/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Context queries and configuration shared by both programming models.

pub use crate::runtime::{Context, Context as CudaContext};
use crate::{DeviceLaunchLimits, DriverError, IntoResult, KeepAlive, Stream};
use std::{
    ffi::c_int,
    mem::MaybeUninit,
    sync::{atomic::Ordering, Arc},
};

/// The device's meaningful stream priorities, from
/// [`Context::stream_priority_range`].
///
/// CUDA orders priorities the opposite way round to intuition: **a lower
/// number is a higher priority**, so the range runs from
/// [`greatest`](Self::greatest) up to [`least`](Self::least) and
/// `greatest <= least` always holds.
///
/// A priority outside the range is not refused when a stream is created. The
/// driver clamps it to the nearest end and reports nothing, so a caller that
/// wants to know what it will get should ask [`clamp`](Self::clamp) rather
/// than assume the request survived.
///
/// A device without priority support reports `0` for both ends, which
/// [`is_supported`](Self::is_supported) reads as unsupported. Creating a
/// stream still succeeds there; every priority simply collapses to the same
/// one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPriorityRange {
    /// Numerically largest value, the *lowest* priority.
    least: i32,
    /// Numerically smallest value, the *highest* priority.
    greatest: i32,
}

impl StreamPriorityRange {
    /// The lowest priority, which is the numerically largest value.
    pub fn least(&self) -> i32 {
        self.least
    }

    /// The highest priority, which is the numerically smallest value.
    pub fn greatest(&self) -> i32 {
        self.greatest
    }

    /// Whether this device implements stream priorities at all.
    ///
    /// False when the driver reports `0` for both ends, which is how
    /// `cuCtxGetStreamPriorityRange` answers on a device without support.
    /// A device with support always reports a range wider than one value.
    pub fn is_supported(&self) -> bool {
        self.least != self.greatest
    }

    /// Whether `priority` lies inside the range, and so survives stream
    /// creation unchanged.
    pub fn contains(&self, priority: i32) -> bool {
        (self.greatest..=self.least).contains(&priority)
    }

    /// The value the driver will actually apply for a request of `priority`.
    ///
    /// Answers the silent clamp in
    /// [`Context::new_stream_with_priority`] ahead of the call.
    pub fn clamp(&self, priority: i32) -> i32 {
        priority.clamp(self.greatest, self.least)
    }
}

/// A per-context device limit (`CU_LIMIT_*`), read with
/// [`Context::limit`] and written with [`Context::set_limit`].
///
/// Every limit is state on the device's **primary** context, so it is shared
/// by every [`Context`] for that device, by any runtime-API user of the
/// same device in this process, and across library boundaries. A limit set
/// here is not scoped to the handle that set it.
///
/// The driver is free to clamp or round a request. Read the limit back after
/// setting it to observe what was actually applied.
///
/// # Ordering constraints
///
/// [`PrintfFifoSize`](Self::PrintfFifoSize) and
/// [`MallocHeapSize`](Self::MallocHeapSize) must be set **before the first
/// kernel launch in the process that uses `printf` or device `malloc`**;
/// afterwards the driver rejects the write with `CUDA_ERROR_INVALID_VALUE`.
/// Because the limit lives on the shared primary context, "first launch"
/// counts launches made through any handle, not just this one.
///
/// # Omitted variants
///
/// `CU_LIMIT_SHMEM_SIZE`, `CU_LIMIT_CIG_ENABLED` and
/// `CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED` are absent. The first two are
/// query-only, all three concern CIG (graphics-interop) contexts, and none
/// predates CUDA 12.5, so naming them would need a header probe and a `cfg`
/// in the manner of `cuda_has_cuEventElapsedTime_v2`. The seven variants
/// below have been present since CUDA 11 and are the ones
/// `cuCtxSetLimit`'s own documentation specifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextLimit {
    /// Stack size in bytes for each GPU thread (`CU_LIMIT_STACK_SIZE`).
    ///
    /// The driver raises this on its own whenever a launched kernel needs a
    /// larger frame and **does not lower it again**, so a read reflects the
    /// high-water mark of everything launched in this context so far. The
    /// reservation scales as roughly `bytes * mp_count * threads_per_mp` and
    /// is invisible to the allocation APIs: it simply reduces the device
    /// memory a later allocation can obtain. Lowering it after a deep-frame
    /// kernel has raised it is how that memory is handed back.
    StackSize,
    /// Size in bytes of the FIFO backing the device `printf`
    /// (`CU_LIMIT_PRINTF_FIFO_SIZE`).
    ///
    /// The FIFO is circular: once a launch fills it, the **oldest** output is
    /// overwritten and lost silently, with no error and no marker in the
    /// stream the host prints. Raising this is the only fix for a kernel
    /// whose output is truncated. Subject to the ordering constraint above.
    PrintfFifoSize,
    /// Size in bytes of the heap backing device `malloc` and `free`
    /// (`CU_LIMIT_MALLOC_HEAP_SIZE`). Subject to the ordering constraint
    /// above.
    MallocHeapSize,
    /// Maximum grid nesting depth at which a device-runtime thread may call
    /// `cudaDeviceSynchronize` (`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`).
    ///
    /// Applies only to devices of compute capability below 9.0; elsewhere the
    /// driver returns `CUDA_ERROR_UNSUPPORTED_LIMIT`. Each level of depth
    /// reserves device memory, so a request the driver cannot back fails with
    /// `CUDA_ERROR_OUT_OF_MEMORY` and leaves the limit settable at a lower
    /// value.
    DevRuntimeSyncDepth,
    /// Maximum number of outstanding device-runtime launches from this
    /// context (`CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`), default 2048.
    ///
    /// Reserves device memory in proportion, and fails the same way as
    /// [`DevRuntimeSyncDepth`](Self::DevRuntimeSyncDepth) when the
    /// reservation cannot be met.
    DevRuntimePendingLaunchCount,
    /// L2 fetch granularity in bytes, 0 to 128
    /// (`CU_LIMIT_MAX_L2_FETCH_GRANULARITY`). A hint: the platform may ignore
    /// or clamp it, and a read need not return what was written.
    MaxL2FetchGranularity,
    /// Bytes of L2 set aside for persisting lines
    /// (`CU_LIMIT_PERSISTING_L2_CACHE_SIZE`). A hint, with the same caveat as
    /// [`MaxL2FetchGranularity`](Self::MaxL2FetchGranularity).
    PersistingL2CacheSize,
}

impl ContextLimit {
    /// Maps to the raw `CUlimit` the driver expects.
    fn to_raw(self) -> cuda_bindings::CUlimit {
        match self {
            ContextLimit::StackSize => cuda_bindings::CUlimit_enum_CU_LIMIT_STACK_SIZE,
            ContextLimit::PrintfFifoSize => cuda_bindings::CUlimit_enum_CU_LIMIT_PRINTF_FIFO_SIZE,
            ContextLimit::MallocHeapSize => cuda_bindings::CUlimit_enum_CU_LIMIT_MALLOC_HEAP_SIZE,
            ContextLimit::DevRuntimeSyncDepth => {
                cuda_bindings::CUlimit_enum_CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
            }
            ContextLimit::DevRuntimePendingLaunchCount => {
                cuda_bindings::CUlimit_enum_CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
            }
            ContextLimit::MaxL2FetchGranularity => {
                cuda_bindings::CUlimit_enum_CU_LIMIT_MAX_L2_FETCH_GRANULARITY
            }
            ContextLimit::PersistingL2CacheSize => {
                cuda_bindings::CUlimit_enum_CU_LIMIT_PERSISTING_L2_CACHE_SIZE
            }
        }
    }
}

/// Context scheduling policy (`CU_CTX_SCHED_*`), governing how the driver
/// waits when the calling host thread blocks on GPU work (a stream or
/// context synchronize).
///
/// Set via [`Context::set_sync_policy`], read via
/// [`Context::sync_policy`]. The trade-off is host CPU against wake
/// latency: [`Spin`](Self::Spin) burns a full core for the lowest latency,
/// [`BlockingSync`](Self::BlockingSync) frees the core but wakes later. On a
/// host with spare cores this trade rarely matters; on a CPU-constrained one
/// (few physical cores, a sync-heavy pipeline) it can cost double-digit
/// percentages of host CPU for no wall-clock benefit, which is the case
/// [`BlockingSync`](Self::BlockingSync) exists to fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Driver-chosen default: spin if the process has more logical CPUs than
    /// active contexts, otherwise yield.
    Auto,
    /// Spin the calling thread while waiting. Lowest latency, highest CPU use.
    Spin,
    /// Yield the calling thread's timeslice while waiting.
    Yield,
    /// Block the calling thread on a synchronization primitive while
    /// waiting. Lowest CPU use, highest wake latency.
    BlockingSync,
}

impl SyncPolicy {
    fn to_raw(self) -> cuda_bindings::CUctx_flags_enum {
        match self {
            SyncPolicy::Auto => cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_AUTO,
            SyncPolicy::Spin => cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_SPIN,
            SyncPolicy::Yield => cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_YIELD,
            SyncPolicy::BlockingSync => cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC,
        }
    }

    /// Decodes the scheduling bits (`CU_CTX_SCHED_MASK`) out of a raw flags
    /// word. `None` for a driver-reserved combination this enum does not
    /// name (the mask is 3 bits wide; only 4 of 8 values are assigned today).
    fn from_raw(raw: cuda_bindings::CUctx_flags_enum) -> Option<Self> {
        match raw & cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_MASK {
            cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_AUTO => Some(SyncPolicy::Auto),
            cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_SPIN => Some(SyncPolicy::Spin),
            cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_YIELD => Some(SyncPolicy::Yield),
            cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC => {
                Some(SyncPolicy::BlockingSync)
            }
            _ => None,
        }
    }
}

impl Context {
    fn device_attribute(
        &self,
        attribute: cuda_bindings::CUdevice_attribute,
    ) -> Result<u32, DriverError> {
        self.bind_to_thread()?;
        let mut value = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGetAttribute(value.as_mut_ptr(), attribute, self.cu_device)
                .result()?;
            u32::try_from(value.assume_init())
                .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
        }
    }
    /// Returns the device's meaningful stream priority range.
    ///
    /// Wraps `cuCtxGetStreamPriorityRange`. See [`StreamPriorityRange`] for
    /// the ordering convention and for what a device without priority support
    /// reports.
    pub fn stream_priority_range(&self) -> Result<StreamPriorityRange, DriverError> {
        self.bind_to_thread()?;
        let mut least = MaybeUninit::uninit();
        let mut greatest = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuCtxGetStreamPriorityRange(least.as_mut_ptr(), greatest.as_mut_ptr())
                .result()?;
            Ok(StreamPriorityRange {
                least: least.assume_init(),
                greatest: greatest.assume_init(),
            })
        }
    }
    /// Queries the device's marketing name (e.g. `"NVIDIA GeForce RTX 5090"`).
    ///
    /// Wraps `cuDeviceGetName` with a 256-byte buffer (driver guarantees
    /// the name fits in 256 bytes including the trailing NUL). Returns the
    /// decoded UTF-8 string with any trailing NULs stripped.
    pub fn device_name(&self) -> Result<String, DriverError> {
        self.bind_to_thread()?;
        // A `u8` buffer sidesteps `c_char`, which is `i8` on x86_64 but `u8`
        // on aarch64 (where a `c as u8` cast trips `clippy::unnecessary_cast`).
        let mut buf = [0u8; 256];
        unsafe {
            cuda_bindings::cuDeviceGetName(
                buf.as_mut_ptr().cast(),
                buf.len() as c_int,
                self.cu_device,
            )
            .result()?;
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }
    /// Queries the compute capability (SM version) of the device.
    ///
    /// Returns `(major, minor)` -- e.g. `(9, 0)` for Hopper H100.
    pub fn compute_capability(&self) -> Result<(i32, i32), DriverError> {
        self.bind_to_thread()?;
        let mut major = MaybeUninit::uninit();
        let mut minor = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.cu_device,
            )
            .result()?;
            cuda_bindings::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.cu_device,
            )
            .result()?;
            Ok((major.assume_init(), minor.assume_init()))
        }
    }
    /// Queries dimension, thread-count, and portable shared-memory launch
    /// limits for this device.
    ///
    /// Cooperative and cluster capabilities are deliberately not queried
    /// here. Typed launch preparation asks for those newer attributes only
    /// when the kernel contract requires the corresponding launch mode.
    pub fn launch_limits(&self) -> Result<DeviceLaunchLimits, DriverError> {
        Ok(DeviceLaunchLimits {
            max_threads_per_block: self.device_attribute(
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            )?,
            max_block_dim: (
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                )?,
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                )?,
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                )?,
            ),
            max_grid_dim: (
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                )?,
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                )?,
                self.device_attribute(
                    cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                )?,
            ),
            max_shared_memory_per_block: self.device_attribute(
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )?,
        })
    }
    /// Queries the non-portable opt-in shared-memory limit per block.
    ///
    /// Typed launch preparation calls this only when static plus dynamic
    /// shared memory exceeds the portable limit.
    pub fn max_opt_in_shared_memory_per_block(&self) -> Result<u32, DriverError> {
        self.device_attribute(
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    /// Returns whether this device supports cooperative kernel launches.
    pub fn supports_cooperative_launch(&self) -> Result<bool, DriverError> {
        self.device_attribute(
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        )
        .map(|value| value != 0)
    }
    /// Returns whether this device supports thread-block cluster launches.
    pub fn supports_cluster_launch(&self) -> Result<bool, DriverError> {
        self.device_attribute(
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
        )
        .map(|value| value != 0)
    }
    /// Returns the number of streaming multiprocessors on this device.
    pub fn multiprocessor_count(&self) -> Result<u32, DriverError> {
        self.device_attribute(
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    /// Returns the maximum number of threads resident on one streaming
    /// multiprocessor.
    ///
    /// Together with [`multiprocessor_count`](Self::multiprocessor_count) this
    /// gives the thread count that
    /// [`set_stack_size`](Self::set_stack_size) multiplies against.
    pub fn max_threads_per_multiprocessor(&self) -> Result<u32, DriverError> {
        self.device_attribute(
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )
    }
    /// Reads a device limit off this context.
    ///
    /// Wraps `cuCtxGetLimit`. See [`ContextLimit`] for what each limit means,
    /// which of them the driver treats as a hint, and why the value read back
    /// need not equal the value written.
    pub fn limit(&self, limit: ContextLimit) -> Result<usize, DriverError> {
        self.bind_to_thread()?;
        let mut value = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuCtxGetLimit(value.as_mut_ptr(), limit.to_raw()).result()?;
            Ok(value.assume_init())
        }
    }
    /// Writes a device limit on this context.
    ///
    /// Wraps `cuCtxSetLimit`. The limit is state on the device's primary
    /// context and therefore shared process-wide; see [`ContextLimit`] for the
    /// per-limit restrictions, including the launch-ordering constraint on
    /// [`PrintfFifoSize`](ContextLimit::PrintfFifoSize) and
    /// [`MallocHeapSize`](ContextLimit::MallocHeapSize).
    ///
    /// The driver may clamp or round the request, so call
    /// [`limit`](Self::limit) afterwards to observe what was applied. A write
    /// takes effect immediately and may block until previously submitted work
    /// completes.
    pub fn set_limit(&self, limit: ContextLimit, value: usize) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        unsafe { cuda_bindings::cuCtxSetLimit(limit.to_raw(), value) }.result()
    }
    /// Queries the per-thread device stack size, in bytes.
    ///
    /// Equivalent to [`limit`](Self::limit) with
    /// [`ContextLimit::StackSize`], whose documentation describes how the
    /// driver maintains this value.
    pub fn stack_size(&self) -> Result<usize, DriverError> {
        self.limit(ContextLimit::StackSize)
    }
    /// Sets the per-thread device stack size, in bytes.
    ///
    /// Equivalent to [`set_limit`](Self::set_limit) with
    /// [`ContextLimit::StackSize`]. The driver reserves `bytes` for *every*
    /// thread that can be resident on the device, so the reservation scales as
    /// roughly `bytes * mp_count * threads_per_mp`, using
    /// [`multiprocessor_count`](Self::multiprocessor_count) and
    /// [`max_threads_per_multiprocessor`](Self::max_threads_per_multiprocessor).
    ///
    /// CUDA may clamp or round the request; call
    /// [`stack_size`](Self::stack_size) afterwards to observe what the driver
    /// actually applied. The call takes effect immediately and may block until
    /// previously submitted work completes.
    pub fn set_stack_size(&self, bytes: usize) -> Result<(), DriverError> {
        self.set_limit(ContextLimit::StackSize, bytes)
    }
    /// Sets the context's scheduling policy (`CU_CTX_SCHED_*`).
    ///
    /// Wraps `cuDevicePrimaryCtxSetFlags_v2`, scoped to this context's
    /// device rather than `cuCtxSetFlags`'s "whatever is current on this
    /// thread": [`Context`] retains and owns a specific device's primary
    /// context, so the device-scoped call is the one that matches what this
    /// type actually holds. Read-modify-write: only the 3 scheduling bits
    /// (`CU_CTX_SCHED_MASK`) are replaced, any other flag bit the process has
    /// set independently (e.g. `CU_CTX_MAP_HOST`) is preserved.
    ///
    /// The policy is process-wide state on the device's primary context:
    /// it affects every [`Context`] clone of this device, and any
    /// runtime-API user of the same device in this process, not just this
    /// handle. The read-modify-write over `CU_CTX_SCHED_MASK` is also not
    /// atomic: a concurrent flag writer on the same device can interleave
    /// between the get and the set, and one side's update is then lost.
    ///
    /// Historical note: before CUDA 11, `cuDevicePrimaryCtxSetFlags` failed
    /// with `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE` once the primary context was
    /// active. That restriction was lifted; the flags now apply to the
    /// already-active context.
    pub fn set_sync_policy(&self, policy: SyncPolicy) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        unsafe {
            let mut current = MaybeUninit::uninit();
            let mut active = MaybeUninit::uninit();
            cuda_bindings::cuDevicePrimaryCtxGetState(
                self.cu_device,
                current.as_mut_ptr(),
                active.as_mut_ptr(),
            )
            .result()?;
            let current = current.assume_init();
            let new_flags =
                (current & !cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_MASK) | policy.to_raw();
            cuda_bindings::cuDevicePrimaryCtxSetFlags_v2(self.cu_device, new_flags).result()
        }
    }
    /// Returns the context's current scheduling policy (`CU_CTX_SCHED_*`).
    ///
    /// `None` if the driver reports a scheduling value this enum does not
    /// name (see [`SyncPolicy::from_raw`](SyncPolicy) -- the mask has more
    /// bit patterns than the driver assigns meanings to today).
    pub fn sync_policy(&self) -> Result<Option<SyncPolicy>, DriverError> {
        self.bind_to_thread()?;
        unsafe {
            let mut flags = MaybeUninit::uninit();
            let mut active = MaybeUninit::uninit();
            cuda_bindings::cuDevicePrimaryCtxGetState(
                self.cu_device,
                flags.as_mut_ptr(),
                active.as_mut_ptr(),
            )
            .result()?;
            Ok(SyncPolicy::from_raw(flags.assume_init()))
        }
    }
    /// Atomically reads and clears the sticky error state.
    ///
    /// Returns `Ok(())` if no error was recorded, or the stored
    /// [`DriverError`] otherwise. The error is cleared after this call.
    pub fn check_err(&self) -> Result<(), DriverError> {
        let error_state = self.error_state.swap(0, Ordering::Relaxed);
        if error_state == 0 {
            Ok(())
        } else {
            Err(DriverError(error_state))
        }
    }
    /// Records a driver error into the sticky error state.
    ///
    /// Used during [`Drop`] paths where returning a `Result` is not possible.
    /// If `result` is `Err`, the raw error code is stored; subsequent
    /// [`check_err`](Self::check_err) or [`bind_to_thread`](Self::bind_to_thread)
    /// calls will surface it. A later store overwrites an earlier one.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            self.error_state.store(err.0, Ordering::Relaxed)
        }
    }
    /// Returns the context's legacy default stream (stream zero).
    pub fn default_stream(self: &Arc<Self>) -> Arc<Stream> {
        Arc::new(Stream {
            cu_stream: std::ptr::null_mut(),
            context: self.clone(),
            owned: false,
            _keep_alive: KeepAlive::none(),
        })
    }
    /// Creates a nonblocking stream with the requested CUDA scheduling priority.
    pub fn new_stream_with_priority(
        self: &Arc<Self>,
        priority: i32,
    ) -> Result<Arc<Stream>, DriverError> {
        self.bind_to_thread()?;
        let mut stream = std::ptr::null_mut();
        unsafe {
            cuda_bindings::cuStreamCreateWithPriority(
                &mut stream,
                cuda_bindings::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
                priority,
            )
            .result()?;
        }
        Ok(Arc::new(Stream {
            cu_stream: stream,
            context: self.clone(),
            owned: true,
            _keep_alive: KeepAlive::none(),
        }))
    }
}
