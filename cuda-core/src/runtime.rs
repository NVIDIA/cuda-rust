/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA runtime types: Device, Stream, Module, Function, LaunchConfig.
//!
//! These are the public API of `cuda-core`. They wrap raw CUDA driver
//! handles with RAII lifetimes and provide `borrow_raw` constructors
//! for interop with external frameworks (cudarc, etc.).

use std::ffi::{c_int, c_void, CString};
use std::sync::Arc;

use crate::cudarc_shim::{ctx, device, module, pool, primary_ctx, stream};
use crate::error::*;
use crate::init;

/// Kernel launch configuration specifying grid, block, and shared memory sizes.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// Grid dimensions `(x, y, z)` in thread blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Bytes of dynamic shared memory per block.
    pub shared_mem_bytes: u32,
}

/// Anything that owns an external CUDA resource. A borrowed handle can hold an
/// `Arc<dyn ForeignOwner>` as a *liveness token*: while the handle (and anything
/// derived from it) is alive, the token's refcount is nonzero, so the external
/// owner cannot be dropped — and the resource it backs cannot be destroyed —
/// out from under cutile. Blanket-implemented, so any `Arc<T>` erases to
/// `Arc<dyn ForeignOwner>`.
pub trait ForeignOwner: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> ForeignOwner for T {}

/// Optional liveness token held by a borrowed handle (see [`ForeignOwner`]).
///
/// Compares equal regardless of contents (the token is an ownership detail, not
/// part of a handle's identity) and prints opaquely, so the handle types keep
/// their `Debug`/`PartialEq`/`Eq` derives.
#[derive(Clone, Default)]
pub struct KeepAlive(Option<Arc<dyn ForeignOwner>>);

impl KeepAlive {
    /// A token holding nothing (owned or raw-borrowed handles).
    pub fn none() -> Self {
        Self(None)
    }
    /// A token keeping `owner` alive for the handle's lifetime.
    pub fn owner(owner: Arc<dyn ForeignOwner>) -> Self {
        Self(Some(owner))
    }
}

impl std::fmt::Debug for KeepAlive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "KeepAlive(owner)"
        } else {
            "KeepAlive(none)"
        })
    }
}

impl PartialEq for KeepAlive {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for KeepAlive {}

/// A GPU device handle wrapping a CUDA primary context.
///
/// Can be **owned** (created via [`Device::new`], releases the primary context
/// on drop), **borrowed** (created via [`Device::borrow_raw`], does NOT release
/// on drop), or **foreign** (created via [`Device::borrow_with_owner`], holds a
/// liveness token so the external owner outlives it).
#[derive(Debug)]
pub struct Device {
    pub(crate) cu_device: cuda_bindings::CUdevice,
    pub(crate) cu_ctx: cuda_bindings::CUcontext,
    pub(crate) ordinal: usize,
    owned: bool,
    _keep_alive: KeepAlive,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _guard = teardown_lock();
        // Streams hold an Arc<Device>, so by the time the last device handle
        // for this ordinal drops, every pooled handle is idle; the context
        // release below reclaims them. Discard so a later re-retain cannot
        // pop a handle from a torn-down context.
        stream_pool()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.ordinal);
        let _ = self.bind_to_thread();
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            let _ = unsafe { primary_ctx::release(self.cu_device) };
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.cu_device == other.cu_device
            && self.cu_ctx == other.cu_ctx
            && self.ordinal == other.ordinal
    }
}
impl Eq for Device {}

/// The driver indexes devices with a C `int`. An ordinal that does not fit is
/// not a device at all; `as c_int` would wrap it onto some other device's
/// index, so it is rejected up front with `CUDA_ERROR_INVALID_DEVICE`.
fn ordinal_to_c_int(ordinal: usize) -> Result<c_int, DriverError> {
    c_int::try_from(ordinal)
        .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_DEVICE))
}

impl Device {
    /// Creates a new owned device on the specified ordinal.
    ///
    /// Errors with `CUDA_ERROR_INVALID_DEVICE` (before touching the driver)
    /// when `ordinal` does not fit the driver's `int` ordinal type.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let cu_ordinal = ordinal_to_c_int(ordinal)?;
        unsafe { init(0)? };
        let cu_device = device::get(cu_ordinal)?;
        let cu_ctx = unsafe { primary_ctx::retain(cu_device) }?;
        let device = Arc::new(Device {
            cu_device,
            cu_ctx,
            ordinal,
            owned: true,
            _keep_alive: KeepAlive::none(),
        });
        device.bind_to_thread()?;
        Ok(device)
    }

    /// Wraps externally-owned CUDA handles without taking ownership.
    ///
    /// Inputs are the raw C primitives (`CUcontext` is an opaque pointer,
    /// `CUdevice` is `int` in the driver API). Accepting primitives rather
    /// than `cuda_bindings::CU*` typedefs keeps this API agnostic to which
    /// binding crate the caller uses — a cudarc `CUcontext`, a fresh
    /// `bindgen` wrapper, or a hand-rolled FFI type all cast in the same way.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_ctx` points to a valid retained `CUcontext` for `cu_device`
    /// - The handles outlive the returned `Device` **and everything derived
    ///   from it**. Streams from [`new_stream`](Self::new_stream) hold the
    ///   device alive and, because the device is borrowed, are never parked in
    ///   the process-wide stream pool: each is synchronized and destroyed
    ///   against `cu_ctx` when it drops (see [`Stream`]). `cu_ctx` must
    ///   therefore still be valid when the last such stream drops.
    /// - No concurrent destruction of the handles
    pub unsafe fn borrow_raw(cu_ctx: *mut c_void, cu_device: c_int, ordinal: usize) -> Arc<Self> {
        Arc::new(Device {
            cu_device: cu_device as cuda_bindings::CUdevice,
            cu_ctx: cu_ctx as cuda_bindings::CUcontext,
            ordinal,
            owned: false,
            _keep_alive: KeepAlive::none(),
        })
    }

    /// Wraps externally-owned CUDA handles, holding `owner` alive for the
    /// returned device's lifetime.
    ///
    /// Same as [`borrow_raw`](Self::borrow_raw), but the liveness obligation is
    /// discharged by construction: `owner` is whatever owns the context (a
    /// cudarc device, a torch context handle, ...), and holding it here
    /// guarantees the handles stay valid as long as this `Device` — or anything
    /// derived from it — lives. Only the point-in-time validity of the handles
    /// remains a caller assertion.
    ///
    /// # Safety
    /// The caller must ensure, *at construction*, that:
    /// - `cu_ctx` points to a valid retained `CUcontext` for `cu_device`, and
    /// - dropping `owner` would release those handles (i.e. `owner` really is
    ///   what keeps them alive).
    pub unsafe fn borrow_with_owner(
        cu_ctx: *mut c_void,
        cu_device: c_int,
        ordinal: usize,
        owner: Arc<dyn ForeignOwner>,
    ) -> Arc<Self> {
        Arc::new(Device {
            cu_device: cu_device as cuda_bindings::CUdevice,
            cu_ctx: cu_ctx as cuda_bindings::CUcontext,
            ordinal,
            owned: false,
            _keep_alive: KeepAlive::owner(owner),
        })
    }

    /// Returns the number of CUDA-capable devices available.
    pub fn device_count() -> Result<i32, DriverError> {
        unsafe { init(0)? };
        device::get_count()
    }

    /// Returns the raw `CUdevice` handle for a given ordinal without
    /// creating a full `Device` (no context retained). Rejects an ordinal
    /// that does not fit `c_int` like [`new`](Self::new) does.
    pub fn raw_device(ordinal: usize) -> Result<cuda_bindings::CUdevice, DriverError> {
        let cu_ordinal = ordinal_to_c_int(ordinal)?;
        unsafe { init(0)? };
        device::get(cu_ordinal)
    }

    /// Get the `ordinal` index of the device this is on.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, DriverError> {
        device::get_name(self.cu_device)
    }

    /// Returns the raw `CUdevice` handle.
    pub fn cu_device(&self) -> cuda_bindings::CUdevice {
        self.cu_device
    }

    /// Returns the raw `CUcontext` handle.
    pub fn cu_ctx(&self) -> cuda_bindings::CUcontext {
        self.cu_ctx
    }

    /// Binds this context to the calling thread if not already current.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        if match ctx::get_current()? {
            Some(curr_ctx) => curr_ctx != self.cu_ctx,
            None => true,
        } {
            unsafe { ctx::set_current(self.cu_ctx) }?;
        }
        Ok(())
    }

    /// Blocks until all work on this device's context is complete.
    ///
    /// # Safety
    /// The caller must ensure this device's context is current on the
    /// calling thread (via [`bind_to_thread`](Device::bind_to_thread)).
    pub unsafe fn synchronize(&self) -> Result<(), DriverError> {
        ctx::synchronize()
    }

    /// Creates a new non-blocking CUDA stream on this device.
    ///
    /// On an owned device the handle may be one parked by an earlier stream's
    /// drop (see [`Stream`]). On a borrowed device it is always freshly
    /// created: the pool is keyed by ordinal alone and holds handles from the
    /// primary contexts this crate retains, which need not be the context the
    /// external owner handed to [`borrow_raw`](Self::borrow_raw).
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<Stream>, DriverError> {
        self.bind_to_thread()?;
        let pooled = if self.owned {
            stream_pool()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get_mut(&self.ordinal)
                .and_then(Vec::pop)
        } else {
            None
        };
        let cu_stream = match pooled {
            Some(handle) => handle as cuda_bindings::CUstream,
            None => stream::create(stream::StreamKind::NonBlocking)?,
        };
        Ok(Arc::new(Stream {
            cu_stream,
            device: self.clone(),
            owned: true,
            _keep_alive: KeepAlive::none(),
        }))
    }

    /// Loads a CUDA module from a PTX source string.
    pub fn load_module_from_ptx_src(
        self: &Arc<Self>,
        ptx_src: &str,
    ) -> Result<Arc<Module>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = {
            let c_src = CString::new(ptx_src).unwrap();
            unsafe { module::load_data(c_src.as_ptr() as *const _) }
        }?;
        Ok(Arc::new(Module {
            cu_module,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Loads a CUDA module from a file path (PTX or cubin).
    pub fn load_module_from_file(
        self: &Arc<Self>,
        filename: &str,
    ) -> Result<Arc<Module>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = { module::load(filename) }?;
        Ok(Arc::new(Module {
            cu_module,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Loads a CUDA module from an in-memory **cubin** image.
    ///
    /// The image must be a cubin — an ELF, which encodes its own length. This is
    /// deliberately *not* a PTX loader: `cuModuleLoadData` parses PTX as a
    /// NUL-terminated C string, and a byte slice carries no terminator, so PTX
    /// bytes would be read past the end of the slice. Use
    /// [`load_module_from_ptx_src`](Self::load_module_from_ptx_src) for PTX; it
    /// builds a `CString`. The ELF-magic check below rejects a non-cubin image
    /// (an empty slice included) with `CUDA_ERROR_INVALID_IMAGE` rather than
    /// letting the driver over-read.
    ///
    /// # Safety
    ///
    /// `image` must be a **complete, well-formed cubin**. `cuModuleLoadData`
    /// takes no length: the driver dereferences the section and program-header
    /// offsets declared inside the image, so a truncated or otherwise malformed
    /// image is read past the end of the slice (a truncated cubin segfaults
    /// inside libcuda). The ELF-magic check only rules out non-ELF input; it
    /// cannot validate those offsets, so the caller must know the bytes are the
    /// whole artifact — produced by the compiler in this process, or read back
    /// from a store whose per-entry checksum verified.
    pub unsafe fn load_module_from_bytes(
        self: &Arc<Self>,
        image: &[u8],
    ) -> Result<Arc<Module>, DriverError> {
        // ELF magic `\x7fELF`; a bounded prefix check that also covers the empty
        // slice, so the raw pointer handed to `cuModuleLoadData` is never PTX
        // text (which it would read up to an out-of-bounds NUL).
        if !image.starts_with(b"\x7fELF") {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_IMAGE,
            ));
        }
        self.bind_to_thread()?;
        // SAFETY: the caller guarantees a complete cubin, so every offset the
        // driver follows from the header lands inside `image`.
        let cu_module = unsafe { module::load_data(image.as_ptr().cast()) }?;
        Ok(Arc::new(Module {
            cu_module,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Creates a new memory pool on this device.
    ///
    /// The returned pool is owned — it will be destroyed when the last `Arc`
    /// is dropped. Pair with [`MemPool::set_release_threshold`] to control
    /// when the pool returns memory to the OS.
    pub fn new_mem_pool(self: &Arc<Self>) -> Result<Arc<MemPool>, DriverError> {
        self.bind_to_thread()?;
        let mut props: cuda_bindings::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = cuda_bindings::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = cuda_bindings::CUmemAllocationHandleType_enum_CU_MEM_HANDLE_TYPE_NONE;
        props.location.type_ = cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
        cuda_bindings::set_mem_location_id(&mut props.location, self.ordinal as c_int);
        let cu_pool = unsafe { pool::create(&props) }?;
        Ok(Arc::new(MemPool {
            cu_pool,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Returns the driver-owned default memory pool for this device.
    ///
    /// The returned wrapper is **not owned** — dropping it does not destroy the
    /// default pool, which is shared across all users of the device.
    pub fn default_mem_pool(self: &Arc<Self>) -> Result<Arc<MemPool>, DriverError> {
        self.bind_to_thread()?;
        let cu_pool = unsafe { pool::get_default(self.cu_device) }?;
        Ok(Arc::new(MemPool {
            cu_pool,
            device: self.clone(),
            owned: false,
        }))
    }
}

/// A CUDA memory pool handle.
///
/// Can be either **owned** (created via [`Device::new_mem_pool`], destroyed on
/// drop) or **borrowed** (created via [`Device::default_mem_pool`], does NOT
/// destroy on drop).
///
/// Used by async tensor allocation via `cuMemAllocFromPoolAsync` when a pool
/// is registered via `cuda_async::device_context::set_device_pool`.
#[derive(Debug)]
pub struct MemPool {
    pub(crate) cu_pool: cuda_bindings::CUmemoryPool,
    pub(crate) device: Arc<Device>,
    owned: bool,
}

unsafe impl Send for MemPool {}
unsafe impl Sync for MemPool {}

impl Drop for MemPool {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _ = self.device.bind_to_thread();
        let _ = unsafe { pool::destroy(self.cu_pool) };
    }
}

impl MemPool {
    /// Returns the raw `CUmemoryPool` handle.
    pub fn cu_pool(&self) -> cuda_bindings::CUmemoryPool {
        self.cu_pool
    }

    /// Returns a reference to the parent device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Sets the release threshold for this pool.
    ///
    /// Memory held by the pool is not returned to the OS until pool usage drops
    /// below this threshold. Use `u64::MAX` to prevent the OS from reclaiming
    /// pool memory (useful for inference workloads with stable memory footprints).
    pub fn set_release_threshold(&self, threshold: u64) -> Result<(), DriverError> {
        self.device.bind_to_thread()?;
        unsafe { pool::set_release_threshold(self.cu_pool, threshold) }
    }

    /// Reads all four memory accounting counters in a single bind-to-thread block.
    ///
    /// The four reads happen back-to-back, but the pool is being mutated by
    /// async allocations on streams — so this is a best-effort snapshot, not
    /// a synchronously consistent reading. For stable values, sync the streams
    /// that allocate from this pool first.
    pub fn mem_stats(&self) -> Result<PoolMemStats, DriverError> {
        self.device.bind_to_thread()?;
        unsafe {
            Ok(PoolMemStats {
                used_current: pool::get_attribute_u64(
                    self.cu_pool,
                    cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                )?,
                used_high: pool::get_attribute_u64(
                    self.cu_pool,
                    cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                )?,
                reserved_current: pool::get_attribute_u64(
                    self.cu_pool,
                    cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                )?,
                reserved_high: pool::get_attribute_u64(
                    self.cu_pool,
                    cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                )?,
            })
        }
    }

    /// Resets `used_high` to the current `used_current`.
    pub fn reset_used_high(&self) -> Result<(), DriverError> {
        self.device.bind_to_thread()?;
        unsafe {
            pool::reset_high_watermark(
                self.cu_pool,
                cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            )
        }
    }

    /// Resets `reserved_high` to the current `reserved_current`.
    pub fn reset_reserved_high(&self) -> Result<(), DriverError> {
        self.device.bind_to_thread()?;
        unsafe {
            pool::reset_high_watermark(
                self.cu_pool,
                cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            )
        }
    }
}

/// Snapshot of a pool's memory accounting counters at a point in time.
///
/// `used_*` tracks memory currently checked out to the application; `reserved_*`
/// tracks memory the pool holds from the OS (which may exceed `used_*` due to
/// internal caching). `*_high` are watermarks since the last reset (or pool
/// creation), readable via [`MemPool::mem_stats`] and resettable via
/// [`MemPool::reset_used_high`] / [`MemPool::reset_reserved_high`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolMemStats {
    pub used_current: u64,
    pub used_high: u64,
    pub reserved_current: u64,
    pub reserved_high: u64,
}

/// A CUDA stream handle.
///
/// Can be **owned** (created via [`Device::new_stream`]), **borrowed** (created
/// via [`Stream::borrow_raw`], does NOT destroy on drop), or **foreign**
/// (created via [`Stream::borrow_with_owner`], holds a liveness token so the
/// external owner outlives it).
///
/// Dropping an owned stream blocks in `cuStreamSynchronize` until the work
/// enqueued on it has drained. What happens to the handle next depends on the
/// device it was created on:
///
/// - On an **owned** device the handle is *not* destroyed: it is parked in a
///   process-wide, per-ordinal pool and handed out again by the next
///   [`Device::new_stream`] on that ordinal (see `stream_pool` for why). The
///   last owned `Device` for the ordinal discards the pool entry, and the
///   primary-context release reclaims the handles.
/// - On a **borrowed** device ([`Device::borrow_raw`] /
///   [`Device::borrow_with_owner`]) the handle is destroyed, under the
///   teardown lock. It must never enter the pool: the pool is keyed by
///   ordinal only, so an owned `Device` created later for the same ordinal
///   would pop a handle belonging to a context the external owner may already
///   have destroyed.
#[derive(Debug, PartialEq, Eq)]
pub struct Stream {
    pub(crate) cu_stream: cuda_bindings::CUstream,
    pub(crate) device: Arc<Device>,
    owned: bool,
    _keep_alive: KeepAlive,
}

/// Per-device pool of idle stream handles (all created `NonBlocking` by
/// [`Device::new_stream`]). `cuStreamDestroy` racing concurrent driver
/// activity from other threads segfaults inside libcuda (reproducible in
/// parallel test runs; a leak-streams experiment ran 10/10 clean where
/// destroy-paths crashed ~half the runs), so owned streams are never
/// destroyed: drops return the handle here, `new_stream` reuses it, and the
/// last `Device` drop for an ordinal discards the pooled handles — the
/// primary-context release reclaims them.
fn stream_pool() -> &'static std::sync::Mutex<std::collections::HashMap<usize, Vec<usize>>> {
    static POOL: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Vec<usize>>>,
    > = std::sync::OnceLock::new();
    POOL.get_or_init(Default::default)
}

/// Serializes destructive driver teardown (stream destroy, module unload,
/// context release). Concurrent teardown racing other threads' driver calls
/// segfaults inside libcuda (observed: cuStreamDestroy_v2 during parallel
/// test runs); teardown is cold, so one process-wide lock is cheap.
pub(crate) fn teardown_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

// ── Event ───────────────────────────────────────────────────────────────────

/// A CUDA event, created with timing enabled, for device-side timing and
/// cross-stream synchronization.
///
/// Owned RAII: the driver handle is destroyed on drop (the driver defers
/// destruction of an event still captured in unfinished work, so dropping
/// early is safe). An event is bound to its device at construction;
/// recording it on a stream of another device is rejected.
pub struct Event {
    cu_event: cuda_bindings::CUevent,
    device: Arc<Device>,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// Records this event on `stream`.
    ///
    /// Errors with `CUDA_ERROR_INVALID_VALUE` if the stream belongs to a
    /// different device than the one this event was created on.
    pub fn record(&self, stream: &Arc<Stream>) -> Result<(), DriverError> {
        if stream.device().ordinal() != self.device.ordinal() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        // Safety: both handles are valid by construction (RAII wrappers),
        // and the same-device check above pins them to one context.
        unsafe { crate::cudarc_shim::event::record(self.cu_event, stream.cu_stream()) }
    }

    /// Blocks the calling thread until this event has completed.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        // Safety: the handle is valid by construction.
        unsafe { crate::cudarc_shim::event::synchronize(self.cu_event) }
    }

    /// Queries completion without blocking: `Ok(true)` once all work
    /// captured by the most recent [`record`](Self::record) has completed
    /// (or the event was never recorded), `Ok(false)` while it is still in
    /// flight. Any other driver result is returned as the error.
    pub fn query(&self) -> Result<bool, DriverError> {
        // Safety: the handle is valid by construction.
        match unsafe { crate::cudarc_shim::event::query(self.cu_event) } {
            Ok(()) => Ok(true),
            Err(DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Milliseconds elapsed on the device between this event and `end`
    /// (called on the start event, `torch.cuda.Event` convention:
    /// `start.elapsed_time(&end)`).
    ///
    /// Both events must have been recorded, and `end` must have completed —
    /// call [`synchronize`](Self::synchronize) on it first, or the driver
    /// reports not-ready.
    pub fn elapsed_time(&self, end: &Event) -> Result<f32, DriverError> {
        // Safety: both handles are valid by construction.
        unsafe { crate::cudarc_shim::event::elapsed(self.cu_event, end.cu_event) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // Safety: owned handle; the driver defers destruction while in use.
        let _ = unsafe { crate::cudarc_shim::event::destroy(self.cu_event) };
    }
}

impl Device {
    /// Creates a timing-enabled event on this device.
    pub fn new_event(self: &Arc<Self>) -> Result<Event, DriverError> {
        self.bind_to_thread()?;
        let cu_event =
            crate::cudarc_shim::event::create(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT)?;
        Ok(Event {
            cu_event,
            device: self.clone(),
        })
    }

    /// Returns the device's L2 cache size in bytes.
    pub fn l2_cache_size_bytes(&self) -> Result<usize, DriverError> {
        let mut value: core::ffi::c_int = 0;
        // Safety: out-pointer is valid; the device handle is valid by
        // construction.
        unsafe {
            cuda_bindings::cuDeviceGetAttribute(
                &mut value,
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                self.cu_device,
            )
            .result()?;
        }
        Ok(value.max(0) as usize)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.owned || self.cu_stream.is_null() {
            return;
        }
        let _ = self.device.bind_to_thread();
        if self.device.owned {
            // Never destroyed (see `stream_pool`): drain, then return the
            // handle for reuse by the next `new_stream` on this device.
            let _ = unsafe { stream::synchronize(self.cu_stream) };
            stream_pool()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(self.device.ordinal)
                .or_default()
                .push(self.cu_stream as usize);
        } else {
            // Borrowed device: the pool is keyed by ordinal only, so parking
            // this handle would let an owned `Device` created later for the
            // same ordinal pop a stream from a context the external owner may
            // by then have destroyed (reproduced as a segfault). Destroy it
            // instead, serialized with the other destructive driver calls.
            let _guard = teardown_lock();
            let _ = unsafe { stream::synchronize(self.cu_stream) };
            let _ = unsafe { stream::destroy(self.cu_stream) };
        }
    }
}

impl Stream {
    /// Wraps an externally-owned CUDA stream without taking ownership.
    ///
    /// `cu_stream` is the raw `CUstream` opaque pointer. See
    /// [`Device::borrow_raw`] for why this is a `*mut c_void` rather than a
    /// `cuda_bindings::CUstream` typedef.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_stream` points to a valid CUDA stream on `device`
    /// - The stream outlives the returned `Stream`
    /// - No concurrent destruction of the stream
    pub unsafe fn borrow_raw(cu_stream: *mut c_void, device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Stream {
            cu_stream: cu_stream as cuda_bindings::CUstream,
            device: device.clone(),
            owned: false,
            _keep_alive: KeepAlive::none(),
        })
    }

    /// Wraps an externally-owned CUDA stream, holding `owner` alive for the
    /// returned stream's lifetime.
    ///
    /// Same as [`borrow_raw`](Self::borrow_raw), but the liveness obligation is
    /// discharged by construction: holding `owner` (whatever owns the stream)
    /// guarantees `cu_stream` stays valid as long as this `Stream` — or anything
    /// derived from it — lives. Only the point-in-time validity of `cu_stream`
    /// remains a caller assertion.
    ///
    /// # Safety
    /// The caller must ensure, *at construction*, that `cu_stream` points to a
    /// valid CUDA stream on `device`, and that dropping `owner` would destroy
    /// that stream (i.e. `owner` really is what keeps it alive).
    pub unsafe fn borrow_with_owner(
        cu_stream: *mut c_void,
        device: &Arc<Device>,
        owner: Arc<dyn ForeignOwner>,
    ) -> Arc<Self> {
        Arc::new(Stream {
            cu_stream: cu_stream as cuda_bindings::CUstream,
            device: device.clone(),
            owned: false,
            _keep_alive: KeepAlive::owner(owner),
        })
    }

    /// Returns the raw `CUstream` handle.
    pub fn cu_stream(&self) -> cuda_bindings::CUstream {
        self.cu_stream
    }

    /// Returns a reference to the parent device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Blocks until all work on this stream is complete.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn synchronize(&self) -> Result<(), DriverError> {
        stream::synchronize(self.cu_stream)
    }

    /// Queries stream completion without blocking: `Ok(true)` when all prior
    /// work has completed, `Ok(false)` when work is still in flight.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn query(&self) -> Result<bool, DriverError> {
        stream::query(self.cu_stream)
    }

    /// Makes all work subsequently enqueued on this stream wait until
    /// `event` has completed (`cuStreamWaitEvent`). This is the
    /// cross-stream dependency primitive: record an event on the producing
    /// stream, then have the consuming stream wait on it.
    ///
    /// Errors with `CUDA_ERROR_INVALID_VALUE` if the event belongs to a
    /// different device than this stream.
    pub fn wait_event(&self, event: &Event) -> Result<(), DriverError> {
        if event.device.ordinal() != self.device.ordinal() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        // Safety: both handles are valid by construction (RAII wrappers),
        // and the same-device check above pins them to one context.
        unsafe {
            stream::wait_event(
                self.cu_stream,
                event.cu_event,
                cuda_bindings::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
            )
        }
    }

    /// Enqueues a host-side callback to execute after all prior stream work completes.
    ///
    /// The driver runs `host_func` on one of its own threads once everything
    /// enqueued on this stream before the call has finished. Two rules carry
    /// over from `cuLaunchHostFunc`:
    ///
    /// - **The callback must not call CUDA APIs.** The driver may report
    ///   `CUDA_ERROR_NOT_PERMITTED`, or simply deadlock its own work queue.
    ///   Use the callback to signal the host (an atomic store, a channel send,
    ///   a waker), never to enqueue or wait on GPU work.
    /// - **Panics are caught and discarded.** Unwinding out of the
    ///   `extern "C"` trampoline would abort the process, and the driver has
    ///   no channel to report a failed callback, so a panicking `host_func`
    ///   is silently swallowed.
    ///
    /// The closure is boxed and handed to the driver as user data, which is
    /// why `F: 'static`. On `Ok` the trampoline reclaims the box when the
    /// callback fires; on `Err` the driver never invokes the callback, so the
    /// box is reclaimed here and the closure's captures are dropped.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn launch_host_function<F: FnOnce() + Send + 'static>(
        &self,
        host_func: F,
    ) -> Result<(), DriverError> {
        Self::enqueue_boxed(host_func, |func, arg| unsafe {
            stream::launch_host_function(self.cu_stream, func, arg)
        })
    }

    /// Like [`launch_host_function`](Self::launch_host_function) but with an
    /// explicit host-task sync mode (`CU_HOST_TASK_BLOCKING` /
    /// `CU_HOST_TASK_SPINWAIT`) via `cuLaunchHostFunc_v2`. Same callback
    /// rules: no CUDA calls inside `host_func`, panics are caught and
    /// discarded, and a refused launch reclaims the boxed closure.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn launch_host_function_with_sync_mode<F: FnOnce() + Send + 'static>(
        &self,
        host_func: F,
        sync_mode: ::core::ffi::c_uint,
    ) -> Result<(), DriverError> {
        Self::enqueue_boxed(host_func, |func, arg| unsafe {
            stream::launch_host_function_v2(self.cu_stream, func, arg, sync_mode)
        })
    }

    /// Boxes `host_func`, hands the trampoline and the box to `enqueue`, and
    /// reclaims the box if the driver refuses the launch.
    ///
    /// Ownership of the box passes to the driver only on a successful
    /// enqueue: that is the one case in which the trampoline — the box's
    /// other reclaimer — will ever run. On `Err` nothing else will free it, so
    /// without this step the closure and everything it captured (a waker, a
    /// channel sender, an `Arc`) would leak.
    fn enqueue_boxed<F: FnOnce() + Send + 'static>(
        host_func: F,
        enqueue: impl FnOnce(unsafe extern "C" fn(*mut c_void), *mut c_void) -> Result<(), DriverError>,
    ) -> Result<(), DriverError> {
        let user_data = Box::into_raw(Box::new(host_func)).cast::<c_void>();
        let result = enqueue(Self::callback_wrapper::<F>, user_data);
        if result.is_err() {
            // SAFETY: `user_data` is the `Box<F>` leaked above, and a refused
            // launch never runs the trampoline, so this is its only reclaim.
            drop(unsafe { Box::from_raw(user_data.cast::<F>()) });
        }
        result
    }

    /// `extern "C"` trampoline the driver invokes on a driver-internal thread
    /// when the host function fires. Reconstructs the `Box<F>` leaked by
    /// [`enqueue_boxed`](Self::enqueue_boxed) and calls the closure, catching
    /// panics so nothing unwinds across the C ABI boundary.
    ///
    /// # Safety
    /// `callback` must be the pointer `enqueue_boxed` produced for an `F`
    /// closure, and must be passed here exactly once (double free otherwise).
    unsafe extern "C" fn callback_wrapper<F: FnOnce() + Send + 'static>(callback: *mut c_void) {
        let _ = std::panic::catch_unwind(|| {
            let callback: Box<F> = unsafe { Box::from_raw(callback.cast::<F>()) };
            callback();
        });
    }

    /// Begins stream capture for CUDA graph construction.
    ///
    /// # Safety
    /// The caller must ensure the context is current and the stream is not
    /// already being captured.
    pub unsafe fn begin_capture(
        &self,
        mode: cuda_bindings::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        stream::begin_capture(self.cu_stream, mode)
    }

    /// Ends stream capture and returns the captured CUDA graph.
    ///
    /// # Safety
    /// The caller must ensure `begin_capture` was previously called on this stream.
    pub unsafe fn end_capture(&self) -> Result<cuda_bindings::CUgraph, DriverError> {
        stream::end_capture(self.cu_stream)
    }
}

/// A loaded CUDA module (PTX/cubin).
///
/// Can be either **owned** (created via [`Device::load_module_from_ptx_src`]
/// / [`Device::load_module_from_file`], unloads on drop) or **borrowed**
/// (created via [`Module::borrow_raw`], does NOT unload on drop).
#[derive(Debug)]
pub struct Module {
    pub(crate) cu_module: cuda_bindings::CUmodule,
    pub(crate) device: Arc<Device>,
    owned: bool,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _guard = teardown_lock();
        let _ = self.device.bind_to_thread();
        let _ = unsafe { module::unload(self.cu_module) };
    }
}

impl Module {
    /// Wraps an externally-owned CUDA module without taking ownership.
    ///
    /// `cu_module` is the raw `CUmodule` opaque pointer. See
    /// [`Device::borrow_raw`] for why this is a `*mut c_void`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_module` points to a valid module loaded on `device`
    /// - The module outlives the returned `Module`
    /// - No concurrent unload of the module
    pub unsafe fn borrow_raw(cu_module: *mut c_void, device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Module {
            cu_module: cu_module as cuda_bindings::CUmodule,
            device: device.clone(),
            owned: false,
        })
    }

    /// Returns the raw `CUmodule` handle.
    pub fn cu_module(&self) -> cuda_bindings::CUmodule {
        self.cu_module
    }

    /// Looks up a device function by name within this module.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<Function, DriverError> {
        let cu_function = unsafe { module::get_function(self.cu_module, fn_name) }?;
        Ok(Function {
            cu_function,
            module: self.clone(),
        })
    }
}

/// Handle to a device function loaded from a [`Module`].
#[derive(Debug, Clone)]
pub struct Function {
    pub(crate) cu_function: cuda_bindings::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<Module>,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Wraps an externally-owned CUDA function without taking ownership.
    ///
    /// `cu_function` is the raw `CUfunction` opaque pointer. The returned
    /// `Function` holds a clone of `module` to keep the parent alive;
    /// there is no `owned` flag because `Function` has no `Drop` (functions
    /// are not freed independently — they live as long as their module).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_function` points to a valid function within `module`
    /// - `module` is the module that `cu_function` was obtained from
    pub unsafe fn borrow_raw(cu_function: *mut c_void, module: &Arc<Module>) -> Function {
        Function {
            cu_function: cu_function as cuda_bindings::CUfunction,
            module: module.clone(),
        }
    }

    /// Returns the raw `CUfunction` handle.
    ///
    /// # Safety
    /// The caller must not use the handle after the parent module is dropped.
    pub unsafe fn cu_function(&self) -> cuda_bindings::CUfunction {
        self.cu_function
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts how often the closure it is captured by runs and how often it
    /// is dropped, so a test can tell "reclaimed" (dropped once, never run)
    /// from "leaked" (never dropped) and from "double-freed" (dropped twice).
    struct Probe {
        calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl Probe {
        fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let probe = Probe {
                calls: calls.clone(),
                drops: drops.clone(),
            };
            (probe, calls, drops)
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn refused_host_function_launch_reclaims_the_boxed_closure() {
        let (probe, calls, drops) = Probe::new();
        let refused = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE);

        let result = Stream::enqueue_boxed(
            move || {
                probe.calls.fetch_add(1, Ordering::SeqCst);
            },
            |_trampoline, _user_data| Err(refused),
        );

        assert_eq!(result, Err(refused), "the driver's error must surface");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a refused launch never runs"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the closure and its captures must be dropped exactly once"
        );
    }

    #[test]
    fn accepted_host_function_launch_hands_the_box_to_the_trampoline() {
        let (probe, calls, drops) = Probe::new();

        // Stand in for the driver: accept the launch and fire the callback.
        let result = Stream::enqueue_boxed(
            move || {
                probe.calls.fetch_add(1, Ordering::SeqCst);
            },
            |trampoline, user_data| {
                // SAFETY: `user_data` is the box `enqueue_boxed` just leaked
                // for this trampoline, and this is its one invocation.
                unsafe { trampoline(user_data) };
                Ok(())
            },
        );

        assert_eq!(result, Ok(()));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the callback runs once");
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the trampoline is the sole reclaimer on success: no leak, no double free"
        );
    }

    fn has_gpu() -> bool {
        Device::device_count().map(|n| n > 0).unwrap_or(false)
    }

    /// The stream pool is process-wide and keyed by ordinal, so the tests that
    /// assert on its contents for ordinal 0 must not overlap: a concurrent
    /// test's `new_stream` pops, and its `Device` drop discards the whole
    /// entry. Nothing else in this test binary touches the pool.
    fn pool_tests_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Snapshot of the parked handles for `ordinal`.
    fn pooled_handles(ordinal: usize) -> Vec<usize> {
        stream_pool()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&ordinal)
            .cloned()
            .unwrap_or_default()
    }

    fn borrow(owner: &Arc<Device>) -> Arc<Device> {
        // SAFETY: `owner` retains the primary context for the duration of
        // every test below, and nothing destroys it concurrently.
        unsafe {
            Device::borrow_raw(
                owner.cu_ctx().cast(),
                owner.cu_device() as c_int,
                owner.ordinal(),
            )
        }
    }

    #[test]
    fn ordinals_beyond_c_int_are_rejected_not_truncated() {
        let invalid = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_DEVICE);
        // Checked before `cuInit`, so this holds with or without a driver.
        assert_eq!(Device::new(usize::MAX).err(), Some(invalid));
        assert_eq!(Device::raw_device(usize::MAX).err(), Some(invalid));
        // The first ordinal past `c_int` would have wrapped to device 0.
        let wraps_to_zero = (c_int::MAX as usize) + 1 + (c_int::MAX as usize) + 1;
        assert_eq!(Device::new(wraps_to_zero).err(), Some(invalid));
        assert_eq!(ordinal_to_c_int(0), Ok(0));
        assert_eq!(ordinal_to_c_int(c_int::MAX as usize), Ok(c_int::MAX));
    }

    #[test]
    fn owned_device_streams_are_parked_and_reused() {
        if !has_gpu() {
            return;
        }
        let _serialized = pool_tests_lock();
        let owner = Device::new(0).unwrap();
        let first = owner.new_stream().unwrap();
        let handle = first.cu_stream() as usize;
        drop(first);
        assert!(
            pooled_handles(0).contains(&handle),
            "an owned device's stream is parked, not destroyed"
        );
        let second = owner.new_stream().unwrap();
        assert_eq!(second.cu_stream() as usize, handle, "and handed out again");
    }

    #[test]
    fn borrowed_device_streams_never_enter_the_pool() {
        if !has_gpu() {
            return;
        }
        let _serialized = pool_tests_lock();
        let owner = Device::new(0).unwrap();
        // Seed the pool with a handle from the owned device so that a
        // borrowed `new_stream` that consulted the pool would be caught.
        let parked = {
            let seed = owner.new_stream().unwrap();
            seed.cu_stream() as usize
        };
        assert!(pooled_handles(0).contains(&parked));

        let borrowed = borrow(&owner);
        let stream = borrowed.new_stream().unwrap();
        let handle = stream.cu_stream() as usize;
        assert_ne!(
            handle, parked,
            "a borrowed device must not pop a pooled handle: the pool's handles \
             belong to contexts this crate retained, not to the borrowed one"
        );
        assert!(
            pooled_handles(0).contains(&parked),
            "and leaves the pool as it was"
        );

        unsafe { stream.synchronize() }.unwrap();
        drop(stream);
        assert!(
            !pooled_handles(0).contains(&handle),
            "a borrowed device's stream must be destroyed on drop, never parked \
             where a later owned Device for this ordinal could pop it"
        );
        assert!(pooled_handles(0).contains(&parked));
    }

    #[test]
    fn panicking_host_function_is_caught_and_still_reclaimed() {
        let (probe, calls, drops) = Probe::new();

        let result = Stream::enqueue_boxed(
            move || {
                probe.calls.fetch_add(1, Ordering::SeqCst);
                panic!("callback panic must not cross the C ABI");
            },
            |trampoline, user_data| {
                // SAFETY: as above; the trampoline must swallow the panic.
                unsafe { trampoline(user_data) };
                Ok(())
            },
        );

        assert_eq!(result, Ok(()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
