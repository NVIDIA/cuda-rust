/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOp, ExecutionContext};
use crate::error::DeviceError;
use anyhow::{Context, Result};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{launch_kernel, DType, Function, LaunchConfig, Stream};
use std::ffi::c_void;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::vec::Vec;

/// A builder for asynchronously launching a CUDA kernel on a stream.
///
/// Arguments are copied into an inline arena by [`push_arg`](Self::push_arg) /
/// [`push_device_ptr`](Self::push_device_ptr) and handed to the driver as a
/// `*mut c_void` array at launch; the driver copies the parameter values out
/// before `cuLaunchKernel` returns, so nothing outlives the launch.
#[derive(Debug)]
pub struct AsyncKernelLaunch {
    pub func: Arc<Function>,
    args: KernelArgStorage,
    cfg: Option<LaunchConfig>,
    programmatic_dependent_launch: bool,
}

/// Inline capacity of the argument arena, in 16-byte value slots and in
/// arguments. A tensor parameter takes `1 + 2 * rank` slots (pointer, shape,
/// strides), so this covers four rank-3 tensors plus scalars; longer
/// parameter lists spill to the heap.
const INLINE_SLOTS: usize = 32;

/// Type-erased kernel argument values, stored in 16-byte slots.
///
/// `cuLaunchKernel` takes an array of `*mut c_void` pointing at the parameter
/// VALUES. Every value pushed here is a plain `Copy` scalar or device pointer
/// (the only two `push` callers), so the storage is a bump arena with no
/// destructor bookkeeping. It lives inline for the common parameter count
/// and moves to the heap only beyond [`INLINE_SLOTS`]; the pointer array is
/// materialized only at launch, after every push, so growth can never
/// invalidate a recorded pointer.
enum KernelArgStorage {
    Inline {
        /// 16-byte-aligned value slots; the first `values_len` are occupied. A
        /// padded value leaves its padding bytes uninitialized.
        values: [MaybeUninit<u128>; INLINE_SLOTS],
        /// Slot index of each argument, in push order; the first `args_len` are set.
        offsets: [u8; INLINE_SLOTS],
        values_len: usize,
        args_len: usize,
    },
    Heap {
        /// The same slots as `Inline::values`, never read as `u128`: a value
        /// with padding leaves those bytes uninitialized, so the slots stay
        /// `MaybeUninit` and are only ever copied or pointed at.
        values: Vec<MaybeUninit<u128>>,
        offsets: Vec<usize>,
    },
}

impl Default for KernelArgStorage {
    fn default() -> Self {
        Self::Inline {
            values: [const { MaybeUninit::uninit() }; INLINE_SLOTS],
            offsets: [0; INLINE_SLOTS],
            values_len: 0,
            args_len: 0,
        }
    }
}

impl Debug for KernelArgStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("KernelArgStorage");
        match self {
            Self::Inline {
                offsets, args_len, ..
            } => d
                .field("len", args_len)
                .field("offsets", &&offsets[..*args_len]),
            Self::Heap { offsets, .. } => d.field("len", &offsets.len()).field("offsets", offsets),
        };
        d.finish()
    }
}

impl KernelArgStorage {
    /// Copies `arg` into the arena. `Copy` makes the missing destructor
    /// bookkeeping sound: dropping the arena drops nothing.
    fn push<T: Copy + Send>(&mut self, arg: T) {
        const SLOT: usize = std::mem::size_of::<u128>();
        const {
            assert!(std::mem::align_of::<T>() <= SLOT);
        }
        let slots = std::mem::size_of::<T>().div_ceil(SLOT).max(1);
        if let Self::Inline {
            values,
            offsets,
            values_len,
            args_len,
        } = self
        {
            if *values_len + slots <= INLINE_SLOTS && *args_len < INLINE_SLOTS {
                let offset = *values_len;
                for slot in &mut values[offset..offset + slots] {
                    slot.write(0);
                }
                // SAFETY: the pointer derives from the whole array, so a value
                // spanning several slots stays inside the borrowed range; the
                // span is at least `size_of::<T>()` bytes and 16-byte aligned,
                // which the const assertion above bounds `T`'s alignment by;
                // `T: Copy` so overwriting drops nothing.
                unsafe { std::ptr::write(values.as_mut_ptr().add(offset).cast::<T>(), arg) };
                offsets[*args_len] = offset as u8;
                *values_len += slots;
                *args_len += 1;
                return;
            }
            // Out of inline room: continue on the heap with everything so far.
            // Slots move as `MaybeUninit`; a padded value's padding bytes are
            // uninitialized and must never be read as an integer.
            let mut heap_values = Vec::with_capacity(2 * (*values_len + slots));
            heap_values.extend_from_slice(&values[..*values_len]);
            let mut heap_offsets = Vec::with_capacity(2 * (*args_len + 1));
            heap_offsets.extend(offsets[..*args_len].iter().map(|&o| o as usize));
            *self = Self::Heap {
                values: heap_values,
                offsets: heap_offsets,
            };
        }
        let Self::Heap { values, offsets } = self else {
            unreachable!("inline arena handled above");
        };
        let offset = values.len();
        values.resize(offset + slots, MaybeUninit::new(0));
        // SAFETY: as above; the span was just reserved and zeroed.
        unsafe { std::ptr::write(values.as_mut_ptr().add(offset).cast::<T>(), arg) };
        offsets.push(offset);
    }

    /// Runs `f` on the parameter pointer array in push order, as
    /// `cuLaunchKernel` wants it. The array is built on the stack for an
    /// inline arena, so a launch with the common parameter count allocates
    /// nothing.
    fn with_param_ptrs<R>(&mut self, f: impl FnOnce(&mut [*mut c_void]) -> R) -> R {
        match self {
            Self::Inline {
                values,
                offsets,
                args_len,
                ..
            } => {
                let base = values.as_mut_ptr();
                // Uninitialized until written. A null fill would be a real
                // pointer value, and the slice passed to `f` must contain
                // only the pointers written below.
                let mut ptrs: [MaybeUninit<*mut c_void>; INLINE_SLOTS] =
                    [const { MaybeUninit::uninit() }; INLINE_SLOTS];
                for (slot, &offset) in ptrs.iter_mut().zip(&offsets[..*args_len]) {
                    // SAFETY: every offset was a valid slot index at push time.
                    slot.write(unsafe { base.add(offset as usize) } as *mut c_void);
                }
                // SAFETY: the prefix of length `args_len` was written above,
                // and `args_len` is at most `INLINE_SLOTS`. `MaybeUninit<T>`
                // has the same layout as `T`.
                let ptrs = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptrs.as_mut_ptr().cast::<*mut c_void>(),
                        *args_len,
                    )
                };
                f(ptrs)
            }
            Self::Heap { values, offsets } => {
                let base = values.as_mut_ptr();
                let mut ptrs: Vec<*mut c_void> = offsets
                    .iter()
                    // SAFETY: every offset was a valid slot index at push time
                    // and the arena only grows.
                    .map(|&offset| unsafe { base.add(offset) } as *mut c_void)
                    .collect();
                f(&mut ptrs)
            }
        }
    }
}

impl AsyncKernelLaunch {
    /// Creates a new kernel launch builder for the given CUDA function.
    pub fn new(func: Arc<Function>) -> AsyncKernelLaunch {
        AsyncKernelLaunch {
            func,
            args: KernelArgStorage::default(),
            cfg: None,
            programmatic_dependent_launch: false,
        }
    }

    /// Pushes a kernel argument by value.
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Pushes a kernel argument from an `Arc` reference.
    #[inline(always)]
    pub fn push_arg_arc<T: ArcKernelArgument>(&mut self, arg: &Arc<T>) -> &mut Self {
        arg.push_arg_arc(self);
        self
    }

    /// Pushes a device pointer as a kernel argument.
    ///
    /// # Safety
    /// `ptr` must stay a valid device allocation, on the device that owns the
    /// stream this launch is executed on, until the kernel has completed — not
    /// merely until the launch is submitted. The kernel signature must expect a
    /// pointer at this position.
    pub unsafe fn push_device_ptr(&mut self, ptr: CUdeviceptr) -> &mut Self {
        self.push_arg_raw(ptr)
    }

    /// Pushes a raw argument to the kernel parameter list.
    ///
    /// # Safety
    /// `T` must match the size and alignment of the kernel's formal parameter
    /// at this position; the driver copies `size_of::<T>()` bytes from the
    /// stored value.
    unsafe fn push_arg_raw<T: Copy + Send>(&mut self, arg: T) -> &mut Self {
        self.args.push(arg);
        self
    }

    /// Sets the grid/block dimensions and shared memory configuration for the launch.
    pub fn set_launch_config(&mut self, cfg: LaunchConfig) -> &mut Self {
        self.cfg = Some(cfg);
        self
    }

    /// Enable programmatic dependent launch for this submission. Off by default.
    ///
    /// # Safety
    /// The kernel must wait for predecessor completion before dependent memory
    /// accesses (token-ordered after `gdc_wait_tko` for Tile kernels). Work before
    /// the wait must not race predecessor work. Neither kernel may depend on
    /// overlap, and all resources must remain valid through their last use.
    pub unsafe fn programmatic_dependent_launch(&mut self) -> &mut Self {
        self.programmatic_dependent_launch = true;
        self
    }

    /// Launches the kernel on the given CUDA stream.
    ///
    /// # Safety
    /// The caller must ensure the kernel arguments and launch config are valid.
    unsafe fn launch(mut self, stream: &Arc<Stream>) -> Result<(), DeviceError> {
        let cfg = self.cfg.ok_or_else(|| {
            DeviceError::Launch("Await called before launching the kernel.".to_string())
        })?;
        let launch = if self.programmatic_dependent_launch {
            let architecture = cuda_core::get_device_sm_name(stream.device().cu_device())?;
            let sm: u32 = architecture
                .strip_prefix("sm_")
                .and_then(|s| s.trim_end_matches(['a', 'f']).parse().ok())
                .unwrap_or(0);
            if sm < 90 {
                return Err(DeviceError::Launch(format!(
                    "programmatic dependent launch requires sm_90 or newer; target {architecture}"
                )));
            }
            cuda_core::launch_kernel_pdl
        } else {
            launch_kernel
        };
        let func = self.func.cu_function();
        self.args
            .with_param_ptrs(|params| {
                launch(
                    func,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    stream.cu_stream(),
                    params,
                )
            })
            .with_context(|| {
                format!(
                    r#"
                Failed to launch kernel.
                args: {:#?}
                cfg: {:#?}"#,
                    self.args, cfg
                )
            })?;
        Ok(())
    }
}

/// A kernel argument that can be pushed from an `Arc` reference.
pub trait ArcKernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunch);
}

/// A kernel argument that can be pushed by value into an `AsyncKernelLaunch`.
pub trait KernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg(self, launcher: &mut AsyncKernelLaunch);
}

/// Safe implementation for scalar types. Values implementing `DType` are copied
/// into the kernel's parameter space during launch — the kernel reads the value,
/// not a device pointer, so no `unsafe` is required.
impl<T: DType> KernelArgument for T {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        // SAFETY: a `DType` scalar is a plain `Copy` value with the layout the
        // compiled kernel declares for that scalar type; the launcher's
        // signature validation (in cutile) checks the position, and the value
        // is copied out by the driver at launch, so nothing outlives the box.
        unsafe {
            launcher.push_arg_raw(self);
        }
    }
}

impl DeviceOp for AsyncKernelLaunch {
    type Output = ();

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.launch(ctx.get_cuda_stream())
    }
}

impl IntoFuture for AsyncKernelLaunch {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            let mut f = DeviceFuture::new();
            f.device_operation = Some(self);
            f.execution_context = Some(ExecutionContext::new(stream));
            Ok(f)
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

#[cfg(test)]
mod arg_storage_tests {
    //! Host-only: the storage never touches the driver.

    use super::*;

    /// Reads a parameter back the way the driver does: `size_of::<T>()`
    /// bytes at a pointer `with_param_ptrs` handed out. Such a pointer
    /// addresses a slot of the arena that owns it, 16-byte aligned, and
    /// stays valid for the closure's duration; every caller names the `T`
    /// the slot was pushed with.
    fn param<T: Copy>(ptr: *mut c_void) -> T {
        // SAFETY: see above.
        unsafe { ptr.cast::<T>().read() }
    }

    /// Values of every accepted width roundtrip through the arena, and the
    /// pointer array, materialized after all pushes, points at the values
    /// even after the arena moved to the heap while growing.
    #[test]
    fn values_roundtrip_and_survive_arena_growth() {
        let mut storage = KernelArgStorage::default();
        storage.push(7u8);
        storage.push(0x1122_3344_5566_7788u64);
        storage.push(-5i32);
        assert!(matches!(storage, KernelArgStorage::Inline { .. }));
        for i in 0..64u64 {
            storage.push(i);
        }
        assert!(matches!(storage, KernelArgStorage::Heap { .. }));
        storage.with_param_ptrs(|ptrs| {
            assert_eq!(ptrs.len(), 3 + 64);
            assert_eq!(param::<u8>(ptrs[0]), 7);
            assert_eq!(param::<u64>(ptrs[1]), 0x1122_3344_5566_7788);
            assert_eq!(param::<i32>(ptrs[2]), -5);
            for i in 0..64usize {
                assert_eq!(param::<u64>(ptrs[3 + i]), i as u64);
            }
        });
    }

    /// A parameter list within the inline capacity never leaves the inline
    /// arena, and its pointers read back the pushed values.
    #[test]
    fn common_parameter_counts_stay_inline() {
        let mut storage = KernelArgStorage::default();
        for i in 0..INLINE_SLOTS {
            storage.push(i as u32);
        }
        assert!(matches!(storage, KernelArgStorage::Inline { .. }));
        storage.with_param_ptrs(|ptrs| {
            assert_eq!(ptrs.len(), INLINE_SLOTS);
            for (i, &ptr) in ptrs.iter().enumerate() {
                assert_eq!(param::<u32>(ptr), i as u32);
            }
        });
        storage.push(1u8);
        assert!(matches!(storage, KernelArgStorage::Heap { .. }));
    }

    /// A value with padding leaves its padding bytes uninitialized after the
    /// typed write, so spilling must move slots without reading them as
    /// integers (Miri flags the alternative).
    #[test]
    fn padded_values_survive_the_spill() {
        #[derive(Clone, Copy)]
        #[repr(C)]
        struct Padded {
            a: u8,
            b: u32,
        }
        let mut storage = KernelArgStorage::default();
        for i in 0..INLINE_SLOTS + 4 {
            storage.push(Padded {
                a: i as u8,
                b: i as u32 * 3,
            });
        }
        assert!(matches!(storage, KernelArgStorage::Heap { .. }));
        storage.with_param_ptrs(|ptrs| {
            assert_eq!(ptrs.len(), INLINE_SLOTS + 4);
            for (i, &ptr) in ptrs.iter().enumerate() {
                let value = param::<Padded>(ptr);
                assert_eq!((value.a, value.b), (i as u8, i as u32 * 3));
            }
        });
    }

    /// A value wider than one slot is written through a pointer derived from
    /// the whole arena, not from its first slot, so the write stays within
    /// the borrowed range (Miri flags the alternative). Covers both arenas.
    #[test]
    fn multi_slot_values_roundtrip_inline_and_on_the_heap() {
        let wide = [11u64, 22, 33];
        let mut storage = KernelArgStorage::default();
        storage.push(wide);
        storage.push(7u8);
        assert!(matches!(storage, KernelArgStorage::Inline { .. }));
        storage.with_param_ptrs(|ptrs| {
            assert_eq!(param::<[u64; 3]>(ptrs[0]), wide);
            assert_eq!(param::<u8>(ptrs[1]), 7);
        });
        for _ in 0..INLINE_SLOTS {
            storage.push(wide);
        }
        assert!(matches!(storage, KernelArgStorage::Heap { .. }));
        storage.with_param_ptrs(|ptrs| {
            assert_eq!(ptrs.len(), 2 + INLINE_SLOTS);
            assert_eq!(param::<[u64; 3]>(ptrs[0]), wide);
            assert_eq!(param::<[u64; 3]>(ptrs[ptrs.len() - 1]), wide);
        });
    }

    /// Every parameter pointer is aligned for its slot (16 bytes), which
    /// bounds all accepted argument types; the `const` assertion in `push`
    /// rejects wider alignments at compile time.
    #[test]
    fn slots_are_sixteen_byte_aligned() {
        let mut storage = KernelArgStorage::default();
        storage.push(1u8);
        storage.push(2u128);
        storage.with_param_ptrs(|ptrs| {
            assert!(ptrs.iter().all(|&p| (p as usize).is_multiple_of(16)));
            assert_eq!(param::<u128>(ptrs[1]), 2);
        });
    }
}
