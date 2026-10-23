/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.
//!
//! [`AsyncKernelLaunchBuilder`] safely accumulates a kernel function reference,
//! launch attributes, and type-erased argument pointers. An explicit unsafe
//! call to [`AsyncKernelLaunchBuilder::finalize_unchecked`] supplies the raw
//! launch geometry and returns an immutable [`AsyncKernelLaunch`] that can be
//! scheduled as a [`DeviceOp`].
//!
//! Arguments use an inline arena with a heap fallback and are freed when
//! the launcher is dropped after submission. This keeps the pointed-to values
//! alive until `cuLaunchKernel` / `cuLaunchKernelEx` has copied the launch
//! parameter values out of the host-side argument array.
//!
//! [`DeviceOp`]: crate::device_operation::DeviceOp

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOp, ExecutionContext};
use crate::error::DeviceError;

use cuda_core::CudaFunction;
use cuda_core::LaunchConfig;
use std::ffi::c_void;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Safe, inert builder for an unchecked CUDA kernel launch.
///
/// Building a launch does not enqueue GPU work. The builder deliberately does
/// not implement [`DeviceOp`] or [`IntoFuture`]: raw launch geometry
/// must first cross the explicit unsafe boundary at
/// [`finalize_unchecked`](Self::finalize_unchecked).
///
/// ```compile_fail,E0277
/// use cuda_async::device_operation::DeviceOp;
/// use cuda_async::launch::AsyncKernelLaunchBuilder;
///
/// fn schedule<O: DeviceOp>(_operation: O) {}
///
/// fn cannot_schedule(builder: AsyncKernelLaunchBuilder<'static>) {
///     schedule(builder);
/// }
/// ```
///
/// Builders stay on the thread that assembles their type-erased argument
/// storage. Only the immutable finalized operation is `Send`:
///
/// ```compile_fail,E0277
/// use cuda_async::launch::AsyncKernelLaunchBuilder;
///
/// fn assert_send<T: Send>() {}
/// assert_send::<AsyncKernelLaunchBuilder<'static>>();
/// ```
#[derive(Debug)]
pub struct AsyncKernelLaunchBuilder<'a> {
    /// Handle to the compiled device function.
    func: Arc<CudaFunction>,
    resources: Vec<(cuda_core::RawDeviceBuffer, bool)>,
    /// Owned, type-erased argument storage passed to the CUDA driver.
    args: KernelArgStorage,
    configured: Option<LaunchConfig>,
    /// Optional thread-block cluster dimensions for `cuLaunchKernelEx`.
    cluster_dim: Option<(u32, u32, u32)>,
    /// When `true`, launch via `cuLaunchKernelEx` with
    /// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` set, which guarantees the whole grid
    /// is co-resident on the device (required for `cuda_device::grid::sync()`).
    cooperative: bool,
    programmatic_dependent_launch: bool,
    /// Ties borrowed device buffers to the lazy launch operation.
    _borrows: PhantomData<&'a mut ()>,
}

/// Immutable, runnable CUDA kernel launch.
///
/// This type can only be created by consuming an
/// [`AsyncKernelLaunchBuilder`]. It exposes no safe geometry or argument
/// setters, so its safety assumptions cannot be invalidated after the raw
/// launch configuration has been accepted.
///
/// ```compile_fail,E0599
/// use cuda_async::launch::AsyncKernelLaunch;
/// use cuda_core::LaunchConfig;
///
/// fn cannot_reconfigure(launch: &mut AsyncKernelLaunch<'_>, config: LaunchConfig) {
///     launch.set_launch_config(config);
/// }
/// ```
/// Chaining finalized launches remains `Send`, including borrowed launches:
///
/// ```
/// use cuda_async::{device_operation::DeviceOp, launch::AsyncKernelLaunch};
/// fn assert_send(_: impl Send) {}
/// fn check<'a>(first: AsyncKernelLaunch<'a>, second: AsyncKernelLaunch<'a>, third: AsyncKernelLaunch<'a>) {
///     assert_send(async move {
///         first.then(move |_| second).and_then_with_context(move |_, ()| third).await
///     });
/// }
/// ```
#[derive(Debug)]
pub struct AsyncKernelLaunch<'a> {
    /// Handle to the compiled device function.
    func: Arc<CudaFunction>,
    resources: Vec<(cuda_core::RawDeviceBuffer, bool)>,
    /// Owned, type-erased argument storage passed to the CUDA driver.
    args: KernelArgStorage,
    /// Complete grid/block dimensions and dynamic shared-memory size.
    cfg: LaunchConfig,
    /// Optional thread-block cluster dimensions for `cuLaunchKernelEx`.
    cluster_dim: Option<(u32, u32, u32)>,
    /// Whether this launch requests cooperative grid residency.
    cooperative: bool,
    programmatic_dependent_launch: bool,
    /// Ties borrowed device buffers to the lazy launch operation.
    _borrows: PhantomData<&'a mut ()>,
}

/// # Safety
///
/// Argument values are owned by the launch. Non-`Copy` values enter only
/// through the explicit boxed argument method, which requires `Send`;
/// copied scalar values have no destructor. The runnable type exposes none of
/// those erased values to safe Rust. The `Arc<CudaFunction>` is `Send + Sync`,
/// and `_borrows` preserves source lifetimes captured while building.
unsafe impl<'a> Send for AsyncKernelLaunch<'a> {}

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
    Mixed {
        plain: Box<KernelArgStorage>,
        boxed: Vec<ErasedArgument>,
        order: Vec<ArgumentIndex>,
    },
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
            Self::Mixed { order, .. } => d.field("len", &order.len()),
            Self::Heap { offsets, .. } => d.field("len", &offsets.len()).field("offsets", offsets),
        };
        d.finish()
    }
}

impl KernelArgStorage {
    /// Copies `arg` into the arena. `Copy` makes the missing destructor
    /// bookkeeping sound: dropping the arena drops nothing.
    fn push<T: Copy>(&mut self, arg: T) {
        const SLOT: usize = std::mem::size_of::<u128>();
        if std::mem::align_of::<T>() > SLOT {
            self.push_box(Box::new(arg));
            return;
        }
        if let Self::Mixed { plain, order, .. } = self {
            let index = plain.len();
            plain.push(arg);
            order.push(ArgumentIndex::Plain(index));
            return;
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
                // which the alignment guard above bounds `T`'s alignment by;
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
            Self::Mixed {
                plain,
                boxed,
                order,
            } => {
                let ptrs = plain.param_vec();
                let mut all: Vec<*mut c_void> = order
                    .iter()
                    .map(|index| match index {
                        ArgumentIndex::Plain(i) => ptrs[*i],
                        ArgumentIndex::Boxed(i) => boxed[*i].ptr,
                    })
                    .collect();
                f(&mut all)
            }
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

#[derive(Debug)]
struct ErasedArgument {
    ptr: *mut c_void,
    drop_value: unsafe fn(*mut c_void),
}
impl Drop for ErasedArgument {
    fn drop(&mut self) {
        unsafe { (self.drop_value)(self.ptr) };
    }
}
enum ArgumentIndex {
    Plain(usize),
    Boxed(usize),
}
impl KernelArgStorage {
    fn param_vec(&mut self) -> Vec<*mut c_void> {
        self.with_param_ptrs(|p| p.to_vec())
    }
    fn len(&self) -> usize {
        match self {
            Self::Inline { args_len, .. } => *args_len,
            Self::Heap { offsets, .. } => offsets.len(),
            Self::Mixed { order, .. } => order.len(),
        }
    }
    fn push_box<T>(&mut self, value: Box<T>) {
        unsafe fn drop_box<T>(ptr: *mut c_void) {
            drop(unsafe { Box::from_raw(ptr.cast::<T>()) });
        }
        if !matches!(self, Self::Mixed { .. }) {
            let order = (0..self.len()).map(ArgumentIndex::Plain).collect();
            let plain = Box::new(std::mem::take(self));
            *self = Self::Mixed {
                plain,
                boxed: Vec::new(),
                order,
            };
        }
        let Self::Mixed { boxed, order, .. } = self else {
            unreachable!()
        };
        order.push(ArgumentIndex::Boxed(boxed.len()));
        boxed.push(ErasedArgument {
            ptr: Box::into_raw(value).cast(),
            drop_value: drop_box::<T>,
        });
    }
}

impl<'a> AsyncKernelLaunchBuilder<'a> {
    /// Creates an inert builder for `func` with no arguments or launch
    /// attributes.
    pub fn new(func: Arc<CudaFunction>) -> Self {
        Self {
            func,
            resources: Vec::new(),
            args: KernelArgStorage::default(),
            cluster_dim: None,
            cooperative: false,
            programmatic_dependent_launch: false,
            configured: None,
            _borrows: PhantomData,
        }
    }

    /// Push a raw CUDA address. Its pointee must remain valid through completion.
    /// # Safety
    /// The address, extent, element type and permissions must match the kernel.
    pub unsafe fn push_device_ptr(&mut self, ptr: cuda_core::sys::CUdeviceptr) -> &mut Self {
        self.push_scalar_arg(ptr)
    }
    pub fn push_arg_arc<T: ArcKernelArgument>(&mut self, arg: &Arc<T>) -> &mut Self {
        arg.push_arg_arc(self);
        self
    }
    /// Sets the geometry on this inert builder. It cannot be scheduled.
    pub fn set_launch_config(&mut self, config: LaunchConfig) -> &mut Self {
        self.configured = Some(config);
        self
    }
    /// Executes a configured builder using the same sealed launch operation.
    /// # Safety
    /// All requirements of `finalize_unchecked` apply to the configured launch.
    pub unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        let config = self
            .configured
            .ok_or_else(|| DeviceError::Launch("missing launch configuration".into()))?;
        unsafe { self.finalize_unchecked(config).execute(ctx) }
    }
    /// Enable programmatic dependent launch.
    /// # Safety
    /// The kernel must perform the required dependency wait before dependent
    /// accesses; neither kernel may require overlap. Retain all used resources.
    pub unsafe fn programmatic_dependent_launch(&mut self) -> &mut Self {
        self.programmatic_dependent_launch = true;
        self
    }
    /// Pushes an explicitly typed, heap-owned argument without a DeviceCopy bound.
    /// # Safety
    /// The object's layout and initialized representation must match the kernel
    /// parameter. It must not carry host ownership the device can dereference.
    pub unsafe fn push_boxed_arg_unchecked<T: Send + 'static>(&mut self, arg: Box<T>) -> &mut Self {
        self.args.push_box(arg);
        self
    }

    /// Retains a shared allocation and records its access mode for submission.
    pub fn retain_buffer(&mut self, storage: cuda_core::RawDeviceBuffer, write: bool) {
        self.resources.push((storage, write));
    }

    /// Appends a by-value `Copy` argument to the kernel launch packet.
    ///
    /// This is the typed-module path used for scalar, raw-pointer, custom
    /// `Copy` structs, and `Copy` closure arguments.
    #[inline(always)]
    pub fn push_scalar_arg<T: Copy + 'a>(&mut self, arg: T) -> &mut Self {
        self.args.push(arg);
        self
    }

    /// Appends a kernel argument to owned storage for the driver call.
    /// Common scalar arguments stay in the inline arena; large argument lists
    /// and over-aligned types use heap storage.
    ///
    /// The allocated storage remains alive until launch submission finishes or
    /// the builder is dropped without launching.
    ///
    /// Values with thread-affine ownership cannot be hidden in the erased
    /// storage that will later become a `Send` operation:
    ///
    /// ```compile_fail,E0277
    /// use cuda_async::launch::AsyncKernelLaunchBuilder;
    /// use std::rc::Rc;
    ///
    /// fn reject_rc(mut builder: AsyncKernelLaunchBuilder<'static>) {
    ///     builder.push_arg(Box::new(Rc::new(1_u32)));
    /// }
    /// ```
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Appends multiple kernel arguments at once from a tuple.
    ///
    /// Equivalent to chained [`push_arg`](Self::push_arg) calls but allows
    /// grouping all arguments in a single expression:
    ///
    /// ```ignore
    /// launch.push_args((m, n, k, alpha, a_ptr, a_len, b_ptr, b_len, beta, c_ptr, c_len));
    /// ```
    ///
    /// Supports tuples up to 32 elements.
    #[inline(always)]
    pub fn push_args<T: KernelArguments>(&mut self, args: T) -> &mut Self {
        args.push_args(self);
        self
    }

    /// Sets thread-block cluster dimensions for a cluster launch.
    pub fn set_cluster_dim(&mut self, cluster_dim: (u32, u32, u32)) -> &mut Self {
        self.cluster_dim = Some(cluster_dim);
        self
    }

    /// Marks this launch as cooperative (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).
    ///
    /// A cooperative launch guarantees every block in the grid is co-resident
    /// on the device, which is the precondition for grid-wide barriers like
    /// `cuda_device::grid::sync()`. May be combined with
    /// [`set_cluster_dim`](Self::set_cluster_dim); both attributes are then
    /// passed to `cuLaunchKernelEx` in the same call.
    pub fn set_cooperative(&mut self, cooperative: bool) -> &mut Self {
        self.cooperative = cooperative;
        self
    }

    /// Accepts an unverified raw launch configuration and seals this builder
    /// into an immutable, schedulable operation.
    ///
    /// Prefer the generated typed module APIs for application code. They check
    /// a kernel's declared launch contract before constructing the runnable
    /// operation.
    ///
    /// # Safety
    ///
    /// The caller must prove all facts that a typed prepared launch normally
    /// checks:
    ///
    /// - `cfg` must match the kernel's indexing dimensionality, launch bounds,
    ///   block shape, and dynamic shared-memory requirements;
    /// - the cluster and cooperative attributes must be legal for the kernel;
    /// - the function signature must exactly match the arguments, in order,
    ///   type, size, and alignment;
    /// - referenced device allocations must remain valid for the complete GPU
    ///   operation and satisfy Rust's aliasing requirements; and
    /// - the scheduling policy must select a stream from the CUDA context that
    ///   owns `func`.
    ///
    /// Violating these requirements can cause memory corruption, data races,
    /// or CUDA driver errors.
    ///
    /// ```compile_fail,E0133
    /// use cuda_async::launch::AsyncKernelLaunchBuilder;
    /// use cuda_core::LaunchConfig;
    ///
    /// fn raw_finalize_requires_unsafe(
    ///     builder: AsyncKernelLaunchBuilder<'static>,
    ///     config: LaunchConfig,
    /// ) {
    ///     let _ = builder.finalize_unchecked(config);
    /// }
    /// ```
    pub unsafe fn finalize_unchecked(self, cfg: LaunchConfig) -> AsyncKernelLaunch<'a> {
        AsyncKernelLaunch {
            func: self.func,
            resources: self.resources,
            args: self.args,
            cfg,
            cluster_dim: self.cluster_dim,
            cooperative: self.cooperative,
            programmatic_dependent_launch: self.programmatic_dependent_launch,
            _borrows: self._borrows,
        }
    }
}

impl<'a> AsyncKernelLaunch<'a> {
    /// Submits the kernel to `stream` via `cuLaunchKernel`, or via
    /// `cuLaunchKernelEx` when cluster dimensions or a cooperative launch
    /// were requested.
    ///
    /// # Safety
    ///
    /// - `self.func` must refer to a kernel loaded from the same CUDA context
    ///   that owns `stream`.
    /// - All argument pointers in `self.args` must point to correctly typed and
    ///   aligned host-side values for the kernel's formal parameters.
    /// - The pointed-to argument values must remain valid until launch
    ///   submission returns. The stream-aware launch helper binds the correct
    ///   context before calling into the CUDA driver.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Driver`] if context binding or launch submission
    /// fails.
    unsafe fn launch(mut self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        let stream = ctx.get_cuda_stream();
        if self.func.context().as_ref() != stream.context().as_ref() {
            return Err(DeviceError::Driver(cuda_core::DriverError(
                cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            )));
        }
        if self.programmatic_dependent_launch && (self.cluster_dim.is_some() || self.cooperative) {
            return Err(DeviceError::Launch(
                "programmatic dependent launch cannot be combined with cluster/cooperative mode"
                    .into(),
            ));
        }
        stream.context().bind_to_thread()?;
        // Register before enqueue: partial failures, unwind and forgotten futures
        // must not shorten the lifetime of any argument or module.
        ctx.retain(self.func.clone())?;
        for (storage, write) in &self.resources {
            ctx.retain_buffer(storage.clone(), *write)?;
        }
        let cfg = self.cfg;
        let function = self.func.clone();
        let cluster = self.cluster_dim;
        let cooperative = self.cooperative;
        let pdl = self.programmatic_dependent_launch;
        if pdl {
            let architecture = cuda_core::get_device_sm_name(stream.context().cu_device())?;
            let sm: u32 = architecture
                .strip_prefix("sm_")
                .and_then(|s| s.trim_end_matches(['a', 'f']).parse().ok())
                .unwrap_or(0);
            if sm < 90 {
                return Err(DeviceError::Launch(format!(
                    "programmatic dependent launch requires sm_90 or newer; target {architecture}"
                )));
            }
        }
        self.args
            .with_param_ptrs(|params| unsafe {
                match (cluster, cooperative, pdl) {
                    (_, _, true) => cuda_core::launch_kernel_pdl(
                        function.cu_function(),
                        cfg.grid_dim,
                        cfg.block_dim,
                        cfg.shared_mem_bytes,
                        stream.cu_stream(),
                        params,
                    ),
                    (Some(cluster), true, false) => {
                        cuda_core::launch_kernel_ex_cooperative_on_stream(
                            &function,
                            cfg.grid_dim,
                            cfg.block_dim,
                            cfg.shared_mem_bytes,
                            cluster,
                            stream,
                            params,
                        )
                    }
                    (Some(cluster), false, false) => cuda_core::launch_kernel_ex_on_stream(
                        &function,
                        cfg.grid_dim,
                        cfg.block_dim,
                        cfg.shared_mem_bytes,
                        cluster,
                        stream,
                        params,
                    ),
                    (None, true, false) => cuda_core::launch_kernel_cooperative_on_stream(
                        &function,
                        cfg.grid_dim,
                        cfg.block_dim,
                        cfg.shared_mem_bytes,
                        stream,
                        params,
                    ),
                    (None, false, false) => cuda_core::launch_kernel_on_stream(
                        &function,
                        cfg.grid_dim,
                        cfg.block_dim,
                        cfg.shared_mem_bytes,
                        stream,
                        params,
                    ),
                }
            })
            .map_err(DeviceError::Driver)
    }
}

/// Returns caller-owned values from a lazy async kernel launch. GPU-visible
/// storage must also be retained in the launch's descriptors: host combinators
/// and cancellation may drop the returned values before GPU completion.
#[derive(Debug)]
pub struct OwnedAsyncKernelLaunch<R: Send> {
    launch: AsyncKernelLaunch<'static>,
    resources: R,
}

impl<R: Send> OwnedAsyncKernelLaunch<R> {
    /// Associates returned values with a launch that already retains its GPU resources.
    pub fn new(launch: AsyncKernelLaunch<'static>, resources: R) -> Self {
        Self { launch, resources }
    }
}

/// A by-value kernel argument encoder. Geometry and signature validation belong
/// to the finalized launch, not to this inert builder.
pub trait KernelArgument {
    fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>);
}
/// All plain device-copyable values use the same marshalling contract in both
/// programming models. Tile dtype validation is performed by the Tile frontend.
impl<T: cuda_core::DeviceCopy> KernelArgument for T {
    fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>) {
        launcher.args.push(self);
    }
}
pub trait ArcKernelArgument {
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunchBuilder<'_>);
}

// ---------------------------------------------------------------------------
// KernelArguments — push multiple heterogeneous args in a single call
// ---------------------------------------------------------------------------

/// Trait for a group of kernel arguments that can be pushed together.
///
/// Implemented for tuples of [`KernelArgument`] types up to arity 32, enabling
/// `launch.push_args((dim_m, dim_n, alpha, ptr, len))` as an alternative to
/// chained `.push_arg()` calls.
#[diagnostic::on_unimplemented(
    message = "cannot push `{Self}` as kernel arguments",
    note = "KernelArguments is implemented for tuples of KernelArgument types up to 32 elements"
)]
pub trait KernelArguments {
    /// Pushes every element into `launcher` in order.
    fn push_args(self, launcher: &mut AsyncKernelLaunchBuilder<'_>);
}

macro_rules! impl_kernel_args_tuple {
    // Base case: empty tuple
    () => {
        impl KernelArguments for () {
            #[inline(always)]
            fn push_args(self, _launcher: &mut AsyncKernelLaunchBuilder<'_>) {}
        }
    };
    // Recursive case: (A, B, C, ...) where each element is a KernelArgument
    ($($idx:tt : $T:ident),+) => {
        impl<$($T: KernelArgument),+> KernelArguments for ($($T,)+) {
            #[inline(always)]
            fn push_args(self, launcher: &mut AsyncKernelLaunchBuilder<'_>) {
                $(launcher.push_arg(self.$idx);)+
            }
        }
    };
}

impl_kernel_args_tuple!();
impl_kernel_args_tuple!(0: A);
impl_kernel_args_tuple!(0: A, 1: B);
impl_kernel_args_tuple!(0: A, 1: B, 2: C);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE, 31: AF);

/// Launches the kernel on the stream bound to the execution context.
impl<'a> DeviceOp for AsyncKernelLaunch<'a> {
    type Output = ();

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        unsafe { self.launch(ctx) }
    }
}

/// Schedules the kernel launch via the thread-local default scheduling policy.
impl<'a> IntoFuture for AsyncKernelLaunch<'a> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch<'a>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| self.schedule(policy)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<R: Send> DeviceOp for OwnedAsyncKernelLaunch<R> {
    type Output = R;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<R, DeviceError> {
        let Self { launch, resources } = self;
        unsafe { launch.launch(ctx) }?;
        Ok(resources)
    }
}

impl<R: Send> IntoFuture for OwnedAsyncKernelLaunch<R> {
    type Output = Result<R, DeviceError>;
    type IntoFuture = DeviceFuture<R, OwnedAsyncKernelLaunch<R>>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| self.schedule(policy)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl crate::device_operation::GraphNode for AsyncKernelLaunch<'_> {}
impl<R: Send> crate::device_operation::GraphNode for OwnedAsyncKernelLaunch<R> {}

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
    /// covers ordinary arguments. Wider alignments use typed heap storage.
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

#[cfg(test)]
mod extra_argument_tests {
    use super::*;
    #[test]
    fn over_aligned_values_keep_order_across_plain_arena_growth() {
        #[derive(Clone, Copy)]
        #[repr(C, align(256))]
        struct Aligned([u8; 256]);
        let mut storage = KernelArgStorage::default();
        storage.push(7u32);
        storage.push(Aligned([19; 256]));
        for i in 0..64u64 {
            storage.push(i);
        }
        storage.with_param_ptrs(|args| unsafe {
            assert_eq!(args.len(), 66);
            assert_eq!(*(args[0] as *const u32), 7);
            assert_eq!(args[1] as usize % 256, 0);
            assert_eq!((*(args[1] as *const Aligned)).0, [19; 256]);
            for i in 0..64 {
                assert_eq!(*(args[i + 2] as *const u64), i as u64);
            }
        });
    }
    #[test]
    fn erased_arguments_drop_with_their_original_type() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Probe(Arc<AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let mut storage = KernelArgStorage::default();
        storage.push_box(Box::new(Probe(drops.clone())));
        drop(storage);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
