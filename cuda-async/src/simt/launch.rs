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
//! scheduled as a [`DeviceOperation`].
//!
//! Arguments are heap-allocated via [`KernelArgument::push_arg`] and freed when
//! the launcher is dropped after submission. This keeps the pointed-to values
//! alive until `cuLaunchKernel` / `cuLaunchKernelEx` has copied the launch
//! parameter values out of the host-side argument array.
//!
//! [`DeviceOperation`]: crate::simt::device_operation::DeviceOperation

use crate::simt::device_context::with_default_device_policy;
use crate::simt::device_future::DeviceFuture;
use crate::simt::device_operation::{DeviceOperation, ExecutionContext};
use crate::simt::error::DeviceError;
use crate::simt::scheduling_policies::SchedulingPolicy;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaFunction, CudaStream};
use std::ffi::c_void;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::Arc;

/// Safe, inert builder for an unchecked CUDA kernel launch.
///
/// Building a launch does not enqueue GPU work. The builder deliberately does
/// not implement [`DeviceOperation`] or [`IntoFuture`]: raw launch geometry
/// must first cross the explicit unsafe boundary at
/// [`finalize_unchecked`](Self::finalize_unchecked).
///
/// ```compile_fail,E0277
/// use cuda_async::simt::device_operation::DeviceOperation;
/// use cuda_async::simt::launch::AsyncKernelLaunchBuilder;
///
/// fn schedule<O: DeviceOperation>(_operation: O) {}
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
/// use cuda_async::simt::launch::AsyncKernelLaunchBuilder;
///
/// fn assert_send<T: Send>() {}
/// assert_send::<AsyncKernelLaunchBuilder<'static>>();
/// ```
#[derive(Debug)]
pub struct AsyncKernelLaunchBuilder<'a> {
    /// Handle to the compiled device function.
    func: Arc<CudaFunction>,
    /// Heap-allocated, type-erased argument storage passed to the CUDA driver.
    args: KernelArgStorage,
    /// Optional thread-block cluster dimensions for `cuLaunchKernelEx`.
    cluster_dim: Option<(u32, u32, u32)>,
    /// When `true`, launch via `cuLaunchKernelEx` with
    /// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` set, which guarantees the whole grid
    /// is co-resident on the device (required for `cuda_device::grid::sync()`).
    cooperative: bool,
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
/// use cuda_async::simt::launch::AsyncKernelLaunch;
/// use cuda_core::simt::LaunchConfig;
///
/// fn cannot_reconfigure(launch: &mut AsyncKernelLaunch<'_>, config: LaunchConfig) {
///     launch.set_launch_config(config);
/// }
/// ```
#[derive(Debug)]
pub struct AsyncKernelLaunch<'a> {
    /// Handle to the compiled device function.
    func: Arc<CudaFunction>,
    /// Heap-allocated, type-erased argument storage passed to the CUDA driver.
    args: KernelArgStorage,
    /// Complete grid/block dimensions and dynamic shared-memory size.
    cfg: LaunchConfig,
    /// Optional thread-block cluster dimensions for `cuLaunchKernelEx`.
    cluster_dim: Option<(u32, u32, u32)>,
    /// Whether this launch requests cooperative grid residency.
    cooperative: bool,
    /// Ties borrowed device buffers to the lazy launch operation.
    _borrows: PhantomData<&'a mut ()>,
}

/// # Safety
///
/// The argument pointers refer to heap allocations owned by the launch.
/// Non-`Copy` values enter that storage only through `KernelArgument: Send`;
/// copied scalar values have no destructor. The runnable type exposes none of
/// those erased values to safe Rust. The `Arc<CudaFunction>` is `Send + Sync`,
/// and `_borrows` preserves source lifetimes captured while building.
unsafe impl<'a> Send for AsyncKernelLaunch<'a> {}

#[derive(Debug, Default)]
struct KernelArgStorage {
    ptrs: Vec<*mut c_void>,
    drops: Vec<unsafe fn(*mut c_void)>,
}

impl Drop for KernelArgStorage {
    fn drop(&mut self) {
        for (arg, drop_arg) in self.ptrs.drain(..).zip(self.drops.drain(..)) {
            unsafe { drop_arg(arg) };
        }
    }
}

impl KernelArgStorage {
    fn push_send_arg<T: Send>(&mut self, arg: Box<T>) {
        unsafe fn drop_box<T>(arg: *mut c_void) {
            let _ = unsafe { Box::from_raw(arg as *mut T) };
        }

        self.ptrs.push(Box::into_raw(arg) as *mut c_void);
        self.drops.push(drop_box::<T>);
    }

    fn push_scalar_arg<T: Copy>(&mut self, arg: T) {
        unsafe fn drop_copy_box<T: Copy>(arg: *mut c_void) {
            let _ = unsafe { Box::from_raw(arg as *mut T) };
        }

        self.ptrs.push(Box::into_raw(Box::new(arg)) as *mut c_void);
        self.drops.push(drop_copy_box::<T>);
    }

    fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
        &mut self.ptrs
    }
}

impl<'a> AsyncKernelLaunchBuilder<'a> {
    /// Creates an inert builder for `func` with no arguments or launch
    /// attributes.
    pub fn new(func: Arc<CudaFunction>) -> Self {
        Self {
            func,
            args: KernelArgStorage::default(),
            cluster_dim: None,
            cooperative: false,
            _borrows: PhantomData,
        }
    }

    /// Appends a by-value `Copy` argument to the kernel launch packet.
    ///
    /// This is the typed-module path used for scalar, raw-pointer, custom
    /// `Copy` structs, and `Copy` closure arguments.
    #[inline(always)]
    pub fn push_scalar_arg<T: Copy + 'a>(&mut self, arg: T) -> &mut Self {
        self.args.push_scalar_arg(arg);
        self
    }

    /// Appends a kernel argument. The value is heap-allocated and its pointer
    /// stored for the driver call.
    ///
    /// Scalars like `u32`, `f32`, `u64` etc. are auto-boxed -- no need to
    /// wrap them in `Box::new`.
    ///
    /// The allocated storage remains alive until launch submission finishes or
    /// the builder is dropped without launching.
    ///
    /// Values with thread-affine ownership cannot be hidden in the erased
    /// storage that will later become a `Send` operation:
    ///
    /// ```compile_fail,E0277
    /// use cuda_async::simt::launch::AsyncKernelLaunchBuilder;
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
    /// use cuda_async::simt::launch::AsyncKernelLaunchBuilder;
    /// use cuda_core::simt::LaunchConfig;
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
            args: self.args,
            cfg,
            cluster_dim: self.cluster_dim,
            cooperative: self.cooperative,
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
    unsafe fn launch(mut self, stream: &Arc<CudaStream>) -> Result<(), DeviceError> {
        let cfg = self.cfg;
        let result = match (self.cluster_dim, self.cooperative) {
            (Some(cluster_dim), true) => unsafe {
                cuda_core::launch_kernel_ex_cooperative_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    cluster_dim,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (Some(cluster_dim), false) => unsafe {
                cuda_core::launch_kernel_ex_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    cluster_dim,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (None, true) => unsafe {
                cuda_core::launch_kernel_cooperative_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (None, false) => unsafe {
                cuda_core::launch_kernel_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
        };
        result.map_err(DeviceError::Driver)?;
        Ok(())
    }
}

/// Owns resources for a lazy async kernel launch and returns them when the GPU
/// work has completed.
#[derive(Debug)]
pub struct OwnedAsyncKernelLaunch<R: Send> {
    launch: AsyncKernelLaunch<'static>,
    resources: R,
}

unsafe impl<R: Send> Send for OwnedAsyncKernelLaunch<R> {}

impl<R: Send> OwnedAsyncKernelLaunch<R> {
    /// Creates an owned async kernel operation from a prepared launch and the
    /// resources that must stay alive until completion.
    pub fn new(launch: AsyncKernelLaunch<'static>, resources: R) -> Self {
        Self { launch, resources }
    }
}

/// Trait for types that can be marshalled into a CUDA kernel argument list.
///
/// Implementors heap-allocate the value and push a `*mut c_void` into the
/// launcher's argument vector.
///
/// Implemented for all common scalar primitives (`u8`–`u64`, `i8`–`i64`,
/// `f32`, `f64`, `usize`, `isize`, `bool`) so callers can write
/// `.push_arg(42u32)` without manual `Box::new`.
pub trait KernelArgument: Send {
    /// Heap-allocates `self` and appends the pointer to `launcher.args`.
    fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>);
}

/// Passes the box's raw pointer directly as a kernel argument.
///
/// This is the low-level escape hatch. Prefer passing scalars directly -- they
/// are auto-boxed via the blanket scalar impls.
impl<T: Send + 'static> KernelArgument for Box<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>) {
        launcher.args.push_send_arg(self);
    }
}

macro_rules! impl_scalar_kernel_arg {
    ($($t:ty),*) => {
        $(
            impl KernelArgument for $t {
                #[inline(always)]
                fn push_arg(self, launcher: &mut AsyncKernelLaunchBuilder<'_>) {
                    launcher.push_scalar_arg(self);
                }
            }
        )*
    };
}

impl_scalar_kernel_arg!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, bool);

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
impl<'a> DeviceOperation for AsyncKernelLaunch<'a> {
    type Output = ();

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        unsafe { self.launch(ctx.get_cuda_stream()) }
    }
}

/// Schedules the kernel launch via the thread-local default scheduling policy.
impl<'a> IntoFuture for AsyncKernelLaunch<'a> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch<'a>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<R: Send + 'static> DeviceOperation for OwnedAsyncKernelLaunch<R> {
    type Output = R;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<R, DeviceError> {
        let Self { launch, resources } = self;
        unsafe { launch.launch(ctx.get_cuda_stream()) }?;
        Ok(resources)
    }
}

impl<R: Send + 'static> IntoFuture for OwnedAsyncKernelLaunch<R> {
    type Output = Result<R, DeviceError>;
    type IntoFuture = DeviceFuture<R, OwnedAsyncKernelLaunch<R>>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestParams {
        scale: f32,
        bias: i32,
    }

    #[test]
    fn scalar_arg_storage_accepts_custom_copy_value() {
        let mut storage = KernelArgStorage::default();
        let params = TestParams {
            scale: 2.0,
            bias: 3,
        };

        storage.push_scalar_arg(params);

        assert_eq!(storage.ptrs.len(), 1);
        assert_eq!(unsafe { *(storage.ptrs[0] as *const TestParams) }, params);
    }

    #[test]
    fn arg_storage_drops_values_with_their_original_type() {
        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut storage = KernelArgStorage::default();

        storage.push_send_arg(Box::new(DropCounter(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        drop(storage);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
