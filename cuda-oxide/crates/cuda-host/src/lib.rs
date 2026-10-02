/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Required for `type_id_u128`'s use of `core::intrinsics::type_id`. The
// intrinsic is the only way to obtain the same 128-bit hash the backend
// uses while keeping the bound at `T: ?Sized` (stable `TypeId::of` would
// force `T: 'static` on every kernel marker — see `type_id.rs` for why
// that would silently reject non-`'static` borrowing closures).
//
// We accept the `internal_features` warning here: cuda-oxide already ships
// `rustc_codegen_cuda` against `rustc_private` and pins a nightly
// toolchain, so the broader project is firmly inside rustc's internal API
// surface. Anyone trying to lift this helper to a stable crate will hit
// the same gate and have to make the same trade-off there.
#![feature(core_intrinsics)]
#![allow(internal_features)]

//! Host-side utilities for CUDA kernel development.
//!
//! This crate provides CPU-side utilities for preparing data and setting up
//! GPU kernel execution. Unlike `cuda-device` which contains device-side primitives,
//! this crate runs entirely on the host.
//!
//! ## Modules
//!
//! - [`embedded`]: Load `#[cuda_module]` artifact bundles embedded in the host
//!   binary (PTX, cubin, NVVM IR, LTOIR)
//! - [`launch`]: Kernel launch traits (`CudaKernel`, `GenericCudaKernel`)
//! - [`kernel_family`]: Bounded variant menus with validated selection,
//!   overrides, and cache provenance
//! - [`ltoir`]: libNVVM + nvJitLink wrappers (`load_kernel_module`, in-memory
//!   cubin builders, and pre-Blackwell PTX compatibility)
//! - [`tiling`]: Layout transformations for tensor core operations (tcgen05)
//!
//! ## Macros
//!
//! - [`cuda_module`]: Generate a typed embedded-module loader and per-kernel
//!   sync launch methods from an inline kernel module. Enable the `async`
//!   feature for borrowed and owned async launch methods.
//! - [`cuda_launch!`]: Unsafe low-level launch macro. It cannot check
//!   argument count or types, so callers must wrap it in `unsafe { }`. Its
//!   niche is modules loaded at runtime by name; for embedded kernels use
//!   `#[cuda_module]`.
//! - `cuda_launch_async!`: Low-level async launch macro retained for
//!   migration when the `async` feature is enabled.
//!
//! ## Usage
//!
//! ```ignore
//! use cuda_device::{kernel, thread, DisjointSlice};
//! use cuda_host::cuda_module;
//! use cuda_core::simt::LaunchConfig;
//! use cuda_core::{CudaContext, DeviceBuffer};
//!
//! #[cuda_module]
//! mod kernels {
//!     use super::*;
//!
//!     #[kernel]
//!     pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
//! }
//!
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let module = kernels::load(&ctx)?;
//!
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
//! let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//!
//! // SAFETY: this raw configuration is one-dimensional and matches vecadd's
//! // index calculation. Prefer a #[launch_contract] and PreparedLaunch when
//! // the kernel's geometry is known.
//! unsafe {
//!     module.vecadd(
//!         &stream,
//!         LaunchConfig::for_num_elems(N as u32),
//!         &a_dev,
//!         &b_dev,
//!         &mut c_dev,
//!     )?;
//! }
//!
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

pub mod embedded;
pub mod entry_registry;
pub mod kernel_family;
pub mod launch;
pub mod ltoir;
mod ltoir_cache;
pub mod tiling;
pub mod type_id;

pub use kernel_family::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelectionCache,
    KernelSelectionError, KernelSelectionResult, KernelSelector, KernelVariant,
    NoKernelSelectionCache, SelectedVariant, SelectionMode, SelectionSource,
};
pub use launch::{
    CudaKernel, CudaKernelArgument, CudaKernelArgumentKind, CudaKernelScalarKind,
    CudaKernelSignature, GenericCudaKernel, HasLength, KernelScalar, ReadOnly, RowWidth,
    RowWidthOwned, Scalar, WriteOnly, push_kernel_device_slice, push_kernel_row_width_device_slice,
    push_kernel_scalar, read_only_device_buffer_arg, row_width_device_buffer_arg,
    writable_device_buffer_arg,
};
#[doc(hidden)]
pub use type_id::__intern_generic_kernel_name;
pub use type_id::{type_id_u128, type_id_u128_of_val};

#[cfg(feature = "async")]
pub use launch::{
    KernelSliceArg, KernelSliceArgMut, PreparedAsyncKernelLaunch, PreparedOwnedAsyncKernelLaunch,
    load_cuda_module_from_async_context, load_kernel_module_async, new_async_kernel_launch_builder,
    new_owned_async_kernel_launch, new_prepared_async_kernel_launch,
    new_prepared_owned_async_kernel_launch, push_async_kernel_scalar,
    push_async_owned_row_width_device_slice, push_async_read_only_device_slice,
    push_async_row_width_device_slice, push_async_writable_device_slice,
    set_async_kernel_cluster_dim, set_async_kernel_cooperative,
};

/// The shared async crate, re-exported whole. Its root is cutile's Tile API;
/// the SIMT surface cuda-host builds on is `cuda_host::cuda_async::simt::*`.
/// Same-named root items such as `DeviceError` or `init_device_contexts` are
/// the Tile ones and do not interoperate with the generated launch methods.
#[cfg(feature = "async")]
pub use cuda_async;
#[cfg(feature = "async")]
pub use cuda_async::simt::launch::{
    AsyncKernelLaunch, AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch,
};

pub use embedded::{
    EmbeddedModuleError, load_all_ptx_bundles_merged, load_embedded_module,
    load_embedded_module_from_anchor, load_first_embedded_module, merge_ptx_bundles,
};
pub use entry_registry::{
    diagnose_generic_kernel_load_error, divergent_type_id_entries, panic_generic_kernel_load_failed,
};
/// Loads a compiled kernel module by name. It prefers PTX, then
/// handles NVVM IR (`<name>.ll`) or an existing `<name>.ltoir`, and finally a
/// standalone cubin. NVVM inputs become a same-target cubin, except that an
/// artifact built for a standard pre-Blackwell target, such as `sm_86`, may be
/// converted to PTX and JIT-compiled by the driver on Blackwell. Most kernels
/// emit PTX directly. See [`ltoir`] for the complete lookup and compatibility
/// rules.
pub use ltoir::{LtoirError, load_kernel_module};

// Re-export launch macros from cuda-macros for convenience.
pub use cuda_macros::{cuda_launch, cuda_module};

/// Re-export of [`cuda_macros::cuda_launch_async`].
///
/// Builds a lazy `cuda_async::simt::launch::AsyncKernelLaunch`. Raw launch
/// configuration is not tied to the kernel's indexing assumptions, so the
/// macro must be called inside `unsafe`. Stream assignment is deferred to the
/// scheduling policy -- call `.sync()` to block or `.await` to suspend.
#[cfg(feature = "async")]
pub use cuda_macros::cuda_launch_async;
pub use tiling::{
    TILE_SIZE, k_major_index, mn_major_index, print_layout_indices, to_k_major_f16, to_mn_major_f16,
};
