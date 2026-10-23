/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-level CUDA primitives with statically checked operands.
//!
//! Physical formats and instruction selectors are Rust types. Election,
//! synchronization, instruction order and asynchronous completion are explicit
//! responsibilities of the kernel author, documented on unsafe operations.
//!
//! Register carriers can be constructed without issuing an instruction:
//!
//! ```
//! use ctm_rs::mma::AccumulatorTile16x8;
//! let tile = AccumulatorTile16x8::<2, 8>::zero(); // 2 × 8 fragments per lane.
//! assert_eq!(tile.fragment::<1, 7>().registers(), [0.0; 4]);
//! ```
//!
//! Setup, coverage and the safety contract are in
//! `cuda-oxide/cute-rs-preview/ctm-rs/README.md`.
//! Enable `host` for `host::tma` descriptor encoding.
//! Users of the reexported kernel macros must also depend directly on
//! `cuda-device`: their expansions refer to `::cuda_device`.

#![feature(f16)]
#![no_std]

#[cfg(feature = "host")]
extern crate std;

#[cfg(feature = "host")]
pub mod host;

/// Kernel macros using cuda-device's launch ABI; require a direct `cuda-device` dependency.
pub use cuda_device::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract};

pub mod async_copy;
pub mod barrier;
pub mod cluster;
pub mod convert;
pub mod debug;
pub mod f32x2;
pub mod fence;
pub mod float;
pub mod groups;
pub mod mbarrier;
pub mod memory;
pub mod mma;
pub mod tcgen05;
pub mod tma;
pub mod warp;

// Keep callable APIs as forwarding functions instead of public function
// reexports. rustc may print a reexport's path as an intrinsic's name; the
// device compiler recognizes raw operations by their cuda-device paths. The
// scoped-index and unroll compiler markers below are intentional exceptions:
// their canonical definitions must survive macro and importer recognition.

/// Hardware thread/block coordinates, without inferred thread mapping.
pub mod thread {
    /// Canonical index witness and compiler markers, preserving their defining crate.
    ///
    /// `index_1d` calls must appear inside a kernel/device body for the macro to
    /// attach its lifetime scope. `__unroll_config` is recognized at its loop
    /// call site. Wrapping either in an ordinary forwarding body is incorrect.
    pub use cuda_device::thread::{__unroll_config, ThreadIndex, index_1d};

    // Scope rewriting preserves the user's `thread::` prefix. Reexport the
    // canonical helpers so that generated calls retain the original witness.
    #[doc(hidden)]
    pub use cuda_device::thread::__internal;

    /// Generate named forwarding functions while preserving the intrinsic’s defining path.
    macro_rules! coordinate {
        ($($name:ident),* $(,)?) => {$(
            #[doc = concat!("Read the hardware coordinate through [`cuda_device::thread::", stringify!($name), "`].")]
            #[allow(non_snake_case)]
            #[inline(always)]
            pub fn $name() -> u32 {
                cuda_device::thread::$name()
            }
        )*};
    }
    coordinate!(
        blockDim_x,
        blockDim_y,
        blockDim_z,
        blockIdx_x,
        blockIdx_y,
        blockIdx_z,
        gridDim_x,
        gridDim_y,
        gridDim_z,
        threadIdx_x,
        threadIdx_y,
        threadIdx_z,
    );

    /// Restricts register redistribution counts to instruction immediates.
    mod sealed {
        /// Implemented only for multiples of eight in 24..=256.
        pub trait RegisterCount<const N: u32> {}
    }
    /// Sealed proof that `N` is a multiple of eight in `24..=256`.
    pub trait ValidRegisterCount<const N: u32>: sealed::RegisterCount<N> {}
    /// Admit the register-budget immediates supported by setmaxnreg.
    macro_rules! register_counts {
        ($($n:literal),* $(,)?) => {$(
            impl sealed::RegisterCount<$n> for () {}
            impl ValidRegisterCount<$n> for () {}
        )*};
    }
    register_counts!(
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
        184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
    );

    /// Increase a warpgroup's register limit to an immediate count.
    /// `N` must be a multiple of eight in `24..=256`.
    ///
    /// # Safety
    /// Every thread in the warpgroup must execute the same operation and count
    /// on supported hardware. The current register limit must be at most `N`.
    /// Synchronize the warpgroup before another register-limit adjustment.
    /// Coordinate releases by other warpgroups to avoid waiting indefinitely
    /// for unavailable registers.
    ///
    /// ```compile_fail,E0277
    /// unsafe { ctm_rs::thread::setmaxnreg_inc::<16>() };
    /// ```
    ///
    /// ```compile_fail,E0277
    /// unsafe { ctm_rs::thread::setmaxnreg_inc::<33>() };
    /// ```
    #[inline(always)]
    pub unsafe fn setmaxnreg_inc<const N: u32>()
    where
        (): ValidRegisterCount<N>,
    {
        unsafe { cuda_device::thread::setmaxnreg_inc::<N>() }
    }

    /// Decrease a warpgroup's register limit to an immediate count.
    /// `N` must be a multiple of eight in `24..=256`.
    ///
    /// # Safety
    /// Every thread in the warpgroup must execute the same operation and count
    /// on supported hardware. The current register limit must be at least `N`.
    /// Synchronize the warpgroup before another register-limit adjustment.
    /// Subsequent code must respect the selected limit.
    ///
    /// ```compile_fail,E0277
    /// unsafe { ctm_rs::thread::setmaxnreg_dec::<264>() };
    /// ```
    ///
    /// ```compile_fail,E0133
    /// ctm_rs::thread::setmaxnreg_dec::<32>();
    /// ```
    #[inline(always)]
    pub unsafe fn setmaxnreg_dec<const N: u32>()
    where
        (): ValidRegisterCount<N>,
    {
        unsafe { cuda_device::thread::setmaxnreg_dec::<N>() }
    }
}
