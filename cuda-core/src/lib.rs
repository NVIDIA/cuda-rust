/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level CUDA driver API bindings and safe wrappers.

#![cfg_attr(feature = "f16", feature(f16))]

mod api;
pub(crate) mod cudarc_shim;
mod dtype;
mod error;
mod runtime;
pub mod simt;
pub mod vmm;

pub use api::*;
pub use cuda_bindings as sys;
pub use dtype::*;
pub use error::*;
pub use runtime::*;

// The cuda-oxide surface, re-exported at the root where its consumers
// expect it. `simt::LaunchConfig` and `simt::vmm` are deliberately absent:
// they collide with this crate's own `LaunchConfig` and with the reviewed
// `vmm` module above (both forks of the same cuda-oxide ancestor), and stay
// reachable only through `simt::`.
pub use simt::embedded;
pub use simt::{
    launch_kernel_cooperative, launch_kernel_cooperative_on_stream, launch_kernel_ex,
    launch_kernel_ex_cooperative, launch_kernel_ex_cooperative_on_stream,
    launch_kernel_ex_on_stream, launch_kernel_on_stream, BlockRequirement, ConstantHandle,
    ContextLimit, CoordinateRequirement, CudaContext, CudaEvent, CudaFunction, CudaModule,
    CudaStream, DeviceBuffer, DeviceCopy, DeviceLaunchLimits, DynamicSharedMemoryRequirement,
    EmbeddedModule, EmbeddedModuleError, KernelLaunchConfig, KernelLaunchContract, LaunchAxis,
    LaunchConfig1D, LaunchConfig2D, LaunchConfig3D, LaunchContractError, LaunchContractSpec,
    LaunchDimension, PinnedHostBuffer, PreparedLaunch, StreamPriorityRange, SyncPolicy,
};
