/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level CUDA driver API bindings and safe wrappers.

#![cfg_attr(feature = "f16", feature(f16))]

pub mod allocation;
mod api;
pub use allocation::{AllocationLease, DeviceAllocation, RawDeviceBuffer};
pub mod completion;
pub mod context;
pub(crate) mod cudarc_shim;
pub mod device_buffer;
mod dtype;
pub mod embedded;
mod error;
pub mod event;
pub mod launch;
mod launch_raw;
pub mod memory;
pub mod module;
pub mod peer;
pub mod pinned_host_buffer;
mod runtime;
pub mod stream;
pub mod vmm;

pub use api::*;
// `DeviceBuffer::cast_elem` and `cast_chunks` are bounded on bytemuck's
// marker traits. The re-export lets a downstream crate
// implement them for its own element types against the same bytemuck.
pub use bytemuck;
pub use cuda_bindings as sys;
pub use cuda_repr::DeviceRepr;
pub use dtype::*;
pub use error::*;
pub use runtime::*;

pub use context::{ContextLimit, StreamPriorityRange, SyncPolicy};
pub use cuda_core_derive::DeviceCopy;
pub use device_buffer::{DeviceBuffer, DeviceBufferView, DeviceCopy};
pub use embedded::{EmbeddedModule, EmbeddedModuleError};
pub use launch::*;
pub use launch_raw::*;
pub use module::ConstantHandle;
pub use oxide_artifacts as artifacts;
pub use pinned_host_buffer::PinnedHostBuffer;

// Compatibility spellings are aliases of the shared handles, not separate runtimes.
pub use runtime::{
    Context as CudaContext, Event as CudaEvent, Function as CudaFunction, Module as CudaModule,
    Stream as CudaStream,
};
