/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Convenience re-exports for common `cutile` types and traits.
//!
//! ```rust,ignore
//! use cutile::prelude::*;
//! ```

// Re-export the cuda-async prelude
pub use cuda_async::prelude::*;

// API functions and extension traits
pub use crate::api::{self, DeviceOpReshape, DeviceOpReshapeShared};

// Error type
pub use crate::error::Error;

// Tensor types and traits
pub use crate::tensor::{
    IntoPartition, KernelInput, KernelInputStored, KernelOutput, KernelOutputStored,
    MappedLaunchPartition, Partition, PartitionMut, Reshape, SpecializationBits, Tensor,
    TensorView, ToHostVec, TryPartition, Unpartition, OWNED,
};

// Tile kernel traits
pub use crate::tile_kernel::{PartitionOp, TileKernel, ToHostVecOp};

// Common types from cuda-core
pub use cuda_core::{DType, Device};

// Common dependencies
pub use std::sync::Arc;
