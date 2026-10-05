/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Convenience re-exports for common `cuda-async` types and traits.
//!
//! ```rust,ignore
//! use cuda_async::prelude::*;
//! ```

// Core traits + execution context
pub use crate::device_operation::DeviceOp;
pub use crate::device_operation::ExecutionContext;
pub use crate::device_operation::GraphNode;

// Constructors and helpers
pub use crate::device_operation::value;
pub use crate::device_operation::with_context;
pub use crate::device_operation::DeviceOpUnwrapArc;
pub use crate::device_operation::IntoDeviceOp;

// Composition traits
pub use crate::device_operation::Unzippable1;
pub use crate::device_operation::Unzippable2;
pub use crate::device_operation::Unzippable3;
pub use crate::device_operation::Unzippable4;
pub use crate::device_operation::Unzippable5;
pub use crate::device_operation::Unzippable6;
pub use crate::device_operation::Zippable;

// Macros
pub use crate::unzip;
pub use crate::zip;

// Concrete types users commonly name
pub use crate::device_operation::BoxedDeviceOp;
pub use crate::device_operation::DeviceOpVec;
pub use crate::device_operation::SharedDeviceOp;
pub use crate::device_operation::Value;

// CUDA graph capture
pub use crate::cuda_graph::{CudaGraph, GraphLaunch, Scope};

// Error type
pub use crate::error::DeviceError;
