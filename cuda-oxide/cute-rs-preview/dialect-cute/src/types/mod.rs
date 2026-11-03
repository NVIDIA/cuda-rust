/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor-view types used to check and lower CuTe operations.
//!
//! A tensor view is a compiler description, not a runtime struct:
//!
//! ```text
//! pointer + length ── cute.tensor_make ──► !cute.tensor_view<...>
//!                                           │
//!                                           └─ gone before LLVM lowering
//! ```
//!
//! A selected backend consumes the description while lowering to native
//! operations or MLIR. A view has no runtime storage and must be removed
//! before the backend produces a kernel signature.
//!
//! After MIR preparation, the verifier follows each view through its uses:
//!
//! ```text
//! make -> zipped_divide -> slice -> load/store
//! ```
//!
//! At this stage, a view cannot be stored inside a pointer or aggregate,
//! passed between blocks, or appear in a function signature. Ordinary scalar
//! values may remain in compiler-owned local storage whose address does not
//! escape, or pass between blocks. The verifier follows those values too.

/// Shared output placement and asynchronous store synchronization contracts.
mod epilogue;
/// Packed-value and scale-view contracts for the block-scaled GEMV flow.
mod gemv;
/// Shared tensor placement, register carriers, and tiled matrix-instruction contracts.
mod mma;
/// Barrier-ring configuration and producer/consumer cursor contracts.
mod pipeline;
/// Persistent output-tile grids and their ordinary cursor provenance.
mod scheduler;
/// Tensor storage, access, format, and selection metadata.
mod tensor;
/// Descriptor-backed raw transport views and their direct copy contracts.
mod tma;

pub use epilogue::CuteEpilogueTileType;
pub use gemv::{CuteFragmentType, CuteScaledViewType};
pub use mma::{CuteSmemTensorType, MmaCarrierShape, mma_carrier_shape};
pub use pipeline::CuteTmaLoadPipelineType;
pub use scheduler::CuteWorkTileType;
pub use tensor::CuteTensorViewType;
pub(crate) use tensor::scalar_transport_layout;
pub use tma::CuteTmaViewType;

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
