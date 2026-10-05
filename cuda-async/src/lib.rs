/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async runtime for CUDA device operations, providing futures-based kernel launching
//! and device memory management.

pub mod cuda_graph;
pub mod device_buffer;
pub mod device_context;
pub mod device_future;
pub mod device_operation;
pub mod error;
pub mod launch;
mod leak;
mod loom_compat;
pub mod predicate;
pub mod prelude;
// The real CUDA backend wraps pinned memory via `AtomicU32::from_ptr`, which is
// incompatible with loom's swapped atomics; under `--cfg loom` the protocol is
// model-checked through `slot_table`'s mock backend instead.
#[cfg(not(loom))]
mod reactor;
pub mod scheduling_policies;
/// SIMT-model async surface, copied from cuda-oxide for the shared
/// host-crate migration. Not re-exported at the root; see the module docs.
pub mod simt;
mod slot_table;
mod submission;

pub use futures;
