/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Entry point for GPU-dependent integration tests.

#[path = "common/mod.rs"]
mod common;

#[path = "gpu/tile_ir_13_4.rs"]
mod tile_ir_13_4;

#[path = "gpu/tensor.rs"]
mod tensor;

#[path = "gpu/num_tiles.rs"]
mod num_tiles;

#[path = "gpu/partial_coverage.rs"]
mod partial_coverage;

#[path = "gpu/launch_preconditions.rs"]
mod launch_preconditions;

// ---------------------------------------------------------------------------
// Migrated from cutile-examples (smoke tests for kernel patterns that were
// previously exposed as runnable examples but don't teach a pattern worth
// keeping in the examples drawer).
// ---------------------------------------------------------------------------

#[path = "gpu/add_basic.rs"]
mod add_basic;

#[path = "gpu/add_ptr.rs"]
mod add_ptr;

#[path = "gpu/const_pointers.rs"]
mod const_pointers;

#[path = "gpu/program_id.rs"]
mod program_id;

#[path = "gpu/launch_site.rs"]
mod launch_site;

#[path = "gpu/add_refs.rs"]
mod add_refs;

#[path = "gpu/global_counter.rs"]
mod global_counter;

#[path = "gpu/inter_module.rs"]
mod inter_module;

#[path = "gpu/tensor_slicing.rs"]
mod tensor_slicing;

#[path = "gpu/async_saxpy_unsafe.rs"]
mod async_saxpy_unsafe;

#[path = "gpu/async_device_op.rs"]
mod async_device_op;

#[path = "gpu/book_snippets.rs"]
mod book_snippets;

#[path = "gpu/tensor_permute.rs"]
mod tensor_permute;

#[path = "gpu/mapped_partition_values.rs"]
mod mapped_partition_values;

#[path = "gpu/mapped_partition_schedule_matrix.rs"]
mod mapped_partition_schedule_matrix;

#[path = "gpu/atomic_red_view.rs"]
mod atomic_red_view;

#[path = "gpu/borrow_raw_parts.rs"]
mod borrow_raw_parts;

#[path = "gpu/device_debug.rs"]
mod device_debug;

// ---------------------------------------------------------------------------
// Soundness guards: shape/storage invariants, launcher-side validation, and
// graph-capture input discipline.
// ---------------------------------------------------------------------------

#[path = "gpu/tensor_guards.rs"]
mod tensor_guards;

#[path = "gpu/launcher_guards.rs"]
mod launcher_guards;

#[path = "gpu/graph_scope_inputs.rs"]
mod graph_scope_inputs;

#[path = "gpu/tensor_token.rs"]
mod tensor_token;

// `gpu/submission_lifetimes.rs` has its own binary (`tests/submission_lifetimes.rs`):
// its stream Gates deadlock with any context-wide synchronize elsewhere in the
// same process (first-use module loads, eviction unloads), so it must not share
// a process with the rest of the suite. See that file's header.
// 2026-08 codegen audit regressions, one module per fix; `audit_common`
// holds the shared compile/transfer/subprocess helpers.
// ---------------------------------------------------------------------------

#[path = "gpu/audit_common.rs"]
mod audit_common;

#[path = "gpu/audit_signedness.rs"]
mod audit_signedness;

#[path = "gpu/audit_int_division.rs"]
mod audit_int_division;

#[path = "gpu/audit_short_circuit.rs"]
mod audit_short_circuit;

#[path = "gpu/audit_control_flow.rs"]
mod audit_control_flow;

#[path = "gpu/audit_check_hoisting.rs"]
mod audit_check_hoisting;

#[path = "gpu/audit_frontend_errors.rs"]
mod audit_frontend_errors;

#[path = "gpu/audit_module_consts.rs"]
mod audit_module_consts;
