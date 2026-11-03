/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Token/ordering coverage is independent of the bounds annotations, but the
// kernels are written in the older spelling.
#![allow(deprecated)]

//! Happens-before (token) ordering for mutable partition stores.
//!
//! The Tile IR memory model
//! (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html>)
//! orders memory ops *only* through tokens: two ops that consume the same input
//! token are unordered (a data race — UB — if they conflict); ordering requires
//! threading one op's output token into the next op's input.
//!
//! The rule is brand-directed: two writes are **ordered** when they hit the same
//! region (same brand / aliasing) and **forked** (share an input token) when
//! their brands are distinct (disjoint). Resource token threading implements it:
//! a view roots at its tensor, and the tensor's token propagates up the borrow
//! link and across every scope boundary — the method-call boundary (a view's
//! token advances its tensor's) and the loop boundary (the tensor's token is
//! carried, each store's output joined into it, the result published at exit).
//!
//! - **straight-line** same-region writes chain (ordered); a later view of the
//!   same tensor is ordered after the prior view's writes;
//! - **loop** distinct-index writes fork off the loop-invariant view token and
//!   join into an accumulator published to the tensor; a constant/repeated index
//!   serializes (the views read the carried accumulator);
//! - **cross-epoch** — a second `partition_mut` seeds from the tensor's published
//!   token, so it is ordered after the first view's writes.
//!
//! One spec remains `#[ignore]`d: `straightline_disjoint_should_fork` is a
//! deferred perf optimization (forking *straight-line* disjoint stores), not a
//! soundness gap — serializing them is correct, just slower.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod token_ordering_module {
    use cutile::core::*;

    /// Two straight-line stores to the **same** column (region). Aliasing →
    /// must be ordered.
    #[cutile::entry()]
    fn straightline_same_index<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut v = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        v.store(tile, coord((0i32, 0i32)));
        v.store(tile, coord((0i32, 0i32)));
    }

    /// Two straight-line stores to **distinct** columns. Disjoint → should fork.
    #[cutile::entry()]
    fn straightline_disjoint<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut v = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        v.store(tile, coord((0i32, 0i32)));
        v.store(tile, coord((0i32, 1i32)));
    }

    /// A single view iterating a branded column index: distinct region per
    /// iteration → the stores should fork off the loop-entry token.
    #[cutile::entry()]
    fn single_view_loop<const N: i32, const BLOCK_SIZE: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let tile_shape = shape![1, BLOCK_SIZE];
        let mut v = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            v.store(tile, coord((0i32, j)));
        }
    }

    /// Two sequential `partition_mut` epochs of the SAME tensor, each writing
    /// the same column. Aliasing across the epochs → view B must be ordered
    /// after view A.
    #[cutile::entry()]
    fn two_epoch_aliasing<const N: i32, const BLOCK_SIZE: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let tile_shape = shape![1, BLOCK_SIZE];
        {
            let mut a = out
                .partition_mut(tile_shape)
                .with_bounds((Dim::new(1), cols));
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0, tile_shape);
            a.store(tile, coord((0i32, 0i32)));
        }
        {
            let mut b = out
                .partition_mut(tile_shape)
                .with_bounds((Dim::new(1), cols));
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(2.0, tile_shape);
            b.store(tile, coord((0i32, 0i32)));
        }
    }

    /// Two sequential epochs, each a store *loop* over the whole column range.
    /// Every column is written by both epochs → view B's loop must be ordered
    /// after view A's.
    #[cutile::entry()]
    fn two_epoch_aliasing_in_loop<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let tile_shape = shape![1, BLOCK_SIZE];
        {
            let mut a = out
                .partition_mut(tile_shape)
                .with_bounds((Dim::new(1), cols));
            for j in cols {
                let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0, tile_shape);
                a.store(tile, coord((0i32, j)));
            }
        }
        {
            let mut b = out
                .partition_mut(tile_shape)
                .with_bounds((Dim::new(1), cols));
            for j in cols {
                let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(2.0, tile_shape);
                b.store(tile, coord((0i32, j)));
            }
        }
    }

    // ── Inductive composition (depth-2 nesting) ────────────────────────────

    /// Nested loops over branded `(row, col)`: every `(i, j)` is a distinct
    /// (disjoint) region, so every leaf store should fork off the *outermost*
    /// loop-invariant token — the fork rule composing across two loop levels.
    #[cutile::entry()]
    fn nested_loop<const M: i32, const N: i32, const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
    ) {
        let rows = Dim::new(M / BM);
        let cols = Dim::new(N / BN);
        let ts = shape![BM, BN];
        let mut v = out.partition_mut(ts).with_bounds((rows, cols));
        for i in rows {
            for j in cols {
                let tile: Tile<f32, { [BM, BN] }> = constant(0.0, ts);
                v.store(tile, coord((i, j)));
            }
        }
    }

    /// Two epochs, each a *nested* loop over the same region → epoch B must be
    /// ordered after epoch A: the epoch-boundary join composing with nesting
    /// (the inner loop's join feeds the outer, which feeds the epoch boundary).
    #[cutile::entry()]
    fn two_epoch_nested_loop<const M: i32, const N: i32, const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
    ) {
        let rows = Dim::new(M / BM);
        let cols = Dim::new(N / BN);
        let ts = shape![BM, BN];
        {
            let mut a = out.partition_mut(ts).with_bounds((rows, cols));
            for i in rows {
                for j in cols {
                    let tile: Tile<f32, { [BM, BN] }> = constant(1.0, ts);
                    a.store(tile, coord((i, j)));
                }
            }
        }
        {
            let mut b = out.partition_mut(ts).with_bounds((rows, cols));
            for i in rows {
                for j in cols {
                    let tile: Tile<f32, { [BM, BN] }> = constant(2.0, ts);
                    b.store(tile, coord((i, j)));
                }
            }
        }
    }

    /// Within one epoch: a store loop, then a straight-line store to a column
    /// the loop already wrote → the trailing store must be ordered after the
    /// loop (ordering composing across the loop boundary inside an epoch).
    #[cutile::entry()]
    fn store_after_loop<const N: i32, const BLOCK_SIZE: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let ts = shape![1, BLOCK_SIZE];
        let mut v = out.partition_mut(ts).with_bounds((Dim::new(1), cols));
        for j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, ts);
            v.store(tile, coord((0i32, j)));
        }
        // Trailing straight-line store aliasing the `j == 0` write.
        let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0, ts);
        v.store(tile, coord((0i32, 0i32)));
    }

    /// Two stores to the SAME branded index within one loop iteration → same
    /// region → must be ordered (the second depends on the first) inside the
    /// body. The base of the induction for same-region ordering under a loop.
    #[cutile::entry()]
    fn within_iteration_aliasing<const N: i32, const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let ts = shape![1, BLOCK_SIZE];
        let mut v = out.partition_mut(ts).with_bounds((Dim::new(1), cols));
        for j in cols {
            let t0: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, ts);
            v.store(t0, coord((0i32, j)));
            let t1: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0, ts);
            v.store(t1, coord((0i32, j))); // same `j` → same address
        }
    }

    /// A loop whose index does NOT vary the written address (constant column 0
    /// every iteration) → every iteration writes the SAME region. The fork
    /// guard (L1): these must serialize, not fork off the loop-entry token.
    #[cutile::entry()]
    fn const_index_loop<const N: i32, const BLOCK_SIZE: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / BLOCK_SIZE);
        let ts = shape![1, BLOCK_SIZE];
        let mut v = out.partition_mut(ts).with_bounds((Dim::new(1), cols));
        for _j in cols {
            let tile: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, ts);
            v.store(tile, coord((0i32, 0i32))); // same address every iteration
        }
    }
}

use token_ordering_module::__module_ast_self;

fn compile_g(function_name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> String {
    let function_name = function_name.to_string();
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    let strides: Vec<(String, Vec<i32>)> = strides
        .iter()
        .map(|(n, s)| (n.to_string(), s.to_vec()))
        .collect();
    common::with_test_stack(move || {
        let strides: Vec<(&str, &[i32])> = strides
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        common::compile_to_ir(
            __module_ast_self,
            "token_ordering_module",
            &function_name,
            &generics,
            &strides,
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .unwrap_or_else(|e| panic!("failed to compile {function_name}: {e}"))
    })
}

fn compile(function_name: &str) -> String {
    compile_g(function_name, &["256", "64"], &[("out", &[256, 1])])
}

use std::collections::{HashMap, HashSet};

/// The input token operand (`token = %NN`) of each `store_view_tko`. Two stores
/// sharing an input token are *unordered* (forked) — the strong assertion for
/// disjoint writes (per the memory model, same input token = concurrent).
fn store_input_tokens(mlir: &str) -> Vec<String> {
    mlir.lines()
        .filter(|l| l.contains("store_view_tko"))
        .filter_map(store_input_token)
        .collect()
}

fn store_input_token(line: &str) -> Option<String> {
    line.split("token = ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_string)
}

/// Per `store_view_tko`, `(output token, input token)` — the LHS result and the
/// `token = %NN` operand, in program order.
fn store_io_tokens(mlir: &str) -> Vec<(String, String)> {
    mlir.lines()
        .filter(|l| l.contains("store_view_tko"))
        .filter_map(|l| {
            let output = l.split_once('=')?.0.trim().to_string();
            let input = store_input_token(l)?;
            output.starts_with('%').then_some((output, input))
        })
        .collect()
}

/// Every `%<ident>` SSA reference in a text fragment.
fn ssa_refs(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut refs = vec![];
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            refs.push(s[start..i].to_string());
        } else {
            i += 1;
        }
    }
    refs
}

/// Map each SSA result `%id` to the SSA operands its defining op reads
/// (`%a = op ... %b, %c` → `%a ↦ [%b, %c]`). Region-aware: a `for` op's result
/// also depends on the operands its body `continue`/`yield`s, so the def-use
/// walk can trace a loop result back into its region (a loop that accumulates a
/// token yields it via `continue`, and the result carries that dependency).
fn ssa_def_operands(mlir: &str) -> HashMap<String, Vec<String>> {
    let mut defs: HashMap<String, Vec<String>> = HashMap::new();
    // Stack of (for-op result, brace depth when it opened) for open regions.
    let mut regions: Vec<(String, i32)> = Vec::new();
    let mut depth = 0i32;
    for line in mlir.lines() {
        if let Some((lhs, rhs)) = line.split_once('=') {
            let results: Vec<String> = lhs
                .split(',')
                .map(str::trim)
                .filter(|s| s.starts_with('%'))
                .map(str::to_string)
                .collect();
            if !results.is_empty() {
                let operands = ssa_refs(rhs);
                for r in &results {
                    defs.entry(r.clone()).or_default().extend(operands.clone());
                }
                if rhs.trim_start().starts_with("for ") && line.contains('{') {
                    regions.push((results[0].clone(), depth));
                }
            }
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("continue ") || trimmed.starts_with("yield ") {
            if let Some((for_result, _)) = regions.last() {
                let ops = ssa_refs(trimmed);
                defs.entry(for_result.clone()).or_default().extend(ops);
            }
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        while regions.last().is_some_and(|(_, open)| depth <= *open) {
            regions.pop();
        }
    }
    defs
}

/// Does `source` transitively depend on `target` in the def-use graph? This is
/// the *actual* happens-before edge — token inequality alone is too weak (a
/// fresh, unrelated token also differs yet is still a race).
fn depends_on(defs: &HashMap<String, Vec<String>>, source: &str, target: &str) -> bool {
    let mut stack = vec![source.to_string()];
    let mut seen = HashSet::new();
    while let Some(v) = stack.pop() {
        if v == target {
            return true;
        }
        if !seen.insert(v.clone()) {
            continue;
        }
        if let Some(ops) = defs.get(&v) {
            stack.extend(ops.iter().cloned());
        }
    }
    false
}

/// Assert store `later` happens-after store `earlier`: `later`'s input token
/// transitively derives from `earlier`'s output token.
fn assert_ordered(mlir: &str, earlier: usize, later: usize) {
    let stores = store_io_tokens(mlir);
    assert!(
        stores.len() > earlier.max(later),
        "expected at least {} stores, found {}:\n{mlir}",
        earlier.max(later) + 1,
        stores.len()
    );
    let defs = ssa_def_operands(mlir);
    let earlier_out = &stores[earlier].0;
    let later_in = &stores[later].1;
    assert!(
        depends_on(&defs, later_in, earlier_out),
        "store {later} (input {later_in}) must depend on store {earlier} (output {earlier_out}); \
         the happens-before edge is missing:\n{mlir}"
    );
}

/// Assert the loop's store forks off a loop-invariant token (a `make_token`
/// defined outside the loop) — the correct, unordered fork for disjoint writes.
fn assert_forks_off_invariant(mlir: &str) {
    let toks = store_input_tokens(mlir);
    assert_eq!(
        toks.len(),
        1,
        "expected one store in the loop body:\n{mlir}"
    );
    assert!(
        mlir.contains(&format!("{} = make_token", toks[0])),
        "loop store should fork off a loop-invariant token, got {}:\n{mlir}",
        toks[0]
    );
}

// ── Already correct: assertions that pass on today's lowering ──────────────

#[test]
fn straightline_same_index_is_ordered() {
    // Store 2's input token derives from store 1's output (chained).
    assert_ordered(&compile("straightline_same_index"), 0, 1);
}

#[test]
fn within_iteration_aliasing_is_ordered() {
    // Two stores to the same index in one iteration must be ordered inside the
    // body — the base case for same-region ordering under a loop.
    assert_ordered(&compile("within_iteration_aliasing"), 0, 1);
}

#[test]
fn loop_stores_fork_off_the_entry_token() {
    assert_forks_off_invariant(&compile("single_view_loop"));
}

// ── Inductive composition: the base rules must compose at depth-2 nesting ───

const NESTED_G: [&str; 4] = ["256", "256", "64", "64"];

#[test]
fn nested_loop_stores_fork_at_all_levels() {
    // Fork composes across two loop levels: distinct (i, j) are disjoint, so the
    // leaf store reads a token defined outside *both* loops.
    assert_forks_off_invariant(&compile_g("nested_loop", &NESTED_G, &[("out", &[256, 1])]));
}

// ── Executable specs of the pending token-dataflow fix (currently fail) ─────
//
// "ordered" now asserts a real def-use dependency (assert_ordered), not token
// inequality — a fix that merely gives a fresh unrelated token would still race
// and must NOT satisfy these.

#[ignore = "target: disjoint straight-line stores should fork; today update_token over-serializes them"]
#[test]
fn straightline_disjoint_should_fork() {
    let toks = store_input_tokens(&compile("straightline_disjoint"));
    assert_eq!(toks.len(), 2, "expected two stores");
    assert_eq!(
        toks[0], toks[1],
        "disjoint straight-line stores should share an input token (forked)"
    );
}

#[test]
fn cross_epoch_aliasing_should_serialize() {
    assert_ordered(&compile("two_epoch_aliasing"), 0, 1);
}

#[test]
fn cross_epoch_aliasing_in_loop_should_serialize() {
    assert_ordered(&compile("two_epoch_aliasing_in_loop"), 0, 1);
}

#[test]
fn cross_epoch_nested_loops_should_serialize() {
    assert_ordered(
        &compile_g("two_epoch_nested_loop", &NESTED_G, &[("out", &[256, 1])]),
        0,
        1,
    );
}

#[test]
fn store_after_loop_in_epoch_should_serialize() {
    // The trailing store (index 1) must depend on the loop store (index 0).
    assert_ordered(&compile("store_after_loop"), 0, 1);
}

#[test]
fn const_index_loop_should_serialize() {
    // The store must read a loop-carried token (an iter-arg), NOT a loop-
    // invariant `make_token` — otherwise every iteration writes the same address
    // unordered.
    let mlir = compile("const_index_loop");
    let toks = store_input_tokens(&mlir);
    assert_eq!(toks.len(), 1, "expected one store in the loop body");
    assert!(
        !mlir.contains(&format!("{} = make_token", toks[0])),
        "const-index loop store must be serialized (carried token), not forked off \
         a loop-invariant make_token, got {}:\n{mlir}",
        toks[0]
    );
}
