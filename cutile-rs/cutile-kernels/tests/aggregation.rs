/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage for the aggregation kernels, plus host reference
//! implementations the GPU tests compare against.

use cutile::compile_api::KernelCompiler;
use cutile_compiler::ast::Module as ModuleAst;
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, Module, OpId, Type};

const TEST_STACK_SIZE: usize = 32_000_000;
const TARGET: &str = "sm_120";

const MODE_OR: i64 = 1;
const MODE_ADD: i64 = 3;
const MODE_ADD_F: i64 = 4;
const MODE_MAX: i64 = 5;
const MODE_UMAX: i64 = 7;
const ORDERING_RELAXED: i64 = 1;
const SCOPE_DEVICE: i64 = 1;

struct Case {
    name: &'static str,
    module_ast: fn() -> ModuleAst,
    module_name: &'static str,
    function_name: &'static str,
    generics: Vec<String>,
    strides: Vec<(&'static str, &'static [i32])>,
    mode: i64,
}

const UNIT: &[i32] = &[1];
const ROW: &[i32] = &[-1, 1];

fn modes_for(elem: &str) -> (i64, i64) {
    // (AddF/Add, Max/UMax) by element.
    let add = if matches!(elem, "f16" | "bf16" | "f32" | "f64") {
        MODE_ADD_F
    } else {
        MODE_ADD
    };
    let max = if matches!(elem, "u32" | "u64") {
        MODE_UMAX
    } else {
        MODE_MAX
    };
    (add, max)
}

const ADD_ELEMS: &[&str] = &["f16", "bf16", "f32", "f64", "i32", "u32", "i64", "u64"];
const INT_ELEMS: &[&str] = &["i32", "u32", "i64", "u64"];

/// Every generic kernel, driven across every element of its trait, with the
/// mode each (kernel, element) pair must emit. One `Case` per instantiation.
fn cases() -> Vec<Case> {
    use cutile_kernels::aggregation::*;
    let mut v = Vec::new();
    for &e in ADD_ELEMS {
        let (add, _) = modes_for(e);
        v.push(Case {
            name: "scatter_add",
            module_ast: scatter_add_module::__module_ast_self,
            module_name: "scatter_add_module",
            function_name: "scatter_add",
            generics: vec![e.to_string()],
            strides: vec![("indices", UNIT), ("values", UNIT), ("out", UNIT)],
            mode: add,
        });
        v.push(Case {
            name: "segment_sum",
            module_ast: segment_sum_module::__module_ast_self,
            module_name: "segment_sum_module",
            function_name: "segment_sum",
            generics: vec![e.to_string()],
            strides: vec![("segment_ids", UNIT), ("values", UNIT), ("out", UNIT)],
            mode: add,
        });
        v.push(Case {
            name: "histogram",
            module_ast: histogram_module::__module_ast_self,
            module_name: "histogram_module",
            function_name: "histogram",
            generics: vec![e.to_string()],
            strides: vec![("bin_ids", UNIT), ("bins", UNIT)],
            mode: add,
        });
        v.push(Case {
            name: "weighted_histogram",
            module_ast: weighted_histogram_module::__module_ast_self,
            module_name: "weighted_histogram_module",
            function_name: "weighted_histogram",
            generics: vec![e.to_string()],
            strides: vec![("bin_ids", UNIT), ("weights", UNIT), ("bins", UNIT)],
            mode: add,
        });
        v.push(Case {
            name: "embedding_backward_scatter_add",
            module_ast: embedding_backward_scatter_add_module::__module_ast_self,
            module_name: "embedding_backward_scatter_add_module",
            function_name: "embedding_backward_scatter_add",
            generics: vec![e.to_string(), "8".to_string()],
            strides: vec![("token_ids", UNIT), ("grads", ROW), ("table_grad", ROW)],
            mode: add,
        });
    }
    for &e in INT_ELEMS {
        let (_, max) = modes_for(e);
        v.push(Case {
            name: "max_reduce",
            module_ast: max_reduce_module::__module_ast_self,
            module_name: "max_reduce_module",
            function_name: "max_reduce",
            generics: vec![e.to_string(), "64".to_string()],
            strides: vec![("values", ROW), ("out", UNIT)],
            mode: max,
        });
        v.push(Case {
            name: "bitmask_or",
            module_ast: bitmask_or_module::__module_ast_self,
            module_name: "bitmask_or_module",
            function_name: "bitmask_or",
            generics: vec![e.to_string()],
            strides: vec![("dest_ids", UNIT), ("masks", UNIT), ("out", UNIT)],
            mode: MODE_OR,
        });
    }
    v
}

fn ops(module: &Module) -> Vec<OpId> {
    fn walk(module: &Module, op: OpId, out: &mut Vec<OpId>) {
        out.push(op);
        for &region in &module.op(op).regions {
            for &block in &module.region(region).blocks {
                for &inner in &module.block(block).ops {
                    walk(module, inner, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for &function in &module.functions {
        walk(module, function, &mut out);
    }
    out
}

fn int_attr(module: &Module, op: OpId, name: &str) -> i64 {
    module
        .op(op)
        .attributes
        .iter()
        .find_map(|(n, attr)| match attr {
            Attribute::Integer(v, _) if n == name => Some(*v),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing integer attribute `{name}`"))
}

/// Every kernel lowers to exactly one atomic reduction with its intended
/// mode (signedness included), relaxed ordering, device scope, and a token
/// as the only result; indices loaded from memory keep a check, recorded in
/// the placement counts rather than silently dropped.
#[test]
fn every_kernel_reaches_its_aggregation_mode() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(|| {
            for case in cases() {
                let artifacts =
                    KernelCompiler::new(case.module_ast, case.module_name, case.function_name)
                        .target(TARGET)
                        .generics(
                            case.generics
                                .iter()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>(),
                        )
                        .strides(&case.strides)
                        .compile()
                        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
                let counts = artifacts.check_counts();
                let module = artifacts.into_module();
                let reductions: Vec<OpId> = ops(&module)
                    .into_iter()
                    .filter(|&id| module.op(id).opcode == Opcode::AtomicRedViewTko)
                    .collect();
                assert_eq!(reductions.len(), 1, "{}: one reduction", case.name);
                let op = reductions[0];
                assert_eq!(
                    int_attr(&module, op, "mode"),
                    case.mode,
                    "{}: mode",
                    case.name
                );
                assert_eq!(
                    int_attr(&module, op, "memory_ordering_semantics"),
                    ORDERING_RELAXED,
                    "{}: ordering",
                    case.name
                );
                assert_eq!(
                    int_attr(&module, op, "memory_scope"),
                    SCOPE_DEVICE,
                    "{}: scope",
                    case.name
                );
                assert_eq!(
                    module.op(op).result_types,
                    vec![Type::Token],
                    "{}: token result",
                    case.name
                );
                let placed = counts.discharged + counts.hoisted + counts.in_place;
                assert!(
                    placed > 0,
                    "{}: no access check was placed at all",
                    case.name
                );
            }
        })
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// The signed and unsigned maxima must not collapse to one comparison.
#[test]
fn signed_and_unsigned_max_select_different_modes() {
    assert_ne!(MODE_MAX, MODE_UMAX);
    // Enforced per kernel by `every_kernel_reaches_its_aggregation_mode`;
    // this test exists so a change collapsing the modes fails by name.
    let modes: Vec<i64> = cases()
        .iter()
        .filter(|c| c.name.starts_with("max_reduce"))
        .map(|c| c.mode)
        .collect();
    assert_eq!(modes, vec![MODE_MAX, MODE_UMAX, MODE_MAX, MODE_UMAX]);
}

// ── Host references, shared with the GPU tests ──────────────────────────────

pub fn scatter_add_ref(indices: &[u32], values: &[f32], out: &mut [f32]) {
    for (&i, &v) in indices.iter().zip(values) {
        out[i as usize] += v;
    }
}

pub fn histogram_ref(bin_ids: &[u32], bins: &mut [u32]) {
    for &b in bin_ids {
        bins[b as usize] += 1;
    }
}

pub fn weighted_histogram_ref(bin_ids: &[u32], weights: &[f32], bins: &mut [f32]) {
    for (&b, &w) in bin_ids.iter().zip(weights) {
        bins[b as usize] += w;
    }
}

pub fn max_reduce_i32_ref(values: &[i32], seed: i32) -> i32 {
    values.iter().copied().fold(seed, i32::max)
}

pub fn max_reduce_u32_ref(values: &[u32], seed: u32) -> u32 {
    values.iter().copied().fold(seed, u32::max)
}

pub fn bitmask_or_ref(dest_ids: &[u32], masks: &[u32], out: &mut [u32]) {
    for (&d, &m) in dest_ids.iter().zip(masks) {
        out[d as usize] |= m;
    }
}

/// Small deterministic inputs: repeated indices, an empty segment, duplicate
/// bins, zeros, negatives for signed max, and values straddling the signed
/// midpoint for unsigned max.
#[test]
fn host_references_handle_the_edge_cases() {
    // Repeated indices accumulate; untouched slots keep the identity seed.
    let mut out = vec![0.0f32; 4];
    scatter_add_ref(&[1, 1, 3, 1], &[1.0, 2.0, 5.0, 0.0], &mut out);
    assert_eq!(out, vec![0.0, 3.0, 0.0, 5.0]);

    // Segment 2 is empty and keeps its seed; segment sums share the scatter
    // reference, which is the same contraction.
    let mut segs = vec![0.0f32; 3];
    scatter_add_ref(&[0, 0, 1], &[2.0, -2.0, 7.0], &mut segs);
    assert_eq!(segs, vec![0.0, 7.0, 0.0]);

    let mut bins = vec![0u32; 3];
    histogram_ref(&[2, 2, 2, 0], &mut bins);
    assert_eq!(bins, vec![1, 0, 3]);

    let mut wbins = vec![0.0f32; 2];
    weighted_histogram_ref(&[1, 1, 0], &[0.5, 0.25, 0.0], &mut wbins);
    assert_eq!(wbins, vec![0.0, 0.75]);

    // Negative values with the signed identity.
    assert_eq!(max_reduce_i32_ref(&[-5, -2, -9], i32::MIN), -2);
    assert_eq!(max_reduce_i32_ref(&[], i32::MIN), i32::MIN);
    // Above the signed midpoint: the unsigned comparison must pick the
    // high-bit value a signed one would call negative.
    assert_eq!(
        max_reduce_u32_ref(&[0x7FFF_FFF0, 0x8000_0010], 0),
        0x8000_0010
    );

    let mut words = vec![0u32; 2];
    bitmask_or_ref(&[0, 0, 1], &[0b01, 0b10, 0b100], &mut words);
    assert_eq!(words, vec![0b11, 0b100]);
}

#[test]
fn reduce_max_uses_the_elements_signedness() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(|| {
            use cutile_kernels::aggregation::max_reduce_module;
            let func = "max_reduce";
            for (elem, want_signed) in [
                ("i32", "signed"),
                ("u32", "unsigned"),
                ("i64", "signed"),
                ("u64", "unsigned"),
            ] {
                let module = KernelCompiler::new(
                    max_reduce_module::__module_ast_self,
                    "max_reduce_module",
                    func,
                )
                .target(TARGET)
                .generics(vec![elem.to_string(), "64".to_string()])
                .strides(&[("values", ROW), ("out", UNIT)])
                .compile()
                .unwrap_or_else(|e| panic!("{func}<{elem}>: {e}"))
                .into_module();
                // Find the MaxI op (the reduce combinator) and read its
                // signedness attribute.
                let mut seen = None;
                for id in ops(&module) {
                    let op = module.op(id);
                    if format!("{:?}", op.opcode).contains("MaxI") {
                        for (n, a) in &op.attributes {
                            if n == "signedness" {
                                if let Attribute::Integer(v, _) = a {
                                    seen = Some(
                                        if *v == 0 { "unsigned" } else { "signed" }.to_string(),
                                    );
                                }
                            }
                        }
                    }
                }
                let got = seen.unwrap_or_else(|| panic!("{func}<{elem}>: no MaxI signedness attr"));
                eprintln!("{func}<{elem}>: MaxI signedness = {got} (want {want_signed})");
                assert!(
                    got.to_lowercase().contains(want_signed),
                    "{func}<{elem}>: MaxI signedness {got}, expected {want_signed}"
                );
            }
        })
        .expect("spawn")
        .join()
        .expect("probe");
}
