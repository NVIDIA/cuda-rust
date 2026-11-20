/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed-point `SsaEnv` operand propagation for phased shared-memory, 2D tiled,
//! `mbarrier`/TMA, and Blackwell `tcgen05` kernels.

use std::collections::HashMap;

use pliron::{
    builtin::types::IntegerType, context::Context, linked_list::ContainsLinkedList,
    operation::Operation, r#type::Typed, value::Value,
};

use super::ir_utils::{
    BlockGraph, as_idx_expr, attribute_field_index, bin_idx_exprs, bin_kinds, eval_int_cast,
    int_attr_u64, is_mbarrier_shared_alloc, is_uniform_or_const, op_kind, op_name,
    shared_alloc_elem_bytes, uniform_if_both,
};
use crate::{
    spec::{KernelVerificationSpec, get_string_attr},
    ssa::{GuardExpr, IdxExpr, SsaEnv, SsaKind},
};

/// Simplify an addition of two [`IdxExpr`]s, folding `0` identities and constant sums.
/// Keeps `TmemAddr` lane and column offset trees compact across bit-shifts and adds.
fn simplify_add_idx(a: IdxExpr, b: IdxExpr) -> IdxExpr {
    match (a, b) {
        (IdxExpr::Const(0), other) | (other, IdxExpr::Const(0)) => other,
        (IdxExpr::Const(x), IdxExpr::Const(y)) => IdxExpr::Const(x.saturating_add(y)),
        (lhs, rhs) => IdxExpr::Add(Box::new(lhs), Box::new(rhs)),
    }
}

/// Extract `(shared_id, size, elem_bytes, byte_offset)` from a shared-memory offset or descriptor.
/// Preserves interior byte offsets (`smem_off + const`) through `Tcgen05SmemDescriptor` packing.
pub(super) fn as_smem_desc_source(kind: &SsaKind) -> Option<(usize, u64, u64, u64)> {
    match *kind {
        SsaKind::SmemSharedOffset {
            shared_id,
            size,
            elem_bytes,
            byte_offset,
        }
        | SsaKind::SmemDesc {
            shared_id,
            size,
            elem_bytes,
            byte_offset,
        } => Some((shared_id, size, elem_bytes, byte_offset)),
        _ => None,
    }
}

/// Combine a `TmemAddr` with a lane/column integer offset (`(lane << 16)` or `col < 65536`).
/// Rejects column expressions whose upper bound could overflow bit `16` into the lane field.
fn combine_tmem_addr(
    base: &SsaKind,
    delta: &SsaKind,
    block_size: u32,
    block_y: u32,
) -> Option<SsaKind> {
    let SsaKind::TmemAddr {
        alloc_cols,
        lane_expr,
        col_expr,
    } = base
    else {
        return None;
    };
    let expr = as_idx_expr(delta)?;
    if let IdxExpr::Const(raw) = &expr {
        return Some(SsaKind::TmemAddr {
            alloc_cols: *alloc_cols,
            lane_expr: simplify_add_idx(lane_expr.clone(), IdxExpr::Const(raw >> 16)),
            col_expr: simplify_add_idx(col_expr.clone(), IdxExpr::Const(raw & 0xFFFF)),
        });
    }
    if let IdxExpr::Mul(inner, factor) = &expr
        && let IdxExpr::Const(f) = factor.as_ref()
        && *f >= 65536
        && *f % 65536 == 0
    {
        let scale = *f / 65536;
        let scaled_lane = if scale == 1 {
            *inner.clone()
        } else {
            IdxExpr::Mul(inner.clone(), Box::new(IdxExpr::Const(scale)))
        };
        return Some(SsaKind::TmemAddr {
            alloc_cols: *alloc_cols,
            lane_expr: simplify_add_idx(lane_expr.clone(), scaled_lane),
            col_expr: col_expr.clone(),
        });
    }
    if expr
        .exclusive_upper_bound(block_size, block_y)
        .is_some_and(|ub| ub <= 65536)
    {
        let next_col = simplify_add_idx(col_expr.clone(), expr);
        if next_col.single_loop_scope().is_valid() {
            return Some(SsaKind::TmemAddr {
                alloc_cols: *alloc_cols,
                lane_expr: lane_expr.clone(),
                col_expr: next_col,
            });
        }
    }
    None
}

/// Evaluate a pure SSA-producing operation during phased `SsaEnv` fixed-point propagation.
/// Centralizes coordinate algebra, `SmemDesc` byte offsets, `TmemAddr`, and `UniformScalar`.
#[allow(clippy::too_many_arguments)]
fn eval_phased_ssa_op(
    ctx: &Context,
    opid: &str,
    op: &Operation,
    b_idx: usize,
    spec: &KernelVerificationSpec,
    graph: &BlockGraph,
    env: &SsaEnv,
    bar_raw_list: &[String],
    shared_raw_list: &[(String, u64)],
    shared_elem_bytes_map: &HashMap<String, u64>,
    tmem_addr_shared_slots: &HashMap<usize, u64>,
    stack_struct_fields: &HashMap<Value, Vec<SsaKind>>,
) -> Option<SsaKind> {
    let (lhs, rhs) = bin_kinds(env, op);
    match opid {
        "nvvm.read_ptx_sreg_tid_x" => Some(SsaKind::Index(IdxExpr::Tx)),
        "nvvm.read_ptx_sreg_tid_y" => Some(SsaKind::Index(IdxExpr::Ty)),
        "nvvm.read_ptx_sreg_ctaid_x" => Some(SsaKind::SregCtaidX { bit_width: 32 }),
        "nvvm.read_ptx_sreg_ntid_x" => Some(SsaKind::SregNtidX { bit_width: 32 }),
        "nvvm.read_ptx_sreg_laneid" => {
            Some(SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)))
        }
        "mir.call" if op.get_num_results() == 1 => {
            let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
            if callee.contains("index_1d") {
                Some(SsaKind::ThreadIndexStruct)
            } else if callee.contains("lane_id") {
                Some(SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)))
            } else if (callee.contains("Tcgen05SmemDescriptorBuilder")
                || callee.contains("Tcgen05SmemDescriptor"))
                && let Some((shared_id, size, elem_bytes, byte_offset)) =
                    lhs.and_then(|k| match k {
                        SsaKind::StructFields(fs) => fs.iter().find_map(as_smem_desc_source),
                        other => as_smem_desc_source(&other),
                    })
            {
                let desc = SsaKind::SmemDesc {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                };
                if op
                    .get_result(0)
                    .get_type(ctx)
                    .deref(ctx)
                    .downcast_ref::<IntegerType>()
                    .is_some()
                {
                    Some(desc)
                } else {
                    Some(SsaKind::StructFields(vec![desc]))
                }
            } else {
                Some(SsaKind::DataValue)
            }
        }
        "mir.alloca" if op.get_num_results() == 1 => Some(SsaKind::StackAlloca(op.get_result(0))),
        "mir.field_addr" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
            let field = attribute_field_index(&op.attributes, "field_index")
                .or_else(|| attribute_field_index(&op.attributes, "index"));
            match (lhs, field) {
                (Some(SsaKind::StackAlloca(alloca)), Some(idx)) => Some(SsaKind::StackFieldPtr {
                    alloca,
                    field: idx as usize,
                }),
                _ => Some(SsaKind::DataValue),
            }
        }
        "mir.div" | "mir.rem" => {
            if let (Some(a), Some(SsaKind::ConstInt { value: g, .. })) =
                (lhs.as_ref().and_then(as_idx_expr), &rhs)
                && *g > 0
                && a.has_coord()
            {
                let d = u64::try_from(*g).unwrap_or(u64::MAX);
                let expr = if opid == "mir.div" {
                    IdxExpr::Div(Box::new(a), d)
                } else {
                    IdxExpr::Mod(Box::new(a), d)
                };
                Some(SsaKind::Index(expr))
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.shr" | "mir.shl" => {
            if let Some((shared_id, size, elem_bytes, byte_offset)) =
                lhs.as_ref().and_then(as_smem_desc_source)
                && matches!(rhs, Some(SsaKind::ConstInt { .. }))
            {
                Some(SsaKind::SmemDesc {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                })
            } else if opid == "mir.shl"
                && let (
                    Some(SsaKind::ConstInt { width, value: vl }),
                    Some(SsaKind::ConstInt { value: sh, .. }),
                ) = (&lhs, &rhs)
                && *sh < 64
            {
                Some(SsaKind::ConstInt {
                    width: *width,
                    value: vl << *sh,
                })
            } else if let (Some(a), Some(SsaKind::ConstInt { value: sh, .. })) =
                (lhs.as_ref().and_then(as_idx_expr), &rhs)
                && *sh <= 32
                && a.has_coord()
            {
                let scale = 1u64 << *sh;
                let expr = if opid == "mir.shr" {
                    IdxExpr::Div(Box::new(a), scale)
                } else {
                    IdxExpr::Mul(Box::new(a), Box::new(IdxExpr::Const(scale)))
                };
                Some(SsaKind::Index(expr))
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.bitand" => match (&lhs, &rhs) {
            (Some(l), Some(SsaKind::ConstInt { .. }))
            | (Some(SsaKind::ConstInt { .. }), Some(l))
                if as_smem_desc_source(l).is_some() =>
            {
                let (shared_id, size, elem_bytes, byte_offset) = as_smem_desc_source(l).unwrap();
                Some(SsaKind::SmemDesc {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                })
            }
            (Some(l), Some(SsaKind::ConstInt { value: m, .. }))
            | (Some(SsaKind::ConstInt { value: m, .. }), Some(l))
                if *m > 0 && (*m + 1).is_power_of_two() =>
            {
                if let Some(a) = as_idx_expr(l)
                    && a.has_coord()
                {
                    let modulus = u64::try_from(*m + 1).unwrap_or(u64::MAX);
                    Some(SsaKind::Index(IdxExpr::Mod(Box::new(a), modulus)))
                } else {
                    Some(uniform_if_both(&lhs, &rhs))
                }
            }
            _ => Some(uniform_if_both(&lhs, &rhs)),
        },
        "mir.bitor" => {
            if let (Some(l), Some(r)) = (&lhs, &rhs)
                && let Some((shared_id, size, elem_bytes, byte_offset)) = as_smem_desc_source(l)
                    .filter(|_| matches!(r, SsaKind::ConstInt { .. }))
                    .or_else(|| {
                        as_smem_desc_source(r).filter(|_| matches!(l, SsaKind::ConstInt { .. }))
                    })
            {
                Some(SsaKind::SmemDesc {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                })
            } else if let (
                Some(SsaKind::ConstInt { width, value: vl }),
                Some(SsaKind::ConstInt { value: vr, .. }),
            ) = (&lhs, &rhs)
            {
                Some(SsaKind::ConstInt {
                    width: *width,
                    value: vl | vr,
                })
            } else if let Some((a, b)) = bin_idx_exprs(&lhs, &rhs)
                && (a.has_coord() || b.has_coord())
                && let Some(sum) = IdxExpr::disjoint_bitor(a, b, spec.block_size, spec.block_y)
                && sum.single_loop_scope().is_valid()
            {
                Some(SsaKind::Index(sum))
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.mul" => {
            if let (
                Some(SsaKind::SregCtaidX { bit_width: w1 }),
                Some(SsaKind::SregNtidX { bit_width: w2 }),
            )
            | (
                Some(SsaKind::SregNtidX { bit_width: w2 }),
                Some(SsaKind::SregCtaidX { bit_width: w1 }),
            ) = (&lhs, &rhs)
                && ((w1 >= &64 && w2 >= &64) || spec.u32_coordinates)
            {
                Some(SsaKind::BlockBase1D { bit_width: *w1 })
            } else if let (
                Some(SsaKind::SregCtaidX { bit_width: w1 }),
                Some(SsaKind::ConstInt {
                    width: w2,
                    value: c,
                }),
            )
            | (
                Some(SsaKind::ConstInt {
                    width: w2,
                    value: c,
                }),
                Some(SsaKind::SregCtaidX { bit_width: w1 }),
            ) = (&lhs, &rhs)
                && *c == u128::from(spec.block_size)
                && ((w1 >= &64 && w2 >= &64) || spec.u32_coordinates)
            {
                Some(SsaKind::BlockBase1D { bit_width: *w1 })
            } else if let Some((a, b)) = bin_idx_exprs(&lhs, &rhs)
                && (a.has_coord() || b.has_coord())
            {
                let prod = IdxExpr::Mul(Box::new(a), Box::new(b));
                Some(if prod.single_loop_scope().is_valid() {
                    SsaKind::Index(prod)
                } else {
                    SsaKind::DataValue
                })
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.sub" => (op.get_num_results() == 1).then(|| uniform_if_both(&lhs, &rhs)),
        "mir.add" => {
            if let (Some(l), Some(r)) = (&lhs, &rhs)
                && let Some(tmem) = combine_tmem_addr(l, r, spec.block_size, spec.block_y)
                    .or_else(|| combine_tmem_addr(r, l, spec.block_size, spec.block_y))
            {
                Some(tmem)
            } else if let (
                Some(SsaKind::SmemSharedOffset {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                }),
                Some(SsaKind::ConstInt { value: delta, .. }),
            )
            | (
                Some(SsaKind::ConstInt { value: delta, .. }),
                Some(SsaKind::SmemSharedOffset {
                    shared_id,
                    size,
                    elem_bytes,
                    byte_offset,
                }),
            ) = (&lhs, &rhs)
            {
                let add_bytes = u64::try_from(*delta).unwrap_or(u64::MAX);
                Some(SsaKind::SmemSharedOffset {
                    shared_id: *shared_id,
                    size: *size,
                    elem_bytes: *elem_bytes,
                    byte_offset: byte_offset.saturating_add(add_bytes),
                })
            } else if let (
                Some(SsaKind::BlockBase1D { bit_width }),
                Some(SsaKind::Index(IdxExpr::Tx)),
            )
            | (
                Some(SsaKind::Index(IdxExpr::Tx)),
                Some(SsaKind::BlockBase1D { bit_width }),
            ) = (&lhs, &rhs)
                && (*bit_width >= 64 || spec.u32_coordinates)
            {
                Some(SsaKind::Global1DIdx {
                    offset: 0,
                    group_size: 1,
                })
            } else if let (
                Some(SsaKind::Global1DIdx { offset, group_size }),
                Some(SsaKind::ConstInt { value: c, .. }),
            )
            | (
                Some(SsaKind::ConstInt { value: c, .. }),
                Some(SsaKind::Global1DIdx { offset, group_size }),
            ) = (&lhs, &rhs)
            {
                Some(SsaKind::Global1DIdx {
                    offset: offset.saturating_add(u64::try_from(*c).unwrap_or(u64::MAX)),
                    group_size: *group_size,
                })
            } else if let Some((a, b)) = bin_idx_exprs(&lhs, &rhs)
                && (a.has_coord() || b.has_coord())
            {
                let sum = IdxExpr::Add(Box::new(a), Box::new(b));
                Some(if sum.single_loop_scope().is_valid() {
                    SsaKind::Index(sum)
                } else {
                    SsaKind::DataValue
                })
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.eq" | "mir.ne" | "mir.ge" => {
            if let Some((a, b)) = bin_idx_exprs(&lhs, &rhs)
                && (a.has_coord() || b.has_coord())
            {
                let g = match opid {
                    "mir.eq" => GuardExpr::Eq(a, b),
                    "mir.ne" => GuardExpr::Ne(a, b),
                    _ => GuardExpr::Ge(a, b),
                };
                Some(SsaKind::Guard(g))
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.lt" => {
            if let Some((a, b)) = bin_idx_exprs(&lhs, &rhs)
                && a.has_coord()
            {
                Some(SsaKind::Guard(GuardExpr::Lt(a, b)))
            } else if let (
                Some(SsaKind::Global1DIdx {
                    offset: 0,
                    group_size,
                }),
                Some(SsaKind::SliceLen(guard_param)),
            ) = (&lhs, &rhs)
            {
                Some(SsaKind::Global1DLtLen {
                    guard_param: *guard_param,
                    group_size: *group_size,
                })
            } else if let (
                Some(SsaKind::SregCtaidX { bit_width }),
                Some(SsaKind::SliceLen(guard_param)),
            ) = (&lhs, &rhs)
                && (*bit_width >= 64 || spec.u32_coordinates)
            {
                Some(SsaKind::Global1DLtLen {
                    guard_param: *guard_param,
                    group_size: u64::from(spec.block_size),
                })
            } else {
                Some(uniform_if_both(&lhs, &rhs))
            }
        }
        "mir.not" => Some(match lhs {
            Some(SsaKind::Guard(g)) => SsaKind::Guard(g.negate()),
            Some(SsaKind::Global1DLtLen {
                guard_param,
                group_size,
            }) => SsaKind::Global1DGeLen {
                guard_param,
                group_size,
            },
            Some(SsaKind::Global1DGeLen {
                guard_param,
                group_size,
            }) => SsaKind::Global1DLtLen {
                guard_param,
                group_size,
            },
            Some(SsaKind::MbarrierReady { bar_id }) => SsaKind::MbarrierNotReady { bar_id },
            Some(SsaKind::MbarrierNotReady { bar_id }) => SsaKind::MbarrierReady { bar_id },
            Some(SsaKind::UniformScalar | SsaKind::ConstInt { .. }) => SsaKind::UniformScalar,
            _ => SsaKind::DataValue,
        }),
        "nvvm.mbarrier_try_wait_parity_shared"
        | "nvvm.mbarrier_try_wait_shared"
        | "nvvm.mbarrier_test_wait_shared"
        | "nvvm.mbarrier_try_wait_parity_cluster"
        | "nvvm.mbarrier_try_wait_parity_cluster_relaxed" => {
            (op.get_num_results() == 1).then_some(match lhs {
                Some(SsaKind::MbarrierPtr { bar_id }) => SsaKind::MbarrierReady { bar_id },
                _ => SsaKind::DataValue,
            })
        }
        "mir.constant" | "builtin.constant" => {
            if let Some(val) = int_attr_u64(
                &op.attributes,
                if opid == "mir.constant" {
                    "value"
                } else {
                    "builtin_constant_value"
                },
            ) {
                Some(SsaKind::ConstInt {
                    width: 64,
                    value: u128::from(val),
                })
            } else {
                (op.get_num_results() == 1).then_some(SsaKind::DataValue)
            }
        }
        "mir.shared_alloc" => {
            let full =
                get_string_attr(&op.attributes, "source_name").unwrap_or_else(|| "SHARED".into());
            let short = full.rsplit("::").next().unwrap_or(&full).to_string();
            if is_mbarrier_shared_alloc(ctx, op) {
                let bar_id = bar_raw_list.iter().position(|n| n == &short).unwrap_or(0);
                Some(SsaKind::MbarrierPtr { bar_id })
            } else {
                let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                let elem_bytes = shared_alloc_elem_bytes(ctx, op);
                let shared_id = shared_raw_list
                    .iter()
                    .position(|(n, _)| n == &short)
                    .unwrap_or(0);
                Some(SsaKind::SharedBasePtr {
                    shared_id,
                    size,
                    elem_bytes,
                })
            }
        }
        "mir.cast" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
            lhs.map(|k| eval_int_cast(ctx, op, k, spec.block_size, spec.block_y))
        }
        "mir.construct_struct" if op.get_num_results() == 1 => {
            let fields = op
                .operands()
                .map(|v| env.get_kind(&v).unwrap_or(SsaKind::DataValue))
                .collect();
            Some(SsaKind::StructFields(fields))
        }
        "mir.insert_field" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
            let field = attribute_field_index(&op.attributes, "field_index")
                .or_else(|| attribute_field_index(&op.attributes, "index"));
            match (lhs, field) {
                (Some(SsaKind::StructFields(mut fs)), Some(idx)) if (idx as usize) < fs.len() => {
                    fs[idx as usize] = rhs.unwrap_or(SsaKind::DataValue);
                    Some(SsaKind::StructFields(fs))
                }
                _ => Some(SsaKind::DataValue),
            }
        }
        "mir.extract_field" => Some(
            match (lhs, attribute_field_index(&op.attributes, "index")) {
                (Some(SsaKind::ThreadIndexStruct), Some(0)) => SsaKind::Global1DIdx {
                    offset: 0,
                    group_size: 1,
                },
                (Some(SsaKind::BufferParam(p)), Some(0)) => SsaKind::SlicePtr(p),
                (Some(SsaKind::BufferParam(p)), Some(1)) => SsaKind::SliceLen(p),
                (Some(SsaKind::TmemLoadedReg { ld_id }), _) => SsaKind::TmemLoadedReg { ld_id },
                (Some(SsaKind::UniformScalar), _) => SsaKind::UniformScalar,
                (Some(SsaKind::StructFields(fs)), Some(idx)) => {
                    fs.get(idx as usize).cloned().unwrap_or(SsaKind::DataValue)
                }
                _ => SsaKind::DataValue,
            },
        ),
        "nvvm.cvta_generic_to_shared_offset"
            if op.get_num_operands() >= 1 && op.get_num_results() == 1 =>
        {
            Some(match lhs {
                Some(SsaKind::SharedBasePtr {
                    shared_id,
                    size,
                    elem_bytes,
                }) if spec.no_oob_shared.contains(&shared_id)
                    && spec.no_data_race.contains(&shared_id) =>
                {
                    SsaKind::SmemSharedOffset {
                        shared_id,
                        size,
                        elem_bytes,
                        byte_offset: 0,
                    }
                }
                Some(SsaKind::SharedElementPtr {
                    shared_id,
                    size,
                    addr: IdxExpr::Const(elem_idx),
                }) if spec.no_oob_shared.contains(&shared_id)
                    && spec.no_data_race.contains(&shared_id) =>
                {
                    let elem_bytes = shared_raw_list
                        .get(shared_id)
                        .and_then(|(n, _)| shared_elem_bytes_map.get(n).copied())
                        .unwrap_or(0);
                    SsaKind::SmemSharedOffset {
                        shared_id,
                        size,
                        elem_bytes,
                        byte_offset: elem_idx.saturating_mul(elem_bytes),
                    }
                }
                _ => SsaKind::DataValue,
            })
        }
        "mir.load" if op.get_num_operands() >= 1 && op.get_num_results() == 1 => Some(match lhs {
            Some(
                SsaKind::SharedBasePtr { shared_id, .. }
                | SsaKind::SharedElementPtr {
                    shared_id,
                    addr: IdxExpr::Const(0),
                    ..
                },
            ) if tmem_addr_shared_slots.contains_key(&shared_id) => SsaKind::TmemAddr {
                alloc_cols: tmem_addr_shared_slots[&shared_id],
                lane_expr: IdxExpr::Const(0),
                col_expr: IdxExpr::Const(0),
            },
            Some(SsaKind::StackAlloca(alloca)) => stack_struct_fields
                .get(&alloca)
                .cloned()
                .map(SsaKind::StructFields)
                .unwrap_or(SsaKind::DataValue),
            Some(SsaKind::StackFieldPtr { alloca, field }) => stack_struct_fields
                .get(&alloca)
                .and_then(|fs| fs.get(field).cloned())
                .unwrap_or(SsaKind::DataValue),
            _ => SsaKind::DataValue,
        }),
        "mir.ptr_offset" => {
            let shared_idx = rhs.as_ref().and_then(as_idx_expr);
            if let (
                Some(SsaKind::SharedBasePtr {
                    shared_id, size, ..
                }),
                Some(addr),
            ) = (lhs.clone(), shared_idx)
            {
                if graph.is_loop_expr_in_scope(&addr, b_idx) {
                    Some(SsaKind::SharedElementPtr {
                        shared_id,
                        size,
                        addr,
                    })
                } else {
                    (op.get_num_results() == 1).then_some(SsaKind::DataValue)
                }
            } else if let (
                Some(SsaKind::SlicePtr(param)),
                Some(SsaKind::Global1DIdx { offset, group_size }),
            ) = (&lhs, &rhs)
            {
                Some(SsaKind::Global1DElementPtr {
                    param: *param,
                    offset: *offset,
                    group_size: *group_size,
                })
            } else if let (Some(SsaKind::SlicePtr(param)), Some(SsaKind::SregCtaidX { bit_width })) =
                (&lhs, &rhs)
                && (*bit_width >= 64 || spec.u32_coordinates)
            {
                Some(SsaKind::Global1DElementPtr {
                    param: *param,
                    offset: 0,
                    group_size: u64::from(spec.block_size),
                })
            } else {
                (op.get_num_results() == 1).then_some(SsaKind::DataValue)
            }
        }
        _ => None,
    }
}

/// Run fixed-point SSA operand propagation over `graph.blocks` and return the converged [`SsaEnv`].
/// Resolves loop counters (`LoopVar`), uniform scalars (`UniformScalar`), `SmemDesc`, and `TmemAddr`.
pub(super) fn propagate_phased_ssa(
    ctx: &Context,
    graph: &BlockGraph,
    spec: &KernelVerificationSpec,
    bar_raw_list: &[String],
    shared_raw_list: &[(String, u64)],
    shared_elem_bytes_map: &HashMap<String, u64>,
    tmem_ld_id_map: &HashMap<(usize, usize), usize>,
) -> SsaEnv {
    let num_blocks = graph.len();
    let mut env = SsaEnv::new();
    for (idx, arg) in graph.blocks[0].deref(ctx).arguments().enumerate() {
        let arg_ty = arg.get_type(ctx).deref(ctx);
        let is_slice = arg_ty
            .downcast_ref::<dialect_mir::types::MirSliceType>()
            .is_some()
            || arg_ty
                .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                .is_some();
        env.bind_arg(
            arg,
            idx,
            if is_slice {
                SsaKind::BufferParam(idx)
            } else {
                SsaKind::UniformScalar
            },
        );
    }

    let mut tmem_addr_shared_slots: HashMap<usize, u64> = HashMap::new();
    let mut stack_struct_fields: HashMap<Value, Vec<SsaKind>> = HashMap::new();
    let max_ssa_passes = num_blocks + 4;

    for pass_idx in 0..max_ssa_passes {
        let mut changed = false;
        let mut stack_changed = false;
        let mut update = |env: &mut SsaEnv, val: Value, kind: SsaKind| {
            let old = env.get_kind(&val);
            if old.as_ref() != Some(&kind) {
                if kind == SsaKind::DataValue
                    && matches!(old, Some(SsaKind::Index(IdxExpr::LoopVar { .. })))
                {
                    return;
                }
                env.bind_result(val, kind);
                changed = true;
            }
        };

        for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
            let block = b_ptr.deref(ctx);
            if b_idx != 0 {
                for (arg_pos, arg) in block.arguments().enumerate() {
                    let mut incoming_uniform: Option<SsaKind> = None;
                    let mut all_agree = true;
                    let mut all_scalar_uniform = true;
                    let mut saw_any = false;
                    let mut saw_missing = false;
                    let mut pred_blocks: Vec<usize> = Vec::new();
                    let mut merge_incoming = |p_idx: usize, opt_k: Option<SsaKind>| {
                        pred_blocks.push(p_idx);
                        let Some(k) = opt_k else {
                            saw_missing = true;
                            all_agree = false;
                            return;
                        };
                        saw_any = true;
                        if !matches!(k, SsaKind::UniformScalar | SsaKind::ConstInt { .. }) {
                            all_scalar_uniform = false;
                        }
                        match &incoming_uniform {
                            None => incoming_uniform = Some(k),
                            Some(prev) if *prev != k => all_agree = false,
                            _ => {}
                        }
                    };
                    for (p_idx, &pred_ptr) in graph.blocks.iter().enumerate() {
                        for pred_op_ptr in pred_ptr.deref(ctx).iter(ctx) {
                            let pred_opid = op_name(ctx, pred_op_ptr);
                            let pred_op = pred_op_ptr.deref(ctx);
                            if pred_opid == "mir.goto"
                                && pred_op.get_num_successors() == 1
                                && pred_op.get_successor(0) == b_ptr
                                && arg_pos < pred_op.get_num_operands()
                            {
                                merge_incoming(p_idx, op_kind(&env, &pred_op, arg_pos));
                            } else if pred_opid == "mir.cond_br"
                                && pred_op.get_num_successors() == 2
                            {
                                let t_len = pred_op.get_successor(0).deref(ctx).get_num_arguments();
                                if pred_op.get_successor(0) == b_ptr && arg_pos < t_len {
                                    merge_incoming(p_idx, op_kind(&env, &pred_op, 1 + arg_pos));
                                }
                                if pred_op.get_successor(1) == b_ptr
                                    && 1 + t_len + arg_pos < pred_op.get_num_operands()
                                {
                                    merge_incoming(
                                        p_idx,
                                        op_kind(&env, &pred_op, 1 + t_len + arg_pos),
                                    );
                                }
                            }
                        }
                    }
                    if !saw_any {
                        continue;
                    }
                    let split_by_nonuniform = (0..num_blocks).any(|br_idx| {
                        graph.blocks[br_idx].deref(ctx).iter(ctx).any(|op_ptr| {
                            let opid = op_name(ctx, op_ptr);
                            let op = op_ptr.deref(ctx);
                            if opid == "mir.cond_br"
                                && op.get_num_operands() >= 1
                                && op.get_num_successors() == 2
                                && !is_uniform_or_const(op_kind(&env, &op, 0).as_ref())
                                && let (Some(t_dst), Some(f_dst)) = (
                                    graph.idx_of(op.get_successor(0)),
                                    graph.idx_of(op.get_successor(1)),
                                )
                            {
                                let in_t = pred_blocks.iter().any(|&p| {
                                    p != br_idx
                                        && !graph.can_reach_without(0, p, br_idx)
                                        && !graph.can_reach_without(f_dst, p, br_idx)
                                }) || (pred_blocks.contains(&br_idx) && t_dst == b_idx);
                                let in_f = pred_blocks.iter().any(|&p| {
                                    p != br_idx
                                        && !graph.can_reach_without(0, p, br_idx)
                                        && !graph.can_reach_without(t_dst, p, br_idx)
                                }) || (pred_blocks.contains(&br_idx) && f_dst == b_idx);
                                in_t && in_f
                            } else {
                                false
                            }
                        })
                    });
                    let arg_kind = match incoming_uniform {
                        Some(k) if all_agree && !saw_missing => k,
                        Some(_) if all_scalar_uniform && !split_by_nonuniform => {
                            SsaKind::UniformScalar
                        }
                        _ => SsaKind::DataValue,
                    };
                    update(&mut env, arg, arg_kind);
                }
            }

            // Detect cyclic loop header `mir.lt(step_arg, const_bound)` -> `LoopVar`.
            let mut lt_in_blk: HashMap<Value, (Value, u64)> = HashMap::new();
            let mut not_in_blk: HashMap<Value, Value> = HashMap::new();
            let mut term_cond_br: Option<(Value, usize, usize)> = None;
            for op_ptr in block.iter(ctx) {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if opid == "mir.lt"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                    && let Some(SsaKind::ConstInt { value: bound, .. }) = op_kind(&env, &op, 1)
                {
                    lt_in_blk.insert(
                        op.get_result(0),
                        (op.get_operand(0), u64::try_from(bound).unwrap_or(u64::MAX)),
                    );
                } else if opid == "mir.not"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    not_in_blk.insert(op.get_result(0), op.get_operand(0));
                } else if opid == "mir.cond_br"
                    && op.get_num_operands() >= 1
                    && op.get_num_successors() == 2
                    && let (Some(t_idx), Some(f_idx)) = (
                        graph.idx_of(op.get_successor(0)),
                        graph.idx_of(op.get_successor(1)),
                    )
                {
                    term_cond_br = Some((op.get_operand(0), t_idx, f_idx));
                }
            }
            if let Some((cond_val, t_idx, f_idx)) = term_cond_br {
                let decoded = lt_in_blk
                    .get(&cond_val)
                    .map(|&(l, b)| (l, b, t_idx, f_idx))
                    .or_else(|| {
                        not_in_blk
                            .get(&cond_val)
                            .and_then(|iv| lt_in_blk.get(iv))
                            .map(|&(l, b)| (l, b, f_idx, t_idx))
                    });
                if let Some((lhs_val, bound, body_blk, exit_blk)) = decoded
                    && graph.can_reach(body_blk, b_idx)
                {
                    for (arg_pos, step_arg) in block.arguments().enumerate() {
                        if step_arg == lhs_val {
                            update(
                                &mut env,
                                step_arg,
                                SsaKind::Index(IdxExpr::LoopVar {
                                    loop_id: b_idx * 16 + arg_pos,
                                    header_block: b_idx,
                                    body_block: body_blk,
                                    exit_block: exit_blk,
                                    bound,
                                }),
                            );
                        }
                    }
                }
            }

            for (op_idx, op_ptr) in block.iter(ctx).enumerate() {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if pass_idx + 1 < max_ssa_passes
                    && op.get_num_operands() >= 1
                    && !matches!(opid.as_str(), "mir.goto" | "mir.cond_br" | "mir.return")
                    && op.operands().any(|v| env.get_kind(&v).is_none())
                {
                    stack_changed = true;
                    continue;
                }
                if opid.starts_with("nvvm.tcgen05_ld_") {
                    if let Some(&ld_id) = tmem_ld_id_map.get(&(b_idx, op_idx)) {
                        for r_idx in 0..op.get_num_results() {
                            update(
                                &mut env,
                                op.get_result(r_idx),
                                SsaKind::TmemLoadedReg { ld_id },
                            );
                        }
                    }
                    continue;
                }
                if matches!(
                    opid.as_str(),
                    "nvvm.tcgen05_alloc" | "nvvm.tcgen05_alloc_cg2"
                ) {
                    if let (
                        Some(
                            SsaKind::SharedBasePtr { shared_id, .. }
                            | SsaKind::SharedElementPtr {
                                shared_id,
                                addr: IdxExpr::Const(0),
                                ..
                            },
                        ),
                        Some(SsaKind::ConstInt { value: cols, .. }),
                    ) = bin_kinds(&env, &op)
                        && let Ok(alloc_cols) = u64::try_from(cols)
                    {
                        tmem_addr_shared_slots.insert(shared_id, alloc_cols);
                    }
                    continue;
                }
                if opid == "mir.store" {
                    if let (Some(dst_k), val_opt) = bin_kinds(&env, &op) {
                        let val_k = val_opt.unwrap_or(SsaKind::DataValue);
                        match (dst_k, val_k) {
                            (SsaKind::StackAlloca(alloca), SsaKind::StructFields(fs)) => {
                                if stack_struct_fields.insert(alloca, fs.clone()).as_ref()
                                    != Some(&fs)
                                {
                                    stack_changed = true;
                                }
                            }
                            (SsaKind::StackFieldPtr { alloca, field }, rhs_k) => {
                                let entry = stack_struct_fields.entry(alloca).or_default();
                                if entry.len() <= field {
                                    entry.resize(field + 1, SsaKind::DataValue);
                                }
                                if entry[field] != rhs_k {
                                    entry[field] = rhs_k;
                                    stack_changed = true;
                                }
                            }
                            _ => {}
                        }
                    }
                    continue;
                }
                if let Some(mapped) = eval_phased_ssa_op(
                    ctx,
                    &opid,
                    &op,
                    b_idx,
                    spec,
                    graph,
                    &env,
                    bar_raw_list,
                    shared_raw_list,
                    shared_elem_bytes_map,
                    &tmem_addr_shared_slots,
                    &stack_struct_fields,
                ) {
                    if op.get_num_results() >= 1 {
                        update(&mut env, op.get_result(0), mapped);
                    }
                } else if op.get_num_results() >= 1 {
                    let is_pure_uniform = opid.starts_with("mir.")
                        && !matches!(
                            opid.as_str(),
                            "mir.load"
                                | "mir.call"
                                | "mir.alloca"
                                | "mir.shared_alloc"
                                | "mir.ptr_offset"
                                | "mir.field_addr"
                                | "mir.array_element_addr"
                        )
                        && op.get_num_operands() >= 1
                        && op
                            .operands()
                            .all(|v| is_uniform_or_const(env.get_kind(&v).as_ref()));
                    let fallback = if is_pure_uniform {
                        SsaKind::UniformScalar
                    } else {
                        SsaKind::DataValue
                    };
                    for r_idx in 0..op.get_num_results() {
                        update(&mut env, op.get_result(r_idx), fallback.clone());
                    }
                }
            }
        }
        if !changed && !stack_changed {
            break;
        }
    }
    env
}
