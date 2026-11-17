/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared `dialect-mir` basic-block graph (`BlockGraph`) and Pliron operand,
//! constant, cast, and attribute helpers reused across all exporter stages.

use std::collections::BTreeSet;

use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        types::IntegerType,
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    printable::Printable,
    r#type::Typed,
};

use super::{AccessKind, PhasedSharedAccess};
use crate::ssa::{GuardExpr, IdxExpr, LoopScopeCheck, SsaEnv, SsaKind, TileAccessPatternKind};

/// Shared basic-block adjacency and structural metadata for a multi-block `mir.func`.
/// Centralizes live successor edges, pure `mir.unreachable` exhaustiveness sinks,
/// `nvvm.barrier0` sites, and reachability queries used during SSA operand resolution.
pub(super) struct BlockGraph {
    /// Ordered basic-block pointers (`0` is the entry block).
    pub(super) blocks: Vec<Ptr<BasicBlock>>,
    /// Live outgoing successor block indices per block (excluding pure `mir.unreachable` sinks).
    pub(super) succs: Vec<Vec<usize>>,
    /// Blocks ending in `mir.unreachable` without `nvvm.trap` (exhaustive enum fallthroughs).
    pub(super) pure_unreachable: BTreeSet<usize>,
    /// Coordinates `(block_idx, ord_in_block)` of every `nvvm.barrier0` operation.
    pub(super) barrier_sites: Vec<(usize, usize)>,
    /// Indices of all basic blocks ending in `mir.return`.
    pub(super) return_blocks: Vec<usize>,
}

impl BlockGraph {
    /// Scan all basic blocks of `blocks` once to populate adjacency lists, pure
    /// `mir.unreachable` sinks, `nvvm.barrier0` sites, and `mir.return` blocks.
    pub(super) fn build(ctx: &Context, blocks: Vec<Ptr<BasicBlock>>) -> Self {
        let n = blocks.len();
        let mut pure_unreachable = BTreeSet::new();
        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            let mut has_unreach = false;
            let mut has_trap = false;
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                match op_name(ctx, op_ptr).as_str() {
                    "mir.unreachable" => has_unreach = true,
                    "nvvm.trap" => has_trap = true,
                    _ => {}
                }
            }
            if has_unreach && !has_trap {
                pure_unreachable.insert(b_idx);
            }
        }
        let mut succs = vec![Vec::new(); n];
        let mut barrier_sites = Vec::new();
        let mut return_blocks = Vec::new();
        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            let mut ord = 0usize;
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                match opid.as_str() {
                    "nvvm.barrier0" => {
                        barrier_sites.push((b_idx, ord));
                        ord += 1;
                    }
                    "mir.return" => return_blocks.push(b_idx),
                    "mir.goto" | "mir.cond_br" => {
                        for s in 0..op.get_num_successors() {
                            if let Some(dst) = blocks.iter().position(|&b| b == op.get_successor(s))
                                && !pure_unreachable.contains(&dst)
                                && !succs[b_idx].contains(&dst)
                            {
                                succs[b_idx].push(dst);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Self {
            blocks,
            succs,
            pure_unreachable,
            barrier_sites,
            return_blocks,
        }
    }

    /// Return the total number of basic blocks in the function region.
    /// Entry block is always index `0` when non-empty.
    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Look up the 0-based block index of `target` in `self.blocks`.
    /// Returns `None` if `target` is outside this function region.
    pub(super) fn idx_of(&self, target: Ptr<BasicBlock>) -> Option<usize> {
        self.blocks.iter().position(|&b| b == target)
    }

    /// Return true if `start` can reach `target` without visiting `avoid1` or `avoid2`.
    /// Used for dominance, loop-header, and cyclic barrier ordering queries.
    pub(super) fn can_reach_avoiding_two(
        &self,
        start: usize,
        target: usize,
        avoid1: usize,
        avoid2: usize,
    ) -> bool {
        if start == avoid1 || start == avoid2 || start >= self.len() {
            return false;
        }
        let mut visited = vec![false; self.len()];
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(curr) = stack.pop() {
            if curr == target {
                return true;
            }
            for &next in &self.succs[curr] {
                if next != avoid1 && next != avoid2 && !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        false
    }

    /// Return true if `start` can reach `target` without visiting `forbidden`.
    /// Delegates to [`Self::can_reach_avoiding_two`] with `usize::MAX` as the second sentinel.
    pub(super) fn can_reach_without(&self, start: usize, target: usize, forbidden: usize) -> bool {
        self.can_reach_avoiding_two(start, target, forbidden, usize::MAX)
    }

    /// Return true if `start` can reach `target` along live CFG edges.
    /// Excludes pure `mir.unreachable` enum-exhaustiveness fallthrough sinks.
    pub(super) fn can_reach(&self, start: usize, target: usize) -> bool {
        self.can_reach_without(start, target, usize::MAX)
    }

    /// Return true when block `b` lies on a directed cycle (`b ->+ b`).
    /// Distinguishes loop-body blocks from acyclic prologue and epilogue blocks.
    pub(super) fn is_cyclic(&self, b: usize) -> bool {
        self.succs[b].iter().any(|&s| self.can_reach(s, b))
    }

    /// Return true if `expr` uses at most one `LoopVar` and `b_idx` sits inside its `< bound` body.
    /// Rejects post-loop `LoopVar` leaks (`i == bound`) and conflicting multi-loop expressions.
    pub(super) fn is_loop_expr_in_scope(&self, expr: &IdxExpr, b_idx: usize) -> bool {
        match expr.single_loop_scope() {
            LoopScopeCheck::None => true,
            LoopScopeCheck::Single {
                header_block,
                body_block,
                exit_block,
                ..
            } => {
                !self.can_reach_without(0, b_idx, body_block)
                    && !self.can_reach_without(exit_block, b_idx, header_block)
            }
            LoopScopeCheck::Conflict => false,
        }
    }
}

/// Return the string opcode identifier (`"mir.add"`, `"nvvm.barrier0"`, ...) of `op_ptr`.
/// Avoids repeating `Operation::get_opid(op_ptr, ctx).to_string()` at every match site.
pub(super) fn op_name(ctx: &Context, op_ptr: Ptr<Operation>) -> String {
    Operation::get_opid(op_ptr, ctx).to_string()
}

/// Look up the [`SsaKind`] of operand `idx` on `op` when `idx < op.get_num_operands()`.
/// Returns `None` if `op` has fewer than `idx + 1` operands or the operand is not yet bound.
pub(super) fn op_kind(env: &SsaEnv, op: &Operation, idx: usize) -> Option<SsaKind> {
    (idx < op.get_num_operands())
        .then(|| env.get_kind(&op.get_operand(idx)))
        .flatten()
}

/// Look up the `(lhs, rhs)` [`SsaKind`] pair of a binary MIR or NVVM operation.
/// Safely returns `None` for missing or not-yet-bound operands.
pub(super) fn bin_kinds(env: &SsaEnv, op: &Operation) -> (Option<SsaKind>, Option<SsaKind>) {
    (op_kind(env, op, 0), op_kind(env, op, 1))
}

/// Extract a compositional [`IdxExpr`] from an [`SsaKind`] (`Index` or `ConstInt`).
/// Returns `None` for non-index SSA values (`DataValue`, `UniformScalar`, pointers).
pub(super) fn as_idx_expr(kind: &SsaKind) -> Option<IdxExpr> {
    match kind {
        SsaKind::Index(expr) => Some(expr.clone()),
        SsaKind::ConstInt { value, .. } => u64::try_from(*value).ok().map(IdxExpr::Const),
        _ => None,
    }
}

/// Extract `(lhs_idx, rhs_idx)` when both operands of a binary op convert to [`IdxExpr`].
/// Used across `mir.add`, `mir.mul`, `mir.bitor`, and coordinate comparison ops.
pub(super) fn bin_idx_exprs(
    lhs: &Option<SsaKind>,
    rhs: &Option<SsaKind>,
) -> Option<(IdxExpr, IdxExpr)> {
    Some((
        lhs.as_ref().and_then(as_idx_expr)?,
        rhs.as_ref().and_then(as_idx_expr)?,
    ))
}

/// Return true when `kind` is a CTA-uniform scalar parameter/counter or integer literal.
/// Used to distinguish uniform loop bounds (`tile < num_tiles`) from thread-divergent branches.
pub(super) fn is_uniform_or_const(kind: Option<&SsaKind>) -> bool {
    matches!(
        kind,
        Some(SsaKind::UniformScalar | SsaKind::ConstInt { .. })
    )
}

/// Return `SsaKind::UniformScalar` when both `lhs` and `rhs` are CTA-uniform, else `DataValue`.
/// Serves as the default fallback for binary arithmetic and comparison operations.
pub(super) fn uniform_if_both(lhs: &Option<SsaKind>, rhs: &Option<SsaKind>) -> SsaKind {
    if is_uniform_or_const(lhs.as_ref()) && is_uniform_or_const(rhs.as_ref()) {
        SsaKind::UniformScalar
    } else {
        SsaKind::DataValue
    }
}

/// Decode an integer constant attribute (`"value"` or `"builtin_constant_value"`) as `(width, value)`.
/// Returns `None` for float or non-integer constant attributes.
pub(super) fn decode_int_constant(op: &Operation, opid: &str) -> Option<(u32, u128)> {
    let key = if opid == "mir.constant" {
        "value"
    } else {
        "builtin_constant_value"
    };
    let int_attr = op
        .attributes
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key)
        .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())?;
    let value = int_attr
        .value()
        .to_string_unsigned_decimal()
        .parse::<u128>()
        .ok()?;
    let width = u32::try_from(int_attr.value().bw()).unwrap_or(64);
    Some((width, value))
}

/// Evaluate an integer/pointer `mir.cast` on `kind`, preserving widening unsigned `IdxExpr`s
/// and wrapping narrowing unsigned casts in `Mod(..., 2^w)` when wraparound is possible.
pub(super) fn eval_int_cast(
    ctx: &Context,
    op: &Operation,
    kind: SsaKind,
    block_size: u32,
    block_y: u32,
) -> SsaKind {
    let src_ty = op.get_operand(0).get_type(ctx).deref(ctx);
    let dst_ty = op.get_result(0).get_type(ctx).deref(ctx);
    match (
        src_ty.downcast_ref::<IntegerType>(),
        dst_ty.downcast_ref::<IntegerType>(),
    ) {
        (Some(si), Some(di)) => {
            if si.is_unsigned() && di.is_unsigned() {
                if di.width() >= si.width() {
                    match kind {
                        SsaKind::SregTidX { .. } => SsaKind::SregTidX {
                            bit_width: di.width(),
                        },
                        SsaKind::SregCtaidX { .. } => SsaKind::SregCtaidX {
                            bit_width: di.width(),
                        },
                        SsaKind::SregNtidX { .. } => SsaKind::SregNtidX {
                            bit_width: di.width(),
                        },
                        SsaKind::BlockBase1D { .. } => SsaKind::BlockBase1D {
                            bit_width: di.width(),
                        },
                        other => other,
                    }
                } else {
                    match kind {
                        SsaKind::ConstInt { value, .. } => {
                            let mask = if di.width() >= 128 {
                                u128::MAX
                            } else {
                                (1u128 << di.width()) - 1
                            };
                            SsaKind::ConstInt {
                                width: di.width(),
                                value: value & mask,
                            }
                        }
                        SsaKind::Index(expr) if di.width() < 64 => {
                            let modulus = 1u64 << di.width();
                            if expr
                                .exclusive_upper_bound(block_size, block_y)
                                .is_some_and(|ub| ub <= modulus)
                            {
                                SsaKind::Index(expr)
                            } else {
                                SsaKind::Index(IdxExpr::Mod(Box::new(expr), modulus))
                            }
                        }
                        SsaKind::UniformScalar => SsaKind::UniformScalar,
                        _ => SsaKind::DataValue,
                    }
                }
            } else if matches!(kind, SsaKind::UniformScalar | SsaKind::ConstInt { .. }) {
                SsaKind::UniformScalar
            } else {
                SsaKind::DataValue
            }
        }
        (None, None) => kind,
        _ => SsaKind::DataValue,
    }
}

/// Extract the numeric index from a `mir.field_index` or enum-variant attribute on `attrs`.
/// Used on `mir.extract_field`, `mir.insert_field`, `mir.field_addr`, and `mir.construct_enum`.
pub(super) fn attribute_field_index(attrs: &AttributeDict, key_name: &str) -> Option<u32> {
    let text = attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key_name)
        .map(|(_, v)| format!("{v:?}"))?;
    text.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<u32>()
        .ok()
}

/// Extract an unsigned integer attribute by key as `u64`.
/// Returns `None` if `key` is absent or does not fit in `u64`.
pub(super) fn int_attr_u64(attrs: &AttributeDict, key: &str) -> Option<u64> {
    attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key)
        .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
        .and_then(|attr| {
            attr.value()
                .to_string_unsigned_decimal()
                .parse::<u64>()
                .ok()
        })
}

/// Compute the byte width of one element in a `mir.shared_alloc` operation (`0` if unknown).
/// Resolves integer bit widths (`u8`, `u16`, `u32`, `u64`) and standard floating-point types.
pub(super) fn shared_alloc_elem_bytes(ctx: &Context, op: &Operation) -> u64 {
    op.attributes
        .0
        .iter()
        .find(|(k, _)| k.to_string() == "elem_type")
        .and_then(|(_, v)| v.downcast_ref::<TypeAttr>())
        .map(|ta| {
            let ty_handle = ta.get_type(ctx);
            let ty_ref = ty_handle.deref(ctx);
            if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
                u64::from(int_ty.width()).div_ceil(8)
            } else {
                match ty_handle.disp(ctx).to_string().trim() {
                    "builtin.fp16" | "builtin.bf16" => 2,
                    "builtin.fp32" => 4,
                    "builtin.fp64" => 8,
                    _ => 0,
                }
            }
        })
        .unwrap_or(0)
}

/// Return true if `op` is a `mir.shared_alloc` for a 64-bit hardware `Barrier` (`mbarrier`).
/// Matches `size == 1` with a 64-bit integer element type (`ui64` / `i64`).
pub(super) fn is_mbarrier_shared_alloc(ctx: &Context, op: &Operation) -> bool {
    int_attr_u64(&op.attributes, "size") == Some(1)
        && op.attributes.0.iter().any(|(k, v)| {
            k.to_string() == "elem_type"
                && v.downcast_ref::<TypeAttr>()
                    .and_then(|ta| {
                        ta.get_type(ctx)
                            .deref(ctx)
                            .downcast_ref::<IntegerType>()
                            .map(|i| i.width() == 64)
                    })
                    .unwrap_or(false)
        })
}

/// Append a [`PhasedSharedAccess`] entry using the resolved `SharedArray` short name.
/// Shared across TMA copies, `tcgen05_alloc`, `tcgen05_mma`/`cp`, and generic `load`/`store` sites.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_shared_access(
    out: &mut Vec<PhasedSharedAccess>,
    shared_raw_list: &[(String, u64)],
    kind: AccessKind,
    shared_id: usize,
    buffer_len: u64,
    phase: usize,
    pattern: TileAccessPatternKind,
    loop_bound: u64,
    leader_only: bool,
    active_threads: u64,
    guard: GuardExpr,
) {
    let shared_name = shared_raw_list
        .get(shared_id)
        .map(|(n, _)| n.clone())
        .unwrap_or_else(|| format!("SHARED_{shared_id}"));
    out.push(PhasedSharedAccess {
        kind,
        shared_id,
        shared_name,
        buffer_len,
        phase,
        pattern,
        loop_bound,
        leader_only,
        active_threads,
        guard,
    });
}
