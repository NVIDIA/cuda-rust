/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared `dialect-mir` basic-block graph (`BlockGraph`) and Pliron attribute,
//! constant, and `tcgen05` shape helpers reused across the exporter.

use std::collections::BTreeSet;

use dialect_mir::types::{MirPtrType, MirStructType};
use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{IntegerAttr, TypeAttr},
        types::IntegerType,
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    printable::Printable,
    r#type::TypeHandle,
};

/// Shared basic-block adjacency and structural metadata for a multi-block `mir.func`.
/// Centralizes live successor edges and pure `mir.unreachable` exhaustiveness sinks.
pub(super) struct BlockGraph {
    /// Ordered basic-block pointers (`0` is the entry block).
    pub(super) blocks: Vec<Ptr<BasicBlock>>,
    /// Live outgoing successor block indices per block (excluding pure `mir.unreachable` sinks).
    pub(super) succs: Vec<Vec<usize>>,
    /// Blocks ending in `mir.unreachable` without `nvvm.trap` (exhaustive enum fallthroughs).
    pub(super) pure_unreachable: BTreeSet<usize>,
}

impl BlockGraph {
    /// Scan all basic blocks of `blocks` once to populate adjacency lists and pure
    /// `mir.unreachable` sinks.
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
        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if matches!(opid.as_str(), "mir.goto" | "mir.cond_br") {
                    for s in 0..op.get_num_successors() {
                        if let Some(dst) = blocks.iter().position(|&b| b == op.get_successor(s))
                            && !pure_unreachable.contains(&dst)
                            && !succs[b_idx].contains(&dst)
                        {
                            succs[b_idx].push(dst);
                        }
                    }
                }
            }
        }
        Self {
            blocks,
            succs,
            pure_unreachable,
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
}

/// Return the string opcode identifier (`"mir.add"`, `"nvvm.barrier0"`, ...) of `op_ptr`.
/// Avoids repeating `Operation::get_opid(op_ptr, ctx).to_string()` at every match site.
pub(super) fn op_name(ctx: &Context, op_ptr: Ptr<Operation>) -> String {
    Operation::get_opid(op_ptr, ctx).to_string()
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

/// Peel single-field `MirStructType` wrappers (such as `Barrier(UnsafeCell<u64>)`)
/// down to their underlying leaf `TypeHandle`.
fn peel_single_field_structs(ctx: &Context, mut ty: TypeHandle) -> TypeHandle {
    for _ in 0..8 {
        let next = {
            let ty_ref = ty.deref(ctx);
            let Some(st) = ty_ref.downcast_ref::<MirStructType>() else {
                break;
            };
            if st.field_types.len() != 1 {
                break;
            }
            st.field_types[0]
        };
        ty = next;
    }
    ty
}

/// Compute the byte width of a `dialect-mir` element `TypeHandle` (`0` if unknown).
/// Resolves integer bit widths (`u8..u128`), floats, and 1-field wrapper structs.
pub(super) fn mir_type_elem_bytes(ctx: &Context, ty: TypeHandle) -> u64 {
    let ty_handle = peel_single_field_structs(ctx, ty);
    let ty_ref = ty_handle.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        u64::from(int_ty.width()).div_ceil(8)
    } else if let Some(st) = ty_ref.downcast_ref::<MirStructType>()
        && st.total_size > 0
    {
        st.total_size
    } else {
        match ty_handle.disp(ctx).to_string().trim() {
            "builtin.fp16" | "builtin.bf16" | "mir.fp16" => 2,
            "builtin.fp32" => 4,
            "builtin.fp64" => 8,
            _ => 0,
        }
    }
}

/// Extract the pointee byte width of a `MirPtrType` (`0` if not a `MirPtrType` or unknown).
pub(super) fn ptr_pointee_elem_bytes(ctx: &Context, ptr_ty: TypeHandle) -> u64 {
    ptr_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map_or(0, |pt| mir_type_elem_bytes(ctx, pt.pointee))
}

/// Compute the byte width of one element in a `mir.shared_alloc` operation (`0` if unknown).
/// Resolves integer bit widths (`u8`, `u16`, `u32`, `u64`), floats, and 1-field wrapper structs.
pub(super) fn shared_alloc_elem_bytes(ctx: &Context, op: &Operation) -> u64 {
    op.attributes
        .0
        .iter()
        .find(|(k, _)| k.to_string() == "elem_type")
        .and_then(|(_, v)| v.downcast_ref::<TypeAttr>())
        .map_or(0, |ta| mir_type_elem_bytes(ctx, ta.get_type(ctx)))
}

/// Return true if `op` is a `mir.shared_alloc` for a scalar `static mut BAR: Barrier`.
/// Matches `size == 1`, direct 64-bit integer element type (`ui64` / `i64`), and no `SharedArray` alignment attribute.
pub(super) fn is_mbarrier_shared_alloc(ctx: &Context, op: &Operation) -> bool {
    int_attr_u64(&op.attributes, "size") == Some(1)
        && !op
            .attributes
            .0
            .iter()
            .any(|(k, _)| k.to_string() == "alignment")
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

/// Parse `(height_lanes, width_cols)` from an `nvvm.tcgen05_ld_*` operation name.
/// Covers `32x32b`, `16x64b`, `16x128b`, `16x256b`, and `16x32bx2` register shapes.
pub(super) fn parse_tcgen05_ld_shape(opid: &str) -> Option<(u64, u64)> {
    let rest = opid.strip_prefix("nvvm.tcgen05_ld_")?;
    let rest = rest.strip_suffix("_raw").unwrap_or(rest);
    let rest = rest.strip_suffix("_pure").unwrap_or(rest);
    let rest = rest.strip_suffix("_pack16").unwrap_or(rest);
    let (shape_part, mult_part) = rest.split_once("_x")?;
    let mult = mult_part.parse::<u64>().ok()?;
    match shape_part {
        "32x32b" => Some((32, mult)),
        "16x64b" => Some((16, 2 * mult)),
        "16x128b" => Some((16, 4 * mult)),
        "16x256b" => Some((16, 8 * mult)),
        "16x32bx2" => Some((32, mult)),
        _ => None,
    }
}

/// Parse `(height_lanes, width_cols)` from an `nvvm.tcgen05_cp_*` operation name.
/// Covers 128-lane and warpx2/warpx4 SMEM-to-TMEM async matrix copy shapes.
pub(super) fn parse_tcgen05_cp_shape(opid: &str) -> Option<(u64, u64)> {
    let rest = opid.strip_prefix("nvvm.tcgen05_cp_")?;
    let rest = rest.strip_suffix("_cg2").unwrap_or(rest);
    match rest {
        "128x256b" => Some((128, 8)),
        "128x128b" => Some((128, 4)),
        "4x256b" => Some((4, 8)),
        "32x128b_warpx2_01_13" | "32x128b_warpx2_02_13" => Some((64, 4)),
        "32x128b_warpx4" => Some((128, 4)),
        _ => None,
    }
}

/// Decode `(height_lanes, width_cols)` from a constant 32-bit `Tcgen05InstructionDescriptor`.
/// Halves the per-CTA `M` lane height when `.cta_group::2` (`is_cg2`) is active.
pub(super) fn decode_tcgen05_idesc_shape(raw: u128, is_cg2: bool) -> Option<(u64, u64)> {
    let v = u32::try_from(raw).ok()?;
    let n_dim = u64::from(((v >> 17) & 0x3F) * 8);
    let m_full = u64::from(((v >> 24) & 0x1F) * 16);
    if n_dim == 0 || m_full == 0 {
        return None;
    }
    Some((if is_cg2 { m_full / 2 } else { m_full }, n_dim))
}
