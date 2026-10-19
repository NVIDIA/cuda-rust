/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-operation bytecode serialization dispatch.
//!
//! Each operation has a specific serialization format determined by its
//! definition in `Ops.td`. The field order is:
//!
//! 1. Result types (type indices)
//! 2. Variadic result count (only if op has variadic results)
//! 3. Flags field (bitfield for optional attributes/operands)
//! 4. Required attributes (in declaration order, inline — no tag prefix)
//! 5. Optional attributes (only if flagged present)
//! 6. Operands (value indices, with or without size encoding)
//! 7. Regions (recursive serialization)
//!
//! Ported from generated `Bytecode.inc` in the `cuda-tile` submodule.

use super::encoding::EncodingWriter;
use super::opcode::Opcode;
use super::writer::{require_version, WriterCtx};
use super::BytecodeVersion;
use crate::ir::{Attribute, Operation};
use crate::requirements::{attribute_feature, Feature};
use crate::{Error, Result};

// =========================================================================
// Per-op dispatch entry point
// =========================================================================

/// Serialize the body of an operation (everything after the opcode varint).
///
/// The opcode itself is written by the caller before calling this function.
/// After this returns, the caller registers the op's result values.
pub(super) fn write_op_body(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    use Opcode::*;
    require_version(op.opcode.name(), op.opcode.minimum_version(), ctx.version)?;
    for (name, attr) in &op.attributes {
        if let Some(feature) = attribute_feature(op.opcode, name, attr) {
            require_version(feature.name(), feature.since(), ctx.version)?;
        }
    }
    match op.opcode {
        // ----- Simple: result types + operands (no size) -----
        AbsF
        | AbsI
        | AndI
        | Atan2
        | Bitcast
        | Broadcast
        | Ceil
        | Cos
        | CosH
        | Floor
        | FPowI
        | IntToPtr
        | Log
        | Log2
        | MakeGatherScatterView
        | MakeStridedView
        | MemoryFenceAliasTko
        | MmaFScaled
        | MulhiI
        | NegF
        | Offset
        | OrI
        | Pack
        | Pow
        | PtrToInt
        | PtrToPtr
        | RemF
        | Reshape
        | Select
        | Sin
        | SinH
        | Tan
        | Unpack
        | XOrI => {
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        Exp => {
            write_result_types(op, w, ctx)?;
            if find_op_attr(op, "rounding_mode")
                .is_some_and(|attr| !matches!(attr, Attribute::Integer(5, _)))
            {
                require_version(
                    "exp.rounding_mode",
                    Feature::ExpRounding.since(),
                    ctx.version,
                )?;
            }
            if ctx.version >= super::enums::BytecodeVersion::V13_3 {
                write_inline_attr_or_default(op, "rounding_mode", 5, w, ctx)?;
            }
            write_operands(op, w, ctx, false)?;
        }

        MmaF => {
            write_result_types(op, w, ctx)?;
            if flag_if_bool_true(op, "fast_acc", 0) != 0 {
                require_version(
                    "mmaf.fast_acc",
                    Feature::FastAccumulation.since(),
                    ctx.version,
                )?;
            }
            if ctx.version >= super::enums::BytecodeVersion::V13_3 {
                let flags = flag_if_bool_true(op, "fast_acc", 0);
                w.write_varint(flags);
            }
            write_operands(op, w, ctx, false)?;
        }

        Alloca => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_bool_true(op, "global", 0);
            w.write_varint(flags);
            write_inline_attr(op, "num_elem", w, ctx)?;
            write_inline_attr(op, "alignment", w, ctx)?;
        }

        // ----- v13.2: NegI gains overflow attr -----
        NegI => {
            write_result_types(op, w, ctx)?;
            // overflow defaults to NONE (0) if not set
            write_inline_attr_or_default(op, "overflow", 0, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        // ----- v13.2: TanH gains rounding_mode attr -----
        TanH => {
            write_result_types(op, w, ctx)?;
            // rounding_mode defaults to FULL (5) if not set
            write_inline_attr_or_default(op, "rounding_mode", 5, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- No operands, just result types -----
        Iota => {
            write_result_types(op, w, ctx)?;
        }

        // ----- Required attributes only -----
        AddI | MulI | SubI | ShLI | TruncI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "overflow", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        ShRI | ExtI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Cat => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Permute => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "permutation", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Assert => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "message", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Assume => {
            write_result_types(op, w, ctx)?;
            write_interface_attr(op, "predicate", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        CmpF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "comparison_predicate", w, ctx)?;
            write_inline_attr(op, "comparison_ordering", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        CmpI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "comparison_predicate", w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Constant => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "value", w, ctx)?;
        }
        DivI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        FToF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        FToI => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_bool_true(op, "saturating", 0);
            if flags != 0 {
                require_version(
                    "ftoi.saturating",
                    Feature::SaturatingFToI.since(),
                    ctx.version,
                )?;
            }
            if ctx.version >= BytecodeVersion::V13_4 {
                w.write_varint(flags);
            }
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        IToF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        MaxI | MinI | RemI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        MmaI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness_lhs", w, ctx)?;
            write_inline_attr(op, "signedness_rhs", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- No operands -----
        GetNumTileBlocks | GetTileBlockId | MakeToken => {
            write_result_types(op, w, ctx)?;
        }

        // ----- Simple: result types + operands (view construction) -----
        MakePartitionView => {
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- JoinTokens: variadic result count + operands with size -----
        JoinTokens => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- MakeTensorView: result count + AttrSizedOperandSegments -----
        // Args: base (ptr), dynamicShape (variadic), dynamicStrides (variadic)
        MakeTensorView => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            // ODS 0: base (1 operand, no size)
            write_operand_group(op, w, ctx, 0, 1)?;
            // ODS 1: dynamicShape (variadic, with size)
            write_variadic_operand_group(op, w, ctx, 1)?;
            // ODS 2: dynamicStrides (variadic, with size)
            write_variadic_operand_group(op, w, ctx, 2)?;
        }
        GetGlobal => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "name", w, ctx)?;
        }

        // ----- Extract: result count + variadic operands -----
        Extract | Insert => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- Flags + optional attributes -----
        AddF | DivF | MulF | SubF => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Fma => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Sqrt => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Exp2 | Rsqrt => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_operands(op, w, ctx, false)?;
        }
        MaxF | MinF => {
            write_result_types(op, w, ctx)?;
            let flags =
                flag_if_present(op, "propagate_nan", 0) | flag_if_present(op, "flush_to_zero", 1);
            w.write_varint(flags);
            write_operands(op, w, ctx, false)?;
        }

        // ----- Variadic results, no operands -----
        GetIndexSpaceShape | GetTensorShape => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- Global: result types + attrs (handled in global section, but also as op) -----
        Global => {
            write_result_types(op, w, ctx)?;
            if flag_if_bool_true(op, "constant", 0) != 0 {
                require_version(
                    "global.constant",
                    Feature::GlobalConstant.since(),
                    ctx.version,
                )?;
            }
            if find_op_attr(op, "symbol_visibility")
                .is_some_and(|attr| !matches!(attr, Attribute::Integer(0, _)))
            {
                require_version(
                    "global.symbol_visibility",
                    Feature::GlobalVisibility.since(),
                    ctx.version,
                )?;
            }
            if ctx.version >= super::enums::BytecodeVersion::V13_3 {
                let flags = flag_if_bool_true(op, "constant", 0);
                w.write_varint(flags);
            }
            write_inline_attr(op, "sym_name", w, ctx)?;
            write_inline_attr(op, "value", w, ctx)?;
            write_inline_attr(op, "alignment", w, ctx)?;
            if ctx.version >= super::enums::BytecodeVersion::V13_3 {
                write_inline_attr_or_default(op, "symbol_visibility", 0, w, ctx)?;
            }
        }

        // ----- Module: result types + attr + regions -----
        Module => {
            write_result_types(op, w, ctx)?;
            if find_op_attr(op, "producer").is_some() {
                require_version("module.producer", Feature::Producer.since(), ctx.version)?;
            }
            if ctx.version >= super::enums::BytecodeVersion::V13_3 {
                let flags = flag_if_present(op, "producer", 0);
                w.write_varint(flags);
            }
            write_inline_attr(op, "sym_name", w, ctx)?;
            if ctx.version >= super::enums::BytecodeVersion::V13_3
                && find_op_attr(op, "producer").is_some()
            {
                write_inline_attr(op, "producer", w, ctx)?;
            }
            write_regions(op, w, ctx)?;
        }

        // ----- Terminators: result count + variadic operands -----
        Break | Continue | Return | Yield => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- Print: result count + flags + `str` + sized `args` + optional token -----
        // Reference (`writePrintTkoOp`): the flags word has bit 0 set when the
        // optional `token` operand is present; the `args` group is written with
        // its size, and the token follows it unsized. The former writer left
        // the flag clear and folded the token into the sized group, so a
        // `print_tko` with an ordering token mis-parsed in every toolkit.
        Print => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let (num_args, has_token) = print_operand_groups(op, ctx);
            w.write_varint(if has_token { 1 } else { 0 });
            write_inline_attr(op, "str", w, ctx)?;
            w.write_varint(num_args as u64);
            for &operand in &op.operands[..num_args] {
                w.write_varint(operand_index(op, operand, ctx)?);
            }
            if has_token {
                w.write_varint(operand_index(op, op.operands[num_args], ctx)?);
            }
        }

        // ----- Variadic results + regions -----
        For => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            // v13.2: flags field for unsignedCmp (bit 0)
            let flags = flag_if_present(op, "unsigned_cmp", 0);
            w.write_varint(flags);
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        Loop => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        If => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
            write_regions(op, w, ctx)?;
        }

        // ----- Variadic results + required attributes + regions -----
        Reduce => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_inline_attr(op, "identities", w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        Scan => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_inline_attr(op, "reverse", w, ctx)?;
            write_inline_attr(op, "identities", w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }

        // ----- AttrSizedOperandSegments + flags -----
        AtomicCAS => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_operand_group(op, 3, 0) | flag_if_operand_group(op, 4, 1);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            write_inline_attr(op, "memory_scope", w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            write_operand_group(op, w, ctx, 2, 1)?;
            if flags & (1 << 0) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
            if flags & (1 << 1) != 0 {
                write_operand_group(op, w, ctx, 4, 1)?;
            }
        }
        AtomicRMW => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_operand_group(op, 2, 0) | flag_if_operand_group(op, 3, 1);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            write_inline_attr(op, "memory_scope", w, ctx)?;
            write_inline_attr(op, "mode", w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            if flags & (1 << 0) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 1) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        AtomicRedViewTko => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let flags = flag_if_operand_group(op, 3, 0);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            write_inline_attr(op, "memory_scope", w, ctx)?;
            write_inline_attr(op, "mode", w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_variadic_operand_group(op, w, ctx, 1)?;
            write_operand_group(op, w, ctx, 2, 1)?;
            if flags & (1 << 0) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        LoadPtrTko => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 1, 2)
                | flag_if_operand_group(op, 2, 3)
                | flag_if_operand_group(op, 3, 4);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 1, 1)?;
            }
            if flags & (1 << 3) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 4) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        LoadViewTko => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 2, 2);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_inbounds(op, 1, w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_variadic_operand_group(op, w, ctx, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
        }
        StorePtrTko => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 2, 2)
                | flag_if_operand_group(op, 3, 3);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 3) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        StoreViewTko => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 3, 2);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_inbounds(op, 2, w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            write_variadic_operand_group(op, w, ctx, 2)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }

        GdcLaunchDependentsTko | GdcWaitTko => {
            if op.operands.len() > 1 {
                return Err(Error::BytecodeWrite(format!(
                    "{} expects at most one token operand",
                    op.opcode.name()
                )));
            }
            write_result_types(op, w, ctx)?;
            w.write_varint(u64::from(!op.operands.is_empty()));
            write_operands(op, w, ctx, false)?;
        }

        // ----- Entry (function-like, with flags + optional attrs + regions) -----
        Entry => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "arg_attrs", 0)
                | flag_if_present(op, "res_attrs", 1)
                | flag_if_present(op, "optimization_hints", 2);
            w.write_varint(flags);
            write_inline_attr(op, "sym_name", w, ctx)?;
            write_inline_attr(op, "function_type", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "arg_attrs", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "res_attrs", w, ctx)?;
            }
            if flags & (1 << 2) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_regions(op, w, ctx)?;
        }

        // All opcodes covered. This arm catches future additions.
        #[allow(unreachable_patterns)]
        _ => {
            return Err(Error::BytecodeWrite(format!(
                "unsupported operation {:?} in bytecode writer",
                op.opcode
            )));
        }
    }
    Ok(())
}

// =========================================================================
// Helpers
// =========================================================================

fn write_result_types(op: &Operation, w: &mut EncodingWriter, ctx: &mut WriterCtx) -> Result<()> {
    for ty in &op.result_types {
        let idx = ctx.types.get_or_insert(ty);
        w.write_varint(idx);
    }
    Ok(())
}

fn write_operands(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    encode_size: bool,
) -> Result<()> {
    if encode_size {
        w.write_varint(op.operands.len() as u64);
    }
    for &operand in &op.operands {
        w.write_varint(operand_index(op, operand, ctx)?);
    }
    Ok(())
}

/// The bytecode value index of `operand`, which must already be defined.
fn operand_index(op: &Operation, operand: crate::ir::Value, ctx: &WriterCtx) -> Result<u64> {
    ctx.value_map.get(&operand).copied().ok_or_else(|| {
        Error::BytecodeWrite(format!(
            "operand {:?} not found in value map for op {:?}",
            operand, op.opcode
        ))
    })
}

/// Splits a `print_tko` op's operands into `(number of format args, has token)`.
///
/// Uses `operandSegmentSizes` when the builder recorded it (the compiler
/// always does); otherwise the token is recognized by its type — it is the
/// only operand a `print_tko` can have of type `token`, and it is always
/// last.
fn print_operand_groups(op: &Operation, ctx: &WriterCtx) -> (usize, bool) {
    if find_op_attr(op, "operandSegmentSizes").is_some() {
        let sizes = operand_segment_sizes(op);
        let num_args = sizes.first().copied().unwrap_or(0).max(0) as usize;
        let has_token = sizes.get(1).copied().unwrap_or(0) > 0;
        return (num_args.min(op.operands.len()), has_token);
    }
    let has_token = op
        .operands
        .last()
        .is_some_and(|&v| matches!(ctx.module.value_type(v), crate::ir::Type::Token));
    (op.operands.len() - usize::from(has_token), has_token)
}

fn write_operand_group(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    group_idx: usize,
    expected_size: usize,
) -> Result<()> {
    let offset = operand_group_offset(op, group_idx);
    for i in 0..expected_size {
        let operand = op.operands[offset + i];
        let idx = ctx.value_map.get(&operand).copied().ok_or_else(|| {
            Error::BytecodeWrite(format!(
                "operand group {group_idx}[{i}] not found in value map"
            ))
        })?;
        w.write_varint(idx);
    }
    Ok(())
}

fn write_variadic_operand_group(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    group_idx: usize,
) -> Result<()> {
    let sizes = operand_segment_sizes(op);
    let size = sizes.get(group_idx).copied().unwrap_or(0) as usize;
    let offset = operand_group_offset(op, group_idx);
    w.write_varint(size as u64);
    for i in 0..size {
        let operand = op.operands[offset + i];
        let idx = ctx.value_map.get(&operand).copied().ok_or_else(|| {
            Error::BytecodeWrite(format!(
                "variadic operand group {group_idx}[{i}] not found in value map"
            ))
        })?;
        w.write_varint(idx);
    }
    Ok(())
}

fn operand_group_offset(op: &Operation, group_idx: usize) -> usize {
    let sizes = operand_segment_sizes(op);
    sizes.iter().take(group_idx).map(|&s| s as usize).sum()
}

fn operand_segment_sizes(op: &Operation) -> Vec<i32> {
    for (name, attr) in &op.attributes {
        if name == "operandSegmentSizes" {
            if let Attribute::Array(arr) = attr {
                return arr
                    .iter()
                    .map(|a| match a {
                        Attribute::Integer(v, _) => *v as i32,
                        _ => 0,
                    })
                    .collect();
            }
        }
    }
    vec![1i32; op.operands.len()]
}

fn write_regions(op: &Operation, w: &mut EncodingWriter, ctx: &mut WriterCtx) -> Result<()> {
    let region_ids: Vec<_> = op.regions.clone();
    w.write_varint(region_ids.len() as u64);
    for region_id in region_ids {
        ctx.write_region(region_id, w)?;
    }
    Ok(())
}

fn write_inline_attr(
    op: &Operation,
    name: &str,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    let attr = find_op_attr(op, name).ok_or_else(|| {
        Error::BytecodeWrite(format!("missing attribute '{name}' on op {:?}", op.opcode))
    })?;
    write_attr_value_inline(attr, w, ctx)
}

fn write_inline_attr_or_default(
    op: &Operation,
    name: &str,
    default: u64,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    match find_op_attr(op, name) {
        Some(attr) => write_attr_value_inline(attr, w, ctx),
        None => {
            w.write_varint(default);
            Ok(())
        }
    }
}

fn write_interface_attr(
    op: &Operation,
    name: &str,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    let attr = find_op_attr(op, name).ok_or_else(|| {
        Error::BytecodeWrite(format!(
            "missing interface attribute '{name}' on op {:?}",
            op.opcode
        ))
    })?;
    super::writer::write_self_contained_attribute(
        attr,
        w,
        &mut ctx.strings,
        &mut ctx.types,
        &mut ctx.constants,
    )
}

fn write_attr_value_inline(
    attr: &Attribute,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    match attr {
        Attribute::Integer(v, _) => {
            w.write_varint(*v as u64);
        }
        Attribute::Float(v, ty) => {
            w.write_ap_float(*v, ty);
        }
        Attribute::Bool(v) => {
            w.write_byte(if *v { 0x01 } else { 0x00 });
        }
        Attribute::String(s) => {
            let idx = ctx.strings.get_or_insert(s);
            w.write_varint(idx);
        }
        Attribute::Type(ty) => {
            let idx = ctx.types.get_or_insert(ty);
            w.write_varint(idx);
        }
        Attribute::DenseI32Array(arr) => {
            w.write_le_var_size_i32(arr);
        }
        Attribute::DenseElements(de) => {
            let mut cdata = EncodingWriter::new();
            cdata.write_varint(de.data.len() as u64);
            cdata.write_bytes(&de.data);
            let idx = ctx.constants.add(cdata.into_bytes());
            w.write_varint(idx);
        }
        Attribute::Array(elems) => {
            // Inline Array: no tag prefix (unlike self-contained).
            // C++ writeSingleAttribute with isSelfContained=false skips the tag.
            w.write_varint(elems.len() as u64);
            for elem in elems {
                super::writer::write_self_contained_attribute(
                    elem,
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        Attribute::Dictionary(entries) => {
            // Inline Dictionary: no tag prefix.
            w.write_varint(entries.len() as u64);
            for (key, val) in entries {
                let key_idx = ctx.strings.get_or_insert(key);
                w.write_varint(key_idx);
                super::writer::write_self_contained_attribute(
                    val,
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        Attribute::OptimizationHints(oh) => {
            // Inline OptimizationHints: no OptimizationHints tag.
            // C++ writes: inner DictionaryAttr inline (no Dictionary tag either).
            // Format: size + entries where each entry value is self-contained.
            w.write_varint(oh.entries.len() as u64);
            for (arch, hints) in &oh.entries {
                let arch_idx = ctx.strings.get_or_insert(arch);
                w.write_varint(arch_idx);
                // Each arch value is a Dictionary, written self-contained
                // (with Dictionary tag) because it's an entry value.
                super::writer::write_self_contained_attribute(
                    &Attribute::Dictionary(
                        hints.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    ),
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        _ => {
            super::writer::write_self_contained_attribute(
                attr,
                w,
                &mut ctx.strings,
                &mut ctx.types,
                &mut ctx.constants,
            )?;
        }
    }
    Ok(())
}

/// 13.4 adds a dense byte-per-bool vector before view operands. An omitted
/// attribute means all false, preserving the pre-13.4 bounds semantics.
fn write_inbounds(
    op: &Operation,
    indices_group: usize,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
) -> Result<()> {
    let rank = operand_segment_sizes(op)
        .get(indices_group)
        .copied()
        .unwrap_or(0);
    let rank = usize::try_from(rank)
        .map_err(|_| Error::BytecodeWrite("negative view index operand count".into()))?;
    let values = match find_op_attr(op, "inbounds") {
        None => vec![false; rank],
        Some(Attribute::Array(attrs)) if attrs.len() == rank => attrs
            .iter()
            .map(|a| match a {
                Attribute::Bool(value) => Ok(*value),
                _ => Err(Error::BytecodeWrite(
                    "inbounds requires boolean elements".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(Error::BytecodeWrite(format!(
                "{}.inbounds must contain {rank} booleans (one per index dimension)",
                op.opcode.name()
            )))
        }
    };
    if values.iter().any(|&value| value) {
        require_version("inbounds=true", Feature::Inbounds.since(), ctx.version)?;
    }
    if ctx.version >= BytecodeVersion::V13_4 {
        w.write_varint(rank as u64);
        for value in values {
            w.write_byte(u8::from(value));
        }
    }
    Ok(())
}

fn find_op_attr<'a>(op: &'a Operation, name: &str) -> Option<&'a Attribute> {
    op.attributes
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

fn flag_if_present(op: &Operation, attr_name: &str, bit: u32) -> u64 {
    if find_op_attr(op, attr_name).is_some() {
        1u64 << bit
    } else {
        0
    }
}

fn flag_if_bool_true(op: &Operation, attr_name: &str, bit: u32) -> u64 {
    match find_op_attr(op, attr_name) {
        Some(Attribute::Bool(true)) => 1u64 << bit,
        Some(Attribute::Integer(v, _)) if *v != 0 => 1u64 << bit,
        _ => 0,
    }
}

fn flag_if_operand_group(op: &Operation, group_idx: usize, bit: u32) -> u64 {
    let sizes = operand_segment_sizes(op);
    let size = sizes.get(group_idx).copied().unwrap_or(0);
    if size > 0 {
        1u64 << bit
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::{
        ConstantManager, DebugInfoCollector, StringManager, TypeManager, WriterCtx,
    };
    use super::*;
    use crate::builder::{append_op, build_single_block_region, OpBuilder};
    use crate::bytecode::enums::BytecodeVersion;
    use crate::ir::{Location, Module, ScalarType, TileElementType, TileType, Type};

    fn tile_f32() -> Type {
        Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::F32),
        })
    }

    /// Serializes the body of the single op in `block` with a fresh context
    /// whose value map numbers the block arguments 0, 1, ...
    fn op_body_bytes(module: &Module, args: &[crate::ir::Value], op: crate::ir::OpId) -> Vec<u8> {
        op_body_bytes_version(module, args, op, BytecodeVersion::CURRENT).expect("write_op_body")
    }

    fn op_body_bytes_version(
        module: &Module,
        args: &[crate::ir::Value],
        op: crate::ir::OpId,
        version: BytecodeVersion,
    ) -> Result<Vec<u8>> {
        let mut ctx = WriterCtx {
            module,
            version,
            value_map: args
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i as u64))
                .collect(),
            next_idx: args.len() as u64,
            strings: StringManager::new(),
            types: TypeManager::new(),
            constants: ConstantManager::new(),
            debug: DebugInfoCollector::new(),
        };
        let mut w = EncodingWriter::new();
        write_op_body(module.op(op), &mut w, &mut ctx)?;
        Ok(w.into_bytes())
    }

    fn compare_python(operation: &str, version: BytecodeVersion, option: bool, actual: &[u8]) {
        let Some(source) = std::env::var_os("CUTILE_PYTHON_SOURCE") else {
            eprintln!("Python differential check skipped: set CUTILE_PYTHON_SOURCE to a reference checkout");
            return;
        };
        let output = std::process::Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/support/python_encoding.py"
            ))
            .arg(source)
            .arg(operation)
            .arg(version.to_string())
            .arg(option.to_string())
            .output()
            .expect("run Python reference encoder");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: String = actual.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            expected,
            "{operation} {version}"
        );
    }

    #[test]
    fn gdc_optional_token_wire_layout_and_version_gate() {
        for opcode in [Opcode::GdcLaunchDependentsTko, Opcode::GdcWaitTko] {
            for present in [false, true] {
                let mut module = Module::new("gdc");
                let (_, _, args) = build_single_block_region(&mut module, &[Type::Token]);
                let mut builder = OpBuilder::new(opcode, Location::Unknown).result(Type::Token);
                if present {
                    builder = builder.operand(args[0]);
                }
                let (op, _) = builder.build(&mut module);
                // result type, optional-operand flag, optional token value.
                assert_eq!(
                    op_body_bytes(&module, &args, op),
                    if present { vec![0, 1, 0] } else { vec![0, 0] }
                );
                compare_python(
                    &format!("{opcode:?}"),
                    BytecodeVersion::V13_4,
                    present,
                    &op_body_bytes(&module, &args, op),
                );
                for old in [BytecodeVersion::V13_2, BytecodeVersion::V13_3] {
                    assert!(op_body_bytes_version(&module, &args, op, old)
                        .unwrap_err()
                        .to_string()
                        .contains("13.4"));
                }
            }
        }
    }

    #[test]
    fn insert_fpowi_alias_fence_wire_layout_and_version_gate() {
        for (opcode, count, expected) in [
            (Opcode::Insert, 3, vec![1, 1, 3, 0, 1, 2]),
            (Opcode::FPowI, 2, vec![1, 0, 1]),
            (Opcode::MemoryFenceAliasTko, 1, vec![0, 0]),
        ] {
            let mut module = Module::new("new_ops");
            let result = if opcode == Opcode::MemoryFenceAliasTko {
                Type::Token
            } else {
                tile_f32()
            };
            let (_, _, args) = build_single_block_region(&mut module, &vec![result.clone(); count]);
            let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                .operands(args.iter().copied())
                .result(result)
                .build(&mut module);
            assert_eq!(op_body_bytes(&module, &args, op), expected, "{opcode:?}");
            compare_python(
                &format!("{opcode:?}"),
                BytecodeVersion::V13_4,
                false,
                &op_body_bytes(&module, &args, op),
            );
            assert!(op_body_bytes_version(&module, &args, op, BytecodeVersion::V13_3).is_err());
        }
    }

    #[test]
    fn ftoi_flags_are_added_only_in_13_4() {
        for saturating in [false, true] {
            let mut module = Module::new("ftoi");
            let (_, _, args) = build_single_block_region(&mut module, &[tile_f32()]);
            let result = Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Scalar(ScalarType::I32),
            });
            let (op, _) = OpBuilder::new(Opcode::FToI, Location::Unknown)
                .operand(args[0])
                .result(result)
                .attr("signedness", Attribute::i32(0))
                .attr("rounding_mode", Attribute::i32(6))
                .attr("saturating", Attribute::Bool(saturating))
                .build(&mut module);
            assert_eq!(
                op_body_bytes(&module, &args, op),
                vec![1, u8::from(saturating), 0, 6, 0]
            );
            compare_python(
                "FToI",
                BytecodeVersion::V13_4,
                saturating,
                &op_body_bytes(&module, &args, op),
            );
            for old in [BytecodeVersion::V13_2, BytecodeVersion::V13_3] {
                let bytes = op_body_bytes_version(&module, &args, op, old);
                if saturating {
                    assert!(bytes.is_err());
                } else {
                    let bytes = bytes.unwrap();
                    assert_eq!(bytes, vec![1, 0, 6, 0]);
                    compare_python("FToI", old, false, &bytes);
                }
            }
        }
    }

    /// `print_tko` with an ordering token: flags bit 0 set, the sized `args`
    #[test]
    fn view_inbounds_layout_matches_python_with_and_without_promises() {
        for opcode in [Opcode::LoadViewTko, Opcode::StoreViewTko] {
            for inbounds in [false, true] {
                let mut module = Module::new("view_flags");
                let count = if opcode == Opcode::LoadViewTko { 3 } else { 4 };
                let (_, _, args) =
                    build_single_block_region(&mut module, &vec![Type::Token; count]);
                let mut builder = OpBuilder::new(opcode, Location::Unknown)
                    .operands(args.iter().copied())
                    .attr("inbounds", Attribute::dense_bool_array([inbounds]))
                    .attr("memory_ordering_semantics", Attribute::i32(0))
                    .attr(
                        "operandSegmentSizes",
                        Attribute::Array(vec![Attribute::i32(1); count]),
                    );
                if opcode == Opcode::LoadViewTko {
                    builder = builder.result(tile_f32());
                }
                let (op, _) = builder.result(Type::Token).build(&mut module);
                for version in BytecodeVersion::SUPPORTED {
                    let bytes = op_body_bytes_version(&module, &args, op, version);
                    if inbounds && version < BytecodeVersion::V13_4 {
                        assert!(bytes.is_err());
                    } else {
                        compare_python(&format!("{opcode:?}"), version, inbounds, &bytes.unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn atan2_body_matches_python_on_all_versions() {
        let mut module = Module::new("atan2");
        let (_, _, args) = build_single_block_region(&mut module, &[tile_f32(), tile_f32()]);
        let (op, _) = OpBuilder::new(Opcode::Atan2, Location::Unknown)
            .operands(args.iter().copied())
            .result(tile_f32())
            .build(&mut module);
        for version in BytecodeVersion::SUPPORTED {
            compare_python(
                "Atan2",
                version,
                false,
                &op_body_bytes_version(&module, &args, op, version).unwrap(),
            );
        }
    }

    /// `print_tko` with an ordering token: flags bit 0 set, the sized `args`
    /// group, then the token unsized — the reference `writePrintTkoOp`
    /// layout. The former writer cleared the flag and folded the token into
    /// the sized group.
    #[test]
    fn print_with_token_sets_the_flag_and_writes_the_token_after_the_args() {
        let mut module = Module::new("print_token");
        let (_region, block, args) =
            build_single_block_region(&mut module, &[tile_f32(), Type::Token]);
        let (op, _) = OpBuilder::new(Opcode::Print, Location::Unknown)
            .attr("str", Attribute::String("v=%f\n".into()))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![Attribute::i32(1), Attribute::i32(1)]),
            )
            .operand(args[0])
            .operand(args[1])
            .result(Type::Token)
            .build(&mut module);
        append_op(&mut module, block, op);
        // [result count = 1][result type idx = 0][flags = 1][str idx = 0]
        // [args size = 1][arg idx = 0][token idx = 1]
        assert_eq!(
            op_body_bytes(&module, &args, op),
            vec![0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01]
        );
    }

    /// Without a token the flag stays clear and nothing follows the args.
    /// The token is recognized by type when `operandSegmentSizes` is absent.
    #[test]
    fn print_without_token_clears_the_flag() {
        for with_segment_sizes in [true, false] {
            let mut module = Module::new("print_no_token");
            let (_region, block, args) =
                build_single_block_region(&mut module, &[tile_f32(), tile_f32()]);
            let mut builder = OpBuilder::new(Opcode::Print, Location::Unknown)
                .attr("str", Attribute::String("a=%f b=%f\n".into()))
                .operand(args[0])
                .operand(args[1])
                .result(Type::Token);
            if with_segment_sizes {
                builder = builder.attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![Attribute::i32(2), Attribute::i32(0)]),
                );
            }
            let (op, _) = builder.build(&mut module);
            append_op(&mut module, block, op);
            // [1][type 0][flags 0][str 0][args size 2][0][1]
            assert_eq!(
                op_body_bytes(&module, &args, op),
                vec![0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01],
                "with_segment_sizes = {with_segment_sizes}"
            );
        }
    }

    /// A token operand is found by type when the builder recorded no
    /// segment sizes.
    #[test]
    fn print_token_is_inferred_from_its_type_without_segment_sizes() {
        let mut module = Module::new("print_inferred");
        let (_region, block, args) =
            build_single_block_region(&mut module, &[tile_f32(), Type::Token]);
        let (op, _) = OpBuilder::new(Opcode::Print, Location::Unknown)
            .attr("str", Attribute::String("v=%f\n".into()))
            .operand(args[0])
            .operand(args[1])
            .result(Type::Token)
            .build(&mut module);
        append_op(&mut module, block, op);
        assert_eq!(
            op_body_bytes(&module, &args, op),
            vec![0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01]
        );
    }
}
