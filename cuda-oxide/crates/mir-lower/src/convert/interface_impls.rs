/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `MirToLlvmConversion` implementations for all MIR and NVVM ops.
//!
//! Each impl delegates directly to a per-op conversion function,
//! bypassing the old category sub-enum dispatch.

use pliron::{
    builtin::ops::ConstantOp,
    context::Context,
    derive::op_interface_impl,
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        rewriter::Rewriter,
    },
    op::Op,
    result::Result,
};

use llvm_export::attributes::{FCmpPredicateAttr, ICmpPredicateAttr};

use crate::conversion_interface::MirToLlvmConversion;

use dialect_mir::ops::{
    MirAddOp, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirBitAndOp, MirBitOrOp,
    MirBitXorOp, MirCallOp, MirCastOp, MirCheckedAddOp, MirCheckedMulOp, MirCheckedSubOp, MirCmpOp,
    MirCondBranchOp, MirConstantOp, MirConstructArrayOp, MirConstructDisjointSliceOp,
    MirConstructEnumOp, MirConstructSliceOp, MirConstructStructOp, MirConstructTupleOp,
    MirDbgValueListOp, MirDbgValueOp, MirDivOp, MirEnumPayloadOp, MirEqOp,
    MirExtractArrayElementOp, MirExtractFieldOp, MirFieldAddrOp, MirFloatConstantOp, MirGeOp,
    MirGetDiscriminantOp, MirGotoOp, MirGtOp, MirInsertFieldOp, MirLeOp, MirLoadOp, MirLtOp,
    MirMemcpyOp, MirMemmoveOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRefOp,
    MirRemOp, MirReturnOp, MirSetDiscriminantOp, MirShlOp, MirShrOp, MirStorageDeadOp,
    MirStorageLiveOp, MirStoreOp, MirSubOp, MirUndefOp, MirUnreachableOp, MirUnrollHintOp,
};
use dialect_nvvm::ops::{
    AssertFailOp, CvtaGenericToSharedOffsetOp, InlinePtxOp, NvvmAtomicCmpxchgOp, NvvmAtomicFenceOp,
    NvvmAtomicLoadOp, NvvmAtomicRmwOp, NvvmAtomicStoreOp, ReadPtxSregClusterIdxOp,
    ReadPtxSregNclusterIdOp, VprintfOp, WgmmaMakeSmemDescOp, WgmmaMmaGroupM64N64K16F32Bf16Op,
    WgmmaMmaGroupValuesM64N64K8F32Tf32Op, WgmmaMmaGroupValuesM64N64K16F32Bf16Op,
    WgmmaMmaGroupValuesM64N64K16F32F16Op, WgmmaMmaGroupValuesM64N128K16F32Bf16Op,
    WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op, WgmmaMmaLoopValuesM64N64K16F32Bf16Op,
    WgmmaMmaLoopValuesM64N64K16F32F16Op, WgmmaMmaM64N64K8F32Tf32Op, WgmmaMmaM64N64K16F32Bf16Op,
    WgmmaMmaM64N64K16F32F16Op, WgmmaMmaM64N128K16F32Bf16Op,
    WgmmaMmaPipelineValuesM64N64K16F32Bf16Op,
};

// ---- Arithmetic ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_add(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_add(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirSubOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_sub(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedSubOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_sub(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirMulOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_mul(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedMulOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_mul(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirDivOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_div(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirRemOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_rem(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirShrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_shr(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirShlOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_shl(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitAndOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitand(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitOrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitor(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitXorOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitxor(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNotOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_not(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNegOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_neg(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SLT,
            ICmpPredicateAttr::ULT,
            FCmpPredicateAttr::OLT,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SLE,
            ICmpPredicateAttr::ULE,
            FCmpPredicateAttr::OLE,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SGT,
            ICmpPredicateAttr::UGT,
            FCmpPredicateAttr::OGT,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SGE,
            ICmpPredicateAttr::UGE,
            FCmpPredicateAttr::OGE,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCmpOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_three_way_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirEqOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::EQ,
            ICmpPredicateAttr::EQ,
            FCmpPredicateAttr::OEQ,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::NE,
            ICmpPredicateAttr::NE,
            FCmpPredicateAttr::UNE,
        )
    }
}

// ---- Memory ops ------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirAllocaOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_alloca(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirStoreOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_store(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirMemcpyOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_memcpy(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirMemmoveOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_memmove(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLoadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_load(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirDbgValueOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_dbg_value(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirDbgValueListOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_dbg_value_list(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirRefOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_ref(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirPtrOffsetOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_ptr_offset(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Constant ops ----------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_integer(ctx, rewriter, self.get_operation(), operands_info)
    }
}

/// A `builtin.constant` that `sccp` materialised carries a signed/unsigned MIR
/// integer type; normalise it to a signless constant, exactly like `mir.constant`.
/// Only a non-signless `builtin.constant` reaches here (see `can_convert_op`), so
/// the emitted signless constant is final and the conversion converges.
#[op_interface_impl]
impl MirToLlvmConversion for ConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = pliron::dyn_clone::clone_box(&*self.get_value(ctx));
        super::ops::constants::convert_builtin_constant(ctx, rewriter, self.get_operation(), value)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirFloatConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_float(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirUndefOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_undef(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Cast op ---------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirCastOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::cast::convert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Aggregate ops ---------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirExtractFieldOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_extract_field(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirInsertFieldOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_insert_field(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructStructOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_struct(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructTupleOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_tuple(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructSliceOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_slice(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructDisjointSliceOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_disjoint_slice(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructArrayOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_array(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirExtractArrayElementOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_extract_array_element(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructEnumOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_enum(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGetDiscriminantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_get_discriminant(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirEnumPayloadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_enum_payload(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirSetDiscriminantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_set_discriminant(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirFieldAddrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_field_addr(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirArrayElementAddrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_array_element_addr(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- Control flow ops ------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirReturnOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_return(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGotoOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_goto(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCondBranchOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_cond_branch(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirAssertOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_assert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirUnreachableOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_unreachable(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- Call op ---------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirCallOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::call::convert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- No-op markers (StorageLive / StorageDead) -----------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirStorageLiveOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirStorageDeadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// Safety net: the loop-unroll pass consumes every `mir.unroll_hint` before
// lowering, so one should never reach here. But if unrolling is skipped (e.g.
// a debug build that bypasses the pass), drop the hint rather than fail the
// conversion: it carries no runtime semantics, only a request to unroll.
#[op_interface_impl]
impl MirToLlvmConversion for MirUnrollHintOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// ---- NVVM Basic ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for InlinePtxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::asm::convert_inline_ptx(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for AssertFailOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_assertfail(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for VprintfOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_vprintf(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterIdxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_idx(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNclusterIdOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_num_clusters(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM memory ops -------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for CvtaGenericToSharedOffsetOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::memory::convert_cvta_generic_to_shared_offset(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM WGMMA ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMakeSmemDescOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_make_smem_desc(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaM64N64K16F32F16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaM64N128K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaM64N64K8F32Tf32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaGroupM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_group(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaGroupValuesM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_group_values(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaGroupValuesM64N64K16F32F16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_group_values_f16(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaGroupValuesM64N128K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_group_values_m64n128_bf16(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaGroupValuesM64N64K8F32Tf32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_group_values_tf32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaLoopValuesM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_loop_values(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaLoopValuesM64N64K16F32F16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_loop_values_f16(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_loop_pipeline_values(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaPipelineValuesM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma_pipeline_values(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM WMMA ops ---------------------------------------------------------

// ---- NVVM Atomic ops -------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicLoadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_load(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicStoreOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_store(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicFenceOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_fence(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicRmwOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_rmw(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicCmpxchgOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_cmpxchg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}
