/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    RegisterMma, RegisterMmaAccumulator, RegisterMmaAdapter, RegisterMmaCompatibilitySource,
    RegisterMmaElement, RegisterMmaLayout, RegisterMmaOperation, RegisterMmaOverflow,
    RegisterMmaParticipation, RegisterMmaShape,
};

pub(in crate::resolve) struct RegisterMmaRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) adapter: RegisterMmaAdapter,
    pub(in crate::resolve) compatibility_source: RegisterMmaCompatibilitySource,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: &'static str,
    pub(in crate::resolve) ptx_modifiers: Vec<&'static str>,
    pub(in crate::resolve) ptx_register_counts: [usize; 4],
}

pub(in crate::resolve) fn integer_register_mma_recipe(
    mma: &RegisterMma,
    common: bool,
) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaCompatibilitySource::{ExistingStub, GeneratedStub};
    use RegisterMmaElement::{S4, S8, U4, U8};
    use RegisterMmaOverflow::{Satfinite, Wrapping};
    use RegisterMmaShape::{M8n8k16, M8n8k32, M16n8k16, M16n8k32, M16n8k64};

    if !common
        || mma.operation != RegisterMmaOperation::Multiply
        || mma.accumulator != RegisterMmaAccumulator::S32
    {
        return None;
    }

    let (id, abi_id, operation_key, source_record, llvm_symbol, compatibility_source) =
        match (mma.shape, mma.a_element, mma.b_element, mma.overflow) {
            (M16n8k32, S8, S8, Wrapping) => (
                "mma_m16n8k32_s32_s8",
                "i0108",
                "matrix.mma.m16n8k32.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                ExistingStub,
            ),
            (M16n8k16, S8, S8, Wrapping) => (
                "mma_m16n8k16_s32_s8",
                "i0110",
                "matrix.mma.m16n8k16.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, U8, Wrapping) => (
                "mma_m16n8k16_s32_s8_u8",
                "i0111",
                "matrix.mma.m16n8k16.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_s8_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.s8.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, U8, Wrapping) => (
                "mma_m16n8k16_s32_u8",
                "i0112",
                "matrix.mma.m16n8k16.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, S8, Wrapping) => (
                "mma_m16n8k16_s32_u8_s8",
                "i0113",
                "matrix.mma.m16n8k16.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_u8_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.u8.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, U8, Wrapping) => (
                "mma_m16n8k32_s32_s8_u8",
                "i0114",
                "matrix.mma.m16n8k32.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s8_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.s8.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, U8, Wrapping) => (
                "mma_m16n8k32_s32_u8",
                "i0115",
                "matrix.mma.m16n8k32.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, S8, Wrapping) => (
                "mma_m16n8k32_s32_u8_s8",
                "i0116",
                "matrix.mma.m16n8k32.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u8_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.u8.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, S8, Satfinite) => (
                "mma_m16n8k16_s32_s8_satfinite",
                "i0117",
                "matrix.mma.m16n8k16.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, U8, Satfinite) => (
                "mma_m16n8k16_s32_s8_u8_satfinite",
                "i0118",
                "matrix.mma.m16n8k16.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, U8, Satfinite) => (
                "mma_m16n8k16_s32_u8_satfinite",
                "i0119",
                "matrix.mma.m16n8k16.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, S8, Satfinite) => (
                "mma_m16n8k16_s32_u8_s8_satfinite",
                "i0120",
                "matrix.mma.m16n8k16.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, S8, Satfinite) => (
                "mma_m16n8k32_s32_s8_satfinite",
                "i0121",
                "matrix.mma.m16n8k32.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, U8, Satfinite) => (
                "mma_m16n8k32_s32_s8_u8_satfinite",
                "i0122",
                "matrix.mma.m16n8k32.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, U8, Satfinite) => (
                "mma_m16n8k32_s32_u8_satfinite",
                "i0123",
                "matrix.mma.m16n8k32.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, S8, Satfinite) => (
                "mma_m16n8k32_s32_u8_s8_satfinite",
                "i0124",
                "matrix.mma.m16n8k32.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, S8, Wrapping) => (
                "mma_m8n8k16_s32_s8",
                "i0125",
                "matrix.mma.m8n8k16.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, U8, Wrapping) => (
                "mma_m8n8k16_s32_s8_u8",
                "i0126",
                "matrix.mma.m8n8k16.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_s8_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.s8.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, U8, Wrapping) => (
                "mma_m8n8k16_s32_u8",
                "i0127",
                "matrix.mma.m8n8k16.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, S8, Wrapping) => (
                "mma_m8n8k16_s32_u8_s8",
                "i0128",
                "matrix.mma.m8n8k16.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_u8_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.u8.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, S8, Satfinite) => (
                "mma_m8n8k16_s32_s8_satfinite",
                "i0129",
                "matrix.mma.m8n8k16.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, U8, Satfinite) => (
                "mma_m8n8k16_s32_s8_u8_satfinite",
                "i0130",
                "matrix.mma.m8n8k16.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, U8, Satfinite) => (
                "mma_m8n8k16_s32_u8_satfinite",
                "i0131",
                "matrix.mma.m8n8k16.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, S8, Satfinite) => (
                "mma_m8n8k16_s32_u8_s8_satfinite",
                "i0132",
                "matrix.mma.m8n8k16.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M8n8k32, S4, S4, Wrapping) => (
                "mma_m8n8k32_s32_s4",
                "i0133",
                "matrix.mma.m8n8k32.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, U4, Wrapping) => (
                "mma_m8n8k32_s32_s4_u4",
                "i0134",
                "matrix.mma.m8n8k32.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_s4_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.s4.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, U4, Wrapping) => (
                "mma_m8n8k32_s32_u4",
                "i0135",
                "matrix.mma.m8n8k32.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, S4, Wrapping) => (
                "mma_m8n8k32_s32_u4_s4",
                "i0136",
                "matrix.mma.m8n8k32.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_u4_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.u4.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, S4, Satfinite) => (
                "mma_m8n8k32_s32_s4_satfinite",
                "i0137",
                "matrix.mma.m8n8k32.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, U4, Satfinite) => (
                "mma_m8n8k32_s32_s4_u4_satfinite",
                "i0138",
                "matrix.mma.m8n8k32.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, U4, Satfinite) => (
                "mma_m8n8k32_s32_u4_satfinite",
                "i0139",
                "matrix.mma.m8n8k32.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, S4, Satfinite) => (
                "mma_m8n8k32_s32_u4_s4_satfinite",
                "i0140",
                "matrix.mma.m8n8k32.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, S4, Wrapping) => (
                "mma_m16n8k32_s32_s4",
                "i0141",
                "matrix.mma.m16n8k32.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, U4, Wrapping) => (
                "mma_m16n8k32_s32_s4_u4",
                "i0142",
                "matrix.mma.m16n8k32.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s4_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.s4.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, U4, Wrapping) => (
                "mma_m16n8k32_s32_u4",
                "i0143",
                "matrix.mma.m16n8k32.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, S4, Wrapping) => (
                "mma_m16n8k32_s32_u4_s4",
                "i0144",
                "matrix.mma.m16n8k32.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u4_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.u4.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, S4, Wrapping) => (
                "mma_m16n8k64_s32_s4",
                "i0145",
                "matrix.mma.m16n8k64.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, U4, Wrapping) => (
                "mma_m16n8k64_s32_s4_u4",
                "i0146",
                "matrix.mma.m16n8k64.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_s4_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.s4.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, U4, Wrapping) => (
                "mma_m16n8k64_s32_u4",
                "i0147",
                "matrix.mma.m16n8k64.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, S4, Wrapping) => (
                "mma_m16n8k64_s32_u4_s4",
                "i0148",
                "matrix.mma.m16n8k64.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_u4_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.u4.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, S4, Satfinite) => (
                "mma_m16n8k32_s32_s4_satfinite",
                "i0149",
                "matrix.mma.m16n8k32.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, U4, Satfinite) => (
                "mma_m16n8k32_s32_s4_u4_satfinite",
                "i0150",
                "matrix.mma.m16n8k32.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, U4, Satfinite) => (
                "mma_m16n8k32_s32_u4_satfinite",
                "i0151",
                "matrix.mma.m16n8k32.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, S4, Satfinite) => (
                "mma_m16n8k32_s32_u4_s4_satfinite",
                "i0152",
                "matrix.mma.m16n8k32.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, S4, Satfinite) => (
                "mma_m16n8k64_s32_s4_satfinite",
                "i0153",
                "matrix.mma.m16n8k64.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, U4, Satfinite) => (
                "mma_m16n8k64_s32_s4_u4_satfinite",
                "i0154",
                "matrix.mma.m16n8k64.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, U4, Satfinite) => (
                "mma_m16n8k64_s32_u4_satfinite",
                "i0155",
                "matrix.mma.m16n8k64.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, S4, Satfinite) => (
                "mma_m16n8k64_s32_u4_s4_satfinite",
                "i0156",
                "matrix.mma.m16n8k64.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            _ => return None,
        };

    let int4 = matches!(mma.a_element, S4 | U4);
    let (rust_arguments, dialect_operands, llvm_arguments, adapter, shape, register_counts) =
        match (mma.shape, int4) {
            (M8n8k16, false) | (M8n8k32, true) => (
                &["[i32; 2]", "u32", "u32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32"] as &'static [&'static str],
                C2I32A1U32B1U32ToD2I32,
                match mma.shape {
                    M8n8k16 => "m8n8k16",
                    M8n8k32 => "m8n8k32",
                    _ => unreachable!(),
                },
                [2, 1, 1, 2],
            ),
            (M16n8k16, false) | (M16n8k32, true) => (
                &["[i32; 4]", "[u32; 2]", "u32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
                C4I32A2U32B1U32ToD4I32,
                match mma.shape {
                    M16n8k16 => "m16n8k16",
                    M16n8k32 => "m16n8k32",
                    _ => unreachable!(),
                },
                [4, 2, 1, 4],
            ),
            (M16n8k32, false) | (M16n8k64, true) => (
                &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &'static [&'static str],
                &[
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
                ] as &'static [&'static str],
                &[
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
                ] as &'static [&'static str],
                C4I32A4U32B2U32ToD4I32,
                match mma.shape {
                    M16n8k32 => "m16n8k32",
                    M16n8k64 => "m16n8k64",
                    _ => unreachable!(),
                },
                [4, 4, 2, 4],
            ),
            _ => return None,
        };
    if mma.adapter != adapter {
        return None;
    }

    let (rust_result, dialect_results, llvm_results, minimum_ptx, minimum_sm) = match mma.shape {
        M8n8k16 | M8n8k32 => (
            "[i32; 2]",
            &["i32", "i32"] as &'static [&'static str],
            &["i32", "i32"] as &'static [&'static str],
            "6.5",
            "sm_75",
        ),
        M16n8k16 | M16n8k32 | M16n8k64 => (
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            "7.0",
            "sm_80",
        ),
        _ => return None,
    };

    let element = |element| match element {
        S4 => Some("s4"),
        U4 => Some("u4"),
        S8 => Some("s8"),
        U8 => Some("u8"),
        _ => None,
    };
    let mut ptx_modifiers = vec!["sync", "aligned", shape, "row", "col"];
    if mma.overflow == Satfinite {
        ptx_modifiers.push("satfinite");
    }
    ptx_modifiers.extend([
        "s32",
        element(mma.a_element)?,
        element(mma.b_element)?,
        "s32",
    ]);

    Some(RegisterMmaRecipe {
        id,
        abi_id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_arguments,
        rust_result,
        dialect_op_type: "RegisterMmaOp",
        dialect_op_name: "nvvm.register_mma",
        dialect_operands,
        dialect_results,
        llvm_arguments,
        llvm_results,
        adapter,
        compatibility_source,
        minimum_ptx,
        minimum_sm,
        ptx_modifiers,
        ptx_register_counts: register_counts,
    })
}

pub(in crate::resolve) fn binary_register_mma_recipe(
    mma: &RegisterMma,
    common: bool,
) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaOperation::{AndPopc, XorPopc};
    use RegisterMmaShape::{M8n8k128, M16n8k128, M16n8k256};

    if !common
        || mma.accumulator != RegisterMmaAccumulator::S32
        || mma.a_element != RegisterMmaElement::B1
        || mma.b_element != RegisterMmaElement::B1
        || mma.overflow != RegisterMmaOverflow::Wrapping
        || mma.compatibility_source != RegisterMmaCompatibilitySource::GeneratedStub
    {
        return None;
    }

    let (id, abi_id, operation_key, source_record, llvm_symbol, operation_name) =
        match (mma.shape, mma.operation) {
            (M8n8k128, XorPopc) => (
                "mma_m8n8k128_s32_b1_xor_popc",
                "i0157",
                "matrix.mma.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m8n8k128_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m8n8k128.row.col.b1",
                "xor",
            ),
            (M16n8k128, XorPopc) => (
                "mma_m16n8k128_s32_b1_xor_popc",
                "i0158",
                "matrix.mma.m16n8k128.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m16n8k128_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1",
                "xor",
            ),
            (M16n8k256, XorPopc) => (
                "mma_m16n8k256_s32_b1_xor_popc",
                "i0159",
                "matrix.mma.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m16n8k256_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m16n8k256.row.col.b1",
                "xor",
            ),
            (M8n8k128, AndPopc) => (
                "mma_m8n8k128_s32_b1_and_popc",
                "i0160",
                "matrix.mma.m8n8k128.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m8n8k128_row_col_b1",
                "llvm.nvvm.mma.and.popc.m8n8k128.row.col.b1",
                "and",
            ),
            (M16n8k128, AndPopc) => (
                "mma_m16n8k128_s32_b1_and_popc",
                "i0161",
                "matrix.mma.m16n8k128.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m16n8k128_row_col_b1",
                "llvm.nvvm.mma.and.popc.m16n8k128.row.col.b1",
                "and",
            ),
            (M16n8k256, AndPopc) => (
                "mma_m16n8k256_s32_b1_and_popc",
                "i0162",
                "matrix.mma.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m16n8k256_row_col_b1",
                "llvm.nvvm.mma.and.popc.m16n8k256.row.col.b1",
                "and",
            ),
            _ => return None,
        };

    let (
        rust_arguments,
        dialect_operands,
        llvm_arguments,
        rust_result,
        result_types,
        adapter,
        counts,
    ) = match mma.shape {
        M8n8k128 => (
            &["[i32; 2]", "u32", "u32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            "[i32; 2]",
            &["i32", "i32"] as &'static [&'static str],
            C2I32A1U32B1U32ToD2I32,
            [2, 1, 1, 2],
        ),
        M16n8k128 => (
            &["[i32; 4]", "[u32; 2]", "u32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            C4I32A2U32B1U32ToD4I32,
            [4, 2, 1, 4],
        ),
        M16n8k256 => (
            &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &'static [&'static str],
            &[
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            ] as &'static [&'static str],
            &[
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            ] as &'static [&'static str],
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            C4I32A4U32B2U32ToD4I32,
            [4, 4, 2, 4],
        ),
        _ => return None,
    };
    if mma.adapter != adapter {
        return None;
    }

    let (minimum_ptx, minimum_sm) = match (mma.operation, mma.shape) {
        (XorPopc, M8n8k128) => ("7.0", "sm_75"),
        (XorPopc, M16n8k128 | M16n8k256) => ("7.0", "sm_80"),
        (AndPopc, M8n8k128 | M16n8k128 | M16n8k256) => ("7.1", "sm_80"),
        _ => return None,
    };
    let shape = match mma.shape {
        M8n8k128 => "m8n8k128",
        M16n8k128 => "m16n8k128",
        M16n8k256 => "m16n8k256",
        _ => return None,
    };

    Some(RegisterMmaRecipe {
        id,
        abi_id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_arguments,
        rust_result,
        dialect_op_type: "RegisterMmaOp",
        dialect_op_name: "nvvm.register_mma",
        dialect_operands,
        dialect_results: result_types,
        llvm_arguments,
        llvm_results: result_types,
        adapter,
        compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
        minimum_ptx,
        minimum_sm,
        ptx_modifiers: vec![
            "sync",
            "aligned",
            shape,
            "row",
            "col",
            "s32",
            "b1",
            "b1",
            "s32",
            operation_name,
            "popc",
        ],
        ptx_register_counts: counts,
    })
}

pub(in crate::resolve) fn register_mma_recipe(mma: &RegisterMma) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAccumulator::{F16 as F16Accumulator, F32, F64};
    use RegisterMmaAdapter::{
        C2F64A1F64B1F64ToD2F64, C2U32A2U32B1U32ToD2U32, C2U32A4U32B2U32ToD2U32,
        C4F32A2U32B1U32ToD4F32, C4F32A4U32B2U32ToD4F32,
    };
    use RegisterMmaElement::{Bf16, F16 as F16Element, F64 as F64Element, Tf32};
    use RegisterMmaShape::{M8n8k4, M16n8k4, M16n8k8, M16n8k16};

    let common = mma.a_layout == RegisterMmaLayout::Row
        && mma.b_layout == RegisterMmaLayout::Col
        && mma.participation
            == RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    if let Some(recipe) = integer_register_mma_recipe(mma, common) {
        return Some(recipe);
    }
    if let Some(recipe) = binary_register_mma_recipe(mma, common) {
        return Some(recipe);
    }
    if mma.operation != RegisterMmaOperation::Multiply {
        return None;
    }
    match (
        mma.shape,
        mma.accumulator,
        mma.a_element,
        mma.b_element,
        mma.overflow,
        mma.adapter,
        common,
    ) {
        (
            M16n8k4,
            F32,
            Tf32,
            Tf32,
            RegisterMmaOverflow::NotApplicable,
            C4F32A2U32B1U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k4_f32_tf32",
            abi_id: "i0520",
            operation_key: "matrix.mma.m16n8k4.row.col.f32.tf32.tf32.f32",
            source_record: "int_nvvm_mma_m16n8k4_row_col_tf32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k4.row.col.tf32",
            rust_arguments: &["[f32; 4]", "[u32; 2]", "u32"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["f32", "f32", "f32", "f32", "i32", "i32", "i32"],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &["i32", "i32", "i32", "f32", "f32", "f32", "f32"],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A2U32B1U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k4", "row", "col", "f32", "tf32", "tf32", "f32",
            ],
            ptx_register_counts: [4, 2, 1, 4],
        }),
        (
            M16n8k8,
            F16Accumulator,
            F16Element,
            F16Element,
            RegisterMmaOverflow::NotApplicable,
            C2U32A2U32B1U32ToD2U32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k8_f16_f16",
            abi_id: "i0521",
            operation_key: "matrix.mma.m16n8k8.row.col.f16.f16.f16.f16",
            source_record: "int_nvvm_mma_m16n8k8_row_col_f16_f16",
            llvm_symbol: "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
            rust_arguments: &["[u32; 2]", "[u32; 2]", "u32"],
            rust_result: "[u32; 2]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["i32", "i32", "i32", "i32", "i32"],
            dialect_results: &["i32", "i32"],
            llvm_arguments: &["v2f16", "v2f16", "v2f16", "v2f16", "v2f16"],
            llvm_results: &["v2f16", "v2f16"],
            adapter: C2U32A2U32B1U32ToD2U32,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k8", "row", "col", "f16", "f16", "f16", "f16",
            ],
            ptx_register_counts: [2, 2, 1, 2],
        }),
        (
            M16n8k8,
            F32,
            Bf16,
            Bf16,
            RegisterMmaOverflow::NotApplicable,
            C4F32A2U32B1U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k8_f32_bf16",
            abi_id: "i0522",
            operation_key: "matrix.mma.m16n8k8.row.col.f32.bf16.bf16.f32",
            source_record: "int_nvvm_mma_m16n8k8_row_col_bf16",
            llvm_symbol: "llvm.nvvm.mma.m16n8k8.row.col.bf16",
            rust_arguments: &["[f32; 4]", "[u32; 2]", "u32"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["f32", "f32", "f32", "f32", "i32", "i32", "i32"],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &["i32", "i32", "i32", "f32", "f32", "f32", "f32"],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A2U32B1U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k8", "row", "col", "f32", "bf16", "bf16", "f32",
            ],
            ptx_register_counts: [4, 2, 1, 4],
        }),
        (
            M16n8k8,
            F32,
            F16Element,
            F16Element,
            RegisterMmaOverflow::NotApplicable,
            C4F32A2U32B1U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k8_f32_f16",
            abi_id: "i0523",
            operation_key: "matrix.mma.m16n8k8.row.col.f32.f16.f16.f32",
            source_record: "int_nvvm_mma_m16n8k8_row_col_f32_f32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            rust_arguments: &["[f32; 4]", "[u32; 2]", "u32"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["f32", "f32", "f32", "f32", "i32", "i32", "i32"],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &["v2f16", "v2f16", "v2f16", "f32", "f32", "f32", "f32"],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A2U32B1U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k8", "row", "col", "f32", "f16", "f16", "f32",
            ],
            ptx_register_counts: [4, 2, 1, 4],
        }),
        (
            M16n8k16,
            F16Accumulator,
            F16Element,
            F16Element,
            RegisterMmaOverflow::NotApplicable,
            C2U32A4U32B2U32ToD2U32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k16_f16_f16",
            abi_id: "i0524",
            operation_key: "matrix.mma.m16n8k16.row.col.f16.f16.f16.f16",
            source_record: "int_nvvm_mma_m16n8k16_row_col_f16_f16",
            llvm_symbol: "llvm.nvvm.mma.m16n8k16.row.col.f16.f16",
            rust_arguments: &["[u32; 2]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[u32; 2]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32"],
            dialect_results: &["i32", "i32"],
            llvm_arguments: &[
                "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16",
            ],
            llvm_results: &["v2f16", "v2f16"],
            adapter: C2U32A4U32B2U32ToD2U32,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k16", "row", "col", "f16", "f16", "f16", "f16",
            ],
            ptx_register_counts: [2, 4, 2, 2],
        }),
        (
            M16n8k16,
            F32,
            Bf16,
            Bf16,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k16_f32_bf16",
            abi_id: "i0105",
            operation_key: "matrix.mma.m16n8k16.row.col.f32.bf16.bf16.f32",
            source_record: "int_nvvm_mma_m16n8k16_row_col_bf16",
            llvm_symbol: "llvm.nvvm.mma.m16n8k16.row.col.bf16",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k16", "row", "col", "f32", "bf16", "bf16", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M16n8k16,
            F32,
            F16Element,
            F16Element,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k16_f32_f16",
            abi_id: "i0106",
            operation_key: "matrix.mma.m16n8k16.row.col.f32.f16.f16.f32",
            source_record: "int_nvvm_mma_m16n8k16_row_col_f32_f32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k16", "row", "col", "f32", "f16", "f16", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M16n8k8,
            F32,
            Tf32,
            Tf32,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k8_f32_tf32",
            abi_id: "i0107",
            operation_key: "matrix.mma.m16n8k8.row.col.f32.tf32.tf32.f32",
            source_record: "int_nvvm_mma_m16n8k8_row_col_tf32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k8", "row", "col", "f32", "tf32", "tf32", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M8n8k4,
            F64,
            F64Element,
            F64Element,
            RegisterMmaOverflow::NotApplicable,
            C2F64A1F64B1F64ToD2F64,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m8n8k4_f64",
            abi_id: "i0109",
            operation_key: "matrix.mma.m8n8k4.row.col.f64.f64.f64.f64",
            source_record: "int_nvvm_mma_m8n8k4_row_col_f64",
            llvm_symbol: "llvm.nvvm.mma.m8n8k4.row.col.f64",
            rust_arguments: &["[f64; 2]", "f64", "f64"],
            rust_result: "[f64; 2]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["f64", "f64", "f64", "f64"],
            dialect_results: &["f64", "f64"],
            llvm_arguments: &["f64", "f64", "f64", "f64"],
            llvm_results: &["f64", "f64"],
            adapter: C2F64A1F64B1F64ToD2F64,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m8n8k4", "row", "col", "f64", "f64", "f64", "f64",
            ],
            ptx_register_counts: [2, 1, 1, 2],
        }),
        _ => None,
    }
}
