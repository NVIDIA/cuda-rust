/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ImportedIntrinsic, OverlayIntrinsic, RuntimeValidation, Tcgen05, Tcgen05Adapter,
    Tcgen05Admission, Tcgen05Cp, Tcgen05CpAdmissionVariant, Tcgen05CpGroup, Tcgen05CpMember,
    Tcgen05Ld, Tcgen05LdAdmissionVariant, Tcgen05LdMultiplicity, Tcgen05LdShape, Tcgen05Operation,
    Tcgen05SourceContract, Tcgen05St, Tcgen05StAdmissionVariant,
};
use crate::ptx::OperandPattern;
use anyhow::{Result, ensure};

use super::*;
use crate::resolve::guards::*;

pub(in crate::resolve) struct Tcgen05CpMemberRecipe {
    pub(in crate::resolve) llvm_suffix: &'static str,
    pub(in crate::resolve) ptx_suffix: &'static str,
    pub(in crate::resolve) op_suffix: &'static str,
    pub(in crate::resolve) selection_stem: &'static str,
}

pub(in crate::resolve) const TCGEN05_CP_MEMBERS: [Tcgen05CpMember; 17] = [
    Tcgen05CpMember::M128x128bB4x16P64,
    Tcgen05CpMember::M128x128bB6x16P32,
    Tcgen05CpMember::M128x128b,
    Tcgen05CpMember::M128x256bB4x16P64,
    Tcgen05CpMember::M128x256bB6x16P32,
    Tcgen05CpMember::M32x128bWarpx4B4x16P64,
    Tcgen05CpMember::M32x128bWarpx4B6x16P32,
    Tcgen05CpMember::M32x128bWarpx4,
    Tcgen05CpMember::M4x256bB4x16P64,
    Tcgen05CpMember::M4x256bB6x16P32,
    Tcgen05CpMember::M4x256b,
    Tcgen05CpMember::M64x128bWarpx2Pair0123B4x16P64,
    Tcgen05CpMember::M64x128bWarpx2Pair0123B6x16P32,
    Tcgen05CpMember::M64x128bWarpx2Pair0123,
    Tcgen05CpMember::M64x128bWarpx2Pair0213B4x16P64,
    Tcgen05CpMember::M64x128bWarpx2Pair0213B6x16P32,
    Tcgen05CpMember::M64x128bWarpx2Pair0213,
];

pub(in crate::resolve) fn tcgen05_cp_member_recipe(
    member: Tcgen05CpMember,
) -> Tcgen05CpMemberRecipe {
    use Tcgen05CpMember::*;
    let (llvm_suffix, ptx_suffix, op_suffix, selection_stem) = match member {
        M128x128bB4x16P64 => (
            "128x128b.b4x16_p64",
            "128x128b.b8x16.b4x16_p64",
            "128x128bB4x16P64",
            "128x128bb4x16_p64",
        ),
        M128x128bB6x16P32 => (
            "128x128b.b6x16_p32",
            "128x128b.b8x16.b6x16_p32",
            "128x128bB6x16P32",
            "128x128bb6x16_p32",
        ),
        M128x128b => ("128x128b", "128x128b", "128x128b", "128x128b"),
        M128x256bB4x16P64 => (
            "128x256b.b4x16_p64",
            "128x256b.b8x16.b4x16_p64",
            "128x256bB4x16P64",
            "128x256bb4x16_p64",
        ),
        M128x256bB6x16P32 => (
            "128x256b.b6x16_p32",
            "128x256b.b8x16.b6x16_p32",
            "128x256bB6x16P32",
            "128x256bb6x16_p32",
        ),
        M32x128bWarpx4B4x16P64 => (
            "32x128b_warpx4.b4x16_p64",
            "32x128b.warpx4.b8x16.b4x16_p64",
            "32x128bWarpx4B4x16P64",
            "32x128b4x16_p64",
        ),
        M32x128bWarpx4B6x16P32 => (
            "32x128b_warpx4.b6x16_p32",
            "32x128b.warpx4.b8x16.b6x16_p32",
            "32x128bWarpx4B6x16P32",
            "32x128b6x16_p32",
        ),
        M32x128bWarpx4 => (
            "32x128b_warpx4",
            "32x128b.warpx4",
            "32x128bWarpx4",
            "32x128",
        ),
        M4x256bB4x16P64 => (
            "4x256b.b4x16_p64",
            "4x256b.b8x16.b4x16_p64",
            "4x256bB4x16P64",
            "4x256bb4x16_p64",
        ),
        M4x256bB6x16P32 => (
            "4x256b.b6x16_p32",
            "4x256b.b8x16.b6x16_p32",
            "4x256bB6x16P32",
            "4x256bb6x16_p32",
        ),
        M4x256b => ("4x256b", "4x256b", "4x256b", "4x256b"),
        M64x128bWarpx2Pair0123B4x16P64 => (
            "64x128b_warpx2_01_23.b4x16_p64",
            "64x128b.warpx2::01_23.b8x16.b4x16_p64",
            "64x128bWarpx2Pair0123B4x16P64",
            "64x128_2b4x16_p64",
        ),
        M64x128bWarpx2Pair0123B6x16P32 => (
            "64x128b_warpx2_01_23.b6x16_p32",
            "64x128b.warpx2::01_23.b8x16.b6x16_p32",
            "64x128bWarpx2Pair0123B6x16P32",
            "64x128_2b6x16_p32",
        ),
        M64x128bWarpx2Pair0123 => (
            "64x128b_warpx2_01_23",
            "64x128b.warpx2::01_23",
            "64x128bWarpx2Pair0123",
            "64x128_2",
        ),
        M64x128bWarpx2Pair0213B4x16P64 => (
            "64x128b_warpx2_02_13.b4x16_p64",
            "64x128b.warpx2::02_13.b8x16.b4x16_p64",
            "64x128bWarpx2Pair0213B4x16P64",
            "64x128_1b4x16_p64",
        ),
        M64x128bWarpx2Pair0213B6x16P32 => (
            "64x128b_warpx2_02_13.b6x16_p32",
            "64x128b.warpx2::02_13.b8x16.b6x16_p32",
            "64x128bWarpx2Pair0213B6x16P32",
            "64x128_1b6x16_p32",
        ),
        M64x128bWarpx2Pair0213 => (
            "64x128b_warpx2_02_13",
            "64x128b.warpx2::02_13",
            "64x128bWarpx2Pair0213",
            "64x128_1",
        ),
    };
    Tcgen05CpMemberRecipe {
        llvm_suffix,
        ptx_suffix,
        op_suffix,
        selection_stem,
    }
}

pub(in crate::resolve) fn materialize_tcgen05_cp_variant(
    base: &OverlayIntrinsic,
    admission: &Tcgen05Admission,
    variant: &Tcgen05CpAdmissionVariant,
) -> OverlayIntrinsic {
    let recipe = tcgen05_cp_member_recipe(variant.member);
    let group = match variant.group {
        Tcgen05CpGroup::Cg1 => 1,
        Tcgen05CpGroup::Cg2 => 2,
    };
    let group_suffix = if group == 1 { "" } else { "_cg2" };
    let id_suffix = recipe.llvm_suffix.replace('.', "_");
    let id = format!("tcgen05_cp_{id_suffix}{group_suffix}");
    let mut record = base.clone();
    record.id = id.clone();
    record.abi_id = variant.abi_id.clone();
    record.operation_key = format!("tcgen05.cp.{}.cg{group}", recipe.llvm_suffix);
    record.source_record = Some(format!("int_nvvm_tcgen05_cp_{}_cg{group}", id_suffix));
    record.rust_name = id.clone();
    record.public_rust_path = format!("cuda_intrinsics::tcgen05::{id}");
    record.compatibility_rust_paths = vec![format!("cuda_device::tcgen05::{id}")];
    record.dialect_op_type = format!(
        "Tcgen05Cp{}{}Op",
        recipe.op_suffix,
        if group == 1 { "" } else { "Cg2" }
    );
    record.dialect_op_name = format!("nvvm.{id}");
    record.llvm_symbol = Some(format!(
        "llvm.nvvm.tcgen05.cp.{}.cg{group}",
        recipe.llvm_suffix
    ));
    record.backend_lowerings[0].evidence_profile = admission
        .cp_llvm_evidence_profile
        .as_ref()
        .expect("validated tcgen05 copy LLVM evidence profile")
        .clone();
    record.backend_lowerings[1].evidence_profile = admission
        .cp_libnvvm_evidence_profile
        .as_ref()
        .expect("validated tcgen05 copy libNVVM evidence profile")
        .clone();
    record.tcgen05 = Some(Tcgen05 {
        operation: if group == 1 {
            Tcgen05Operation::CpSmemToTmem
        } else {
            Tcgen05Operation::CpSmemToTmemCg2
        },
        cp: Some(Tcgen05Cp {
            member: variant.member,
            group: variant.group,
        }),
        ld: None,
        st: None,
        mma: None,
        adapter: Tcgen05Adapter::TmemDescriptorToVoid,
        source_contract: Tcgen05SourceContract::ExactTablegenSelection,
        runtime_validation: admission.runtime_validation,
    });
    record.expected_ptx.modifiers = std::iter::once("cp".into())
        .chain(std::iter::once(format!("cta_group::{group}")))
        .chain(recipe.ptx_suffix.split('.').map(Into::into))
        .collect();
    record.summary = format!(
        "Copies one {} tile from shared memory to tensor memory.",
        recipe.ptx_suffix
    );
    record
}

pub(in crate::resolve) const TCGEN05_LD_VARIANTS: [(Tcgen05LdShape, Tcgen05LdMultiplicity); 29] = {
    use Tcgen05LdMultiplicity::*;
    use Tcgen05LdShape::*;
    [
        (M16x64b, X1),
        (M16x64b, X2),
        (M16x64b, X4),
        (M16x64b, X8),
        (M16x64b, X16),
        (M16x64b, X32),
        (M16x64b, X64),
        (M16x64b, X128),
        (M16x128b, X1),
        (M16x128b, X2),
        (M16x128b, X4),
        (M16x128b, X8),
        (M16x128b, X16),
        (M16x128b, X32),
        (M16x128b, X64),
        (M16x256b, X1),
        (M16x256b, X2),
        (M16x256b, X4),
        (M16x256b, X8),
        (M16x256b, X16),
        (M16x256b, X32),
        (M32x32b, X1),
        (M32x32b, X2),
        (M32x32b, X4),
        (M32x32b, X8),
        (M32x32b, X16),
        (M32x32b, X32),
        (M32x32b, X64),
        (M32x32b, X128),
    ]
};

pub(in crate::resolve) const TCGEN05_OFFSET_LDST_VARIANTS: [(
    Tcgen05LdShape,
    Tcgen05LdMultiplicity,
); 8] = {
    use Tcgen05LdMultiplicity::*;
    use Tcgen05LdShape::*;
    [
        (M16x32bx2, X1),
        (M16x32bx2, X2),
        (M16x32bx2, X4),
        (M16x32bx2, X8),
        (M16x32bx2, X16),
        (M16x32bx2, X32),
        (M16x32bx2, X64),
        (M16x32bx2, X128),
    ]
};

pub(in crate::resolve) fn tcgen05_ld_shape_name(shape: Tcgen05LdShape) -> &'static str {
    match shape {
        Tcgen05LdShape::M16x32bx2 => "16x32bx2",
        Tcgen05LdShape::M16x64b => "16x64b",
        Tcgen05LdShape::M16x128b => "16x128b",
        Tcgen05LdShape::M16x256b => "16x256b",
        Tcgen05LdShape::M32x32b => "32x32b",
    }
}

pub(in crate::resolve) fn tcgen05_ld_multiplicity_name(
    multiplicity: Tcgen05LdMultiplicity,
) -> &'static str {
    match multiplicity {
        Tcgen05LdMultiplicity::X1 => "x1",
        Tcgen05LdMultiplicity::X2 => "x2",
        Tcgen05LdMultiplicity::X4 => "x4",
        Tcgen05LdMultiplicity::X8 => "x8",
        Tcgen05LdMultiplicity::X16 => "x16",
        Tcgen05LdMultiplicity::X32 => "x32",
        Tcgen05LdMultiplicity::X64 => "x64",
        Tcgen05LdMultiplicity::X128 => "x128",
    }
}

pub(in crate::resolve) fn tcgen05_ld_register_count(ld: Tcgen05Ld) -> usize {
    ld.shape.register_multiplier() * ld.multiplicity.count()
}

pub(in crate::resolve) fn tcgen05_ld_id(ld: Tcgen05Ld) -> String {
    format!(
        "tcgen05_ld_{}_{}_{}",
        tcgen05_ld_shape_name(ld.shape),
        tcgen05_ld_multiplicity_name(ld.multiplicity),
        if ld.pack16 { "pack16" } else { "raw" }
    )
}

pub(in crate::resolve) fn tcgen05_ld_source_record(ld: Tcgen05Ld) -> String {
    format!(
        "int_nvvm_tcgen05_ld_{}_{}",
        tcgen05_ld_shape_name(ld.shape),
        tcgen05_ld_multiplicity_name(ld.multiplicity)
    )
}

pub(in crate::resolve) fn tcgen05_ld_llvm_symbol(ld: Tcgen05Ld) -> String {
    format!(
        "llvm.nvvm.tcgen05.ld.{}.{}",
        tcgen05_ld_shape_name(ld.shape),
        tcgen05_ld_multiplicity_name(ld.multiplicity)
    )
}

pub(in crate::resolve) fn tcgen05_ld_rust_result(ld: Tcgen05Ld) -> String {
    match tcgen05_ld_register_count(ld) {
        1 => "u32".into(),
        count => format!("[u32; {count}]"),
    }
}

/// The pinned LLVM 23 TableGen dump models tcgen05 ld/st data as one
/// OVERLOADED vector type-variable per register count (LLVM 22 declared
/// concrete `i32`/`vNi32` types). These anonymous record names are stable
/// for the dump hashes recorded in intrinsics/upstream.lock and follow one
/// arithmetic ladder: count 1 -> anonymous_9933, then +4 per doubling.
pub(in crate::resolve) fn tcgen05_overloaded_data_token(register_count: usize) -> String {
    let anonymous = match register_count {
        1 => 9933,
        2 => 9937,
        4 => 9941,
        8 => 9945,
        16 => 9949,
        32 => 9953,
        64 => 9957,
        128 => 9961,
        other => unreachable!("tcgen05 ld/st register count {other} has no imported record"),
    };
    format!("anonymous_{anonymous}")
}

pub(in crate::resolve) fn tcgen05_ld_llvm_result(ld: Tcgen05Ld) -> String {
    tcgen05_overloaded_data_token(tcgen05_ld_register_count(ld))
}

pub(in crate::resolve) fn tcgen05_ld_op_type(ld: Tcgen05Ld) -> String {
    let shape = tcgen05_ld_shape_name(ld.shape);
    let multiplicity = tcgen05_ld_multiplicity_name(ld.multiplicity)
        .strip_prefix('x')
        .unwrap();
    format!(
        "Tcgen05Ld{shape}X{multiplicity}{}Op",
        if ld.pack16 { "Pack16" } else { "Raw" }
    )
}

pub(in crate::resolve) fn tcgen05_ld_modifiers(ld: Tcgen05Ld) -> Vec<String> {
    let mut modifiers = vec![
        "ld".into(),
        "sync".into(),
        "aligned".into(),
        tcgen05_ld_shape_name(ld.shape).into(),
        tcgen05_ld_multiplicity_name(ld.multiplicity).into(),
    ];
    if ld.pack16 {
        modifiers.push("pack::16b".into());
    }
    modifiers.push("b32".into());
    modifiers
}

pub(in crate::resolve) fn tcgen05_ld_operands(ld: Tcgen05Ld) -> Vec<OperandPattern> {
    let result = OperandPattern::RegisterList {
        length: tcgen05_ld_register_count(ld),
    };
    let mut operands = vec![result, OperandPattern::Address];
    if ld.shape == Tcgen05LdShape::M16x32bx2 {
        operands.push(OperandPattern::Immediate);
    }
    operands
}

pub(in crate::resolve) fn materialize_tcgen05_ld_variant(
    base: &OverlayIntrinsic,
    admission: &Tcgen05Admission,
    variant: &Tcgen05LdAdmissionVariant,
) -> OverlayIntrinsic {
    let ld = Tcgen05Ld {
        shape: variant.shape,
        multiplicity: variant.multiplicity,
        pack16: variant.pack16,
    };
    let id = tcgen05_ld_id(ld);
    let register_count = tcgen05_ld_register_count(ld);
    let rust_result = tcgen05_ld_rust_result(ld);
    let llvm_result = tcgen05_ld_llvm_result(ld);
    let mode = if ld.pack16 { "pack16" } else { "raw" };
    let mut record = base.clone();
    record.id = id.clone();
    record.abi_id = variant.abi_id.clone();
    record.operation_key = format!(
        "tcgen05.ld.{}.{}.{}",
        tcgen05_ld_shape_name(ld.shape),
        tcgen05_ld_multiplicity_name(ld.multiplicity),
        mode
    );
    record.source_record = Some(tcgen05_ld_source_record(ld));
    record.rust_name = id.clone();
    let has_half_split_offset = ld.shape == Tcgen05LdShape::M16x32bx2;
    record.rust_arguments = if has_half_split_offset {
        vec!["u32".into(), "i64".into()]
    } else {
        vec!["u32".into()]
    };
    record.rust_result = rust_result.clone();
    record.must_use = true;
    record.public_rust_path = format!("cuda_intrinsics::tcgen05::{id}");
    record.compatibility_rust_paths = vec![format!(
        "cuda_device::tcgen05::{}",
        if has_half_split_offset {
            format!("__{id}")
        } else {
            id.clone()
        }
    )];
    record.dialect_op_type = tcgen05_ld_op_type(ld);
    record.dialect_op_name = format!("nvvm.{id}");
    record.dialect_operands = if has_half_split_offset {
        vec!["i32".into(), "i64".into()]
    } else {
        vec!["i32".into()]
    };
    record.dialect_results = vec!["i32".into(); register_count];
    record.llvm_symbol = Some(tcgen05_ld_llvm_symbol(ld));
    record.resolved_llvm_symbol = None;
    record.llvm_arguments = if has_half_split_offset {
        vec!["tmem_ptr".into(), "i64".into(), "i1".into()]
    } else {
        vec!["tmem_ptr".into(), "i1".into()]
    };
    record.llvm_results = vec![llvm_result];
    record.ptx_result = rust_result;
    record.execution_scope = Tcgen05Operation::Ld.execution_scope().into();
    record.backend_lowerings[0].evidence_profile = if has_half_split_offset {
        &admission.offset_llvm_evidence_profile
    } else {
        &admission.ld_llvm_evidence_profile
    }
    .as_ref()
    .expect("validated tcgen05 load LLVM evidence profile")
    .clone();
    record.backend_lowerings[1].evidence_profile = if has_half_split_offset {
        &admission.offset_libnvvm_evidence_profile
    } else {
        &admission.ld_libnvvm_evidence_profile
    }
    .as_ref()
    .expect("validated tcgen05 load libNVVM evidence profile")
    .clone();
    record.tcgen05 = Some(Tcgen05 {
        operation: Tcgen05Operation::Ld,
        cp: None,
        ld: Some(ld),
        st: None,
        mma: None,
        adapter: if has_half_split_offset {
            Tcgen05Adapter::TmemHalfSplitOffsetInjectPack16ToU32Registers
        } else {
            Tcgen05Adapter::TmemInjectPack16ToU32Registers
        },
        source_contract: Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        runtime_validation: admission.runtime_validation,
    });
    record.expected_ptx.modifiers = tcgen05_ld_modifiers(ld);
    record.expected_ptx.operands = tcgen05_ld_operands(ld);
    record.summary = format!(
        "Loads {register_count} {} 32-bit register value{} from tensor memory.",
        if ld.pack16 { "packed" } else { "raw" },
        if register_count == 1 { "" } else { "s" }
    );
    record
}

pub(in crate::resolve) const TCGEN05_ST_VARIANTS: [(Tcgen05LdShape, Tcgen05LdMultiplicity); 29] =
    TCGEN05_LD_VARIANTS;

pub(in crate::resolve) fn tcgen05_st_register_count(st: Tcgen05St) -> usize {
    st.shape.register_multiplier() * st.multiplicity.count()
}

pub(in crate::resolve) fn tcgen05_st_id(st: Tcgen05St) -> String {
    format!(
        "tcgen05_st_{}_{}_{}",
        tcgen05_ld_shape_name(st.shape),
        tcgen05_ld_multiplicity_name(st.multiplicity),
        if st.unpack16 { "unpack16" } else { "raw" }
    )
}

pub(in crate::resolve) fn tcgen05_st_source_record(st: Tcgen05St) -> String {
    format!(
        "int_nvvm_tcgen05_st_{}_{}",
        tcgen05_ld_shape_name(st.shape),
        tcgen05_ld_multiplicity_name(st.multiplicity)
    )
}

pub(in crate::resolve) fn tcgen05_st_llvm_symbol(st: Tcgen05St) -> String {
    format!(
        "llvm.nvvm.tcgen05.st.{}.{}",
        tcgen05_ld_shape_name(st.shape),
        tcgen05_ld_multiplicity_name(st.multiplicity)
    )
}

pub(in crate::resolve) fn tcgen05_st_rust_data(st: Tcgen05St) -> String {
    match tcgen05_st_register_count(st) {
        1 => "u32".into(),
        count => format!("[u32; {count}]"),
    }
}

pub(in crate::resolve) fn tcgen05_st_llvm_data(st: Tcgen05St) -> String {
    tcgen05_overloaded_data_token(tcgen05_st_register_count(st))
}

pub(in crate::resolve) fn tcgen05_st_op_type(st: Tcgen05St) -> String {
    let shape = tcgen05_ld_shape_name(st.shape);
    let multiplicity = tcgen05_ld_multiplicity_name(st.multiplicity)
        .strip_prefix('x')
        .unwrap();
    format!(
        "Tcgen05St{shape}X{multiplicity}{}Op",
        if st.unpack16 { "Unpack16" } else { "Raw" }
    )
}

pub(in crate::resolve) fn tcgen05_st_modifiers(st: Tcgen05St) -> Vec<String> {
    let mut modifiers = vec![
        "st".into(),
        "sync".into(),
        "aligned".into(),
        tcgen05_ld_shape_name(st.shape).into(),
        tcgen05_ld_multiplicity_name(st.multiplicity).into(),
    ];
    if st.unpack16 {
        modifiers.push("unpack::16b".into());
    }
    modifiers.push("b32".into());
    modifiers
}

pub(in crate::resolve) fn tcgen05_st_operands(st: Tcgen05St) -> Vec<OperandPattern> {
    let data = OperandPattern::RegisterList {
        length: tcgen05_st_register_count(st),
    };
    let mut operands = vec![OperandPattern::Address];
    if st.shape == Tcgen05LdShape::M16x32bx2 {
        operands.push(OperandPattern::Immediate);
    }
    operands.push(data);
    operands
}

pub(in crate::resolve) fn materialize_tcgen05_st_variant(
    base: &OverlayIntrinsic,
    admission: &Tcgen05Admission,
    variant: &Tcgen05StAdmissionVariant,
) -> OverlayIntrinsic {
    let st = Tcgen05St {
        shape: variant.shape,
        multiplicity: variant.multiplicity,
        unpack16: variant.unpack16,
    };
    let id = tcgen05_st_id(st);
    let register_count = tcgen05_st_register_count(st);
    let rust_data = tcgen05_st_rust_data(st);
    let llvm_data = tcgen05_st_llvm_data(st);
    let mode = if st.unpack16 { "unpack16" } else { "raw" };
    let mut record = base.clone();
    record.id = id.clone();
    record.abi_id = variant.abi_id.clone();
    record.operation_key = format!(
        "tcgen05.st.{}.{}.{}",
        tcgen05_ld_shape_name(st.shape),
        tcgen05_ld_multiplicity_name(st.multiplicity),
        mode
    );
    record.source_record = Some(tcgen05_st_source_record(st));
    record.rust_name = id.clone();
    let has_half_split_offset = st.shape == Tcgen05LdShape::M16x32bx2;
    record.rust_arguments = if has_half_split_offset {
        vec!["u32".into(), "i64".into(), rust_data]
    } else {
        vec!["u32".into(), rust_data]
    };
    record.rust_result = "()".into();
    record.must_use = false;
    record.public_rust_path = format!("cuda_intrinsics::tcgen05::{id}");
    record.compatibility_rust_paths = vec![format!(
        "cuda_device::tcgen05::{}",
        if has_half_split_offset {
            format!("__{id}")
        } else {
            id.clone()
        }
    )];
    record.dialect_op_type = tcgen05_st_op_type(st);
    record.dialect_op_name = format!("nvvm.{id}");
    record.dialect_operands = std::iter::once("i32".into())
        .chain(has_half_split_offset.then(|| "i64".into()))
        .chain(std::iter::repeat_n("i32".into(), register_count))
        .collect();
    record.dialect_results.clear();
    record.llvm_symbol = Some(tcgen05_st_llvm_symbol(st));
    record.resolved_llvm_symbol = None;
    record.llvm_arguments = if has_half_split_offset {
        vec!["tmem_ptr".into(), "i64".into(), llvm_data, "i1".into()]
    } else {
        vec!["tmem_ptr".into(), llvm_data, "i1".into()]
    };
    record.llvm_results.clear();
    record.memory = "write".into();
    record.ptx_result = "()".into();
    record.execution_scope = Tcgen05Operation::St.execution_scope().into();
    record.backend_lowerings[0].evidence_profile = if has_half_split_offset {
        &admission.offset_llvm_evidence_profile
    } else {
        &admission.st_llvm_evidence_profile
    }
    .as_ref()
    .expect("validated tcgen05 store LLVM evidence profile")
    .clone();
    record.backend_lowerings[1].evidence_profile = if has_half_split_offset {
        &admission.offset_libnvvm_evidence_profile
    } else {
        &admission.st_libnvvm_evidence_profile
    }
    .as_ref()
    .expect("validated tcgen05 store libNVVM evidence profile")
    .clone();
    record.tcgen05 = Some(Tcgen05 {
        operation: Tcgen05Operation::St,
        cp: None,
        ld: None,
        st: Some(st),
        mma: None,
        adapter: if has_half_split_offset {
            Tcgen05Adapter::TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid
        } else {
            Tcgen05Adapter::TmemU32RegistersInjectUnpack16ToVoid
        },
        source_contract: Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        runtime_validation: admission.runtime_validation,
    });
    record.expected_ptx.modifiers = tcgen05_st_modifiers(st);
    record.expected_ptx.operands = tcgen05_st_operands(st);
    record.summary = format!(
        "Stores {register_count} {} 32-bit register value{} to tensor memory.",
        if st.unpack16 { "unpacked" } else { "raw" },
        if register_count == 1 { "" } else { "s" }
    );
    record
}

pub(in crate::resolve) fn validate_tcgen05_ld_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tcgen05: &Tcgen05,
    ld: Tcgen05Ld,
) -> Result<()> {
    let has_half_split_offset = ld.shape == Tcgen05LdShape::M16x32bx2;
    let variants: &[(Tcgen05LdShape, Tcgen05LdMultiplicity)] = if has_half_split_offset {
        &TCGEN05_OFFSET_LDST_VARIANTS
    } else {
        &TCGEN05_LD_VARIANTS
    };
    ensure!(
        variants.contains(&(ld.shape, ld.multiplicity)),
        "{} has an unsupported tcgen05 load identity",
        policy.id
    );
    let id = tcgen05_ld_id(ld);
    let source_record = tcgen05_ld_source_record(ld);
    let llvm_symbol = tcgen05_ld_llvm_symbol(ld);
    let rust_result = tcgen05_ld_rust_result(ld);
    let llvm_result = tcgen05_ld_llvm_result(ld);
    let register_count = tcgen05_ld_register_count(ld);
    let mode = if ld.pack16 { "pack16" } else { "raw" };
    let expected_rust_arguments = if has_half_split_offset {
        vec!["u32", "i64"]
    } else {
        vec!["u32"]
    };
    let compatibility_name = if has_half_split_offset {
        format!("__{id}")
    } else {
        id.clone()
    };
    let expected_dialect_operands = if has_half_split_offset {
        vec!["i32", "i64"]
    } else {
        vec!["i32"]
    };
    let expected_llvm_arguments = if has_half_split_offset {
        vec!["tmem_ptr", "i64", "i1"]
    } else {
        vec!["tmem_ptr", "i1"]
    };
    let expected_properties = if has_half_split_offset {
        vec![
            "ImmArg<arg1>",
            "ImmArg<arg2>",
            "IntrArgMemOnly",
            "IntrConvergent",
            "NoCapture<arg0>",
        ]
    } else {
        vec![
            "ImmArg<arg1>",
            "IntrArgMemOnly",
            "IntrConvergent",
            "NoCapture<arg0>",
        ]
    };
    ensure!(
        policy.id == id
            && policy.operation_key
                == format!(
                    "tcgen05.ld.{}.{}.{}",
                    tcgen05_ld_shape_name(ld.shape),
                    tcgen05_ld_multiplicity_name(ld.multiplicity),
                    mode
                )
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == source_record
            && declaration.llvm_name == llvm_symbol,
        "{} tcgen05 load identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == id
            && policy.rust_arguments == expected_rust_arguments
            && policy.rust_result == rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{id}")
            && policy.compatibility_rust_paths
                == [format!("cuda_device::tcgen05::{compatibility_name}")],
        "{} tcgen05 load Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == tcgen05_ld_op_type(ld)
            && policy.dialect_op_name == format!("nvvm.{id}")
            && policy.dialect_operands == expected_dialect_operands
            && policy.dialect_results == vec!["i32"; register_count]
            && policy.llvm_arguments == expected_llvm_arguments
            && policy.llvm_results == [llvm_result.as_str()]
            && declaration.arguments == expected_llvm_arguments
            && declaration.results == [llvm_result.as_str()]
            && declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_TCGEN05_LD"]
            && declaration.properties == expected_properties
            && declaration.selections.is_empty()
            && policy.lowering == "generated_tcgen05",
        "{} tcgen05 load carrier or imported declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read"
            && policy.convergent
            && policy.execution_scope == Tcgen05Operation::Ld.execution_scope()
            && tcgen05.operation == Tcgen05Operation::Ld
            && tcgen05.cp.is_none()
            && tcgen05.ld == Some(ld)
            && tcgen05.st.is_none()
            && tcgen05.adapter
                == if has_half_split_offset {
                    Tcgen05Adapter::TmemHalfSplitOffsetInjectPack16ToU32Registers
                } else {
                    Tcgen05Adapter::TmemInjectPack16ToU32Registers
                }
            && tcgen05.source_contract == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 load semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LLVM_TARGETS
            && policy.ptx_isa_version == "8.6"
            && policy.ptx_result == rust_result
            && policy.expected_ptx.mnemonic == "tcgen05"
            && policy.expected_ptx.modifiers == tcgen05_ld_modifiers(ld)
            && policy.expected_ptx.operands == tcgen05_ld_operands(ld),
        "{} tcgen05 load target or PTX contract changed",
        policy.id
    );
    validate_tcgen05_backend_routes(policy, "tcgen05 load")?;
    ensure_no_other_family_contract(policy, "tcgen05 load")?;
    Ok(())
}

pub(in crate::resolve) fn validate_tcgen05_st_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tcgen05: &Tcgen05,
    st: Tcgen05St,
) -> Result<()> {
    let has_half_split_offset = st.shape == Tcgen05LdShape::M16x32bx2;
    let variants: &[(Tcgen05LdShape, Tcgen05LdMultiplicity)] = if has_half_split_offset {
        &TCGEN05_OFFSET_LDST_VARIANTS
    } else {
        &TCGEN05_ST_VARIANTS
    };
    ensure!(
        variants.contains(&(st.shape, st.multiplicity)),
        "{} has an unsupported tcgen05 store identity",
        policy.id
    );
    let id = tcgen05_st_id(st);
    let source_record = tcgen05_st_source_record(st);
    let llvm_symbol = tcgen05_st_llvm_symbol(st);
    let rust_data = tcgen05_st_rust_data(st);
    let llvm_data = tcgen05_st_llvm_data(st);
    let register_count = tcgen05_st_register_count(st);
    let mode = if st.unpack16 { "unpack16" } else { "raw" };
    let expected_rust_arguments = if has_half_split_offset {
        vec!["u32", "i64", rust_data.as_str()]
    } else {
        vec!["u32", rust_data.as_str()]
    };
    let compatibility_name = if has_half_split_offset {
        format!("__{id}")
    } else {
        id.clone()
    };
    let expected_dialect_operands = std::iter::once("i32".to_owned())
        .chain(has_half_split_offset.then(|| "i64".to_owned()))
        .chain(std::iter::repeat_n("i32".to_owned(), register_count))
        .collect::<Vec<_>>();
    let expected_llvm_arguments = if has_half_split_offset {
        vec!["tmem_ptr", "i64", llvm_data.as_str(), "i1"]
    } else {
        vec!["tmem_ptr", llvm_data.as_str(), "i1"]
    };
    let expected_properties = if has_half_split_offset {
        vec![
            "ImmArg<arg1>",
            "ImmArg<arg3>",
            "IntrArgMemOnly",
            "IntrConvergent",
            "NoCapture<arg0>",
        ]
    } else {
        vec![
            "ImmArg<arg2>",
            "IntrArgMemOnly",
            "IntrConvergent",
            "NoCapture<arg0>",
        ]
    };
    ensure!(
        policy.id == id
            && policy.operation_key
                == format!(
                    "tcgen05.st.{}.{}.{}",
                    tcgen05_ld_shape_name(st.shape),
                    tcgen05_ld_multiplicity_name(st.multiplicity),
                    mode
                )
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == source_record
            && declaration.llvm_name == llvm_symbol,
        "{} tcgen05 store identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == id
            && policy.rust_arguments == expected_rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{id}")
            && policy.compatibility_rust_paths
                == [format!("cuda_device::tcgen05::{compatibility_name}")],
        "{} tcgen05 store Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == tcgen05_st_op_type(st)
            && policy.dialect_op_name == format!("nvvm.{id}")
            && policy.dialect_operands == expected_dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == expected_llvm_arguments
            && policy.llvm_results.is_empty()
            && declaration.arguments == expected_llvm_arguments
            && declaration.results.is_empty()
            && declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_TCGEN05_ST"]
            && declaration.properties == expected_properties
            && declaration.selections.is_empty()
            && policy.lowering == "generated_tcgen05",
        "{} tcgen05 store carrier or imported declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "write"
            && policy.convergent
            && policy.execution_scope == Tcgen05Operation::St.execution_scope()
            && tcgen05.operation == Tcgen05Operation::St
            && tcgen05.cp.is_none()
            && tcgen05.ld.is_none()
            && tcgen05.st == Some(st)
            && tcgen05.adapter
                == if has_half_split_offset {
                    Tcgen05Adapter::TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid
                } else {
                    Tcgen05Adapter::TmemU32RegistersInjectUnpack16ToVoid
                }
            && tcgen05.source_contract == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 store semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LLVM_TARGETS
            && policy.ptx_isa_version == "8.6"
            && policy.ptx_result == "()"
            && policy.expected_ptx.mnemonic == "tcgen05"
            && policy.expected_ptx.modifiers == tcgen05_st_modifiers(st)
            && policy.expected_ptx.operands == tcgen05_st_operands(st),
        "{} tcgen05 store target or PTX contract changed",
        policy.id
    );
    validate_tcgen05_backend_routes(policy, "tcgen05 store")?;
    ensure_no_other_family_contract(policy, "tcgen05 store")?;
    Ok(())
}

pub(in crate::resolve) fn validate_tcgen05_cp_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tcgen05: &Tcgen05,
    cp: Tcgen05Cp,
) -> Result<()> {
    ensure!(
        TCGEN05_CP_MEMBERS.contains(&cp.member),
        "{} has an unsupported tcgen05 copy identity",
        policy.id
    );
    let recipe = tcgen05_cp_member_recipe(cp.member);
    let group = match cp.group {
        Tcgen05CpGroup::Cg1 => 1,
        Tcgen05CpGroup::Cg2 => 2,
    };
    let group_suffix = if group == 1 { "" } else { "_cg2" };
    let id_suffix = recipe.llvm_suffix.replace('.', "_");
    let id = format!("tcgen05_cp_{id_suffix}{group_suffix}");
    let operation = if group == 1 {
        Tcgen05Operation::CpSmemToTmem
    } else {
        Tcgen05Operation::CpSmemToTmemCg2
    };
    let op_type = format!(
        "Tcgen05Cp{}{}Op",
        recipe.op_suffix,
        if group == 1 { "" } else { "Cg2" }
    );
    let source_record = format!("int_nvvm_tcgen05_cp_{}_cg{group}", id_suffix);
    let llvm_symbol = format!("llvm.nvvm.tcgen05.cp.{}.cg{group}", recipe.llvm_suffix);
    let modifiers = std::iter::once("cp".into())
        .chain(std::iter::once(format!("cta_group::{group}")))
        .chain(recipe.ptx_suffix.split('.').map(Into::into))
        .collect::<Vec<String>>();
    ensure!(
        policy.id == id
            && policy.operation_key == format!("tcgen05.cp.{}.cg{group}", recipe.llvm_suffix)
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == source_record
            && declaration.llvm_name == llvm_symbol,
        "{} tcgen05 copy identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == id
            && policy.rust_arguments == ["u32", "u64"]
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{id}")
            && policy.compatibility_rust_paths == [format!("cuda_device::tcgen05::{id}")],
        "{} tcgen05 copy Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == op_type
            && policy.dialect_op_name == format!("nvvm.{id}")
            && policy.dialect_operands == ["i32", "i64"]
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == ["tmem_ptr", "i64"]
            && policy.llvm_results.is_empty()
            && declaration.arguments == ["tmem_ptr", "i64"]
            && declaration.results.is_empty()
            && declaration.classes == ["SDPatternOperator", "Intrinsic"]
            && declaration.properties
                == [
                    "IntrConvergent",
                    "IntrInaccessibleMemOrArgMemOnly",
                    "NoCapture<arg0>",
                ]
            && policy.lowering == "generated_tcgen05",
        "{} tcgen05 copy carrier or declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "thread"
            && tcgen05.operation == operation
            && tcgen05.ld.is_none()
            && tcgen05.st.is_none()
            && tcgen05.adapter == Tcgen05Adapter::TmemDescriptorToVoid
            && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 copy semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LLVM_TARGETS
            && policy.ptx_isa_version == "8.6"
            && policy.ptx_result == "()"
            && policy.expected_ptx.mnemonic == "tcgen05"
            && policy.expected_ptx.modifiers == modifiers
            && policy.expected_ptx.operands == [OperandPattern::Address, OperandPattern::Register],
        "{} tcgen05 copy target or PTX contract changed",
        policy.id
    );
    validate_tcgen05_backend_routes(policy, "tcgen05 copy")?;
    ensure!(
        declaration.selections.len() == 1,
        "{} must keep one exact tcgen05 copy selection",
        policy.id
    );
    let selection = &declaration.selections[0];
    ensure!(
        selection.source_record == format!("TCGEN05_CP_{}_cg{group}", recipe.selection_stem)
            && selection.asm
                == format!(
                    "tcgen05.cp.cta_group::{group}.{} \t[$tmem_addr], $sdesc;",
                    recipe.ptx_suffix
                )
            && selection.predicates == ["Subtarget->hasTcgen05InstSupport()"]
            && selection.constraints.is_empty(),
        "{} exact tcgen05 copy selection changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "tcgen05 copy")?;
    Ok(())
}
