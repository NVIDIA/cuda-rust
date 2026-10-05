/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    OverlayIntrinsic, PackedAluAdapter, PackedAluFormat, PackedAluOperation,
    PackedAtomicAccessContract, PackedAtomicAdapter, PackedAtomicAtomicity,
    PackedAtomicCodegenContract, PackedAtomicFormat, PackedAtomicOperation, PackedAtomicOrdering,
    PackedAtomicPointerContract, PackedAtomicReturnContract, PackedAtomicRounding,
    PackedAtomicScope, PackedAtomicScopeContract, PackedAtomicStateSpace, PackedAtomicSubnormal,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

pub(in crate::resolve) fn validate_packed_atomic_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
) -> Result<()> {
    let packed = policy
        .packed_atomic
        .as_ref()
        .with_context(|| format!("{} has no closed packed-atomic contract", policy.id))?;
    ensure!(
        packed.operation == PackedAtomicOperation::Add
            && packed.state_space == PackedAtomicStateSpace::Global
            && packed.ordering == PackedAtomicOrdering::Relaxed
            && packed.scope == PackedAtomicScope::Gpu
            && packed.rounding == PackedAtomicRounding::NearestEven
            && packed.subnormal == PackedAtomicSubnormal::Preserve
            && packed.atomicity == PackedAtomicAtomicity::PerElement
            && packed.pointer_contract == PackedAtomicPointerContract::MutableGlobalU32Aligned4
            && packed.access_contract
                == PackedAtomicAccessContract::NoMixedWholeWordOrNonAtomicAccess
            && packed.scope_contract == PackedAtomicScopeContract::RacingAtomicsMutuallyInclusive
            && packed.codegen_contract == PackedAtomicCodegenContract::ExactNativeInstruction
            && packed.return_contract
                == PackedAtomicReturnContract::OldValuesPerElementMayBeNoncoherent
            && packed.adapter == PackedAtomicAdapter::OldPackedU32,
        "{} requests an unsupported packed-atomic semantic or safety contract",
        policy.id
    );
    let (format, native_sm, minimum_ptx, minimum_sm, public_name) = match packed.format {
        PackedAtomicFormat::F16x2 => ("f16x2", 60, "6.2", "sm_70", "atom_add_f16x2"),
        PackedAtomicFormat::Bf16x2 => ("bf16x2", 90, "7.8", "sm_90", "atom_add_bf16x2"),
    };
    ensure!(
        packed.native_minimum_sm == native_sm,
        "{} PTX-native hardware floor does not match the selected packed format",
        policy.id
    );
    ensure!(
        source
            == &IntrinsicSource::PtxNative {
                instruction: format!("atom.global.add.noftz.{format}"),
            },
        "{} PTX-native source does not match its packed format",
        policy.id
    );
    ensure!(
        policy.rust_module == "atomic"
            && policy.rust_name == public_name
            && policy.rust_arguments == ["*mut u32", "u32"]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::atomic::{public_name}")
            && policy.compatibility_rust_paths == [format!("cuda_device::atomic::{public_name}")],
        "{} must preserve the unsafe must-use packed atomic raw/compatibility API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "PackedAtomicAddOp"
            && policy.dialect_op_name == "nvvm.packed_atomic_add"
            && policy.dialect_operands == ["ptr", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.lowering == "generated_packed_atomic_inline_ptx",
        "{} is outside the one closed packed-atomic dialect recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == Some(minimum_sm)
            && policy.targets == "all"
            && policy.ptx_result == "u32"
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.selected_address_space.is_none(),
        "{} packed-atomic effects, carrier, or native target floor disagree",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "atom"
            && policy.expected_ptx.modifiers == ["global", "add", "noftz", format]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
        "{} expected PTX must match the exact packed global add spelling",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        backend_pairs
            == BTreeSet::from([
                (
                    IntrinsicBackend::LlvmNvptx,
                    BackendLoweringMechanism::InlinePtx,
                ),
                (
                    IntrinsicBackend::LibNvvm,
                    BackendLoweringMechanism::InlinePtx,
                ),
            ]),
        "{} must define exactly the reviewed LLVM-NVPTX and libNVVM inline-PTX routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let expected_sm = match (packed.format, lowering.backend) {
            (PackedAtomicFormat::F16x2, IntrinsicBackend::LlvmNvptx) => "sm_70",
            (PackedAtomicFormat::F16x2, IntrinsicBackend::LibNvvm) => "sm_75",
            (PackedAtomicFormat::Bf16x2, _) => "sm_90",
        };
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == Some(expected_sm)
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact reviewed profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

pub(in crate::resolve) enum PackedAluRecipeSource {
    Imported {
        record: &'static str,
        symbol: &'static str,
        resolved_symbol: Option<&'static str>,
        arguments: &'static [&'static str],
        results: &'static [&'static str],
        properties: &'static [&'static str],
        selection: &'static str,
        selection_asm: &'static str,
    },
    PtxNative,
}

pub(in crate::resolve) struct PackedAluRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) arity: usize,
    pub(in crate::resolve) must_use: bool,
    pub(in crate::resolve) ptx_mnemonic: &'static str,
    pub(in crate::resolve) modifiers: &'static [&'static str],
    pub(in crate::resolve) native_minimum_sm: u16,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: &'static str,
    pub(in crate::resolve) ptx_isa_section: &'static str,
    pub(in crate::resolve) ptx_isa_url: &'static str,
    pub(in crate::resolve) source: PackedAluRecipeSource,
}

/// Returns the closed recipe for a packed-ALU (format, operation) pair.
///
/// `None` means the pair is outside the family: the operation exists for some
/// other format but this one has no reviewed lowering for it.
pub(in crate::resolve) fn packed_alu_recipe(
    format: PackedAluFormat,
    operation: PackedAluOperation,
) -> Option<PackedAluRecipe> {
    match format {
        PackedAluFormat::Bf16x2 => packed_bf16x2_alu_recipe(operation),
        PackedAluFormat::F16x2 => packed_f16x2_alu_recipe(operation),
        PackedAluFormat::F32x2 => packed_f32x2_alu_recipe(operation),
    }
}

pub(in crate::resolve) fn packed_f32x2_alu_recipe(
    operation: PackedAluOperation,
) -> Option<PackedAluRecipe> {
    let (
        id,
        abi_id,
        operation_key,
        rust_name,
        dialect_op_type,
        dialect_op_name,
        arity,
        head,
        modifiers,
        section,
        url,
    ) = match operation {
        PackedAluOperation::Add => (
            "add_f32x2",
            "i0995",
            "packed.alu.f32x2.add",
            "add_f32x2",
            "AddF32x2Op",
            "nvvm.add_f32x2",
            2,
            "add.rn.f32x2",
            &["rn", "f32x2"][..],
            "9.7.3.1 Floating Point Instructions: add",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add",
        ),
        PackedAluOperation::AddFtz => (
            "add_ftz_f32x2",
            "i0996",
            "packed.alu.f32x2.add.ftz",
            "add_ftz_f32x2",
            "AddFtzF32x2Op",
            "nvvm.add_ftz_f32x2",
            2,
            "add.rn.ftz.f32x2",
            &["rn", "ftz", "f32x2"][..],
            "9.7.3.1 Floating Point Instructions: add",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add",
        ),
        PackedAluOperation::Sub => (
            "sub_f32x2",
            "i0997",
            "packed.alu.f32x2.sub",
            "sub_f32x2",
            "SubF32x2Op",
            "nvvm.sub_f32x2",
            2,
            "sub.rn.f32x2",
            &["rn", "f32x2"][..],
            "9.7.3.2 Floating Point Instructions: sub",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sub",
        ),
        PackedAluOperation::SubFtz => (
            "sub_ftz_f32x2",
            "i0998",
            "packed.alu.f32x2.sub.ftz",
            "sub_ftz_f32x2",
            "SubFtzF32x2Op",
            "nvvm.sub_ftz_f32x2",
            2,
            "sub.rn.ftz.f32x2",
            &["rn", "ftz", "f32x2"][..],
            "9.7.3.2 Floating Point Instructions: sub",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sub",
        ),
        PackedAluOperation::Mul => (
            "mul_f32x2",
            "i0999",
            "packed.alu.f32x2.mul",
            "mul_f32x2",
            "MulF32x2Op",
            "nvvm.mul_f32x2",
            2,
            "mul.rn.f32x2",
            &["rn", "f32x2"][..],
            "9.7.3.3 Floating Point Instructions: mul",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul",
        ),
        PackedAluOperation::MulFtz => (
            "mul_ftz_f32x2",
            "i1000",
            "packed.alu.f32x2.mul.ftz",
            "mul_ftz_f32x2",
            "MulFtzF32x2Op",
            "nvvm.mul_ftz_f32x2",
            2,
            "mul.rn.ftz.f32x2",
            &["rn", "ftz", "f32x2"][..],
            "9.7.3.3 Floating Point Instructions: mul",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul",
        ),
        PackedAluOperation::Fma => (
            "fma_f32x2",
            "i1001",
            "packed.alu.f32x2.fma",
            "fma_f32x2",
            "FmaF32x2Op",
            "nvvm.fma_f32x2",
            3,
            "fma.rn.f32x2",
            &["rn", "f32x2"][..],
            "9.7.3.4 Floating Point Instructions: fma",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma",
        ),
        PackedAluOperation::FmaFtz => (
            "fma_ftz_f32x2",
            "i1002",
            "packed.alu.f32x2.fma.ftz",
            "fma_ftz_f32x2",
            "FmaFtzF32x2Op",
            "nvvm.fma_ftz_f32x2",
            3,
            "fma.rn.ftz.f32x2",
            &["rn", "ftz", "f32x2"][..],
            "9.7.3.4 Floating Point Instructions: fma",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma",
        ),
        _ => return None,
    };
    Some(PackedAluRecipe {
        id,
        abi_id,
        operation_key,
        rust_name,
        dialect_op_type,
        dialect_op_name,
        arity,
        must_use: true,
        ptx_mnemonic: head,
        modifiers,
        native_minimum_sm: 100,
        minimum_ptx: "8.6",
        minimum_sm: "sm_100",
        ptx_isa_section: section,
        ptx_isa_url: url,
        source: PackedAluRecipeSource::PtxNative,
    })
}

pub(in crate::resolve) fn packed_bf16x2_alu_recipe(
    operation: PackedAluOperation,
) -> Option<PackedAluRecipe> {
    const PURE: &[&str] = &["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"];
    const COMMUTATIVE_PURE: &[&str] = &[
        "Commutative",
        "IntrNoCreateUndefOrPoison",
        "IntrNoMem",
        "IntrSpeculatable",
    ];
    Some(match operation {
        PackedAluOperation::Fma => PackedAluRecipe {
            id: "fma_bf16x2",
            abi_id: "i0062",
            operation_key: "packed.alu.bf16x2.fma",
            rust_name: "fma_bf16x2",
            dialect_op_type: "FmaBf16x2Op",
            dialect_op_name: "nvvm.fma_bf16x2",
            arity: 3,
            must_use: false,
            ptx_mnemonic: "fma.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_bf16x2",
                symbol: "llvm.nvvm.fma.rn.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_bf16x2",
                selection_asm: "fma.rn.bf16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaRelu => PackedAluRecipe {
            id: "fma_relu_bf16x2",
            abi_id: "i0063",
            operation_key: "packed.alu.bf16x2.fma.relu",
            rust_name: "fma_relu_bf16x2",
            dialect_op_type: "FmaReluBf16x2Op",
            dialect_op_name: "nvvm.fma_relu_bf16x2",
            arity: 3,
            must_use: false,
            ptx_mnemonic: "fma.rn.relu.bf16x2",
            modifiers: &["rn", "relu", "bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_relu_bf16x2",
                symbol: "llvm.nvvm.fma.rn.relu.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_relu_bf16x2",
                selection_asm: "fma.rn.relu.bf16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::Add => PackedAluRecipe {
            id: "add_bf16x2",
            abi_id: "i0064",
            operation_key: "packed.alu.bf16x2.add",
            rust_name: "add_bf16x2",
            dialect_op_type: "AddBf16x2Op",
            dialect_op_name: "nvvm.add_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "add.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.1 Half Precision Floating Point Instructions: add",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Sub => PackedAluRecipe {
            id: "sub_bf16x2",
            abi_id: "i0065",
            operation_key: "packed.alu.bf16x2.sub",
            rust_name: "sub_bf16x2",
            dialect_op_type: "SubBf16x2Op",
            dialect_op_name: "nvvm.sub_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "sub.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.2 Half Precision Floating Point Instructions: sub",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Mul => PackedAluRecipe {
            id: "mul_bf16x2",
            abi_id: "i0066",
            operation_key: "packed.alu.bf16x2.mul",
            rust_name: "mul_bf16x2",
            dialect_op_type: "MulBf16x2Op",
            dialect_op_name: "nvvm.mul_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "mul.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.3 Half Precision Floating Point Instructions: mul",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Min => PackedAluRecipe {
            id: "min_bf16x2",
            abi_id: "i0067",
            operation_key: "packed.alu.bf16x2.min",
            rust_name: "min_bf16x2",
            dialect_op_type: "MinBf16x2Op",
            dialect_op_name: "nvvm.min_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "min.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.7 Half Precision Floating Point Instructions: min",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmin_bf16x2",
                symbol: "llvm.nvvm.fmin.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMIN_bf16x2",
                selection_asm: "min.bf16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Max => PackedAluRecipe {
            id: "max_bf16x2",
            abi_id: "i0068",
            operation_key: "packed.alu.bf16x2.max",
            rust_name: "max_bf16x2",
            dialect_op_type: "MaxBf16x2Op",
            dialect_op_name: "nvvm.max_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "max.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.8 Half Precision Floating Point Instructions: max",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmax_bf16x2",
                symbol: "llvm.nvvm.fmax.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMAN_bf16x2",
                selection_asm: "max.bf16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Neg => PackedAluRecipe {
            id: "neg_bf16x2",
            abi_id: "i0069",
            operation_key: "packed.alu.bf16x2.neg",
            rust_name: "neg_bf16x2",
            dialect_op_type: "NegBf16x2Op",
            dialect_op_name: "nvvm.neg_bf16x2",
            arity: 1,
            must_use: false,
            ptx_mnemonic: "neg.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.5 Half Precision Floating Point Instructions: neg",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-neg",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_neg_bf16x2",
                symbol: "llvm.nvvm.neg.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_NEG_BF16X2",
                selection_asm: "neg.bf16x2 \t$dst, $src0;",
            },
        },
        PackedAluOperation::Abs => PackedAluRecipe {
            id: "abs_bf16x2",
            abi_id: "i0070",
            operation_key: "packed.alu.bf16x2.abs",
            rust_name: "abs_bf16x2",
            dialect_op_type: "AbsBf16x2Op",
            dialect_op_name: "nvvm.abs_bf16x2",
            arity: 1,
            must_use: false,
            ptx_mnemonic: "abs.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.6 Half Precision Floating Point Instructions: abs",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fabs",
                symbol: "llvm.nvvm.fabs",
                resolved_symbol: Some("llvm.nvvm.fabs.v2bf16"),
                arguments: &["anonymous_8"],
                results: &["anyfloat"],
                properties: PURE,
                selection: "ABS_BF16X2",
                selection_asm: "abs.bf16x2 \t$dst, $src0;",
            },
        },
        // The f32x2-only operation keys are outside this admission. The ftz
        // and sat fma forms also do not exist for bf16x2 in the PTX ISA at
        // all: ptxas rejects hand-written `fma.rn.ftz.bf16x2` with
        // "Illegal modifier '.ftz' for instruction 'fma'", and accepts
        // `fma.rn.relu.bf16x2` beside it. LLVM declares the intrinsics anyway,
        // so instruction selection is what fails first ("Cannot select:
        // intrinsic %llvm.nvvm.fma.rn.ftz.bf16x2"), but the assembler is the
        // authority here. Reject them rather than admit a recipe that has no
        // instruction to lower to.
        PackedAluOperation::AddFtz
        | PackedAluOperation::SubFtz
        | PackedAluOperation::MulFtz
        | PackedAluOperation::FmaFtz
        | PackedAluOperation::FmaSat
        | PackedAluOperation::FmaFtzSat
        | PackedAluOperation::FmaFtzRelu => return None,
    })
}

pub(in crate::resolve) fn packed_f16x2_alu_recipe(
    operation: PackedAluOperation,
) -> Option<PackedAluRecipe> {
    const PURE: &[&str] = &["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"];
    const COMMUTATIVE_PURE: &[&str] = &[
        "Commutative",
        "IntrNoCreateUndefOrPoison",
        "IntrNoMem",
        "IntrSpeculatable",
    ];
    Some(match operation {
        PackedAluOperation::Fma => PackedAluRecipe {
            id: "fma_f16x2",
            abi_id: "i0072",
            operation_key: "packed.alu.f16x2.fma",
            rust_name: "fma_f16x2",
            dialect_op_type: "FmaF16x2Op",
            dialect_op_name: "nvvm.fma_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_f16x2",
                symbol: "llvm.nvvm.fma.rn.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_f16x2",
                selection_asm: "fma.rn.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaFtz => PackedAluRecipe {
            id: "fma_ftz_f16x2",
            abi_id: "i0854",
            operation_key: "packed.alu.f16x2.fma.ftz",
            rust_name: "fma_ftz_f16x2",
            dialect_op_type: "FmaFtzF16x2Op",
            dialect_op_name: "nvvm.fma_ftz_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.ftz.f16x2",
            modifiers: &["rn", "ftz", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_ftz_f16x2",
                symbol: "llvm.nvvm.fma.rn.ftz.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_ftz_f16x2",
                selection_asm: "fma.rn.ftz.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaSat => PackedAluRecipe {
            id: "fma_sat_f16x2",
            abi_id: "i0855",
            operation_key: "packed.alu.f16x2.fma.sat",
            rust_name: "fma_sat_f16x2",
            dialect_op_type: "FmaSatF16x2Op",
            dialect_op_name: "nvvm.fma_sat_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.sat.f16x2",
            modifiers: &["rn", "sat", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_sat_f16x2",
                symbol: "llvm.nvvm.fma.rn.sat.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_sat_f16x2",
                selection_asm: "fma.rn.sat.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaFtzSat => PackedAluRecipe {
            id: "fma_ftz_sat_f16x2",
            abi_id: "i0856",
            operation_key: "packed.alu.f16x2.fma.ftz.sat",
            rust_name: "fma_ftz_sat_f16x2",
            dialect_op_type: "FmaFtzSatF16x2Op",
            dialect_op_name: "nvvm.fma_ftz_sat_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.ftz.sat.f16x2",
            modifiers: &["rn", "ftz", "sat", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_ftz_sat_f16x2",
                symbol: "llvm.nvvm.fma.rn.ftz.sat.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_ftz_sat_f16x2",
                selection_asm: "fma.rn.ftz.sat.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaRelu => PackedAluRecipe {
            id: "fma_relu_f16x2",
            abi_id: "i0073",
            operation_key: "packed.alu.f16x2.fma.relu",
            rust_name: "fma_relu_f16x2",
            dialect_op_type: "FmaReluF16x2Op",
            dialect_op_name: "nvvm.fma_relu_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.relu.f16x2",
            modifiers: &["rn", "relu", "f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_relu_f16x2",
                symbol: "llvm.nvvm.fma.rn.relu.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_relu_f16x2",
                selection_asm: "fma.rn.relu.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaFtzRelu => PackedAluRecipe {
            id: "fma_ftz_relu_f16x2",
            abi_id: "i0857",
            operation_key: "packed.alu.f16x2.fma.ftz.relu",
            rust_name: "fma_ftz_relu_f16x2",
            dialect_op_type: "FmaFtzReluF16x2Op",
            dialect_op_name: "nvvm.fma_ftz_relu_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.ftz.relu.f16x2",
            modifiers: &["rn", "ftz", "relu", "f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_ftz_relu_f16x2",
                symbol: "llvm.nvvm.fma.rn.ftz.relu.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_ftz_relu_f16x2",
                selection_asm: "fma.rn.ftz.relu.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::Add => PackedAluRecipe {
            id: "add_f16x2",
            abi_id: "i0074",
            operation_key: "packed.alu.f16x2.add",
            rust_name: "add_f16x2",
            dialect_op_type: "AddF16x2Op",
            dialect_op_name: "nvvm.add_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "add.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.1 Half Precision Floating Point Instructions: add",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Sub => PackedAluRecipe {
            id: "sub_f16x2",
            abi_id: "i0075",
            operation_key: "packed.alu.f16x2.sub",
            rust_name: "sub_f16x2",
            dialect_op_type: "SubF16x2Op",
            dialect_op_name: "nvvm.sub_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "sub.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.2 Half Precision Floating Point Instructions: sub",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Mul => PackedAluRecipe {
            id: "mul_f16x2",
            abi_id: "i0076",
            operation_key: "packed.alu.f16x2.mul",
            rust_name: "mul_f16x2",
            dialect_op_type: "MulF16x2Op",
            dialect_op_name: "nvvm.mul_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "mul.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.3 Half Precision Floating Point Instructions: mul",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Min => PackedAluRecipe {
            id: "min_f16x2",
            abi_id: "i0077",
            operation_key: "packed.alu.f16x2.min",
            rust_name: "min_f16x2",
            dialect_op_type: "MinF16x2Op",
            dialect_op_name: "nvvm.min_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "min.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.7 Half Precision Floating Point Instructions: min",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmin_f16x2",
                symbol: "llvm.nvvm.fmin.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16"],
                results: &["v2f16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMIN_f16x2",
                selection_asm: "min.f16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Max => PackedAluRecipe {
            id: "max_f16x2",
            abi_id: "i0078",
            operation_key: "packed.alu.f16x2.max",
            rust_name: "max_f16x2",
            dialect_op_type: "MaxF16x2Op",
            dialect_op_name: "nvvm.max_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "max.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.8 Half Precision Floating Point Instructions: max",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmax_f16x2",
                symbol: "llvm.nvvm.fmax.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16"],
                results: &["v2f16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMAN_f16x2",
                selection_asm: "max.f16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Neg => PackedAluRecipe {
            id: "neg_f16x2",
            abi_id: "i0079",
            operation_key: "packed.alu.f16x2.neg",
            rust_name: "neg_f16x2",
            dialect_op_type: "NegF16x2Op",
            dialect_op_name: "nvvm.neg_f16x2",
            arity: 1,
            must_use: true,
            ptx_mnemonic: "neg.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "6.0",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.5 Half Precision Floating Point Instructions: neg",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-neg",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Abs => PackedAluRecipe {
            id: "abs_f16x2",
            abi_id: "i0080",
            operation_key: "packed.alu.f16x2.abs",
            rust_name: "abs_f16x2",
            dialect_op_type: "AbsF16x2Op",
            dialect_op_name: "nvvm.abs_f16x2",
            arity: 1,
            must_use: true,
            ptx_mnemonic: "abs.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "6.5",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.6 Half Precision Floating Point Instructions: abs",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fabs",
                symbol: "llvm.nvvm.fabs",
                resolved_symbol: Some("llvm.nvvm.fabs.v2f16"),
                arguments: &["anonymous_8"],
                results: &["anyfloat"],
                properties: PURE,
                selection: "ABS_F16X2",
                selection_asm: "abs.f16x2 \t$dst, $src0;",
            },
        },
        // These operations were added for the f32x2 admission. Keeping them
        // outside the f16x2 recipe avoids silently expanding this PR's scope,
        // even where a similarly spelled PTX form may exist.
        PackedAluOperation::AddFtz | PackedAluOperation::SubFtz | PackedAluOperation::MulFtz => {
            return None;
        }
    })
}

/// Per-backend PTX and SM floors, which can sit above the recipe's own floor.
///
/// Takes the caller's already-resolved recipe rather than looking it up again,
/// so this cannot be reached with a pair the family rejects.
pub(in crate::resolve) fn packed_alu_backend_floor(
    recipe: &PackedAluRecipe,
    format: PackedAluFormat,
    operation: PackedAluOperation,
    backend: IntrinsicBackend,
) -> (&'static str, &'static str) {
    match (format, operation, backend) {
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::Fma
            | PackedAluOperation::Add
            | PackedAluOperation::Sub
            | PackedAluOperation::Mul,
            IntrinsicBackend::LlvmNvptx,
        ) => ("6.0", "sm_70"),
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::Fma
            | PackedAluOperation::Add
            | PackedAluOperation::Sub
            | PackedAluOperation::Mul,
            IntrinsicBackend::LibNvvm,
        ) => ("4.2", "sm_75"),
        // The ftz and sat fma forms share the plain fma floors: PTX 4.2 natively,
        // but the pinned LLVM backend needs 6.0 and CUDA 13.3 ptxas no longer
        // targets sm_70, so libNVVM is checked at sm_75.
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaFtz | PackedAluOperation::FmaSat | PackedAluOperation::FmaFtzSat,
            IntrinsicBackend::LlvmNvptx,
        ) => ("6.0", "sm_70"),
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaFtz | PackedAluOperation::FmaSat | PackedAluOperation::FmaFtzSat,
            IntrinsicBackend::LibNvvm,
        ) => ("4.2", "sm_75"),
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, IntrinsicBackend::LlvmNvptx) => {
            ("6.0", "sm_70")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, IntrinsicBackend::LibNvvm) => {
            ("6.0", "sm_75")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, IntrinsicBackend::LlvmNvptx) => {
            ("6.5", "sm_70")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, IntrinsicBackend::LibNvvm) => {
            ("6.5", "sm_75")
        }
        _ => (recipe.minimum_ptx, recipe.minimum_sm),
    }
}

pub(in crate::resolve) fn validate_packed_alu_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let packed = policy
        .packed_alu
        .as_ref()
        .with_context(|| format!("{} has no closed packed-ALU contract", policy.id))?;
    let recipe = packed_alu_recipe(packed.format, packed.operation)
        .with_context(|| format!("{} is outside the closed packed-ALU recipe", policy.id))?;
    let (rust_module, rust_type, dialect_type, expected_adapter) = match packed.format {
        PackedAluFormat::Bf16x2 => ("bf16x2", "u32", "i32", PackedAluAdapter::DirectPackedU32),
        PackedAluFormat::F16x2 => ("f16x2", "u32", "i32", PackedAluAdapter::DirectPackedU32),
        PackedAluFormat::F32x2 => ("f32x2", "u64", "i64", PackedAluAdapter::DirectPackedU64),
    };
    ensure!(
        packed.adapter == expected_adapter,
        "{} requests an unsupported packed-ALU adapter",
        policy.id
    );
    let rust_arguments = vec![rust_type; recipe.arity];
    let dialect_operands = vec![dialect_type; recipe.arity];
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} packed-ALU identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == rust_module
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == rust_type
            && policy.safe
            && policy.must_use == recipe.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path
                == format!("cuda_intrinsics::{rust_module}::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::{rust_module}::{}", recipe.rust_name)],
        "{} must preserve its reviewed safe packed-ALU API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results == [dialect_type]
            && policy.lowering == "generated_packed_alu_inline_ptx",
        "{} is outside the closed packed-ALU dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == rust_type
            && policy.targets == "all"
            && packed.native_minimum_sm == recipe.native_minimum_sm,
        "{} packed-ALU effects, carrier, or target floor disagree",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} packed-ALU PTX provenance does not match its reviewed instruction section",
        policy.id
    );
    let expected_operands = vec![OperandPattern::Register; recipe.arity + 1];
    ensure!(
        policy.expected_ptx.mnemonic
            == recipe.ptx_mnemonic.split('.').next().expect("PTX mnemonic")
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its exact packed-ALU instruction",
        policy.id
    );

    match &recipe.source {
        PackedAluRecipeSource::PtxNative => {
            ensure!(
                source
                    == &IntrinsicSource::PtxNative {
                        instruction: recipe.ptx_mnemonic.to_owned(),
                    }
                    && declaration.is_none(),
                "{} packed-ALU source does not match its PTX-native recipe",
                policy.id
            );
        }
        PackedAluRecipeSource::Imported {
            record,
            symbol,
            resolved_symbol,
            arguments,
            results,
            properties,
            selection,
            selection_asm,
        } => {
            let declaration = declaration.context("imported packed ALU has no declaration")?;
            ensure!(
                source
                    == &IntrinsicSource::LlvmImported {
                        source_record: (*record).to_owned(),
                    }
                    && policy.llvm_symbol.as_deref() == Some(*symbol)
                    && policy.resolved_llvm_symbol.as_deref() == *resolved_symbol
                    && policy.llvm_arguments == *arguments
                    && policy.llvm_results == *results,
                "{} packed-ALU LLVM source or signature changed",
                policy.id
            );
            let matching_selections: Vec<_> = declaration
                .selections
                .iter()
                .filter(|candidate| candidate.source_record == *selection)
                .collect();
            let expected_selection_count = if *record == "int_nvvm_fabs" { 6 } else { 1 };
            ensure!(
                declaration.properties == *properties
                    && declaration.selections.len() == expected_selection_count
                    && matching_selections.len() == 1
                    && matching_selections[0].asm == *selection_asm
                    && matching_selections[0].predicates
                        == [
                            format!("Subtarget->getSmVersion() >= {}", recipe.native_minimum_sm),
                            format!(
                                "Subtarget->getPTXVersion() >= {}",
                                recipe.minimum_ptx.replace('.', "")
                            ),
                        ]
                    && matching_selections[0].constraints.is_empty(),
                "{} packed-ALU imported properties or selection changed",
                policy.id
            );
        }
    }
    let llvm_floor = packed_alu_backend_floor(
        &recipe,
        packed.format,
        packed.operation,
        IntrinsicBackend::LlvmNvptx,
    );
    let libnvvm_floor = packed_alu_backend_floor(
        &recipe,
        packed.format,
        packed.operation,
        IntrinsicBackend::LibNvvm,
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                llvm_floor.0,
                Some(llvm_floor.1),
            ),
            (
                IntrinsicBackend::LibNvvm,
                libnvvm_floor.0,
                Some(libnvvm_floor.1),
            ),
        ],
        "packed-ALU",
    )?;
    ensure_no_other_family_contract(policy, "packed-ALU")?;
    Ok(())
}
