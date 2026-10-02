/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ClusterSregAdmission, ImportedIntrinsic, IntrinsicBackend,
    IntrinsicSource, OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, SpecialRegister,
    SpecialRegisterAdmission, SpecialRegisterKind, SpecialRegisterLlvmExclusion,
    SpecialRegisterLlvmExclusionReason, SpecialRegisterObservation,
    SpecialRegisterOutputConstraint, SpecialRegisterPtxType, SpecialRegisterWidth,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;

use crate::resolve::guards::*;

pub(in crate::resolve) const REVIEWED_SPECIAL_REGISTERS: [SpecialRegisterKind; 12] = [
    SpecialRegisterKind::Clock,
    SpecialRegisterKind::Clock64,
    SpecialRegisterKind::Globaltimer,
    SpecialRegisterKind::Envreg1,
    SpecialRegisterKind::Envreg2,
    SpecialRegisterKind::Smid,
    SpecialRegisterKind::Nsmid,
    SpecialRegisterKind::Gridid,
    SpecialRegisterKind::Warpid,
    SpecialRegisterKind::Nwarpid,
    SpecialRegisterKind::DynamicSmemSize,
    SpecialRegisterKind::TotalSmemSize,
];

#[derive(Clone, Copy)]
pub(in crate::resolve) struct SpecialRegisterRecipe {
    kind: SpecialRegisterKind,
    id: &'static str,
    operation_key: &'static str,
    source_record: Option<&'static str>,
    llvm_symbol: Option<&'static str>,
    rust_module: &'static str,
    compatibility_paths: &'static [&'static str],
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    register_spelling: &'static str,
    observation: SpecialRegisterObservation,
    result_width: SpecialRegisterWidth,
    ptx_type: SpecialRegisterPtxType,
    output_constraint: SpecialRegisterOutputConstraint,
    llvm_mechanism: BackendLoweringMechanism,
    libnvvm_mechanism: BackendLoweringMechanism,
    minimum_ptx: &'static str,
    minimum_sm: Option<&'static str>,
    execution_scope: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    selection_record: Option<&'static str>,
    selection_asm: Option<&'static str>,
    summary: &'static str,
}

pub(in crate::resolve) fn special_register_recipe(
    kind: SpecialRegisterKind,
) -> SpecialRegisterRecipe {
    use BackendLoweringMechanism::{InlinePtx, TypedNvvm};
    use SpecialRegisterKind::*;
    use SpecialRegisterObservation::{StablePure, VolatileObservation};
    use SpecialRegisterOutputConstraint::{Register32, Register64};
    use SpecialRegisterPtxType::{B32, U32, U64};
    use SpecialRegisterWidth::{B32 as Width32, B64 as Width64};

    match kind {
        Clock => SpecialRegisterRecipe {
            kind,
            id: "clock",
            operation_key: "debug.clock",
            source_record: Some("int_nvvm_read_ptx_sreg_clock"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.clock"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::clock"],
            dialect_op_type: "ReadPtxSregClockOp",
            dialect_op_name: "nvvm.read_ptx_sreg_clock",
            register_spelling: "%clock",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "1.0",
            minimum_sm: None,
            execution_scope: "sm",
            ptx_isa_section: "10.23 Special Registers: %clock, %clock_hi",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock-clock-hi",
            selection_record: Some("SREG_CLOCK"),
            selection_asm: Some("mov.u32 \t$d, %clock;"),
            summary: "Samples the current SM's 32-bit clock counter.",
        },
        Clock64 => SpecialRegisterRecipe {
            kind,
            id: "clock64",
            operation_key: "debug.clock64",
            source_record: Some("int_nvvm_read_ptx_sreg_clock64"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.clock64"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::clock64"],
            dialect_op_type: "ReadPtxSregClock64Op",
            dialect_op_name: "nvvm.read_ptx_sreg_clock64",
            register_spelling: "%clock64",
            observation: VolatileObservation,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "sm",
            ptx_isa_section: "10.24 Special Registers: %clock64",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64",
            selection_record: Some("SREG_CLOCK64"),
            selection_asm: Some("mov.u64 \t$d, %clock64;"),
            summary: "Samples the current SM's 64-bit clock counter.",
        },
        Globaltimer => SpecialRegisterRecipe {
            kind,
            id: "globaltimer",
            operation_key: "debug.global_timer",
            source_record: Some("int_nvvm_read_ptx_sreg_globaltimer"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.globaltimer"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::globaltimer"],
            dialect_op_type: "ReadPtxSregGlobaltimerOp",
            dialect_op_name: "nvvm.read_ptx_sreg_globaltimer",
            register_spelling: "%globaltimer",
            observation: VolatileObservation,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "3.1",
            minimum_sm: Some("sm_30"),
            execution_scope: "device",
            ptx_isa_section: "10.28 Special Registers: %globaltimer, %globaltimer_lo, %globaltimer_hi",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer",
            selection_record: Some("SREG_GLOBALTIMER"),
            selection_asm: Some("mov.u64 \t$d, %globaltimer;"),
            summary: "Samples the device-wide 64-bit global timer.",
        },
        Envreg1 => SpecialRegisterRecipe {
            kind,
            id: "envreg1",
            operation_key: "grid.environment_register.1",
            source_record: Some("int_nvvm_read_ptx_sreg_envreg1"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.envreg1"),
            rust_module: "grid",
            compatibility_paths: &["cuda_device::grid::envreg1"],
            dialect_op_type: "ReadPtxSregEnvReg1Op",
            dialect_op_name: "nvvm.read_ptx_sreg_envreg1",
            register_spelling: "%envreg1",
            observation: StablePure,
            result_width: Width32,
            ptx_type: B32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.1",
            minimum_sm: None,
            execution_scope: "grid",
            ptx_isa_section: "10.27 Special Registers: %envreg<32>",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-envreg",
            selection_record: None,
            selection_asm: None,
            summary: "Reads PTX environment register 1.",
        },
        Envreg2 => SpecialRegisterRecipe {
            kind,
            id: "envreg2",
            operation_key: "grid.environment_register.2",
            source_record: Some("int_nvvm_read_ptx_sreg_envreg2"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.envreg2"),
            rust_module: "grid",
            compatibility_paths: &["cuda_device::grid::envreg2"],
            dialect_op_type: "ReadPtxSregEnvReg2Op",
            dialect_op_name: "nvvm.read_ptx_sreg_envreg2",
            register_spelling: "%envreg2",
            observation: StablePure,
            result_width: Width32,
            ptx_type: B32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.1",
            minimum_sm: None,
            execution_scope: "grid",
            ptx_isa_section: "10.27 Special Registers: %envreg<32>",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-envreg",
            selection_record: None,
            selection_asm: None,
            summary: "Reads PTX environment register 2.",
        },
        Smid => SpecialRegisterRecipe {
            kind,
            id: "smid",
            operation_key: "execution.sm_identifier",
            source_record: Some("int_nvvm_read_ptx_sreg_smid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.smid"),
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::smid", "cuda_device::smid"],
            dialect_op_type: "ReadPtxSregSmIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_smid",
            register_spelling: "%smid",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "1.3",
            minimum_sm: None,
            execution_scope: "thread",
            ptx_isa_section: "10.8 Special Registers: %smid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-smid",
            selection_record: Some("SREG_SMID"),
            selection_asm: Some("mov.u32 \t$d, %smid;"),
            summary: "Samples the SM currently executing this thread.",
        },
        Nsmid => SpecialRegisterRecipe {
            kind,
            id: "nsmid",
            operation_key: "execution.sm_identifier_bound",
            source_record: Some("int_nvvm_read_ptx_sreg_nsmid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.nsmid"),
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::nsmid", "cuda_device::nsmid"],
            dialect_op_type: "ReadPtxSregNsmIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_nsmid",
            register_spelling: "%nsmid",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "device",
            ptx_isa_section: "10.9 Special Registers: %nsmid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid",
            selection_record: Some("SREG_NSMID"),
            selection_asm: Some("mov.u32 \t$d, %nsmid;"),
            summary: "Returns the upper bound for SM identifiers.",
        },
        Gridid => SpecialRegisterRecipe {
            kind,
            id: "gridid",
            operation_key: "launch.grid_identifier",
            source_record: None,
            llvm_symbol: None,
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::gridid", "cuda_device::gridid"],
            dialect_op_type: "ReadPtxSregGridIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_gridid",
            register_spelling: "%gridid",
            observation: StablePure,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "3.0",
            minimum_sm: Some("sm_30"),
            execution_scope: "grid",
            ptx_isa_section: "10.10 Special Registers: %gridid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-gridid",
            selection_record: None,
            selection_asm: None,
            summary: "Returns the full 64-bit temporal grid identifier.",
        },
        Warpid => SpecialRegisterRecipe {
            kind,
            id: "warpid",
            operation_key: "warp.hardware_identifier",
            source_record: Some("int_nvvm_read_ptx_sreg_warpid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.warpid"),
            rust_module: "warp",
            compatibility_paths: &["cuda_device::warp::warpid"],
            dialect_op_type: "ReadPtxSregWarpIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_warpid",
            register_spelling: "%warpid",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "1.3",
            minimum_sm: None,
            execution_scope: "cta",
            ptx_isa_section: "10.4 Special Registers: %warpid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-warpid",
            selection_record: Some("SREG_WARPID"),
            selection_asm: Some("mov.u32 \t$d, %warpid;"),
            summary: "Samples the hardware warp currently executing this thread.",
        },
        Nwarpid => SpecialRegisterRecipe {
            kind,
            id: "nwarpid",
            operation_key: "warp.hardware_identifier_bound",
            source_record: Some("int_nvvm_read_ptx_sreg_nwarpid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.nwarpid"),
            rust_module: "warp",
            compatibility_paths: &["cuda_device::warp::nwarpid"],
            dialect_op_type: "ReadPtxSregNwarpIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_nwarpid",
            register_spelling: "%nwarpid",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.5 Special Registers: %nwarpid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nwarpid",
            selection_record: Some("SREG_NWARPID"),
            selection_asm: Some("mov.u32 \t$d, %nwarpid;"),
            summary: "Returns the upper bound for hardware warp identifiers.",
        },
        DynamicSmemSize => SpecialRegisterRecipe {
            kind,
            id: "dynamic_smem_size",
            operation_key: "shared.dynamic_size",
            source_record: Some("int_nvvm_read_ptx_sreg_dynamic_smem_size"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.dynamic_smem_size"),
            rust_module: "shared",
            compatibility_paths: &["cuda_device::shared::dynamic_smem_size"],
            dialect_op_type: "ReadPtxSregDynamicSmemSizeOp",
            dialect_op_name: "nvvm.read_ptx_sreg_dynamic_smem_size",
            register_spelling: "%dynamic_smem_size",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "4.1",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.32 Special Registers: %dynamic_smem_size",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size",
            selection_record: Some("INT_PTX_SREG_DYNAMIC_SMEM_SIZE"),
            selection_asm: Some("mov.u32 \t$d, %dynamic_smem_size;"),
            summary: "Returns the launch-time dynamic shared-memory size in bytes.",
        },
        TotalSmemSize => SpecialRegisterRecipe {
            kind,
            id: "total_smem_size",
            operation_key: "shared.total_size",
            source_record: Some("int_nvvm_read_ptx_sreg_total_smem_size"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.total_smem_size"),
            rust_module: "shared",
            compatibility_paths: &["cuda_device::shared::total_smem_size"],
            dialect_op_type: "ReadPtxSregTotalSmemSizeOp",
            dialect_op_name: "nvvm.read_ptx_sreg_total_smem_size",
            register_spelling: "%total_smem_size",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "4.1",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.30 Special Registers: %total_smem_size",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-total-smem-size",
            selection_record: Some("INT_PTX_SREG_TOTAL_SMEM_SIZE"),
            selection_asm: Some("mov.u32 \t$d, %total_smem_size;"),
            summary: "Returns the total user shared-memory allocation in bytes.",
        },
    }
}

pub(in crate::resolve) fn special_register_ptx_type(
    ptx_type: SpecialRegisterPtxType,
) -> &'static str {
    match ptx_type {
        SpecialRegisterPtxType::B32 => "b32",
        SpecialRegisterPtxType::U32 => "u32",
        SpecialRegisterPtxType::U64 => "u64",
    }
}

pub(in crate::resolve) fn special_register_backend_floor(
    recipe: SpecialRegisterRecipe,
    backend: IntrinsicBackend,
) -> (Option<&'static str>, Option<&'static str>) {
    match backend {
        IntrinsicBackend::LlvmNvptx => {
            let minimum_ptx = if matches!(recipe.minimum_ptx, "4.1") {
                "4.1"
            } else {
                "3.2"
            };
            let minimum_sm = if recipe.minimum_sm == Some("sm_30") {
                "sm_30"
            } else {
                "sm_20"
            };
            (Some(minimum_ptx), Some(minimum_sm))
        }
        IntrinsicBackend::LibNvvm => (None, Some("sm_75")),
    }
}

pub(in crate::resolve) fn special_register_contract(
    recipe: SpecialRegisterRecipe,
) -> SpecialRegister {
    let llvm_exclusion =
        (recipe.kind == SpecialRegisterKind::Gridid).then(|| SpecialRegisterLlvmExclusion {
            source_record: "int_nvvm_read_ptx_sreg_gridid".into(),
            llvm_symbol: "llvm.nvvm.read.ptx.sreg.gridid".into(),
            imported_result_width: SpecialRegisterWidth::B32,
            reason: SpecialRegisterLlvmExclusionReason::ResultWidthMismatch,
        });
    SpecialRegister {
        register: recipe.kind,
        observation: recipe.observation,
        result_width: recipe.result_width,
        ptx_type: recipe.ptx_type,
        output_constraint: recipe.output_constraint,
        llvm_exclusion,
    }
}

pub(in crate::resolve) fn expand_special_register_admission(
    admission: &SpecialRegisterAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "special-register runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact special-register admission requires both backend evidence profiles"
    );
    ensure!(
        admission.registers == REVIEWED_SPECIAL_REGISTERS
            && admission.product_count == REVIEWED_SPECIAL_REGISTERS.len(),
        "compact special-register admission must list the canonical 12 registers exactly once and in order"
    );

    admission
        .registers
        .iter()
        .copied()
        .map(|kind| {
            let recipe = special_register_recipe(kind);
            let width = recipe.result_width.bits();
            let rust_result = format!("u{width}");
            let dialect_result = format!("i{width}");
            let source = match recipe.source_record {
                Some(_) => None,
                None => Some(IntrinsicSource::PtxNative {
                    instruction: format!(
                        "mov.{} {}",
                        special_register_ptx_type(recipe.ptx_type),
                        recipe.register_spelling
                    ),
                }),
            };
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: String::new(),
                operation_key: recipe.operation_key.into(),
                family: "sreg".into(),
                source,
                source_record: recipe.source_record.map(str::to_owned),
                rust_module: recipe.rust_module.into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: rust_result.clone(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    "reading this special register has no caller obligations.".into(),
                ),
                public_rust_path: format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id),
                compatibility_rust_paths: recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).into())
                    .collect(),
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![dialect_result.clone()],
                llvm_symbol: recipe.llvm_symbol.map(str::to_owned),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: recipe
                    .source_record
                    .map(|_| dialect_result)
                    .into_iter()
                    .collect(),
                pure: recipe.observation == SpecialRegisterObservation::StablePure,
                memory: if matches!(
                    recipe.kind,
                    SpecialRegisterKind::Clock
                        | SpecialRegisterKind::Clock64
                        | SpecialRegisterKind::Globaltimer
                ) {
                    "inaccessible_read_write".into()
                } else {
                    "none".into()
                },
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(str::to_owned),
                ptx_result: rust_result,
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: recipe.ptx_isa_url.into(),
                lowering: "generated_special_register".into(),
                backend_lowerings: [
                    (
                        IntrinsicBackend::LlvmNvptx,
                        recipe.llvm_mechanism,
                        &admission.llvm_evidence_profile,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        recipe.libnvvm_mechanism,
                        &admission.libnvvm_evidence_profile,
                    ),
                ]
                .into_iter()
                .map(
                    |(backend, mechanism, evidence_profile)| OverlayBackendLowering {
                        minimum_ptx: special_register_backend_floor(recipe, backend)
                            .0
                            .map(str::to_owned),
                        minimum_sm: special_register_backend_floor(recipe, backend)
                            .1
                            .map(str::to_owned),
                        backend,
                        mechanism,
                        evidence_profile: evidence_profile.clone(),
                        targets: None,
                    },
                )
                .collect(),
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                integer_minmax: None,
                packed_conversion: None,
                scalar_conversion: None,
                scalar_arithmetic: None,
                scalar_math: None,
                extended_minmax: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                wgmma_control: None,
                special_register: Some(special_register_contract(recipe)),
                debug_control: None,
                cluster_memory: None,
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "mov".into(),
                    modifiers: vec![special_register_ptx_type(recipe.ptx_type).into()],
                    operands: vec![
                        OperandPattern::Register,
                        OperandPattern::Exact {
                            value: recipe.register_spelling.into(),
                        },
                    ],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_special_register_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let special = policy
        .special_register
        .as_ref()
        .with_context(|| format!("{} has no closed special-register contract", policy.id))?;
    let recipe = special_register_recipe(special.register);
    ensure!(
        special == &special_register_contract(recipe),
        "{} special-register width, PTX type, constraint, observation, or LLVM exclusion changed",
        policy.id
    );
    let expected_source = match recipe.source_record {
        Some(source_record) => IntrinsicSource::LlvmImported {
            source_record: source_record.into(),
        },
        None => IntrinsicSource::PtxNative {
            instruction: format!(
                "mov.{} {}",
                special_register_ptx_type(recipe.ptx_type),
                recipe.register_spelling
            ),
        },
    };
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && source == &expected_source
            && policy.source_record.as_deref() == recipe.source_record
            && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments.is_empty(),
        "{} special-register identity or source changed",
        policy.id
    );
    let width = recipe.result_width.bits();
    let rust_result = format!("u{width}");
    let dialect_result = format!("i{width}");
    let expected_llvm_results = recipe
        .source_record
        .map(|_| dialect_result.clone())
        .into_iter()
        .collect::<Vec<_>>();
    ensure!(
        policy.rust_module == recipe.rust_module
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == rust_result
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path
                == format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id)
            && policy.compatibility_rust_paths
                == recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).to_owned())
                    .collect::<Vec<_>>(),
        "{} special-register Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results == [dialect_result.as_str()]
            && policy.llvm_results == expected_llvm_results
            && policy.lowering == "generated_special_register",
        "{} special-register carrier, result width, or lowering changed",
        policy.id
    );
    let expected_pure = recipe.observation == SpecialRegisterObservation::StablePure;
    let expected_memory = if matches!(
        recipe.kind,
        SpecialRegisterKind::Clock
            | SpecialRegisterKind::Clock64
            | SpecialRegisterKind::Globaltimer
    ) {
        "inaccessible_read_write"
    } else {
        "none"
    };
    ensure!(
        policy.pure == expected_pure
            && policy.memory == expected_memory
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} special-register observation or effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.ptx_result == rust_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} special-register target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "mov".into(),
                modifiers: vec![special_register_ptx_type(recipe.ptx_type).into()],
                operands: vec![
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: recipe.register_spelling.into(),
                    },
                ],
            },
        "{} special-register PTX shape changed",
        policy.id
    );
    let expected_routes = [
        (
            IntrinsicBackend::LlvmNvptx,
            recipe.llvm_mechanism,
            special_register_backend_floor(recipe, IntrinsicBackend::LlvmNvptx),
        ),
        (
            IntrinsicBackend::LibNvvm,
            recipe.libnvvm_mechanism,
            special_register_backend_floor(recipe, IntrinsicBackend::LibNvvm),
        ),
    ];
    ensure!(
        policy.backend_lowerings.len() == expected_routes.len()
            && policy.backend_lowerings.iter().zip(expected_routes).all(
                |(actual, (backend, mechanism, (minimum_ptx, minimum_sm)))| {
                    actual.backend == backend
                        && actual.mechanism == mechanism
                        && !actual.evidence_profile.trim().is_empty()
                        && actual.minimum_ptx.as_deref() == minimum_ptx
                        && actual.minimum_sm.as_deref() == minimum_sm
                }
            ),
        "{} special-register backend routes changed",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
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
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with a special register",
        policy.id
    );

    match (recipe.source_record, declaration) {
        (None, None) => {}
        (Some(_), Some(declaration)) => {
            let timer = matches!(
                recipe.kind,
                SpecialRegisterKind::Clock
                    | SpecialRegisterKind::Clock64
                    | SpecialRegisterKind::Globaltimer
            );
            let expected_properties: &[&str] = if timer {
                &[
                    "IntrInaccessibleMemOnly",
                    "IntrNoCallback",
                    "IntrNoFree",
                    "IntrWillReturn",
                    "NoUndef<ret>",
                ]
            } else {
                &["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"]
            };
            ensure!(
                declaration.arguments.is_empty()
                    && declaration.results == [dialect_result.as_str()]
                    && declaration.properties
                        == expected_properties
                            .iter()
                            .map(|property| (*property).to_owned())
                            .collect::<Vec<_>>()
                    && if timer {
                        declaration
                            .classes
                            .iter()
                            .any(|class| class == "PTXReadNCSRegIntrinsic")
                    } else {
                        declaration
                            .classes
                            .iter()
                            .any(|class| class == "NVVMPureIntrinsic")
                    },
                "{} imported special-register signature, class, or properties changed",
                policy.id
            );
            match (recipe.selection_record, recipe.selection_asm) {
                (None, None) => ensure!(
                    declaration.selections.is_empty(),
                    "{} selectionless environment-register contract changed",
                    policy.id
                ),
                (Some(selection_record), Some(selection_asm)) => ensure!(
                    declaration.selections.len() == 1
                        && declaration.selections[0].source_record == selection_record
                        && declaration.selections[0].asm == selection_asm
                        && declaration.selections[0].predicates.is_empty()
                        && declaration.selections[0].constraints.is_empty(),
                    "{} imported special-register selection changed",
                    policy.id
                ),
                _ => unreachable!("closed special-register selection recipe"),
            }
        }
        _ => bail!(
            "{} special-register source and imported declaration disagree",
            policy.id
        ),
    }
    Ok(())
}

pub(in crate::resolve) fn validate_special_register_llvm_exclusion(
    policy: &OverlayIntrinsic,
    imported_by_record: &BTreeMap<&str, &ImportedIntrinsic>,
) -> Result<()> {
    let Some(exclusion) = policy
        .special_register
        .as_ref()
        .and_then(|special| special.llvm_exclusion.as_ref())
    else {
        return Ok(());
    };
    let declaration = imported_by_record
        .get(exclusion.source_record.as_str())
        .with_context(|| {
            format!(
                "{} excludes missing imported LLVM record {}",
                policy.id, exclusion.source_record
            )
        })?;
    ensure!(
        policy.id == "gridid"
            && exclusion.reason == SpecialRegisterLlvmExclusionReason::ResultWidthMismatch
            && exclusion.imported_result_width == SpecialRegisterWidth::B32
            && declaration.llvm_name == exclusion.llvm_symbol
            && declaration.arguments.is_empty()
            && declaration.results == ["i32"]
            && declaration.properties == ["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"]
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == "SREG_GRIDID"
            && declaration.selections[0].asm == "mov.u32 \t$d, %gridid;"
            && declaration.selections[0].predicates.is_empty()
            && declaration.selections[0].constraints.is_empty()
            && policy.rust_result == "u64"
            && policy.dialect_results == ["i64"],
        "{} LLVM exclusion no longer proves the reviewed i32-to-u64 gridid width mismatch",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_sreg_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    if policy.special_register.is_some() {
        return validate_special_register_policy(policy, source, declaration);
    }
    let declaration = declaration.context("sreg requires imported LLVM declaration")?;
    ensure!(
        policy.rust_arguments.is_empty() && policy.llvm_arguments.is_empty(),
        "{} is not a zero-operand intrinsic; the sreg recipe cannot lower it",
        policy.id
    );
    ensure!(
        matches!(policy.rust_result.as_str(), "u32" | "u64"),
        "{} has unsupported raw scalar result {}",
        policy.id,
        policy.rust_result
    );
    let expected_llvm_result = match policy.rust_result.as_str() {
        "u32" => "i32",
        "u64" => "i64",
        _ => unreachable!(),
    };
    ensure!(
        policy.llvm_results == [expected_llvm_result]
            && policy.ptx_result == policy.rust_result
            && policy.lowering == "direct_nvvm",
        "{} has a signature or lowering outside the scalar direct-NVVM sreg recipe",
        policy.id
    );
    ensure!(
        policy.resolved_llvm_symbol.is_none()
            && policy.backend_lowerings.is_empty()
            && policy.special_register.is_none(),
        "{} uses a backend contract outside the direct-NVVM sreg recipe",
        policy.id
    );
    ensure_no_other_family_contract(policy, "sreg")?;
    if policy.id.starts_with("lanemask_") {
        validate_lanemask_policy(policy, declaration)?;
    }
    if is_cluster_sreg_source(&declaration.source_record) {
        validate_cluster_sreg_policy(policy, declaration)?;
    }
    Ok(())
}

pub(in crate::resolve) fn is_cluster_sreg_source(source_record: &str) -> bool {
    source_record == "int_nvvm_read_ptx_sreg_cluster_ctarank"
        || source_record == "int_nvvm_read_ptx_sreg_cluster_nctarank"
        || source_record.starts_with("int_nvvm_read_ptx_sreg_cluster_ctaid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_cluster_nctaid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_clusterid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_nclusterid_")
}

#[derive(Clone)]
pub(in crate::resolve) struct ClusterSregRecipe {
    id: String,
    operation_key: String,
    source_suffix: String,
    llvm_suffix: String,
    selection_record: String,
    ptx_register: String,
    compatibility_path: Option<String>,
    op_type: String,
    scope: &'static str,
    section: &'static str,
    anchor: &'static str,
    range: Option<&'static str>,
    safe_reason: String,
    summary: String,
}

#[derive(Clone, Copy)]
pub(in crate::resolve) struct ClusterSregXyzFamilyRecipe {
    id_prefix: &'static str,
    operation_key_prefix: &'static str,
    source_prefix: &'static str,
    llvm_prefix: &'static str,
    selection_prefix: &'static str,
    ptx_prefix: &'static str,
    compatibility_prefix: Option<&'static str>,
    op_type_prefix: &'static str,
    scope: &'static str,
    section: &'static str,
    anchor: &'static str,
    x_range: &'static str,
    yz_range: &'static str,
    safe_reason: &'static str,
    summary: &'static str,
}

pub(in crate::resolve) const CLUSTER_SREG_AXES: [&str; 3] = ["x", "y", "z"];

pub(in crate::resolve) const CLUSTER_SREG_XYZ_FAMILIES: [ClusterSregXyzFamilyRecipe; 4] = [
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_block_idx",
        operation_key_prefix: "launch.cluster.block_index",
        source_prefix: "cluster_ctaid_",
        llvm_prefix: "cluster.ctaid.",
        selection_prefix: "INT_PTX_SREG_CLUSTER_CTAID_",
        ptx_prefix: "%cluster_ctaid.",
        compatibility_prefix: Some("cuda_device::cluster::cluster_ctaid"),
        op_type_prefix: "ReadPtxSregClusterCtaid",
        scope: "cta",
        section: "10.14 Special Registers: %cluster_ctaid",
        anchor: "cluster-ctaid",
        x_range: "Range<ret,0,2147483647>",
        yz_range: "Range<ret,0,65535>",
        safe_reason: "reading the read-only block index within its cluster has no caller obligations",
        summary: "Returns the block's {axis} index within its thread block cluster.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_dim",
        operation_key_prefix: "launch.cluster.dimension",
        source_prefix: "cluster_nctaid_",
        llvm_prefix: "cluster.nctaid.",
        selection_prefix: "INT_PTX_SREG_CLUSTER_NCTAID_",
        ptx_prefix: "%cluster_nctaid.",
        compatibility_prefix: Some("cuda_device::cluster::cluster_nctaid"),
        op_type_prefix: "ReadPtxSregClusterNctaid",
        scope: "cluster",
        section: "10.15 Special Registers: %cluster_nctaid",
        anchor: "cluster-nctaid",
        x_range: "Range<ret,1,2147483648>",
        yz_range: "Range<ret,1,65536>",
        safe_reason: "reading the read-only cluster dimension has no caller obligations",
        summary: "Returns the number of blocks in the cluster's {axis} dimension.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_idx",
        operation_key_prefix: "launch.cluster.index",
        source_prefix: "clusterid_",
        llvm_prefix: "clusterid.",
        selection_prefix: "INT_PTX_SREG_CLUSTERID_",
        ptx_prefix: "%clusterid.",
        compatibility_prefix: Some("cuda_device::cluster::__cluster_idx"),
        op_type_prefix: "ReadPtxSregClusterId",
        scope: "cluster",
        section: "10.12 Special Registers: %clusterid",
        anchor: "clusterid",
        x_range: "Range<ret,0,2147483647>",
        yz_range: "Range<ret,0,65535>",
        safe_reason: "reading the read-only cluster index has no caller obligations",
        summary: "Returns the cluster's {axis} index within the grid.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_grid_dim",
        operation_key_prefix: "launch.cluster.grid_dimension",
        source_prefix: "nclusterid_",
        llvm_prefix: "nclusterid.",
        selection_prefix: "INT_PTX_SREG_NCLUSTERID_",
        ptx_prefix: "%nclusterid.",
        compatibility_prefix: Some("cuda_device::cluster::__cluster_grid_dim"),
        op_type_prefix: "ReadPtxSregNclusterId",
        scope: "grid",
        section: "10.13 Special Registers: %nclusterid",
        anchor: "nclusterid",
        x_range: "Range<ret,1,2147483648>",
        yz_range: "Range<ret,1,65536>",
        safe_reason: "reading the read-only cluster-grid dimension has no caller obligations",
        summary: "Returns the number of clusters in the grid's {axis} dimension.",
    },
];

pub(in crate::resolve) fn cluster_sreg_recipes() -> Vec<ClusterSregRecipe> {
    let mut recipes = Vec::with_capacity(14);
    for family in CLUSTER_SREG_XYZ_FAMILIES {
        for axis in CLUSTER_SREG_AXES {
            let axis_upper = axis.to_ascii_uppercase();
            recipes.push(ClusterSregRecipe {
                id: format!("{}_{axis}", family.id_prefix),
                operation_key: format!("{}.{axis}", family.operation_key_prefix),
                source_suffix: format!("{}{axis}", family.source_prefix),
                llvm_suffix: format!("{}{axis}", family.llvm_prefix),
                selection_record: format!("{}{axis}", family.selection_prefix),
                ptx_register: format!("{}{axis}", family.ptx_prefix),
                compatibility_path: family
                    .compatibility_prefix
                    .map(|prefix| format!("{prefix}{axis_upper}")),
                op_type: format!("{}{axis_upper}Op", family.op_type_prefix),
                scope: family.scope,
                section: family.section,
                anchor: family.anchor,
                range: Some(if axis == "x" {
                    family.x_range
                } else {
                    family.yz_range
                }),
                safe_reason: family.safe_reason.into(),
                summary: family.summary.replace("{axis}", &axis_upper),
            });
        }
    }
    recipes.extend([
        ClusterSregRecipe {
            id: "cluster_block_rank".into(),
            operation_key: "launch.cluster.block_rank".into(),
            source_suffix: "cluster_ctarank".into(),
            llvm_suffix: "cluster.ctarank".into(),
            selection_record: "INT_PTX_SREG_CLUSTER_CTARANK".into(),
            ptx_register: "%cluster_ctarank".into(),
            compatibility_path: None,
            op_type: "ReadPtxSregClusterCtarankOp".into(),
            scope: "cta",
            section: "10.16 Special Registers: %cluster_ctarank",
            anchor: "cluster-ctarank",
            range: None,
            safe_reason:
                "reading the read-only block rank within its cluster has no caller obligations"
                    .into(),
            summary: "Returns the block's linear rank within its thread block cluster.".into(),
        },
        ClusterSregRecipe {
            id: "cluster_block_count".into(),
            operation_key: "launch.cluster.block_count".into(),
            source_suffix: "cluster_nctarank".into(),
            llvm_suffix: "cluster.nctarank".into(),
            selection_record: "INT_PTX_SREG_CLUSTER_NCTARANK".into(),
            ptx_register: "%cluster_nctarank".into(),
            compatibility_path: None,
            op_type: "ReadPtxSregClusterNctarankOp".into(),
            scope: "cluster",
            section: "10.17 Special Registers: %cluster_nctarank",
            anchor: "cluster-nctarank",
            range: None,
            safe_reason: "reading the read-only block count has no caller obligations".into(),
            summary: "Returns the total number of blocks in the thread block cluster.".into(),
        },
    ]);
    recipes
}

pub(in crate::resolve) fn expand_cluster_sreg_admission(
    admission: &ClusterSregAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.axes == CLUSTER_SREG_AXES,
        "cluster-sreg axes must be exactly x, y, z"
    );
    ensure!(
        admission.xyz_product_count == 12 && admission.record_count == 14,
        "cluster-sreg admission must expand to 12 xyz and 14 total records"
    );
    let recipes = cluster_sreg_recipes();
    ensure!(
        recipes.len() == admission.record_count,
        "cluster-sreg recipe count disagrees with its admission"
    );
    Ok(recipes.into_iter().map(cluster_sreg_policy).collect())
}

pub(in crate::resolve) fn cluster_sreg_policy(recipe: ClusterSregRecipe) -> OverlayIntrinsic {
    let compatibility_rust_paths = recipe.compatibility_path.iter().cloned().collect();
    OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: String::new(),
        operation_key: recipe.operation_key,
        family: "sreg".into(),
        source: None,
        source_record: Some(format!("int_nvvm_read_ptx_sreg_{}", recipe.source_suffix)),
        rust_module: "sreg".into(),
        rust_name: recipe.id.clone(),
        rust_arguments: vec![],
        rust_result: "u32".into(),
        safe: true,
        must_use: false,
        safe_allowlist_reason: Some(recipe.safe_reason),
        public_rust_path: format!("cuda_intrinsics::sreg::{}", recipe.id),
        compatibility_rust_paths,
        dialect_op_type: recipe.op_type,
        dialect_op_name: format!("nvvm.read_ptx_sreg_{}", recipe.source_suffix),
        dialect_operands: vec![],
        dialect_results: vec![],
        llvm_symbol: Some(format!("llvm.nvvm.read.ptx.sreg.{}", recipe.llvm_suffix)),
        resolved_llvm_symbol: None,
        llvm_arguments: vec![],
        llvm_results: vec!["i32".into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: recipe.scope.into(),
        minimum_ptx: "7.8".into(),
        minimum_sm: Some("sm_90".into()),
        ptx_result: "u32".into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: recipe.section.into(),
        ptx_isa_url: format!(
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-{}",
            recipe.anchor
        ),
        lowering: "direct_nvvm".into(),
        backend_lowerings: vec![],
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        integer_minmax: None,
        packed_conversion: None,
        scalar_conversion: None,
        scalar_arithmetic: None,
        scalar_math: None,
        extended_minmax: None,
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: None,
        prmt: None,
        cluster_barrier: None,
        wgmma_control: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        tma: None,
        tcgen05: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "mov".into(),
            modifiers: vec!["u32".into()],
            operands: vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: recipe.ptx_register,
                },
            ],
        },
        summary: recipe.summary,
    }
}

pub(in crate::resolve) fn validate_cluster_sreg_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    ensure!(
        !declaration.source_record.ends_with("_w"),
        "{} selects an unused always-zero fourth cluster-register component",
        policy.id
    );
    let recipe = cluster_sreg_recipes()
        .into_iter()
        .find(|recipe| recipe.id == policy.id)
        .with_context(|| format!("{} is not a reviewed cluster special register", policy.id))?;

    let source_record = format!("int_nvvm_read_ptx_sreg_{}", recipe.source_suffix);
    let llvm_symbol = format!("llvm.nvvm.read.ptx.sreg.{}", recipe.llvm_suffix);
    let compatibility_paths = recipe
        .compatibility_path
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} cluster-register identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "sreg"
            && policy.rust_name == policy.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && !policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::sreg::{}", policy.id)
            && policy.compatibility_rust_paths == compatibility_paths,
        "{} must preserve its reviewed raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.op_type
            && policy.dialect_op_name == format!("nvvm.read_ptx_sreg_{}", recipe.source_suffix)
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "direct_nvvm",
        "{} is outside the closed cluster-register lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == recipe.scope
            && policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} cluster-register effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-{}",
                    recipe.anchor
                ),
        "{} cluster-register PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    let mut properties = vec!["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"];
    properties.extend(recipe.range);
    ensure!(
        declaration.arguments.is_empty()
            && declaration.results == ["i32"]
            && declaration.classes
                == [
                    "SDPatternOperator",
                    "Intrinsic",
                    "DefaultAttrsIntrinsic",
                    "NVVMPureIntrinsic",
                    "PTXReadSRegIntrinsicNB_r32",
                ]
            && declaration.properties == properties,
        "{} declaration shape or properties disagree with LLVM TableGen",
        policy.id
    );
    let [selection] = declaration.selections.as_slice() else {
        bail!("{} must have exactly one LLVM selection", policy.id);
    };
    ensure!(
        selection.source_record == recipe.selection_record
            && selection.asm == format!("mov.u32 \t$d, {};", recipe.ptx_register)
            && selection.predicates
                == [
                    "Subtarget->getSmVersion() >= 90",
                    "Subtarget->getPTXVersion() >= 78",
                ]
            && selection.constraints.address_space.is_none()
            && selection.constraints.immediate_bindings.is_empty(),
        "{} selector disagrees with LLVM TableGen",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mov"
            && policy.expected_ptx.modifiers == ["u32"]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: recipe.ptx_register,
                    },
                ],
        "{} expected PTX does not match its cluster register",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_lanemask_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let (suffix, abi_id, section, op_type) = match policy.id.as_str() {
        "lanemask_lt" => ("lt", "i0035", "10.13", "ReadPtxSregLanemaskLtOp"),
        "lanemask_le" => ("le", "i0036", "10.12", "ReadPtxSregLanemaskLeOp"),
        "lanemask_eq" => ("eq", "i0037", "10.11", "ReadPtxSregLanemaskEqOp"),
        "lanemask_ge" => ("ge", "i0038", "10.14", "ReadPtxSregLanemaskGeOp"),
        "lanemask_gt" => ("gt", "i0039", "10.15", "ReadPtxSregLanemaskGtOp"),
        _ => bail!("{} is not a reviewed lane-mask special register", policy.id),
    };
    ensure!(
        policy.abi_id == abi_id
            && policy.operation_key == format!("warp.lane_mask.{suffix}")
            && policy.source.is_none()
            && policy.source_record.as_deref()
                == Some(format!("int_nvvm_read_ptx_sreg_lanemask_{suffix}").as_str())
            && policy.llvm_symbol.as_deref()
                == Some(format!("llvm.nvvm.read.ptx.sreg.lanemask.{suffix}").as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} lane-mask identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "sreg"
            && policy.rust_name == policy.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::sreg::{}", policy.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::warp::{}", policy.id)],
        "{} must preserve its safe must-use raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == op_type
            && policy.dialect_op_name == format!("nvvm.read_ptx_sreg_lanemask_{suffix}")
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "direct_nvvm",
        "{} is outside the closed lane-mask lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "2.0"
            && policy.minimum_sm.as_deref() == Some("sm_20")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} lane-mask effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == format!("{section} Special Registers: %lanemask_{suffix}")
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-{suffix}"
                ),
        "{} lane-mask PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"],
        "{} lane-mask properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mov"
            && policy.expected_ptx.modifiers == ["u32"]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: format!("%lanemask_{suffix}"),
                    },
                ],
        "{} expected PTX does not match its lane-mask register",
        policy.id
    );
    Ok(())
}
