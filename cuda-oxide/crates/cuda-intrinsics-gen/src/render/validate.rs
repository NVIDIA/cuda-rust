/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, CatalogHardwareAlternative, CatalogHardwareTarget,
    ClcAdapter, ClcOperation, ClusterBarrierMode, ClusterBarrierOrdering, ClusterMemoryAdapter,
    ClusterMemoryOperation, ClusterMemorySourceContract, CpAsyncMbarrierAdapter,
    CpAsyncMbarrierStateSpace, DebugControlAdapter, DebugControlOperation,
    ExecutionControlOperation, ExtendedMinMaxAdapter, ExtendedMinMaxFormat, IntegerMinMaxFormat,
    IntrinsicBackend, IntrinsicSource, LdmatrixElement, MbarrierBasicAdapter,
    MbarrierBasicOperation, MbarrierExtendedAdapter, MbarrierExtendedOperation,
    MbarrierExtendedSourceContract, MbarrierStateSpace, RuntimeValidation, ScalarArithmeticFormat,
    ScalarArithmeticOperation, ScalarMathFormat, SparseMmaAccumulator, SpecialRegisterObservation,
    TmaAdapter, TmaOperation, WarpBarrierAdapter, WarpShuffleAdapter, WarpShuffleOperandEncoding,
    WarpShuffleValueKind, WgmmaControlAdapter, WgmmaControlMode, WgmmaControlParticipation,
};
use crate::render::common::llvm;
use crate::render::families::{
    packed_alu_format_shape, packed_conversion_dialect_operands, packed_conversion_dialect_type,
    packed_conversion_is_closed_recipe, packed_conversion_lowering_name,
    packed_conversion_rust_arguments, packed_conversion_rust_type,
    register_mma_extra_operand_count, sparse_mma_fragment_counts, stmatrix_variant,
    tcgen05_render_contract, threadfence_ptx_level,
};
use anyhow::{Result, ensure};
use std::collections::BTreeSet;

pub(super) fn validate_renderable(catalog: &CatalogFile) -> Result<()> {
    ensure!(
        !catalog.intrinsics.is_empty(),
        "catalog contains no intrinsics"
    );
    for record in &catalog.intrinsics {
        match record.family.as_str() {
            "sreg" => {
                if let Some(special) = &record.special_register {
                    ensure!(
                        matches!(record.rust.module.as_str(), "debug" | "grid" | "thread" | "warp" | "shared")
                            && record.rust.arguments.is_empty()
                            && record.dialect.operands.is_empty()
                            && record.dialect.results
                                == [format!("i{}", special.result_width.bits())]
                            && record.scalar_width() == Some(special.result_width.bits())
                            && record.lowering == "generated_special_register"
                            && record.backend_lowerings.len() == 2
                            && record.semantics.pure
                                == (special.observation
                                    == SpecialRegisterObservation::StablePure),
                        "{} is outside the closed generated special-register recipe",
                        record.id
                    );
                } else {
                    ensure!(
                        record.rust.module == "sreg"
                            && record.rust.arguments.is_empty()
                            && llvm(record).arguments.is_empty()
                            && record.lowering == "direct_nvvm"
                            && record.scalar_width().is_some(),
                        "{} is outside the zero-operand scalar direct-NVVM sreg recipe",
                        record.id
                    );
                }
            }
            "ldmatrix" => ensure!(
                record.rust.module == "matrix"
                    && record.lowering == "generated_ldmatrix"
                    && record.ldmatrix.as_ref().is_some_and(|ldmatrix| {
                        record.rust.arguments
                            == [if ldmatrix.variant.element == LdmatrixElement::B16 {
                                "*const u32"
                            } else {
                                "*const u8"
                            }]
                    }),
                "{} is outside the generated ldmatrix recipe",
                record.id
            ),
            "stmatrix" => ensure!(
                stmatrix_variant(record).is_some()
                    && record.rust.module == "matrix"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.rust.result == "()"
                    && record.dialect.results.is_empty()
                    && record.llvm.as_ref().is_some_and(|llvm| llvm.results.is_empty())
                    && record.semantics.memory == "write"
                    && record.semantics.convergent
                    && record.lowering == "generated_stmatrix",
                "{} is outside the closed generated stmatrix recipe",
                record.id
            ),
            "register_mma" => ensure!(
                record.rust.module == "matrix"
                    && record.rust.arguments.len() == 3 + register_mma_extra_operand_count(record)
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.semantics.memory == "none"
                    && record.semantics.convergent
                    && record.dialect.op_type == "RegisterMmaOp"
                    && record.dialect.op_name == "nvvm.register_mma"
                    && record.lowering == "generated_register_mma"
                    && record.register_mma.is_some(),
                "{} is outside the closed generated register-MMA recipe",
                record.id
            ),
            "movmatrix" => ensure!(
                record.rust.module == "matrix"
                    && record.rust.arguments == ["u32"]
                    && record.rust.result == "u32"
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "MovmatrixTransB16Op"
                    && record.dialect.op_name == "nvvm.movmatrix_trans_b16"
                    && record.dialect.operands == ["i32"]
                    && record.dialect.results == ["i32"]
                    && !record.semantics.pure
                    && record.semantics.memory == "inaccessible_read_write"
                    && record.semantics.convergent
                    && record.lowering == "generated_movmatrix_inline_ptx"
                    && record.movmatrix.is_some(),
                "{} is outside the generated movmatrix recipe",
                record.id
            ),
            "sparse_mma" => {
                ensure!(
                    record.sparse_mma.is_some(),
                    "{} is outside the closed generated sparse-MMA recipe",
                    record.id
                );
                let (c_count, a_count, b_count, d_count) =
                    sparse_mma_fragment_counts(record);
                let accumulator = match record.sparse_mma.as_ref().unwrap().accumulator {
                    SparseMmaAccumulator::F16 => "u32",
                    SparseMmaAccumulator::F32 => "f32",
                    SparseMmaAccumulator::S32 => "i32",
                };
                let arguments = [
                    format!("[{accumulator}; {c_count}]"),
                    format!("[u32; {a_count}]"),
                    format!("[u32; {b_count}]"),
                    "u32".to_owned(),
                    "u32".to_owned(),
                ];
                ensure!(
                    record.rust.module == "matrix"
                        && record.rust.arguments == arguments
                        && record.rust.result == format!("[{accumulator}; {d_count}]")
                        && !record.rust.safe
                        && record.rust.must_use
                        && record.semantics.memory == "none"
                        && record.semantics.convergent
                        && record.dialect.op_type == "SparseMmaOp"
                        && record.dialect.op_name == "nvvm.sparse_mma"
                        && record.lowering == "generated_sparse_mma"
                        && record.sparse_mma.is_some(),
                    "{} is outside the closed generated sparse-MMA recipe",
                    record.id
                )
            }
            "packed_atomic" => ensure!(
                record.rust.module == "atomic"
                    && record.rust.arguments == ["*mut u32", "u32"]
                    && record.rust.result == "u32"
                    && record.rust.must_use
                    && record.llvm.is_none()
                    && record.lowering == "generated_packed_atomic_inline_ptx"
                    && record.packed_atomic.is_some(),
                "{} is outside the closed generated packed-atomic recipe",
                record.id
            ),
            "redux" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value]
                        if mask == "u32" && value == &record.rust.result)
                    && matches!(record.rust.result.as_str(), "u32" | "i32" | "f32")
                    && !record.rust.safe
                    && matches!(record.dialect.results.as_slice(), [result]
                        if matches!(result.as_str(), "i32" | "f32"))
                    && matches!(record.dialect.results.as_slice(), [result]
                        if record.llvm.as_ref().is_some_and(|llvm| {
                            llvm.arguments == [result.as_str(), "i32"]
                                && llvm.results == [result.as_str()]
                        }) && record.dialect.operands == ["i32", result.as_str()])
                    && record.lowering == "generated_redux"
                    && record.redux.is_some(),
                "{} is outside the closed generated redux recipe",
                record.id
            ),
            "dotprod" => ensure!(
                record.rust.module == "dotprod"
                    && matches!(record.rust.arguments.as_slice(), [a, b, c]
                        if a == "u32" && b == "u32" && matches!(c.as_str(), "u32" | "i32"))
                    && record.rust.result == *record.rust.arguments.last().unwrap()
                    && record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.results == ["i32"]
                            && (matches!(llvm.arguments.as_slice(), [a, b, c]
                                if a == "i32" && b == "i32" && c == "i32")
                                || matches!(llvm.arguments.as_slice(), [a, b, selector, c]
                                    if a == "i32" && b == "i32" && selector == "i1" && c == "i32"))
                    })
                    && record.dialect.operands == ["i32", "i32", "i32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_dotprod"
                    && record.dot_product.is_some(),
                "{} is outside the closed generated dot-product recipe",
                record.id
            ),
            "packed_alu" => ensure!(
                record.packed_alu.as_ref().is_some_and(|packed| {
                    let (module, must_use, rust_type, dialect_type, adapter) =
                        packed_alu_format_shape(packed.format);
                    record.rust.module == module
                        && record.rust.must_use == must_use
                        && packed.adapter == adapter
                        && record
                            .rust
                            .arguments
                            .iter()
                            .all(|argument| argument == rust_type)
                        && record.rust.result == rust_type
                        && record
                            .dialect
                            .operands
                            .iter()
                            .all(|operand| operand == dialect_type)
                        && record.dialect.results == [dialect_type]
                })
                    && (1..=3).contains(&record.rust.arguments.len())
                    && record.rust.safe
                    && record.dialect.operands.len() == record.rust.arguments.len()
                    && record.lowering == "generated_packed_alu_inline_ptx",
                "{} is outside the closed generated packed-ALU recipe",
                record.id
            ),
            "packed_conversion" => ensure!(
                record.rust.module == "convert"
                    && record.rust.arguments == packed_conversion_rust_arguments(record)
                    && record.rust.result == packed_conversion_rust_type(record)
                    && record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.operands == packed_conversion_dialect_operands(record)
                    && record.dialect.results == [packed_conversion_dialect_type(record)]
                    && record.lowering == packed_conversion_lowering_name(record)
                    && record
                        .packed_conversion
                        .as_ref()
                        .is_some_and(packed_conversion_is_closed_recipe),
                "{} is outside the closed generated packed-conversion recipe",
                record.id
            ),
            "scalar_conversion" => ensure!(
                record.rust.module == "convert"
                    && record.rust.arguments == ["f32"]
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "ScalarConversionOp"
                    && record.dialect.op_name == "nvvm.scalar_conversion"
                    && record.dialect.operands == ["f32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_scalar_conversion"
                    && record.scalar_conversion.is_some(),
                "{} is outside the closed generated scalar-conversion recipe",
                record.id
            ),
            "scalar_arithmetic" => ensure!(
                record.scalar_arithmetic.as_ref().is_some_and(|arithmetic| {
                    let ty = match arithmetic.format {
                        ScalarArithmeticFormat::F32 => "f32",
                        ScalarArithmeticFormat::F64 => "f64",
                    };
                    let arity = match arithmetic.operation {
                        ScalarArithmeticOperation::Mul
                        | ScalarArithmeticOperation::Div
                        | ScalarArithmeticOperation::Add => 2,
                        ScalarArithmeticOperation::Fma => 3,
                    };
                    record.rust.arguments == vec![ty; arity]
                        && record.rust.result == ty
                        && record.dialect.operands == vec![ty; arity]
                        && record.dialect.results == [ty]
                        && record.semantics.pure
                            == (arithmetic.operation != ScalarArithmeticOperation::Div)
                })
                    && record.rust.module == "float"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "ScalarArithmeticOp"
                    && record.dialect.op_name == "nvvm.scalar_arithmetic"
                    && record.semantics.memory == "none"
                    && !record.semantics.convergent
                    && record.lowering == "generated_scalar_arithmetic",
                "{} is outside the closed generated scalar-arithmetic recipe",
                record.id
            ),
            "scalar_math" => ensure!(
                record.scalar_math.as_ref().is_some_and(|math| {
                    let (rust_ty, dialect_ty) = match math.format {
                        ScalarMathFormat::F16 => ("u16", "i16"),
                        ScalarMathFormat::F32 => ("f32", "f32"),
                        ScalarMathFormat::F64 => ("f64", "f64"),
                    };
                    record.rust.arguments == [rust_ty]
                        && record.rust.result == rust_ty
                        && record.dialect.operands == [dialect_ty]
                        && record.dialect.results == [dialect_ty]
                        && record.semantics.pure
                })
                    && record.rust.module == "float"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "ScalarMathOp"
                    && record.dialect.op_name == "nvvm.scalar_math"
                    && record.semantics.memory == "none"
                    && !record.semantics.convergent
                    && record.lowering == "generated_scalar_math",
                "{} is outside the closed generated scalar-math recipe",
                record.id
            ),
            "extended_minmax" => ensure!(
                record.extended_minmax.as_ref().is_some_and(|minmax| {
                    let (module, rust_type, dialect_type, adapter) = match minmax.format {
                        ExtendedMinMaxFormat::F32 => (
                            "float",
                            "f32",
                            "f32",
                            ExtendedMinMaxAdapter::DirectF32,
                        ),
                        ExtendedMinMaxFormat::F16 => (
                            "f16",
                            "u16",
                            "i16",
                            ExtendedMinMaxAdapter::DirectHalfU16,
                        ),
                        ExtendedMinMaxFormat::Bf16 => (
                            "bf16",
                            "u16",
                            "i16",
                            ExtendedMinMaxAdapter::DirectHalfU16,
                        ),
                        ExtendedMinMaxFormat::F16x2 => (
                            "f16x2",
                            "u32",
                            "i32",
                            ExtendedMinMaxAdapter::DirectPackedU32,
                        ),
                        ExtendedMinMaxFormat::Bf16x2 => (
                            "bf16x2",
                            "u32",
                            "i32",
                            ExtendedMinMaxAdapter::DirectPackedU32,
                        ),
                    };
                    record.rust.module == module
                        && record.rust.arguments == [rust_type, rust_type]
                        && record.rust.result == rust_type
                        && record.dialect.operands == [dialect_type, dialect_type]
                        && record.dialect.results == [dialect_type]
                        && minmax.adapter == adapter
                })
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "ExtendedMinMaxOp"
                    && record.dialect.op_name == "nvvm.extended_minmax"
                    && record.semantics.pure
                    && record.semantics.memory == "none"
                    && !record.semantics.convergent
                    && record.lowering == "generated_extended_minmax",
                "{} is outside the closed generated extended-minmax recipe",
                record.id
            ),
            "prmt" => ensure!(
                record.rust.module == "prmt"
                    && matches!(record.rust.arguments.len(), 2 | 3)
                    && record.rust.arguments.iter().all(|argument| argument == "u32")
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.op_type == "PrmtOp"
                    && record.dialect.op_name == "nvvm.prmt"
                    && record.dialect.operands.len() == record.rust.arguments.len()
                    && record.dialect.operands.iter().all(|operand| operand == "i32")
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_prmt"
                    && record.prmt.is_some(),
                "{} is outside the closed generated prmt recipe",
                record.id
            ),
            "cluster_barrier" => ensure!(
                matches!(
                    record.id.as_str(),
                    "barrier_cluster_arrive"
                        | "barrier_cluster_arrive_aligned"
                        | "barrier_cluster_arrive_relaxed"
                        | "barrier_cluster_arrive_relaxed_aligned"
                        | "barrier_cluster_wait"
                        | "barrier_cluster_wait_aligned"
                )
                    && record.rust.module == "cluster"
                    && record.rust.arguments.is_empty()
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.op_type == "ClusterBarrierOp"
                    && record.dialect.op_name == "nvvm.cluster_barrier"
                    && record.dialect.operands.is_empty()
                    && record.dialect.results.is_empty()
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments.is_empty() && llvm.results.is_empty()
                    })
                    && record.cluster_barrier.as_ref().is_some_and(|barrier| {
                        matches!(
                            (barrier.mode, barrier.ordering, barrier.aligned),
                            (
                                ClusterBarrierMode::Arrive,
                                ClusterBarrierOrdering::Release,
                                false
                            ) | (
                                ClusterBarrierMode::ArriveAligned,
                                ClusterBarrierOrdering::Release,
                                true
                            ) | (
                                ClusterBarrierMode::ArriveRelaxed,
                                ClusterBarrierOrdering::Relaxed,
                                false
                            ) | (
                                ClusterBarrierMode::ArriveRelaxedAligned,
                                ClusterBarrierOrdering::Relaxed,
                                true
                            ) | (
                                ClusterBarrierMode::Wait,
                                ClusterBarrierOrdering::Acquire,
                                false
                            ) | (
                                ClusterBarrierMode::WaitAligned,
                                ClusterBarrierOrdering::Acquire,
                                true
                            )
                        )
                    })
                    && record.lowering == "generated_cluster_barrier",
                "{} is outside the closed generated cluster-barrier recipe",
                record.id
            ),
            "cluster_memory" => ensure!(
                record.rust.module == "cluster"
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.semantics.convergent
                    && record.semantics.execution_scope == "cluster"
                    && record.target.minimum_ptx.to_string() == "7.8"
                    && matches!(
                        &record.target.hardware,
                        CatalogHardwareTarget::AnyOf { alternatives }
                            if alternatives.as_slice()
                                == [CatalogHardwareAlternative::MinimumSm { sm: 90 }]
                    )
                    && record.lowering == "generated_cluster_memory_inline_ptx"
                    && record.backend_lowerings.len() == 2
                    && record.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                            && lowering.target.minimum_ptx.to_string() == "7.8"
                            && matches!(
                                &lowering.target.hardware,
                                CatalogHardwareTarget::AnyOf { alternatives }
                                    if alternatives.as_slice()
                                        == [CatalogHardwareAlternative::MinimumSm { sm: 90 }]
                            )
                    })
                    && record.cluster_memory.as_ref().is_some_and(|cluster| {
                        cluster.runtime_validation == RuntimeValidation::Unexecuted
                            && match (cluster.operation, cluster.adapter, cluster.source_contract) {
                                (
                                    ClusterMemoryOperation::MapSharedRank,
                                    ClusterMemoryAdapter::GenericConstAndMutPointerRankToSamePointer,
                                    ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx,
                                ) => {
                                    record.id == "map_shared_rank"
                                        && record.rust.arguments == ["*const u8", "u32"]
                                        && record.rust.result == "*const u8"
                                        && record.rust.compatibility_paths
                                            == [
                                                "cuda_device::cluster::map_shared_rank",
                                                "cuda_device::cluster::map_shared_rank_mut",
                                            ]
                                        && record.dialect.op_type == "MapaSharedClusterOp"
                                        && record.dialect.op_name == "nvvm.mapa_shared_cluster"
                                        && record.dialect.operands == ["ptr", "i32"]
                                        && record.dialect.results == ["ptr"]
                                        && record.semantics.memory == "none"
                                        && matches!(
                                            record.source,
                                            IntrinsicSource::LlvmImported { ref source_record }
                                                if source_record == "int_nvvm_mapa_shared_cluster"
                                        )
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.symbol == "llvm.nvvm.mapa.shared.cluster"
                                                && llvm.arguments == ["shared_ptr", "i32"]
                                                && llvm.results == ["shared_cluster_ptr"]
                                                && llvm.properties
                                                    == [
                                                        "IntrNoMem",
                                                        "IntrSpeculatable",
                                                        "NoCapture<arg0>",
                                                    ]
                                        })
                                        && record
                                            .selections
                                            .iter()
                                            .map(|selection| selection.source_record.as_str())
                                            .collect::<BTreeSet<_>>()
                                            == BTreeSet::from([
                                                "mapa_shared_cluster_64",
                                                "mapa_shared_cluster_64i",
                                            ])
                                }
                                (
                                    ClusterMemoryOperation::ReadU32,
                                    ClusterMemoryAdapter::ConstU32PointerRankToU32,
                                    ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad,
                                ) => {
                                    record.id == "dsmem_read_u32"
                                        && record.rust.arguments == ["*const u32", "u32"]
                                        && record.rust.result == "u32"
                                        && record.rust.compatibility_paths
                                            == ["cuda_device::cluster::dsmem_read_u32"]
                                        && record.dialect.op_type == "DsmemReadU32Op"
                                        && record.dialect.op_name == "nvvm.dsmem_read_u32"
                                        && record.dialect.operands == ["ptr", "i32"]
                                        && record.dialect.results == ["i32"]
                                        && record.semantics.memory == "read"
                                        && record.llvm.is_none()
                                        && record.selections.is_empty()
                                        && matches!(
                                            record.source,
                                            IntrinsicSource::PtxNative { ref instruction }
                                                if instruction
                                                    == "mapa.shared::cluster.u64 + ld.shared::cluster.u32"
                                        )
                                }
                                _ => false,
                            }
                    }),
                "{} is outside the closed generated cluster-memory recipe",
                record.id
            ),
            "debug_control" => ensure!(
                record.rust.module == "debug"
                    && record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.operands.is_empty()
                    && record.dialect.results.is_empty()
                    && record.llvm.is_none()
                    && record.lowering == "generated_debug_control"
                    && record.debug_control.as_ref().is_some_and(|debug| {
                        debug.runtime_validation == RuntimeValidation::Unexecuted
                            && match debug.operation {
                                DebugControlOperation::Trap => {
                                    debug.adapter == DebugControlAdapter::Direct
                                        && record.rust.arguments.is_empty()
                                        && record.rust.result == "!"
                                        && record.dialect.op_type == "TrapOp"
                                        && record.dialect.op_name == "nvvm.trap"
                                }
                                DebugControlOperation::Breakpoint => {
                                    debug.adapter == DebugControlAdapter::Direct
                                        && record.rust.arguments.is_empty()
                                        && record.rust.result == "()"
                                        && record.dialect.op_type == "BreakpointOp"
                                        && record.dialect.op_name == "nvvm.brkpt"
                                }
                                DebugControlOperation::Pmevent => {
                                    debug.adapter
                                        == DebugControlAdapter::ConstGenericToImmediateU32
                                        && record.rust.arguments == ["u32"]
                                        && record.rust.result == "()"
                                        && record.dialect.op_type == "PmEventOp"
                                        && record.dialect.op_name == "nvvm.pmevent"
                                }
                            }
                    }),
                "{} is outside the closed generated debug-control recipe",
                record.id
            ),
            "clc" => ensure!(
                record.rust.module == "clc"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.lowering == "generated_clc"
                    && record.llvm.is_some()
                    && record.clc.as_ref().is_some_and(|clc| {
                        clc.runtime_validation == RuntimeValidation::Unexecuted
                            && match (clc.operation, clc.adapter) {
                                (
                                    ClcOperation::TryCancel
                                    | ClcOperation::TryCancelMulticast,
                                    ClcAdapter::GenericPointersToShared,
                                ) => {
                                    record.rust.arguments == ["*mut u8", "*mut u64"]
                                        && record.rust.result == "()"
                                        && record.dialect.operands == ["ptr", "ptr"]
                                        && record.dialect.results.is_empty()
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr", "shared_ptr"]
                                                && llvm.results.is_empty()
                                        })
                                }
                                (
                                    ClcOperation::QueryIsCanceled,
                                    ClcAdapter::PairU64ToI128BoolToU32,
                                ) => {
                                    record.rust.arguments == ["u64", "u64"]
                                        && record.rust.result == "u32"
                                        && record.dialect.operands == ["i64", "i64"]
                                        && record.dialect.results == ["i32"]
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["i128"]
                                                && llvm.results == ["i1"]
                                        })
                                }
                                (
                                    ClcOperation::QueryGetFirstCtaidX
                                    | ClcOperation::QueryGetFirstCtaidY
                                    | ClcOperation::QueryGetFirstCtaidZ,
                                    ClcAdapter::PairU64ToI128U32,
                                ) => {
                                    record.rust.arguments == ["u64", "u64"]
                                        && record.rust.result == "u32"
                                        && record.dialect.operands == ["i64", "i64"]
                                        && record.dialect.results == ["i32"]
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["i128"]
                                                && llvm.results == ["i32"]
                                        })
                                }
                                _ => false,
                            }
                    }),
                "{} is outside the closed generated CLC recipe",
                record.id
            ),
            "wgmma_control" => ensure!(
                record.rust.module == "wgmma"
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.results.is_empty()
                    && record.llvm.as_ref().is_some_and(|llvm| llvm.results.is_empty())
                    && !record.semantics.pure
                    && record.semantics.memory == "read_write"
                    && record.semantics.convergent
                    && record.semantics.execution_scope == "warpgroup"
                    && record.target.minimum_ptx.encoded() == 80
                    && record.target.hardware
                        == CatalogHardwareTarget::AnyOf {
                            alternatives: vec![CatalogHardwareAlternative::ExactArchitecture {
                                sm: 90,
                            }],
                        }
                    && record.backend_lowerings.len() == 2
                    && record.backend_lowerings.iter().any(|lowering| {
                        lowering.backend == IntrinsicBackend::LlvmNvptx
                            && lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    })
                    && record.backend_lowerings.iter().any(|lowering| {
                        lowering.backend == IntrinsicBackend::LibNvvm
                            && lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    })
                    && record.lowering == "generated_wgmma_control"
                    && record.wgmma_control.as_ref().is_some_and(|control| {
                        control.participation
                            == WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction
                            && match control.mode {
                                WgmmaControlMode::Fence => {
                                    control.adapter == WgmmaControlAdapter::NoArguments
                                        && record.id == "wgmma_fence"
                                        && record.rust.arguments.is_empty()
                                        && record.dialect.op_type == "WgmmaFenceSyncAlignedOp"
                                        && record.dialect.op_name
                                            == "nvvm.wgmma_fence_sync_aligned"
                                        && record.dialect.operands.is_empty()
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments.is_empty()
                                        })
                                }
                                WgmmaControlMode::CommitGroup => {
                                    control.adapter == WgmmaControlAdapter::NoArguments
                                        && record.id == "wgmma_commit_group"
                                        && record.rust.arguments.is_empty()
                                        && record.dialect.op_type
                                            == "WgmmaCommitGroupSyncAlignedOp"
                                        && record.dialect.op_name
                                            == "nvvm.wgmma_commit_group_sync_aligned"
                                        && record.dialect.operands.is_empty()
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments.is_empty()
                                        })
                                }
                                WgmmaControlMode::WaitGroup => {
                                    control.adapter
                                        == WgmmaControlAdapter::ConstGenericU32ToI64Immediate
                                        && record.id == "wgmma_wait_group"
                                        && record.rust.arguments == ["u64"]
                                        && record.rust.compatibility_paths
                                            == ["cuda_device::wgmma::__wgmma_wait_group"]
                                        && record.dialect.op_type
                                            == "WgmmaWaitGroupSyncAlignedOp"
                                        && record.dialect.op_name
                                            == "nvvm.wgmma_wait_group_sync_aligned"
                                        && record.dialect.operands == ["i64"]
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["i64"]
                                        })
                                }
                            }
                    }),
                "{} is outside the closed generated WGMMA-control recipe",
                record.id
            ),
            "tma" => ensure!(
                record.rust.module == "tma"
                    && record.rust.result == "()"
                    && !record.rust.must_use
                    && record.dialect.results.is_empty()
                    && record.llvm.as_ref().is_some_and(|llvm| llvm.results.is_empty())
                    && record.lowering == "generated_tma"
                    && record.tma.as_ref().is_some_and(|tma| {
                        tma.runtime_validation == RuntimeValidation::Unexecuted
                            && match (tma.operation, tma.adapter) {
                                (
                                    TmaOperation::G2sTile1d
                                    | TmaOperation::G2sTile2d
                                    | TmaOperation::G2sTile3d
                                    | TmaOperation::G2sTile4d
                                    | TmaOperation::G2sTile5d,
                                    TmaAdapter::G2sPointersCoordinatesBarrierInjectDefaults,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::G2sTile2dMulticast
                                    | TmaOperation::G2sTile2dMulticastCg2,
                                    TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::S2gTile1d
                                    | TmaOperation::S2gTile2d
                                    | TmaOperation::S2gTile3d
                                    | TmaOperation::S2gTile4d
                                    | TmaOperation::S2gTile5d,
                                    TmaAdapter::S2gPointersCoordinatesInjectDefaults,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::Reduce,
                                    TmaAdapter::ReductionPointersCoordinatesInjectDefaults,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                        && tma.reduction.is_some()
                                }
                                (TmaOperation::CommitGroup, TmaAdapter::NoOperands) => {
                                    record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::WaitGroup | TmaOperation::WaitGroupRead,
                                    TmaAdapter::CompileTimeConstantMaxPending,
                                ) => {
                                    record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::PrefetchTensorMap,
                                    TmaAdapter::DescriptorPointer,
                                ) => {
                                    !record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "read"
                                }
                                (
                                    TmaOperation::PrefetchTile1d
                                    | TmaOperation::PrefetchTile2d
                                    | TmaOperation::PrefetchTile3d
                                    | TmaOperation::PrefetchTile4d
                                    | TmaOperation::PrefetchTile5d
                                    | TmaOperation::PrefetchTileGather4TwoDimensional,
                                    TmaAdapter::DescriptorCoordinatesInjectDefaults,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read"
                                }
                                (
                                    TmaOperation::PrefetchTile1dCacheHint
                                    | TmaOperation::PrefetchTile2dCacheHint
                                    | TmaOperation::PrefetchTile3dCacheHint
                                    | TmaOperation::PrefetchTile4dCacheHint
                                    | TmaOperation::PrefetchTile5dCacheHint
                                    | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint,
                                    TmaAdapter::DescriptorCoordinatesCacheHintInjectFlag,
                                ) => {
                                    !record.rust.safe
                                        && record.semantics.convergent
                                        && record.semantics.memory == "read"
                                }
                                (
                                    TmaOperation::ReplaceGlobalAddress,
                                    TmaAdapter::DescriptorAndAddressPointers,
                                )
                                | (
                                    TmaOperation::ReplaceBoxDim
                                    | TmaOperation::ReplaceElementStride
                                    | TmaOperation::ReplaceGlobalDim,
                                    TmaAdapter::DescriptorOrdinalAndU32,
                                )
                                | (
                                    TmaOperation::ReplaceGlobalStride,
                                    TmaAdapter::DescriptorOrdinalAndU64,
                                )
                                | (
                                    TmaOperation::ReplaceElementType
                                    | TmaOperation::ReplaceFillMode
                                    | TmaOperation::ReplaceInterleaveLayout
                                    | TmaOperation::ReplaceSwizzleAtomicity
                                    | TmaOperation::ReplaceSwizzleMode,
                                    TmaAdapter::DescriptorAndImmediateU32,
                                )
                                | (
                                    TmaOperation::ReplaceRank,
                                    TmaAdapter::DescriptorAndRuntimeU32,
                                ) => {
                                    !record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "write"
                                }
                                (
                                    TmaOperation::FenceProxyTensorMapAcquireCluster
                                    | TmaOperation::FenceProxyTensorMapAcquireCta
                                    | TmaOperation::FenceProxyTensorMapAcquireGpu
                                    | TmaOperation::FenceProxyTensorMapAcquireSystem,
                                    TmaAdapter::DescriptorPointerInjectBytes,
                                ) => {
                                    !record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                (
                                    TmaOperation::FenceProxyTensorMapReleaseCluster
                                    | TmaOperation::FenceProxyTensorMapReleaseCta
                                    | TmaOperation::FenceProxyTensorMapReleaseGpu
                                    | TmaOperation::FenceProxyTensorMapReleaseSystem,
                                    TmaAdapter::NoOperands,
                                ) => {
                                    record.rust.safe
                                        && !record.semantics.convergent
                                        && record.semantics.memory == "read_write"
                                }
                                _ => false,
                            }
                    }),
                "{} is outside the closed generated TMA recipe",
                record.id
            ),
            "counted_barrier" | "grid_dependency" | "register_control" => ensure!(
                ExecutionControlOperation::from_catalog_id(&record.id).is_some_and(|operation| {
                    record.family == operation.family()
                        && record.rust.result == "()"
                        && !record.rust.safe
                        && !record.rust.must_use
                        && record.dialect.results.is_empty()
                        && record
                            .llvm
                            .as_ref()
                            .is_some_and(|llvm| llvm.results.is_empty())
                        && record.lowering == "generated_execution_control"
                        && match operation {
                            ExecutionControlOperation::BarrierCtaSync
                            | ExecutionControlOperation::BarrierCtaSyncAligned
                            | ExecutionControlOperation::BarrierCtaArrive
                            | ExecutionControlOperation::BarrierCtaArriveAligned => {
                                record.rust.module == "barrier"
                                    && record.rust.arguments == ["u32", "u32"]
                                    && record.dialect.operands == ["i32", "i32"]
                                    && record.llvm.as_ref().is_some_and(|llvm| {
                                        llvm.arguments == ["i32", "i32"]
                                    })
                            }
                            ExecutionControlOperation::GridDependencyLaunchDependents
                            | ExecutionControlOperation::GridDependencyWait => {
                                record.rust.module == "grid"
                                    && record.rust.arguments.is_empty()
                                    && record.dialect.operands.is_empty()
                                    && record
                                        .llvm
                                        .as_ref()
                                        .is_some_and(|llvm| llvm.arguments.is_empty())
                            }
                            ExecutionControlOperation::SetMaxNRegInc
                            | ExecutionControlOperation::SetMaxNRegDec => {
                                record.rust.module == "thread"
                                    && record.rust.arguments == ["u32"]
                                    && record.dialect.operands.is_empty()
                                    && record
                                        .llvm
                                        .as_ref()
                                        .is_some_and(|llvm| llvm.arguments == ["i32"])
                            }
                        }
                }),
                "{} is outside the closed generated execution-control recipe",
                record.id
            ),
            "tcgen05" => ensure!(
                tcgen05_render_contract(record),
                "{} is outside the closed generated tcgen05 recipe",
                record.id
            ),
            "cp_async_copy" => ensure!(
                record.rust.module == "async_copy"
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        matches!(llvm.arguments.as_slice(), [dst, src]
                            if dst == "shared_ptr" && src == "global_ptr")
                            || matches!(llvm.arguments.as_slice(), [dst, src, size]
                                if dst == "shared_ptr" && src == "global_ptr" && size == "i32")
                    })
                    && record.llvm.as_ref().is_some_and(|llvm| llvm.results.is_empty())
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_cp_async_copy"
                    && record.cp_async_copy.is_some(),
                "{} is outside the closed generated cp.async copy recipe",
                record.id
            ),
            "cp_async_control" => ensure!(
                record.rust.module == "async_copy"
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.results.is_empty()
                            && (llvm.arguments.is_empty() || llvm.arguments == ["i32"])
                    })
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_cp_async_control"
                    && record.cp_async_control.is_some(),
                "{} is outside the closed generated cp.async control recipe",
                record.id
            ),
            "cp_async_mbarrier" => ensure!(
                record.rust.module == "async_copy"
                    && record.rust.arguments == ["*mut u64"]
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.operands == ["ptr"]
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_cp_async_mbarrier"
                    && record.cp_async_mbarrier.as_ref().is_some_and(|bridge| {
                        bridge.adapter == CpAsyncMbarrierAdapter::PointerToVoid
                            && record.llvm.as_ref().is_some_and(|llvm| {
                                llvm.results.is_empty()
                                    && match bridge.state_space {
                                        CpAsyncMbarrierStateSpace::Generic => {
                                            llvm.arguments == ["ptr"]
                                        }
                                        CpAsyncMbarrierStateSpace::Shared => {
                                            llvm.arguments == ["shared_ptr"]
                                        }
                                    }
                            })
                    }),
                "{} is outside the closed generated cp.async mbarrier recipe",
                record.id
            ),
            "mbarrier_basic" => ensure!(
                record.rust.module == "barrier"
                    && !record.rust.safe
                    && record.lowering == "generated_mbarrier_basic"
                    && record.mbarrier_basic.as_ref().is_some_and(|mbarrier| {
                        mbarrier.state_space == MbarrierStateSpace::Shared
                            && match (mbarrier.operation, mbarrier.adapter) {
                                (
                                    MbarrierBasicOperation::Init,
                                    MbarrierBasicAdapter::InitPointerCountToVoid,
                                ) => {
                                    record.rust.arguments == ["*mut u64", "u32"]
                                        && record.rust.result == "()"
                                        && !record.rust.must_use
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr", "i32"]
                                                && llvm.results.is_empty()
                                        })
                                        && record.dialect.operands == ["ptr", "i32"]
                                        && record.dialect.results.is_empty()
                                }
                                (
                                    MbarrierBasicOperation::Arrive,
                                    MbarrierBasicAdapter::ArrivePointerToToken,
                                ) => {
                                    record.rust.arguments == ["*const u64"]
                                        && record.rust.result == "u64"
                                        && record.rust.must_use
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr"]
                                                && llvm.results == ["i64"]
                                        })
                                        && record.dialect.operands == ["ptr"]
                                        && record.dialect.results == ["i64"]
                                }
                                (
                                    MbarrierBasicOperation::ArriveNoComplete,
                                    MbarrierBasicAdapter::ArriveNoCompletePointerCountToToken,
                                ) => {
                                    record.rust.arguments == ["*const u64", "u32"]
                                        && record.rust.result == "u64"
                                        && record.rust.must_use
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr", "i32"]
                                                && llvm.results == ["i64"]
                                        })
                                        && record.dialect.operands == ["ptr", "i32"]
                                        && record.dialect.results == ["i64"]
                                }
                                (
                                    MbarrierBasicOperation::TestWait,
                                    MbarrierBasicAdapter::TestWaitPointerTokenToPredicate,
                                ) => {
                                    record.rust.arguments == ["*const u64", "u64"]
                                        && record.rust.result == "bool"
                                        && record.rust.must_use
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr", "i64"]
                                                && llvm.results == ["i1"]
                                        })
                                        && record.dialect.operands == ["ptr", "i64"]
                                        && record.dialect.results == ["i1"]
                                }
                                (
                                    MbarrierBasicOperation::Inval,
                                    MbarrierBasicAdapter::InvalPointerToVoid,
                                ) => {
                                    record.rust.arguments == ["*mut u64"]
                                        && record.rust.result == "()"
                                        && !record.rust.must_use
                                        && record.llvm.as_ref().is_some_and(|llvm| {
                                            llvm.arguments == ["shared_ptr"]
                                                && llvm.results.is_empty()
                                        })
                                        && record.dialect.operands == ["ptr"]
                                        && record.dialect.results.is_empty()
                                }
                                _ => false,
                            }
                    }),
                "{} is outside the closed generated basic mbarrier recipe",
                record.id
            ),
            "mbarrier_extended" => ensure!(
                record.rust.module == "barrier"
                    && !record.rust.safe
                    && record.semantics.memory == "read_write"
                    && record.semantics.convergent
                    && record.lowering == "generated_mbarrier_extended_inline_ptx"
                    && record.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    })
                    && record.mbarrier_extended.as_ref().is_some_and(|mbarrier| {
                        let source_matches = match mbarrier.source_contract {
                            MbarrierExtendedSourceContract::LlvmImported => record.llvm.is_some(),
                            MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => {
                                record.llvm.is_none()
                                    && matches!(record.source, IntrinsicSource::PtxNative { .. })
                            }
                        };
                        source_matches
                            && match (mbarrier.operation, mbarrier.adapter) {
                                (
                                    MbarrierExtendedOperation::ArriveExpectTxCta,
                                    MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
                                ) => record.id == "mbarrier_arrive_expect_tx",
                                (
                                    MbarrierExtendedOperation::ArriveExpectTxCluster,
                                    MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
                                ) => record.id == "mbarrier_arrive_expect_tx_cluster",
                                (
                                    MbarrierExtendedOperation::ArriveRemoteCluster,
                                    MbarrierExtendedAdapter::RawClusterAddressToVoid,
                                ) => record.id == "mbarrier_arrive_cluster",
                                (
                                    MbarrierExtendedOperation::TryWaitTokenCta,
                                    MbarrierExtendedAdapter::PointerTokenToPredicate,
                                ) => record.id == "mbarrier_try_wait",
                                (
                                    MbarrierExtendedOperation::TryWaitParityCta,
                                    MbarrierExtendedAdapter::PointerParityToPredicate,
                                ) => record.id == "mbarrier_try_wait_parity",
                                (
                                    MbarrierExtendedOperation::TryWaitParityCluster,
                                    MbarrierExtendedAdapter::PointerParityToPredicate,
                                ) => record.id == "mbarrier_try_wait_parity_cluster",
                                (
                                    MbarrierExtendedOperation::FenceProxyAsyncSharedCta,
                                    MbarrierExtendedAdapter::ZeroOperandsToVoid,
                                ) => record.id == "fence_proxy_async_shared_cta",
                                (
                                    MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster,
                                    MbarrierExtendedAdapter::ZeroOperandsToVoid,
                                ) => record.id == "fence_mbarrier_init_release_cluster",
                                (
                                    MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster,
                                    MbarrierExtendedAdapter::ZeroOperandsToVoid,
                                ) => record.id
                                    == "fence_proxy_async_generic_release_shared_cta_cluster",
                                (
                                    MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster,
                                    MbarrierExtendedAdapter::ZeroOperandsToVoid,
                                ) => record.id
                                    == "fence_proxy_async_generic_acquire_shared_cluster_cluster",
                                (
                                    MbarrierExtendedOperation::Nanosleep,
                                    MbarrierExtendedAdapter::NanosecondsToVoid,
                                ) => record.id == "nanosleep",
                                _ => false,
                            }
                    }),
                "{} is outside the closed generated extended-mbarrier recipe",
                record.id
            ),
            "sync" => {
                if record.id == "sync_threads" {
                    ensure!(
                        record.rust.module == "thread"
                            && record.rust.name == "sync_threads"
                            && record.rust.arguments.is_empty()
                            && record.rust.result == "()"
                            && !record.rust.safe
                            && !record.rust.must_use
                            && record.llvm.as_ref().is_some_and(|llvm| {
                                llvm.symbol == "llvm.nvvm.barrier.cta.sync.aligned.all"
                                    && llvm.arguments == ["i32"]
                                    && llvm.results.is_empty()
                            })
                            && record.dialect.op_type == "Barrier0Op"
                            && record.dialect.op_name == "nvvm.barrier0"
                            && record.dialect.operands.is_empty()
                            && record.dialect.results.is_empty()
                            && record.lowering == "generated_sync_threads",
                        "{} is outside the fixed-zero generated sync_threads recipe",
                        record.id
                    );
                } else {
                    ensure!(
                        threadfence_ptx_level(record).is_some()
                            && record.rust.module == "fence"
                            && record.rust.name == record.id
                            && record.rust.arguments.is_empty()
                            && record.rust.result == "()"
                            && record.rust.safe
                            && !record.rust.must_use
                            && record.llvm.as_ref().is_some_and(|llvm| {
                                llvm.arguments.is_empty() && llvm.results.is_empty()
                            })
                            && record.dialect.operands.is_empty()
                            && record.dialect.results.is_empty()
                            && record.lowering == "direct_nvvm"
                            && record.semantics.memory == "read_write"
                            && !record.semantics.pure
                            && !record.semantics.convergent,
                        "{} is outside the closed generated thread-fence recipe",
                        record.id
                    );
                }
            }
            "vote" => ensure!(
                record.rust.module == "warp"
                    && record.rust.arguments == ["u32", "bool"]
                    && matches!(record.rust.result.as_str(), "bool" | "u32")
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments == ["i32", "i1"]
                            && matches!(llvm.results.as_slice(), [result]
                                if result == "i1" || result == "i32")
                    })
                    && record.dialect.operands == ["i32", "i1"]
                    && matches!(record.dialect.results.as_slice(), [result]
                        if result == "i1" || result == "i32")
                    && record.lowering == "generated_vote"
                    && record.vote.is_some(),
                "{} is outside the closed generated vote.sync recipe",
                record.id
            ),
            "active_mask" => ensure!(
                record.id == "active_mask"
                    && record.rust.module == "warp"
                    && record.rust.arguments.is_empty()
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments.is_empty() && llvm.results == ["i32"]
                    })
                    && record.dialect.operands.is_empty()
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_active_mask"
                    && record.active_mask.is_some(),
                "{} is outside the closed generated active-mask recipe",
                record.id
            ),
            "warp_match" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value]
                        if mask == "u32" && matches!(value.as_str(), "u32" | "u64"))
                    && record.rust.result == "u32"
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        matches!(llvm.arguments.as_slice(), [mask, value]
                            if mask == "i32" && matches!(value.as_str(), "i32" | "i64"))
                            && (matches!(llvm.results.as_slice(), [mask] if mask == "i32")
                                || matches!(llvm.results.as_slice(), [mask, predicate]
                                    if mask == "i32" && predicate == "i1"))
                    })
                    && matches!(record.dialect.operands.as_slice(), [mask, value]
                        if mask == "i32" && matches!(value.as_str(), "i32" | "i64"))
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_warp_match"
                    && record.warp_match.is_some(),
                "{} is outside the closed generated match.sync recipe",
                record.id
            ),
            "elect" => ensure!(
                record.id == "elect_sync"
                    && record.rust.module == "warp"
                    && record.rust.name == "elect_sync"
                    && record.rust.arguments == ["u32"]
                    && record.rust.result == "(u32, bool)"
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.symbol == "llvm.nvvm.elect.sync"
                            && llvm.arguments == ["i32"]
                            && llvm.results == ["i32", "i1"]
                    })
                    && record.dialect.op_type == "ElectSyncOp"
                    && record.dialect.op_name == "nvvm.elect_sync"
                    && record.dialect.operands == ["i32"]
                    && record.dialect.results == ["i32", "i1"]
                    && record.lowering == "generated_elect",
                "{} is outside the closed generated elect.sync recipe",
                record.id
            ),
            "warp_barrier" => ensure!(
                record.id == "sync_mask"
                    && record.rust.module == "warp"
                    && record.rust.name == "sync_mask"
                    && record.rust.arguments == ["u32"]
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.symbol == "llvm.nvvm.bar.warp.sync"
                            && llvm.arguments == ["i32"]
                            && llvm.results.is_empty()
                    })
                    && record.dialect.op_type == "BarWarpSyncOp"
                    && record.dialect.op_name == "nvvm.bar_warp_sync"
                    && record.dialect.operands == ["i32"]
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_warp_barrier"
                    && record.warp_barrier.as_ref().is_some_and(|barrier| {
                        barrier.adapter == WarpBarrierAdapter::DirectMemberMask
                    }),
                "{} is outside the closed generated bar.warp.sync recipe",
                record.id
            ),
            "warp_shuffle" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value, lane]
                        if mask == "u32"
                            && lane == "u32"
                            && matches!(value.as_str(), "u32" | "f32" | "u64")
                            && value == &record.rust.result)
                    && !record.rust.safe
                    && record.rust.must_use
                    && matches!(record.dialect.operands.as_slice(), [mask, value, lane]
                        if mask == "i32"
                            && lane == "i32"
                            && matches!(value.as_str(), "i32" | "f32" | "i64"))
                    && matches!(record.dialect.results.as_slice(), [result]
                        if result == &record.dialect.operands[1])
                    && record.warp_shuffle.as_ref().is_some_and(|shuffle| {
                        match shuffle.value_kind {
                            WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32 => {
                                let (value_ty, rust_ty) = match shuffle.value_kind {
                                    WarpShuffleValueKind::I32 => ("i32", "u32"),
                                    WarpShuffleValueKind::F32 => ("f32", "f32"),
                                    WarpShuffleValueKind::I64 => unreachable!(),
                                };
                                shuffle.adapter
                                    == WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
                                    && shuffle.lane_encoding
                                        == WarpShuffleOperandEncoding::RegisterOrImmediate
                                    && shuffle.mask_encoding
                                        == WarpShuffleOperandEncoding::RegisterOrImmediate
                                    && record.lowering == "generated_warp_shuffle"
                                    && record.rust.result == rust_ty
                                    && record.dialect.operands[1] == value_ty
                                    && record.llvm.as_ref().is_some_and(|llvm| {
                                        matches!(llvm.arguments.as_slice(), [mask, value, lane, clamp]
                                            if mask == "i32"
                                                && value == value_ty
                                                && lane == "i32"
                                                && clamp == "i32")
                                            && llvm.results == [value_ty]
                                    })
                            }
                            WarpShuffleValueKind::I64 => {
                                shuffle.adapter
                                    == WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
                                    && shuffle.lane_encoding
                                        == WarpShuffleOperandEncoding::RegisterOnly
                                    && shuffle.mask_encoding
                                        == WarpShuffleOperandEncoding::RegisterOnly
                                    && record.rust.result == "u64"
                                    && record.dialect.operands[1] == "i64"
                                    && record.lowering
                                        == "generated_warp_shuffle_i64_inline_ptx"
                                    && record.llvm.is_none()
                                    && matches!(record.source, IntrinsicSource::PtxNative { .. })
                            }
                        }
                    }),
                "{} is outside the closed generated shfl.sync recipe",
                record.id
            ),
            "integer_minmax" => ensure!(
                record.integer_minmax.as_ref().is_some_and(|minmax| {
                    let module = match minmax.format {
                        IntegerMinMaxFormat::S32 => "int",
                        IntegerMinMaxFormat::S16x2 | IntegerMinMaxFormat::U16x2 => "i16x2",
                    };
                    record.rust.module == module
                })
                    && record.rust.arguments.len() == 2
                    && record
                        .rust
                        .arguments
                        .iter()
                        .all(|argument| argument == &record.rust.result)
                    && matches!(record.rust.result.as_str(), "i32" | "u32")
                    && record.rust.safe
                    && record.rust.must_use
                    && record.dialect.operands == ["i32", "i32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_integer_minmax_inline_ptx",
                "{} is outside the closed generated integer-min/max recipe",
                record.id
            ),
            family => ensure!(false, "{} has unrenderable family {family}", record.id),
        };
    }
    Ok(())
}
