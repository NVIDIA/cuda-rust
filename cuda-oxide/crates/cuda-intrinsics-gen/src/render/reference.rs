/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, ClusterBarrierOrdering, ClusterMemorySourceContract,
    CpAsyncMbarrierOperation, CpAsyncMbarrierStateSpace, EvidenceArtifactKind, IntrinsicBackend,
    MbarrierBasicOperation, MbarrierExtendedSourceContract, PackedAtomicFormat,
    PackedConversionRounding, PackedConversionSaturation, RegisterMmaAdapter, RegisterMmaOperation,
    RegisterMmaOverflow, RuntimeValidation, SparseMmaOverflow, WarpMatchAdapter,
    WarpShuffleValueKind, WgmmaControlMode,
};
use crate::render::common::{
    backend_label, evidence_stage_label, hardware_target_label, llvm, lowering_mechanism_label,
    markdown_header, source_label,
};
use crate::render::families::{
    BLACKWELL_LDMATRIX_EFFECTIVE_FLOORS, active_masks, cluster_barriers, cluster_memory,
    cp_async_mbarriers, expected_ptx_head, is_blackwell_ldmatrix, mbarrier_basics,
    mbarrier_extended, movmatrix, packed_alu_format_shape, packed_alu_ptx_mnemonic, packed_alus,
    packed_atomics, packed_conversion_destination, packed_conversion_ptx_mnemonic,
    packed_conversion_typed_llvm_name, packed_conversions, redux, register_mmas,
    scalar_arithmetic_llvm_mechanism, scalar_arithmetic_ptx_mnemonic, scalar_arithmetics,
    scalar_conversion_ptx_mnemonic, scalar_conversions, sparse_mma_metadata_rule,
    sparse_mma_ptx_head, sparse_mma_selector_description, sparse_mmas, sync_intrinsics,
    threadfence_ptx_level, vote_intrinsics, warp_barriers, warp_matches, warp_shuffles,
    wgmma_controls,
};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Escape a value for a GFM table cell.
///
/// A `|` splits the row before inline parsing runs, so it splits the cell even
/// inside a code span. GFM's own rule is that a literal pipe must be written
/// `\|` wherever it appears. Three catalog entries carry a literal pipe today:
/// the `elect.sync` and `match.all.sync` PTX patterns write their two
/// destinations as `<register|predicate>`. Their rows rendered with a seventh
/// cell, which dropped the backend-evidence column.
fn escape_table_cell(value: impl std::fmt::Display) -> String {
    value.to_string().replace('|', "\\|")
}

pub(super) fn render_reference(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = markdown_header(catalog, hash);
    output.push_str(
        "# Generated Intrinsic Reference\n\nThis table is generated from the resolved catalog. Evidence stages below distinguish backend code generation, terminal validation, and GPU runtime execution; no runtime claim is made unless an executed stage is recorded.\n\n| Rust function | CUDA operation | Source | Effects | Availability | Backend evidence |\n|:--|:--|:--|:--|:--|:--|\n",
    );
    for record in &catalog.intrinsics {
        let safety = if record.rust.safe { "safe" } else { "unsafe" };
        let availability = if is_blackwell_ldmatrix(record) {
            format!(
                "Instruction floor PTX {} on {}; effective LLVM target floors: {BLACKWELL_LDMATRIX_EFFECTIVE_FLOORS}",
                record.target.minimum_ptx,
                hardware_target_label(&record.target.hardware)
            )
        } else {
            format!(
                "PTX {} on {}",
                record.target.minimum_ptx,
                hardware_target_label(&record.target.hardware)
            )
        };
        writeln!(
            output,
            "| `{}` | `{}` | {} | {safety}; scope {}; {}; memory {}; convergent {} | {availability} ([PTX ISA {}, {}]({})) | {} on `{}`; expects `{}` (`{}`, SHA-256 `{}`) |",
            escape_table_cell(&record.rust.public_path),
            escape_table_cell(&record.dialect.op_name),
            escape_table_cell(source_label(record)),
            escape_table_cell(&record.semantics.execution_scope),
            if record.semantics.pure { "pure" } else { "impure" },
            escape_table_cell(&record.semantics.memory),
            record.semantics.convergent,
            escape_table_cell(&record.target.ptx_isa_version),
            escape_table_cell(&record.target.ptx_isa_section),
            escape_table_cell(&record.target.ptx_isa_url),
            escape_table_cell(&record.backend.status),
            escape_table_cell(&record.backend.profile),
            escape_table_cell(&record.expected_ptx),
            escape_table_cell(&record.backend.version),
            escape_table_cell(&record.backend.sha256),
        )
        .unwrap();
    }
    output.push_str("\n## Compiler identity and compatibility paths\n\n");
    for record in &catalog.intrinsics {
        writeln!(output, "- `{}` (`{}`)", record.id, record.rust.abi_id).unwrap();
        writeln!(
            output,
            "  - canonical compiler path: `{}`",
            record.rust.canonical_path
        )
        .unwrap();
        writeln!(
            output,
            "  - public source path: `{}`",
            record.rust.public_path
        )
        .unwrap();
        for path in &record.rust.compatibility_paths {
            writeln!(output, "  - compatibility compiler path: `{path}`").unwrap();
        }
    }
    output.push_str("\n## Register-MMA contracts\n\n");
    for record in register_mmas(catalog) {
        let mma = record.register_mma.as_ref().unwrap();
        let operation = match mma.operation {
            RegisterMmaOperation::Multiply => "multiply-accumulate",
            RegisterMmaOperation::AndPopc => "AND, population count, and accumulate",
            RegisterMmaOperation::XorPopc => "XOR, population count, and accumulate",
        };
        let overflow = match mma.overflow {
            RegisterMmaOverflow::NotApplicable => "not applicable",
            RegisterMmaOverflow::Wrapping => "wrapping",
            RegisterMmaOverflow::Satfinite => "finite saturation",
        };
        let runtime = match mma.runtime_validation {
            RuntimeValidation::Unexecuted => "not executed on a GPU",
            RuntimeValidation::Executed => "executed on a GPU",
        };
        if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 {
            writeln!(
                output,
                "- `{}` takes C, A, B, scale-A data/selectors, and scale-B data/selectors in PTX operand order, performs {operation}, and lowers to one convergent, register-only `{}` instruction. For `scale_vec::1X`, byte selectors must be in `0..=3`, the A thread selector in `0..=1`, and the B thread selector in `0..=3`. Every non-exited warp lane must execute the same instruction and qualifiers. Integer overflow is {overflow}; runtime validation is {runtime}.",
                record.id,
                expected_ptx_head(record),
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "- `{}` takes fragments in C, A, B order, performs {operation}, and lowers to one convergent, register-only `{}` instruction. Every non-exited warp lane must execute the same instruction and qualifiers. Integer overflow is {overflow}; runtime validation is {runtime}.",
                record.id,
                expected_ptx_head(record),
            )
            .unwrap();
        }
    }
    output.push_str("\n## Sparse-MMA contracts\n\n");
    for record in sparse_mmas(catalog) {
        let mma = record.sparse_mma.as_ref().unwrap();
        let overflow = match mma.overflow {
            SparseMmaOverflow::NotApplicable => "Overflow mode is not applicable.",
            SparseMmaOverflow::Wrapping => "Integer overflow wraps.",
            SparseMmaOverflow::Satfinite => "Integer overflow uses finite saturation.",
        };
        let runtime = match mma.runtime_validation {
            RuntimeValidation::Unexecuted => "not executed on a GPU",
            RuntimeValidation::Executed => "executed on a GPU",
        };
        let metadata = sparse_mma_metadata_rule(mma);
        writeln!(
            output,
            "- `{}` takes fragments in C, A, B, metadata, selector order and lowers to one convergent, register-only `{}` instruction. Its LLVM source record uses A, B, C, metadata, selector order. The selector must be {}. Every non-exited warp lane must execute the same instruction and qualifiers. {metadata} {overflow} Runtime validation is {runtime}.",
            record.id,
            sparse_mma_ptx_head(record),
            sparse_mma_selector_description(record),
        )
        .unwrap();
    }
    output.push_str("\n## Packed-atomic contracts\n\n");
    for record in packed_atomics(catalog) {
        let packed = record.packed_atomic.as_ref().unwrap();
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "f16x2",
            PackedAtomicFormat::Bf16x2 => "bf16x2",
        };
        writeln!(
            output,
            "- `{}` (`{format}`): the native PTX instruction starts at `sm_{}`; cuda-oxide admits it from {}. Omitted `.sem` / `.scope` mean relaxed GPU scope. Each 16-bit element rounds to nearest-even and `.noftz` preserves subnormal inputs and results. The elements are atomic independently, so the returned `u32` contains old per-element values that may not form one coherent pair. The pointer must address four writable, four-byte-aligned global bytes; do not mix whole-word or non-atomic overlapping access, and every racing atomic must use a mutually inclusive scope.",
            record.id,
            packed.native_minimum_sm,
            hardware_target_label(&record.target.hardware),
        )
        .unwrap();
    }
    output.push_str("\n## Redux contracts\n\n");
    for record in redux(catalog) {
        writeln!(
            output,
            "- `{}`: raw and dialect operands are `[member_mask, value]`, adapted to LLVM `(value, membermask)`. The executing lane must be named in the mask, and every non-exited named lane must execute the same instruction with the same qualifiers and mask.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Packed-ALU contracts\n\n");
    for record in packed_alus(catalog) {
        let packed = record.packed_alu.as_ref().unwrap();
        let (format, _, carrier, _, _) = packed_alu_format_shape(packed.format);
        let backend_floors = record
            .backend_lowerings
            .iter()
            .map(|lowering| {
                let backend = match lowering.backend {
                    IntrinsicBackend::LlvmNvptx => "LLVM-NVPTX",
                    IntrinsicBackend::LibNvvm => "libNVVM",
                };
                format!(
                    "{backend} PTX {} on {}",
                    lowering.target.minimum_ptx,
                    hardware_target_label(&lowering.target.hardware),
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        writeln!(
            output,
            "- `{}` carries one packed `{format}` value in a `{carrier}` and lowers to one pure `{}` instruction. The native instruction starts at PTX {} / `sm_{}`; cuda-oxide admits it from {}. Backend profile floors: {backend_floors}.",
            record.id,
            packed_alu_ptx_mnemonic(record),
            record.target.minimum_ptx,
            packed.native_minimum_sm,
            hardware_target_label(&record.target.hardware),
        )
        .unwrap();
    }
    output.push_str("\n## Packed-conversion contract\n\n");
    for record in packed_conversions(catalog) {
        let conversion = record.packed_conversion.as_ref().unwrap();
        let rounding = match conversion.rounding {
            PackedConversionRounding::NearestEven => "nearest-even",
            PackedConversionRounding::TowardZero => "toward-zero",
        };
        let saturation = match conversion.saturation {
            PackedConversionSaturation::None => "without saturation",
            PackedConversionSaturation::Relu => "with ReLU",
            PackedConversionSaturation::Satfinite => "with finite saturation",
            PackedConversionSaturation::SatfiniteRelu => "with finite saturation and ReLU",
        };
        if packed_conversion_typed_llvm_name(record).is_some() {
            writeln!(
                output,
                "- `{}` converts two `f32` inputs to packed `{}` using {rounding} rounding {saturation}. LLVM-NVPTX uses typed `{}` with `[high, low]` inputs; libNVVM uses pure `{}` inline PTX. The public first input becomes the low lane and the second becomes the high lane.",
                record.id,
                packed_conversion_destination(record),
                llvm(record).symbol,
                packed_conversion_ptx_mnemonic(record),
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "- `{}` converts two `f32` inputs to packed `{}` using {rounding} rounding {saturation}. It lowers to pure `{}` inline PTX. The first input becomes the low lane and the second becomes the high lane, so PTX prints the inputs in reverse order.",
                record.id,
                packed_conversion_destination(record),
                packed_conversion_ptx_mnemonic(record),
            )
            .unwrap();
        }
    }
    output.push_str("\n## Scalar TF32-conversion contract\n\n");
    for record in scalar_conversions(catalog) {
        writeln!(
            output,
            "- `{}` converts one `f32` value to raw TF32 bits with `{}`. LLVM-NVPTX uses the typed intrinsic; libNVVM uses pure inline PTX so the native target floor is preserved.",
            record.id,
            scalar_conversion_ptx_mnemonic(record),
        )
        .unwrap();
    }
    output.push_str("\n## Explicit-rounding scalar-arithmetic contracts\n\n");
    for record in scalar_arithmetics(catalog) {
        let llvm_route = match scalar_arithmetic_llvm_mechanism(record) {
            BackendLoweringMechanism::TypedNvvm => "the typed intrinsic",
            BackendLoweringMechanism::InlinePtx => "pure inline PTX",
        };
        writeln!(
            output,
            "- `{}` performs `{}`. LLVM-NVPTX uses {llvm_route}; libNVVM uses pure inline PTX.",
            record.id,
            scalar_arithmetic_ptx_mnemonic(record),
        )
        .unwrap();
    }
    output.push_str("\n## Warp vote contracts\n\n");
    for record in vote_intrinsics(catalog) {
        writeln!(
            output,
            "- `{}` keeps raw and dialect operands in `[member_mask, predicate]` order. The executing lane must be named in the mask, and every non-exited named lane must execute the same `vote.sync` instruction with the same qualifiers and mask. On `sm_6x` and earlier, all named lanes must execute in convergence and no unnamed lane may be active. Both immediate and register member masks are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Active-mask contract\n\n");
    for record in active_masks(catalog) {
        writeln!(
            output,
            "- `{}` observes the lanes active at the instruction. LLVM uses the typed intrinsic; libNVVM uses reviewed convergent, side-effecting inline PTX because that backend does not select the intrinsic.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-match contracts\n\n");
    for record in warp_matches(catalog) {
        let adapter = match record.warp_match.as_ref().unwrap().adapter {
            WarpMatchAdapter::DirectMask => "returns LLVM's mask directly",
            WarpMatchAdapter::ProjectMaskDiscardPredicate => {
                "projects field 0 from LLVM's `{i32, i1}` result"
            }
        };
        writeln!(
            output,
            "- `{}` keeps operands in `[member_mask, value]` order and {adapter}. The executing lane must be named in the mask, and every non-exited named lane must execute the same `match.sync` operation with the same qualifiers and mask. All register/immediate value and mask forms are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-barrier contract\n\n");
    for record in warp_barriers(catalog) {
        writeln!(
            output,
            "- `{}` passes the 32-bit member mask directly to the typed LLVM intrinsic on both backends. The executing lane must be named in the mask, and every non-exited named lane must execute the same `bar.warp.sync` operation with the same mask. On `sm_6x` and earlier, all named lanes must execute the barrier in convergence, and no unnamed lane may be active when it executes. The barrier orders memory accesses among participating lanes. Both immediate and register masks are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-shuffle contracts\n\n");
    for record in warp_shuffles(catalog) {
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        let lowering = match shuffle.value_kind {
            WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32 => {
                "Register and immediate lane/mask forms are admitted."
            }
            WarpShuffleValueKind::I64 => {
                "One convergent, side-effecting inline-PTX block splits `i64`, performs low then high `b32` shuffles with the same register lane/mask, and reassembles it."
            }
        };
        writeln!(
            output,
            "- `{}` keeps raw and dialect operands in `[member_mask, value, lane_or_delta]` order and inserts clamp `{}` during lowering. The executing lane must be named in the mask, and every non-exited named lane must execute the same `shfl.sync` operation with the same qualifiers and mask. On `sm_6x` and earlier, all named lanes must execute in convergence and no unnamed lane may be active. A computed in-range source must be active and named; if PTX marks it out of range, the calling lane's input is copied. {lowering}",
            record.id,
            shuffle.clamp,
        )
        .unwrap();
    }
    output.push_str("\n## Cluster-barrier contracts\n\n");
    for record in cluster_barriers(catalog) {
        let barrier = record.cluster_barrier.as_ref().unwrap();
        let ordering = match barrier.ordering {
            ClusterBarrierOrdering::Release => "release",
            ClusterBarrierOrdering::Relaxed => "relaxed",
            ClusterBarrierOrdering::Acquire => "acquire",
        };
        let alignment = if barrier.aligned {
            "Every non-exited warp thread must execute the same aligned instruction in identical control flow."
        } else {
            "The instruction is not aligned."
        };
        writeln!(
            output,
            "- `{}` has {ordering} ordering. Each non-exited cluster thread must arrive exactly once before completion, then execute the matching wait. {alignment}",
            record.id,
        )
        .unwrap();
    }
    if cluster_barriers(catalog).next().is_some() {
        output.push_str(
            "- `cuda_device::cluster::cluster_sync` is the generated compatibility operation: aligned arrive followed by aligned wait.\n",
        );
    }
    if cluster_memory(catalog).next().is_some() {
        output.push_str("\n## Cluster-memory contracts\n\n");
        for record in cluster_memory(catalog) {
            match record.cluster_memory.as_ref().unwrap().source_contract {
                ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx => {
                    writeln!(
                        output,
                        "- `{}` preserves LLVM 22's address-space-7 result. Both backends use exact convergent `mapa.shared::cluster.u64` inline PTX, and ordinary Rust dereference of the mapped pointer lowers through cluster shared memory.",
                        record.id
                    )
                    .unwrap();
                }
                ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad => {
                    writeln!(
                        output,
                        "- `{}` has no one-to-one LLVM intrinsic. Both backends use one convergent `mapa.shared::cluster.u64` plus `ld.shared::cluster.u32` block with a compiler memory clobber.",
                        record.id
                    )
                    .unwrap();
                }
            }
        }
    }
    if movmatrix(catalog).next().is_some() {
        output.push_str("\n## movmatrix contracts\n\n");
        for record in movmatrix(catalog) {
            writeln!(
                output,
                "- `{}` transposes one packed b16 fragment in registers. All warp lanes must execute the same instruction, and no lane may have exited.",
                record.id,
            )
            .unwrap();
        }
    }
    if mbarrier_extended(catalog).next().is_some() {
        output.push_str("\n## Extended mbarrier contracts\n\n");
        for record in mbarrier_extended(catalog) {
            let contract = record.mbarrier_extended.as_ref().unwrap();
            let source = match contract.source_contract {
                MbarrierExtendedSourceContract::LlvmImported => "LLVM 22 TableGen",
                MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => {
                    "PTX-native raw-address contract"
                }
            };
            writeln!(
                output,
                "- `{}` keeps its established CUDA-device signature and exact convergent inline-PTX memory clobber. Its source identity is {source}.",
                record.id,
            )
            .unwrap();
        }
    }
    if wgmma_controls(catalog).next().is_some() {
        output.push_str("\n## WGMMA-control contracts\n\n");
        for record in wgmma_controls(catalog) {
            let detail = match record.wgmma_control.as_ref().unwrap().mode {
                WgmmaControlMode::Fence => {
                    "It orders register writes before later WGMMA operations."
                }
                WgmmaControlMode::CommitGroup => {
                    "It commits the warpgroup's prior uncommitted WGMMA operations."
                }
                WgmmaControlMode::WaitGroup => {
                    "Its `u64` argument must be a compile-time constant."
                }
            };
            writeln!(
                output,
                "- `{}` must be executed by all four warps in the warpgroup with the same control flow. {detail}",
                record.id,
            )
            .unwrap();
        }
    }
    if sync_intrinsics(catalog).any(|record| threadfence_ptx_level(record).is_some()) {
        output.push_str("\n## Synchronization contracts\n\n");
    } else {
        output.push_str("\n## CTA synchronization contracts\n\n");
    }
    for record in sync_intrinsics(catalog) {
        if record.id == "sync_threads" {
            writeln!(
                output,
                "- `{}` inserts the fixed barrier ID `0`. Every active CTA thread must reach the same barrier; divergent use can deadlock the CTA.",
                record.id,
            )
            .unwrap();
        } else {
            let level = threadfence_ptx_level(record)
                .expect("resolver admitted an unknown generated synchronization record");
            writeln!(
                output,
                "- `{}` emits `membar.{level}` through the reviewed typed NVVM route on both backends.",
                record.id,
            )
            .unwrap();
        }
    }
    output.push_str("\n## cp.async mbarrier contracts\n\n");
    for record in cp_async_mbarriers(catalog) {
        let bridge = record.cp_async_mbarrier.as_ref().unwrap();
        let address = match bridge.state_space {
            CpAsyncMbarrierStateSpace::Generic => "generic",
            CpAsyncMbarrierStateSpace::Shared => "explicit shared",
        };
        let counting = match bridge.operation {
            CpAsyncMbarrierOperation::Arrive => {
                "increments the pending count before the later asynchronous decrement"
            }
            CpAsyncMbarrierOperation::ArriveNoInc => {
                "does not increment the pending count, so initialization must already include the later asynchronous decrement"
            }
        };
        writeln!(
            output,
            "- `{}` uses {address} addressing and {counting}. It associates only the executing thread's prior `cp.async` operations with a live, eight-byte-aligned shared-memory barrier. LLVM-NVPTX uses the typed intrinsic; libNVVM uses reviewed convergent, side-effecting inline PTX with a memory clobber.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Basic mbarrier contracts\n\n");
    for record in mbarrier_basics(catalog) {
        let mbarrier = record.mbarrier_basic.as_ref().unwrap();
        let contract = match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                "initializes one eight-byte-aligned shared-memory object exactly once with an expected count in `1..=0xFFFFF`"
            }
            MbarrierBasicOperation::Arrive => {
                "records one expected arrival and returns a token for the same barrier phase"
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                "records a dynamic arrival count without completing the current phase and returns the prior opaque state"
            }
            MbarrierBasicOperation::TestWait => {
                "tests one token from the same barrier phase with convergent, side-effecting inline PTX"
            }
            MbarrierBasicOperation::Inval => {
                "invalidates an initialized barrier only after all users are finished"
            }
        };
        let mechanism = if mbarrier.operation == MbarrierBasicOperation::TestWait {
            "inline PTX on both backends"
        } else {
            "the typed NVVM intrinsic with LLVM-NVPTX and inline PTX with libNVVM"
        };
        writeln!(
            output,
            "- `{}` {contract}. It lowers through {mechanism}; generic MIR pointers are normalized to shared address space during lowering.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Imported LLVM properties and result facts\n\n");
    for record in &catalog.intrinsics {
        let Some(llvm) = &record.llvm else {
            writeln!(
                output,
                "- `{}`: PTX-native source; no LLVM record, symbol, properties, or selection facts.",
                record.id
            )
            .unwrap();
            continue;
        };
        let properties = record
            .llvm
            .as_ref()
            .unwrap()
            .properties
            .iter()
            .map(|property| format!("`{property}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let range = record
            .llvm
            .as_ref()
            .unwrap()
            .result_facts
            .range
            .as_ref()
            .map(|range| format!("[{}, {})", range.lower, range.upper_exclusive))
            .unwrap_or_else(|| "none".to_owned());
        let selection_records = record
            .selections
            .iter()
            .map(|selection| format!("`{}`", selection.source_record))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        let selection_predicates = record
            .selections
            .iter()
            .flat_map(|selection| selection.predicates.iter())
            .map(|predicate| format!("`{predicate}`"))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "- `{}`: properties {}; result `NoUndef` {}; half-open result range `{}`; selection records {}; selection predicates {}",
            record.id,
            properties,
            llvm.result_facts.no_undef,
            range,
            selection_records,
            if selection_predicates.is_empty() {
                "none"
            } else {
                &selection_predicates
            }
        )
        .unwrap();
    }
    output.push_str("\n## Backend-specific lowering evidence\n\n");
    for record in &catalog.intrinsics {
        if record.backend_lowerings.is_empty() {
            continue;
        }
        let runtime = record
            .ldmatrix
            .as_ref()
            .map(|record| format!("{:?}", record.safety.runtime_validation).to_lowercase())
            .or_else(|| {
                record
                    .packed_atomic
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .mbarrier_basic
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .movmatrix
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .mbarrier_extended
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .cp_async_mbarrier
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .register_mma
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .sparse_mma
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .debug_control
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .cluster_memory
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| {
                record
                    .tcgen05
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .or_else(|| (record.family == "stmatrix").then(|| "unexecuted".to_owned()))
            .unwrap_or_else(|| "not recorded".to_owned());
        writeln!(output, "- `{}`: runtime `{runtime}`", record.id).unwrap();
        for lowering in &record.backend_lowerings {
            writeln!(
                output,
                "  - `{}` uses `{}` from profile `{}` at PTX {} / {}: status `{}` (`{}`, SHA-256 `{}`)",
                backend_label(lowering.backend),
                lowering_mechanism_label(lowering.mechanism),
                lowering.evidence_profile,
                lowering.target.minimum_ptx,
                hardware_target_label(&lowering.target.hardware),
                lowering.status,
                lowering.version,
                lowering.sha256,
            )
            .unwrap();
            for stage in &lowering.stages {
                let tool = match (
                    stage.tool_path.as_deref(),
                    stage.tool_version.as_deref(),
                    stage.tool_sha256.as_deref(),
                ) {
                    (Some(path), Some(version), Some(sha256)) => {
                        format!(" Tool `{path}` reports `{version}` (SHA-256 `{sha256}`).")
                    }
                    _ => String::new(),
                };
                let artifact = match stage.artifact_kind {
                    Some(EvidenceArtifactKind::Cubin) => " Artifact `cubin`.",
                    None => "",
                };
                writeln!(
                    output,
                    "    - {} on `{}`: `{}` — {}{}{}",
                    evidence_stage_label(stage.stage),
                    stage.targets.join(", "),
                    stage.outcome,
                    stage.detail,
                    tool,
                    artifact,
                )
                .unwrap();
            }
        }
    }
    output
}

pub(super) fn render_compiler_path_patterns(
    output: &mut String,
    catalog: &CatalogFile,
    indent: &str,
) {
    let paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .flat_map(|record| {
            std::iter::once(record.rust.canonical_path.as_str())
                .chain(record.rust.compatibility_paths.iter().map(String::as_str))
        })
        .collect();
    render_string_patterns(output, &paths, indent);
}

pub(super) fn render_string_patterns(output: &mut String, values: &[&str], indent: &str) {
    for (index, value) in values.iter().enumerate() {
        if index == 0 {
            writeln!(output, "{indent}{value:?}").unwrap();
        } else {
            writeln!(output, "{indent}| {value:?}").unwrap();
        }
    }
}

pub(super) fn render_inline_patterns(output: &mut String, values: &[&str]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(" | ");
        }
        write!(output, "{value:?}").unwrap();
    }
}

#[allow(dead_code)]
pub(super) fn modules(catalog: &CatalogFile) -> BTreeSet<&str> {
    catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.module.as_str())
        .collect()
}
