/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, CatalogIntrinsic, ClcAdapter, ClusterMemoryOperation,
    CpAsyncControlOperation, CpAsyncMbarrierStateSpace, CpAsyncSourceSize, DebugControlOperation,
    DotProductAdapter, ExecutionControlOperation, ExtendedMinMaxFormat, IntrinsicBackend,
    MbarrierBasicOperation, MbarrierExtendedAdapter, PackedAtomicFormat,
    PackedConversionSourceFormat, ReduxAdapter, RegisterMmaAdapter, SparseMmaAccumulator,
    SpecialRegisterObservation, Tcgen05LdShape, Tcgen05Mma, Tcgen05MmaBUsage, Tcgen05MmaForm,
    Tcgen05MmaKind, Tcgen05Operation, TmaOperation, VoteAdapter, VoteMode, WarpBarrierAdapter,
    WarpMatchMode, WarpShuffleAdapter, WarpShuffleMode, WarpShuffleValueKind, WgmmaControlMode,
};
use crate::render::common::{backend_label, llvm, llvm_header};
use crate::render::families::{
    extended_minmax_contract, extended_minmax_ptx_mnemonic, integer_minmax_ptx_mnemonic,
    movmatrix_template, packed_alu_ptx_mnemonic, packed_alu_register_constraint, packed_alu_width,
    packed_conversion_constraint, packed_conversion_dialect_type, packed_conversion_ptx_mnemonic,
    packed_conversion_source, packed_conversion_source_width, packed_conversion_typed_llvm_name,
    register_mma_constraints, register_mma_fragment_counts, register_mma_template,
    scalar_arithmetic_arity, scalar_arithmetic_llvm_mechanism, scalar_arithmetic_llvm_type,
    scalar_arithmetic_ptx_mnemonic, scalar_math_llvm_mechanism, scalar_math_llvm_type,
    scalar_math_ptx_mnemonic, sparse_mma_constraints, sparse_mma_fragment_counts,
    sparse_mma_selector_values, sparse_mma_template, special_register_backend_mechanism,
    special_register_inline_template, special_register_output_constraint, stmatrix_variant,
    tcgen05_inline_asm, tcgen05_ld_register_count, tcgen05_mma_inline_asm, tcgen05_mma_is_ws,
    tcgen05_mma_runtime_parameters, tcgen05_st_register_count, threadfence_ptx_level,
};
use std::fmt::Write as _;

pub(super) fn render_special_register_probe(
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
    hash: &str,
    backend: IntrinsicBackend,
) -> String {
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    let width = record.scalar_width().expect("special-register width");
    let function_name = format!("probe_{}_{}", record.id, backend_label(backend));
    match special_register_backend_mechanism(record, backend) {
        BackendLoweringMechanism::TypedNvvm => {
            let symbol = &record
                .llvm
                .as_ref()
                .expect("typed special-register route")
                .symbol;
            writeln!(output, "declare i{width} @{symbol}()\n").unwrap();
            writeln!(output, "define i{width} @{function_name}() {{").unwrap();
            writeln!(output, "  %result = call i{width} @{symbol}()").unwrap();
        }
        BackendLoweringMechanism::InlinePtx => {
            writeln!(output, "define i{width} @{function_name}() {{").unwrap();
            let sideeffect = if record.special_register.as_ref().is_some_and(|special| {
                special.observation == SpecialRegisterObservation::VolatileObservation
            }) {
                " sideeffect"
            } else {
                ""
            };
            writeln!(
                output,
                "  %result = call i{width} asm{sideeffect} {:?}, {:?}()",
                special_register_inline_template(record),
                special_register_output_constraint(record)
            )
            .unwrap();
        }
    }
    writeln!(output, "  ret i{width} %result\n}}").unwrap();
    output
}

fn render_tma_probe(catalog: &CatalogFile, record: &CatalogIntrinsic, hash: &str) -> String {
    let tma = record.tma.as_ref().expect("TMA contract");
    let operation = tma.operation;
    let llvm = llvm(record);
    let symbol = llvm.resolved_symbol.as_ref().unwrap_or(&llvm.symbol);
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    if let Some(dimensions) = tma.dimensions() {
        let is_g2s = matches!(
            operation,
            TmaOperation::G2sTile1d
                | TmaOperation::G2sTile2d
                | TmaOperation::G2sTile2dMulticast
                | TmaOperation::G2sTile2dMulticastCg2
                | TmaOperation::G2sTile3d
                | TmaOperation::G2sTile4d
                | TmaOperation::G2sTile5d
        );
        let coordinates = std::iter::repeat_n("i32", dimensions)
            .collect::<Vec<_>>()
            .join(", ");
        let coordinate_parameters = (0..dimensions)
            .map(|index| format!("i32 %coord{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let coordinate_arguments = (0..dimensions)
            .map(|index| format!("i32 %coord{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        if is_g2s {
            let multicast = matches!(
                operation,
                TmaOperation::G2sTile2dMulticast | TmaOperation::G2sTile2dMulticastCg2
            );
            let declaration_coordinates = if coordinates.is_empty() {
                String::new()
            } else {
                format!(", {coordinates}")
            };
            writeln!(
                output,
                "declare void @{symbol}(ptr addrspace(7), ptr addrspace(3), ptr{declaration_coordinates}, i16, i64, i1, i1, i32) #0\n"
            )
            .unwrap();
            let parameters = if coordinate_parameters.is_empty() {
                String::new()
            } else {
                format!(", {coordinate_parameters}")
            };
            let mask_parameter = if multicast { ", i16 %cta_mask" } else { "" };
            writeln!(
                output,
                "define void @probe_{}(ptr %dst_generic, ptr %barrier_generic, ptr %tensor_map{parameters}{mask_parameter}) #0 {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %dst = addrspacecast ptr %dst_generic to ptr addrspace(7)\n  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
            );
            let arguments = if coordinate_arguments.is_empty() {
                String::new()
            } else {
                format!(", {coordinate_arguments}")
            };
            let mask = if multicast { "%cta_mask" } else { "0" };
            let group = if operation == TmaOperation::G2sTile2dMulticastCg2 {
                2
            } else {
                0
            };
            writeln!(
                output,
                "  call void @{symbol}(ptr addrspace(7) %dst, ptr addrspace(3) %barrier, ptr %tensor_map{arguments}, i16 {mask}, i64 0, i1 {multicast}, i1 false, i32 {group}) #0"
            )
            .unwrap();
        } else {
            let declaration_coordinates = if coordinates.is_empty() {
                String::new()
            } else {
                format!(", {coordinates}")
            };
            writeln!(
                output,
                "declare void @{symbol}(ptr addrspace(3), ptr{declaration_coordinates}, i64, i1) #0\n"
            )
            .unwrap();
            let parameters = if coordinate_parameters.is_empty() {
                String::new()
            } else {
                format!(", {coordinate_parameters}")
            };
            writeln!(
                output,
                "define void @probe_{}(ptr %src_generic, ptr %tensor_map{parameters}) #0 {{",
                record.id
            )
            .unwrap();
            output.push_str("  %src = addrspacecast ptr %src_generic to ptr addrspace(3)\n");
            let arguments = if coordinate_arguments.is_empty() {
                String::new()
            } else {
                format!(", {coordinate_arguments}")
            };
            writeln!(
                output,
                "  call void @{symbol}(ptr addrspace(3) %src, ptr %tensor_map{arguments}, i64 0, i1 false) #0"
            )
            .unwrap();
        }
        output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    } else if operation == TmaOperation::PrefetchTensorMap {
        writeln!(output, "declare void @{symbol}(ptr)\n").unwrap();
        writeln!(
            output,
            "define void @probe_{}(ptr %tensor_map) {{",
            record.id
        )
        .unwrap();
        writeln!(output, "  call void @{symbol}(ptr %tensor_map)").unwrap();
        output.push_str("  ret void\n}\n");
    } else if let Some(coordinate_count) = operation.prefetch_coordinate_count() {
        let coordinates = std::iter::repeat_n("i32", coordinate_count)
            .collect::<Vec<_>>()
            .join(", ");
        let declaration_coordinates = if coordinates.is_empty() {
            String::new()
        } else {
            format!(", {coordinates}")
        };
        let coordinate_parameters = (0..coordinate_count)
            .map(|index| format!("i32 %coord{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let parameters = if coordinate_parameters.is_empty() {
            String::new()
        } else {
            format!(", {coordinate_parameters}")
        };
        let cache_hint_parameter = if operation.uses_prefetch_cache_hint() {
            ", i64 %cache_hint"
        } else {
            ""
        };
        let coordinate_arguments = (0..coordinate_count)
            .map(|index| format!("i32 %coord{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = if coordinate_arguments.is_empty() {
            String::new()
        } else {
            format!(", {coordinate_arguments}")
        };
        writeln!(
            output,
            "declare void @{symbol}(ptr{declaration_coordinates}, i64, i1) #0\n"
        )
        .unwrap();
        writeln!(
            output,
            "define void @probe_{}(ptr %tensor_map{parameters}{cache_hint_parameter}) #0 {{",
            record.id,
        )
        .unwrap();
        let cache_hint = if operation.uses_prefetch_cache_hint() {
            "%cache_hint"
        } else {
            "0"
        };
        let use_cache_hint = operation.uses_prefetch_cache_hint();
        writeln!(
            output,
            "  call void @{symbol}(ptr %tensor_map{arguments}, i64 {cache_hint}, i1 {use_cache_hint}) #0"
        )
        .unwrap();
        output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    } else if matches!(
        operation,
        TmaOperation::ReplaceBoxDim
            | TmaOperation::ReplaceElementStride
            | TmaOperation::ReplaceElementType
            | TmaOperation::ReplaceFillMode
            | TmaOperation::ReplaceGlobalAddress
            | TmaOperation::ReplaceGlobalDim
            | TmaOperation::ReplaceGlobalStride
            | TmaOperation::ReplaceInterleaveLayout
            | TmaOperation::ReplaceRank
            | TmaOperation::ReplaceSwizzleAtomicity
            | TmaOperation::ReplaceSwizzleMode
    ) {
        let (field, width, ordinal, value_ty, immediate, address) = match operation {
            TmaOperation::ReplaceBoxDim => ("box_dim", "b32", true, "i32", false, false),
            TmaOperation::ReplaceElementStride => {
                ("element_stride", "b32", true, "i32", false, false)
            }
            TmaOperation::ReplaceElementType => ("elemtype", "b32", false, "i32", true, false),
            TmaOperation::ReplaceFillMode => ("fill_mode", "b32", false, "i32", true, false),
            TmaOperation::ReplaceGlobalAddress => {
                ("global_address", "b64", false, "i64", false, true)
            }
            TmaOperation::ReplaceGlobalDim => ("global_dim", "b32", true, "i32", false, false),
            TmaOperation::ReplaceGlobalStride => {
                ("global_stride", "b64", true, "i64", false, false)
            }
            TmaOperation::ReplaceInterleaveLayout => {
                ("interleave_layout", "b32", false, "i32", true, false)
            }
            TmaOperation::ReplaceRank => ("rank", "b32", false, "i32", false, false),
            TmaOperation::ReplaceSwizzleAtomicity => {
                ("swizzle_atomicity", "b32", false, "i32", true, false)
            }
            TmaOperation::ReplaceSwizzleMode => ("swizzle_mode", "b32", false, "i32", true, false),
            _ => unreachable!("TMA tensor-map replace operation was matched"),
        };
        let value_parameter = if immediate {
            String::new()
        } else if address {
            ", ptr %new_address".into()
        } else {
            format!(", {value_ty} %new_value")
        };
        writeln!(
            output,
            "define void @probe_{}(ptr %tensor_map{value_parameter}) {{",
            record.id
        )
        .unwrap();
        let mut arguments = vec!["ptr %tensor_map".to_owned()];
        let mut constraints = vec!["l"];
        if ordinal {
            arguments.push("i32 0".into());
            constraints.push("n");
        }
        if immediate {
            arguments.push(format!("{value_ty} 0"));
            constraints.push("n");
        } else if address {
            arguments.push("ptr %new_address".into());
            constraints.push("l");
        } else {
            arguments.push(format!("{value_ty} %new_value"));
            constraints.push(if value_ty == "i64" { "l" } else { "r" });
        }
        constraints.push("~{memory}");
        let template = if ordinal {
            format!("tensormap.replace.tile.{field}.global.b1024.{width} [$0], $1, $2;")
        } else {
            format!("tensormap.replace.tile.{field}.global.b1024.{width} [$0], $1;")
        };
        writeln!(
            output,
            "  call void asm sideeffect {template:?}, {:?}({})",
            constraints.join(","),
            arguments.join(", ")
        )
        .unwrap();
        output.push_str("  ret void\n}\n");
    } else if matches!(
        operation,
        TmaOperation::FenceProxyTensorMapAcquireCluster
            | TmaOperation::FenceProxyTensorMapAcquireCta
            | TmaOperation::FenceProxyTensorMapAcquireGpu
            | TmaOperation::FenceProxyTensorMapAcquireSystem
    ) {
        writeln!(output, "declare void @{symbol}(ptr, i32)\n").unwrap();
        writeln!(
            output,
            "define void @probe_{}(ptr %tensor_map) {{",
            record.id
        )
        .unwrap();
        writeln!(output, "  call void @{symbol}(ptr %tensor_map, i32 128)").unwrap();
        output.push_str("  ret void\n}\n");
    } else if operation == TmaOperation::CommitGroup
        || matches!(
            operation,
            TmaOperation::FenceProxyTensorMapReleaseCluster
                | TmaOperation::FenceProxyTensorMapReleaseCta
                | TmaOperation::FenceProxyTensorMapReleaseGpu
                | TmaOperation::FenceProxyTensorMapReleaseSystem
        )
    {
        writeln!(output, "declare void @{symbol}()\n").unwrap();
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        writeln!(output, "  call void @{symbol}()").unwrap();
        output.push_str("  ret void\n}\n");
    } else {
        writeln!(output, "declare void @{symbol}(i32)\n").unwrap();
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        writeln!(output, "  call void @{symbol}(i32 0)").unwrap();
        output.push_str("  ret void\n}\n");
    }
    output
}

fn render_execution_control_probe(
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
    hash: &str,
) -> String {
    let operation = ExecutionControlOperation::from_catalog_id(&record.id)
        .expect("closed execution-control operation");
    let symbol = &llvm(record).symbol;
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    match operation {
        ExecutionControlOperation::BarrierCtaSync
        | ExecutionControlOperation::BarrierCtaSyncAligned
        | ExecutionControlOperation::BarrierCtaArrive
        | ExecutionControlOperation::BarrierCtaArriveAligned => {
            writeln!(output, "declare void @{symbol}(i32, i32)\n").unwrap();
            for (suffix, parameters, barrier_id, thread_count) in [
                (
                    "rr",
                    "i32 %barrier_id, i32 %thread_count",
                    "%barrier_id",
                    "%thread_count",
                ),
                ("ri", "i32 %barrier_id", "%barrier_id", "32"),
                ("ir", "i32 %thread_count", "1", "%thread_count"),
                ("ii", "", "1", "32"),
            ] {
                writeln!(
                    output,
                    "define void @probe_{}_{suffix}({parameters}) #0 {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  call void @{symbol}(i32 {barrier_id}, i32 {thread_count})"
                )
                .unwrap();
                output.push_str("  ret void\n}\n\n");
            }
            output.push_str("attributes #0 = { convergent }\n");
        }
        ExecutionControlOperation::GridDependencyLaunchDependents
        | ExecutionControlOperation::GridDependencyWait => {
            writeln!(output, "declare void @{symbol}()\n").unwrap();
            writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
            writeln!(output, "  call void @{symbol}()").unwrap();
            output.push_str("  ret void\n}\n");
        }
        ExecutionControlOperation::SetMaxNRegInc | ExecutionControlOperation::SetMaxNRegDec => {
            writeln!(output, "declare void @{symbol}(i32)\n").unwrap();
            writeln!(output, "define void @probe_{}() #0 {{", record.id).unwrap();
            writeln!(output, "  call void @{symbol}(i32 64)").unwrap();
            output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
        }
    }
    output
}

fn render_tcgen05_mma_probe(
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
    hash: &str,
    mma: &Tcgen05Mma,
) -> String {
    let (kind, b_buffer, b_usage) = mma.fixed_selectors.map_or(
        (Tcgen05MmaKind::F16, 0, Tcgen05MmaBUsage::Discard),
        |fixed| (fixed.kind, fixed.b_buffer, fixed.b_usage),
    );
    let (template, constraints) = if tcgen05_mma_is_ws(mma.form) {
        tcgen05_mma_inline_asm(mma.form, kind, 1, None, Some(b_buffer), Some(b_usage))
    } else {
        tcgen05_mma_inline_asm(mma.form, kind, 1, Some("discard"), None, None)
    };
    let parameters = tcgen05_mma_runtime_parameters(mma);
    let llvm_parameter = |(name, ty): &(&str, &str)| {
        let ty = match *ty {
            "u32" => "i32",
            "u64" => "i64",
            "bool" => "i1",
            _ => unreachable!("closed tcgen05 MMA probe type"),
        };
        format!("{ty} %{name}")
    };
    let signature = parameters
        .iter()
        .map(llvm_parameter)
        .collect::<Vec<_>>()
        .join(", ");
    let argument_indices: Vec<usize> =
        if mma.alias.is_some() && mma.form == Tcgen05MmaForm::WsTensor {
            vec![0, 1, 3, 4, 5]
        } else {
            (0..parameters.len()).collect()
        };
    let arguments = argument_indices
        .iter()
        .map(|&index| llvm_parameter(&parameters[index]))
        .collect::<Vec<_>>()
        .join(", ");

    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    writeln!(
        output,
        "; LLVM source: {}; route: inline PTX; source contract: {:?}",
        llvm(record).symbol,
        record.tcgen05.as_ref().unwrap().source_contract
    )
    .unwrap();
    writeln!(
        output,
        "define void @probe_{}({signature}) #0 {{",
        record.id
    )
    .unwrap();
    writeln!(
        output,
        "  call void asm sideeffect {template:?}, {constraints:?}({arguments}) #0"
    )
    .unwrap();
    output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    output
}

fn render_tcgen05_probe(catalog: &CatalogFile, record: &CatalogIntrinsic, hash: &str) -> String {
    let tcgen05 = record.tcgen05.as_ref().expect("tcgen05 contract");
    if let Some(mma) = &tcgen05.mma {
        return render_tcgen05_mma_probe(catalog, record, hash, mma);
    }
    let operation = tcgen05.operation;
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    writeln!(
        output,
        "; LLVM source: {}; route: inline PTX; source contract: {:?}",
        llvm(record).symbol,
        record.tcgen05.as_ref().unwrap().source_contract
    )
    .unwrap();
    let (parameters, result_ty, template, constraints, arguments) = match operation {
        Tcgen05Operation::Alloc | Tcgen05Operation::AllocCg2 => {
            let group = if operation == Tcgen05Operation::AllocCg2 {
                "2"
            } else {
                "1"
            };
            (
                "ptr %dst, i32 %ncols".into(),
                "void".into(),
                format!(
                    "{{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::{group}.sync.aligned.shared::cta.b32 [%shared32], $1; }}"
                ),
                "l,r,~{memory}".into(),
                "ptr %dst, i32 %ncols".into(),
            )
        }
        Tcgen05Operation::Dealloc | Tcgen05Operation::DeallocCg2 => {
            let group = if operation == Tcgen05Operation::DeallocCg2 {
                "2"
            } else {
                "1"
            };
            (
                "i32 %tmem, i32 %ncols".into(),
                "void".into(),
                format!("tcgen05.dealloc.cta_group::{group}.sync.aligned.b32 $0, $1;"),
                "r,r,~{memory}".into(),
                "i32 %tmem, i32 %ncols".into(),
            )
        }
        Tcgen05Operation::RelinquishAllocPermit | Tcgen05Operation::RelinquishAllocPermitCg2 => {
            let group = if operation == Tcgen05Operation::RelinquishAllocPermitCg2 {
                "2"
            } else {
                "1"
            };
            (
                String::new(),
                "void".into(),
                format!("tcgen05.relinquish_alloc_permit.cta_group::{group}.sync.aligned;"),
                "~{memory}".into(),
                String::new(),
            )
        }
        Tcgen05Operation::FenceBeforeThreadSync | Tcgen05Operation::FenceAfterThreadSync => {
            let modifier = if operation == Tcgen05Operation::FenceBeforeThreadSync {
                "before_thread_sync"
            } else {
                "after_thread_sync"
            };
            (
                String::new(),
                "void".into(),
                format!("tcgen05.fence::{modifier};"),
                "~{memory}".into(),
                String::new(),
            )
        }
        Tcgen05Operation::Commit
        | Tcgen05Operation::CommitSharedCluster
        | Tcgen05Operation::CommitCg2
        | Tcgen05Operation::CommitSharedClusterCg2 => {
            let group = if matches!(
                operation,
                Tcgen05Operation::CommitCg2 | Tcgen05Operation::CommitSharedClusterCg2
            ) {
                "2"
            } else {
                "1"
            };
            let shared = matches!(
                operation,
                Tcgen05Operation::CommitSharedCluster | Tcgen05Operation::CommitSharedClusterCg2
            );
            (
                "i32 %mbar".into(),
                "void".into(),
                format!(
                    "tcgen05.commit.cta_group::{group}.mbarrier::arrive::one{}.b64 [$0];",
                    if shared { ".shared::cluster" } else { "" }
                ),
                "r,~{memory}".into(),
                "i32 %mbar".into(),
            )
        }
        Tcgen05Operation::MmaWsF16 | Tcgen05Operation::MmaWsBf16 | Tcgen05Operation::MmaWsTf32 => {
            let kind = if operation == Tcgen05Operation::MmaWsTf32 {
                "tf32"
            } else {
                "f16"
            };
            (
                "i32 %d, i32 %a, i64 %legacy_a_desc, i64 %b, i32 %idesc, i1 %enable".into(),
                "void".into(),
                format!(
                    "{{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $5, 0; tcgen05.mma.ws.cta_group::1.kind::{kind} [$0], [$1], $3, $4, %enable_pred; }}"
                ),
                "r,r,l,l,r,r,~{memory}".into(),
                "i32 %d, i32 %a, i64 %legacy_a_desc, i64 %b, i32 %idesc, i1 %enable".into(),
            )
        }
        Tcgen05Operation::MmaF16 | Tcgen05Operation::MmaF16Cg2 => {
            let group = if operation == Tcgen05Operation::MmaF16Cg2 {
                "2"
            } else {
                "1"
            };
            let zeros = if operation == Tcgen05Operation::MmaF16Cg2 {
                "%z, %z, %z, %z, %z, %z, %z, %z"
            } else {
                "%z, %z, %z, %z"
            };
            (
                "i32 %d, i64 %a, i64 %b, i32 %idesc, i1 %enable".into(),
                "void".into(),
                format!(
                    "{{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::{group}.kind::f16 [$0], $1, $2, $3, {{{zeros}}}, %enable_pred; }}"
                ),
                "r,l,l,r,r,~{memory}".into(),
                "i32 %d, i64 %a, i64 %b, i32 %idesc, i1 %enable".into(),
            )
        }
        Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2 => (
            "i32 %tmem, i64 %desc".into(),
            "void".into(),
            format!(
                "{}.{} [$0], $1;",
                record.expected_ptx.mnemonic,
                record.expected_ptx.modifiers.join(".")
            ),
            "r,l,~{memory}".into(),
            "i32 %tmem, i64 %desc".into(),
        ),
        Tcgen05Operation::Ld16x256bX8Pure | Tcgen05Operation::Ld16x256bPure => {
            let count = if operation == Tcgen05Operation::Ld16x256bX8Pure {
                32
            } else {
                4
            };
            let registers = (0..count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(",");
            let constraints = std::iter::repeat_n("=f", count)
                .chain(std::iter::once("r"))
                .collect::<Vec<_>>()
                .join(",");
            let fields = std::iter::repeat_n("float", count)
                .collect::<Vec<_>>()
                .join(", ");
            (
                "i32 %tmem".into(),
                format!("{{ {fields} }}"),
                format!(
                    "tcgen05.ld.sync.aligned.16x256b.{}.b32 {{{registers}}}, [${count}];",
                    if count == 32 { "x8" } else { "x1" }
                ),
                constraints,
                "i32 %tmem".into(),
            )
        }
        Tcgen05Operation::St => {
            let count = tcgen05_st_register_count(record);
            let has_half_split_offset = record
                .tcgen05
                .as_ref()
                .and_then(|tcgen05| tcgen05.st)
                .is_some_and(|st| st.shape == Tcgen05LdShape::M16x32bx2);
            let parameters = std::iter::once("i32 %tmem".into())
                .chain((0..count).map(|index| format!("i32 %d{index}")))
                .collect::<Vec<_>>()
                .join(", ");
            let arguments = std::iter::once("i32 %tmem".into())
                .chain(has_half_split_offset.then(|| "i64 16".into()))
                .chain((0..count).map(|index| format!("i32 %d{index}")))
                .collect::<Vec<_>>()
                .join(", ");
            let (template, constraints, _) = tcgen05_inline_asm(record);
            (parameters, "void".into(), template, constraints, arguments)
        }
        Tcgen05Operation::Ld => {
            let count = tcgen05_ld_register_count(record);
            let has_half_split_offset = record
                .tcgen05
                .as_ref()
                .and_then(|tcgen05| tcgen05.ld)
                .is_some_and(|ld| ld.shape == Tcgen05LdShape::M16x32bx2);
            let fields = std::iter::repeat_n("i32", count)
                .collect::<Vec<_>>()
                .join(", ");
            let (template, constraints, _) = tcgen05_inline_asm(record);
            (
                "i32 %tmem".into(),
                if count == 1 {
                    "i32".into()
                } else {
                    format!("{{ {fields} }}")
                },
                template,
                constraints,
                if has_half_split_offset {
                    "i32 %tmem, i64 16".into()
                } else {
                    "i32 %tmem".into()
                },
            )
        }
        Tcgen05Operation::LoadWait | Tcgen05Operation::StoreWait => {
            let kind = if operation == Tcgen05Operation::LoadWait {
                "ld"
            } else {
                "st"
            };
            (
                String::new(),
                "void".into(),
                format!("tcgen05.wait::{kind}.sync.aligned;"),
                "~{memory}".into(),
                String::new(),
            )
        }
        Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2 => {
            let group = if operation == Tcgen05Operation::CommitMulticastCg2 {
                2
            } else {
                1
            };
            (
                "i32 %mbar, i16 %mask".into(),
                "void".into(),
                format!(
                    "tcgen05.commit.cta_group::{group}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;"
                ),
                "r,h,~{memory}".into(),
                "i32 %mbar, i16 %mask".into(),
            )
        }
        Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2 => {
            let group = if operation == Tcgen05Operation::ShiftDownCg2 {
                2
            } else {
                1
            };
            (
                "i32 %tmem".into(),
                "void".into(),
                format!("tcgen05.shift.cta_group::{group}.down [$0];"),
                "r,~{memory}".into(),
                "i32 %tmem".into(),
            )
        }
        Tcgen05Operation::Mma => unreachable!("generic MMA probe handled above"),
    };
    let (lowering_template, lowering_constraints, lowering_results) = tcgen05_inline_asm(record);
    assert_eq!(template, lowering_template);
    assert_eq!(constraints, lowering_constraints);
    assert_eq!(result_ty != "void", lowering_results.is_some());
    writeln!(
        output,
        "define void @probe_{}({parameters}) #0 {{",
        record.id
    )
    .unwrap();
    if result_ty == "void" {
        writeln!(
            output,
            "  call void asm sideeffect {template:?}, {constraints:?}({arguments}) #0"
        )
        .unwrap();
    } else {
        writeln!(
            output,
            "  %result = call {result_ty} asm sideeffect {template:?}, {constraints:?}({arguments}) #0"
        )
        .unwrap();
    }
    output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    output
}

pub(super) fn render_elect_probe(
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
    hash: &str,
    backend: IntrinsicBackend,
) -> String {
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    let mechanism = record
        .backend_lowerings
        .iter()
        .find(|route| route.backend == backend)
        .expect("elect backend route")
        .mechanism;
    match mechanism {
        BackendLoweringMechanism::TypedNvvm => {
            writeln!(
                output,
                "declare {{ i32, i1 }} @{}(i32) #0\n",
                llvm(record).symbol
            )
            .unwrap();
            for (suffix, parameters, mask) in
                [("register", "i32 %mask", "%mask"), ("immediate", "", "-1")]
            {
                writeln!(
                    output,
                    "define {{ i32, i1 }} @probe_{}_{}_{suffix}({parameters}) #0 {{",
                    record.id,
                    backend_label(backend)
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call {{ i32, i1 }} @{}(i32 {mask}) #0",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret { i32, i1 } %result\n}\n");
            }
        }
        BackendLoweringMechanism::InlinePtx => {
            for (suffix, parameters, mask) in [
                ("register", "i32 %mask", "%mask"),
                ("constant_mask", "", "-1"),
            ] {
                writeln!(
                    output,
                    "define {{ i32, i1 }} @probe_{}_{}_{suffix}({parameters}) #0 {{",
                    record.id,
                    backend_label(backend)
                )
                .unwrap();
                writeln!(
                    output,
                    "  %raw = call {{ i32, i32 }} asm sideeffect \"{{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }}\", \"=r,=r,r\"(i32 {mask}) #0"
                )
                .unwrap();
                output.push_str(
                    "  %leader = extractvalue { i32, i32 } %raw, 0\n\
                     \x20 %predicate.i32 = extractvalue { i32, i32 } %raw, 1\n\
                     \x20 %predicate = trunc i32 %predicate.i32 to i1\n\
                     \x20 %result.0 = insertvalue { i32, i1 } poison, i32 %leader, 0\n\
                     \x20 %result.1 = insertvalue { i32, i1 } %result.0, i1 %predicate, 1\n\
                     \x20 ret { i32, i1 } %result.1\n}\n",
                );
            }
        }
    }
    output.push_str("\nattributes #0 = { convergent }\n");
    output
}

pub(crate) fn render_probe(catalog: &CatalogFile, record: &CatalogIntrinsic, hash: &str) -> String {
    if record.special_register.is_some() {
        return render_special_register_probe(catalog, record, hash, IntrinsicBackend::LlvmNvptx);
    }
    if record.tma.is_some() {
        return render_tma_probe(catalog, record, hash);
    }
    if ExecutionControlOperation::from_catalog_id(&record.id).is_some() {
        return render_execution_control_probe(catalog, record, hash);
    }
    if record.tcgen05.is_some() {
        return render_tcgen05_probe(catalog, record, hash);
    }
    if record.family == "elect" {
        return render_elect_probe(catalog, record, hash, IntrinsicBackend::LlvmNvptx);
    }
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    if let Some(cluster) = &record.cluster_memory {
        let (template, constraints) =
            crate::resolve::cluster_memory_inline_recipe(cluster.operation);
        match cluster.operation {
            ClusterMemoryOperation::MapSharedRank => {
                writeln!(
                    output,
                    "define ptr addrspace(7) @probe_{}(ptr %source_generic, i32 %rank) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %source = addrspacecast ptr %source_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %mapped_integer = call i64 asm sideeffect {template:?}, {constraints:?}(ptr addrspace(3) %source, i32 %rank) #0"
                )
                .unwrap();
                output.push_str(
                    "  %mapped = inttoptr i64 %mapped_integer to ptr addrspace(7)\n  ret ptr addrspace(7) %mapped\n}\n\nattributes #0 = { convergent }\n",
                );
            }
            ClusterMemoryOperation::ReadU32 => {
                writeln!(
                    output,
                    "define i32 @probe_{}(ptr %source_generic, i32 %rank) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %source = addrspacecast ptr %source_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %value = call i32 asm sideeffect {template:?}, {constraints:?}(ptr addrspace(3) %source, i32 %rank) #0"
                )
                .unwrap();
                output.push_str("  ret i32 %value\n}\n\nattributes #0 = { convergent }\n");
            }
        }
    } else if let Some(clc) = &record.clc {
        match clc.adapter {
            ClcAdapter::GenericPointersToShared => {
                writeln!(
                    output,
                    "declare void @{}(ptr addrspace(3), ptr addrspace(3))\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define void @probe_{}(ptr %response_generic, ptr %mbarrier_generic) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %response = addrspacecast ptr %response_generic to ptr addrspace(3)\n\
                     \x20 %mbarrier = addrspacecast ptr %mbarrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  call void @{}(ptr addrspace(3) %response, ptr addrspace(3) %mbarrier)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret void\n}\n");
            }
            ClcAdapter::PairU64ToI128BoolToU32 | ClcAdapter::PairU64ToI128U32 => {
                let llvm_result = if clc.adapter == ClcAdapter::PairU64ToI128BoolToU32 {
                    "i1"
                } else {
                    "i32"
                };
                writeln!(
                    output,
                    "declare {llvm_result} @{}(i128)\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define i32 @probe_{}(i64 %response_low, i64 %response_high) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %response_low_i128 = zext i64 %response_low to i128\n\
                     \x20 %response_high_i128 = zext i64 %response_high to i128\n\
                     \x20 %response_high_shifted = shl i128 %response_high_i128, 64\n\
                     \x20 %response = or i128 %response_low_i128, %response_high_shifted\n",
                );
                writeln!(
                    output,
                    "  %raw_result = call {llvm_result} @{}(i128 %response)",
                    llvm(record).symbol
                )
                .unwrap();
                if clc.adapter == ClcAdapter::PairU64ToI128BoolToU32 {
                    output.push_str(
                        "  %result = zext i1 %raw_result to i32\n\
                         \x20 ret i32 %result\n}\n",
                    );
                } else {
                    output.push_str("  ret i32 %raw_result\n}\n");
                }
            }
        }
    } else if let Some(bridge) = &record.cp_async_mbarrier {
        let shared = bridge.state_space == CpAsyncMbarrierStateSpace::Shared;
        let pointer_type = if shared { "ptr addrspace(3)" } else { "ptr" };
        writeln!(
            output,
            "declare void @{}({pointer_type})\n",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(
            output,
            "define void @probe_{}(ptr %barrier_generic) {{",
            record.id
        )
        .unwrap();
        if shared {
            output
                .push_str("  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n");
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %barrier)",
                llvm(record).symbol
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "  call void @{}(ptr %barrier_generic)",
                llvm(record).symbol
            )
            .unwrap();
        }
        output.push_str("  ret void\n}\n");
    } else if let Some(mbarrier) = &record.mbarrier_basic {
        match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                writeln!(
                    output,
                    "declare void @{}(ptr addrspace(3), i32)\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define void @probe_{}(ptr %barrier_generic, i32 %expected_count) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  call void @{}(ptr addrspace(3) %barrier, i32 %expected_count)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret void\n}\n");
            }
            MbarrierBasicOperation::Arrive => {
                writeln!(
                    output,
                    "declare i64 @{}(ptr addrspace(3))\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define i64 @probe_{}(ptr %barrier_generic) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %token = call i64 @{}(ptr addrspace(3) %barrier)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret i64 %token\n}\n");
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                writeln!(
                    output,
                    "declare i64 @{}(ptr addrspace(3), i32)\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define i64 @probe_{}(ptr %barrier_generic, i32 %count) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %state = call i64 @{}(ptr addrspace(3) %barrier, i32 %count)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret i64 %state\n}\n");
            }
            MbarrierBasicOperation::TestWait => {
                writeln!(
                    output,
                    "define i1 @probe_{}(ptr %barrier_generic, i64 %token) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n\
                     \x20 %result_i32 = call i32 asm sideeffect \"{ .reg .pred %p0; mbarrier.test_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }\", \"=r,l,l,~{memory}\"(ptr addrspace(3) %barrier, i64 %token) #0\n\
                     \x20 %result = trunc i32 %result_i32 to i1\n\
                     \x20 ret i1 %result\n\
                     }\n\n\
                     attributes #0 = { convergent }\n",
                );
            }
            MbarrierBasicOperation::Inval => {
                writeln!(
                    output,
                    "declare void @{}(ptr addrspace(3))\n",
                    llvm(record).symbol
                )
                .unwrap();
                writeln!(
                    output,
                    "define void @probe_{}(ptr %barrier_generic) {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  call void @{}(ptr addrspace(3) %barrier)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push_str("  ret void\n}\n");
            }
        }
    } else if record.movmatrix.is_some() {
        writeln!(output, "define i32 @probe_{}(i32 %value) #0 {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i32 asm {:?}, \"=r,r\"(i32 %value) #0",
            movmatrix_template(record),
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n\nattributes #0 = { convergent }\n");
    } else if let Some(mbarrier) = &record.mbarrier_extended {
        let (template, constraints) =
            crate::resolve::mbarrier_extended_inline_recipe(mbarrier.operation);
        match mbarrier.adapter {
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount => {
                writeln!(
                    output,
                    "define i64 @probe_{}(ptr %barrier_generic, i32 %bytes) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %token = call i64 asm sideeffect {template:?}, {constraints:?}(ptr addrspace(3) %barrier, i32 %bytes) #0"
                )
                .unwrap();
                output.push_str("  ret i64 %token\n}\n\nattributes #0 = { convergent }\n");
            }
            MbarrierExtendedAdapter::PointerTokenToPredicate => {
                writeln!(
                    output,
                    "define i1 @probe_{}(ptr %barrier_generic, i64 %token) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %result_i32 = call i32 asm sideeffect {template:?}, {constraints:?}(ptr addrspace(3) %barrier, i64 %token) #0"
                )
                .unwrap();
                output.push_str(
                    "  %result = trunc i32 %result_i32 to i1\n  ret i1 %result\n}\n\nattributes #0 = { convergent }\n",
                );
            }
            MbarrierExtendedAdapter::PointerParityToPredicate => {
                writeln!(
                    output,
                    "define i1 @probe_{}(ptr %barrier_generic, i32 %parity) #0 {{",
                    record.id
                )
                .unwrap();
                output.push_str(
                    "  %barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)\n",
                );
                writeln!(
                    output,
                    "  %result_i32 = call i32 asm sideeffect {template:?}, {constraints:?}(ptr addrspace(3) %barrier, i32 %parity) #0"
                )
                .unwrap();
                output.push_str(
                    "  %result = trunc i32 %result_i32 to i1\n  ret i1 %result\n}\n\nattributes #0 = { convergent }\n",
                );
            }
            MbarrierExtendedAdapter::RawClusterAddressToVoid => {
                writeln!(
                    output,
                    "define void @probe_{}(i64 %address) #0 {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  call void asm sideeffect {template:?}, {constraints:?}(i64 %address) #0"
                )
                .unwrap();
                output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
            }
            MbarrierExtendedAdapter::ZeroOperandsToVoid => {
                writeln!(output, "define void @probe_{}() #0 {{", record.id).unwrap();
                writeln!(
                    output,
                    "  call void asm sideeffect {template:?}, {constraints:?}() #0"
                )
                .unwrap();
                output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
            }
            MbarrierExtendedAdapter::NanosecondsToVoid => {
                writeln!(output, "define void @probe_{}(i32 %ns) #0 {{", record.id).unwrap();
                writeln!(
                    output,
                    "  call void asm sideeffect {template:?}, {constraints:?}(i32 %ns) #0"
                )
                .unwrap();
                output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
            }
        }
    } else if record.integer_minmax.is_some() {
        let parameters = "i32 %arg0, i32 %arg1";
        writeln!(output, "define i32 @probe_{}({parameters}) {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i32 asm \"{} $0, $1, $2;\", \"=r,r,r\"(i32 %arg0, i32 %arg1)",
            integer_minmax_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if record.packed_alu.is_some() {
        let arity = record.rust.arguments.len();
        let width = packed_alu_width(record);
        let llvm_type = format!("i{width}");
        let register_constraint = packed_alu_register_constraint(record);
        let parameters = (0..arity)
            .map(|index| format!("{llvm_type} %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..arity)
            .map(|index| format!("{llvm_type} %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = (0..=arity)
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let constraints = std::iter::once(format!("={register_constraint}"))
            .chain(std::iter::repeat_n(register_constraint.to_owned(), arity))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            output,
            "define {llvm_type} @probe_{}({parameters}) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {llvm_type} asm \"{} {operands};\", \"{constraints}\"({arguments})",
            packed_alu_ptx_mnemonic(record)
        )
        .unwrap();
        writeln!(output, "  ret {llvm_type} %result\n}}").unwrap();
    } else if record.packed_conversion.is_some() {
        let result_ty = packed_conversion_dialect_type(record);
        if packed_conversion_typed_llvm_name(record).is_some() {
            let llvm = llvm(record);
            let symbol = llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol);
            writeln!(output, "declare {result_ty} @{symbol}(float, float)\n").unwrap();
            writeln!(
                output,
                "define {result_ty} @probe_{}(float %low, float %high) {{",
                record.id,
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result_ty} @{symbol}(float %high, float %low)"
            )
            .unwrap();
        } else if packed_conversion_source(record) == PackedConversionSourceFormat::F32x2 {
            writeln!(
                output,
                "define {result_ty} @probe_{}(float %low, float %high) {{",
                record.id,
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result_ty} asm \"{} $0, $2, $1;\", \"{}\"(float %low, float %high)",
                packed_conversion_ptx_mnemonic(record),
                packed_conversion_constraint(record),
            )
            .unwrap();
        } else {
            // A packed source arrives in one integer register, so the probe
            // takes a single operand and needs no reordering.
            let source_ty = format!("i{}", packed_conversion_source_width(record));
            writeln!(
                output,
                "define {result_ty} @probe_{}({source_ty} %packed) {{",
                record.id,
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result_ty} asm \"{} $0, $1;\", \"{}\"({source_ty} %packed)",
                packed_conversion_ptx_mnemonic(record),
                packed_conversion_constraint(record),
            )
            .unwrap();
        }
        writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
    } else if let Some(packed) = &record.packed_atomic {
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "f16x2",
            PackedAtomicFormat::Bf16x2 => "bf16x2",
        };
        writeln!(
            output,
            "define i32 @probe_{}(ptr %address, i32 %addend) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %old = call i32 asm sideeffect \"atom.global.add.noftz.{format} $0, [$1], $2;\", \"=r,l,r,~{{memory}}\"(ptr %address, i32 %addend)"
        )
        .unwrap();
        output.push_str("  ret i32 %old\n}\n");
    } else if let Some(copy) = &record.cp_async_copy {
        let dynamic_source_size = copy.source_size == CpAsyncSourceSize::Runtime;
        let declaration_arguments = if dynamic_source_size {
            "ptr addrspace(3), ptr addrspace(1), i32"
        } else {
            "ptr addrspace(3), ptr addrspace(1)"
        };
        writeln!(
            output,
            "declare void @{}({declaration_arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        if dynamic_source_size {
            writeln!(
                output,
                "define void @probe_{}_register(ptr %shared_generic, ptr %global_generic, i32 %source_size) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global, i32 %source_size)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
            writeln!(
                output,
                "define void @probe_{}_immediate(ptr %shared_generic, ptr %global_generic) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global, i32 3)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
        } else {
            writeln!(
                output,
                "define void @probe_{}(ptr %shared_generic, ptr %global_generic) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
        }
    } else if let Some(control) = &record.cp_async_control {
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let declaration_arguments = if has_immediate { "i32" } else { "" };
        writeln!(
            output,
            "declare void @{}({declaration_arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        if has_immediate {
            writeln!(output, "  call void @{}(i32 3)", llvm(record).symbol).unwrap();
        } else {
            writeln!(output, "  call void @{}()", llvm(record).symbol).unwrap();
        }
        output.push_str("  ret void\n}\n");
    } else if let Some(control) = &record.wgmma_control {
        if control.mode == WgmmaControlMode::WaitGroup {
            writeln!(
                output,
                "declare void @{}(i64 immarg) #0\n",
                llvm(record).symbol
            )
            .unwrap();
        } else {
            writeln!(output, "declare void @{}() #0\n", llvm(record).symbol).unwrap();
        }
        writeln!(output, "define void @probe_{}() #0 {{", record.id).unwrap();
        if control.mode == WgmmaControlMode::WaitGroup {
            writeln!(output, "  call void @{}(i64 0) #0", llvm(record).symbol,).unwrap();
        } else {
            writeln!(output, "  call void @{}() #0", llvm(record).symbol,).unwrap();
        }
        output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    } else if record.cluster_barrier.is_some() {
        writeln!(output, "declare void @{}() #0", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define void @probe_{}() #0 {{", record.id).unwrap();
        writeln!(output, "  call void @{}() #0", llvm(record).symbol).unwrap();
        output.push_str("  ret void\n}\n\nattributes #0 = { convergent }\n");
    } else if record.family == "sync" {
        let arguments = if record.id == "sync_threads" {
            "i32"
        } else {
            debug_assert!(threadfence_ptx_level(record).is_some());
            ""
        };
        writeln!(output, "declare void @{}({arguments})", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        let call_arguments = if record.id == "sync_threads" {
            "i32 0"
        } else {
            ""
        };
        writeln!(
            output,
            "  call void @{}({call_arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret void\n}\n");
    } else if let Some(warp_barrier) = &record.warp_barrier {
        debug_assert_eq!(warp_barrier.adapter, WarpBarrierAdapter::DirectMemberMask);
        writeln!(output, "declare void @{}(i32)", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(
            output,
            "define void @probe_{}(i32 %member_mask) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  call void @{}(i32 %member_mask)",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret void\n}\n");
        writeln!(output, "define void @probe_{}_immediate() {{", record.id).unwrap();
        writeln!(output, "  call void @{}(i32 -1)", llvm(record).symbol).unwrap();
        output.push_str("  ret void\n}\n");
    } else if let Some(dot) = &record.dot_product {
        match dot.adapter {
            DotProductAdapter::DirectThreeOperands => {
                writeln!(
                    output,
                    "declare i32 @{}(i32, i32, i32)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push('\n');
                writeln!(
                    output,
                    "define i32 @probe_{}(i32 %a, i32 %b, i32 %c) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call i32 @{}(i32 %a, i32 %b, i32 %c)",
                    llvm(record).symbol
                )
                .unwrap();
            }
            DotProductAdapter::InsertLowHalfFalse => {
                writeln!(
                    output,
                    "declare i32 @{}(i32, i32, i1, i32)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push('\n');
                writeln!(
                    output,
                    "define i32 @probe_{}(i32 %a, i32 %b, i32 %c) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call i32 @{}(i32 %a, i32 %b, i1 false, i32 %c)",
                    llvm(record).symbol
                )
                .unwrap();
            }
        }
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(vote) = &record.vote {
        debug_assert_eq!(vote.adapter, VoteAdapter::DirectMaskPredicate);
        let result_ty = match vote.mode {
            VoteMode::All | VoteMode::Any | VoteMode::Uni => "i1",
            VoteMode::Ballot => "i32",
        };
        writeln!(
            output,
            "declare {result_ty} @{}(i32, i1)",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        writeln!(
            output,
            "define {result_ty} @probe_{}(i32 %member_mask, i1 %predicate) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {result_ty} @{}(i32 %member_mask, i1 %predicate)",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
        writeln!(
            output,
            "define {result_ty} @probe_{}_immediate(i1 %predicate) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {result_ty} @{}(i32 -1, i1 %predicate)",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
    } else if let Some(warp_match) = &record.warp_match {
        let value_ty = format!("i{}", warp_match.value_width.bits());
        let result_ty = match warp_match.mode {
            WarpMatchMode::Any => "i32".to_owned(),
            WarpMatchMode::All => "{ i32, i1 }".to_owned(),
        };
        writeln!(
            output,
            "declare {result_ty} @{}(i32, {value_ty})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        let forms = [
            (
                "rr",
                "i32 %member_mask, ",
                "i32 %member_mask",
                format!("{value_ty} %value"),
            ),
            ("ri", "", "i32 -1", format!("{value_ty} %value")),
            (
                "ir",
                "i32 %member_mask",
                "i32 %member_mask",
                format!("{value_ty} 7"),
            ),
            ("ii", "", "i32 -1", format!("{value_ty} 7")),
        ];
        for (suffix, first_parameter, mask, value) in forms {
            let parameters = match suffix {
                "rr" => format!("{first_parameter}{value_ty} %value"),
                "ri" => format!("{value_ty} %value"),
                "ir" => first_parameter.to_owned(),
                "ii" => String::new(),
                _ => unreachable!(),
            };
            writeln!(
                output,
                "define {result_ty} @probe_{}_{suffix}({parameters}) {{",
                record.id
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result_ty} @{}({mask}, {value})",
                llvm(record).symbol
            )
            .unwrap();
            writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
        }
    } else if let Some(warp_shuffle) = &record.warp_shuffle {
        if warp_shuffle.value_kind == WarpShuffleValueKind::I64 {
            debug_assert_eq!(
                warp_shuffle.adapter,
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
            );
            let mode = match warp_shuffle.mode {
                WarpShuffleMode::Idx => "idx",
                WarpShuffleMode::Bfly => "bfly",
                WarpShuffleMode::Down => "down",
                WarpShuffleMode::Up => "up",
            };
            let asm = format!(
                "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; shfl.sync.{mode}.b32 lo, lo, $2, {}, $3; shfl.sync.{mode}.b32 hi, hi, $2, {}, $3; mov.b64 $0, {{lo, hi}}; }}",
                warp_shuffle.clamp, warp_shuffle.clamp
            );
            writeln!(
                output,
                "define i64 @probe_{}(i32 %member_mask, i64 %value, i32 %lane) #0 {{",
                record.id
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call i64 asm sideeffect {asm:?}, \"=l,l,r,r\"(i64 %value, i32 %lane, i32 %member_mask) #0"
            )
            .unwrap();
            output.push_str("  ret i64 %result\n}\n\nattributes #0 = { convergent }\n");
        } else {
            debug_assert_eq!(
                warp_shuffle.adapter,
                WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
            );
            let value_ty = match warp_shuffle.value_kind {
                WarpShuffleValueKind::I32 => "i32",
                WarpShuffleValueKind::F32 => "float",
                WarpShuffleValueKind::I64 => unreachable!(),
            };
            writeln!(
                output,
                "declare {value_ty} @{}(i32, {value_ty}, i32, i32)",
                llvm(record).symbol
            )
            .unwrap();
            output.push('\n');
            let forms = [
                (
                    "rr",
                    format!("i32 %member_mask, {value_ty} %value, i32 %lane"),
                    "i32 %member_mask",
                    "i32 %lane",
                ),
                (
                    "ri",
                    format!("{value_ty} %value, i32 %lane"),
                    "i32 -1",
                    "i32 %lane",
                ),
                (
                    "ir",
                    format!("i32 %member_mask, {value_ty} %value"),
                    "i32 %member_mask",
                    "i32 1",
                ),
                ("ii", format!("{value_ty} %value"), "i32 -1", "i32 1"),
            ];
            for (suffix, parameters, member_mask, lane) in forms {
                writeln!(
                    output,
                    "define {value_ty} @probe_{}_{suffix}({parameters}) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call {value_ty} @{}({member_mask}, {value_ty} %value, {lane}, i32 {})",
                    llvm(record).symbol,
                    warp_shuffle.clamp,
                )
                .unwrap();
                writeln!(output, "  ret {value_ty} %result\n}}").unwrap();
            }
        }
    } else if let Some(redux) = &record.redux {
        debug_assert_eq!(redux.adapter, ReduxAdapter::MaskValueToSourceMemberMask);
        let value_type = match llvm(record).results[0].as_str() {
            "i32" => "i32",
            "f32" => "float",
            other => panic!("unsupported redux value type {other}"),
        };
        writeln!(
            output,
            "declare {value_type} @{}({value_type}, i32)",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        writeln!(
            output,
            "define {value_type} @probe_{}(i32 %member_mask, {value_type} %value) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {value_type} @{}({value_type} %value, i32 %member_mask)",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret {value_type} %result\n}}").unwrap();
    } else if let Some(sparse_mma) = &record.sparse_mma {
        let accumulator = match sparse_mma.accumulator {
            SparseMmaAccumulator::F16 => "i32",
            SparseMmaAccumulator::F32 => "float",
            SparseMmaAccumulator::S32 => "i32",
        };
        let (c_count, a_count, b_count, d_count) = sparse_mma_fragment_counts(record);
        let result = format!(
            "{{ {} }}",
            std::iter::repeat_n(accumulator, d_count)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let parameters = (0..c_count)
            .map(|index| format!("{accumulator} %c{index}"))
            .chain((0..a_count).map(|index| format!("i32 %a{index}")))
            .chain((0..b_count).map(|index| format!("i32 %b{index}")))
            .chain(std::iter::once("i32 %metadata".to_owned()))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..c_count)
            .map(|index| format!("{accumulator} %c{index}"))
            .chain((0..a_count).map(|index| format!("i32 %a{index}")))
            .chain((0..b_count).map(|index| format!("i32 %b{index}")))
            .chain(std::iter::once("i32 %metadata".to_owned()))
            .collect::<Vec<_>>()
            .join(", ");
        for selector in sparse_mma_selector_values(record) {
            writeln!(
                output,
                "define {result} @probe_{}_selector_{selector}({parameters}) #0 {{",
                record.id
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result} asm sideeffect {:?}, {:?}({arguments}, i32 {selector}) #0",
                sparse_mma_template(record),
                sparse_mma_constraints(record),
            )
            .unwrap();
            writeln!(output, "  ret {result} %result\n}}").unwrap();
        }
        output.push_str("\nattributes #0 = { convergent }\n");
    } else if let Some(mma) = &record.register_mma {
        let (c_count, a_count, b_count, d_count) = register_mma_fragment_counts(record);
        let (c_type, packed_type, result_type) = match mma.adapter {
            RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32
            | RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => ("i32", "i32", "i32"),
            RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
            | RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32
            | RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => {
                ("float", "i32", "float")
            }
            RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 => ("double", "double", "double"),
            RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32
            | RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
            | RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => ("i32", "i32", "i32"),
        };
        let mut parameter_values = (0..c_count)
            .map(|index| format!("{c_type} %c{index}"))
            .chain((0..a_count).map(|index| format!("{packed_type} %a{index}")))
            .chain((0..b_count).map(|index| format!("{packed_type} %b{index}")))
            .collect::<Vec<_>>();
        let mut argument_values = (0..c_count)
            .map(|index| format!("{c_type} %c{index}"))
            .chain((0..a_count).map(|index| format!("{packed_type} %a{index}")))
            .chain((0..b_count).map(|index| format!("{packed_type} %b{index}")))
            .collect::<Vec<_>>();
        if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 {
            let block_scale_parameters = [
                "i32 %scale_a",
                "i16 %byte_id_a",
                "i16 %thread_id_a",
                "i32 %scale_b",
                "i16 %byte_id_b",
                "i16 %thread_id_b",
            ];
            parameter_values.extend(block_scale_parameters.map(str::to_owned));
            argument_values.extend(block_scale_parameters.map(str::to_owned));
        }
        let parameters = parameter_values.join(", ");
        let arguments = argument_values.join(", ");
        let result = format!(
            "{{ {} }}",
            std::iter::repeat_n(result_type, d_count)
                .collect::<Vec<_>>()
                .join(", ")
        );
        writeln!(
            output,
            "define {result} @probe_{}({parameters}) #0 {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {result} asm sideeffect {:?}, {:?}({arguments}) #0",
            register_mma_template(record),
            register_mma_constraints(record),
        )
        .unwrap();
        output.push_str("  ret ");
        output.push_str(&result);
        output.push_str(" %result\n}\n\nattributes #0 = { convergent }\n");
    } else if let Some(debug) = &record.debug_control {
        let template = match debug.operation {
            DebugControlOperation::Trap => "trap;",
            DebugControlOperation::Breakpoint => "brkpt;",
            DebugControlOperation::Pmevent => "pmevent 15;",
        };
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        writeln!(output, "  call void asm sideeffect {:?}, \"\"()", template).unwrap();
        output.push_str("  ret void\n}\n");
    } else if record.family == "stmatrix" {
        let (multiplicity, _) = stmatrix_variant(record).expect("stmatrix variant");
        let count = multiplicity.register_count();
        let symbol = llvm(record)
            .resolved_symbol
            .as_ref()
            .expect("stmatrix resolved symbol");
        let declaration = std::iter::once("ptr addrspace(3)".to_owned())
            .chain(std::iter::repeat_n("i32".to_owned(), count))
            .collect::<Vec<_>>()
            .join(", ");
        let parameters = std::iter::once("ptr %generic".to_owned())
            .chain((0..count).map(|index| format!("i32 %r{index}")))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = std::iter::once("ptr addrspace(3) %shared".to_owned())
            .chain((0..count).map(|index| format!("i32 %r{index}")))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "declare void @{symbol}({declaration})\n").unwrap();
        writeln!(output, "define void @probe_{}({parameters}) {{", record.id).unwrap();
        output.push_str("  %shared = addrspacecast ptr %generic to ptr addrspace(3)\n");
        writeln!(output, "  call void @{symbol}({arguments})").unwrap();
        output.push_str("  ret void\n}\n");
    } else if record.prmt.is_some() {
        let arity = record.rust.arguments.len();
        let declaration = std::iter::repeat_n("i32", arity)
            .collect::<Vec<_>>()
            .join(", ");
        let parameters = (0..arity)
            .map(|index| format!("i32 %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..arity)
            .map(|index| format!("i32 %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "declare i32 @{}({declaration})\n",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "define i32 @probe_{}({parameters}) {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i32 @{}({arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(ldmatrix) = &record.ldmatrix {
        let register_count = ldmatrix.variant.register_count();
        let result_ty = if register_count == 1 {
            "i32".to_owned()
        } else {
            format!(
                "{{ {} }}",
                std::iter::repeat_n("i32", register_count)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let symbol = record
            .llvm
            .as_ref()
            .expect("ldmatrix LLVM facts")
            .resolved_symbol
            .as_ref()
            .expect("ldmatrix resolved symbol");
        writeln!(output, "declare {result_ty} @{symbol}(ptr addrspace(3))").unwrap();
        output.push('\n');
        writeln!(
            output,
            "define {result_ty} @probe_{}(ptr %generic) {{",
            record.id
        )
        .unwrap();
        output.push_str("  %shared = addrspacecast ptr %generic to ptr addrspace(3)\n");
        writeln!(
            output,
            "  %result = call {result_ty} @{symbol}(ptr addrspace(3) %shared)"
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}\n").unwrap();
    } else if record.scalar_arithmetic.is_some() {
        let ty = scalar_arithmetic_llvm_type(record);
        let arity = scalar_arithmetic_arity(record);
        let parameters = (0..arity)
            .map(|index| format!("{ty} %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..arity)
            .map(|index| format!("{ty} %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        if scalar_arithmetic_llvm_mechanism(record) == BackendLoweringMechanism::TypedNvvm {
            let declaration = std::iter::repeat_n(ty, arity)
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                output,
                "declare {ty} @{}({declaration})\n",
                llvm(record).symbol
            )
            .unwrap();
        }
        writeln!(output, "define {ty} @probe_{}({parameters}) {{", record.id).unwrap();
        match scalar_arithmetic_llvm_mechanism(record) {
            BackendLoweringMechanism::TypedNvvm => {
                writeln!(
                    output,
                    "  %result = call {ty} @{}({arguments})",
                    llvm(record).symbol
                )
                .unwrap();
            }
            BackendLoweringMechanism::InlinePtx => {
                let register = if ty == "double" { "d" } else { "f" };
                let constraints = std::iter::once(format!("={register}"))
                    .chain(std::iter::repeat_n(register.to_owned(), arity))
                    .collect::<Vec<_>>()
                    .join(",");
                let operands = (0..=arity)
                    .map(|index| format!("${index}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(
                    output,
                    "  %result = call {ty} asm {:?}, {:?}({arguments})",
                    format!("{} {operands};", scalar_arithmetic_ptx_mnemonic(record)),
                    constraints,
                )
                .unwrap();
            }
        }
        writeln!(output, "  ret {ty} %result\n}}").unwrap();
    } else if record.scalar_math.is_some() {
        let ty = scalar_math_llvm_type(record);
        let parameters = format!("{ty} %arg0");
        let arguments = format!("{ty} %arg0");
        if scalar_math_llvm_mechanism(record) == BackendLoweringMechanism::TypedNvvm {
            writeln!(output, "declare {ty} @{}({ty})\n", llvm(record).symbol).unwrap();
        }
        writeln!(output, "define {ty} @probe_{}({parameters}) {{", record.id).unwrap();
        match scalar_math_llvm_mechanism(record) {
            BackendLoweringMechanism::TypedNvvm => {
                writeln!(
                    output,
                    "  %result = call {ty} @{}({arguments})",
                    llvm(record).symbol
                )
                .unwrap();
            }
            BackendLoweringMechanism::InlinePtx => {
                let register = match ty {
                    "double" => "d",
                    "i16" => "h",
                    _ => "f",
                };
                writeln!(
                    output,
                    "  %result = call {ty} asm {:?}, {:?}({arguments})",
                    format!("{} $0, $1;", scalar_math_ptx_mnemonic(record)),
                    format!("={register},{register}"),
                )
                .unwrap();
            }
        }
        writeln!(output, "  ret {ty} %result\n}}").unwrap();
    } else if record.extended_minmax.is_some() {
        let (ty, register) = match extended_minmax_contract(record).format {
            ExtendedMinMaxFormat::F32 => ("float", "f"),
            ExtendedMinMaxFormat::F16 | ExtendedMinMaxFormat::Bf16 => ("i16", "h"),
            ExtendedMinMaxFormat::F16x2 | ExtendedMinMaxFormat::Bf16x2 => ("i32", "r"),
        };
        writeln!(
            output,
            "define {ty} @probe_{}({ty} %a, {ty} %b) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {ty} asm {:?}, {:?}({ty} %a, {ty} %b)",
            format!("{} $0, $1, $2;", extended_minmax_ptx_mnemonic(record)),
            format!("={register},{register},{register}"),
        )
        .unwrap();
        writeln!(output, "  ret {ty} %result\n}}").unwrap();
    } else if record.scalar_conversion.is_some() {
        writeln!(output, "declare i32 @{}(float)", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define i32 @probe_{}(float %value) {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i32 @{}(float %value)",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(width) = record.scalar_width() {
        writeln!(output, "declare i{width} @{}()", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define i{width} @probe_{}() {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i{width} @{}()",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret i{width} %result\n}}\n").unwrap();
    } else {
        unreachable!("generated intrinsic has no probe renderer: {}", record.id);
    }
    output
}
