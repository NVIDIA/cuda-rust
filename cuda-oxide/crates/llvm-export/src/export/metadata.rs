/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM annotations and version metadata.

use std::collections::HashSet;
use std::fmt::Write;

use super::state::{KernelBlockGeometry, ModuleExportState};

pub(super) fn needs_nvvm_annotations(
    state: &ModuleExportState,
    emit_all_annotations: bool,
) -> bool {
    let has_required_annotations = !state.cluster_kernels.is_empty()
        || !state.launch_bounds_kernels.is_empty()
        || !state.function_abi_alignments.is_empty()
        || !state.grid_constant_kernels.is_empty();
    has_required_annotations || (emit_all_annotations && !state.all_kernels.is_empty())
}

/// Emit `!nvvm.annotations` metadata nodes for kernels.
///
/// Metadata IDs come from `ModuleExportState` because `!nvvm.annotations`,
/// `!nvvmir.version`, and future debug-info nodes all share LLVM's flat
/// module metadata namespace.
pub(super) fn emit_nvvm_annotations(
    output: &mut String,
    state: &mut ModuleExportState,
    emit_all_annotations: bool,
) -> Result<(), String> {
    let mut metadata_refs = Vec::new();

    // Cluster annotations already carry `!"kernel"`. Launch-bounds annotations
    // do not, so bounded kernels still need the basic annotation when the
    // backend requests annotations for every kernel.
    let cluster_kernel_names: HashSet<String> = state
        .cluster_kernels
        .iter()
        .map(|k| k.name.clone())
        .collect();

    // Emit basic annotations for kernels whose other metadata does not already
    // identify them as kernels.
    if emit_all_annotations {
        let basic_kernels: Vec<String> = state
            .all_kernels
            .iter()
            .filter(|kernel| !cluster_kernel_names.contains(&kernel.name))
            .map(|kernel| kernel.name.clone())
            .collect();

        for kernel_name in basic_kernels {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &kernel_name)?;
            writeln!(output, ", !\"kernel\", i32 1}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }
    }

    // Emit non-natural ABI alignments for direct aggregate arguments and
    // returns. NVVM encodes position in the high 16 bits (0 = return,
    // arguments start at 1) and byte alignment in the low 16 bits.
    let function_abi_alignments: Vec<_> = state
        .function_abi_alignments
        .iter()
        .map(|entry| (entry.name.clone(), entry.position, entry.alignment))
        .collect();

    for (name, position, alignment) in function_abi_alignments {
        let encoded = (u32::from(position) << 16) | u32::from(alignment);
        let md_id = state.alloc_metadata_id();
        write!(output, "!{md_id} = !{{").unwrap();
        emit_function_reference(output, state, &name)?;
        writeln!(output, ", !\"align\", i32 {encoded}}}").unwrap();
        metadata_refs.push(format!("!{}", md_id));
    }

    // NVVM represents grid-constant positions indirectly: the annotation's
    // value is a metadata node containing the 1-based parameter indices.
    let grid_constant_kernels: Vec<_> = state
        .grid_constant_kernels
        .iter()
        .map(|kernel| (kernel.name.clone(), kernel.positions.clone()))
        .collect();

    for (name, positions) in grid_constant_kernels {
        let positions_id = state.alloc_metadata_id();
        let positions = positions
            .iter()
            .map(|position| format!("i32 {position}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "!{positions_id} = !{{{positions}}}").unwrap();

        let annotation_id = state.alloc_metadata_id();
        write!(output, "!{annotation_id} = !{{").unwrap();
        emit_function_reference(output, state, &name)?;
        writeln!(output, ", !\"grid_constant\", !{positions_id}}}").unwrap();
        metadata_refs.push(format!("!{annotation_id}"));
    }

    // Emit cluster config annotations
    let cluster_kernels: Vec<_> = state
        .cluster_kernels
        .iter()
        .map(|cfg| (cfg.name.clone(), cfg.dim_x, cfg.dim_y, cfg.dim_z))
        .collect();

    for (name, dim_x, dim_y, dim_z) in cluster_kernels {
        let md_id = state.alloc_metadata_id();
        write!(output, "!{md_id} = !{{").unwrap();
        emit_function_reference(output, state, &name)?;
        writeln!(
            output,
            ", !\"kernel\", i32 1, !\"cluster_dim_x\", i32 {dim_x}, !\"cluster_dim_y\", i32 {dim_y}, !\"cluster_dim_z\", i32 {dim_z}}}"
        )
        .unwrap();
        metadata_refs.push(format!("!{}", md_id));
    }

    // Emit launch bounds annotations
    let launch_bounds_kernels: Vec<_> = state
        .launch_bounds_kernels
        .iter()
        .map(|bounds| (bounds.name.clone(), bounds.geometry, bounds.min_blocks))
        .collect();

    for (name, geometry, min_blocks) in launch_bounds_kernels {
        let annotations = match geometry {
            KernelBlockGeometry::ExactBlock(x, y, z) => {
                [("reqntidx", x), ("reqntidy", y), ("reqntidz", z)]
            }
            KernelBlockGeometry::MaxThreads(max_threads) => {
                [("maxntidx", max_threads), ("maxntidy", 1), ("maxntidz", 1)]
            }
        };
        for (key, value) in annotations {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &name)?;
            writeln!(output, ", !\"{key}\", i32 {value}}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }

        if let Some(min_blocks) = min_blocks {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &name)?;
            writeln!(output, ", !\"minctasm\", i32 {min_blocks}}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }
    }

    // Emit named metadata referencing all annotation nodes
    if !metadata_refs.is_empty() {
        writeln!(
            output,
            "!nvvm.annotations = !{{{}}}",
            metadata_refs.join(", ")
        )
        .unwrap();
    }
    Ok(())
}

fn emit_function_reference(
    output: &mut String,
    state: &ModuleExportState<'_>,
    name: &str,
) -> Result<(), String> {
    if state.legacy_typed_pointers() {
        state.export_named_function_pointer_type(name, output)?;
    } else {
        write!(output, "ptr").unwrap();
    }
    write!(output, " @{name}").unwrap();
    Ok(())
}

pub(super) fn emit_nvvmir_version(
    output: &mut String,
    state: &mut ModuleExportState,
    version: [i32; 4],
) {
    let md_id = state.alloc_metadata_id();
    writeln!(output, "!nvvmir.version = !{{!{}}}", md_id).unwrap();
    writeln!(
        output,
        "!{} = !{{i32 {}, i32 {}, i32 {}, i32 {}}}",
        md_id, version[0], version[1], version[2], version[3]
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::config::DebugKind;
    use crate::export::state::{
        FunctionAbiAlignment, KernelBlockGeometry, KernelClusterConfig, KernelGridConstants,
        KernelInfo, KernelLaunchBounds, ModuleExportState,
    };
    use pliron::context::Context;

    fn test_state<'a>(ctx: &'a Context) -> ModuleExportState<'a> {
        ModuleExportState::new(
            ctx,
            false,
            DebugKind::Off,
            None,
            super::super::config::FunctionLocalStaticPlacement::CompileUnitGlobals,
        )
    }

    #[test]
    fn allocator_returns_contiguous_module_metadata_ids() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);

        assert_eq!(state.alloc_metadata_id(), 0);
        assert_eq!(state.alloc_metadata_id(), 1);
        assert_eq!(state.alloc_metadata_id(), 2);
        assert_eq!(state.next_metadata_id(), 3);
    }

    #[test]
    fn nvvm_metadata_uses_one_shared_allocator() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "plain".into(),
        });
        state.all_kernels.push(KernelInfo {
            name: "clustered".into(),
        });
        state.all_kernels.push(KernelInfo {
            name: "bounded".into(),
        });
        state.cluster_kernels.push(KernelClusterConfig {
            name: "clustered".into(),
            dim_x: 2,
            dim_y: 3,
            dim_z: 4,
        });
        state.launch_bounds_kernels.push(KernelLaunchBounds {
            name: "bounded".into(),
            geometry: KernelBlockGeometry::MaxThreads(256),
            min_blocks: Some(2),
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, true).unwrap();
        emit_nvvmir_version(&mut output, &mut state, [2, 0, 3, 2]);

        assert_eq!(
            output,
            concat!(
                "!0 = !{ptr @plain, !\"kernel\", i32 1}\n",
                "!1 = !{ptr @bounded, !\"kernel\", i32 1}\n",
                "!2 = !{ptr @clustered, !\"kernel\", i32 1, !\"cluster_dim_x\", i32 2, !\"cluster_dim_y\", i32 3, !\"cluster_dim_z\", i32 4}\n",
                "!3 = !{ptr @bounded, !\"maxntidx\", i32 256}\n",
                "!4 = !{ptr @bounded, !\"maxntidy\", i32 1}\n",
                "!5 = !{ptr @bounded, !\"maxntidz\", i32 1}\n",
                "!6 = !{ptr @bounded, !\"minctasm\", i32 2}\n",
                "!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6}\n",
                "!nvvmir.version = !{!7}\n",
                "!7 = !{i32 2, i32 0, i32 3, i32 2}\n",
            )
        );
        assert_eq!(state.next_metadata_id(), 8);
    }

    #[test]
    fn non_natural_function_abi_alignment_emits_nvvm_align_property() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.function_abi_alignments.push(FunctionAbiAlignment {
            name: "packed_param".into(),
            position: 1,
            alignment: 2,
        });
        state.function_abi_alignments.push(FunctionAbiAlignment {
            name: "packed_return".into(),
            position: 0,
            alignment: 2,
        });

        assert!(needs_nvvm_annotations(&state, false));

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, false).unwrap();

        assert_eq!(
            output,
            concat!(
                "!0 = !{ptr @packed_param, !\"align\", i32 65538}\n",
                "!1 = !{ptr @packed_return, !\"align\", i32 2}\n",
                "!nvvm.annotations = !{!0, !1}\n",
            )
        );
        assert_eq!(state.next_metadata_id(), 2);
    }

    #[test]
    fn grid_constant_emits_parameter_list_and_annotation() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.grid_constant_kernels.push(KernelGridConstants {
            name: "copy_maps".into(),
            positions: vec![1, 3],
        });

        assert!(needs_nvvm_annotations(&state, false));

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, false).unwrap();

        assert_eq!(
            output,
            concat!(
                "!0 = !{i32 1, i32 3}\n",
                "!1 = !{ptr @copy_maps, !\"grid_constant\", !0}\n",
                "!nvvm.annotations = !{!1}\n",
            )
        );
        assert_eq!(state.next_metadata_id(), 2);
    }

    #[test]
    fn cluster_and_launch_bounds_emit_one_kernel_identity() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "clustered_bounded".into(),
        });
        state.cluster_kernels.push(KernelClusterConfig {
            name: "clustered_bounded".into(),
            dim_x: 2,
            dim_y: 1,
            dim_z: 1,
        });
        state.launch_bounds_kernels.push(KernelLaunchBounds {
            name: "clustered_bounded".into(),
            geometry: KernelBlockGeometry::MaxThreads(128),
            min_blocks: None,
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, true).unwrap();

        assert_eq!(output.matches("!\"kernel\", i32 1").count(), 1);
        assert!(output.contains(
            "!0 = !{ptr @clustered_bounded, !\"kernel\", i32 1, !\"cluster_dim_x\", i32 2, !\"cluster_dim_y\", i32 1, !\"cluster_dim_z\", i32 1}"
        ));
        assert!(output.contains("!1 = !{ptr @clustered_bounded, !\"maxntidx\", i32 128}"));
        assert!(output.contains("!2 = !{ptr @clustered_bounded, !\"maxntidy\", i32 1}"));
        assert!(output.contains("!3 = !{ptr @clustered_bounded, !\"maxntidz\", i32 1}"));
        assert_eq!(state.next_metadata_id(), 4);
    }

    #[test]
    fn exact_block_emits_reqntid_and_suppresses_maxntid() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "exact".into(),
        });
        // The kernel carries both `#[launch_bounds(256, 2)]` and an exact
        // `#[launch_contract(block = ...)]`, which is the shape of the
        // `cuda_module_contract` example. ptxas rejects an entry declaring both
        // `.maxntid` and `.reqntid`, so only `.reqntid` may survive. The
        // occupancy hint is independent and stays.
        state.launch_bounds_kernels.push(KernelLaunchBounds {
            name: "exact".into(),
            geometry: KernelBlockGeometry::ExactBlock(256, 1, 1),
            min_blocks: Some(2),
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, true).unwrap();

        assert!(output.contains("!1 = !{ptr @exact, !\"reqntidx\", i32 256}"));
        assert!(output.contains("!2 = !{ptr @exact, !\"reqntidy\", i32 1}"));
        assert!(output.contains("!3 = !{ptr @exact, !\"reqntidz\", i32 1}"));
        assert!(output.contains("!4 = !{ptr @exact, !\"minctasm\", i32 2}"));
        assert!(!output.contains("maxntid"));
        assert_eq!(state.next_metadata_id(), 5);
    }

    #[test]
    fn exact_block_without_launch_bounds_still_emits_reqntid() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "contract_only".into(),
        });
        state.launch_bounds_kernels.push(KernelLaunchBounds {
            name: "contract_only".into(),
            geometry: KernelBlockGeometry::ExactBlock(16, 16, 1),
            min_blocks: None,
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, true).unwrap();

        assert!(output.contains("!1 = !{ptr @contract_only, !\"reqntidx\", i32 16}"));
        assert!(output.contains("!2 = !{ptr @contract_only, !\"reqntidy\", i32 16}"));
        assert!(output.contains("!3 = !{ptr @contract_only, !\"reqntidz\", i32 1}"));
        assert!(!output.contains("maxntid"));
        assert_eq!(state.next_metadata_id(), 4);
    }

    #[test]
    fn basic_kernel_annotations_are_skipped_when_backend_does_not_need_them() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "plain".into(),
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, false).unwrap();

        assert!(output.is_empty());
        assert_eq!(state.next_metadata_id(), 0);
    }
}
