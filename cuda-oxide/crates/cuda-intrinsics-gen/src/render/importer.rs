/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogFile, CatalogIntrinsic, ClcOperation, ClusterBarrierMode, CpAsyncControlOperation,
    CpAsyncMbarrierAdapter, CpAsyncSourceSize, DebugControlOperation, ExecutionControlOperation,
    MbarrierBasicAdapter, MbarrierBasicOperation, MbarrierExtendedAdapter, PackedAtomicFormat,
    PrmtMode, ReduxAdapter, RegisterMmaAdapter, Tcgen05LdShape, Tcgen05MmaForm,
    Tcgen05MmaSelectorLayout, Tcgen05Operation, TmaAdapter, TmaOperation, VoteAdapter,
    WarpBarrierAdapter, WarpShuffleAdapter, WgmmaControlMode,
};
use crate::render::common::{intrinsic_marker, rust_header, uses_identifier};
use crate::render::families::dialect_nvvm_ops_import_candidates;
use crate::render::families::{
    active_masks, clc_intrinsics, cluster_barrier_attr, cluster_barriers, cluster_memory,
    cp_async_controls, cp_async_copies, cp_async_mbarriers, debug_controls, dot_products,
    elect_intrinsics, execution_controls, extended_minmax, extended_minmax_format_attr,
    extended_minmax_nan_attr, extended_minmax_operation_attr, extended_minmax_subnormal_attr,
    extended_minmax_xorsign_abs_attr, integer_minmaxes, ldmatrix, ldmatrix_attr_variants,
    mbarrier_basics, mbarrier_extended, movmatrix, packed_alus, packed_atomics, packed_conversions,
    prmts, redux, register_mma_attr_variants, register_mmas, scalar_arithmetic_arity,
    scalar_arithmetic_format_attr, scalar_arithmetic_operation_attr,
    scalar_arithmetic_rounding_attr, scalar_arithmetic_saturation_attr,
    scalar_arithmetic_subnormal_attr, scalar_arithmetics, scalar_conversion_rounding_attr,
    scalar_conversion_saturation_attr, scalar_conversions, scalar_math_format_attr,
    scalar_math_operation_attr, scalar_math_precision_attr, scalar_math_subnormal_attr,
    scalar_maths, sparse_mma_attr_variants, sparse_mma_import_adapter, sparse_mma_selector_error,
    sparse_mmas, sregs, stmatrices, stmatrix_compatibility_name, stmatrix_variant, sync_intrinsics,
    tcgen05_intrinsics, tcgen05_ld_register_count, tcgen05_mma_b_usage_attr, tcgen05_mma_form_attr,
    tcgen05_mma_intrinsics, tcgen05_mma_kind_attr, tcgen05_st_register_count, tma_intrinsics,
    vote_intrinsics, warp_barriers, warp_matches, warp_shuffles, wgmma_controls,
};
use crate::render::reference::{
    render_compiler_path_patterns, render_inline_patterns, render_string_patterns,
};
use std::fmt::Write as _;
use std::path::PathBuf;

fn render_prepare_destination_write(output: &mut String, prev_op: &str) {
    writeln!(
        output,
        "            let (prepared_destination, prepared_last_op) = helpers::prepare_destination_write(\n                ctx, body, destination, value_map, block_ptr, {prev_op}, loc.clone(),\n            )?;"
    )
    .unwrap();
}

fn render_importer_pure_value_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    writeln!(
        output,
        "            require_arity(name, args.len(), {}, &loc)?;",
        record.rust.arguments.len()
    )
    .unwrap();
    for index in 0..record.rust.arguments.len() {
        let previous = if index == 0 { "prev_op" } else { "last_op" };
        writeln!(
            output,
            "            let (arg{index}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
        )
        .unwrap();
    }
    let arguments = (0..record.rust.arguments.len())
        .map(|index| format!("arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "            let intrinsic = {}::build(ctx, {arguments});",
        record.dialect.op_type
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n            let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_scalar_conversion_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    output.push_str(
        "            require_arity(name, args.len(), 1, &loc)?;\n\
                     let (value, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
                     )?;\n",
    );
    writeln!(
        output,
        "            let intrinsic = ScalarConversionOp::build(ctx, value, {}, {});",
        scalar_conversion_rounding_attr(record),
        scalar_conversion_saturation_attr(record),
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n\
                     let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_scalar_arithmetic_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    let arity = scalar_arithmetic_arity(record);
    writeln!(
        output,
        "            require_arity(name, args.len(), {arity}, &loc)?;"
    )
    .unwrap();
    for index in 0..arity {
        let previous = if index == 0 { "prev_op" } else { "last_op" };
        writeln!(
            output,
            "            let (arg{index}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
        )
        .unwrap();
    }
    let arguments = (0..arity)
        .map(|index| format!("arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "            let intrinsic = ScalarArithmeticOp::build(ctx, vec![{arguments}], {}, {}, {}, {}, {});",
        scalar_arithmetic_format_attr(record),
        scalar_arithmetic_operation_attr(record),
        scalar_arithmetic_rounding_attr(record),
        scalar_arithmetic_subnormal_attr(record),
        scalar_arithmetic_saturation_attr(record),
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n\
                     let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_scalar_math_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    writeln!(
        output,
        "            require_arity(name, args.len(), 1, &loc)?;"
    )
    .unwrap();
    writeln!(
        output,
        "            let (arg0, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;"
    )
    .unwrap();
    writeln!(
        output,
        "            let intrinsic = ScalarMathOp::build(ctx, arg0, {}, {}, {}, {});",
        scalar_math_format_attr(record),
        scalar_math_operation_attr(record),
        scalar_math_precision_attr(record),
        scalar_math_subnormal_attr(record),
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n\
                     let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_extended_minmax_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    output.push_str(
        "            require_arity(name, args.len(), 2, &loc)?;\n\
                     let (a, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
                     )?;\n\
                     let (b, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n\
                     )?;\n",
    );
    writeln!(
        output,
        "            let intrinsic = ExtendedMinMaxOp::build(ctx, a, b, {}, {}, {}, {}, {});",
        extended_minmax_format_attr(record),
        extended_minmax_operation_attr(record),
        extended_minmax_subnormal_attr(record),
        extended_minmax_nan_attr(record),
        extended_minmax_xorsign_abs_attr(record),
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n\
                     let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_elect_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    output.push_str(
        "            require_arity(name, args.len(), 1, &loc)?;\n\
         \n\
                     let tuple_ty = crate::translator::types::translate_destination_type(\n\
                         ctx, body, destination, &loc,\n\
                     )?;\n\
                     let (leader_ty, elected_ty) = {\n\
                         let ty = tuple_ty.deref(ctx);\n\
                         match ty.downcast_ref::<MirTupleType>() {\n\
                             Some(tuple) if tuple.get_types().len() == 2 => {\n\
                                 (tuple.get_types()[0], tuple.get_types()[1])\n\
                             }\n\
                             _ => {\n\
                                 return input_err!(\n\
                                     loc.clone(),\n\
                                     TranslationErr::unsupported(\n\
                                         \"warp::elect_sync destination must be a (u32, bool) tuple\"\n\
                                             .to_owned()\n\
                                     )\n\
                                 );\n\
                             }\n\
                         }\n\
                     };\n\
                     let (mask, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
                     )?;\n",
    );
    writeln!(
        output,
        "            let elect = Operation::new(\n                ctx, {}::get_concrete_op_info(), vec![leader_ty, elected_ty],\n                vec![mask], vec![], 0,\n            );",
        record.dialect.op_type
    )
    .unwrap();
    output.push_str("            elect.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, elect, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    render_prepare_destination_write(output, "last_op");
    output.push_str(
        "            helpers::insert_op(ctx, elect, block_ptr, prepared_last_op);\n\
                     let leader = elect.deref(ctx).get_result(0);\n\
                     let elected = elect.deref(ctx).get_result(1);\n\
                     let tuple = Operation::new(\n\
                         ctx, MirConstructTupleOp::get_concrete_op_info(), vec![tuple_ty],\n\
                         vec![leader, elected],\n\
                         vec![], 0,\n\
                     );\n\
                     tuple.deref_mut(ctx).set_loc(loc.clone());\n\
                     tuple.insert_after(ctx, elect);\n\
                     let result = tuple.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, tuple, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer_tcgen05_mma_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mma = record
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref())
        .expect("tcgen05 MMA record");
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    writeln!(
        output,
        "            require_arity(name, args.len(), {}, &loc)?;",
        record.rust.arguments.len()
    )
    .unwrap();

    let runtime_indices = if mma.alias.is_some() && mma.form == Tcgen05MmaForm::WsTensor {
        vec![0, 1, 3, 4, 5]
    } else if mma.alias.is_some() {
        (0..record.rust.arguments.len()).collect()
    } else {
        let selector_indices = match mma.selector_layout {
            Tcgen05MmaSelectorLayout::Base {
                kind_argument,
                cta_group_argument,
                collector_a_argument,
                ..
            } => [
                kind_argument as usize,
                cta_group_argument as usize,
                collector_a_argument as usize,
            ],
            Tcgen05MmaSelectorLayout::WarpSpecialized {
                kind_argument,
                b_buffer_argument,
                b_usage_argument,
            } => [
                kind_argument as usize,
                b_buffer_argument as usize,
                b_usage_argument as usize,
            ],
        };
        (0..record.rust.arguments.len())
            .filter(|index| !selector_indices.contains(index))
            .collect()
    };
    writeln!(
        output,
        "            let runtime_indices: &[usize] = &{runtime_indices:?};"
    )
    .unwrap();
    output.push_str(
        "            let mut last_op = prev_op;\n            let mut operands = Vec::with_capacity(runtime_indices.len());\n            for &index in runtime_indices {\n                let (value, translated) = rvalue::translate_operand(\n                    ctx, body, &args[index], value_map, block_ptr, last_op, loc.clone(),\n                )?;\n                last_op = translated;\n                operands.push(value);\n            }\n",
    );

    if let Some(fixed) = mma.fixed_selectors {
        writeln!(
            output,
            "            let kind = {};",
            tcgen05_mma_kind_attr(fixed.kind)
        )
        .unwrap();
        match mma.selector_layout {
            Tcgen05MmaSelectorLayout::Base { .. } => {
                output.push_str(
                    "            let cta_group = Tcgen05MmaCtaGroupAttr::Cg1;\n            let collector_a = Tcgen05MmaCollectorAAttr::Discard;\n",
                );
            }
            Tcgen05MmaSelectorLayout::WarpSpecialized { .. } => {
                let b_buffer = match fixed.b_buffer {
                    0 => "Tcgen05MmaBBufferAttr::B0",
                    1 => "Tcgen05MmaBBufferAttr::B1",
                    2 => "Tcgen05MmaBBufferAttr::B2",
                    3 => "Tcgen05MmaBBufferAttr::B3",
                    _ => panic!("admitted tcgen05 MMA B buffer"),
                };
                writeln!(output, "            let b_buffer = {b_buffer};").unwrap();
                writeln!(
                    output,
                    "            let b_usage = {};",
                    tcgen05_mma_b_usage_attr(fixed.b_usage)
                )
                .unwrap();
            }
        }
    } else {
        let (kind_argument, second_argument, third_argument) = match mma.selector_layout {
            Tcgen05MmaSelectorLayout::Base {
                kind_argument,
                cta_group_argument,
                collector_a_argument,
                ..
            } => (kind_argument, cta_group_argument, collector_a_argument),
            Tcgen05MmaSelectorLayout::WarpSpecialized {
                kind_argument,
                b_buffer_argument,
                b_usage_argument,
            } => (kind_argument, b_buffer_argument, b_usage_argument),
        };
        writeln!(
            output,
            "            let kind = match generated_tcgen05_mma_selector(args.get({kind_argument}usize)) {{\n                Some(0) => Tcgen05MmaKindAttr::F16,\n                Some(1) => Tcgen05MmaKindAttr::Tf32,\n                Some(2) => Tcgen05MmaKindAttr::F8f6f4,\n                Some(3) => Tcgen05MmaKindAttr::I8,\n                _ => return input_err!(loc.clone(), TranslationErr::unsupported(\"tcgen05 MMA kind must be a compile-time u32 constant in 0..=3\".to_owned())),\n            }};"
        )
        .unwrap();
        match mma.selector_layout {
            Tcgen05MmaSelectorLayout::Base {
                collector_a_upper_exclusive,
                ..
            } => {
                writeln!(
                    output,
                    "            let cta_group = match generated_tcgen05_mma_selector(args.get({second_argument}usize)) {{\n                Some(1) => Tcgen05MmaCtaGroupAttr::Cg1,\n                Some(2) => Tcgen05MmaCtaGroupAttr::Cg2,\n                _ => {{\n                    return input_err!(\n                        loc.clone(),\n                        TranslationErr::unsupported(\n                            \"tcgen05 MMA CTA group must be the compile-time u32 constant 1 or 2\"\n                                .to_owned()\n                        )\n                    );\n                }}\n            }};"
                )
                .unwrap();
                let collector_arms = if collector_a_upper_exclusive == 2 {
                    "                Some(0) => Tcgen05MmaCollectorAAttr::Discard,\n                Some(1) => Tcgen05MmaCollectorAAttr::LastUse,\n"
                } else {
                    "                Some(0) => Tcgen05MmaCollectorAAttr::Discard,\n                Some(1) => Tcgen05MmaCollectorAAttr::LastUse,\n                Some(2) => Tcgen05MmaCollectorAAttr::Fill,\n                Some(3) => Tcgen05MmaCollectorAAttr::Use,\n"
                };
                writeln!(
                    output,
                    "            let collector_a = match generated_tcgen05_mma_selector(args.get({third_argument}usize)) {{\n{collector_arms}                _ => return input_err!(loc.clone(), TranslationErr::unsupported(\"tcgen05 MMA collector-A selector is outside its closed range\".to_owned())),\n            }};"
                )
                .unwrap();
            }
            Tcgen05MmaSelectorLayout::WarpSpecialized { .. } => {
                writeln!(
                    output,
                    "            let b_buffer = match generated_tcgen05_mma_selector(args.get({second_argument}usize)) {{\n                Some(0) => Tcgen05MmaBBufferAttr::B0,\n                Some(1) => Tcgen05MmaBBufferAttr::B1,\n                Some(2) => Tcgen05MmaBBufferAttr::B2,\n                Some(3) => Tcgen05MmaBBufferAttr::B3,\n                _ => return input_err!(loc.clone(), TranslationErr::unsupported(\"tcgen05 MMA B buffer must be a compile-time u32 constant in 0..=3\".to_owned())),\n            }};"
                )
                .unwrap();
                writeln!(
                    output,
                    "            let b_usage = match generated_tcgen05_mma_selector(args.get({third_argument}usize)) {{\n                Some(0) => Tcgen05MmaBUsageAttr::Discard,\n                Some(1) => Tcgen05MmaBUsageAttr::LastUse,\n                Some(2) => Tcgen05MmaBUsageAttr::Fill,\n                Some(3) => Tcgen05MmaBUsageAttr::Use,\n                _ => return input_err!(loc.clone(), TranslationErr::unsupported(\"tcgen05 MMA B usage must be a compile-time u32 constant in 0..=3\".to_owned())),\n            }};"
                )
                .unwrap();
            }
        }
    }

    output.push_str(
        "            let intrinsic = Operation::new(ctx, Tcgen05MmaOp::get_concrete_op_info(), vec![], operands, vec![], 0);\n            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n            let mma_op = Tcgen05MmaOp::new(intrinsic);\n",
    );
    writeln!(
        output,
        "            mma_op.set_attr_nvvm_tcgen05_mma_form(ctx, {});",
        tcgen05_mma_form_attr(mma.form)
    )
    .unwrap();
    output.push_str("            mma_op.set_attr_nvvm_tcgen05_mma_kind(ctx, kind);\n");
    match mma.selector_layout {
        Tcgen05MmaSelectorLayout::Base { .. } => output.push_str(
            "            mma_op.set_attr_nvvm_tcgen05_mma_cta_group(ctx, cta_group);\n            mma_op.set_attr_nvvm_tcgen05_mma_collector_a(ctx, collector_a);\n",
        ),
        Tcgen05MmaSelectorLayout::WarpSpecialized { .. } => output.push_str(
            "            mma_op.set_attr_nvvm_tcgen05_mma_b_buffer(ctx, b_buffer);\n            mma_op.set_attr_nvvm_tcgen05_mma_b_usage(ctx, b_usage);\n",
        ),
    }
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, intrinsic, block_map, loc)))\n            } else {\n",
    );
    writeln!(
        output,
        "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("            }\n        }\n");
}

fn render_importer_tcgen05_non_mma_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let tcgen05 = record.tcgen05.as_ref().unwrap();
    let operation = tcgen05.operation;
    let has_half_split_offset = tcgen05
        .ld
        .is_some_and(|ld| ld.shape == Tcgen05LdShape::M16x32bx2)
        || tcgen05
            .st
            .is_some_and(|st| st.shape == Tcgen05LdShape::M16x32bx2);
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    if operation == Tcgen05Operation::St {
        let count = tcgen05_st_register_count(record);
        writeln!(
            output,
            "            require_arity(name, args.len(), {}, &loc)?;",
            if has_half_split_offset { 3 } else { 2 }
        )
        .unwrap();
        if has_half_split_offset {
            output.push_str(
                    "            if !matches!(args.get(1), Some(mir::Operand::Constant(_))) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"tcgen05 16x32bx2 half-split offset must be a compile-time constant\".to_owned()\n                    )\n                );\n            }\n",
                );
        }
        writeln!(
                output,
                "            let (operands, last_op) = import_generated_tcgen05_store_operands(\n                ctx, body, args, {count}, {has_half_split_offset}, block_ptr, prev_op, value_map, loc.clone(),\n            )?;"
            )
            .unwrap();
        writeln!(
                output,
                "            let intrinsic = Operation::new(ctx, {}::get_concrete_op_info(), vec![], operands, vec![], 0);",
                record.dialect.op_type
            )
            .unwrap();
        output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
                "            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, intrinsic, block_map, loc)))\n            } else {\n",
            );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
        return;
    }
    let arity = record.dialect.operands.len();
    writeln!(
        output,
        "            require_arity(name, args.len(), {arity}, &loc)?;"
    )
    .unwrap();
    if has_half_split_offset {
        output.push_str(
                "            if !matches!(args.get(1), Some(mir::Operand::Constant(_))) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"tcgen05 16x32bx2 half-split offset must be a compile-time constant\".to_owned()\n                    )\n                );\n            }\n",
            );
    }
    output.push_str(
            "            let mut last_op = prev_op;\n            let mut operands = Vec::with_capacity(args.len());\n            for arg in args {\n                let (value, translated) = rvalue::translate_operand(\n                    ctx, body, arg, value_map, block_ptr, last_op, loc.clone(),\n                )?;\n                last_op = translated;\n                operands.push(value);\n            }\n",
        );
    if has_half_split_offset {
        output.push_str(
                "            if operands.get(1).and_then(|value| value.defining_op()).and_then(|op| Operation::get_op::<MirConstantOp>(op, ctx)).is_none() {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"tcgen05 16x32bx2 half-split offset must lower to a constant\".to_owned()\n                    )\n                );\n            }\n",
            );
    }
    let load = match operation {
        Tcgen05Operation::Ld16x256bX8Pure => Some((32, "FP32Type::get(ctx).into()")),
        Tcgen05Operation::Ld16x256bPure => Some((4, "FP32Type::get(ctx).into()")),
        Tcgen05Operation::Ld => Some((
            tcgen05_ld_register_count(record),
            "IntegerType::get(ctx, 32, Signedness::Unsigned).into()",
        )),
        _ => None,
    };
    if let Some((count, result_ty)) = load {
        writeln!(
            output,
            "            let result_ty: TypeHandle = {result_ty};"
        )
        .unwrap();
        writeln!(
            output,
            "            let result_types = (0..{count}).map(|_| result_ty).collect();"
        )
        .unwrap();
        writeln!(
                    output,
                    "            let intrinsic = Operation::new(ctx, {}::get_concrete_op_info(), result_types, operands, vec![], 0);",
                    record.dialect.op_type
                )
                .unwrap();
        output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n",
        );
        if count == 1 {
            output.push_str(
                        "            let result = intrinsic.deref(ctx).get_result(0);\n            let value = intrinsic;\n",
                    );
        } else {
            writeln!(
                        output,
                        "            let results: Vec<Value> = (0..{count}).map(|index| intrinsic.deref(ctx).get_result(index)).collect();"
                    )
                    .unwrap();
            writeln!(
                output,
                "            let array_ty = MirArrayType::get(ctx, result_ty, {count});"
            )
            .unwrap();
            output.push_str(
                        "            let array = Operation::new(ctx, MirConstructArrayOp::get_concrete_op_info(), vec![array_ty.into()], results, vec![], 0);\n            array.deref_mut(ctx).set_loc(loc.clone());\n            array.insert_after(ctx, intrinsic);\n            let destination_rust_ty = match destination.ty(body.locals()) {\n                Ok(ty) => ty,\n                Err(error) => {\n                    return input_err!(\n                        loc,\n                        TranslationErr::unsupported(format!(\n                            \"failed to resolve destination type for intrinsic result: {error:?}\"\n                        ))\n                    );\n                }\n            };\n            let destination_ty = types::translate_type(ctx, &destination_rust_ty)?;\n            let array_result = array.deref(ctx).get_result(0);\n            let (result, value) = if destination_ty == array_result.get_type(ctx) {\n                (array_result, array)\n            } else {\n                let value = Operation::new(ctx, MirConstructStructOp::get_concrete_op_info(), vec![destination_ty], vec![array_result], vec![], 0);\n                value.deref_mut(ctx).set_loc(loc.clone());\n                value.insert_after(ctx, array);\n                (value.deref(ctx).get_result(0), value)\n            };\n            helpers::set_compiler_result_bundle_marker(ctx, value);\n",
                    );
        }
        writeln!(
                    output,
                    "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, value, value_map, block_map, loc,\n                {:?},\n            )?))",
                    format!("{} call without target block", record.rust.name)
                )
                .unwrap();
    } else {
        writeln!(
                    output,
                    "            let intrinsic = Operation::new(ctx, {}::get_concrete_op_info(), vec![], operands, vec![], 0);",
                    record.dialect.op_type
                )
                .unwrap();
        output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
                    "            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, intrinsic, block_map, loc)))\n            } else {\n",
                );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n");
    }
    output.push_str("        }\n");
}

fn append_importer_classification(output: &mut String, catalog: &CatalogFile) {
    writeln!(
        output,
        "pub const GENERATED_INTRINSIC_ABI: u32 = {};",
        catalog.intrinsic_abi
    )
    .unwrap();
    writeln!(
        output,
        "pub const GENERATED_INTRINSIC_ABI_NAMESPACE: &str = {:?};\n",
        format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi)
    )
    .unwrap();
    output.push_str(
        "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum RawIntrinsicIdentity {\n    NotRawCrate,\n    Known(String),\n    UnsupportedAbi(String),\n    UnknownId(String),\n}\n\n/// Classify the unrewritten raw intrinsic DefPath. rustc's ordinary name\n/// printer may prefer a public re-export path, which is source API rather than ABI.\npub fn classify_raw_intrinsic(fn_def: FnDef) -> RawIntrinsicIdentity {\n    let crate_name = fn_def.krate().name.to_string();\n    if !matches!(crate_name.as_str(), \"cuda_intrinsics\" | \"cuda-intrinsics\") {\n        return RawIntrinsicIdentity::NotRawCrate;\n    }\n\n    let mut segments = Vec::new();\n    let mut current = Some(fn_def.def_id());\n    while let Some(def_id) = current {\n        let printed = def_id.name();\n        let segment = printed.as_str().rsplit(\"::\").next().unwrap_or_default();\n        if segment != crate_name {\n            segments.push(segment.to_owned());\n        }\n        current = def_id.parent();\n    }\n    segments.reverse();\n    classify_raw_intrinsic_path(&crate_name, format!(\"{crate_name}::{}\", segments.join(\"::\")))\n}\n\nfn classify_raw_intrinsic_path(crate_name: &str, path: String) -> RawIntrinsicIdentity {\n    if !matches!(crate_name, \"cuda_intrinsics\" | \"cuda-intrinsics\") {\n        return RawIntrinsicIdentity::NotRawCrate;\n    }\n    if is_raw_generated_intrinsic_path(&path) {\n        return RawIntrinsicIdentity::Known(path);\n    }\n    let namespace = path.split(\"::\").nth(1).unwrap_or_default();\n    if namespace.starts_with(\"__cuda_oxide_intrinsic_abi_v\")\n        && namespace != GENERATED_INTRINSIC_ABI_NAMESPACE\n    {\n        RawIntrinsicIdentity::UnsupportedAbi(path)\n    } else {\n        RawIntrinsicIdentity::UnknownId(path)\n    }\n}\n\npub fn require_supported_raw_intrinsic(\n    fn_def: FnDef,\n    loc: &Location,\n) -> TranslationResult<Option<String>> {\n    match classify_raw_intrinsic(fn_def) {\n        RawIntrinsicIdentity::NotRawCrate => Ok(None),\n        RawIntrinsicIdentity::Known(path) => Ok(Some(path)),\n        RawIntrinsicIdentity::UnsupportedAbi(path) => input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"cuda-intrinsics ABI mismatch: `{path}` uses an unsupported intrinsic ABI; this compiler supports ABI v{GENERATED_INTRINSIC_ABI}\"\n            ))\n        ),\n        RawIntrinsicIdentity::UnknownId(path) => input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"cuda-intrinsics ABI mismatch: `{path}` is not a known intrinsic ID in ABI v{GENERATED_INTRINSIC_ABI}\"\n            ))\n        ),\n    }\n}\n\n",
    );
    output
        .push_str("pub fn is_generated_intrinsic_path(name: &str) -> bool {\n    matches!(name,\n");
    render_compiler_path_patterns(output, catalog, "        ");
    output.push_str("    )\n}\n\npub fn is_raw_generated_intrinsic_path(name: &str) -> bool {\n    matches!(name,\n");
    let raw_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.canonical_path.as_str())
        .collect();
    render_string_patterns(output, &raw_paths, "        ");
    // Test-only. The dispatch arms rendered below carry each marker as a
    // literal, and the op-name-keyed lookup the compiler actually calls lives
    // in `cuda-oxide-codegen`; nothing outside this file's own generated tests
    // reads this path-keyed table. Gate it rather than leave a 2000-line match
    // sitting unreferenced in the compiler.
    output.push_str("    )\n}\n\n#[cfg(test)]\npub fn generated_intrinsic_marker(name: &str) -> Option<&'static str> {\n    match name {\n");
    for record in &catalog.intrinsics {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(output, &path_refs);
        writeln!(output, " => Some({:?}),", intrinsic_marker(catalog, record)).unwrap();
    }
    output.push_str("        _ => None,\n    }\n}\n\n");
}

fn sreg_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in sregs(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), 0, &loc)?;"
        )
        .unwrap();
        let helper = if record.scalar_width() == Some(64) {
            "emit_generated_nvvm_intrinsic_u64"
        } else {
            "emit_generated_nvvm_intrinsic"
        };
        writeln!(output, "            Ok(Some(helpers::{helper}(").unwrap();
        writeln!(
            output,
            "                ctx, body, {}::get_concrete_op_info(), {:?}, destination, target, block_ptr,",
            record.dialect.op_type,
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "                prev_op, value_map, block_map, loc,\n            )?))\n        }\n",
        );
    }
    output
}

fn active_mask_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in active_masks(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
        writeln!(
            output,
            "            let active_mask = {}::build(ctx);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            active_mask.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, active_mask, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "prev_op");
        output.push_str(
            "            helpers::insert_op(ctx, active_mask, block_ptr, prepared_last_op);\n            let result = active_mask.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, active_mask, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn ldmatrix_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in ldmatrix(catalog) {
        let (shape, multiplicity, layout, element, state_space) = ldmatrix_attr_variants(record);
        let register_count = record.ldmatrix.as_ref().unwrap().variant.register_count();
        output.push_str("        ");
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 1, &loc)?;\n");
        output.push_str(
            "            let (address, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let load = LdmatrixOp::build(ctx, address, {shape}, {multiplicity}, {layout}, {element}, {state_space});"
        )
        .unwrap();
        output.push_str("            load.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, load, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output
            .push_str("            helpers::insert_op(ctx, load, block_ptr, prepared_last_op);\n");
        if register_count == 1 {
            output.push_str("            let value = load.deref(ctx).get_result(0);\n");
            output.push_str("            let last_op = load;\n");
        } else {
            writeln!(
                output,
                "            let (value, last_op) = helpers::bundle_generated_u32_results_as_array(ctx, load, {register_count}, loc.clone());"
            )
            .unwrap();
        }
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, value, target, block_ptr, last_op, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn stmatrix_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in stmatrices(catalog) {
        let (multiplicity, _) = stmatrix_variant(record).expect("stmatrix variant");
        let arity = multiplicity.register_count() + 1;
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        output.push_str(
            "            let mut last_op = prev_op;\n            let mut operands = Vec::with_capacity(args.len());\n            for arg in args {\n                let (value, translated) = rvalue::translate_operand(\n                    ctx, body, arg, value_map, block_ptr, last_op, loc.clone(),\n                )?;\n                last_op = translated;\n                operands.push(value);\n            }\n",
        );
        writeln!(
            output,
            "            let store = Operation::new(ctx, {}::get_concrete_op_info(), vec![], operands, vec![], 0);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            store.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, store, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, store, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, store, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!(
                "{} call without target block",
                stmatrix_compatibility_name(record)
            )
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn register_mma_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in register_mmas(catalog) {
        let adapter = match record.register_mma.as_ref().unwrap().adapter {
            RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32 => {
                "GeneratedMmaImportAdapter::C2U32A2U32B1U32ToD2U32"
            }
            RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => {
                "GeneratedMmaImportAdapter::C2U32A4U32B2U32ToD2U32"
            }
            RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32 => {
                "GeneratedMmaImportAdapter::C4F32A2U32B1U32ToD4F32"
            }
            RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32 => {
                "GeneratedMmaImportAdapter::C4F32A4U32B2U32ToD4F32"
            }
            RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 => {
                "GeneratedMmaImportAdapter::C2F64A1F64B1F64ToD2F64"
            }
            RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32 => {
                "GeneratedMmaImportAdapter::C2I32A1U32B1U32ToD2I32"
            }
            RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32 => {
                "GeneratedMmaImportAdapter::C4I32A4U32B2U32ToD4I32"
            }
            RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => {
                "GeneratedMmaImportAdapter::C4I32A2U32B1U32ToD4I32"
            }
            RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => {
                "GeneratedMmaImportAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32"
            }
        };
        let (
            shape,
            operation,
            kind,
            accumulator,
            a_element,
            b_element,
            a_layout,
            b_layout,
            overflow,
        ) = register_mma_attr_variants(record);
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        let arity = record.rust.arguments.len();
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        writeln!(
            output,
            "            let (operands, last_op, result_ty, result_count) = import_generated_mma_operands(ctx, body, args, block_ptr, prev_op, value_map, loc.clone(), {adapter})?;"
        )
        .unwrap();
        output.push_str(
            "            let mma = Operation::new(ctx, RegisterMmaOp::get_concrete_op_info(), vec![result_ty; result_count], operands, vec![], 0);\n            mma.deref_mut(ctx).set_loc(loc.clone());\n            let mma = RegisterMmaOp::new(mma);\n",
        );
        writeln!(
            output,
            "            mma.set_attr_nvvm_register_mma_shape(ctx, {shape});\n            mma.set_attr_nvvm_register_mma_operation(ctx, {operation});\n            mma.set_attr_nvvm_register_mma_kind(ctx, {kind});\n            mma.set_attr_nvvm_register_mma_accumulator(ctx, {accumulator});\n            mma.set_attr_nvvm_register_mma_a_element(ctx, {a_element});\n            mma.set_attr_nvvm_register_mma_b_element(ctx, {b_element});\n            mma.set_attr_nvvm_register_mma_a_layout(ctx, {a_layout});\n            mma.set_attr_nvvm_register_mma_b_layout(ctx, {b_layout});\n            mma.set_attr_nvvm_register_mma_overflow(ctx, {overflow});"
        )
        .unwrap();
        output.push_str("            let mma = mma.get_operation();\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, mma, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "Some(last_op)");
        output.push_str(
            "            helpers::insert_op(ctx, mma, block_ptr, prepared_last_op);\n            let (result, last_op) = bundle_generated_mma_results(ctx, mma, result_ty, result_count, loc.clone());\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, last_op, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn sparse_mma_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in sparse_mmas(catalog) {
        let (
            shape,
            accumulator,
            a_element,
            b_element,
            a_layout,
            b_layout,
            overflow,
            metadata,
            selector,
        ) = sparse_mma_attr_variants(record);
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 5, &loc)?;\n");
        writeln!(
            output,
            "            if !matches!(&args[4], mir::Operand::Constant(_)) {{\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        {:?}.to_owned()\n                    )\n                );\n            }}",
            sparse_mma_selector_error(record)
        )
        .unwrap();
        writeln!(
            output,
            "            let (mut operands, last_op, result_ty, result_count) = import_generated_mma_operands(ctx, body, args, block_ptr, prev_op, value_map, loc.clone(), {})?;",
            sparse_mma_import_adapter(record)
        )
        .unwrap();
        output.push_str(
            "            let (metadata, last_op) = rvalue::translate_operand(\n                ctx, body, &args[3], value_map, block_ptr, Some(last_op), loc.clone(),\n            )?;\n            operands.push(metadata);\n            let (selector_value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[4], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n            operands.push(selector_value);\n",
        );
        output.push_str(
            "            let mma = Operation::new(ctx, SparseMmaOp::get_concrete_op_info(), vec![result_ty; result_count], operands, vec![], 0);\n            mma.deref_mut(ctx).set_loc(loc.clone());\n            let mma = SparseMmaOp::new(mma);\n",
        );
        writeln!(
            output,
            "            mma.set_attr_nvvm_sparse_mma_shape(ctx, {shape});\n            mma.set_attr_nvvm_sparse_mma_accumulator(ctx, {accumulator});\n            mma.set_attr_nvvm_sparse_mma_a_element(ctx, {a_element});\n            mma.set_attr_nvvm_sparse_mma_b_element(ctx, {b_element});\n            mma.set_attr_nvvm_sparse_mma_a_layout(ctx, {a_layout});\n            mma.set_attr_nvvm_sparse_mma_b_layout(ctx, {b_layout});\n            mma.set_attr_nvvm_sparse_mma_overflow(ctx, {overflow});\n            mma.set_attr_nvvm_sparse_mma_metadata(ctx, {metadata});\n            mma.set_attr_nvvm_sparse_mma_selector(ctx, {selector});"
        )
        .unwrap();
        output.push_str("            let mma = mma.get_operation();\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, mma, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, mma, block_ptr, prepared_last_op);\n            let (result, last_op) = bundle_generated_mma_results(ctx, mma, result_ty, result_count, loc.clone());\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, last_op, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn packed_atomic_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in packed_atomics(catalog) {
        let format = match record.packed_atomic.as_ref().unwrap().format {
            PackedAtomicFormat::F16x2 => "PackedAtomicFormatAttr::F16x2",
            PackedAtomicFormat::Bf16x2 => "PackedAtomicFormatAttr::Bf16x2",
        };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (address, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (addend, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let atom = PackedAtomicAddOp::build(ctx, address, addend, {format});"
        )
        .unwrap();
        output.push_str("            atom.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, atom, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output
            .push_str("            helpers::insert_op(ctx, atom, block_ptr, prepared_last_op);\n");
        output.push_str("            let value = atom.deref(ctx).get_result(0);\n");
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, value, target, block_ptr, atom, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn redux_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in redux(catalog) {
        debug_assert_eq!(
            record.redux.as_ref().unwrap().adapter,
            ReduxAdapter::MaskValueToSourceMemberMask
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let reduction = {}::build(ctx, member_mask, value);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            reduction.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, reduction, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, reduction, block_ptr, prepared_last_op);\n            let result = reduction.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, reduction, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn vote_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in vote_intrinsics(catalog) {
        debug_assert_eq!(
            record.vote.as_ref().unwrap().adapter,
            VoteAdapter::DirectMaskPredicate
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (predicate, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let vote = {}::build(ctx, member_mask, predicate);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            vote.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, vote, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, vote, block_ptr, prepared_last_op);\n            let result = vote.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, vote, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn warp_match_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in warp_matches(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let warp_match = {}::build(ctx, member_mask, value);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            warp_match.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, warp_match, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, warp_match, block_ptr, prepared_last_op);\n            let result = warp_match.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, warp_match, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn elect_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in elect_intrinsics(catalog) {
        render_importer_elect_dispatch(&mut output, catalog, record);
    }
    output
}

fn warp_barrier_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in warp_barriers(catalog) {
        debug_assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 1, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let barrier = {}::build(ctx, member_mask);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            barrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, barrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, barrier, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, barrier, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn warp_shuffle_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in warp_shuffles(catalog) {
        debug_assert!(matches!(
            record.warp_shuffle.as_ref().unwrap().adapter,
            WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
                | WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
        ));
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 3, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (lane_or_delta, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let shuffle = {}::build(ctx, member_mask, value, lane_or_delta);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            shuffle.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, shuffle, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, shuffle, block_ptr, prepared_last_op);\n            let result = shuffle.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, shuffle, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn dotprod_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in dot_products(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 3, &loc)?;\n");
        output.push_str(
            "            let (a, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (b, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (c, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let dot = {}::build(ctx, a, b, c);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            dot.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, dot, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, dot, block_ptr, prepared_last_op);\n            let result = dot.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, dot, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn packed_alu_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in packed_alus(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    output
}

fn integer_minmax_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in integer_minmaxes(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    output
}

fn packed_conversion_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in packed_conversions(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    output
}

fn scalar_conversion_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in scalar_conversions(catalog) {
        render_importer_scalar_conversion_dispatch(&mut output, catalog, record);
    }
    output
}

fn scalar_arithmetic_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in scalar_arithmetics(catalog) {
        render_importer_scalar_arithmetic_dispatch(&mut output, catalog, record);
    }
    output
}

fn scalar_math_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in scalar_maths(catalog) {
        render_importer_scalar_math_dispatch(&mut output, catalog, record);
    }
    output
}

fn extended_minmax_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in extended_minmax(catalog) {
        render_importer_extended_minmax_dispatch(&mut output, catalog, record);
    }
    output
}

fn movmatrix_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in movmatrix(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    output
}

fn prmt_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in prmts(catalog) {
        let mode = match record.prmt.as_ref().unwrap().mode {
            PrmtMode::Generic => "PrmtModeAttr::Generic",
            PrmtMode::F4e => "PrmtModeAttr::F4e",
            PrmtMode::B4e => "PrmtModeAttr::B4e",
            PrmtMode::Rc8 => "PrmtModeAttr::Rc8",
            PrmtMode::Ecl => "PrmtModeAttr::Ecl",
            PrmtMode::Ecr => "PrmtModeAttr::Ecr",
            PrmtMode::Rc16 => "PrmtModeAttr::Rc16",
        };
        let arity = record.rust.arguments.len();
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        for index in 0..arity {
            let previous = if index == 0 { "prev_op" } else { "last_op" };
            writeln!(
                output,
                "            let (arg{index}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
            )
            .unwrap();
        }
        let arguments = (0..arity)
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "            let prmt = PrmtOp::build(ctx, vec![{arguments}], {mode});"
        )
        .unwrap();
        output.push_str("            prmt.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, prmt, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, prmt, block_ptr, prepared_last_op);\n            let result = prmt.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, prmt, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn cluster_barrier_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    if cluster_barriers(catalog).next().is_some() {
        let arrive = cluster_barriers(catalog)
            .find(|record| {
                record
                    .cluster_barrier
                    .as_ref()
                    .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::ArriveAligned)
            })
            .expect("aligned cluster arrival");
        let wait = cluster_barriers(catalog)
            .find(|record| {
                record
                    .cluster_barrier
                    .as_ref()
                    .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::WaitAligned)
            })
            .expect("aligned cluster wait");
        output.push_str(
            "        \"cuda_device::cluster::cluster_sync\" => {\n            require_arity(name, args.len(), 0, &loc)?;\n            let arrive = ClusterBarrierOp::build(ctx, ClusterBarrierModeAttr::ArriveAligned);\n            arrive.deref_mut(ctx).set_loc(loc.clone());\n",
        );
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, arrive, {:?});",
            intrinsic_marker(catalog, arrive)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, arrive, block_ptr, prev_op);\n            let wait = ClusterBarrierOp::build(ctx, ClusterBarrierModeAttr::WaitAligned);\n            wait.deref_mut(ctx).set_loc(loc.clone());\n",
        );
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, wait, {:?});",
            intrinsic_marker(catalog, wait)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, wait, block_ptr, Some(arrive));\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, wait, block_map, loc)))\n            } else {\n                input_err!(loc, TranslationErr::unsupported(\"cluster_sync call without target block\".to_owned()))\n            }\n        }\n",
        );
    }
    for record in cluster_barriers(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n            require_arity(name, args.len(), 0, &loc)?;\n");
        writeln!(
            output,
            "            let barrier = ClusterBarrierOp::build(ctx, {});",
            cluster_barrier_attr(record)
        )
        .unwrap();
        output.push_str("            barrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, barrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, barrier, block_ptr, prev_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, barrier, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn cluster_memory_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in cluster_memory(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str(
            "            require_arity(name, args.len(), 2, &loc)?;\n\
             let (source, last_op) = rvalue::translate_operand(\n\
                 ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
             )?;\n\
             let (rank, last_op) = rvalue::translate_operand(\n\
                 ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n\
             )?;\n",
        );
        writeln!(
            output,
            "            let cluster = {}::build(ctx, source, rank);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            cluster.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, cluster, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        render_prepare_destination_write(&mut output, "last_op");
        output.push_str(
            "            helpers::insert_op(ctx, cluster, block_ptr, prepared_last_op);\n\
             let result = cluster.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, cluster, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    output
}

fn debug_control_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in debug_controls(catalog) {
        let operation = record.debug_control.as_ref().unwrap().operation;
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        match operation {
            DebugControlOperation::Trap => {
                output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
                output.push_str("            let debug = TrapOp::build(ctx);\n");
                output.push_str("            debug.deref_mut(ctx).set_loc(loc.clone());\n");
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, debug, {:?});",
                    intrinsic_marker(catalog, record)
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, debug, block_ptr, prev_op);\n\
                     let unreachable = Operation::new(\n\
                         ctx, MirUnreachableOp::get_concrete_op_info(), vec![], vec![], vec![], 0,\n\
                     );\n\
                     unreachable.deref_mut(ctx).set_loc(loc);\n\
                     helpers::insert_op(ctx, unreachable, block_ptr, Some(debug));\n\
                     Ok(Some(unreachable))\n",
                );
            }
            DebugControlOperation::Breakpoint => {
                output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
                output.push_str("            let debug = BreakpointOp::build(ctx);\n");
                output.push_str("            debug.deref_mut(ctx).set_loc(loc.clone());\n");
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, debug, {:?});",
                    intrinsic_marker(catalog, record)
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, debug, block_ptr, prev_op);\n\
                     if let Some(target_idx) = target {\n\
                         Ok(Some(helpers::emit_goto(ctx, *target_idx, debug, block_map, loc)))\n\
                     } else {\n",
                );
                writeln!(
                    output,
                    "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                    format!("{} call without target block", record.rust.name)
                )
                .unwrap();
                output.push_str("            }\n");
            }
            DebugControlOperation::Pmevent => {
                output.push_str(
                    "            require_arity(name, args.len(), 1, &loc)?;\n\
                     if !matches!(&args[0], mir::Operand::Constant(_)) {\n\
                         return input_err!(\n\
                             loc,\n\
                             TranslationErr::unsupported(\n\
                                 \"prof_trigger requires a compile-time constant event ID in 0..=15\".to_owned()\n\
                             )\n\
                         );\n\
                     }\n\
                     let (event_id_value, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
                     )?;\n\
                     let event_id = event_id_value\n\
                         .defining_op()\n\
                         .and_then(|defining_op| Operation::get_op::<MirConstantOp>(defining_op, ctx))\n\
                         .and_then(|constant| constant.get_attr_value(ctx))\n\
                         .map(|value| value.value().to_u64())\n\
                         .and_then(|value| u32::try_from(value).ok())\n\
                         .filter(|value| *value <= 15);\n\
                     let Some(event_id) = event_id else {\n\
                         return input_err!(\n\
                             loc,\n\
                             TranslationErr::unsupported(\n\
                                 \"prof_trigger requires a compile-time constant event ID in 0..=15\".to_owned()\n\
                             )\n\
                         );\n\
                     };\n\
                     let debug = PmEventOp::build(ctx, event_id);\n\
                     debug.deref_mut(ctx).set_loc(loc.clone());\n",
                );
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, debug, {:?});",
                    intrinsic_marker(catalog, record)
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, debug, block_ptr, last_op);\n\
                     if let Some(target_idx) = target {\n\
                         Ok(Some(helpers::emit_goto(ctx, *target_idx, debug, block_map, loc)))\n\
                     } else {\n",
                );
                writeln!(
                    output,
                    "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                    format!("{} call without target block", record.rust.name)
                )
                .unwrap();
                output.push_str("            }\n");
            }
        }
        output.push_str("        }\n");
    }
    output
}

fn clc_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in clc_intrinsics(catalog) {
        let query = !matches!(
            record.clc.as_ref().unwrap().operation,
            ClcOperation::TryCancel | ClcOperation::TryCancelMulticast
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (arg0, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n            let (arg1, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        let result_types = if query {
            "vec![pliron::builtin::types::IntegerType::get(ctx, 32, pliron::builtin::types::Signedness::Unsigned).into()]"
        } else {
            "vec![]"
        };
        writeln!(
            output,
            "            let intrinsic = Operation::new(ctx, {}::get_concrete_op_info(), {result_types}, vec![arg0, arg1], vec![], 0);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        if query {
            render_prepare_destination_write(&mut output, "last_op");
            output.push_str(
                "            helpers::insert_op(ctx, intrinsic, block_ptr, prepared_last_op);\n",
            );
            output.push_str("            let result = intrinsic.deref(ctx).get_result(0);\n");
            writeln!(
                output,
                "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
        } else {
            output
                .push_str("            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n");
            output.push_str("            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, intrinsic, block_map, loc)))\n            } else {\n");
            writeln!(
                output,
                "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
            output.push_str("            }\n");
        }
        output.push_str("        }\n");
    }
    output
}

fn wgmma_control_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in wgmma_controls(catalog) {
        let control = record.wgmma_control.as_ref().unwrap();
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        match control.mode {
            WgmmaControlMode::Fence | WgmmaControlMode::CommitGroup => {
                output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
                writeln!(
                    output,
                    "            let control = {}::build(ctx);",
                    record.dialect.op_type
                )
                .unwrap();
                output.push_str("            control.deref_mut(ctx).set_loc(loc.clone());\n");
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, control, {:?});",
                    intrinsic_marker(catalog, record)
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, control, block_ptr, prev_op);\n",
                );
            }
            WgmmaControlMode::WaitGroup => {
                output.push_str(
                    "            require_arity(name, args.len(), 1, &loc)?;\n\
                     if !matches!(&args[0], mir::Operand::Constant(_)) {\n\
                         return input_err!(\n\
                             loc,\n\
                             TranslationErr::unsupported(\n\
                                 \"wgmma_wait_group requires a compile-time constant\".to_owned()\n\
                             )\n\
                         );\n\
                     }\n\
                     let (max_pending, last_op) = rvalue::translate_operand(\n\
                         ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n\
                     )?;\n\
                     if max_pending\n\
                         .defining_op()\n\
                         .and_then(|op| Operation::get_op::<MirConstantOp>(op, ctx))\n\
                         .is_none()\n\
                     {\n\
                         return input_err!(\n\
                             loc,\n\
                             TranslationErr::unsupported(\n\
                                 \"wgmma_wait_group requires a compile-time constant\".to_owned()\n\
                             )\n\
                         );\n\
                     }\n\
                     let control = WgmmaWaitGroupSyncAlignedOp::build(ctx, max_pending);\n\
                     control.deref_mut(ctx).set_loc(loc.clone());\n",
                );
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, control, {:?});",
                    intrinsic_marker(catalog, record)
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, control, block_ptr, last_op);\n",
                );
            }
        }
        output.push_str(
            "            if let Some(target_idx) = target {\n\
                 Ok(Some(helpers::emit_goto(ctx, *target_idx, control, block_map, loc)))\n\
             } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn cp_async_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().unwrap();
        let dynamic = copy.source_size == CpAsyncSourceSize::Runtime;
        let arity = if dynamic { 3 } else { 2 };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        output.push_str(
            "            let (shared_dst, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (global_src, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        if dynamic {
            output.push_str(
                "            let (source_size, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
            );
            writeln!(
                output,
                "            let copy = {}::build(ctx, shared_dst, global_src, source_size);",
                record.dialect.op_type
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "            let copy = {}::build(ctx, shared_dst, global_src);",
                record.dialect.op_type
            )
            .unwrap();
        }
        output.push_str("            copy.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, copy, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, copy, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, copy, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    for record in cp_async_controls(catalog) {
        let control = record.cp_async_control.as_ref().unwrap();
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let arity = usize::from(has_immediate);
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        if has_immediate {
            output.push_str(
                "            if !matches!(&args[0], mir::Operand::Constant(_)) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"cp_async_wait_group requires a compile-time constant max_pending value\".to_owned()\n                    )\n                );\n            }\n",
            );
            output.push_str(
                "            let (max_pending, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
            );
            writeln!(
                output,
                "            let control = {}::build(ctx, max_pending);",
                record.dialect.op_type
            )
            .unwrap();
        } else {
            output.push_str("            let last_op = prev_op;\n");
            writeln!(
                output,
                "            let control = {}::build(ctx);",
                record.dialect.op_type
            )
            .unwrap();
        }
        output.push_str("            control.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, control, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, control, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, control, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    for record in cp_async_mbarriers(catalog) {
        debug_assert_eq!(
            record.cp_async_mbarrier.as_ref().unwrap().adapter,
            CpAsyncMbarrierAdapter::PointerToVoid
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 1, &loc)?;\n");
        output.push_str(
            "            let (barrier, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let bridge = {}::build(ctx, barrier);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            bridge.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, bridge, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, bridge, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, bridge, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn mbarrier_basic_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in mbarrier_basics(catalog) {
        let mbarrier = record.mbarrier_basic.as_ref().unwrap();
        let (argument_names, returns_value) = match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::InitPointerCountToVoid
                );
                (&["barrier", "expected_count"][..], false)
            }
            MbarrierBasicOperation::Arrive => {
                debug_assert_eq!(mbarrier.adapter, MbarrierBasicAdapter::ArrivePointerToToken);
                (&["barrier"][..], true)
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::ArriveNoCompletePointerCountToToken
                );
                (&["barrier", "count"][..], true)
            }
            MbarrierBasicOperation::TestWait => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::TestWaitPointerTokenToPredicate
                );
                (&["barrier", "token"][..], true)
            }
            MbarrierBasicOperation::Inval => {
                debug_assert_eq!(mbarrier.adapter, MbarrierBasicAdapter::InvalPointerToVoid);
                (&["barrier"][..], false)
            }
        };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {}, &loc)?;",
            argument_names.len()
        )
        .unwrap();
        for (index, argument_name) in argument_names.iter().enumerate() {
            let previous = if index == 0 { "prev_op" } else { "last_op" };
            writeln!(
                output,
                "            let ({argument_name}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
            )
            .unwrap();
        }
        writeln!(
            output,
            "            let mbarrier = {}::build(ctx, {});",
            record.dialect.op_type,
            argument_names.join(", ")
        )
        .unwrap();
        output.push_str("            mbarrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, mbarrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        if returns_value {
            render_prepare_destination_write(&mut output, "last_op");
            output.push_str(
                "            helpers::insert_op(ctx, mbarrier, block_ptr, prepared_last_op);\n",
            );
            output.push_str("            let result = mbarrier.deref(ctx).get_result(0);\n");
            writeln!(
                output,
                "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, mbarrier, value_map, block_map, loc,\n                {:?},\n            )?))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
        } else {
            output.push_str("            helpers::insert_op(ctx, mbarrier, block_ptr, last_op);\n");
            output.push_str(
                "            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, mbarrier, block_map, loc)))\n            } else {\n",
            );
            writeln!(
                output,
                "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
            output.push_str("            }\n");
        }
        output.push_str("        }\n");
    }
    output
}

fn mbarrier_extended_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in mbarrier_extended(catalog) {
        let contract = record.mbarrier_extended.as_ref().unwrap();
        let (arguments, returns_value): (Vec<(usize, &str)>, bool) = match contract.adapter {
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount => {
                (vec![(0, "barrier"), (2, "bytes")], true)
            }
            MbarrierExtendedAdapter::RawClusterAddressToVoid => (vec![(0, "address")], false),
            MbarrierExtendedAdapter::PointerTokenToPredicate => {
                (vec![(0, "barrier"), (1, "token")], true)
            }
            MbarrierExtendedAdapter::PointerParityToPredicate => {
                (vec![(0, "barrier"), (1, "parity")], true)
            }
            MbarrierExtendedAdapter::ZeroOperandsToVoid => (vec![], false),
            MbarrierExtendedAdapter::NanosecondsToVoid => (vec![(0, "ns")], false),
        };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {}, &loc)?;",
            record.rust.arguments.len()
        )
        .unwrap();
        if arguments.is_empty() {
            output.push_str("            let last_op = prev_op;\n");
        }
        for (position, (argument_index, argument_name)) in arguments.iter().enumerate() {
            let previous = if position == 0 { "prev_op" } else { "last_op" };
            writeln!(
                output,
                "            let ({argument_name}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{argument_index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
            )
            .unwrap();
        }
        let build_arguments = arguments
            .iter()
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "            let extended = {}::build(ctx{}{});",
            record.dialect.op_type,
            if build_arguments.is_empty() { "" } else { ", " },
            build_arguments,
        )
        .unwrap();
        output.push_str("            extended.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, extended, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        if returns_value {
            render_prepare_destination_write(&mut output, "last_op");
            output.push_str(
                "            helpers::insert_op(ctx, extended, block_ptr, prepared_last_op);\n",
            );
            output.push_str("            let result = extended.deref(ctx).get_result(0);\n");
            writeln!(
                output,
                "            Ok(Some(helpers::emit_prepared_result_and_goto(\n                ctx, prepared_destination, result, target, block_ptr, extended, value_map, block_map, loc,\n                {:?},\n            )?))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
        } else {
            output.push_str("            helpers::insert_op(ctx, extended, block_ptr, last_op);\n");
            output.push_str(
                "            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, extended, block_map, loc)))\n            } else {\n",
            );
            writeln!(
                output,
                "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                format!("{} call without target block", record.rust.name)
            )
            .unwrap();
            output.push_str("            }\n");
        }
        output.push_str("        }\n");
    }
    output
}

fn sync_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in sync_intrinsics(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
        writeln!(
            output,
            "            let barrier = Operation::new(ctx, {}::get_concrete_op_info(), vec![], vec![], vec![], 0);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            barrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, barrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, barrier, block_ptr, prev_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, barrier, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn execution_control_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in execution_controls(catalog) {
        let operation = ExecutionControlOperation::from_catalog_id(&record.id)
            .expect("closed execution-control record");
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {}, &loc)?;",
            operation.operand_count()
        )
        .unwrap();
        if operation.requires_immediate_operands() {
            output.push_str(
                "            if !matches!(&args[0], mir::Operand::Constant(_)) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"setmaxnreg requires a compile-time register count in 24..=256 divisible by 8\".to_owned()\n                    )\n                );\n            }\n            let (register_count_value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n            let register_count = register_count_value\n                .defining_op()\n                .and_then(|defining_op| Operation::get_op::<MirConstantOp>(defining_op, ctx))\n                .and_then(|constant| constant.get_attr_value(ctx))\n                .map(|value| value.value().to_u64())\n                .and_then(|value| u32::try_from(value).ok())\n                .filter(|value| (24..=256).contains(value) && value % 8 == 0);\n            let Some(register_count) = register_count else {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"setmaxnreg requires a compile-time register count in 24..=256 divisible by 8\".to_owned()\n                    )\n                );\n            };\n",
            );
            writeln!(
                output,
                "            let control = {}::build(ctx, register_count);",
                record.dialect.op_type
            )
            .unwrap();
        } else if operation.operand_count() == 0 {
            output.push_str("            let last_op = prev_op;\n");
            writeln!(
                output,
                "            let control = Operation::new(ctx, {}::get_concrete_op_info(), vec![], vec![], vec![], 0);",
                record.dialect.op_type
            )
            .unwrap();
        } else {
            output.push_str(
                "            let mut last_op = prev_op;\n            let mut operands = Vec::with_capacity(args.len());\n            for arg in args {\n                let (value, translated) = rvalue::translate_operand(\n                    ctx, body, arg, value_map, block_ptr, last_op, loc.clone(),\n                )?;\n                last_op = translated;\n                operands.push(value);\n            }\n",
            );
            writeln!(
                output,
                "            let control = Operation::new(ctx, {}::get_concrete_op_info(), vec![], operands, vec![], 0);",
                record.dialect.op_type
            )
            .unwrap();
        }
        output.push_str("            control.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, control, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, control, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, control, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output
}

fn tma_arms(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    for record in tma_intrinsics(catalog) {
        let operation = record.tma.as_ref().unwrap().operation;
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        if matches!(
            operation,
            TmaOperation::WaitGroup | TmaOperation::WaitGroupRead
        ) {
            output.push_str(
                "            require_arity(name, args.len(), 1, &loc)?;\n            if !matches!(args.first(), Some(mir::Operand::Constant(_))) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"TMA wait-group count must be a compile-time constant\".to_owned()\n                    )\n                );\n            }\n",
            );
        }
        let marker = intrinsic_marker(catalog, record);
        match operation {
            TmaOperation::G2sTile1d
            | TmaOperation::G2sTile2d
            | TmaOperation::G2sTile3d
            | TmaOperation::G2sTile4d
            | TmaOperation::G2sTile5d => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_g2s(\n                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, {}, {marker:?},\n            )?))",
                    operation.dimensions().unwrap()
                )
                .unwrap();
            }
            TmaOperation::G2sTile2dMulticast => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_g2s_multicast(\n                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, {marker:?},\n            )?))"
                )
                .unwrap();
            }
            TmaOperation::G2sTile2dMulticastCg2 => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_g2s_multicast_cg2(\n                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, {marker:?},\n            )?))"
                )
                .unwrap();
            }
            TmaOperation::S2gTile1d
            | TmaOperation::S2gTile2d
            | TmaOperation::S2gTile3d
            | TmaOperation::S2gTile4d
            | TmaOperation::S2gTile5d => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_s2g(\n                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, {}, {marker:?},\n            )?))",
                    operation.dimensions().unwrap()
                )
                .unwrap();
            }
            TmaOperation::CommitGroup => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_commit_group(\n                ctx, args, target, block_ptr, prev_op, block_map, loc, {marker:?},\n            )?))"
                )
                .unwrap();
            }
            TmaOperation::WaitGroup | TmaOperation::WaitGroupRead => {
                writeln!(
                    output,
                    "            Ok(Some(super::super::tma::emit_tma_wait_group(\n                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, {}, {marker:?},\n            )?))",
                    operation == TmaOperation::WaitGroupRead
                )
                .unwrap();
            }
            _ => {
                let arity = record.dialect.operands.len();
                writeln!(
                    output,
                    "            require_arity(name, args.len(), {arity}, &loc)?;"
                )
                .unwrap();
                if matches!(
                    record.tma.as_ref().unwrap().adapter,
                    TmaAdapter::DescriptorOrdinalAndU32
                        | TmaAdapter::DescriptorOrdinalAndU64
                        | TmaAdapter::DescriptorAndImmediateU32
                ) {
                    output.push_str(
                        "            if !matches!(args.get(1), Some(mir::Operand::Constant(_))) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"tensor-map replacement selector must be a compile-time constant\".to_owned()\n                    )\n                );\n            }\n",
                    );
                }
                output.push_str(
                    "            let mut last_op = prev_op;\n            let mut operands = Vec::with_capacity(args.len());\n            for arg in args {\n                let (value, translated) = rvalue::translate_operand(\n                    ctx, body, arg, value_map, block_ptr, last_op, loc.clone(),\n                )?;\n                last_op = translated;\n                operands.push(value);\n            }\n",
                );
                writeln!(
                    output,
                    "            let intrinsic = Operation::new(ctx, dialect_nvvm::ops::{}::get_concrete_op_info(), vec![], operands, vec![], 0);",
                    record.dialect.op_type
                )
                .unwrap();
                output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
                writeln!(
                    output,
                    "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {marker:?});"
                )
                .unwrap();
                output.push_str(
                    "            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, intrinsic, block_map, loc)))\n            } else {\n",
                );
                writeln!(
                    output,
                    "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
                    format!("{} call without target block", record.rust.name)
                )
                .unwrap();
                output.push_str("            }\n");
            }
        }
        output.push_str("        }\n");
    }
    output
}

fn importer_helpers_body(catalog: &CatalogFile) -> String {
    let mut output = String::new();
    output.push_str(
        "pub(super) fn require_arity(\n    name: &str,\n    actual: usize,\n    expected: usize,\n    loc: &Location,\n) -> TranslationResult<()> {\n    if actual != expected {\n        return input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"generated intrinsic `{name}` expects {expected} arguments, got {actual}\"\n            ))\n        );\n    }\n    Ok(())\n}\n",
    );
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        output.push_str(
            r#"

pub(super) fn generated_tcgen05_mma_selector(operand: Option<&mir::Operand>) -> Option<u32> {
    use rustc_public::ty::{ConstantKind, TyConstKind};
    let mir::Operand::Constant(constant) = operand? else { return None; };
    let value: u128 = match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => alloc.read_uint().ok()?,
        ConstantKind::Ty(value) => match value.kind() {
            TyConstKind::Value(_, alloc) => alloc.read_uint().ok()?,
            _ => value.eval_target_usize().ok()? as u128,
        },
        ConstantKind::Unevaluated(_) | ConstantKind::Param(_) => return None,
        ConstantKind::ZeroSized => return None,
    };
    u32::try_from(value).ok()
}
"#,
        );
    }
    if tcgen05_intrinsics(catalog)
        .any(|record| record.tcgen05.as_ref().unwrap().operation == Tcgen05Operation::St)
    {
        output.push_str(
            r#"

#[allow(clippy::too_many_arguments)]
pub(super) fn import_generated_tcgen05_store_operands(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    expected_len: usize,
    has_half_split_offset: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: Location,
) -> TranslationResult<(Vec<Value>, Option<Ptr<Operation>>)> {
    let expected_arity = if has_half_split_offset { 3 } else { 2 };
    if args.len() != expected_arity || !(1..=128).contains(&expected_len) {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "generated tcgen05 store has an invalid address, offset, or register count".to_owned()
            )
        );
    }
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let (address, last_op) = rvalue::translate_operand(
        ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),
    )?;
    let (half_split_offset, last_op) = if has_half_split_offset {
        let (offset, translated) = rvalue::translate_operand(
            ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),
        )?;
        if offset
            .defining_op()
            .and_then(|op| Operation::get_op::<MirConstantOp>(op, ctx))
            .is_none()
        {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "tcgen05 16x32bx2 half-split offset must lower to a constant".to_owned()
                )
            );
        }
        (Some(offset), translated)
    } else {
        (None, last_op)
    };
    let data_index = if has_half_split_offset { 2 } else { 1 };
    let (data, mut last_op) = rvalue::translate_operand(
        ctx, body, &args[data_index], value_map, block_ptr, last_op, loc.clone(),
    )?;
    if expected_len == 1 {
        if data.get_type(ctx) != u32_ty {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "generated tcgen05 store data must be a u32 register".to_owned()
                )
            );
        }
        let mut operands = vec![address];
        operands.extend(half_split_offset);
        operands.push(data);
        return Ok((operands, last_op));
    }

    let data_ty = data.get_type(ctx);
    let direct_array = data_ty
        .deref(ctx)
        .downcast_ref::<MirArrayType>()
        .is_some_and(|array| {
            array.size() == expected_len as u64 && array.element_type() == u32_ty
        });
    let array = if direct_array {
        data
    } else {
        let wrapped_array_ty = data_ty
            .deref(ctx)
            .downcast_ref::<MirStructType>()
            .filter(|structure| structure.field_count() == 1)
            .and_then(|structure| structure.get_field_type(0));
        let valid_wrapper = wrapped_array_ty.is_some_and(|field_ty| {
            field_ty
                .deref(ctx)
                .downcast_ref::<MirArrayType>()
                .is_some_and(|array| {
                    array.size() == expected_len as u64 && array.element_type() == u32_ty
                })
        });
        if !valid_wrapper {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "generated tcgen05 store data must contain {expected_len} u32 registers"
                ))
            );
        }
        let array_ty = wrapped_array_ty.expect("validated tcgen05 store wrapper");
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![array_ty],
            vec![data],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        let extract = MirExtractFieldOp::new(extract);
        extract.set_attr_index(ctx, FieldIndexAttr(0));
        helpers::insert_op(ctx, extract.get_operation(), block_ptr, last_op);
        last_op = Some(extract.get_operation());
        extract.get_operation().deref(ctx).get_result(0)
    };

    let mut operands = Vec::with_capacity(expected_len + 1 + usize::from(has_half_split_offset));
    operands.push(address);
    operands.extend(half_split_offset);
    for index in 0..expected_len {
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![u32_ty],
            vec![array],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        let extract = MirExtractFieldOp::new(extract);
        extract.set_attr_index(ctx, FieldIndexAttr(index as u32));
        helpers::insert_op(ctx, extract.get_operation(), block_ptr, last_op);
        last_op = Some(extract.get_operation());
        operands.push(extract.get_operation().deref(ctx).get_result(0));
    }
    Ok((operands, last_op))
}
"#,
        );
    }
    if register_mmas(catalog).next().is_some() || sparse_mmas(catalog).next().is_some() {
        output.push_str(
            r#"

#[derive(Clone, Copy)]
pub(super) enum GeneratedMmaImportAdapter {
    C2U32A2U32B1U32ToD2U32,
    C2U32A2U32B2U32ToD2U32,
    C2U32A4U32B2U32ToD2U32,
    C2U32A4U32B4U32ToD2U32,
    C4F32A2U32B1U32ToD4F32,
    C4F32A2U32B2U32ToD4F32,
    C4F32A4U32B2U32ToD4F32,
    C4F32A4U32B4U32ToD4F32,
    C2F64A1F64B1F64ToD2F64,
    C2I32A1U32B1U32ToD2I32,
    C4I32A4U32B2U32ToD4I32,
    C4I32A4U32B4U32ToD4I32,
    C4I32A2U32B1U32ToD4I32,
    C4I32A2U32B2U32ToD4I32,
    C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32,
}

#[allow(clippy::too_many_arguments)]
fn extract_generated_mma_array(
    ctx: &mut Context,
    array: Value,
    expected_element_ty: TypeHandle,
    expected_len: usize,
    block_ptr: Ptr<BasicBlock>,
    mut last_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Vec<Value>, Ptr<Operation>)> {
    let valid = array
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirArrayType>()
        .is_some_and(|array_ty| {
            array_ty.size() == expected_len as u64
                && array_ty.element_type() == expected_element_ty
        });
    if !valid {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "generated MMA fragment must be an array of {expected_len} scalar registers"
            ))
        );
    }
    let mut registers = Vec::with_capacity(expected_len);
    for index in 0..expected_len {
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![expected_element_ty],
            vec![array],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        let extract = MirExtractFieldOp::new(extract);
        extract.set_attr_index(ctx, FieldIndexAttr(index as u32));
        helpers::insert_op(ctx, extract.get_operation(), block_ptr, last_op);
        last_op = Some(extract.get_operation());
        registers.push(extract.get_operation().deref(ctx).get_result(0));
    }
    Ok((registers, last_op.expect("non-empty generated MMA fragment")))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn import_generated_mma_operands(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: Location,
    adapter: GeneratedMmaImportAdapter,
) -> TranslationResult<(Vec<Value>, Ptr<Operation>, TypeHandle, usize)> {
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let f64_ty: TypeHandle = FP64Type::get(ctx).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let (c_ty, c_count, a_ty, a_count, a_array, b_ty, b_count, b_array, result_ty, result_count) =
        match adapter {
            GeneratedMmaImportAdapter::C2U32A2U32B1U32ToD2U32 =>
                (u32_ty, 2, u32_ty, 2, true, u32_ty, 1, false, u32_ty, 2),
            GeneratedMmaImportAdapter::C2U32A2U32B2U32ToD2U32 =>
                (u32_ty, 2, u32_ty, 2, true, u32_ty, 2, true, u32_ty, 2),
            GeneratedMmaImportAdapter::C2U32A4U32B2U32ToD2U32 =>
                (u32_ty, 2, u32_ty, 4, true, u32_ty, 2, true, u32_ty, 2),
            GeneratedMmaImportAdapter::C2U32A4U32B4U32ToD2U32 =>
                (u32_ty, 2, u32_ty, 4, true, u32_ty, 4, true, u32_ty, 2),
            GeneratedMmaImportAdapter::C4F32A2U32B1U32ToD4F32 =>
                (f32_ty, 4, u32_ty, 2, true, u32_ty, 1, false, f32_ty, 4),
            GeneratedMmaImportAdapter::C4F32A2U32B2U32ToD4F32 =>
                (f32_ty, 4, u32_ty, 2, true, u32_ty, 2, true, f32_ty, 4),
            GeneratedMmaImportAdapter::C4F32A4U32B2U32ToD4F32 =>
                (f32_ty, 4, u32_ty, 4, true, u32_ty, 2, true, f32_ty, 4),
            GeneratedMmaImportAdapter::C4F32A4U32B4U32ToD4F32 =>
                (f32_ty, 4, u32_ty, 4, true, u32_ty, 4, true, f32_ty, 4),
            GeneratedMmaImportAdapter::C2F64A1F64B1F64ToD2F64 =>
                (f64_ty, 2, f64_ty, 1, false, f64_ty, 1, false, f64_ty, 2),
            GeneratedMmaImportAdapter::C2I32A1U32B1U32ToD2I32 =>
                (i32_ty, 2, u32_ty, 1, false, u32_ty, 1, false, i32_ty, 2),
            GeneratedMmaImportAdapter::C4I32A4U32B2U32ToD4I32 =>
                (i32_ty, 4, u32_ty, 4, true, u32_ty, 2, true, i32_ty, 4),
            GeneratedMmaImportAdapter::C4I32A4U32B4U32ToD4I32 =>
                (i32_ty, 4, u32_ty, 4, true, u32_ty, 4, true, i32_ty, 4),
            GeneratedMmaImportAdapter::C4I32A2U32B1U32ToD4I32 =>
                (i32_ty, 4, u32_ty, 2, true, u32_ty, 1, false, i32_ty, 4),
            GeneratedMmaImportAdapter::C4I32A2U32B2U32ToD4I32 =>
                (i32_ty, 4, u32_ty, 2, true, u32_ty, 2, true, i32_ty, 4),
            GeneratedMmaImportAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 =>
                (f32_ty, 4, u32_ty, 4, true, u32_ty, 2, true, f32_ty, 4),
        };
    let (c_array, last_op) = rvalue::translate_operand(
        ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),
    )?;
    let (mut operands, last_op) = extract_generated_mma_array(
        ctx, c_array, c_ty, c_count, block_ptr, last_op, loc.clone(),
    )?;
    let (a_value, last_after_a) = rvalue::translate_operand(
        ctx, body, &args[1], value_map, block_ptr, Some(last_op), loc.clone(),
    )?;
    let (a_registers, last_op) = if a_array {
        extract_generated_mma_array(
            ctx, a_value, a_ty, a_count, block_ptr, last_after_a, loc.clone(),
        )?
    } else {
        (vec![a_value], last_after_a.expect("generated MMA A translation keeps predecessor"))
    };
    operands.extend(a_registers);
    let (b_value, last_after_b) = rvalue::translate_operand(
        ctx, body, &args[2], value_map, block_ptr, Some(last_op), loc.clone(),
    )?;
    let (b_registers, last_op) = if b_array {
        extract_generated_mma_array(
            ctx, b_value, b_ty, b_count, block_ptr, last_after_b, loc.clone(),
        )?
    } else {
        (vec![b_value], last_after_b.expect("generated MMA B translation keeps predecessor"))
    };
    operands.extend(b_registers);
    let mut last_op = last_op;
    if matches!(
        adapter,
        GeneratedMmaImportAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
    ) {
        for arg in &args[3..] {
            let (value, translated) = rvalue::translate_operand(
                ctx,
                body,
                arg,
                value_map,
                block_ptr,
                Some(last_op),
                loc.clone(),
            )?;
            last_op = translated.expect("generated block-scale MMA operand translation keeps predecessor");
            operands.push(value);
        }
    }
    Ok((operands, last_op, result_ty, result_count))
}

pub(super) fn bundle_generated_mma_results(
    ctx: &mut Context,
    mma: Ptr<Operation>,
    result_ty: TypeHandle,
    result_count: usize,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let results = (0..result_count)
        .map(|index| mma.deref(ctx).get_result(index))
        .collect();
    let array_ty = MirArrayType::get(ctx, result_ty, result_count as u64);
    let array = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array.deref_mut(ctx).set_loc(loc);
    array.insert_after(ctx, mma);
    (array.deref(ctx).get_result(0), array)
}
"#,
        );
    }
    output
}

fn render_importer_record_test(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    writeln!(
        output,
        "    #[test]\n    fn {}_uses_only_canonical_or_compatibility_defpaths() {{",
        record.id
    )
    .unwrap();
    writeln!(
        output,
        "        assert!(is_generated_intrinsic_path({:?}));",
        record.rust.canonical_path
    )
    .unwrap();
    writeln!(
        output,
        "        assert!(is_raw_generated_intrinsic_path({:?}));",
        record.rust.canonical_path
    )
    .unwrap();
    writeln!(
        output,
        "        assert_eq!(generated_intrinsic_marker({:?}), Some({:?}));",
        record.rust.canonical_path,
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    writeln!(
            output,
            "        assert!(matches!(classify_raw_intrinsic_path(\"cuda_intrinsics\", {:?}.into()), RawIntrinsicIdentity::Known(_)));",
            record.rust.canonical_path
        )
        .unwrap();
    writeln!(
        output,
        "        assert!(!is_generated_intrinsic_path({:?}));",
        record.rust.public_path
    )
    .unwrap();
    for compatibility_path in &record.rust.compatibility_paths {
        writeln!(
                output,
                "        assert!(is_generated_intrinsic_path({compatibility_path:?}));\n        assert!(!is_raw_generated_intrinsic_path({compatibility_path:?}));\n        assert_eq!(generated_intrinsic_marker({compatibility_path:?}), Some({:?}));",
                intrinsic_marker(catalog, record)
            )
            .unwrap();
    }
    writeln!(
            output,
            "        assert!(!is_generated_intrinsic_path(\"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v{}::{}\"));",
            catalog.intrinsic_abi + 1,
            record.rust.abi_id
        )
        .unwrap();
    output.push_str("    }\n");
}

const IMPORTER_GENERATED_DIR: &str =
    "crates/mir-importer/src/translator/terminator/intrinsics/generated";

/// tcgen05 dispatch sub-shards, keyed off the catalog contract member each
/// record carries (the same fields `tcgen05_mma_intrinsics` /
/// `tcgen05_non_mma_intrinsics` inspect, extended to ld/st/cp).
const TCGEN05_BUCKETS: &[&str] = &[
    "tcgen05_mma",
    "tcgen05_ld",
    "tcgen05_st",
    "tcgen05_cp",
    "tcgen05_other",
];

fn tcgen05_member_bucket(record: &CatalogIntrinsic) -> &'static str {
    let tcgen05 = record.tcgen05.as_ref().expect("tcgen05 record");
    if tcgen05.mma.is_some() {
        "tcgen05_mma"
    } else if tcgen05.ld.is_some() {
        "tcgen05_ld"
    } else if tcgen05.st.is_some() {
        "tcgen05_st"
    } else if tcgen05.cp.is_some() {
        "tcgen05_cp"
    } else {
        "tcgen05_other"
    }
}

fn tcgen05_bucket_arms(catalog: &CatalogFile, bucket: &str) -> String {
    let mut output = String::new();
    for record in
        tcgen05_intrinsics(catalog).filter(|record| tcgen05_member_bucket(record) == bucket)
    {
        if record.tcgen05.as_ref().unwrap().mma.is_some() {
            render_importer_tcgen05_mma_dispatch(&mut output, catalog, record);
        } else {
            render_importer_tcgen05_non_mma_dispatch(&mut output, catalog, record);
        }
    }
    output
}

/// Dispatch shards in the old single-file family emission order. `cp_async`
/// and `execution_control` coalesce the same catalog families as
/// `dialect-nvvm/src/ops/generated/`.
fn importer_dispatch_shards(catalog: &CatalogFile) -> Vec<(&'static str, String)> {
    let mut shards: Vec<(&'static str, String)> = vec![
        ("sreg", sreg_arms(catalog)),
        ("active_mask", active_mask_arms(catalog)),
        ("ldmatrix", ldmatrix_arms(catalog)),
        ("stmatrix", stmatrix_arms(catalog)),
        ("register_mma", register_mma_arms(catalog)),
        ("sparse_mma", sparse_mma_arms(catalog)),
        ("packed_atomic", packed_atomic_arms(catalog)),
        ("redux", redux_arms(catalog)),
        ("vote", vote_arms(catalog)),
        ("warp_match", warp_match_arms(catalog)),
        ("elect", elect_arms(catalog)),
        ("warp_barrier", warp_barrier_arms(catalog)),
        ("warp_shuffle", warp_shuffle_arms(catalog)),
        ("dotprod", dotprod_arms(catalog)),
        ("packed_alu", packed_alu_arms(catalog)),
        ("integer_minmax", integer_minmax_arms(catalog)),
        ("packed_conversion", packed_conversion_arms(catalog)),
        ("scalar_conversion", scalar_conversion_arms(catalog)),
        ("scalar_arithmetic", scalar_arithmetic_arms(catalog)),
        ("scalar_math", scalar_math_arms(catalog)),
        ("extended_minmax", extended_minmax_arms(catalog)),
        ("movmatrix", movmatrix_arms(catalog)),
        ("prmt", prmt_arms(catalog)),
        ("cluster_barrier", cluster_barrier_arms(catalog)),
        ("cluster_memory", cluster_memory_arms(catalog)),
        ("debug_control", debug_control_arms(catalog)),
        ("clc", clc_arms(catalog)),
        ("wgmma_control", wgmma_control_arms(catalog)),
        ("cp_async", cp_async_arms(catalog)),
        ("mbarrier_basic", mbarrier_basic_arms(catalog)),
        ("mbarrier_extended", mbarrier_extended_arms(catalog)),
        ("sync", sync_arms(catalog)),
        ("execution_control", execution_control_arms(catalog)),
        ("tma", tma_arms(catalog)),
    ];
    for bucket in TCGEN05_BUCKETS {
        shards.push((bucket, tcgen05_bucket_arms(catalog, bucket)));
    }
    shards.retain(|(_, arms)| !arms.is_empty());
    shards
}

/// Test shards keep whole families together (tcgen05 fits one test file).
const IMPORTER_TEST_SHARD_ORDER: &[&str] = &[
    "sreg",
    "active_mask",
    "ldmatrix",
    "stmatrix",
    "register_mma",
    "sparse_mma",
    "packed_atomic",
    "redux",
    "vote",
    "warp_match",
    "elect",
    "warp_barrier",
    "warp_shuffle",
    "dotprod",
    "packed_alu",
    "integer_minmax",
    "packed_conversion",
    "scalar_conversion",
    "scalar_arithmetic",
    "scalar_math",
    "extended_minmax",
    "movmatrix",
    "prmt",
    "cluster_barrier",
    "cluster_memory",
    "debug_control",
    "clc",
    "wgmma_control",
    "cp_async",
    "mbarrier_basic",
    "mbarrier_extended",
    "sync",
    "execution_control",
    "tma",
    "tcgen05",
];

fn importer_shard_for_family(family: &str) -> &'static str {
    match family {
        "cp_async_copy" | "cp_async_control" | "cp_async_mbarrier" => "cp_async",
        "counted_barrier" | "grid_dependency" | "register_control" => "execution_control",
        _ => IMPORTER_TEST_SHARD_ORDER
            .iter()
            .copied()
            .find(|shard| *shard == family)
            .unwrap_or_else(|| panic!("unmapped generated intrinsic family `{family}`")),
    }
}

fn importer_test_shards(catalog: &CatalogFile) -> Vec<(&'static str, Vec<&CatalogIntrinsic>)> {
    let mut shards: Vec<(&'static str, Vec<&CatalogIntrinsic>)> = IMPORTER_TEST_SHARD_ORDER
        .iter()
        .map(|name| (*name, Vec::new()))
        .collect();
    for record in &catalog.intrinsics {
        let shard = importer_shard_for_family(&record.family);
        shards
            .iter_mut()
            .find(|(name, _)| *name == shard)
            .expect("closed importer shard list")
            .1
            .push(record);
    }
    shards.retain(|(_, records)| !records.is_empty());
    shards
}

fn push_use_group<S: AsRef<str>>(output: &mut String, root: &str, items: &[S]) {
    match items {
        [] => {}
        [only] => writeln!(output, "use {root}::{};", only.as_ref()).unwrap(),
        _ => writeln!(
            output,
            "use {root}::{{{}}};",
            items
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap(),
    }
}

fn nested_use_items(prefix: &str, items: &[&str]) -> Option<String> {
    match items {
        [] => None,
        [only] => Some(format!("{prefix}::{only}")),
        _ => Some(format!("{prefix}::{{{}}}", items.join(", "))),
    }
}

/// Build the exact `use` list one importer shard needs by scanning its body.
/// `dispatch_signature` marks files whose `try_dispatch` signature always
/// needs the dispatch parameter types; `helpers_module` suppresses the
/// `super::helpers` group inside the helpers module itself.
fn importer_shard_imports(
    catalog: &CatalogFile,
    body: &str,
    dispatch_signature: bool,
    helpers_module: bool,
) -> String {
    let mut output = String::new();
    let mut error_items = Vec::new();
    if uses_identifier(body, "TranslationErr") {
        error_items.push("TranslationErr");
    }
    if dispatch_signature || uses_identifier(body, "TranslationResult") {
        error_items.push("TranslationResult");
    }
    push_use_group(&mut output, "crate::error", &error_items);

    let mut translator_items = Vec::new();
    if body.contains("rvalue::") {
        translator_items.push("rvalue");
    }
    if body.contains("helpers::") {
        translator_items.push("terminator::helpers");
    }
    if uses_identifier(body, "types") {
        translator_items.push("types");
    }
    if dispatch_signature || uses_identifier(body, "ValueMap") {
        translator_items.push("values::ValueMap");
    }
    push_use_group(&mut output, "crate::translator", &translator_items);

    let mut mir_dialect = Vec::new();
    if uses_identifier(body, "FieldIndexAttr") {
        mir_dialect.push("attributes::FieldIndexAttr".to_owned());
    }
    let mir_ops: Vec<&str> = [
        "MirConstantOp",
        "MirConstructArrayOp",
        "MirConstructStructOp",
        "MirConstructTupleOp",
        "MirExtractFieldOp",
        "MirUnreachableOp",
    ]
    .into_iter()
    .filter(|item| uses_identifier(body, item))
    .collect();
    if let Some(nested) = nested_use_items("ops", &mir_ops) {
        mir_dialect.push(nested);
    }
    let mir_types: Vec<&str> = ["MirArrayType", "MirStructType", "MirTupleType"]
        .into_iter()
        .filter(|item| uses_identifier(body, item))
        .collect();
    if let Some(nested) = nested_use_items("types", &mir_types) {
        mir_dialect.push(nested);
    }
    push_use_group(&mut output, "dialect_mir", &mir_dialect);

    let nvvm: Vec<String> = dialect_nvvm_ops_import_candidates(catalog)
        .into_iter()
        .filter(|item| uses_identifier(body, item))
        .collect();
    push_use_group(&mut output, "dialect_nvvm::ops", &nvvm);

    if dispatch_signature || uses_identifier(body, "BasicBlock") {
        output.push_str("use pliron::basic_block::BasicBlock;\n");
    }
    let builtin: Vec<&str> = ["FP32Type", "FP64Type", "IntegerType", "Signedness"]
        .into_iter()
        .filter(|item| uses_identifier(body, item))
        .collect();
    push_use_group(&mut output, "pliron::builtin::types", &builtin);
    if dispatch_signature || uses_identifier(body, "Context") || uses_identifier(body, "Ptr") {
        output.push_str("use pliron::context::{Context, Ptr};\n");
    }
    if body.contains("input_err!") {
        output.push_str("use pliron::input_err;\n");
    }
    let mut location_items = Vec::new();
    if body.contains(".set_loc(") {
        location_items.push("Located");
    }
    if dispatch_signature || uses_identifier(body, "Location") {
        location_items.push("Location");
    }
    push_use_group(&mut output, "pliron::location", &location_items);
    if body.contains("::get_concrete_op_info(") || body.contains(".get_operation()") {
        output.push_str("use pliron::op::Op;\n");
    }
    if dispatch_signature || uses_identifier(body, "Operation") {
        output.push_str("use pliron::operation::Operation;\n");
    }
    let mut type_items = Vec::new();
    if uses_identifier(body, "TypeHandle") {
        type_items.push("TypeHandle");
    }
    if body.contains(".get_type(") {
        type_items.push("Typed");
    }
    push_use_group(&mut output, "pliron::r#type", &type_items);
    if uses_identifier(body, "Value") {
        output.push_str("use pliron::value::Value;\n");
    }
    if dispatch_signature || uses_identifier(body, "mir") {
        output.push_str("use rustc_public::mir;\n");
    }

    if !helpers_module {
        let glue: Vec<&str> = [
            "GeneratedMmaImportAdapter",
            "bundle_generated_mma_results",
            "generated_tcgen05_mma_selector",
            "import_generated_mma_operands",
            "import_generated_tcgen05_store_operands",
            "require_arity",
        ]
        .into_iter()
        .filter(|item| uses_identifier(body, item))
        .collect();
        push_use_group(&mut output, "super::helpers", &glue);
    }
    output
}

const IMPORTER_DISPATCH_PARAMS: &[(&str, &str)] = &[
    ("ctx", "&mut Context"),
    ("body", "&mir::Body"),
    ("name", "&str"),
    ("args", "&[mir::Operand]"),
    ("destination", "&mir::Place"),
    ("target", "&Option<usize>"),
    ("block_ptr", "Ptr<BasicBlock>"),
    ("prev_op", "Option<Ptr<Operation>>"),
    ("value_map", "&mut ValueMap"),
    ("block_map", "&[Ptr<BasicBlock>]"),
    ("loc", "Location"),
];

fn importer_dispatch_file(catalog: &CatalogFile, hash: &str, shard: &str, arms: &str) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "//! Generated raw/compatibility dispatch arms: `{shard}` intrinsics.\n"
    )
    .unwrap();
    output.push_str(&importer_shard_imports(catalog, arms, true, false));
    output.push_str("\n#[allow(clippy::too_many_arguments)]\npub(super) fn try_dispatch(\n");
    for (parameter, ty) in IMPORTER_DISPATCH_PARAMS {
        let silent = *parameter != "name" && !uses_identifier(arms, parameter);
        writeln!(
            output,
            "    {}{parameter}: {ty},",
            if silent { "_" } else { "" }
        )
        .unwrap();
    }
    output.push_str(") -> TranslationResult<Option<Ptr<Operation>>> {\n    match name {\n");
    output.push_str(arms);
    output.push_str("        _ => Ok(None),\n    }\n}\n");
    output
}

fn importer_helpers_file(catalog: &CatalogFile, hash: &str) -> String {
    let body = importer_helpers_body(catalog);
    let mut output = rust_header(catalog, hash);
    output.push_str("//! Shared glue for the generated intrinsic dispatch shards.\n\n");
    output.push_str(&importer_shard_imports(catalog, &body, false, true));
    output.push('\n');
    output.push_str(&body);
    output.push('\n');
    output
}

fn importer_mod_file(
    catalog: &CatalogFile,
    hash: &str,
    shards: &[(&'static str, String)],
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated raw/compatibility path dispatch for CUDA intrinsics.\n\nuse crate::error::{TranslationErr, TranslationResult};\nuse crate::translator::values::ValueMap;\nuse pliron::basic_block::BasicBlock;\nuse pliron::context::{Context, Ptr};\nuse pliron::input_err;\nuse pliron::location::Location;\nuse pliron::operation::Operation;\nuse rustc_public::{CrateDef, mir, ty::FnDef};\n\nmod helpers;\n",
    );
    for (shard, _) in shards {
        writeln!(output, "mod {shard};").unwrap();
    }
    output.push_str("#[cfg(test)]\nmod tests;\n\n");
    append_importer_classification(&mut output, catalog);
    output.push_str(
        "#[allow(clippy::too_many_arguments)]\npub fn try_dispatch_generated_intrinsic(\n    ctx: &mut Context,\n    body: &mir::Body,\n    name: &str,\n    args: &[mir::Operand],\n    destination: &mir::Place,\n    target: &Option<usize>,\n    block_ptr: Ptr<BasicBlock>,\n    prev_op: Option<Ptr<Operation>>,\n    value_map: &mut ValueMap,\n    block_map: &[Ptr<BasicBlock>],\n    loc: Location,\n) -> TranslationResult<Option<Ptr<Operation>>> {\n",
    );
    for (shard, _) in shards {
        writeln!(
            output,
            "    if let Some(operation) = {shard}::try_dispatch(\n        ctx, body, name, args, destination, target, block_ptr, prev_op, value_map, block_map,\n        loc.clone(),\n    )? {{\n        return Ok(Some(operation));\n    }}"
        )
        .unwrap();
    }
    output.push_str("    Ok(None)\n}\n");
    output
}

fn importer_tests_mod_file(
    catalog: &CatalogFile,
    hash: &str,
    shards: &[(&'static str, Vec<&CatalogIntrinsic>)],
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated dispatch-path tests, grouped per intrinsic family.\n\nuse super::*;\n\n",
    );
    for (shard, _) in shards {
        writeln!(output, "mod {shard};").unwrap();
    }
    output.push_str(
        "\n    #[test]\n    fn raw_intrinsic_identity_classification_fails_closed() {\n        assert_eq!(\n            classify_raw_intrinsic_path(\"serde\", \"serde::helper\".into()),\n            RawIntrinsicIdentity::NotRawCrate\n        );\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v2::i0001\".into()),\n            RawIntrinsicIdentity::UnsupportedAbi(_)\n        ));\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i9999\".into()),\n            RawIntrinsicIdentity::UnknownId(_)\n        ));\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::helper\".into()),\n            RawIntrinsicIdentity::UnknownId(_)\n        ));\n    }\n",
    );
    output
}

fn importer_tests_shard_file(
    catalog: &CatalogFile,
    hash: &str,
    shard: &str,
    records: &[&CatalogIntrinsic],
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "//! Generated dispatch-path tests: `{shard}` intrinsics.\n\nuse super::super::*;\n"
    )
    .unwrap();
    for record in records {
        render_importer_record_test(&mut output, catalog, record);
    }
    output
}

pub(super) fn render_importer_files(catalog: &CatalogFile, hash: &str) -> Vec<(PathBuf, String)> {
    let shards = importer_dispatch_shards(catalog);
    let test_shards = importer_test_shards(catalog);
    let mut files = vec![
        (
            PathBuf::from(format!("{IMPORTER_GENERATED_DIR}/mod.rs")),
            importer_mod_file(catalog, hash, &shards),
        ),
        (
            PathBuf::from(format!("{IMPORTER_GENERATED_DIR}/helpers.rs")),
            importer_helpers_file(catalog, hash),
        ),
        (
            PathBuf::from(format!("{IMPORTER_GENERATED_DIR}/tests/mod.rs")),
            importer_tests_mod_file(catalog, hash, &test_shards),
        ),
    ];
    for (shard, arms) in &shards {
        files.push((
            PathBuf::from(format!("{IMPORTER_GENERATED_DIR}/{shard}.rs")),
            importer_dispatch_file(catalog, hash, shard, arms),
        ));
    }
    for (shard, records) in &test_shards {
        files.push((
            PathBuf::from(format!("{IMPORTER_GENERATED_DIR}/tests/{shard}.rs")),
            importer_tests_shard_file(catalog, hash, shard, records),
        ));
    }
    files
}

#[cfg(test)]
pub(super) fn render_importer(catalog: &CatalogFile, hash: &str) -> String {
    render_importer_files(catalog, hash)
        .into_iter()
        .map(|(_, contents)| contents)
        .collect::<Vec<_>>()
        .join("\n")
}
