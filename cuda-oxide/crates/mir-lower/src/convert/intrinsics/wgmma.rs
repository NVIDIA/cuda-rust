/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WGMMA conversion for Hopper `sm_90a`.

use crate::convert::intrinsics::common::*;
use dialect_nvvm::ops::{
    WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op, WgmmaMmaPipelineValuesM64N64K16F32Bf16Op,
};
use llvm_export::ops as llvm;
use llvm_export::types::{self as llvm_types, VoidType};
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeHandle;

const VALUE_ACCUMULATOR_COUNT: usize = 32;
const M64N128_VALUE_ACCUMULATOR_COUNT: usize = 64;
const COUNTED_LOOP_CONTROL_COUNT: usize = 5;
const COUNTED_PIPELINE_MAX_PENDING_GROUPS: u8 = 2;

#[derive(Clone, Copy)]
enum WgmmaInputKind {
    Bf16,
    F16,
    Tf32,
}

impl WgmmaInputKind {
    fn ptx_suffix(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16.bf16",
            Self::F16 => "f16.f16",
            Self::Tf32 => unreachable!("TF32 has no counted K-loop lowering"),
        }
    }
}

/// Convert WGMMA make_smem_desc to inline PTX.
pub(crate) fn convert_make_smem_desc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("wgmma_make_smem_desc requires operand");
    }
    let ptr = operands[0];
    let ptr_casted = cast_to_shared_addrspace(ctx, rewriter, ptr);

    let asm_template = r#"{
    .reg .u64 addr;
    cvta.to.shared.u64 addr, $1;
    shr.u64 addr, addr, 4;
    and.b64 addr, addr, 0x3FFF;
    or.b64 $0, addr, 0xC000000800080000;
}"#;

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i64_ty.into(),
        vec![ptr_casted],
        asm_template,
        "=l,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

fn accumulator_register_list() -> String {
    (0..32)
        .map(|index| format!("%acc{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn deferred_group_template(mma_count: usize) -> String {
    let mut template = String::from("{\n    .reg .f32 %acc<32>;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    ld.f32 %acc{index}, [$0 + {offset}];\n"));
    }

    template.push_str("    wgmma.fence.sync.aligned;\n");
    let registers = accumulator_register_list();
    for mma_index in 0..mma_count {
        let desc_a = 1 + mma_index * 2;
        let desc_b = desc_a + 1;
        template.push_str(&format!(
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{registers}}}, ${desc_a}, ${desc_b}, 1, 1, 1, 0, 0;\n"
        ));
    }
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    st.f32 [$0 + {offset}], %acc{index};\n"));
    }
    template.push('}');
    template
}

fn value_accumulator_operand_list(accumulator_count: usize) -> String {
    (0..accumulator_count)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn value_group_template_for_mnemonic(
    mma_count: usize,
    accumulator_count: usize,
    mnemonic: &str,
    controls: &str,
) -> String {
    let mut template = String::from("{\n    wgmma.fence.sync.aligned;\n");
    let accumulators = value_accumulator_operand_list(accumulator_count);
    let descriptor_base = accumulator_count * 2;

    for mma_index in 0..mma_count {
        let desc_a = descriptor_base + mma_index * 2;
        let desc_b = desc_a + 1;
        template.push_str(&format!(
            "    {mnemonic} \
             {{{accumulators}}}, ${desc_a}, ${desc_b}, {controls};\n"
        ));
    }

    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn value_group_template(mma_count: usize, input_kind: WgmmaInputKind) -> String {
    let (mnemonic, controls) = match input_kind {
        WgmmaInputKind::Bf16 => (
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16",
            "1, 1, 1, 0, 0",
        ),
        WgmmaInputKind::F16 => (
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16",
            "1, 1, 1, 0, 0",
        ),
        WgmmaInputKind::Tf32 => (
            "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32",
            "1, 1, 1",
        ),
    };
    value_group_template_for_mnemonic(mma_count, VALUE_ACCUMULATOR_COUNT, mnemonic, controls)
}

fn value_group_template_m64n128_bf16(mma_count: usize) -> String {
    value_group_template_for_mnemonic(
        mma_count,
        M64N128_VALUE_ACCUMULATOR_COUNT,
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
        "1, 1, 1, 0, 0",
    )
}

fn value_group_constraints_for_count(accumulator_count: usize, descriptor_count: usize) -> String {
    let mut constraints = vec!["=f".to_owned(); accumulator_count];
    constraints.extend((0..accumulator_count).map(|index| index.to_string()));
    constraints.extend((0..descriptor_count).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

#[cfg(test)]
fn value_group_constraints(descriptor_count: usize) -> String {
    value_group_constraints_for_count(VALUE_ACCUMULATOR_COUNT, descriptor_count)
}

fn counted_loop_template(input_kind: WgmmaInputKind) -> String {
    let accumulators = value_accumulator_operand_list(VALUE_ACCUMULATOR_COUNT);
    let descriptor_base = VALUE_ACCUMULATOR_COUNT * 2;
    let desc_a_base = descriptor_base;
    let desc_b_base = descriptor_base + 1;
    let desc_a_step = descriptor_base + 2;
    let desc_b_step = descriptor_base + 3;
    let trip_count = descriptor_base + 4;

    let mut template = String::from(
        "{\n    .reg .u64 %desc_a;\n    .reg .u64 %desc_b;\n    .reg .u64 %remaining;\n    .reg .pred %loop_more;\n",
    );
    template.push_str(&format!("    mov.u64 %desc_a, ${desc_a_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b, ${desc_b_base};\n"));
    template.push_str(&format!("    mov.u64 %remaining, ${trip_count};\n"));
    template.push_str("    wgmma.fence.sync.aligned;\n");
    template.push_str("    setp.eq.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_done_${:uid};\n");
    template.push_str("L__wgmma_loop_${:uid}:\n");
    let input_types = input_kind.ptx_suffix();
    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.{input_types} \
         {{{accumulators}}}, %desc_a, %desc_b, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str(&format!("    add.u64 %desc_a, %desc_a, ${desc_a_step};\n"));
    template.push_str(&format!("    add.u64 %desc_b, %desc_b, ${desc_b_step};\n"));
    template.push_str("    sub.u64 %remaining, %remaining, 1;\n");
    template.push_str("    setp.ne.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_loop_${:uid};\n");
    template.push_str("L__wgmma_done_${:uid}:\n");
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn counted_pipeline_template(slot_count: usize, max_pending_groups: u8) -> String {
    let result_count = slot_count * VALUE_ACCUMULATOR_COUNT;
    let control_base = result_count * 2;
    let step_base = control_base + slot_count * 2;
    let trip_count = control_base + slot_count * 4;

    let mut template = String::from("{\n");
    for slot in 0..slot_count {
        template.push_str(&format!("    .reg .u64 %desc_a{slot};\n"));
        template.push_str(&format!("    .reg .u64 %desc_b{slot};\n"));
    }
    template.push_str("    .reg .u64 %remaining;\n    .reg .pred %loop_more;\n");

    for slot in 0..slot_count {
        let desc_a_base = control_base + slot * 2;
        let desc_b_base = desc_a_base + 1;
        template.push_str(&format!("    mov.u64 %desc_a{slot}, ${desc_a_base};\n"));
        template.push_str(&format!("    mov.u64 %desc_b{slot}, ${desc_b_base};\n"));
    }
    template.push_str(&format!("    mov.u64 %remaining, ${trip_count};\n"));
    template.push_str("    wgmma.fence.sync.aligned;\n");
    template.push_str("    setp.eq.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_pipeline_done_${:uid};\n");
    template.push_str("L__wgmma_pipeline_loop_${:uid}:\n");

    for slot in 0..slot_count {
        let accumulators = pipeline_accumulator_operand_list(slot);
        template.push_str(&format!(
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{accumulators}}}, %desc_a{slot}, %desc_b{slot}, 1, 1, 1, 0, 0;\n"
        ));
        template.push_str("    wgmma.commit_group.sync.aligned;\n");
        template.push_str(&format!(
            "    wgmma.wait_group.sync.aligned {max_pending_groups};\n"
        ));
    }

    for slot in 0..slot_count {
        let desc_a_step = step_base + slot * 2;
        let desc_b_step = desc_a_step + 1;
        template.push_str(&format!(
            "    add.u64 %desc_a{slot}, %desc_a{slot}, ${desc_a_step};\n"
        ));
        template.push_str(&format!(
            "    add.u64 %desc_b{slot}, %desc_b{slot}, ${desc_b_step};\n"
        ));
    }

    template.push_str("    sub.u64 %remaining, %remaining, 1;\n");
    template.push_str("    setp.ne.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_pipeline_loop_${:uid};\n");
    template.push_str("L__wgmma_pipeline_done_${:uid}:\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn counted_pipeline_constraints(result_count: usize, control_count: usize) -> String {
    let mut constraints = vec!["=f".to_owned(); result_count];
    constraints.extend((0..result_count).map(|index| index.to_string()));
    constraints.extend((0..control_count).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

fn pipeline_accumulator_operand_list(slot: usize) -> String {
    let base = slot * VALUE_ACCUMULATOR_COUNT;
    (base..base + VALUE_ACCUMULATOR_COUNT)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn pipeline_template(slot_count: usize, group_count: usize, max_pending_groups: u8) -> String {
    let result_count = slot_count * VALUE_ACCUMULATOR_COUNT;
    let descriptor_base = result_count * 2;
    let mut template = String::from("{\n    wgmma.fence.sync.aligned;\n");

    for group_index in 0..group_count {
        let slot = group_index % slot_count;
        let accumulators = pipeline_accumulator_operand_list(slot);
        let desc_a = descriptor_base + group_index * 2;
        let desc_b = desc_a + 1;
        template.push_str(&format!(
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{accumulators}}}, ${desc_a}, ${desc_b}, 1, 1, 1, 0, 0;\n"
        ));
        template.push_str("    wgmma.commit_group.sync.aligned;\n");
        if group_index + 1 >= slot_count {
            template.push_str(&format!(
                "    wgmma.wait_group.sync.aligned {max_pending_groups};\n"
            ));
        }
    }

    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn pipeline_constraints(result_count: usize, descriptor_count: usize) -> String {
    let mut constraints = vec!["=f".to_owned(); result_count];
    constraints.extend((0..result_count).map(|index| index.to_string()));
    constraints.extend((0..descriptor_count).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

fn counted_loop_constraints() -> String {
    let mut constraints = vec!["=f".to_owned(); VALUE_ACCUMULATOR_COUNT];
    constraints.extend((0..VALUE_ACCUMULATOR_COUNT).map(|index| index.to_string()));
    constraints.extend((0..COUNTED_LOOP_CONTROL_COUNT).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

/// Lower a complete deferred BF16 WGMMA group.
///
/// The inline-PTX scope owns 32 explicit accumulator registers. It loads them
/// before the fence, issues every MMA, commits, waits for zero pending groups,
/// and writes them back only after the wait. This avoids exposing pending
/// accumulator values to LLVM or to memory.
pub(crate) fn convert_mma_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 || operands.len() % 2 == 0 {
        return pliron::input_err_noloc!(
            "deferred WGMMA group requires one accumulator pointer and one or more descriptor pairs"
        );
    }

    let mma_count = (operands.len() - 1) / 2;
    let template = deferred_group_template(mma_count);
    let mut constraints = vec!["l"; operands.len()];
    constraints.push("~{memory}");
    let constraints = constraints.join(",");

    inline_asm_convergent(
        ctx,
        rewriter,
        op,
        VoidType::get(ctx).into(),
        operands,
        &template,
        &constraints,
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Lower a value-form BF16 WGMMA group to one multi-result inline-PTX scope.
///
/// The first 32 input operands are tied to 32 `=f` outputs. Descriptor operands
/// follow the tied inputs. The entire fence/MMA+/commit/wait sequence remains in
/// one convergent side-effecting asm statement, so LLVM cannot insert a spill
/// boundary while an asynchronous WGMMA group is in flight.
pub(crate) fn convert_mma_group_values(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_group_values_for_kind(ctx, rewriter, op, WgmmaInputKind::Bf16)
}

/// Lower a value-form F16 WGMMA full-drain group to one multi-result inline-PTX scope.
pub(crate) fn convert_mma_group_values_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_group_values_for_kind(ctx, rewriter, op, WgmmaInputKind::F16)
}

/// Lower a value-form BF16 m64n128 WGMMA full-drain group.
pub(crate) fn convert_mma_group_values_m64n128_bf16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_group_values_for_shape(
        ctx,
        rewriter,
        op,
        M64N128_VALUE_ACCUMULATOR_COUNT,
        value_group_template_m64n128_bf16,
    )
}

/// Lower a value-form TF32 K=8 WGMMA full-drain group to one inline-PTX scope.
pub(crate) fn convert_mma_group_values_tf32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_group_values_for_kind(ctx, rewriter, op, WgmmaInputKind::Tf32)
}

fn convert_mma_group_values_for_kind(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    input_kind: WgmmaInputKind,
) -> Result<()> {
    convert_mma_group_values_for_shape(ctx, rewriter, op, VALUE_ACCUMULATOR_COUNT, |mma_count| {
        value_group_template(mma_count, input_kind)
    })
}

fn convert_mma_group_values_for_shape<F>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    accumulator_count: usize,
    template_builder: F,
) -> Result<()>
where
    F: FnOnce(usize) -> String,
{
    let loc = op.deref(ctx).loc();
    let result_count = op.deref(ctx).get_num_results();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    if result_count != accumulator_count {
        return pliron::input_err_noloc!(
            "value-form WGMMA group requires exactly {accumulator_count} accumulator results"
        );
    }
    if operands.len() < accumulator_count + 2 {
        return pliron::input_err_noloc!(
            "value-form WGMMA group requires {accumulator_count} accumulator inputs and one or more descriptor pairs"
        );
    }

    let descriptor_count = operands.len() - accumulator_count;
    if !descriptor_count.is_multiple_of(2) {
        return pliron::input_err_noloc!("value-form WGMMA group descriptors must form pairs");
    }

    let mma_count = descriptor_count / 2;
    let template = template_builder(mma_count);
    let constraints = value_group_constraints_for_count(accumulator_count, descriptor_count);

    let f32_ty = FP32Type::get(ctx);
    let struct_ty: TypeHandle = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![f32_ty.into(); accumulator_count],
            llvm_types::StructLayout::Unpacked,
        ),
    )
    .into();

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty,
        operands,
        &template,
        &constraints,
    );

    let aggregate = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(accumulator_count);
    for index in 0..accumulator_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error!(loc.clone(), "{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        extracted_values.push(extract.get_operation().deref(ctx).get_result(0));
    }

    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

/// Lower a counted BF16 WGMMA K-loop to one multi-result inline-PTX scope.
pub(crate) fn convert_mma_loop_values(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_loop_values_for_kind(ctx, rewriter, op, WgmmaInputKind::Bf16)
}

/// Lower a counted F16 WGMMA K-loop to one multi-result inline-PTX scope.
pub(crate) fn convert_mma_loop_values_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mma_loop_values_for_kind(ctx, rewriter, op, WgmmaInputKind::F16)
}

/// Lower one counted WGMMA K-loop to a tied-register inline-PTX scope.
///
/// The first 32 operands/results are tied accumulator registers. The remaining
/// operands are the descriptor bases, descriptor deltas, and trip count. Loop
/// control and descriptor updates stay inside the same convergent asm statement
/// as the asynchronous WGMMA instructions, so LLVM never observes an in-flight
/// accumulator lifetime.
fn convert_mma_loop_values_for_kind(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    input_kind: WgmmaInputKind,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let result_count = op.deref(ctx).get_num_results();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    if result_count != VALUE_ACCUMULATOR_COUNT {
        return pliron::input_err_noloc!(
            "counted-loop value-form WGMMA requires exactly 32 accumulator results"
        );
    }
    if operands.len() != VALUE_ACCUMULATOR_COUNT + COUNTED_LOOP_CONTROL_COUNT {
        return pliron::input_err_noloc!(
            "counted-loop value-form WGMMA requires 32 accumulator inputs and five loop-control operands"
        );
    }

    let template = counted_loop_template(input_kind);
    let constraints = counted_loop_constraints();

    let f32_ty = FP32Type::get(ctx);
    let struct_ty: TypeHandle = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![f32_ty.into(); VALUE_ACCUMULATOR_COUNT],
            llvm_types::StructLayout::Unpacked,
        ),
    )
    .into();

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty,
        operands,
        &template,
        &constraints,
    );
    let aggregate = asm_op.deref(ctx).get_result(0);

    let mut extracted_values = Vec::with_capacity(VALUE_ACCUMULATOR_COUNT);
    for index in 0..VALUE_ACCUMULATOR_COUNT {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error!(loc.clone(), "{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        extracted_values.push(extract.get_operation().deref(ctx).get_result(0));
    }

    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

/// Lower a counted WGMMA pipeline to one convergent inline-PTX scope.
///
/// The production counted-loop carrier supports the validated `wait_group<1>`
/// two-slot and `wait_group<2>` three-slot forms. Every accumulator slot is tied
/// across the same asm statement together with its descriptor recurrence and the
/// trip counter, so LLVM never observes a live asynchronous accumulator value.
pub(crate) fn convert_mma_loop_pipeline_values(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let result_count = op.deref(ctx).get_num_results();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let pipeline = Operation::get_op::<WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op>(op, ctx)
        .expect("counted-pipeline conversion must be invoked for the counted-pipeline WGMMA op");
    let Some(max_pending_groups) = pipeline.max_pending_groups(ctx) else {
        return pliron::input_err_noloc!("counted-pipeline WGMMA is missing max_pending_groups");
    };
    if !(1..=COUNTED_PIPELINE_MAX_PENDING_GROUPS).contains(&max_pending_groups) {
        return pliron::input_err_noloc!(
            "counted-pipeline WGMMA supports only max_pending_groups 1 or 2"
        );
    }

    let slot_count = usize::from(max_pending_groups) + 1;
    let expected_result_count = slot_count * VALUE_ACCUMULATOR_COUNT;
    let control_count = slot_count * 4 + 1;
    if result_count != expected_result_count {
        return pliron::input_err_noloc!(
            "counted-pipeline WGMMA requires max_pending_groups + 1 accumulator slots"
        );
    }
    if operands.len() != result_count + control_count {
        return pliron::input_err_noloc!(
            "counted-pipeline WGMMA requires two descriptor bases and two descriptor steps per accumulator slot plus one trip count"
        );
    }

    let template = counted_pipeline_template(slot_count, max_pending_groups);
    let constraints = counted_pipeline_constraints(result_count, control_count);
    let f32_ty = FP32Type::get(ctx);
    let struct_ty: TypeHandle = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![f32_ty.into(); result_count],
            llvm_types::StructLayout::Unpacked,
        ),
    )
    .into();

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty,
        operands,
        &template,
        &constraints,
    );
    let aggregate = asm_op.deref(ctx).get_result(0);

    let mut extracted_values = Vec::with_capacity(result_count);
    for index in 0..result_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error!(loc.clone(), "{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        extracted_values.push(extract.get_operation().deref(ctx).get_result(0));
    }

    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

/// Lower a multi-slot BF16 WGMMA pipeline to one convergent inline-PTX scope.
///
/// Each accumulator slot owns 32 tied `f32` registers. Groups are committed
/// independently and issued round-robin across `N + 1` slots for
/// `wait_group<N>`, ensuring a slot is not reused until its previous group has
/// completed. A final `wait_group<0>` occurs before any result escapes to LLVM.
pub(crate) fn convert_mma_pipeline_values(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let result_count = op.deref(ctx).get_num_results();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    if result_count == 0 || !result_count.is_multiple_of(VALUE_ACCUMULATOR_COUNT) {
        return pliron::input_err_noloc!(
            "pipeline value-form WGMMA requires whole 32-value accumulator slots"
        );
    }
    if operands.len() < result_count + 2 {
        return pliron::input_err_noloc!(
            "pipeline value-form WGMMA requires accumulator inputs and descriptor pairs"
        );
    }
    let descriptor_count = operands.len() - result_count;
    if !descriptor_count.is_multiple_of(2) {
        return pliron::input_err_noloc!("pipeline WGMMA descriptors must form pairs");
    }

    let pipeline = Operation::get_op::<WgmmaMmaPipelineValuesM64N64K16F32Bf16Op>(op, ctx)
        .expect("pipeline conversion must be invoked for the pipeline WGMMA op");
    let Some(max_pending_groups) = pipeline.max_pending_groups(ctx) else {
        return pliron::input_err_noloc!("pipeline WGMMA is missing max_pending_groups");
    };
    if !(1..=7).contains(&max_pending_groups) {
        return pliron::input_err_noloc!("pipeline WGMMA max_pending_groups must be in 1..=7");
    }
    let slot_count = usize::from(max_pending_groups) + 1;
    if result_count != slot_count * VALUE_ACCUMULATOR_COUNT {
        return pliron::input_err_noloc!(
            "pipeline WGMMA requires max_pending_groups + 1 accumulator slots"
        );
    }
    let group_count = descriptor_count / 2;
    if group_count < slot_count {
        return pliron::input_err_noloc!(
            "pipeline WGMMA requires at least max_pending_groups + 1 committed groups"
        );
    }

    let template = pipeline_template(slot_count, group_count, max_pending_groups);
    let constraints = pipeline_constraints(result_count, descriptor_count);
    let f32_ty = FP32Type::get(ctx);
    let struct_ty: TypeHandle = llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![f32_ty.into(); result_count],
            llvm_types::StructLayout::Unpacked,
        ),
    )
    .into();

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        struct_ty,
        operands,
        &template,
        &constraints,
    );
    let aggregate = asm_op.deref(ctx).get_result(0);

    let mut extracted_values = Vec::with_capacity(result_count);
    for index in 0..result_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error!(loc.clone(), "{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        extracted_values.push(extract.get_operation().deref(ctx).get_result(0));
    }

    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

/// Reject an unfused pointer-form MMA operation.
///
/// Reaching this converter means the pre-lowering adapter could not prove a
/// complete and sound straight-line region or canonical counted K-loop.
pub(crate) fn convert_mma(
    _ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    pliron::input_err_noloc!(
        "WGMMA MMA reached lowering without deferred accumulator fusion; expected a supported linear wait_group<0> region, a proven partial-wait pipeline, a canonical counted K-loop, or a supported counted pipeline"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        WgmmaInputKind, counted_loop_constraints, counted_loop_template,
        counted_pipeline_constraints, counted_pipeline_template, deferred_group_template,
        pipeline_constraints, pipeline_template, value_group_constraints, value_group_template,
        value_group_template_m64n128_bf16,
    };

    #[test]
    fn deferred_template_keeps_loads_before_wait_and_stores_after_wait() {
        let template = deferred_group_template(2);
        assert_eq!(template.matches("ld.f32 %acc").count(), 32);
        assert_eq!(template.matches("st.f32 [$0").count(), 32);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);

        let first_mma = template.find("wgmma.mma_async").unwrap();
        let wait = template.find("wgmma.wait_group.sync.aligned 0").unwrap();
        let first_store = template.find("st.f32 [$0").unwrap();
        assert!(first_mma < wait);
        assert!(wait < first_store);
    }

    #[test]
    fn value_template_uses_tied_accumulators_without_memory_round_trip() {
        let template = value_group_template(2, WgmmaInputKind::Bf16);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);
        assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            1
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(!template.contains(".reg .f32"));
        assert!(template.contains("$64, $65"));
        assert!(template.contains("$66, $67"));

        let constraints = value_group_constraints(4);
        assert_eq!(
            constraints
                .split(',')
                .filter(|value| *value == "=f")
                .count(),
            32
        );
        for index in 0..32 {
            let expected = index.to_string();
            assert!(
                constraints
                    .split(',')
                    .any(|value| value == expected.as_str())
            );
        }
        assert_eq!(
            constraints.split(',').filter(|value| *value == "l").count(),
            4
        );
        assert!(constraints.ends_with("~{memory}"));
    }

    #[test]
    fn f16_value_template_uses_k16_with_transpose_controls() {
        let template = value_group_template(2, WgmmaInputKind::F16);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);
        assert!(template.contains("m64n64k16.f32.f16.f16"));
        assert!(!template.contains(".bf16.bf16"));
        assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            1
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(template.contains("$64, $65"));
        assert!(template.contains("$66, $67"));
    }

    #[test]
    fn m64n128_value_template_uses_sixty_four_tied_accumulators() {
        let template = value_group_template_m64n128_bf16(2);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);
        assert!(template.contains("m64n128k16.f32.bf16.bf16"));
        assert!(template.contains("{$0, $1, $2"));
        assert!(template.contains("$63}"));
        assert!(template.contains("$128, $129"));
        assert!(template.contains("$130, $131"));
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
    }

    #[test]
    fn tf32_value_template_uses_k8_without_transpose_controls() {
        let template = value_group_template(2, WgmmaInputKind::Tf32);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);
        assert!(template.contains("m64n64k8.f32.tf32.tf32"));
        assert!(!template.contains("m64n64k16"));
        assert!(!template.contains(".bf16.bf16"));
        assert!(!template.contains(".f16.f16"));
        assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            1
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(template.contains("$64, $65, 1, 1, 1;"));
        assert!(template.contains("$66, $67, 1, 1, 1;"));
        assert!(!template.contains("$64, $65, 1, 1, 1, 0, 0;"));
    }

    #[test]
    fn counted_loop_template_keeps_descriptor_recurrence_inside_one_scope() {
        let template = counted_loop_template(WgmmaInputKind::Bf16);
        assert!(template.contains("m64n64k16.f32.bf16.bf16"));
        assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(template.matches("wgmma.mma_async").count(), 1);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            1
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(template.contains("L__wgmma_loop_${:uid}:"));
        assert!(template.contains("L__wgmma_done_${:uid}:"));
        assert!(template.contains("@%loop_more bra.uni L__wgmma_done_${:uid};"));
        assert!(template.contains("@%loop_more bra.uni L__wgmma_loop_${:uid};"));
        assert!(template.contains("mov.u64 %desc_a, $64;"));
        assert!(template.contains("mov.u64 %desc_b, $65;"));
        assert!(template.contains("add.u64 %desc_a, %desc_a, $66;"));
        assert!(template.contains("add.u64 %desc_b, %desc_b, $67;"));
        assert!(template.contains("mov.u64 %remaining, $68;"));
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(!template.contains(".reg .f32"));

        let constraints = counted_loop_constraints();
        assert_eq!(
            constraints
                .split(',')
                .filter(|value| *value == "=f")
                .count(),
            32
        );
        for index in 0..32 {
            let expected = index.to_string();
            assert!(
                constraints
                    .split(',')
                    .any(|value| value == expected.as_str())
            );
        }
        assert_eq!(
            constraints.split(',').filter(|value| *value == "l").count(),
            5
        );
        assert!(constraints.ends_with("~{memory}"));
    }

    #[test]
    fn f16_counted_loop_template_changes_only_the_input_element_types() {
        let template = counted_loop_template(WgmmaInputKind::F16);
        assert_eq!(template.matches("wgmma.mma_async").count(), 1);
        assert!(template.contains("m64n64k16.f32.f16.f16"));
        assert!(!template.contains(".bf16.bf16"));
        assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            1
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(template.contains("L__wgmma_loop_${:uid}:"));
        assert!(template.contains("L__wgmma_done_${:uid}:"));
        assert!(template.contains("mov.u64 %desc_a, $64;"));
        assert!(template.contains("mov.u64 %desc_b, $65;"));
        assert!(template.contains("add.u64 %desc_a, %desc_a, $66;"));
        assert!(template.contains("add.u64 %desc_b, %desc_b, $67;"));
        assert!(template.contains("mov.u64 %remaining, $68;"));
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(!template.contains(".reg .f32"));
    }

    #[test]
    fn counted_pipeline_template_supports_two_and_three_slot_wait_depths() {
        let two_slot = counted_pipeline_template(2, 1);
        assert_eq!(two_slot.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(two_slot.matches("wgmma.mma_async").count(), 2);
        assert_eq!(
            two_slot.matches("wgmma.commit_group.sync.aligned").count(),
            2
        );
        assert_eq!(
            two_slot.matches("wgmma.wait_group.sync.aligned 1").count(),
            2
        );
        assert_eq!(
            two_slot.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(two_slot.contains("{$0, $1, $2"));
        assert!(two_slot.contains("{$32, $33, $34"));
        assert!(two_slot.contains("mov.u64 %desc_a0, $128;"));
        assert!(two_slot.contains("mov.u64 %desc_b1, $131;"));
        assert!(two_slot.contains("add.u64 %desc_a0, %desc_a0, $132;"));
        assert!(two_slot.contains("add.u64 %desc_b1, %desc_b1, $135;"));
        assert!(two_slot.contains("mov.u64 %remaining, $136;"));

        let three_slot = counted_pipeline_template(3, 2);
        assert_eq!(three_slot.matches("wgmma.fence.sync.aligned").count(), 1);
        assert_eq!(three_slot.matches("wgmma.mma_async").count(), 3);
        assert_eq!(
            three_slot
                .matches("wgmma.commit_group.sync.aligned")
                .count(),
            3
        );
        assert_eq!(
            three_slot
                .matches("wgmma.wait_group.sync.aligned 2")
                .count(),
            3
        );
        assert_eq!(
            three_slot
                .matches("wgmma.wait_group.sync.aligned 0")
                .count(),
            1
        );
        assert!(three_slot.contains("{$0, $1, $2"));
        assert!(three_slot.contains("{$32, $33, $34"));
        assert!(three_slot.contains("{$64, $65, $66"));
        assert!(three_slot.contains("mov.u64 %desc_a0, $192;"));
        assert!(three_slot.contains("mov.u64 %desc_b2, $197;"));
        assert!(three_slot.contains("add.u64 %desc_a0, %desc_a0, $198;"));
        assert!(three_slot.contains("add.u64 %desc_b2, %desc_b2, $203;"));
        assert!(three_slot.contains("mov.u64 %remaining, $204;"));

        for template in [&two_slot, &three_slot] {
            assert!(template.contains("L__wgmma_pipeline_loop_${:uid}:"));
            assert!(template.contains("L__wgmma_pipeline_done_${:uid}:"));
            assert!(!template.contains("ld.f32"));
            assert!(!template.contains("st.f32"));
            assert!(!template.contains(".reg .f32"));
            assert!(
                !template.contains("\\\n"),
                "counted-pipeline PTX must not contain a literal line-continuation backslash: {template}"
            );
        }

        let two_slot_constraints = counted_pipeline_constraints(64, 9);
        assert_eq!(
            two_slot_constraints
                .split(',')
                .filter(|value| *value == "=f")
                .count(),
            64
        );
        assert_eq!(
            two_slot_constraints
                .split(',')
                .filter(|value| *value == "l")
                .count(),
            9
        );

        let three_slot_constraints = counted_pipeline_constraints(96, 13);
        assert_eq!(
            three_slot_constraints
                .split(',')
                .filter(|value| *value == "=f")
                .count(),
            96
        );
        assert_eq!(
            three_slot_constraints
                .split(',')
                .filter(|value| *value == "l")
                .count(),
            13
        );
        assert!(three_slot_constraints.ends_with("~{memory}"));
    }

    #[test]
    fn pipeline_template_throttles_groups_and_finishes_with_wait_zero() {
        let template = pipeline_template(2, 4, 1);
        assert_eq!(template.matches("wgmma.mma_async").count(), 4);
        assert_eq!(
            template.matches("wgmma.commit_group.sync.aligned").count(),
            4
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 1").count(),
            3
        );
        assert_eq!(
            template.matches("wgmma.wait_group.sync.aligned 0").count(),
            1
        );
        assert!(template.contains("{$0, $1, $2"));
        assert!(template.contains("{$32, $33, $34"));
        assert!(!template.contains("ld.f32"));
        assert!(!template.contains("st.f32"));
        assert!(!template.contains(".reg .f32"));

        let constraints = pipeline_constraints(64, 8);
        assert_eq!(
            constraints
                .split(',')
                .filter(|value| *value == "=f")
                .count(),
            64
        );
        assert_eq!(
            constraints.split(',').filter(|value| *value == "l").count(),
            8
        );
        assert!(constraints.ends_with("~{memory}"));
    }
}
