//! SM120 matrix instructions and TMA mappings with selector and identity rejection cases.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::{
    ops::{MirConstantOp, MirFuncOp, MirReturnOp},
    types::MirPtrType,
};
use dialect_nvvm::ops::*;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};

/// MIR/NVVM function with shared pointers and register carriers for SM120 primitive export tests.
struct Fixture {
    /// Source IR context owning the fixture's registered types and operations.
    ctx: Context,
    /// Source module containing the operations under test.
    module: ModuleOp,
    /// Function entry block where the fixture appends source operations.
    block: Ptr<BasicBlock>,
    /// Source function argument types used to build well-typed primitive operations.
    types: Vec<TypeHandle>,
}
impl Fixture {
    /// Build the reusable SM120 source context, module, and typed argument block.
    fn new() -> Self {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        let i8 = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let i32 = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let i64 = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32 = FP32Type::get(&ctx).into();
        let types = vec![
            MirPtrType::get(&mut ctx, i32, true, 0).into(),
            MirPtrType::get(&mut ctx, i8, true, 3).into(),
            MirPtrType::get(&mut ctx, i64, true, 3).into(),
            MirPtrType::get(&mut ctx, i8, false, 0).into(),
            i32,
            i32,
            f32,
            f32,
            f32,
            f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
        ];
        let module = ModuleOp::new(&mut ctx, "sm120".try_into().unwrap());
        let ft = FunctionType::get(&ctx, types.clone(), vec![]);
        let raw = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, raw, TypeAttr::new(ft.into()))
            .set_symbol_name(&mut ctx, "sm120_primitives".try_into().unwrap());
        raw.deref_mut(&ctx).attributes.set(
            "gpu_kernel".try_into().unwrap(),
            StringAttr::new("true".into()),
        );
        module.append_operation(&mut ctx, raw, 0);
        let block = BasicBlock::new(&mut ctx, None, types.clone());
        block.insert_at_back(raw.deref(&ctx).get_region(0), &ctx);
        Self {
            ctx,
            module,
            block,
            types,
        }
    }
    /// Return the indexed source function argument for constructing a test operation.
    fn arg(&self, i: usize) -> Value {
        self.block.deref(&self.ctx).get_argument(i)
    }
    /// Append a typed source operation to the fixture block and return its operation handle.
    fn append<O: Op>(
        &mut self,
        marker: &str,
        operands: Vec<Value>,
        types: Vec<TypeHandle>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            &mut self.ctx,
            O::get_concrete_op_info(),
            types,
            operands,
            vec![],
            0,
        );
        if !marker.is_empty() {
            op.deref_mut(&self.ctx).attributes.set(
                "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                StringAttr::new(marker.into()),
            );
        }
        op.insert_at_back(self.block, &self.ctx);
        op
    }
    /// Append an unsigned constant with a chosen bit width for intrinsic selectors.
    fn constant(&mut self, width: u32, value: u64) -> Value {
        let ty = IntegerType::get(&self.ctx, width, Signedness::Unsigned);
        let op = self.append::<MirConstantOp>("", vec![], vec![ty.into()]);
        MirConstantOp::new(op).set_attr_value(
            &self.ctx,
            IntegerAttr::new(
                ty,
                APInt::from_u64(value, (width as usize).try_into().unwrap()),
            ),
        );
        op.deref(&self.ctx).get_result(0)
    }
    /// Terminate the source function after the test's primitive operations.
    fn finish(&mut self) {
        self.append::<MirReturnOp>("", vec![], vec![]);
    }
    /// Export the fixture with the SM120 primitives profile and render its target MLIR.
    fn translate(&self) -> Result<String, String> {
        CutlassPrimitivesMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&self.ctx, &self.module)
            .and_then(|m| render_module(&m).map_err(Into::into))
            .map_err(|e| e.to_string())
    }
    /// Append the canonical reviewed shared matrix-load operation for mutation tests.
    fn load(&mut self) -> Ptr<Operation> {
        let address = self.arg(0);
        let load = LdmatrixOp::build(
            &mut self.ctx,
            address,
            LdmatrixShapeAttr::M8n8,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B16,
            LdmatrixStateSpaceAttr::Shared,
        );
        load.deref_mut(&self.ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new("v1:i0013".into()),
        );
        load.insert_at_back(self.block, &self.ctx);
        load
    }

    /// Append the canonical reviewed MXF4 register-MMA operation for mutation tests.
    fn mma(&mut self) -> Ptr<Operation> {
        let mut operands = (6..16).map(|i| self.arg(i)).collect::<Vec<_>>();
        // Distinct definitions prove each selector and scale retains its slot.
        operands.push(self.arg(16));
        operands.push(self.constant(16, 0));
        operands.push(self.constant(16, 0));
        operands.push(self.arg(17));
        operands.push(self.constant(16, 0));
        operands.push(self.constant(16, 0));
        let op = self.append::<RegisterMmaOp>("v1:i1039", operands, vec![self.types[6]; 4]);
        let mma = RegisterMmaOp::new(op);
        mma.set_attr_nvvm_register_mma_shape(&self.ctx, RegisterMmaShapeAttr::M16n8k64);
        mma.set_attr_nvvm_register_mma_operation(&self.ctx, RegisterMmaOperationAttr::Multiply);
        mma.set_attr_nvvm_register_mma_kind(&self.ctx, RegisterMmaKindAttr::Mxf4);
        mma.set_attr_nvvm_register_mma_accumulator(&self.ctx, RegisterMmaAccumulatorAttr::F32);
        mma.set_attr_nvvm_register_mma_a_element(&self.ctx, RegisterMmaElementAttr::E2m1);
        mma.set_attr_nvvm_register_mma_b_element(&self.ctx, RegisterMmaElementAttr::E2m1);
        mma.set_attr_nvvm_register_mma_a_layout(&self.ctx, RegisterMmaLayoutAttr::Row);
        mma.set_attr_nvvm_register_mma_b_layout(&self.ctx, RegisterMmaLayoutAttr::Col);
        mma.set_attr_nvvm_register_mma_overflow(&self.ctx, RegisterMmaOverflowAttr::NotApplicable);
        op
    }
}

/// Check MXF4 operands enter NVVM in A/B/C/scale order and every accumulator result is recovered.
#[test]
fn sm120_mxf4_preserves_a_b_c_scale_order_and_extracts_all_results() {
    let mut f = Fixture::new();
    let mma = f.mma();
    let results = mma.deref(&f.ctx).results().collect::<Vec<_>>();
    let top = f.append::<CvtF16x2F32Op>("v1:i0081", results[..2].to_vec(), vec![f.types[4]]);
    let bottom = f.append::<CvtF16x2F32Op>("v1:i0081", results[2..].to_vec(), vec![f.types[4]]);
    let top = top.deref(&f.ctx).get_result(0);
    let bottom = bottom.deref(&f.ctx).get_result(0);
    f.append::<StmatrixM8n8X2Op>("v1:i0301", vec![f.arg(1), top, bottom], vec![]);
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("\"nvvm.mma.block_scale\"(%v10, %v11, %v12, %v13, %v14, %v15, %v6, %v7, %v8, %v9, %v16, %v18, %v19, %v17, %v20, %v21)"), "{text}");
    for expected in [
        "#nvvm.shape<m = 16, n = 8, k = 64>",
        "#nvvm.scale_vec_size<x2>",
        "#nvvm.block_scale_kind<mxf4>",
        "#nvvm.block_scale_format<ue8m0>",
        "operandSegmentSizes = array<i32: 4, 2, 4, 1, 1, 1, 1, 1, 1>",
    ] {
        assert!(text.contains(expected), "{text}");
    }
    assert_eq!(text.matches("#nvvm.mma_type<e2m1>").count(), 2);
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}

/// Reject unsupported MXF4 selectors and catalog identities before emission.
#[test]
fn sm120_mxf4_rejects_other_variants_and_forged_identity() {
    let mut f = Fixture::new();
    let op = f.mma();
    f.finish();
    let mma = RegisterMmaOp::new(op);
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k32);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Mxf8f6f4);
    let error = f.translate().unwrap_err();
    assert!(error.contains("supports m16n8k64 MXF4"), "{error}");
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k64);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Mxf4);
    op.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0858".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i1039")
    );
}

/// Reject the SM120-only MXF4 instruction when targeting SM100.
#[test]
fn sm100_rejects_sm120_mxf4() {
    let mut f = Fixture::new();
    f.mma();
    f.finish();
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&f.ctx, &f.module)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("MXF4 register MMA requires sm_120a"),
        "{error}"
    );
    let error = CutlassFullCuteMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&f.ctx, &f.module)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("MXF4 register MMA requires sm_120a"),
        "{error}"
    );
}

/// Check packed FP16 operand lanes and scalar FP32 accumulator positions survive export.
#[test]
fn fp16_mma_preserves_packed_a_b_and_f32_accumulator_order() {
    let mut f = Fixture::new();
    let operands = (6..16).map(|i| f.arg(i)).collect();
    let op = f.append::<RegisterMmaOp>("v1:i0106", operands, vec![f.types[6]; 4]);
    let mma = RegisterMmaOp::new(op);
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k16);
    mma.set_attr_nvvm_register_mma_operation(&f.ctx, RegisterMmaOperationAttr::Multiply);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Standard);
    mma.set_attr_nvvm_register_mma_accumulator(&f.ctx, RegisterMmaAccumulatorAttr::F32);
    mma.set_attr_nvvm_register_mma_a_element(&f.ctx, RegisterMmaElementAttr::F16);
    mma.set_attr_nvvm_register_mma_b_element(&f.ctx, RegisterMmaElementAttr::F16);
    mma.set_attr_nvvm_register_mma_a_layout(&f.ctx, RegisterMmaLayoutAttr::Row);
    mma.set_attr_nvvm_register_mma_b_layout(&f.ctx, RegisterMmaLayoutAttr::Col);
    mma.set_attr_nvvm_register_mma_overflow(&f.ctx, RegisterMmaOverflowAttr::NotApplicable);
    f.finish();
    let text = f.translate().unwrap();
    for register in 10..16 {
        assert!(
            text.contains(&format!(
                "\"llvm.bitcast\"(%v{register}) : (i32) -> (vector<2xf16>)"
            )),
            "{text}"
        );
    }
    let issue = text
        .lines()
        .find(|line| line.contains("\"nvvm.mma.sync\""))
        .unwrap();
    let packed_values: Vec<_> = (10..16)
        .map(|register| {
            text.lines()
                .find(|line| line.contains(&format!("\"llvm.bitcast\"(%v{register})")))
                .unwrap()
                .trim()
                .split(" = ")
                .next()
                .unwrap()
        })
        .collect();
    assert!(
        issue.contains(&format!(
            "\"nvvm.mma.sync\"({}, %v6, %v7, %v8, %v9)",
            packed_values.join(", ")
        )),
        "{issue}"
    );
    for arch in ["sm_100a", "sm_120a"] {
        let text = CutlassFullCuteMlir22::new(arch)
            .unwrap()
            .translate_module(&f.ctx, &f.module)
            .and_then(|m| render_module(&m).map_err(Into::into))
            .unwrap();
        assert!(text.contains("\"nvvm.mma.sync\""), "{arch}: {text}");
    }
    assert!(
        issue.contains("operandSegmentSizes = array<i32: 4, 2, 4>"),
        "{issue}"
    );
    assert!(
        issue.contains("#nvvm.shape<m = 16, n = 8, k = 16>"),
        "{issue}"
    );
    assert!(issue.contains("layoutA = #nvvm.mma_layout<row>"), "{issue}");
    assert!(issue.contains("layoutB = #nvvm.mma_layout<col>"), "{issue}");
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);

    op.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i1039".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0106")
    );
}

/// Build the BF16 instruction with independent dtype and identity selectors.
fn bf16_mma_fixture() -> (Fixture, RegisterMmaOp) {
    let mut f = Fixture::new();
    let operands = (6..16).map(|i| f.arg(i)).collect();
    let op = f.append::<RegisterMmaOp>("v1:i0105", operands, vec![f.types[6]; 4]);
    let mma = RegisterMmaOp::new(op);
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k16);
    mma.set_attr_nvvm_register_mma_operation(&f.ctx, RegisterMmaOperationAttr::Multiply);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Standard);
    mma.set_attr_nvvm_register_mma_accumulator(&f.ctx, RegisterMmaAccumulatorAttr::F32);
    mma.set_attr_nvvm_register_mma_a_element(&f.ctx, RegisterMmaElementAttr::Bf16);
    mma.set_attr_nvvm_register_mma_b_element(&f.ctx, RegisterMmaElementAttr::Bf16);
    mma.set_attr_nvvm_register_mma_a_layout(&f.ctx, RegisterMmaLayoutAttr::Row);
    mma.set_attr_nvvm_register_mma_b_layout(&f.ctx, RegisterMmaLayoutAttr::Col);
    mma.set_attr_nvvm_register_mma_overflow(&f.ctx, RegisterMmaOverflowAttr::NotApplicable);
    f.finish();
    (f, mma)
}

/// NVVM consumes BF16 register pairs as i32, with the original low/high bits unchanged.
#[test]
fn bf16_mma_preserves_integer_packing_and_fp32_accumulator_order() {
    let (f, _) = bf16_mma_fixture();
    let text = f.translate().unwrap();
    let issue = text
        .lines()
        .find(|line| line.contains("\"nvvm.mma.sync\""))
        .unwrap();
    assert!(
        issue.contains("\"nvvm.mma.sync\"(%v10, %v11, %v12, %v13, %v14, %v15, %v6, %v7, %v8, %v9)"),
        "{issue}"
    );
    assert!(
        issue.contains("(i32, i32, i32, i32, i32, i32, f32, f32, f32, f32)"),
        "{issue}"
    );
    assert!(
        issue.contains("!llvm.struct<(f32, f32, f32, f32)>"),
        "{issue}"
    );
    assert!(
        issue.contains("operandSegmentSizes = array<i32: 4, 2, 4>"),
        "{issue}"
    );
    assert!(
        issue.contains("#nvvm.shape<m = 16, n = 8, k = 16>"),
        "{issue}"
    );
    assert!(issue.contains("layoutA = #nvvm.mma_layout<row>"), "{issue}");
    assert!(issue.contains("layoutB = #nvvm.mma_layout<col>"), "{issue}");
    assert!(
        issue.contains("multiplicandAPtxType = #nvvm.mma_type<bf16>"),
        "{issue}"
    );
    assert!(
        issue.contains("multiplicandBPtxType = #nvvm.mma_type<bf16>"),
        "{issue}"
    );
    assert!(!text.contains("\"llvm.bitcast\""), "{text}");
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);
    for arch in ["sm_100a", "sm_120a"] {
        let text = CutlassFullCuteMlir22::new(arch)
            .unwrap()
            .translate_module(&f.ctx, &f.module)
            .and_then(|m| render_module(&m).map_err(Into::into))
            .unwrap();
        assert!(
            text.contains("multiplicandAPtxType = #nvvm.mma_type<bf16>"),
            "{arch}: {text}"
        );
    }
}

/// A BF16 shape cannot impersonate the FP16 intrinsic with an otherwise identical register ABI.
#[test]
fn bf16_mma_rejects_fp16_intrinsic_identity() {
    let (f, mma) = bf16_mma_fixture();
    mma.get_operation().deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0106".into()),
    );
    let error = f.translate().unwrap_err();
    assert!(
        error.contains("expected intrinsic marker v1:i0105"),
        "{error}"
    );
}

/// Same-size FP16 and BF16 registers do not make mixed-dtype instructions valid.
#[test]
fn bf16_mma_rejects_mixed_operand_dtypes() {
    for change_a in [false, true] {
        let (f, mma) = bf16_mma_fixture();
        if change_a {
            mma.set_attr_nvvm_register_mma_a_element(&f.ctx, RegisterMmaElementAttr::F16);
        } else {
            mma.set_attr_nvvm_register_mma_b_element(&f.ctx, RegisterMmaElementAttr::F16);
        }
        assert!(f.translate().is_err());
    }
}

/// Check each reviewed matrix-load form retains its transpose flag and exact register count.
#[test]
fn matrix_loads_preserve_transposition_and_register_count() {
    for (multiplicity, layout, marker, spelling, count) in [
        (
            LdmatrixMultiplicityAttr::X2,
            LdmatrixLayoutAttr::Normal,
            "v1:i0027",
            "row",
            2,
        ),
        (
            LdmatrixMultiplicityAttr::X2,
            LdmatrixLayoutAttr::Transposed,
            "v1:i0028",
            "col",
            2,
        ),
        (
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Transposed,
            "v1:i0029",
            "col",
            4,
        ),
    ] {
        let mut f = Fixture::new();
        let address = f.arg(0);
        let load = LdmatrixOp::build(
            &mut f.ctx,
            address,
            LdmatrixShapeAttr::M8n8,
            multiplicity,
            layout,
            LdmatrixElementAttr::B16,
            LdmatrixStateSpaceAttr::Shared,
        );
        load.deref_mut(&f.ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new(marker.into()),
        );
        load.insert_at_back(f.block, &f.ctx);
        f.finish();
        let text = f.translate().unwrap();
        assert!(text.contains(&format!("num = {count} : i32")), "{text}");
        assert!(
            text.contains(&format!("#nvvm.mma_layout<{spelling}>")),
            "{text}"
        );
        assert_eq!(text.matches("\"llvm.extractvalue\"").count(), count);
        load.deref_mut(&f.ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new("v1:i0013".into()),
        );
        assert!(
            f.translate()
                .unwrap_err()
                .contains(&format!("expected intrinsic marker {marker}"))
        );
    }
}

/// Check a load/FP16-pack/store chain preserves scalar register and packed-lane ordering.
#[test]
fn sm120_load_pack_and_store_preserve_register_and_lane_order() {
    let mut f = Fixture::new();
    let load = f.load();
    let pack = f.append::<CvtF16x2F32Op>("v1:i0081", vec![f.arg(6), f.arg(7)], vec![f.types[4]]);
    let registers = load.deref(&f.ctx).results().collect::<Vec<_>>();
    let packed = pack.deref(&f.ctx).get_result(0);
    f.append::<StmatrixM8n8X2Op>("v1:i0301", vec![f.arg(1), packed, registers[3]], vec![]);
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("\"nvvm.ldmatrix\""), "{text}");
    assert!(text.contains("num = 4 : i32"), "{text}");
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);
    assert!(text.contains("\"llvm.call_intrinsic\"(%v7, %v6)"), "{text}");
    assert!(text.contains("llvm.nvvm.ff2f16x2.rn"), "{text}");
    assert!(text.contains("\"nvvm.stmatrix\""), "{text}");
    assert!(text.contains("#nvvm.mma_layout<row>"), "{text}");
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}

/// Reject matrix-load shapes, packing choices, or identities outside the reviewed subset.
#[test]
fn sm120_rejects_other_matrix_load_selectors_and_forged_identity() {
    let mut f = Fixture::new();
    // An integer shared address is valid for both tested matrix element formats.
    let address = f.arg(4);
    let load = LdmatrixOp::build(
        &mut f.ctx,
        address,
        LdmatrixShapeAttr::M8n8,
        LdmatrixMultiplicityAttr::X4,
        LdmatrixLayoutAttr::Normal,
        LdmatrixElementAttr::B16,
        LdmatrixStateSpaceAttr::Shared,
    );
    load.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0013".into()),
    );
    load.insert_at_back(f.block, &f.ctx);
    f.finish();
    LdmatrixOp::new(load).set_attr_nvvm_ldmatrix_shape(&f.ctx, LdmatrixShapeAttr::M8n16);
    LdmatrixOp::new(load)
        .set_attr_nvvm_ldmatrix_element(&f.ctx, LdmatrixElementAttr::B8x16B4x16P64);
    assert!(f.translate().unwrap_err().contains("m8n8.x2/x4.b16"));
    LdmatrixOp::new(load).set_attr_nvvm_ldmatrix_shape(&f.ctx, LdmatrixShapeAttr::M8n8);
    LdmatrixOp::new(load).set_attr_nvvm_ldmatrix_element(&f.ctx, LdmatrixElementAttr::B16);
    load.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0001".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0013")
    );
}

/// Check TMA uses the local shared completion barrier without introducing multicast.
#[test]
fn sm120_tma_uses_local_shared_barrier_and_no_multicast() {
    let mut f = Fixture::new();
    let mask = f.constant(16, 0);
    let hint = f.constant(64, 0);
    f.append::<CpAsyncBulkTensorG2sTile2dOp>(
        "v1:i0329",
        vec![f.arg(1), f.arg(2), f.arg(3), f.arg(4), f.arg(5), mask, hint],
        vec![],
    );
    f.finish();
    let text = f.translate().unwrap();
    assert!(
        text.contains("\"nvvm.cp.async.bulk.tensor.shared.cluster.global\""),
        "{text}"
    );
    assert!(
        text.contains("operandSegmentSizes = array<i32: 1, 1, 2, 1, 0, 0, 0, 0>"),
        "{text}"
    );
    assert!(text.contains("useIntrinsic = false"), "{text}");
    assert!(!text.contains("cta_2"));
}

/// Reject unsupported TMA selectors instead of silently dropping their semantics.
#[test]
fn sm120_tma_rejects_nonzero_multicast_mask_or_cache_hint() {
    for (mask, hint) in [(1, 0), (0, 1)] {
        let mut f = Fixture::new();
        let mask = f.constant(16, mask);
        let hint = f.constant(64, hint);
        f.append::<CpAsyncBulkTensorG2sTile2dOp>(
            "v1:i0329",
            vec![f.arg(1), f.arg(2), f.arg(3), f.arg(4), f.arg(5), mask, hint],
            vec![],
        );
        f.finish();
        assert!(
            f.translate()
                .unwrap_err()
                .contains("zero multicast mask and cache hint")
        );
    }
}

/// Reject SM100-only tensor-memory instructions during source architecture validation.
#[test]
fn sm120_rejects_tcgen05_before_translation() {
    let mut f = Fixture::new();
    f.append::<Tcgen05LoadWaitOp>("v1:i0357", vec![], vec![]);
    f.finish();
    let error = CutlassFullCuteMlir22::new("sm_120a")
        .unwrap()
        .translate_module(&f.ctx, &f.module)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("TCGen05 is unsupported on sm_120a"),
        "{error}"
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("TCGen05 is unsupported on sm_120a")
    );
}

/// Reject a raw integer barrier carrier where the reviewed TMA contract requires a pointer.
#[test]
fn sm120_tma_requires_pointer_barrier_carrier() {
    let mut f = Fixture::new();
    let mask = f.constant(16, 0);
    let hint = f.constant(64, 0);
    f.append::<CpAsyncBulkTensorG2sTile2dOp>(
        "v1:i0329",
        vec![f.arg(1), f.arg(4), f.arg(3), f.arg(4), f.arg(5), mask, hint],
        vec![],
    );
    f.finish();
    assert!(
        f.translate()
            .unwrap_err()
            .contains("TMA requires pointer carriers")
    );
}
