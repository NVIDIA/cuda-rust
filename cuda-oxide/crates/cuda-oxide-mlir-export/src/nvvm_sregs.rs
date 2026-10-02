/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA thread and grid coordinates in MLIR's NVVM dialect.
//!
//! These operations already have a one-to-one target operation. The pack is
//! intentionally narrow: another NVVM operation needs its own reviewed recipe
//! instead of being renamed by a catch-all rule.

use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregNctaidXOp,
    ReadPtxSregNctaidYOp, ReadPtxSregNctaidZOp, ReadPtxSregNtidXOp, ReadPtxSregNtidYOp,
    ReadPtxSregNtidZOp, ReadPtxSregTidXOp, ReadPtxSregTidYOp, ReadPtxSregTidZOp,
};
use pliron::{context::Context, context::Ptr, operation::Operation};
use pliron_mlir_export::{
    MlirOperation, OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};

/// Register the zero-operand thread, block, and grid dimension reads.
pub fn register_nvvm_sreg_pack(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_operation::<ReadPtxSregTidXOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.tid.x",
        marker: "v1:i0001",
    })?;
    registry.register_operation::<ReadPtxSregTidYOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.tid.y",
        marker: "v1:i0005",
    })?;
    registry.register_operation::<ReadPtxSregTidZOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.tid.z",
        marker: "v1:i0009",
    })?;
    registry.register_operation::<ReadPtxSregNtidXOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ntid.x",
        marker: "v1:i0003",
    })?;
    registry.register_operation::<ReadPtxSregNtidYOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ntid.y",
        marker: "v1:i0007",
    })?;
    registry.register_operation::<ReadPtxSregNtidZOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ntid.z",
        marker: "v1:i0011",
    })?;
    registry.register_operation::<ReadPtxSregCtaidXOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ctaid.x",
        marker: "v1:i0002",
    })?;
    registry.register_operation::<ReadPtxSregCtaidYOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ctaid.y",
        marker: "v1:i0006",
    })?;
    registry.register_operation::<ReadPtxSregCtaidZOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.ctaid.z",
        marker: "v1:i0010",
    })?;
    registry.register_operation::<ReadPtxSregNctaidXOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.nctaid.x",
        marker: "v1:i0004",
    })?;
    registry.register_operation::<ReadPtxSregNctaidYOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.nctaid.y",
        marker: "v1:i0008",
    })?;
    registry.register_operation::<ReadPtxSregNctaidZOp>(SregTranslation {
        target: "nvvm.read.ptx.sreg.nctaid.z",
        marker: "v1:i0012",
    })?;
    Ok(())
}

struct SregTranslation {
    target: &'static str,
    marker: &'static str,
}

impl OperationTranslation for SregTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let marker = input
            .attributes
            .remove("cuda_oxide_intrinsic_marker")
            .ok_or_else(|| {
                format!(
                    "{} has no cuda_oxide_intrinsic_marker; expected {}",
                    self.target, self.marker
                )
            })?;
        if marker != pliron_mlir_export::MlirAttribute::String(self.marker.into()) {
            return Err(format!(
                "{} expected intrinsic marker {}, got {marker:?}",
                self.target, self.marker
            ));
        }
        if !input.operands.is_empty()
            || input.results.len() != 1
            || !input.successors.is_empty()
            || !input.regions.is_empty()
            || !input.attributes.is_empty()
        {
            return Err(format!(
                "{} expected no operands, one result, no successors, no regions, and no attributes",
                self.target
            ));
        }

        let mut target = MlirOperation::new(self.target)?;
        target.results = input.results;
        target.location = input.location;
        Ok(vec![target])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CutlassFullCuteMlir22, MlirConsumerProfile,
        profile::render_mapping_module_without_cutlass_envelope,
    };
    use dialect_mir::ops::{MirFuncOp, MirReturnOp};
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        identifier::Identifier,
        op::Op,
        r#type::TypedHandle,
    };

    fn append_sreg<O: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        result_type: TypedHandle<IntegerType>,
        marker: &str,
    ) -> Ptr<Operation> {
        let operation = Operation::new(
            ctx,
            O::get_concrete_op_info(),
            vec![result_type.into()],
            vec![],
            vec![],
            0,
        );
        operation.deref_mut(ctx).attributes.set(
            Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
            StringAttr::new(marker.into()),
        );
        operation.insert_at_back(block, ctx);
        operation
    }

    #[test]
    fn thread_block_and_grid_dimensions_map_one_to_one() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, Identifier::try_from("sregs").unwrap());
        let function_type = FunctionType::get(&ctx, vec![], vec![]);
        let function_operation = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(
            &mut ctx,
            function_operation,
            TypeAttr::new(function_type.into()),
        );
        function.set_symbol_name(&mut ctx, Identifier::try_from("coordinates").unwrap());
        module.append_operation(&mut ctx, function_operation, 0);

        let region = function_operation.deref(&ctx).get_region(0);
        let entry = BasicBlock::new(&mut ctx, None, vec![]);
        entry.insert_at_back(region, &ctx);
        let i32_type = IntegerType::get(&ctx, 32, Signedness::Unsigned);

        let tid_x = append_sreg::<ReadPtxSregTidXOp>(&mut ctx, entry, i32_type, "v1:i0001");
        append_sreg::<ReadPtxSregTidYOp>(&mut ctx, entry, i32_type, "v1:i0005");
        append_sreg::<ReadPtxSregTidZOp>(&mut ctx, entry, i32_type, "v1:i0009");
        append_sreg::<ReadPtxSregNtidXOp>(&mut ctx, entry, i32_type, "v1:i0003");
        append_sreg::<ReadPtxSregNtidYOp>(&mut ctx, entry, i32_type, "v1:i0007");
        append_sreg::<ReadPtxSregNtidZOp>(&mut ctx, entry, i32_type, "v1:i0011");
        append_sreg::<ReadPtxSregCtaidXOp>(&mut ctx, entry, i32_type, "v1:i0002");
        append_sreg::<ReadPtxSregCtaidYOp>(&mut ctx, entry, i32_type, "v1:i0006");
        append_sreg::<ReadPtxSregCtaidZOp>(&mut ctx, entry, i32_type, "v1:i0010");
        append_sreg::<ReadPtxSregNctaidXOp>(&mut ctx, entry, i32_type, "v1:i0004");
        append_sreg::<ReadPtxSregNctaidYOp>(&mut ctx, entry, i32_type, "v1:i0008");
        append_sreg::<ReadPtxSregNctaidZOp>(&mut ctx, entry, i32_type, "v1:i0012");

        let return_operation = Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_operation.insert_at_back(entry, &ctx);

        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let target = profile.translate_module(&ctx, &module).unwrap();
        let text = render_mapping_module_without_cutlass_envelope(&target, "sregs");
        let expected = r#""builtin.module"() <{sym_name = "sregs"}> ({
  ^bb0:
    "func.func"() <{function_type = () -> (), sym_name = "coordinates"}> ({
      ^bb1:
        %v0 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32
        %v1 = "nvvm.read.ptx.sreg.tid.y"() : () -> i32
        %v2 = "nvvm.read.ptx.sreg.tid.z"() : () -> i32
        %v3 = "nvvm.read.ptx.sreg.ntid.x"() : () -> i32
        %v4 = "nvvm.read.ptx.sreg.ntid.y"() : () -> i32
        %v5 = "nvvm.read.ptx.sreg.ntid.z"() : () -> i32
        %v6 = "nvvm.read.ptx.sreg.ctaid.x"() : () -> i32
        %v7 = "nvvm.read.ptx.sreg.ctaid.y"() : () -> i32
        %v8 = "nvvm.read.ptx.sreg.ctaid.z"() : () -> i32
        %v9 = "nvvm.read.ptx.sreg.nctaid.x"() : () -> i32
        %v10 = "nvvm.read.ptx.sreg.nctaid.y"() : () -> i32
        %v11 = "nvvm.read.ptx.sreg.nctaid.z"() : () -> i32
        "func.return"() : () -> ()
    }) : () -> ()
}) : () -> ()
"#;
        assert_eq!(text, expected);

        tid_x.deref_mut(&ctx).attributes.set(
            Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
            StringAttr::new("v1:i0002".into()),
        );
        let error = profile
            .translate_module(&ctx, &module)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("expected intrinsic marker v1:i0001"),
            "{error}"
        );
    }
}
