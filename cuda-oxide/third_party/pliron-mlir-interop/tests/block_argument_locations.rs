/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod common;

use pliron::{
    builtin::{
        ops::FuncOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::Context,
    location::Location,
    op::Op,
};
use pliron_llvm::ops::ReturnOp;

#[test]
fn block_location_is_preserved_on_arguments() {
    let ctx = &mut Context::new();
    let i32 = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let function_type = FunctionType::get(ctx, vec![i32], vec![]);
    let function = FuncOp::new(ctx, "locations".try_into().unwrap(), function_type);
    let region = function.get_operation().deref(ctx).get_region(0);
    let block = region.deref(ctx).get_entry_block().unwrap();
    block.deref_mut(ctx).set_loc(Location::Named {
        name: "argument origin".into(),
        child_loc: Box::new(Location::Unknown),
    });
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(block, ctx);
    let text = common::to_mlir(ctx, &function.get_operation());
    assert!(text.contains(": i32 loc(\"argument origin\")"), "{text}");
    common::verify_with_mlir_opt(&text);
}

#[test]
fn individual_argument_locations_override_the_block() {
    use pliron::{context::Ptr, operation::Operation, printable::State};
    use pliron_mlir_interop::{
        MlirPrinterT,
        printers::{print_op, set_block_argument_locations},
    };
    struct WithLocations(Ptr<Operation>);
    impl MlirPrinterT for WithLocations {
        fn mlir_print(
            &self,
            ctx: &Context,
            state: &State,
            f: &mut std::fmt::Formatter<'_>,
        ) -> pliron::result::Result<()> {
            let region = self.0.deref(ctx).get_region(0);
            let block = region.deref(ctx).get_entry_block().unwrap();
            set_block_argument_locations(
                state,
                [
                    (
                        block.deref(ctx).get_argument(0),
                        Location::Named {
                            name: "first".into(),
                            child_loc: Box::new(Location::Unknown),
                        },
                    ),
                    (block.deref(ctx).get_argument(1), Location::Unknown),
                ],
            );
            print_op(ctx, self.0, state, f)
        }
    }
    let ctx = &mut Context::new();
    let i32 = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let function_type = FunctionType::get(ctx, vec![i32, i32], vec![]);
    let function = FuncOp::new(ctx, "locations".try_into().unwrap(), function_type);
    let region = function.get_operation().deref(ctx).get_region(0);
    let block = region.deref(ctx).get_entry_block().unwrap();
    block.deref_mut(ctx).set_loc(Location::Named {
        name: "fallback".into(),
        child_loc: Box::new(Location::Unknown),
    });
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(block, ctx);
    let text = common::to_mlir(ctx, &WithLocations(function.get_operation()));
    assert!(text.contains(": i32 loc(\"first\"), %v1: i32)"), "{text}");
    assert!(!text.contains("fallback"), "{text}");
    common::verify_with_mlir_opt(&text);
}
