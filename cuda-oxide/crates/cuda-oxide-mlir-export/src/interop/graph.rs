/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build a real target SSA graph before invoking upstream generic printers.

use std::{collections::HashMap, fmt, rc::Rc};

use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    context::{Context, Ptr},
    derive::{op_interface_impl, pliron_op},
    ident, input_err_noloc,
    location::Location,
    op::Op,
    operation::Operation,
    printable::State,
    value::Value,
};
use pliron_mlir_interop::{
    MlirPrinter, MlirPrinterT, ToMlirOp,
    printers::{GenericOp, print_op, set_block_argument_locations},
};

use super::tokens::{
    dictionary_key, materialize_attribute, materialize_location, materialize_type, printer_error,
};
use crate::{MlirBlockId, MlirModule, MlirOperation, MlirValueId, TranslationError};

struct OperationData {
    mnemonic: String,
    properties: Vec<(String, AttrObj)>,
    attributes: Vec<(String, AttrObj)>,
}

type OperationDataMap = HashMap<Ptr<Operation>, Rc<OperationData>>;

#[pliron_op(name = "cudaoxide_export.target", format, verifier = "succ")]
struct TargetOp;

#[op_interface_impl]
impl ToMlirOp for TargetOp {
    fn to_mlir(
        &self,
        ctx: &Context,
        state: &State,
        f: &mut fmt::Formatter<'_>,
    ) -> pliron::result::Result<()> {
        // Release the state borrow before GenericOp changes indentation while
        // recursively printing regions. The metadata itself is immutable.
        let metadata = state
            .aux_data_ref()
            .get(&ident!("cudaoxide_export_operations"))
            .and_then(|data| data.downcast_ref::<OperationDataMap>())
            .and_then(|data| data.get(&self.get_operation()).cloned());
        let Some(metadata) = metadata else {
            return input_err_noloc!("target operation has no MLIR mapping metadata");
        };
        let mut generic = GenericOp::new(self.get_operation(), &metadata.mnemonic);
        for (key, value) in &metadata.properties {
            generic = generic.prop_attr(key, &**value);
        }
        for (key, value) in &metadata.attributes {
            generic = generic.attr_attr(key, &**value);
        }
        generic.print(ctx, state, f)
    }
}

struct TargetPrinter {
    root: Ptr<Operation>,
    operations: OperationDataMap,
    argument_locations: Vec<(Value, Location)>,
}

impl MlirPrinterT for TargetPrinter {
    fn mlir_print(
        &self,
        ctx: &Context,
        state: &State,
        f: &mut fmt::Formatter<'_>,
    ) -> pliron::result::Result<()> {
        state.aux_data_mut().insert(
            ident!("cudaoxide_export_operations"),
            Box::new(self.operations.clone()),
        );
        set_block_argument_locations(state, self.argument_locations.iter().cloned());
        print_op(ctx, self.root, state, f)
    }
}

#[derive(Default)]
struct GraphBuilder<'a> {
    values: HashMap<MlirValueId, Value>,
    blocks: HashMap<MlirBlockId, Ptr<BasicBlock>>,
    pending: Vec<(&'a MlirOperation, Ptr<Operation>)>,
    operations: OperationDataMap,
    argument_locations: Vec<(Value, Location)>,
}

impl<'a> GraphBuilder<'a> {
    fn allocate(
        &mut self,
        ctx: &mut Context,
        plan: &'a MlirOperation,
    ) -> Result<Ptr<Operation>, TranslationError> {
        MlirOperation::new(plan.name.clone()).map_err(TranslationError::InvalidTarget)?;
        let types = plan
            .results
            .iter()
            .map(|result| materialize_type(ctx, &result.ty))
            .collect::<Result<_, _>>()?;
        let operation = Operation::new(
            ctx,
            TargetOp::get_concrete_op_info(),
            types,
            vec![],
            vec![],
            plan.regions.len(),
        );
        operation
            .deref_mut(ctx)
            .set_loc(materialize_location(ctx, &plan.location)?);
        for (result, value) in plan.results.iter().zip(operation.deref(ctx).results()) {
            self.values.insert(result.id, value);
        }
        let properties = plan
            .properties
            .iter()
            .map(|(name, attr)| Ok((dictionary_key(name), materialize_attribute(ctx, attr)?)))
            .collect::<Result<_, TranslationError>>()?;
        let attributes = plan
            .attributes
            .iter()
            .map(|(name, attr)| Ok((dictionary_key(name), materialize_attribute(ctx, attr)?)))
            .collect::<Result<_, TranslationError>>()?;
        self.operations.insert(
            operation,
            Rc::new(OperationData {
                mnemonic: plan.name.clone(),
                properties,
                attributes,
            }),
        );
        self.pending.push((plan, operation));
        for (index, region_plan) in plan.regions.iter().enumerate() {
            let region = operation.deref(ctx).get_region(index);
            for block_plan in &region_plan.blocks {
                let types = block_plan
                    .arguments
                    .iter()
                    .map(|arg| materialize_type(ctx, &arg.ty))
                    .collect::<Result<_, _>>()?;
                let block = BasicBlock::new(ctx, None, types);
                block.insert_at_back(region, ctx);
                self.blocks.insert(block_plan.id, block);
                for (argument, value) in block_plan
                    .arguments
                    .iter()
                    .zip(block.deref(ctx).arguments())
                {
                    self.values.insert(argument.id, value);
                    self.argument_locations
                        .push((value, materialize_location(ctx, &argument.location)?));
                }
                for child_plan in &block_plan.operations {
                    let child = self.allocate(ctx, child_plan)?;
                    child.insert_at_back(block, ctx);
                }
            }
        }
        Ok(operation)
    }

    fn wire(&self, ctx: &Context) -> Result<(), TranslationError> {
        // All definitions exist first, including forward CFG destinations and
        // temporaries introduced by one-to-many mappings.
        for (plan, operation) in &self.pending {
            for operand in &plan.operands {
                let value = self.values.get(&operand.id).ok_or_else(|| {
                    TranslationError::InvalidTarget(format!(
                        "operand {:?} has no definition",
                        operand.id
                    ))
                })?;
                Operation::push_operand(*operation, ctx, *value);
            }
            for successor in &plan.successors {
                let block = self.blocks.get(successor).ok_or_else(|| {
                    TranslationError::InvalidTarget(format!(
                        "successor {successor:?} has no block definition"
                    ))
                })?;
                Operation::push_successor(*operation, ctx, *block);
            }
        }
        Ok(())
    }
}

pub(super) fn render(module: &MlirModule) -> Result<String, TranslationError> {
    let ctx = &mut Context::new();
    let mut graph = GraphBuilder::default();
    let root = graph.allocate(ctx, &module.root)?;
    graph.wire(ctx)?;
    let printer = TargetPrinter {
        root,
        operations: graph.operations,
        argument_locations: graph.argument_locations,
    };
    let mut text = String::try_from(MlirPrinter::new(ctx, &printer))
        .map_err(|error| printer_error(ctx, error))?;
    text.push('\n');
    Ok(text)
}
