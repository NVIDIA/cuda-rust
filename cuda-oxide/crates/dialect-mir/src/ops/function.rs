/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR function operations.
//!
//! This module defines the function operation for the MIR dialect.

use combine::{Parser, optional, token};
use once_cell::sync::Lazy;
use pliron::{
    attribute::AttributeDict,
    attribute::attr_cast,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{StringAttr, TypeAttr},
        op_interfaces::{
            ATTR_KEY_SYM_NAME, IsolatedFromAboveInterface, NOpdsInterface, NRegionsInterface,
            NResultsInterface, OneRegionInterface, SymbolOpInterface,
        },
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    indented_block, input_err,
    irfmt::{
        parsers::{spaced, type_parser},
        printers::op::{region, typed_symb_op_header},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{Printable, State, indented_nl},
    region::Region,
    result::Error,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast},
    verify_err,
};
use pliron_derive::pliron_op;

use crate::{
    attributes::ReferenceParamValidityAttr,
    types::{MirPtrType, MirSliceType},
};

const REFERENCE_PARAM_VALIDITY_ATTR_PREFIX: &str = "reference_param_validity_";

fn reference_param_validity_key(index: usize) -> Identifier {
    Identifier::try_new(format!("{REFERENCE_PARAM_VALIDITY_ATTR_PREFIX}{index}"))
        .expect("reference parameter validity attribute name is valid")
}

/// MIR function operation.
///
/// Represents a function in MIR. Contains a single region with basic blocks.
///
/// # Attributes
///
/// ```text
/// | Name           | Type      | Description                        |
/// |----------------|-----------|------------------------------------|
/// | `sym_name`     | StringAttr| Function name (from SymbolOpInterface) |
/// | `mir_func_type`| TypeAttr  | Function type (mir.func_type)      |
/// | `reference_param_validity_N` | ReferenceParamValidityAttr | Proven nonnull/alignment for source argument `N` on a kernel entry |
/// ```
///
/// # Verification
///
/// - Must have a `mir_func_type` attribute that implements `FunctionTypeInterface`.
/// - The entry block arguments must match the function input types.
#[pliron_op(
    name = "mir.func",
    interfaces = [
        SymbolOpInterface,
        IsolatedFromAboveInterface,
        NRegionsInterface<1>,
        OneRegionInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (mir_func_type: TypeAttr)
)]
pub struct MirFuncOp;

impl MirFuncOp {
    /// Create a new MirFuncOp.
    pub fn new(ctx: &mut Context, op_ptr: Ptr<Operation>, func_type_attr: TypeAttr) -> Self {
        let op = MirFuncOp { op: op_ptr };
        op.set_attr_mir_func_type(ctx, func_type_attr);
        op
    }

    /// Create a MirFuncOp from an existing operation pointer.
    ///
    /// Returns `None` if the operation is not a `mir.func`.
    pub fn wrap(ctx: &Context, op: Ptr<Operation>) -> Option<Self> {
        if Operation::get_opid(op, ctx) == Self::get_opid_static() {
            Some(MirFuncOp { op })
        } else {
            None
        }
    }

    /// Get the function type.
    pub fn get_type(&self, ctx: &Context) -> TypedHandle<FunctionType> {
        let ty = attr_cast::<dyn TypedAttrInterface>(&*self.get_attr_mir_func_type(ctx).unwrap())
            .unwrap()
            .get_type(ctx);
        TypedHandle::from_handle(ty, ctx).unwrap()
    }

    /// Record rustc-proven validity for one source-level kernel argument.
    ///
    /// Presence proves `nonnull`; the payload carries the rustc ABI alignment
    /// of the pointee. The source argument index is deliberately retained until
    /// kernel ABI lowering decides which physical LLVM parameter represents it.
    pub fn set_reference_param_validity(
        &self,
        ctx: &mut Context,
        index: usize,
        validity: ReferenceParamValidityAttr,
    ) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(reference_param_validity_key(index), validity);
    }

    /// Return the rustc-proven validity fact for one source argument, if any.
    pub fn reference_param_validity(
        &self,
        ctx: &Context,
        index: usize,
    ) -> Option<ReferenceParamValidityAttr> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<ReferenceParamValidityAttr>(&reference_param_validity_key(index))
            .copied()
    }
}

impl Typed for MirFuncOp {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        self.get_type(ctx).into()
    }
}

impl Printable for MirFuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        typed_symb_op_header(self).fmt(ctx, state, f)?;
        let mut attributes_to_print_separately = self
            .get_operation()
            .deref(ctx)
            .attributes
            .clone_skip_outlined(ctx);

        attributes_to_print_separately
            .0
            .retain(|key, _| key != &*ATTR_KEY_MIR_FUNC_TYPE && key != &ATTR_KEY_SYM_NAME);

        if !attributes_to_print_separately.0.is_empty() {
            indented_block!(state, {
                write!(f, "{}", indented_nl(state))?;
                attributes_to_print_separately.fmt(ctx, state, f)?;
            });
        }
        write!(f, " ")?;
        region(self).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for MirFuncOp {
    type Arg = Vec<(Identifier, pliron::location::Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                pliron::builtin::op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let mut parser = (
            spaced(token('@').with(Identifier::parser(()))).skip(spaced(token(':'))),
            spaced(type_parser()),
            spaced(AttributeDict::parser(())),
            spaced(optional(Region::parser(op))),
        );
        parser
            .parse_stream(state_stream)
            .map(|(fname, fty, attrs, _region)| -> OpObj {
                let ctx = &mut state_stream.state.ctx;
                op.deref_mut(ctx).attributes = attrs;
                let ty_attr = TypeAttr::new(fty);
                let opop = MirFuncOp { op };
                opop.set_symbol_name(ctx, fname);
                opop.set_attr_mir_func_type(ctx, ty_attr);
                OpObj::new(opop)
            })
            .into()
    }
}

impl Verify for MirFuncOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Verify function type attribute
        let func_ty = self.get_type(ctx);
        let func_ty_ref = func_ty.deref(ctx);

        // Check inputs via interface
        let interface = match type_cast::<dyn FunctionTypeInterface>(&*func_ty_ref) {
            Some(i) => i,
            None => {
                return verify_err!(
                    op.loc(),
                    "FunctionType does not implement FunctionTypeInterface"
                );
            }
        };

        // Reference validity is an importer-produced kernel-entry proof.
        // Verify only structural consistency here; semantic facts such as
        // non-nullness and alignment are never re-derived downstream.
        let kernel_key: Identifier = "gpu_kernel".try_into().unwrap();
        let is_kernel = op.attributes.get::<StringAttr>(&kernel_key).is_some();
        let inputs = interface.arg_types();

        for (key, _) in &op.attributes.0 {
            let key_text = key.to_string();
            let Some(index_text) = key_text.strip_prefix(REFERENCE_PARAM_VALIDITY_ATTR_PREFIX)
            else {
                continue;
            };
            let Ok(index) = index_text.parse::<usize>() else {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity attribute `{}` has an invalid source argument index",
                    key_text
                );
            };
            if !is_kernel {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity fact on argument {} is only valid on a kernel entry",
                    index
                );
            }
            if index >= inputs.len() {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity argument index {} is out of range for {} inputs",
                    index,
                    inputs.len()
                );
            }
            let Some(validity) = op.attributes.get::<ReferenceParamValidityAttr>(key) else {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity attribute `{}` has the wrong attribute type",
                    key_text
                );
            };
            if validity.0 == 0 || !validity.0.is_power_of_two() {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity alignment on argument {} must be a non-zero power of two, found {}",
                    index,
                    validity.0
                );
            }

            let input_ref = inputs[index].deref(ctx);
            let is_reference = input_ref
                .downcast_ref::<MirPtrType>()
                .is_some_and(|pointer| pointer.pointer_kind().is_reference())
                || input_ref
                    .downcast_ref::<MirSliceType>()
                    .is_some_and(|slice| slice.pointer_kind().is_reference());
            if !is_reference {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference validity fact on argument {} requires a rustc-proven Rust reference type",
                    index
                );
            }
        }

        // Verify region arguments match function type inputs
        let region = op.get_region(0).deref(ctx);

        // Check if there is an entry block
        if let Some(entry_block_ptr) = region.get_head() {
            let entry_block = entry_block_ptr.deref(ctx);
            if entry_block.get_num_arguments() != inputs.len() {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp entry block argument count must match function type inputs"
                );
            }

            for (i, arg) in entry_block.arguments().enumerate() {
                if arg.get_type(ctx) != inputs[i] {
                    return verify_err!(
                        op.loc(),
                        "MirFuncOp entry block argument {} type mismatch",
                        i
                    );
                }
            }
        }

        Ok(())
    }
}

/// Attribute key for the MIR function type.
pub static ATTR_KEY_MIR_FUNC_TYPE: Lazy<Identifier> =
    Lazy::new(|| "mir_func_type".try_into().unwrap());

/// Register function operations into the given context.
pub fn register(ctx: &mut Context) {
    MirFuncOp::register(ctx);
}
