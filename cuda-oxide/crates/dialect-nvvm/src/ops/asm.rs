/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! User-authored inline PTX operations.

use dialect_mir::{attributes::MirPointerKindAuthorityAttr, types::pointer_carriers_in_type};
use pliron::{
    builtin::attributes::{BoolAttr, StringAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

/// User-authored inline PTX.
///
/// This operation is produced by the MIR importer for `cuda_device::ptx_asm!`
/// marker calls and lowered to LLVM inline assembly.
///
/// Supports zero or more results for multi-output PTX instructions
/// (e.g. MMA, vectorized loads). A result type that directly or recursively
/// carries a MIR pointer must carry `InlineAsm` pointer-kind authority. The
/// MIR importer attaches that authority from rustc's destination type; the
/// PTX text alone is never trusted to invent a pointer category.
#[pliron_op(
    name = "nvvm.inline_ptx",
    format,
    attributes = (
        ptx_template: StringAttr,
        ptx_constraints: StringAttr,
        ptx_sideeffect: BoolAttr,
        ptx_convergent: BoolAttr,
        inline_ptx_pointer_kind_authority: MirPointerKindAuthorityAttr
    )
)]
pub struct InlinePtxOp;

impl InlinePtxOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        InlinePtxOp { op }
    }

    /// Build an inline PTX operation with zero or more results.
    pub fn build(
        ctx: &mut Context,
        result_tys: Vec<TypeHandle>,
        inputs: Vec<Value>,
        template: &str,
        constraints: &str,
        sideeffect: bool,
        convergent: bool,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_tys,
            inputs,
            vec![],
            0,
        );
        let wrapped = InlinePtxOp { op };
        wrapped.set_attr_ptx_template(ctx, StringAttr::new(template.to_string()));
        wrapped.set_attr_ptx_constraints(ctx, StringAttr::new(constraints.to_string()));
        wrapped.set_attr_ptx_sideeffect(ctx, BoolAttr::new(sideeffect));
        wrapped.set_attr_ptx_convergent(ctx, BoolAttr::new(convergent));
        wrapped.get_operation()
    }

    /// Mark pointer-carrying results as originating at an inline-assembly
    /// output boundary whose exact destination type came from rustc MIR.
    pub fn set_pointer_kind_authority(
        &self,
        ctx: &mut Context,
        authority: MirPointerKindAuthorityAttr,
    ) {
        self.set_attr_inline_ptx_pointer_kind_authority(ctx, authority);
    }

    /// Count the output constraints in an LLVM-style constraint string.
    ///
    /// Output constraints are the comma-separated tokens prefixed with `=`
    /// (e.g. `=r`, `=f`). Each op result binds to exactly one output
    /// constraint, in order. This is the canonical counting rule; both the
    /// op verifier and the MIR importer's `ptx_asm!` translation use it so
    /// they can never disagree on how many results an inline PTX block
    /// produces.
    pub fn count_output_constraints(constraints: &str) -> usize {
        constraints
            .split(',')
            .filter(|constraint| constraint.starts_with('='))
            .count()
    }
}

impl Verify for InlinePtxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if self.get_attr_ptx_template(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.inline_ptx requires ptx_template attribute");
        }
        let Some(constraints) = self
            .get_attr_ptx_constraints(ctx)
            .map(|attr| String::from((*attr).clone()))
        else {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_constraints attribute"
            );
        };
        if self.get_attr_ptx_sideeffect(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_sideeffect attribute"
            );
        }
        if self.get_attr_ptx_convergent(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_convergent attribute"
            );
        }
        // Each result binds to exactly one `=`-prefixed output constraint,
        // in order; a mismatch would mis-bind PTX registers and is only
        // diagnosed much later (and worse) by llc.
        let num_outputs = Self::count_output_constraints(&constraints);
        let num_results = op.get_num_results();
        if num_results != num_outputs {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx has {num_results} results but {num_outputs} `=` output constraints"
            );
        }

        let has_pointer_result = (0..num_results).any(|index| {
            let result_ty = op.get_result(index).get_type(ctx);
            !pointer_carriers_in_type(ctx, result_ty).is_empty()
        });
        let authority = self.get_attr_inline_ptx_pointer_kind_authority(ctx);
        match (has_pointer_result, authority) {
            (true, Some(authority)) if *authority == MirPointerKindAuthorityAttr::InlineAsm => {}
            (true, Some(authority)) => {
                return verify_err!(
                    op.loc(),
                    "nvvm.inline_ptx pointer-carrying results require InlineAsm pointer-kind authority, found {:?}",
                    authority
                );
            }
            (true, None) => {
                return verify_err!(
                    op.loc(),
                    "nvvm.inline_ptx pointer-carrying results require InlineAsm pointer-kind authority"
                );
            }
            (false, Some(authority)) => {
                return verify_err!(
                    op.loc(),
                    "nvvm.inline_ptx pointer-kind authority {:?} is spurious for pointer-free results",
                    authority
                );
            }
            (false, None) => {}
        }
        Ok(())
    }
}

/// Register inline PTX operations with the context.
pub(super) fn register(ctx: &mut Context) {
    InlinePtxOp::register(ctx);
}
