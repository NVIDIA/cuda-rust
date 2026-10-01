/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structured PTX operations.
//!
//! These operations can be created either by projecting a lossless syntax
//! document or directly by a producer. Source locations intentionally live in
//! [`crate::Projection`]'s lineage table rather than in the operations, so a
//! freshly-built module does not need to invent source spans.

use crate::attributes::{CallableKindAttr, PredicateAttr, TerminatorKindAttr};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{BoolAttr, StringAttr, VecAttr},
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
            NoTerminatorInterface, OneRegionInterface, SingleBlockRegionInterface,
        },
    },
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    region::Region,
    result::Error,
    verify_err,
};
use pliron_derive::pliron_op;

/// Root of one structured PTX module.
#[pliron_op(
    name = "ptx.module",
    format,
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ]
)]
pub struct PtxModuleOp;

impl PtxModuleOp {
    pub fn build(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let region = op.deref(ctx).get_region(0);
        BasicBlock::new(ctx, None, vec![]).insert_at_back(region, ctx);
        Self { op }
    }

    pub fn body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_head()
            .expect("ptx.module always has a body block")
    }
}

impl Verify for PtxModuleOp {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// One PTX directive at module, callable, or lexical-scope level.
#[pliron_op(
    name = "ptx.directive",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (
        directive_labels: VecAttr,
        directive_name: StringAttr,
        directive_arguments: StringAttr
    )
)]
pub struct PtxDirectiveOp;

impl PtxDirectiveOp {
    pub fn build(ctx: &mut Context, name: &str, arguments: &str) -> Self {
        Self::build_labeled(ctx, std::iter::empty::<&str>(), name, arguments)
    }

    pub fn build_labeled<'label>(
        ctx: &mut Context,
        labels: impl IntoIterator<Item = &'label str>,
        name: &str,
        arguments: &str,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_directive_labels(
            ctx,
            VecAttr::new(
                labels
                    .into_iter()
                    .map(|label| StringAttr::new(label.to_string()).into())
                    .collect(),
            ),
        );
        wrapped.set_attr_directive_name(ctx, StringAttr::new(name.to_string()));
        wrapped.set_attr_directive_arguments(ctx, StringAttr::new(arguments.to_string()));
        wrapped
    }

    pub fn labels(&self, ctx: &Context) -> Vec<String> {
        self.get_attr_directive_labels(ctx)
            .expect("verified ptx.directive has labels")
            .0
            .iter()
            .map(|label| {
                label
                    .downcast_ref::<StringAttr>()
                    .expect("verified PTX directive labels are strings")
                    .as_str()
                    .to_string()
            })
            .collect()
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_directive_name(ctx)
            .expect("verified ptx.directive has a name")
            .as_str()
            .to_string()
    }

    pub fn arguments(&self, ctx: &Context) -> String {
        self.get_attr_directive_arguments(ctx)
            .expect("verified ptx.directive has arguments")
            .as_str()
            .to_string()
    }
}

/// Declaration point for an indexed-branch target table in native CFG form.
///
/// Target labels are intentionally not stored here. They are derived from the
/// successors of every [`PtxTerminatorOp`] which names this table, making CFG
/// edges the sole authority after raising.
#[pliron_op(
    name = "ptx.branch_targets",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (branch_targets_name: StringAttr)
)]
pub struct PtxBranchTargetsOp;

impl PtxBranchTargetsOp {
    pub fn build(ctx: &mut Context, name: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_branch_targets_name(ctx, StringAttr::new(name.to_string()));
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_branch_targets_name(ctx)
            .expect("verified ptx.branch_targets has a name")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxBranchTargetsOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(name) = self.get_attr_branch_targets_name(ctx) else {
            return verify_err!(operation.loc(), "ptx.branch_targets requires a name");
        };
        if name.as_str().is_empty() {
            return verify_err!(
                operation.loc(),
                "PTX branch-target table name must not be empty"
            );
        }
        Ok(())
    }
}

impl Verify for PtxDirectiveOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(labels) = self.get_attr_directive_labels(ctx) else {
            return verify_err!(operation.loc(), "ptx.directive requires labels");
        };
        if labels.0.iter().any(|label| {
            label
                .downcast_ref::<StringAttr>()
                .is_none_or(|label| label.as_str().is_empty())
        }) {
            return verify_err!(
                operation.loc(),
                "PTX directive labels must be non-empty strings"
            );
        }
        let Some(name) = self.get_attr_directive_name(ctx) else {
            return verify_err!(operation.loc(), "ptx.directive requires a name");
        };
        if !name.as_str().starts_with('.') {
            return verify_err!(operation.loc(), "PTX directive name must start with '.'");
        }
        if self.get_attr_directive_arguments(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.directive requires arguments");
        }
        Ok(())
    }
}

/// One PTX statement label. Labels remain explicit operations until control
/// flow recovery resolves them to Pliron basic blocks.
#[pliron_op(
    name = "ptx.label",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (label_name: StringAttr)
)]
pub struct PtxLabelOp;

impl PtxLabelOp {
    pub fn build(ctx: &mut Context, name: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_label_name(ctx, StringAttr::new(name.to_string()));
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_label_name(ctx)
            .expect("verified ptx.label has a name")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxLabelOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(name) = self.get_attr_label_name(ctx) else {
            return verify_err!(operation.loc(), "ptx.label requires a name");
        };
        if name.as_str().is_empty() {
            return verify_err!(operation.loc(), "PTX label name must not be empty");
        }
        Ok(())
    }
}

/// A declaration or definition of a PTX `.entry` or `.func`.
///
/// Declarations have no regions. Definitions own a stable callable identity
/// containing exactly one surface or CFG body-form operation. `header` is the complete spelling before the
/// declaration semicolon or definition opening brace; consumers can gradually
/// replace its generic pieces with typed attributes without losing syntax.
///
/// # Header/attribute contract
///
/// The typed attributes (`callable_name`, `callable_kind`,
/// `callable_external`) are the queryable truth; `callable_header` is the
/// print form and still carries syntax the dialect does not model yet
/// (parameter lists, performance directives such as `.maxntid`). The emitter
/// prints only the header, so [`Verify`] re-parses the header with
/// [`ptx_parse`] and rejects the operation whenever the header disagrees with
/// the typed attributes. Mutating one side without the other can therefore
/// never silently desync: it is a verification error, and emission verifies
/// first.
#[pliron_op(
    name = "ptx.callable",
    format,
    interfaces = [
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (
        callable_name: StringAttr,
        callable_kind: CallableKindAttr,
        callable_external: BoolAttr,
        callable_header: StringAttr
    )
)]
pub struct PtxCallableOp;

impl PtxCallableOp {
    pub fn build_declaration(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
    ) -> Self {
        Self::build(ctx, name, kind, is_extern, header, None)
    }

    pub fn build_definition(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
    ) -> Self {
        Self::build(
            ctx,
            name,
            kind,
            is_extern,
            header,
            Some(CallableBodyKind::Surface),
        )
    }

    pub fn build_cfg_definition(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
    ) -> Self {
        Self::build(
            ctx,
            name,
            kind,
            is_extern,
            header,
            Some(CallableBodyKind::Cfg),
        )
    }

    fn build(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
        body_kind: Option<CallableBodyKind>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            usize::from(body_kind.is_some()),
        );
        let wrapped = Self { op };
        wrapped.set_attr_callable_name(ctx, StringAttr::new(name.to_string()));
        wrapped.set_attr_callable_kind(ctx, kind);
        wrapped.set_attr_callable_external(ctx, BoolAttr::new(is_extern));
        wrapped.set_attr_callable_header(ctx, StringAttr::new(header.to_string()));
        if let Some(body_kind) = body_kind {
            let region = op.deref(ctx).get_region(0);
            let container = BasicBlock::new(ctx, None, vec![]);
            container.insert_at_back(region, ctx);
            let body = match body_kind {
                CallableBodyKind::Surface => PtxSurfaceBodyOp::build(ctx).get_operation(),
                CallableBodyKind::Cfg => PtxCfgBodyOp::build(ctx).get_operation(),
            };
            body.insert_at_back(container, ctx);
        }
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_callable_name(ctx)
            .expect("verified ptx.callable has a name")
            .as_str()
            .to_string()
    }

    pub fn kind(&self, ctx: &Context) -> CallableKindAttr {
        *self
            .get_attr_callable_kind(ctx)
            .expect("verified ptx.callable has a kind")
    }

    pub fn is_external(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_callable_external(ctx)
                .expect("verified ptx.callable has an external flag")
                .clone(),
        )
    }

    pub fn header(&self, ctx: &Context) -> String {
        self.get_attr_callable_header(ctx)
            .expect("verified ptx.callable has a header")
            .as_str()
            .to_string()
    }

    pub fn region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        (self.get_operation().deref(ctx).num_regions() == 1)
            .then(|| self.get_operation().deref(ctx).get_region(0))
    }

    pub fn entry_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        self.surface_body(ctx).map(|body| body.body(ctx))
    }

    pub fn is_definition(&self, ctx: &Context) -> bool {
        self.region(ctx).is_some()
    }

    pub fn body_operation(&self, ctx: &Context) -> Option<Ptr<Operation>> {
        self.region(ctx)?
            .deref(ctx)
            .get_entry_block()?
            .deref(ctx)
            .iter(ctx)
            .next()
    }

    pub fn surface_body(&self, ctx: &Context) -> Option<PtxSurfaceBodyOp> {
        Operation::get_op(self.body_operation(ctx)?, ctx)
    }

    pub fn cfg_body(&self, ctx: &Context) -> Option<PtxCfgBodyOp> {
        Operation::get_op(self.body_operation(ctx)?, ctx)
    }
}

impl Verify for PtxCallableOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_callable_name(ctx).is_none()
            || self.get_attr_callable_kind(ctx).is_none()
            || self.get_attr_callable_external(ctx).is_none()
            || self.get_attr_callable_header(ctx).is_none()
        {
            return verify_err!(
                operation.loc(),
                "ptx.callable requires name, kind, external flag, and header"
            );
        }
        // The header is the print carrier; the typed attributes are the
        // queryable truth. Re-parse the header as a declaration and require
        // agreement so neither side can silently desync from the other.
        let header = self.header(ctx);
        let declaration = format!("{};\n", header.trim());
        let document = match ptx_parse::Document::parse(&declaration) {
            Ok(document) => document,
            Err(error) => {
                return verify_err!(
                    operation.loc(),
                    "ptx.callable header {header:?} does not parse as PTX: {error}"
                );
            }
        };
        let [parsed] = document.callables() else {
            return verify_err!(
                operation.loc(),
                "ptx.callable header {header:?} does not spell exactly one PTX callable"
            );
        };
        if parsed.name() != self.name(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header names {:?} but callable_name is {:?}",
                parsed.name(),
                self.name(ctx)
            );
        }
        if CallableKindAttr::from(parsed.kind()) != self.kind(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header spells {:?} but callable_kind is {:?}",
                CallableKindAttr::from(parsed.kind()),
                self.kind(ctx)
            );
        }
        if parsed.is_extern() != self.is_external(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header spells external = {} but callable_external is {}",
                parsed.is_extern(),
                self.is_external(ctx)
            );
        }
        if operation.num_regions() > 1 {
            return verify_err!(
                operation.loc(),
                "ptx.callable supports at most one body region"
            );
        }
        if let Some(region) = self.region(ctx) {
            let Some(container) = region.deref(ctx).get_entry_block() else {
                return verify_err!(
                    operation.loc(),
                    "PTX callable definition requires a body container"
                );
            };
            let bodies: Vec<_> = container.deref(ctx).iter(ctx).collect();
            if bodies.len() != 1
                || (Operation::get_op::<PtxSurfaceBodyOp>(bodies[0], ctx).is_none()
                    && Operation::get_op::<PtxCfgBodyOp>(bodies[0], ctx).is_none())
            {
                return verify_err!(
                    operation.loc(),
                    "PTX callable definition requires exactly one surface or CFG body"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum CallableBodyKind {
    Surface,
    Cfg,
}

/// Lossless/canonical lexical body form of one [`PtxCallableOp`].
#[pliron_op(
    name = "ptx.surface_body",
    format,
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ]
)]
pub struct PtxSurfaceBodyOp;

impl PtxSurfaceBodyOp {
    pub fn build(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let body = op.deref(ctx).get_region(0);
        BasicBlock::new(ctx, None, vec![]).insert_at_back(body, ctx);
        Self { op }
    }

    pub fn body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .expect("ptx.surface_body always has one block")
    }
}

impl Verify for PtxSurfaceBodyOp {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// Native multi-block CFG body form of one [`PtxCallableOp`].
#[pliron_op(
    name = "ptx.cfg_body",
    format,
    interfaces = [NRegionsInterface<1>, OneRegionInterface, NOpdsInterface<0>, NResultsInterface<0>]
)]
pub struct PtxCfgBodyOp;

impl PtxCfgBodyOp {
    pub fn build(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        Self { op }
    }

    pub fn region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn append_block(&self, ctx: &mut Context) -> Ptr<BasicBlock> {
        let region = self.region(ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        block
    }
}

impl Verify for PtxCfgBodyOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.region(ctx).deref(ctx).get_entry_block().is_none() {
            return verify_err!(
                operation.loc(),
                "ptx.cfg_body requires at least one native CFG block"
            );
        }
        verify_cfg_layout(ctx, &operation, self.region(ctx))?;
        Ok(())
    }
}

/// An anonymous or header-owned lexical PTX scope.
///
/// # Header/attribute contract
///
/// `scope_header` is the print form emitted before the opening brace, and is
/// empty for anonymous scopes. PTX callables are the one brace-headed
/// construct this dialect models with typed attributes, so [`Verify`]
/// re-parses a non-empty header with [`ptx_parse`] and rejects callable
/// headers here: routing a callable through `ptx.scope` would bypass
/// `ptx.callable`'s queryable name/kind/external attributes.
#[pliron_op(
    name = "ptx.scope",
    format,
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (scope_header: StringAttr)
)]
pub struct PtxScopeOp;

impl PtxScopeOp {
    pub fn build(ctx: &mut Context, header: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let wrapped = Self { op };
        wrapped.set_attr_scope_header(ctx, StringAttr::new(header.to_string()));
        let region = op.deref(ctx).get_region(0);
        BasicBlock::new(ctx, None, vec![]).insert_at_back(region, ctx);
        wrapped
    }

    pub fn header(&self, ctx: &Context) -> String {
        self.get_attr_scope_header(ctx)
            .expect("verified ptx.scope has a header")
            .as_str()
            .to_string()
    }

    pub fn body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .expect("ptx.scope always has a body block")
    }
}

impl Verify for PtxScopeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_scope_header(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.scope requires a header attribute");
        }
        let header = self.header(ctx);
        let header = header.trim();
        if header.is_empty() {
            return Ok(());
        }
        if header.ends_with(';') || header.ends_with('{') {
            return verify_err!(
                operation.loc(),
                "ptx.scope header {header:?} must not carry its own terminator; \
                 the emitter prints the brace"
            );
        }
        // A callable header smuggled into a scope would print as a valid
        // definition while bypassing ptx.callable's typed attributes.
        let declaration = format!("{header};\n");
        if let Ok(document) = ptx_parse::Document::parse(&declaration)
            && let [parsed] = document.callables()
        {
            return verify_err!(
                operation.loc(),
                "ptx.scope header spells the PTX callable {:?}; use ptx.callable so the \
                 name, kind, and external flag stay queryable",
                parsed.name()
            );
        }
        Ok(())
    }
}

/// One structurally discovered or directly constructed PTX instruction.
///
/// Predication is the only instruction prefix PTX defines, so the guard is a
/// typed, optional [`PredicateAttr`] rather than free-form prefix text. The
/// emitter derives the `@%p` / `@!%p` spelling from it.
#[pliron_op(
    name = "ptx.instruction",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (
        instruction_predicate: PredicateAttr,
        instruction_head: StringAttr,
        instruction_operands: VecAttr
    )
)]
pub struct PtxInstructionOp;

impl PtxInstructionOp {
    pub fn build<'operand>(
        ctx: &mut Context,
        predicate: Option<PredicateAttr>,
        head: &str,
        operands: impl IntoIterator<Item = &'operand str>,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        if let Some(predicate) = predicate {
            wrapped.set_attr_instruction_predicate(ctx, predicate);
        }
        wrapped.set_attr_instruction_head(ctx, StringAttr::new(head.to_string()));
        wrapped.set_attr_instruction_operands(
            ctx,
            VecAttr::new(
                operands
                    .into_iter()
                    .map(|operand| StringAttr::new(operand.to_string()).into())
                    .collect(),
            ),
        );
        wrapped
    }

    pub fn predicate(&self, ctx: &Context) -> Option<PredicateAttr> {
        self.get_attr_instruction_predicate(ctx)
            .map(|predicate| predicate.clone())
    }

    pub fn head(&self, ctx: &Context) -> String {
        self.get_attr_instruction_head(ctx)
            .expect("verified ptx.instruction has a head")
            .as_str()
            .to_string()
    }

    pub fn operands(&self, ctx: &Context) -> Vec<String> {
        self.get_attr_instruction_operands(ctx)
            .expect("verified ptx.instruction has operands")
            .0
            .iter()
            .map(|operand| {
                operand
                    .downcast_ref::<StringAttr>()
                    .expect("verified PTX operands are strings")
                    .as_str()
                    .to_string()
            })
            .collect()
    }
}

impl Verify for PtxInstructionOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(head) = self.get_attr_instruction_head(ctx) else {
            return verify_err!(operation.loc(), "ptx.instruction requires a head");
        };
        if head.as_str().is_empty() {
            return verify_err!(operation.loc(), "PTX instruction head must not be empty");
        }
        let Some(operands) = self.get_attr_instruction_operands(ctx) else {
            return verify_err!(operation.loc(), "ptx.instruction requires operands");
        };
        if operands
            .0
            .iter()
            .any(|operand| operand.downcast_ref::<StringAttr>().is_none())
        {
            return verify_err!(operation.loc(), "PTX instruction operands must be strings");
        }
        Ok(())
    }
}

/// A PTX instruction which terminates one native Pliron basic block.
///
/// Source PTX has implicit fallthrough edges. `has_fallthrough` makes that
/// relation explicit: when present, successor zero is the fallthrough block
/// and all remaining successors are textual branch targets. A synthetic
/// `Fallthrough` terminator emits no instruction.
///
/// Like [`PtxInstructionOp`], the guard is a typed, optional
/// [`PredicateAttr`] rather than free-form prefix text; the emitter derives
/// the `@%p` / `@!%p` spelling from it.
#[pliron_op(
    name = "ptx.terminator",
    format,
    interfaces = [
        NOpdsInterface<0>,
        NResultsInterface<0>,
        IsTerminatorInterface
    ],
    attributes = (
        terminator_kind: TerminatorKindAttr,
        terminator_predicate: PredicateAttr,
        terminator_head: StringAttr,
        terminator_operands: VecAttr,
        terminator_target_table: StringAttr,
        terminator_has_fallthrough: BoolAttr
    )
)]
pub struct PtxTerminatorOp;

pub struct PtxTerminatorSpec<'source> {
    pub kind: TerminatorKindAttr,
    pub predicate: Option<PredicateAttr>,
    pub head: &'source str,
    pub operands: Vec<&'source str>,
    pub target_table: &'source str,
    pub has_fallthrough: bool,
}

impl PtxTerminatorOp {
    pub fn build(
        ctx: &mut Context,
        syntax: PtxTerminatorSpec<'_>,
        successors: impl IntoIterator<Item = Ptr<BasicBlock>>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            successors.into_iter().collect(),
            0,
        );
        let wrapped = Self { op };
        wrapped.set_attr_terminator_kind(ctx, syntax.kind);
        if let Some(predicate) = syntax.predicate {
            wrapped.set_attr_terminator_predicate(ctx, predicate);
        }
        wrapped.set_attr_terminator_head(ctx, StringAttr::new(syntax.head.to_string()));
        wrapped.set_attr_terminator_operands(
            ctx,
            VecAttr::new(
                syntax
                    .operands
                    .into_iter()
                    .map(|operand| StringAttr::new(operand.to_string()).into())
                    .collect(),
            ),
        );
        wrapped.set_attr_terminator_target_table(
            ctx,
            StringAttr::new(syntax.target_table.to_string()),
        );
        wrapped.set_attr_terminator_has_fallthrough(ctx, BoolAttr::new(syntax.has_fallthrough));
        wrapped
    }

    pub fn fallthrough(ctx: &mut Context, target: Ptr<BasicBlock>) -> Self {
        Self::build(
            ctx,
            PtxTerminatorSpec {
                kind: TerminatorKindAttr::Fallthrough,
                predicate: None,
                head: "",
                operands: Vec::new(),
                target_table: "",
                has_fallthrough: false,
            },
            [target],
        )
    }

    pub fn kind(&self, ctx: &Context) -> TerminatorKindAttr {
        *self
            .get_attr_terminator_kind(ctx)
            .expect("verified ptx.terminator has a kind")
    }

    pub fn predicate(&self, ctx: &Context) -> Option<PredicateAttr> {
        self.get_attr_terminator_predicate(ctx)
            .map(|predicate| predicate.clone())
    }

    pub fn head(&self, ctx: &Context) -> String {
        self.get_attr_terminator_head(ctx)
            .expect("verified ptx.terminator has a head")
            .as_str()
            .to_string()
    }

    pub fn operands(&self, ctx: &Context) -> Vec<String> {
        self.get_attr_terminator_operands(ctx)
            .expect("verified ptx.terminator has operands")
            .0
            .iter()
            .map(|operand| {
                operand
                    .downcast_ref::<StringAttr>()
                    .expect("verified PTX terminator operands are strings")
                    .as_str()
                    .to_string()
            })
            .collect()
    }

    pub fn has_fallthrough(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_terminator_has_fallthrough(ctx)
                .expect("verified ptx.terminator has a fallthrough flag")
                .clone(),
        )
    }

    pub fn target_table(&self, ctx: &Context) -> String {
        self.get_attr_terminator_target_table(ctx)
            .expect("verified ptx.terminator has a target table")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxTerminatorOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(kind) = self
            .get_attr_terminator_kind(ctx)
            .map(|attribute| *attribute)
        else {
            return verify_err!(operation.loc(), "ptx.terminator requires a kind");
        };
        let Some(head) = self.get_attr_terminator_head(ctx) else {
            return verify_err!(operation.loc(), "ptx.terminator requires a head");
        };
        let Some(operands) = self.get_attr_terminator_operands(ctx) else {
            return verify_err!(operation.loc(), "ptx.terminator requires operands");
        };
        let Some(target_table) = self.get_attr_terminator_target_table(ctx) else {
            return verify_err!(operation.loc(), "ptx.terminator requires a target table");
        };
        if operands
            .0
            .iter()
            .any(|operand| operand.downcast_ref::<StringAttr>().is_none())
        {
            return verify_err!(operation.loc(), "PTX terminator operands must be strings");
        }
        let Some(has_fallthrough) = self
            .get_attr_terminator_has_fallthrough(ctx)
            .map(|attribute| bool::from(attribute.clone()))
        else {
            return verify_err!(
                operation.loc(),
                "ptx.terminator requires a fallthrough flag"
            );
        };
        let successor_count = operation.get_num_successors();
        let head_parts = head.as_str().split('.').collect::<Vec<_>>();
        let head_matches = match kind {
            TerminatorKindAttr::Fallthrough => head.as_str().is_empty(),
            TerminatorKindAttr::Branch => head_parts.first() == Some(&"bra"),
            TerminatorKindAttr::IndexedBranch => {
                head_parts.first() == Some(&"brx") && head_parts.get(1) == Some(&"idx")
            }
            TerminatorKindAttr::Return => head_parts.first() == Some(&"ret"),
            TerminatorKindAttr::ThreadExit => head_parts.first() == Some(&"exit"),
            TerminatorKindAttr::Trap => head_parts.first() == Some(&"trap"),
        };
        if !head_matches {
            return verify_err!(
                operation.loc(),
                "PTX terminator kind does not match its instruction head"
            );
        }
        let is_predicated = self.get_attr_terminator_predicate(ctx).is_some();
        match kind {
            TerminatorKindAttr::Fallthrough => {
                if has_fallthrough
                    || is_predicated
                    || !operands.0.is_empty()
                    || !target_table.as_str().is_empty()
                    || successor_count != 1
                {
                    return verify_err!(
                        operation.loc(),
                        "synthetic PTX fallthrough requires no text and exactly one successor"
                    );
                }
            }
            TerminatorKindAttr::Branch => {
                if has_fallthrough != is_predicated
                    || !operands.0.is_empty()
                    || !target_table.as_str().is_empty()
                    || successor_count != 1 + usize::from(has_fallthrough)
                {
                    return verify_err!(
                        operation.loc(),
                        "PTX branch requires one target and an optional predicated fallthrough"
                    );
                }
            }
            TerminatorKindAttr::IndexedBranch => {
                if has_fallthrough != is_predicated
                    || operands.0.is_empty()
                    || target_table.as_str().is_empty()
                    || successor_count <= usize::from(has_fallthrough)
                {
                    return verify_err!(
                        operation.loc(),
                        "PTX indexed branch requires targets and an optional predicated fallthrough"
                    );
                }
            }
            TerminatorKindAttr::Return
            | TerminatorKindAttr::ThreadExit
            | TerminatorKindAttr::Trap => {
                if has_fallthrough != is_predicated
                    || !operands.0.is_empty()
                    || !target_table.as_str().is_empty()
                    || successor_count != usize::from(has_fallthrough)
                {
                    return verify_err!(
                        operation.loc(),
                        "PTX exit terminator permits only a predicated fallthrough successor"
                    );
                }
            }
        }
        Ok(())
    }
}

fn verify_cfg_layout(
    ctx: &Context,
    callable: &Operation,
    region: Ptr<Region>,
) -> Result<(), Error> {
    use std::collections::HashMap;

    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let block_indices: HashMap<_, _> = blocks
        .iter()
        .copied()
        .enumerate()
        .map(|(index, block)| (block, index))
        .collect();
    let mut primary_labels = HashMap::new();
    let mut tables = HashMap::new();
    for (block_index, block) in blocks.iter().copied().enumerate() {
        for (operation_index, operation) in block.deref(ctx).iter(ctx).enumerate() {
            if let Some(label) = Operation::get_op::<PtxLabelOp>(operation, ctx) {
                primary_labels
                    .entry(block)
                    .or_insert_with(|| label.name(ctx));
            }
            if let Some(table) = Operation::get_op::<PtxBranchTargetsOp>(operation, ctx)
                && tables
                    .insert(table.name(ctx), (block_index, operation_index))
                    .is_some()
            {
                return verify_err!(
                    callable.loc(),
                    "PTX native CFG defines an indexed-branch table more than once"
                );
            }
        }
    }

    let mut table_users: HashMap<String, Vec<Ptr<BasicBlock>>> = HashMap::new();
    for (block_index, block) in blocks.iter().copied().enumerate() {
        let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let Some(operation) = operations.last().copied() else {
            return verify_err!(callable.loc(), "PTX native CFG block must not be empty");
        };
        let Some(terminator) = Operation::get_op::<PtxTerminatorOp>(operation, ctx) else {
            return verify_err!(
                callable.loc(),
                "PTX native CFG block must end in ptx.terminator"
            );
        };
        let fallthrough = terminator.has_fallthrough(ctx)
            || terminator.kind(ctx) == TerminatorKindAttr::Fallthrough;
        if fallthrough {
            let expected = blocks.get(block_index + 1).copied();
            if expected != Some(operation.deref(ctx).get_successor(0)) {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX fallthrough successor must be the next emitted block"
                );
            }
        }
        let first_target = usize::from(fallthrough);
        let targets: Vec<_> = operation
            .deref(ctx)
            .successors()
            .skip(first_target)
            .collect();
        for target in &targets {
            if !block_indices.contains_key(target) {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX branch successor must belong to the same callable"
                );
            }
            if !primary_labels.contains_key(target) {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX branch successor requires a source label"
                );
            }
        }
        if terminator.kind(ctx) == TerminatorKindAttr::IndexedBranch {
            let table = terminator.target_table(ctx);
            let Some(&(table_block, table_operation)) = tables.get(&table) else {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX indexed branch names an undeclared target table"
                );
            };
            let terminator_operation = operations.len() - 1;
            if table_block > block_index
                || (table_block == block_index && table_operation >= terminator_operation)
            {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX indexed-branch table must be emitted before its use"
                );
            }
            if let Some(previous) = table_users.insert(table, targets.clone())
                && previous != targets
            {
                return verify_err!(
                    operation.deref(ctx).loc(),
                    "PTX indexed-branch table users must have identical successors"
                );
            }
        }
    }
    if tables.keys().any(|table| !table_users.contains_key(table)) {
        return verify_err!(
            callable.loc(),
            "PTX native CFG target table must be used by an indexed branch"
        );
    }
    Ok(())
}

/// A structurally retained statement for syntax not yet modeled by this dialect.
#[pliron_op(
    name = "ptx.raw",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (raw_text: StringAttr)
)]
pub struct PtxRawOp;

impl PtxRawOp {
    pub fn build(ctx: &mut Context, text: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_raw_text(ctx, StringAttr::new(text.to_string()));
        wrapped
    }

    pub fn text(&self, ctx: &Context) -> String {
        self.get_attr_raw_text(ctx)
            .expect("verified ptx.raw has text")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxRawOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_raw_text(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.raw requires text");
        }
        Ok(())
    }
}

pub fn register(ctx: &mut Context) {
    PtxModuleOp::register(ctx);
    PtxDirectiveOp::register(ctx);
    PtxBranchTargetsOp::register(ctx);
    PtxLabelOp::register(ctx);
    PtxCallableOp::register(ctx);
    PtxSurfaceBodyOp::register(ctx);
    PtxCfgBodyOp::register(ctx);
    PtxScopeOp::register(ctx);
    PtxInstructionOp::register(ctx);
    PtxTerminatorOp::register(ctx);
    PtxRawOp::register(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::attributes::StringAttr;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    #[test]
    fn callable_with_agreeing_header_and_attributes_verifies() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".visible .entry kernel(.param .u64 p0)",
        );
        callable.verify(&ctx).unwrap();
    }

    #[test]
    fn callable_name_mutated_without_header_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".visible .entry kernel()",
        );
        callable.set_attr_callable_name(&ctx, StringAttr::new("renamed".to_string()));
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header names \"kernel\" but callable_name is \"renamed\""),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_kind_desync_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "helper",
            CallableKindAttr::Entry,
            true,
            ".extern .func helper(.param .b32 x)",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header spells Function but callable_kind is Entry"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_external_desync_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "helper",
            CallableKindAttr::Function,
            false,
            ".extern .func helper(.param .b32 x)",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header spells external = true but callable_external is false"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_header_that_is_not_a_callable_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".pragma \"not a callable\"",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("does not spell exactly one PTX callable"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn anonymous_and_plain_scope_headers_verify() {
        let mut ctx = test_context();
        PtxScopeOp::build(&mut ctx, "").verify(&ctx).unwrap();
    }

    #[test]
    fn scope_smuggling_a_callable_header_fails_verification() {
        let mut ctx = test_context();
        let scope = PtxScopeOp::build(&mut ctx, ".visible .entry kernel()");
        let error = scope.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("spells the PTX callable \"kernel\""),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unpredicated_instruction_verifies_without_a_predicate_attribute() {
        let mut ctx = test_context();
        let instruction = PtxInstructionOp::build(&mut ctx, None, "ret", []);
        instruction.verify(&ctx).unwrap();
        assert_eq!(instruction.predicate(&ctx), None);
    }

    #[test]
    fn predicate_register_must_be_percent_prefixed() {
        let mut ctx = test_context();
        let instruction = PtxInstructionOp::build(
            &mut ctx,
            Some(PredicateAttr::new("p1", false)),
            "bra",
            ["L0"],
        );
        let error = instruction
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must be a %-prefixed register name"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn scope_header_carrying_its_own_brace_fails_verification() {
        let mut ctx = test_context();
        let scope = PtxScopeOp::build(&mut ctx, ".pragma \"x\" {");
        let error = scope.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("must not carry its own terminator"),
            "unexpected error: {error}"
        );
    }
}
