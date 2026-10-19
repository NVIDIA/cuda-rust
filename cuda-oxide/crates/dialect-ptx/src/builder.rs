/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ergonomic construction of structured PTX operations.

use crate::attributes::{CallableKindAttr, PredicateAttr};
use crate::emitter::{EmitError, emit_canonical_module};
use crate::ops::{
    PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxModuleOp, PtxRawOp, PtxScopeOp,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;

/// Builder for a complete structured PTX module.
pub struct PtxBuilder<'ctx> {
    ctx: &'ctx mut Context,
    module: PtxModuleOp,
}

impl<'ctx> PtxBuilder<'ctx> {
    pub fn new(ctx: &'ctx mut Context) -> Self {
        Self {
            module: PtxModuleOp::build(ctx),
            ctx,
        }
    }

    pub fn version(&mut self, version: &str) -> &mut Self {
        self.directive(".version", version)
    }

    pub fn target(&mut self, target: &str) -> &mut Self {
        self.directive(".target", target)
    }

    pub fn address_size(&mut self, bits: u8) -> &mut Self {
        self.directive(".address_size", &bits.to_string())
    }

    pub fn directive(&mut self, name: &str, arguments: &str) -> &mut Self {
        PtxDirectiveOp::build(self.ctx, name, arguments)
            .get_operation()
            .insert_at_back(self.module.body(self.ctx), self.ctx);
        self
    }

    pub fn raw(&mut self, text: &str) -> &mut Self {
        PtxRawOp::build(self.ctx, text)
            .get_operation()
            .insert_at_back(self.module.body(self.ctx), self.ctx);
        self
    }

    pub fn callable_declaration(
        &mut self,
        name: &str,
        kind: CallableKindAttr,
        is_external: bool,
        header: &str,
    ) -> PtxCallableOp {
        let callable = PtxCallableOp::build_declaration(self.ctx, name, kind, is_external, header);
        callable
            .get_operation()
            .insert_at_back(self.module.body(self.ctx), self.ctx);
        callable
    }

    pub fn callable_definition(
        &mut self,
        name: &str,
        kind: CallableKindAttr,
        is_external: bool,
        header: &str,
        build_body: impl FnOnce(&mut PtxBodyBuilder<'_>),
    ) -> PtxCallableOp {
        let callable = PtxCallableOp::build_definition(self.ctx, name, kind, is_external, header);
        let body = callable
            .entry_block(self.ctx)
            .expect("a definition has an entry block");
        callable
            .get_operation()
            .insert_at_back(self.module.body(self.ctx), self.ctx);
        build_body(&mut PtxBodyBuilder {
            ctx: self.ctx,
            block: body,
        });
        callable
    }

    /// Convenience for the common public kernel form.
    pub fn visible_entry(
        &mut self,
        name: &str,
        parameters: &str,
        build_body: impl FnOnce(&mut PtxBodyBuilder<'_>),
    ) -> PtxCallableOp {
        let header = format!(".visible .entry {name}{parameters}");
        self.callable_definition(name, CallableKindAttr::Entry, false, &header, build_body)
    }

    pub fn module(&self) -> PtxModuleOp {
        self.module
    }

    pub fn finish(self) -> PtxModuleOp {
        self.module
    }

    pub fn emit(self) -> Result<String, EmitError> {
        emit_canonical_module(self.ctx, &self.module)
    }
}

/// Builder for one callable or lexical-scope body.
pub struct PtxBodyBuilder<'builder> {
    ctx: &'builder mut Context,
    block: Ptr<BasicBlock>,
}

impl PtxBodyBuilder<'_> {
    pub fn label(&mut self, name: &str) -> &mut Self {
        PtxLabelOp::build(self.ctx, name)
            .get_operation()
            .insert_at_back(self.block, self.ctx);
        self
    }

    pub fn directive(&mut self, name: &str, arguments: &str) -> &mut Self {
        PtxDirectiveOp::build(self.ctx, name, arguments)
            .get_operation()
            .insert_at_back(self.block, self.ctx);
        self
    }

    pub fn instruction<I, S>(&mut self, head: &str, operands: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.build_instruction(None, head, operands)
    }

    /// An instruction guarded by a predicate register, e.g. `@!%p1 bra L0;`.
    pub fn predicated_instruction<I, S>(
        &mut self,
        predicate: PredicateAttr,
        head: &str,
        operands: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.build_instruction(Some(predicate), head, operands)
    }

    fn build_instruction<I, S>(
        &mut self,
        predicate: Option<PredicateAttr>,
        head: &str,
        operands: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operands: Vec<String> = operands
            .into_iter()
            .map(|operand| operand.as_ref().to_string())
            .collect();
        PtxInstructionOp::build(
            self.ctx,
            predicate,
            head,
            operands.iter().map(String::as_str),
        )
        .get_operation()
        .insert_at_back(self.block, self.ctx);
        self
    }

    pub fn raw(&mut self, text: &str) -> &mut Self {
        PtxRawOp::build(self.ctx, text)
            .get_operation()
            .insert_at_back(self.block, self.ctx);
        self
    }

    pub fn scope(
        &mut self,
        header: &str,
        build_body: impl FnOnce(&mut PtxBodyBuilder<'_>),
    ) -> &mut Self {
        let scope = PtxScopeOp::build(self.ctx, header);
        let body = scope.body(self.ctx);
        scope.get_operation().insert_at_back(self.block, self.ctx);
        build_body(&mut PtxBodyBuilder {
            ctx: self.ctx,
            block: body,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_constructs_the_dialect_and_emits_ptx() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let mut builder = PtxBuilder::new(&mut ctx);
        builder.version("8.9").target("sm_120a").address_size(64);
        builder.visible_entry("kernel", "()", |body| {
            body.directive(".reg", ".b32 %r<2>;");
            body.label("L0");
            body.instruction("mov.u32", ["%r0", "7"]);
            body.instruction("ret", std::iter::empty::<&str>());
        });
        let emitted = builder.emit().unwrap();
        assert_eq!(
            emitted,
            "\
.version 8.9
.target sm_120a
.address_size 64
.visible .entry kernel()
{
    .reg .b32 %r<2>;
    L0:
    mov.u32 %r0, 7;
    ret;
}
"
        );
        assert!(
            ptx_parse::Document::parse(&emitted)
                .unwrap()
                .coverage()
                .is_complete()
        );
    }
}
