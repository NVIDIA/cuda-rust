/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::stream::position::SourcePosition;
use llvm_export::export::{DebugKind, ExportBackendConfig, FunctionLocalStaticPlacement};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::{Location, Source},
    op::Op,
};
use std::path::PathBuf;

pub(super) struct DebugConfig<C> {
    pub(super) inner: C,
    pub(super) debug_kind: DebugKind,
}

impl<C: ExportBackendConfig> ExportBackendConfig for DebugConfig<C> {
    fn datalayout(&self) -> &str {
        self.inner.datalayout()
    }

    fn emit_llvm_used(&self) -> bool {
        self.inner.emit_llvm_used()
    }

    fn emit_nvvmir_version(&self) -> bool {
        self.inner.emit_nvvmir_version()
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        self.inner.nvvmir_version()
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        self.inner.emit_all_kernel_annotations()
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        self.inner.emit_ptx_kernel_keyword()
    }

    fn nvvm_ir_dialect(&self) -> Option<llvm_export::export::NvvmIrDialect> {
        self.inner.nvvm_ir_dialect()
    }

    fn debug_kind(&self) -> DebugKind {
        self.debug_kind
    }
}

/// Selects where function-local statics are retained, delegating everything
/// else (including the debug tier) to the wrapped config.
pub(super) struct PlacementConfig<C> {
    pub(super) inner: C,
    pub(super) placement: FunctionLocalStaticPlacement,
}

impl<C: ExportBackendConfig> ExportBackendConfig for PlacementConfig<C> {
    fn datalayout(&self) -> &str {
        self.inner.datalayout()
    }

    fn emit_llvm_used(&self) -> bool {
        self.inner.emit_llvm_used()
    }

    fn emit_nvvmir_version(&self) -> bool {
        self.inner.emit_nvvmir_version()
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        self.inner.nvvmir_version()
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        self.inner.emit_all_kernel_annotations()
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        self.inner.emit_ptx_kernel_keyword()
    }

    fn nvvm_ir_dialect(&self) -> Option<llvm_export::export::NvvmIrDialect> {
        self.inner.nvvm_ir_dialect()
    }

    fn debug_kind(&self) -> DebugKind {
        self.inner.debug_kind()
    }

    fn function_local_static_placement(&self) -> FunctionLocalStaticPlacement {
        self.placement
    }
}

pub(super) fn src_location(ctx: &mut Context, file: &str, line: i32, column: i32) -> Location {
    Location::SrcPos {
        src: Source::new_from_file(ctx, PathBuf::from(file)),
        pos: SourcePosition { line, column },
    }
}

pub(super) fn module_top_block(ctx: &mut Context, module: &ModuleOp) -> Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    {
        let region = module_region.deref(ctx);
        if let Some(block) = region.iter(ctx).next() {
            return block;
        }
    }

    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(module_region, ctx);
    block
}

pub(super) fn metadata_id<'a>(ir: &'a str, needle: &str) -> &'a str {
    ir.lines()
        .find(|line| line.contains(needle))
        .and_then(|line| line.split_once(" = ").map(|(id, _)| id))
        .unwrap_or_else(|| panic!("missing metadata node containing {needle:?}:\n{ir}"))
}

/// Scans the textual LLVM IR and asserts that every `%vN` token appearing in
/// an operand position has a corresponding `%vN = ...` definition somewhere
/// in the module. Operates on `%v` temporaries only because that's the
/// exporter's naming scheme; named values like `%entry` (block labels) are
/// ignored by construction.
pub(super) fn assert_no_undefined_temporaries(ir: &str) {
    use std::collections::HashSet;

    let mut defined: HashSet<String> = HashSet::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("%v") {
            continue;
        }
        let Some((lhs, _)) = trimmed.split_once('=') else {
            continue;
        };
        defined.insert(lhs.trim().to_string());
    }

    let mut referenced: HashSet<String> = HashSet::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        // Skip the lhs of a definition; only operand positions can be stale.
        let body = if trimmed.starts_with("%v")
            && let Some(eq) = trimmed.find('=')
        {
            &trimmed[eq + 1..]
        } else {
            trimmed
        };
        for tok in body.split(|c: char| !c.is_alphanumeric() && c != '%' && c != '_') {
            if let Some(num) = tok.strip_prefix("%v")
                && !num.is_empty()
                && num.chars().all(|c| c.is_ascii_digit())
            {
                referenced.insert(format!("%v{num}"));
            }
        }
    }

    let mut undefined: Vec<&String> = referenced.difference(&defined).collect();
    undefined.sort();
    assert!(
        undefined.is_empty(),
        "IR references undefined SSA temporaries: {undefined:?}\nIR:\n{ir}"
    );
}
