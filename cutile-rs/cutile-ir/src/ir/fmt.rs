/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MLIR-like text formatter for the Tile IR.
//!
//! Produces human-readable output that matches the cuda-tile C++ printer:
//! - Types printed without `!cuda_tile.` prefix (shorthand form).
//! - Per-op assembly format dispatch matching `Ops.td` declarations.
//! - Entry functions printed with named block arguments.
//!
//! The output is for debugging, `print_ir`, `dump_mlir_dir`, error messages,
//! and test assertions that check for operation name substrings.

use std::fmt::Write;

use super::attr::Attribute;
use super::module::{Global, Module};
use super::types::*;
use super::value::{BlockId, OpId, RegionId, Value};
use crate::bytecode::Opcode;

impl Module {
    /// Produce a nicely formatted MLIR-like text representation.
    pub fn to_mlir_text(&self) -> String {
        let mut out = String::new();
        let mut printer = ModulePrinter::new(self);
        printer.print_module(&mut out);
        out
    }
}

impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_mlir_text())
    }
}

struct ModulePrinter<'a> {
    module: &'a Module,
    indent: usize,
}

impl<'a> ModulePrinter<'a> {
    fn new(module: &'a Module) -> Self {
        Self { module, indent: 0 }
    }

    fn print_module(&mut self, out: &mut String) {
        writeln!(out, "cuda_tile.module @{} {{", self.module.name).unwrap();
        self.indent += 2;

        for global in &self.module.globals {
            self.print_global(global, out);
        }

        for &func_id in &self.module.functions {
            self.print_function(func_id, out);
        }

        self.indent -= 2;
        writeln!(out, "}}").unwrap();
    }

    fn print_global(&self, global: &Global, out: &mut String) {
        write!(out, "{}", " ".repeat(self.indent)).unwrap();
        writeln!(
            out,
            "cuda_tile.global @{} alignment = {} : {}",
            global.sym_name,
            global.alignment,
            format_type(&global.value.element_type)
        )
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // Entry function — `entry @name(%arg0: type, ...) { ... }`
    // -----------------------------------------------------------------------

    fn print_function(&mut self, func_id: OpId, out: &mut String) {
        let op = self.module.op(func_id);

        let name = find_str_attr(&op.attributes, "sym_name").unwrap_or("?");

        write!(out, "{}", " ".repeat(self.indent)).unwrap();
        write!(out, "entry @{name}").unwrap();

        // Print function signature using block args for names.
        if let Some(&region_id) = op.regions.first() {
            let region = self.module.region(region_id);
            if let Some(&block_id) = region.blocks.first() {
                let block = self.module.block(block_id);
                write!(out, "(").unwrap();
                for (j, (val, ty)) in block.args.iter().enumerate() {
                    if j > 0 {
                        write!(out, ", ").unwrap();
                    }
                    write!(out, "%{}: {}", val.index(), format_type(ty)).unwrap();
                }
                write!(out, ")").unwrap();
            }
        }

        // Print non-signature attributes (optimization_hints, etc.)
        for (name, attr) in &op.attributes {
            if name == "sym_name" || name == "function_type" {
                continue;
            }
            write!(out, " {name} = {}", format_attr(attr)).unwrap();
        }

        writeln!(out, " {{").unwrap();

        self.indent += 2;
        if let Some(&region_id) = op.regions.first() {
            self.print_region_body(region_id, true, out);
        }
        self.indent -= 2;

        write!(out, "{}", " ".repeat(self.indent)).unwrap();
        writeln!(out, "}}").unwrap();
    }

    // -----------------------------------------------------------------------
    // Region / block printing
    // -----------------------------------------------------------------------

    /// Print the body of a region. If `skip_entry_args` is true the first
    /// block's argument header is suppressed (for entry/for/loop/if bodies
    /// where args are printed inline).
    fn print_region_body(&mut self, region_id: RegionId, skip_entry_args: bool, out: &mut String) {
        let region = self.module.region(region_id);
        for (i, &block_id) in region.blocks.iter().enumerate() {
            let block = self.module.block(block_id);
            let show_label = if i == 0 {
                !skip_entry_args && !block.args.is_empty()
            } else {
                true
            };
            if show_label {
                write!(out, "{}^bb{i}", " ".repeat(self.indent.saturating_sub(2))).unwrap();
                if !block.args.is_empty() {
                    write!(out, "(").unwrap();
                    for (j, (val, ty)) in block.args.iter().enumerate() {
                        if j > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "%{}: {}", val.index(), format_type(ty)).unwrap();
                    }
                    write!(out, ")").unwrap();
                }
                writeln!(out, ":").unwrap();
            }
            self.print_block_ops(block_id, out);
        }
    }

    fn print_block_ops(&mut self, block_id: BlockId, out: &mut String) {
        let ops: Vec<OpId> = self.module.block(block_id).ops.clone();
        for op_id in ops {
            self.print_operation(op_id, out);
        }
    }

    // -----------------------------------------------------------------------
    // Operation printing — per-opcode dispatch
    // -----------------------------------------------------------------------

    fn print_operation(&mut self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        let pad = " ".repeat(self.indent);
        let name = opcode_name(op.opcode);

        // Collect results for the LHS of the assignment.
        let results = find_result_values(self.module, op_id);

        // Write result prefix.
        write!(out, "{pad}").unwrap();
        if !results.is_empty() {
            for (i, v) in results.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "%{}", v.index()).unwrap();
            }
            write!(out, " = ").unwrap();
        }

        // Write opcode name.
        write!(out, "{name}").unwrap();

        // Dispatch per opcode.
        let opcode = op.opcode;
        match opcode {
            // ---- Pattern A: Simple unary `$source attr-dict : type($result)` ----
            Opcode::AbsF
            | Opcode::AbsI
            | Opcode::Ceil
            | Opcode::Cos
            | Opcode::CosH
            | Opcode::Exp
            | Opcode::Floor
            | Opcode::Iota
            | Opcode::Log
            | Opcode::Log2
            | Opcode::MakeToken
            | Opcode::NegF
            | Opcode::NegI
            | Opcode::Sin
            | Opcode::SinH
            | Opcode::Tan
            | Opcode::TanH => {
                self.print_unary_result_type(op_id, out);
            }

            // ---- Rsqrt: `$source (flush_to_zero)? attr-dict : type($result)` ----
            Opcode::Rsqrt => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                self.print_attr_dict(op_id, &["flush_to_zero"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Exp2: `$source (flush_to_zero)? attr-dict : type($result)` ----
            Opcode::Exp2 => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                self.print_attr_dict(op_id, &["flush_to_zero"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Sqrt: `$source rounding? (flush_to_zero)? attr-dict : type($result)` ----
            Opcode::Sqrt => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                self.print_optional_rounding(op_id, "rounding_mode", out);
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                self.print_attr_dict(op_id, &["rounding_mode", "flush_to_zero"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern B: Binary same-type `$lhs, $rhs attr-dict : type($result)` ----
            Opcode::AndI
            | Opcode::Atan2
            | Opcode::OrI
            | Opcode::XOrI
            | Opcode::RemF
            | Opcode::MulhiI => {
                self.print_binary_result_type(op_id, out);
            }

            // ---- Pattern C: Binary with overflow ----
            Opcode::AddI | Opcode::MulI | Opcode::ShLI | Opcode::SubI => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                if let Some(ov) = find_int_attr(&op.attributes, "overflow") {
                    let ov_str = overflow_name(ov);
                    if !ov_str.is_empty() {
                        write!(out, " overflow {ov_str}").unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["overflow"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern D: Float arith with rounding ----
            Opcode::AddF | Opcode::DivF | Opcode::MulF | Opcode::SubF => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                self.print_optional_rounding(op_id, "rounding_mode", out);
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                self.print_attr_dict(op_id, &["rounding_mode", "flush_to_zero"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Fma: `$lhs, $rhs, $acc rounding? (flush_to_zero)? attr-dict : type($result)` ----
            Opcode::Fma => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 3 {
                    write!(
                        out,
                        " %{}, %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index(),
                        op.operands[2].index()
                    )
                    .unwrap();
                }
                self.print_optional_rounding(op_id, "rounding_mode", out);
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                self.print_attr_dict(op_id, &["rounding_mode", "flush_to_zero"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern E: CmpF ----
            Opcode::CmpF => {
                let op = self.module.op(op_id);
                let pred = find_int_attr(&op.attributes, "comparison_predicate").unwrap_or(0);
                let ord = find_int_attr(&op.attributes, "comparison_ordering").unwrap_or(0);
                write!(
                    out,
                    " {} {}",
                    comparison_predicate_name(pred),
                    comparison_ordering_name(ord)
                )
                .unwrap();
                self.print_binary_operands(op_id, out);
                self.print_attr_dict(op_id, &["comparison_predicate", "comparison_ordering"], out);
                // `: type($lhs) -> type($result)`
                if !op.operands.is_empty() {
                    let lhs_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(lhs_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern E: CmpI ----
            Opcode::CmpI => {
                let op = self.module.op(op_id);
                let pred = find_int_attr(&op.attributes, "comparison_predicate").unwrap_or(0);
                write!(out, " {}", comparison_predicate_name(pred)).unwrap();
                self.print_binary_operands(op_id, out);
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, ", {}", signedness_name(sign)).unwrap();
                self.print_attr_dict(op_id, &["comparison_predicate", "signedness"], out);
                if !op.operands.is_empty() {
                    let lhs_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(lhs_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern F: Signedness ops ----
            Opcode::MaxI | Opcode::MinI | Opcode::RemI | Opcode::ShRI => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, " {}", signedness_name(sign)).unwrap();
                self.print_attr_dict(op_id, &["signedness"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern G: DivI ----
            Opcode::DivI => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, " {}", signedness_name(sign)).unwrap();
                // Optional rounding — default is ZERO, omitted when zero.
                if let Some(r) = find_int_attr(&op.attributes, "rounding") {
                    if r != 1 {
                        // 1 = ZERO is default; print non-default
                        write!(out, " rounding {}", rounding_mode_name(r)).unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["signedness", "rounding"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- MaxF / MinF: `$lhs, $rhs oilist(flush_to_zero | propagate_nan) attr-dict : type($result)` ----
            Opcode::MaxF | Opcode::MinF => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                if find_bool_attr(&op.attributes, "flush_to_zero") == Some(true) {
                    write!(out, " flush_to_zero").unwrap();
                }
                if find_bool_attr(&op.attributes, "propagate_nan") == Some(true) {
                    write!(out, " propagate_nan").unwrap();
                }
                self.print_attr_dict(op_id, &["flush_to_zero", "propagate_nan"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern H: Cast/convert `$source attr-dict : type($source) -> type($result)` ----
            Opcode::Bitcast
            | Opcode::Broadcast
            | Opcode::IntToPtr
            | Opcode::Pack
            | Opcode::PtrToInt
            | Opcode::PtrToPtr
            | Opcode::Reshape
            | Opcode::Unpack => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- ExtI: `$from signedness attr-dict : type($from) -> type($to)` ----
            Opcode::ExtI => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, " {}", signedness_name(sign)).unwrap();
                self.print_attr_dict(op_id, &["signedness"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- TruncI: `$from (overflow)? attr-dict : type($from) -> type($to)` ----
            Opcode::TruncI => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                if let Some(ov) = find_int_attr(&op.attributes, "overflow") {
                    let ov_str = overflow_name(ov);
                    if !ov_str.is_empty() {
                        write!(out, " overflow {ov_str}").unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["overflow"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- FToF: `$from rounding? attr-dict : type($from) -> type($to)` ----
            Opcode::FToF => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                self.print_optional_rounding(op_id, "rounding_mode", out);
                self.print_attr_dict(op_id, &["rounding_mode"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- FToI: `$from signedness rounding? attr-dict : type($from) -> type($to)` ----
            Opcode::FToI => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, " {}", signedness_name(sign)).unwrap();
                // FToI uses custom<IntegerRoundingMode> — omit if NEAREST_INT_TO_ZERO (6)
                if let Some(r) = find_int_attr(&op.attributes, "rounding_mode") {
                    if r != 6 {
                        write!(out, " rounding<{}>", rounding_mode_name(r)).unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["signedness", "rounding_mode"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- IToF: `$from signedness rounding? attr-dict : type($from) -> type($to)` ----
            Opcode::IToF => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                let sign = find_int_attr(&op.attributes, "signedness").unwrap_or(1);
                write!(out, " {}", signedness_name(sign)).unwrap();
                self.print_optional_rounding(op_id, "rounding_mode", out);
                self.print_attr_dict(op_id, &["signedness", "rounding_mode"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern I: Constant ----
            Opcode::Constant => {
                let op = self.module.op(op_id);
                write!(out, " ").unwrap();
                // Find the DenseElements value attribute.
                if let Some(de) = find_dense_elements_attr(&op.attributes, "value") {
                    print_dense_constant(de, &op.result_types, out);
                } else {
                    // Fallback.
                    write!(out, "dense<? bytes>").unwrap();
                    if !op.result_types.is_empty() {
                        write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["value"], out);
                writeln!(out).unwrap();
            }

            // ---- Pattern J: Terminators ----
            Opcode::Break | Opcode::Continue | Opcode::Return | Opcode::Yield => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " ").unwrap();
                    for (i, &v) in op.operands.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "%{}", v.index()).unwrap();
                    }
                    write!(out, " : ").unwrap();
                    // Types come from the operand values.
                    for (i, &v) in op.operands.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        let ty = self.module.value_type(v);
                        write!(out, "{}", format_type(ty)).unwrap();
                    }
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern K: Assume ----
            Opcode::Assume => {
                let op = self.module.op(op_id);
                // Print predicate without #cuda_tile. prefix.
                let pred_str = find_attr(&op.attributes, "predicate")
                    .map(|a| {
                        let s = format_attr(a);
                        // Strip `#cuda_tile.` prefix if present.
                        if let Some(rest) = s.strip_prefix("#cuda_tile.") {
                            rest.to_string()
                        } else {
                            s
                        }
                    })
                    .unwrap_or_default();
                write!(out, " {pred_str}").unwrap();
                if !op.operands.is_empty() {
                    write!(out, ", %{}", op.operands[0].index()).unwrap();
                }
                self.print_attr_dict(op_id, &["predicate"], out);
                // `: type($value)` — value is the operand.
                if !op.operands.is_empty() {
                    let val_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(val_ty)).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pattern L: For loop ----
            Opcode::For => {
                self.print_for_op(op_id, out);
            }

            // ---- Pattern M: If ----
            Opcode::If => {
                self.print_if_op(op_id, out);
            }

            // ---- Pattern N: Loop ----
            Opcode::Loop => {
                self.print_loop_op(op_id, out);
            }

            // ---- GetTileBlockId / GetNumTileBlocks ----
            Opcode::GetTileBlockId | Opcode::GetNumTileBlocks => {
                let op = self.module.op(op_id);
                self.print_attr_dict(op_id, &[], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- GetGlobal: `$name attr-dict : type($result)` ----
            Opcode::GetGlobal => {
                let op = self.module.op(op_id);
                let gname = find_str_attr(&op.attributes, "name").unwrap_or("?");
                write!(out, " @{gname}").unwrap();
                self.print_attr_dict(op_id, &["name"], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- MakePartitionView: `$source attr-dict : type($result)` ----
            Opcode::MakePartitionView => {
                self.print_unary_result_type(op_id, out);
            }

            // ---- MakeTensorView ----
            Opcode::MakeTensorView => {
                self.print_make_tensor_view(op_id, out);
            }

            // ---- JoinTokens: `$tokens attr-dict : type($result)` ----
            Opcode::JoinTokens => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " ").unwrap();
                    for (i, &v) in op.operands.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "%{}", v.index()).unwrap();
                    }
                }
                self.print_attr_dict(op_id, &[], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Print: `$str (, $args^)? attr-dict (: type($args)^)?` ----
            Opcode::Print => {
                let op = self.module.op(op_id);
                let s = find_str_attr(&op.attributes, "str").unwrap_or("");
                write!(out, " {:?}", s).unwrap();
                if !op.operands.is_empty() {
                    write!(out, ", ").unwrap();
                    for (i, &v) in op.operands.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "%{}", v.index()).unwrap();
                    }
                }
                self.print_attr_dict(op_id, &["str"], out);
                if !op.operands.is_empty() {
                    write!(out, " : ").unwrap();
                    for (i, &v) in op.operands.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        let ty = self.module.value_type(v);
                        write!(out, "{}", format_type(ty)).unwrap();
                    }
                }
                writeln!(out).unwrap();
            }

            // ---- Assert: `$condition, $message attr-dict : type($condition)` ----
            Opcode::Assert => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                let msg = find_str_attr(&op.attributes, "message").unwrap_or("");
                write!(out, ", {:?}", msg).unwrap();
                self.print_attr_dict(op_id, &["message"], out);
                if !op.operands.is_empty() {
                    let cond_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(cond_ty)).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Select: `$cond, $val_if_true, $val_if_false attr-dict : type($cond), type($result)` ----
            Opcode::Select => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 3 {
                    write!(
                        out,
                        " %{}, %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index(),
                        op.operands[2].index()
                    )
                    .unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                if !op.operands.is_empty() {
                    let cond_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(cond_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, ", {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Permute: `$source $permutation attr-dict : type($source) -> type($result)` ----
            Opcode::Permute => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                // Print DenseI32ArrayAttr permutation.
                if let Some(perm) = find_attr(&op.attributes, "permutation") {
                    write!(out, " {}", format_dense_i32_array(perm)).unwrap();
                }
                self.print_attr_dict(op_id, &["permutation"], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Cat: `$lhs, $rhs dim = $dim attr-dict : type($lhs), type($rhs) -> type($result)` ----
            Opcode::Cat => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 2 {
                    write!(
                        out,
                        " %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index()
                    )
                    .unwrap();
                }
                let dim = find_int_attr(&op.attributes, "dim").unwrap_or(0);
                write!(out, " dim = {dim}").unwrap();
                self.print_attr_dict(op_id, &["dim"], out);
                if op.operands.len() >= 2 {
                    let lhs_ty = self.module.value_type(op.operands[0]);
                    let rhs_ty = self.module.value_type(op.operands[1]);
                    write!(out, " : {}, {}", format_type(lhs_ty), format_type(rhs_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Offset: `$ptr, $offset attr-dict : type($ptr), type($offset) -> type($result)` ----
            Opcode::Offset => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 2 {
                    write!(
                        out,
                        " %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index()
                    )
                    .unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                if op.operands.len() >= 2 {
                    let ptr_ty = self.module.value_type(op.operands[0]);
                    let off_ty = self.module.value_type(op.operands[1]);
                    write!(out, " : {}, {}", format_type(ptr_ty), format_type(off_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Pow: `$source, $exponent attr-dict : type($result)` ----
            Opcode::Pow => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 2 {
                    write!(
                        out,
                        " %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index()
                    )
                    .unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                if !op.result_types.is_empty() {
                    write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            Opcode::FPowI => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                self.print_attr_dict(op_id, &[], out);
                if op.operands.len() >= 2 {
                    write!(
                        out,
                        " : {}, {}",
                        format_type(self.module.value_type(op.operands[0])),
                        format_type(self.module.value_type(op.operands[1]))
                    )
                    .unwrap();
                }
                writeln!(out).unwrap();
            }

            Opcode::Insert => {
                let op = self.module.op(op_id);
                self.print_binary_operands(op_id, out);
                write!(out, "[").unwrap();
                for (i, value) in op.operands.iter().skip(2).enumerate() {
                    if i != 0 {
                        write!(out, ", ").unwrap();
                    }
                    write!(out, "%{}", value.index()).unwrap();
                }
                write!(out, "]").unwrap();
                self.print_attr_dict(op_id, &[], out);
                if let (Some(source), Some(result)) = (op.operands.first(), op.result_types.first())
                {
                    write!(
                        out,
                        " : {}, {}",
                        format_type(self.module.value_type(*source)),
                        format_type(result)
                    )
                    .unwrap();
                }
                writeln!(out).unwrap();
            }

            Opcode::GdcLaunchDependentsTko | Opcode::GdcWaitTko => {
                let op = self.module.op(op_id);
                if let Some(token) = op.operands.first() {
                    write!(out, " token = %{}", token.index()).unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                writeln!(out, " -> token").unwrap();
            }

            Opcode::MemoryFenceAliasTko => {
                let op = self.module.op(op_id);
                if let Some(token) = op.operands.first() {
                    write!(out, " %{}", token.index()).unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                writeln!(out).unwrap();
            }

            // ---- Extract: `$source[$indices] attr-dict : type($source) -> type($result)` ----
            Opcode::Extract => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                }
                write!(out, "[").unwrap();
                for (i, &v) in op.operands.iter().skip(1).enumerate() {
                    if i > 0 {
                        write!(out, ", ").unwrap();
                    }
                    write!(out, "%{}", v.index()).unwrap();
                }
                write!(out, "]").unwrap();
                self.print_attr_dict(op_id, &[], out);
                if !op.operands.is_empty() {
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- GetIndexSpaceShape / GetTensorShape ----
            Opcode::GetIndexSpaceShape | Opcode::GetTensorShape => {
                let op = self.module.op(op_id);
                if !op.operands.is_empty() {
                    write!(out, " %{}", op.operands[0].index()).unwrap();
                    let src_ty = self.module.value_type(op.operands[0]);
                    write!(out, " : {}", format_type(src_ty)).unwrap();
                }
                if !op.result_types.is_empty() {
                    write!(out, " -> {}", format_type(&op.result_types[0])).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- Memory ops: LoadViewTko ----
            Opcode::LoadViewTko => {
                self.print_load_view_tko(op_id, out);
            }

            // ---- Memory ops: StoreViewTko ----
            Opcode::StoreViewTko => {
                self.print_store_view_tko(op_id, out);
            }

            // ---- Memory ops: LoadPtrTko ----
            Opcode::LoadPtrTko => {
                self.print_load_ptr_tko(op_id, out);
            }

            // ---- Memory ops: StorePtrTko ----
            Opcode::StorePtrTko => {
                self.print_store_ptr_tko(op_id, out);
            }

            // ---- MmaF: `$lhs, $rhs, $acc attr-dict : type(lhs), type(rhs), type(acc)` ----
            Opcode::MmaF => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 3 {
                    write!(
                        out,
                        " %{}, %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index(),
                        op.operands[2].index()
                    )
                    .unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                write!(out, " :").unwrap();
                for (i, &v) in op.operands.iter().enumerate() {
                    if i > 0 {
                        write!(out, ",").unwrap();
                    }
                    let ty = self.module.value_type(v);
                    write!(out, " {}", format_type(ty)).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- MmaFScaled: `$lhs, $rhs, $acc, $lhs_scale, $rhs_scale attr-dict : type(...)` ----
            Opcode::MmaFScaled => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 5 {
                    write!(
                        out,
                        " %{}, %{}, %{}, %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index(),
                        op.operands[2].index(),
                        op.operands[3].index(),
                        op.operands[4].index()
                    )
                    .unwrap();
                }
                self.print_attr_dict(op_id, &[], out);
                write!(out, " :").unwrap();
                for (i, &v) in op.operands.iter().enumerate() {
                    if i > 0 {
                        write!(out, ",").unwrap();
                    }
                    let ty = self.module.value_type(v);
                    write!(out, " {}", format_type(ty)).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- MmaI: `$lhs, $rhs, $acc signedness_lhs signedness_rhs attr-dict : type(lhs), type(rhs), type(acc)` ----
            Opcode::MmaI => {
                let op = self.module.op(op_id);
                if op.operands.len() >= 3 {
                    write!(
                        out,
                        " %{}, %{}, %{}",
                        op.operands[0].index(),
                        op.operands[1].index(),
                        op.operands[2].index()
                    )
                    .unwrap();
                }
                let sign_lhs = find_int_attr(&op.attributes, "signedness_lhs").unwrap_or(1);
                let sign_rhs = find_int_attr(&op.attributes, "signedness_rhs").unwrap_or(1);
                write!(
                    out,
                    " {} {}",
                    signedness_name(sign_lhs),
                    signedness_name(sign_rhs)
                )
                .unwrap();
                self.print_attr_dict(op_id, &["signedness_lhs", "signedness_rhs"], out);
                write!(out, " :").unwrap();
                for (i, &v) in op.operands.iter().enumerate() {
                    if i > 0 {
                        write!(out, ",").unwrap();
                    }
                    let ty = self.module.value_type(v);
                    write!(out, " {}", format_type(ty)).unwrap();
                }
                writeln!(out).unwrap();
            }

            // ---- AtomicCAS ----
            Opcode::AtomicCAS => {
                self.print_atomic_cas(op_id, out);
            }

            // ---- AtomicRMW ----
            Opcode::AtomicRMW => {
                self.print_atomic_rmw(op_id, out);
            }

            // ---- Reduce / Scan ----
            Opcode::Reduce => {
                self.print_reduce_or_scan(op_id, out);
            }
            Opcode::Scan => {
                self.print_reduce_or_scan(op_id, out);
            }

            // ---- Global / Module / Entry handled elsewhere ----
            Opcode::Alloca
            | Opcode::AtomicRedViewTko
            | Opcode::Global
            | Opcode::MakeGatherScatterView
            | Opcode::MakeStridedView
            | Opcode::Module
            | Opcode::Entry => {
                writeln!(out, " <structural-op>").unwrap();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Shared printing helpers
    // -----------------------------------------------------------------------

    /// Print ` %lhs, %rhs` from the first two operands.
    fn print_binary_operands(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        if op.operands.len() >= 2 {
            write!(
                out,
                " %{}, %{}",
                op.operands[0].index(),
                op.operands[1].index()
            )
            .unwrap();
        }
    }

    /// Pattern A helper: `$source attr-dict : type($result)`.
    fn print_unary_result_type(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        if !op.operands.is_empty() {
            write!(out, " %{}", op.operands[0].index()).unwrap();
        }
        self.print_attr_dict(op_id, &[], out);
        if !op.result_types.is_empty() {
            write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// Pattern B helper: `$lhs, $rhs attr-dict : type($result)`.
    fn print_binary_result_type(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        self.print_binary_operands(op_id, out);
        self.print_attr_dict(op_id, &[], out);
        if !op.result_types.is_empty() {
            write!(out, " : {}", format_type(&op.result_types[0])).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// Print rounding mode if not nearest_even (mode 0).
    fn print_optional_rounding(&self, op_id: OpId, attr_name: &str, out: &mut String) {
        let op = self.module.op(op_id);
        if let Some(r) = find_int_attr(&op.attributes, attr_name) {
            if r != 0 {
                // 0 = NEAREST_EVEN, omitted.
                write!(out, " rounding<{}>", rounding_mode_name(r)).unwrap();
            }
        }
    }

    /// Print `{key = val, ...}` for attributes not already printed inline.
    /// `consumed` is the list of attribute names that were already printed.
    fn print_attr_dict(&self, op_id: OpId, consumed: &[&str], out: &mut String) {
        let op = self.module.op(op_id);
        let extras: Vec<_> = op
            .attributes
            .iter()
            .filter(|(k, _)| !is_structural_attr(k) && !consumed.contains(&k.as_str()))
            .collect();
        if !extras.is_empty() {
            write!(out, " {{").unwrap();
            for (i, (k, v)) in extras.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{k} = {}", format_attr(v)).unwrap();
            }
            write!(out, "}}").unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // Hand-written complex op printers
    // -----------------------------------------------------------------------

    /// ForOp: `for %iv in (%lb to %ub, step %step) : type iter_values(...) -> (...) { ... }`
    fn print_for_op(&mut self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        let pad = " ".repeat(self.indent);

        // Operands: [lb, ub, step, init_values...]
        let lb = op.operands.first().map(|v| v.index());
        let ub = op.operands.get(1).map(|v| v.index());
        let step = op.operands.get(2).map(|v| v.index());
        let init_values = &op.operands[3.min(op.operands.len())..];

        // Region block args: [iv, iter_args...]
        let region_id = op.regions.first().copied();
        let block_id = region_id.and_then(|r| self.module.region(r).blocks.first().copied());
        let block_args = block_id
            .map(|b| self.module.block(b).args.clone())
            .unwrap_or_default();
        let iv = block_args.first().map(|(v, _)| v.index());
        let iter_args = &block_args[1.min(block_args.len())..];

        // `%iv in (%lb to %ub, step %step) : iv_type`
        if let Some(iv_idx) = iv {
            write!(out, " %{iv_idx} in (").unwrap();
        } else {
            write!(out, " %? in (").unwrap();
        }
        if let Some(l) = lb {
            write!(out, "%{l}").unwrap();
        }
        write!(out, " to ").unwrap();
        if let Some(u) = ub {
            write!(out, "%{u}").unwrap();
        }
        write!(out, ", step ").unwrap();
        if let Some(s) = step {
            write!(out, "%{s}").unwrap();
        }
        write!(out, ") : ").unwrap();

        // IV type.
        if let Some((_, ty)) = block_args.first() {
            write!(out, "{}", format_type(ty)).unwrap();
        }
        write!(out, " ").unwrap();

        // iter_values if present.
        if !init_values.is_empty() {
            write!(out, "iter_values(").unwrap();
            for (i, ((arg_val, _), init_val)) in
                iter_args.iter().zip(init_values.iter()).enumerate()
            {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "%{} = %{}", arg_val.index(), init_val.index()).unwrap();
            }
            write!(out, ") -> (").unwrap();
            for (i, iv) in init_values.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                let ty = self.module.value_type(*iv);
                write!(out, "{}", format_type(ty)).unwrap();
            }
            write!(out, ") ").unwrap();
        }

        // Region body.
        writeln!(out, "{{").unwrap();
        self.indent += 2;
        if let Some(rid) = region_id {
            self.print_region_body(rid, true, out);
        }
        self.indent -= 2;
        writeln!(out, "{pad}}}").unwrap();
    }

    /// IfOp: `if %cond (-> (types))? { ... } (else { ... })?`
    fn print_if_op(&mut self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        let pad = " ".repeat(self.indent);

        // Condition.
        if !op.operands.is_empty() {
            write!(out, " %{}", op.operands[0].index()).unwrap();
        }

        // Result types.
        if !op.result_types.is_empty() {
            write!(out, " -> (").unwrap();
            for (i, ty) in op.result_types.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{}", format_type(ty)).unwrap();
            }
            write!(out, ")").unwrap();
        }

        // Then region.
        writeln!(out, " {{").unwrap();
        self.indent += 2;
        if let Some(&rid) = op.regions.first() {
            self.print_region_body_if(rid, out);
        }
        self.indent -= 2;

        // Else region.
        if op.regions.len() > 1 {
            let else_rid = op.regions[1];
            let else_region = self.module.region(else_rid);
            let else_empty = else_region.blocks.iter().all(|&bid| {
                let block = self.module.block(bid);
                block.ops.is_empty()
                    || (block.ops.len() == 1
                        && self.module.op(block.ops[0]).opcode == Opcode::Yield
                        && self.module.op(block.ops[0]).operands.is_empty())
            });
            if !else_empty {
                writeln!(out, "{pad}}} else {{").unwrap();
                self.indent += 2;
                self.print_region_body_if(else_rid, out);
                self.indent -= 2;
            }
        }

        writeln!(out, "{pad}}}").unwrap();
    }

    /// Print a region body for if, suppressing the implicit yield terminator
    /// if it has no operands.
    fn print_region_body_if(&mut self, region_id: RegionId, out: &mut String) {
        let region = self.module.region(region_id);
        for &block_id in &region.blocks {
            let ops: Vec<OpId> = self.module.block(block_id).ops.clone();
            for op_id in ops {
                let op = self.module.op(op_id);
                // Suppress implicit yield with no operands.
                if op.opcode == Opcode::Yield && op.operands.is_empty() {
                    continue;
                }
                self.print_operation(op_id, out);
            }
        }
    }

    /// LoopOp: `cuda_tile.loop (iter_values(...))? (: types (-> types)?)? { ... }`
    fn print_loop_op(&mut self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        let pad = " ".repeat(self.indent);
        let init_values = &op.operands;
        let has_iters = !init_values.is_empty();
        let has_return = !op.result_types.is_empty();

        let region_id = op.regions.first().copied();
        let block_id = region_id.and_then(|r| self.module.region(r).blocks.first().copied());
        let block_args = block_id
            .map(|b| self.module.block(b).args.clone())
            .unwrap_or_default();

        write!(out, " ").unwrap();

        if has_iters {
            write!(out, "iter_values(").unwrap();
            for (i, ((arg_val, _), init_val)) in
                block_args.iter().zip(init_values.iter()).enumerate()
            {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "%{} = %{}", arg_val.index(), init_val.index()).unwrap();
            }
            write!(out, ") ").unwrap();
        }

        if has_iters || has_return {
            write!(out, ": ").unwrap();
        }

        if has_iters {
            for (i, iv) in init_values.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                let ty = self.module.value_type(*iv);
                write!(out, "{}", format_type(ty)).unwrap();
            }
            write!(out, " ").unwrap();
            if has_return {
                write!(out, "-> ").unwrap();
            }
        }

        if has_return {
            for (i, ty) in op.result_types.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{}", format_type(ty)).unwrap();
            }
            write!(out, " ").unwrap();
        }

        writeln!(out, "{{").unwrap();
        self.indent += 2;
        if let Some(rid) = region_id {
            self.print_region_body(rid, true, out);
        }
        self.indent -= 2;
        writeln!(out, "{pad}}}").unwrap();
    }

    /// MakeTensorView: `$base, shape = [...], strides = [...] (: dyn_type ->)? type($result)`
    fn print_make_tensor_view(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        // Operands: [base, dynamic_shape..., dynamic_strides...]
        // The result type tells us the static shape/strides.
        let result_ty = op.result_types.first();
        let tv = match result_ty {
            Some(Type::TensorView(tv)) => Some(tv),
            _ => None,
        };

        // First operand is base pointer.
        if !op.operands.is_empty() {
            write!(out, " %{}", op.operands[0].index()).unwrap();
        }

        // Figure out operandSegmentSizes to split dynamic operands.
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_base, n_dyn_shape, n_dyn_stride) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2))
            }
            _ => (1, 0, 0),
        };

        let dyn_shape_ops = &op.operands
            [n_base..n_base + n_dyn_shape.min(op.operands.len().saturating_sub(n_base))];
        let dyn_stride_ops = &op.operands
            [n_base + n_dyn_shape..(n_base + n_dyn_shape + n_dyn_stride).min(op.operands.len())];

        // Print shape.
        write!(out, ", shape = [").unwrap();
        if let Some(tv) = tv {
            let mut dyn_idx = 0;
            for (i, &dim) in tv.shape.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if dim == DYNAMIC || dim < 0 {
                    if dyn_idx < dyn_shape_ops.len() {
                        write!(out, "%{}", dyn_shape_ops[dyn_idx].index()).unwrap();
                        dyn_idx += 1;
                    } else {
                        write!(out, "?").unwrap();
                    }
                } else {
                    write!(out, "{dim}").unwrap();
                }
            }
        }
        write!(out, "], strides = [").unwrap();
        if let Some(tv) = tv {
            let mut dyn_idx = 0;
            for (i, &stride) in tv.strides.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if stride == DYNAMIC || stride < 0 {
                    if dyn_idx < dyn_stride_ops.len() {
                        write!(out, "%{}", dyn_stride_ops[dyn_idx].index()).unwrap();
                        dyn_idx += 1;
                    } else {
                        write!(out, "?").unwrap();
                    }
                } else {
                    write!(out, "{stride}").unwrap();
                }
            }
        }
        write!(out, "]").unwrap();

        // attr-dict (excluding operandSegmentSizes).
        let extras: Vec<_> = op
            .attributes
            .iter()
            .filter(|(k, _)| !is_structural_attr(k))
            .collect();
        if !extras.is_empty() {
            write!(out, " {{").unwrap();
            for (i, (k, v)) in extras.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{k} = {}", format_attr(v)).unwrap();
            }
            write!(out, "}}").unwrap();
        }

        write!(out, " : ").unwrap();

        // If there are dynamic operands, print their type and arrow.
        if !dyn_shape_ops.is_empty() || !dyn_stride_ops.is_empty() {
            let dyn_val = if !dyn_shape_ops.is_empty() {
                dyn_shape_ops[0]
            } else {
                dyn_stride_ops[0]
            };
            let dyn_ty = self.module.value_type(dyn_val);
            write!(out, "{} -> ", format_type(dyn_ty)).unwrap();
        }

        if let Some(rty) = result_ty {
            write!(out, "{}", format_type(rty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// LoadViewTko: `memory_ordering (scope)? $view[$index] (token = $tok)? (opt_hints)? ... : types`
    fn print_load_view_tko(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope");

        write!(out, " {}", memory_ordering_name(mem_ord)).unwrap();
        if let Some(scope) = mem_scope {
            write!(out, " {}", memory_scope_name(scope)).unwrap();
        }

        // operands: determined by operandSegmentSizes
        // [view, index..., token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_view, n_index, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2))
            }
            _ => (1, 0, 0),
        };

        let view_ops = &op.operands[..n_view.min(op.operands.len())];
        let index_start = n_view;
        let index_ops = &op.operands[index_start..(index_start + n_index).min(op.operands.len())];
        let token_start = index_start + n_index;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        // $view[$index]
        if !view_ops.is_empty() {
            write!(out, " %{}", view_ops[0].index()).unwrap();
        }
        write!(out, "[").unwrap();
        for (i, &v) in index_ops.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "%{}", v.index()).unwrap();
        }
        write!(out, "]").unwrap();

        if !token_ops.is_empty() {
            write!(out, " token = %{}", token_ops[0].index()).unwrap();
        }

        // optimization_hints
        if let Some(oh) = find_attr(&op.attributes, "optimization_hints") {
            write!(out, " optimization_hints = {}", format_attr(oh)).unwrap();
        }

        // types: view_type, index_type -> tile_type, token_type
        write!(out, " : ").unwrap();
        if !view_ops.is_empty() {
            let vty = self.module.value_type(view_ops[0]);
            write!(out, "{}", format_type(vty)).unwrap();
        }
        if !index_ops.is_empty() {
            // All index operands should share the same type.
            let ity = self.module.value_type(index_ops[0]);
            write!(out, ", {}", format_type(ity)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// StoreViewTko
    fn print_store_view_tko(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope");

        write!(out, " {}", memory_ordering_name(mem_ord)).unwrap();
        if let Some(scope) = mem_scope {
            write!(out, " {}", memory_scope_name(scope)).unwrap();
        }

        // operands: [tile, view, index..., token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_tile, n_view, n_index, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2), get(3))
            }
            _ => (1, 1, 0, 0),
        };

        let tile_ops = &op.operands[..n_tile.min(op.operands.len())];
        let view_start = n_tile;
        let view_ops = &op.operands[view_start..(view_start + n_view).min(op.operands.len())];
        let index_start = view_start + n_view;
        let index_ops = &op.operands[index_start..(index_start + n_index).min(op.operands.len())];
        let token_start = index_start + n_index;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        // $tile, $view[$index]
        if !tile_ops.is_empty() {
            write!(out, " %{}", tile_ops[0].index()).unwrap();
        }
        if !view_ops.is_empty() {
            write!(out, ", %{}", view_ops[0].index()).unwrap();
        }
        write!(out, "[").unwrap();
        for (i, &v) in index_ops.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "%{}", v.index()).unwrap();
        }
        write!(out, "]").unwrap();

        if !token_ops.is_empty() {
            write!(out, " token = %{}", token_ops[0].index()).unwrap();
        }

        if let Some(oh) = find_attr(&op.attributes, "optimization_hints") {
            write!(out, " optimization_hints = {}", format_attr(oh)).unwrap();
        }

        // types: tile_type, view_type, index_type -> token_type
        write!(out, " : ").unwrap();
        if !tile_ops.is_empty() {
            let tty = self.module.value_type(tile_ops[0]);
            write!(out, "{}", format_type(tty)).unwrap();
        }
        if !view_ops.is_empty() {
            let vty = self.module.value_type(view_ops[0]);
            write!(out, ", {}", format_type(vty)).unwrap();
        }
        if !index_ops.is_empty() {
            let ity = self.module.value_type(index_ops[0]);
            write!(out, ", {}", format_type(ity)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// LoadPtrTko
    fn print_load_ptr_tko(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope");

        write!(out, " {}", memory_ordering_name(mem_ord)).unwrap();
        if let Some(scope) = mem_scope {
            write!(out, " {}", memory_scope_name(scope)).unwrap();
        }

        // operands: [source, mask?, paddingValue?, token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_src, n_mask, n_pad, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2), get(3))
            }
            _ => (1, 0, 0, 0),
        };

        let src_ops = &op.operands[..n_src.min(op.operands.len())];
        let mask_start = n_src;
        let mask_ops = &op.operands[mask_start..(mask_start + n_mask).min(op.operands.len())];
        let pad_start = mask_start + n_mask;
        let pad_ops = &op.operands[pad_start..(pad_start + n_pad).min(op.operands.len())];
        let token_start = pad_start + n_pad;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        if !src_ops.is_empty() {
            write!(out, " %{}", src_ops[0].index()).unwrap();
        }
        if !mask_ops.is_empty() {
            write!(out, ", %{}", mask_ops[0].index()).unwrap();
        }
        if !pad_ops.is_empty() {
            write!(out, ", %{}", pad_ops[0].index()).unwrap();
        }
        if !token_ops.is_empty() {
            write!(out, " token=%{}", token_ops[0].index()).unwrap();
        }

        if let Some(oh) = find_attr(&op.attributes, "optimization_hints") {
            write!(out, " optimization_hints = {}", format_attr(oh)).unwrap();
        }

        // types: source, (mask,)? (pad,)? -> result, token
        write!(out, " : ").unwrap();
        if !src_ops.is_empty() {
            let sty = self.module.value_type(src_ops[0]);
            write!(out, "{}", format_type(sty)).unwrap();
        }
        if !mask_ops.is_empty() {
            let mty = self.module.value_type(mask_ops[0]);
            write!(out, ", {}", format_type(mty)).unwrap();
        }
        if !pad_ops.is_empty() {
            let pty = self.module.value_type(pad_ops[0]);
            write!(out, ", {}", format_type(pty)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// StorePtrTko
    fn print_store_ptr_tko(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope");

        write!(out, " {}", memory_ordering_name(mem_ord)).unwrap();
        if let Some(scope) = mem_scope {
            write!(out, " {}", memory_scope_name(scope)).unwrap();
        }

        // operands: [destination, value, mask?, token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_dst, n_val, n_mask, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2), get(3))
            }
            _ => (1, 1, 0, 0),
        };

        let dst_ops = &op.operands[..n_dst.min(op.operands.len())];
        let val_start = n_dst;
        let val_ops = &op.operands[val_start..(val_start + n_val).min(op.operands.len())];
        let mask_start = val_start + n_val;
        let mask_ops = &op.operands[mask_start..(mask_start + n_mask).min(op.operands.len())];
        let token_start = mask_start + n_mask;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        if !dst_ops.is_empty() {
            write!(out, " %{}", dst_ops[0].index()).unwrap();
        }
        if !val_ops.is_empty() {
            write!(out, ", %{}", val_ops[0].index()).unwrap();
        }
        if !mask_ops.is_empty() {
            write!(out, ", %{}", mask_ops[0].index()).unwrap();
        }
        if !token_ops.is_empty() {
            write!(out, " token=%{}", token_ops[0].index()).unwrap();
        }

        if let Some(oh) = find_attr(&op.attributes, "optimization_hints") {
            write!(out, " optimization_hints = {}", format_attr(oh)).unwrap();
        }

        // types: dst, val, (mask,)? -> token
        write!(out, " : ").unwrap();
        if !dst_ops.is_empty() {
            let dty = self.module.value_type(dst_ops[0]);
            write!(out, "{}", format_type(dty)).unwrap();
        }
        if !val_ops.is_empty() {
            let vty = self.module.value_type(val_ops[0]);
            write!(out, ", {}", format_type(vty)).unwrap();
        }
        if !mask_ops.is_empty() {
            let mty = self.module.value_type(mask_ops[0]);
            write!(out, ", {}", format_type(mty)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// AtomicCAS
    fn print_atomic_cas(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope").unwrap_or(0);
        write!(
            out,
            " {} {}",
            memory_ordering_name(mem_ord),
            memory_scope_name(mem_scope)
        )
        .unwrap();

        // operands: [pointers, cmp, val, mask?, token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_ptr, n_cmp, n_val, n_mask, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2), get(3), get(4))
            }
            _ => (1, 1, 1, 0, 0),
        };

        let ptr_ops = &op.operands[..n_ptr.min(op.operands.len())];
        let cmp_start = n_ptr;
        let cmp_ops = &op.operands[cmp_start..(cmp_start + n_cmp).min(op.operands.len())];
        let val_start = cmp_start + n_cmp;
        let val_ops = &op.operands[val_start..(val_start + n_val).min(op.operands.len())];
        let mask_start = val_start + n_val;
        let mask_ops = &op.operands[mask_start..(mask_start + n_mask).min(op.operands.len())];
        let token_start = mask_start + n_mask;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        if !ptr_ops.is_empty() {
            write!(out, " %{}", ptr_ops[0].index()).unwrap();
        }
        if !cmp_ops.is_empty() {
            write!(out, ", %{}", cmp_ops[0].index()).unwrap();
        }
        if !val_ops.is_empty() {
            write!(out, ", %{}", val_ops[0].index()).unwrap();
        }
        if !mask_ops.is_empty() {
            write!(out, ", %{}", mask_ops[0].index()).unwrap();
        }
        if !token_ops.is_empty() {
            write!(out, " token=%{}", token_ops[0].index()).unwrap();
        }

        self.print_attr_dict(op_id, &["memory_ordering_semantics", "memory_scope"], out);

        // types
        write!(out, " : ").unwrap();
        // Print operand types, then result types.
        for (i, &v) in op.operands.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            let ty = self.module.value_type(v);
            write!(out, "{}", format_type(ty)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// AtomicRMW
    fn print_atomic_rmw(&self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);

        let mem_ord = find_int_attr(&op.attributes, "memory_ordering_semantics").unwrap_or(0);
        let mem_scope = find_int_attr(&op.attributes, "memory_scope").unwrap_or(0);
        write!(
            out,
            " {} {}",
            memory_ordering_name(mem_ord),
            memory_scope_name(mem_scope)
        )
        .unwrap();

        // operands: [pointers, mode, arg, mask?, token?]
        let seg_sizes = find_attr(&op.attributes, "operandSegmentSizes");
        let (n_ptr, n_mode, n_arg, n_mask, n_token) = match seg_sizes {
            Some(Attribute::Array(arr)) => {
                let get = |i: usize| match arr.get(i) {
                    Some(Attribute::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2), get(3), get(4))
            }
            _ => (1, 1, 1, 0, 0),
        };

        let ptr_ops = &op.operands[..n_ptr.min(op.operands.len())];
        let mode_start = n_ptr;
        let mode_ops = &op.operands[mode_start..(mode_start + n_mode).min(op.operands.len())];
        let arg_start = mode_start + n_mode;
        let arg_ops = &op.operands[arg_start..(arg_start + n_arg).min(op.operands.len())];
        let mask_start = arg_start + n_arg;
        let mask_ops = &op.operands[mask_start..(mask_start + n_mask).min(op.operands.len())];
        let token_start = mask_start + n_mask;
        let token_ops = &op.operands[token_start..(token_start + n_token).min(op.operands.len())];

        if !ptr_ops.is_empty() {
            write!(out, " %{}", ptr_ops[0].index()).unwrap();
        }
        if !mode_ops.is_empty() {
            write!(out, ", %{}", mode_ops[0].index()).unwrap();
        }
        if !arg_ops.is_empty() {
            write!(out, ", %{}", arg_ops[0].index()).unwrap();
        }
        if !mask_ops.is_empty() {
            write!(out, ", %{}", mask_ops[0].index()).unwrap();
        }
        if !token_ops.is_empty() {
            write!(out, " token=%{}", token_ops[0].index()).unwrap();
        }

        self.print_attr_dict(op_id, &["memory_ordering_semantics", "memory_scope"], out);

        write!(out, " : ").unwrap();
        for (i, &v) in op.operands.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            let ty = self.module.value_type(v);
            write!(out, "{}", format_type(ty)).unwrap();
        }
        write!(out, " -> ").unwrap();
        for (i, ty) in op.result_types.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", format_type(ty)).unwrap();
        }
        writeln!(out).unwrap();
    }

    /// Reduce / Scan
    fn print_reduce_or_scan(&mut self, op_id: OpId, out: &mut String) {
        let op = self.module.op(op_id);
        let pad = " ".repeat(self.indent);
        let is_scan = op.opcode == Opcode::Scan;

        // Print operands.
        if !op.operands.is_empty() {
            write!(out, " ").unwrap();
            for (i, &v) in op.operands.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "%{}", v.index()).unwrap();
            }
        }

        // dim = N identities = [...]
        let dim = find_int_attr(&op.attributes, "dim").unwrap_or(0);
        write!(out, " dim={dim}").unwrap();

        if is_scan {
            let reverse = find_bool_attr(&op.attributes, "reverse").unwrap_or(false);
            write!(out, " reverse={reverse}").unwrap();
        }

        if let Some(ids) = find_attr(&op.attributes, "identities") {
            write!(out, " identities={}", format_attr(ids)).unwrap();
        }

        // types
        write!(out, " : ").unwrap();
        // Print operand types.
        for (i, &v) in op.operands.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            let ty = self.module.value_type(v);
            write!(out, "{}", format_type(ty)).unwrap();
        }
        if !op.result_types.is_empty() {
            write!(out, " -> ").unwrap();
            for (i, ty) in op.result_types.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{}", format_type(ty)).unwrap();
            }
        }

        // Region body.
        if !op.regions.is_empty() {
            writeln!(out, " {{").unwrap();
            self.indent += 2;
            for &rid in &op.regions {
                self.print_region_body(rid, false, out);
            }
            self.indent -= 2;
            write!(out, "{pad}}}").unwrap();
        }
        writeln!(out).unwrap();
    }
}

// =========================================================================
// Formatting helpers
// =========================================================================

/// Map opcode to its MLIR operation name.
fn opcode_name(opcode: Opcode) -> &'static str {
    if opcode == Opcode::Loop {
        "cuda_tile.loop"
    } else {
        opcode.name()
    }
}

/// Format a tile-ir `Type` as an MLIR type string (shorthand, no `!cuda_tile.` prefix).
pub fn format_type(ty: &Type) -> String {
    match ty {
        Type::WithPointerAttribute(base, _) => {
            let mut text = format_type(base);
            if let Some(start) = text.find("ptr<").or_else(|| text.find("tensor_view<")) {
                if let Some(end) = text[start..].find('>') {
                    text.insert_str(start + end, ", #cuda_tile.ptr_attr<none>");
                }
            }
            text
        }
        Type::Scalar(s) => format_scalar(*s),
        Type::Pointer(p) => format!("ptr<{}>", format_scalar(p.pointee)),
        Type::Tile(t) => {
            let elem = match &t.element_type {
                TileElementType::Scalar(s) => format_scalar(*s),
                TileElementType::Pointer(p) => {
                    format!("ptr<{}>", format_scalar(p.pointee))
                }
            };
            if t.shape.is_empty() {
                format!("tile<{elem}>")
            } else {
                let shape = t
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("tile<{shape}x{elem}>")
            }
        }
        Type::TensorView(tv) => {
            let elem = format_scalar(tv.element_type);
            let shape = tv
                .shape
                .iter()
                .map(|d| {
                    if *d == DYNAMIC || *d < 0 {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("x");
            let strides = tv
                .strides
                .iter()
                .map(|d| {
                    if *d == DYNAMIC || *d < 0 {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("tensor_view<{shape}x{elem}, strides=[{strides}]>")
        }
        Type::PartitionView(pv) => {
            let tv = format_type(&Type::TensorView(pv.tensor_view.clone()));
            let tile_shape = pv
                .tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let padding = match &pv.padding_value {
                Some(PaddingValue::Zero) => ", padding_value = zero",
                Some(PaddingValue::NegZero) => ", padding_value = neg_zero",
                Some(PaddingValue::Nan) => ", padding_value = nan",
                Some(PaddingValue::PosInf) => ", padding_value = pos_inf",
                Some(PaddingValue::NegInf) => ", padding_value = neg_inf",
                None => "",
            };
            // Emit dim_map when it differs from the identity mapping [0,1,...,n-1].
            let is_identity = pv.dim_map.iter().enumerate().all(|(i, &d)| d == i as i32);
            let dim_map_str = if is_identity {
                String::new()
            } else {
                let vals = pv
                    .dim_map
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!(", dim_map=[{vals}]")
            };
            format!("partition_view<tile=({tile_shape}){padding}, {tv}{dim_map_str}>")
        }
        Type::GatherScatterView(gsv) => {
            let tv = format_type(&Type::TensorView(gsv.tensor_view.clone()));
            let tile_shape = gsv
                .tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let padding = match &gsv.padding_value {
                Some(PaddingValue::Zero) => ", padding_value = zero",
                Some(PaddingValue::NegZero) => ", padding_value = neg_zero",
                Some(PaddingValue::Nan) => ", padding_value = nan",
                Some(PaddingValue::PosInf) => ", padding_value = pos_inf",
                Some(PaddingValue::NegInf) => ", padding_value = neg_inf",
                None => "",
            };
            format!(
                "gather_scatter_view<tile=({tile_shape}){padding}, {tv}, sparse_dim={}>",
                gsv.sparse_dim
            )
        }
        Type::StridedView(sv) => {
            let tv = format_type(&Type::TensorView(sv.tensor_view.clone()));
            let tile_shape = sv
                .tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let traversal_strides = sv
                .traversal_strides
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let padding = match &sv.padding_value {
                Some(PaddingValue::Zero) => ", padding_value = zero",
                Some(PaddingValue::NegZero) => ", padding_value = neg_zero",
                Some(PaddingValue::Nan) => ", padding_value = nan",
                Some(PaddingValue::PosInf) => ", padding_value = pos_inf",
                Some(PaddingValue::NegInf) => ", padding_value = neg_inf",
                None => "",
            };
            let is_identity = sv.dim_map.iter().enumerate().all(|(i, &d)| d == i as i32);
            let dim_map_str = if is_identity {
                String::new()
            } else {
                let vals = sv
                    .dim_map
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!(", dim_map=[{vals}]")
            };
            format!(
                "strided_view<tile=({tile_shape}){padding}, traversal_strides=[{traversal_strides}], {tv}{dim_map_str}>"
            )
        }
        Type::Func(f) => {
            let inputs = f
                .inputs
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(", ");
            let results = f
                .results
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inputs}) -> ({results})")
        }
        Type::Token => "token".into(),
    }
}

fn format_scalar(s: ScalarType) -> String {
    match s {
        ScalarType::I1 => "i1",
        ScalarType::I4 => "i4",
        ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F16 => "f16",
        ScalarType::BF16 => "bf16",
        ScalarType::F32 => "f32",
        ScalarType::TF32 => "tf32",
        ScalarType::F64 => "f64",
        ScalarType::F8E4M3FN => "f8e4m3fn",
        ScalarType::F8E5M2 => "f8e5m2",
        ScalarType::F8E8M0FNU => "f8e8m0fnu",
        ScalarType::F8E5M3FNU => "f8e5m3fnu",
        ScalarType::F4E2M1FN => "f4e2m1fn",
    }
    .into()
}

fn format_attr(attr: &Attribute) -> String {
    match attr {
        Attribute::Integer(v, _) => v.to_string(),
        Attribute::Float(v, _) => format!("{v}"),
        Attribute::Bool(v) => if *v { "true" } else { "false" }.into(),
        Attribute::String(s) => format!("{s:?}"),
        Attribute::Type(ty) => format_type(ty),
        Attribute::Array(arr) => {
            let elems: Vec<_> = arr.iter().map(format_attr).collect();
            format!("[{}]", elems.join(", "))
        }
        Attribute::DenseElements(de) => {
            format!(
                "dense<{} bytes> : {}",
                de.data.len(),
                format_type(&de.element_type)
            )
        }
        Attribute::DenseI32Array(arr) => {
            let vals: Vec<_> = arr.iter().map(|v| v.to_string()).collect();
            format!("[{}]", vals.join(", "))
        }
        Attribute::DivBy(db) => {
            let mut s = format!("#cuda_tile.div_by<{}", db.divisor);
            if let (Some(every), Some(along)) = (db.every, db.along) {
                write!(s, ", every {every} along {along}").unwrap();
            }
            s.push('>');
            s
        }
        Attribute::SameElements(se) => {
            let vals: Vec<_> = se.values.iter().map(|v| v.to_string()).collect();
            format!("#cuda_tile.same_elements<[{}]>", vals.join(", "))
        }
        Attribute::Dictionary(entries) => {
            let kvs: Vec<_> = entries
                .iter()
                .map(|(k, v)| format!("{k} = {}", format_attr(v)))
                .collect();
            format!("{{{}}}", kvs.join(", "))
        }
        Attribute::OptimizationHints(oh) => {
            let archs: Vec<_> = oh
                .entries
                .iter()
                .map(|(arch, hints)| {
                    let hs: Vec<_> = hints
                        .iter()
                        .map(|(k, v)| format!("{k} = {}", format_attr(v)))
                        .collect();
                    format!("{arch} = {{{}}}", hs.join(", "))
                })
                .collect();
            format!("#cuda_tile.optimization_hints<{}>", archs.join(", "))
        }
        Attribute::Bounded(b) => {
            let lb = b.lb.map(|v| v.to_string()).unwrap_or_else(|| "?".into());
            let ub = b.ub.map(|v| v.to_string()).unwrap_or_else(|| "?".into());
            format!("#cuda_tile.bounded<{lb}, {ub}>")
        }
    }
}

// =========================================================================
// Attribute lookup helpers
// =========================================================================

fn find_str_attr<'a>(attrs: &'a [(String, Attribute)], name: &str) -> Option<&'a str> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::String(s) = v {
                return Some(s.as_str());
            }
        }
        None
    })
}

fn find_int_attr(attrs: &[(String, Attribute)], name: &str) -> Option<i64> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::Integer(i, _) = v {
                return Some(*i);
            }
        }
        None
    })
}

fn find_bool_attr(attrs: &[(String, Attribute)], name: &str) -> Option<bool> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            match v {
                Attribute::Bool(b) => return Some(*b),
                // UnitAttr is stored as Bool(true) in the builder, but
                // just its presence counts as true.
                _ => return Some(true),
            }
        }
        None
    })
}

fn find_attr<'a>(attrs: &'a [(String, Attribute)], name: &str) -> Option<&'a Attribute> {
    attrs
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

fn find_dense_elements_attr<'a>(
    attrs: &'a [(String, Attribute)],
    name: &str,
) -> Option<&'a super::attr::DenseElements> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::DenseElements(de) = v {
                return Some(de);
            }
        }
        None
    })
}

fn is_structural_attr(name: &str) -> bool {
    matches!(
        name,
        "sym_name" | "function_type" | "operandSegmentSizes" | "arg_attrs" | "res_attrs"
    )
}

// =========================================================================
// Enum name helpers
// =========================================================================

fn signedness_name(v: i64) -> &'static str {
    match v {
        0 => "unsigned",
        1 => "signed",
        _ => "signed",
    }
}

fn rounding_mode_name(v: i64) -> &'static str {
    match v {
        0 => "nearest_even",
        1 => "zero",
        2 => "negative_inf",
        3 => "positive_inf",
        4 => "approx",
        5 => "full",
        6 => "nearest_int_to_zero",
        7 => "nearest_away",
        _ => "nearest_even",
    }
}

fn comparison_predicate_name(v: i64) -> &'static str {
    match v {
        0 => "equal",
        1 => "not_equal",
        2 => "less_than",
        3 => "less_than_or_equal",
        4 => "greater_than",
        5 => "greater_than_or_equal",
        _ => "equal",
    }
}

fn comparison_ordering_name(v: i64) -> &'static str {
    match v {
        0 => "unordered",
        1 => "ordered",
        _ => "ordered",
    }
}

fn memory_ordering_name(v: i64) -> &'static str {
    match v {
        0 => "weak",
        1 => "relaxed",
        2 => "acquire",
        3 => "release",
        4 => "acq_rel",
        _ => "weak",
    }
}

fn memory_scope_name(v: i64) -> &'static str {
    match v {
        0 => "tl_blk",
        1 => "device",
        2 => "sys",
        _ => "tl_blk",
    }
}

fn overflow_name(v: i64) -> &'static str {
    match v {
        0 => "", // NONE — omit
        1 => "nsw",
        2 => "nuw",
        3 => "nw",
        _ => "",
    }
}

// =========================================================================
// Dense constant printing
// =========================================================================

/// Print a DenseElements constant in the format `<element_type: value> : tile_type`.
fn print_dense_constant(de: &super::attr::DenseElements, result_types: &[Type], out: &mut String) {
    // Extract the element type. The DenseElements element_type is the full
    // tile type; extract the scalar element from it.
    let (elem_name, scalar) = match &de.element_type {
        Type::Tile(t) => match &t.element_type {
            TileElementType::Scalar(s) => (format_scalar(*s), Some(*s)),
            TileElementType::Pointer(_) => ("ptr".into(), None),
        },
        Type::Scalar(s) => (format_scalar(*s), Some(*s)),
        _ => ("?".into(), None),
    };

    // Compute total element count from shape.
    let total: i64 = if de.shape.is_empty() {
        1
    } else {
        de.shape.iter().product()
    };

    write!(out, "<{elem_name}: ").unwrap();

    if let Some(sc) = scalar {
        if total == 1 {
            // Single value.
            write!(out, "{}", decode_scalar(sc, &de.data, 0)).unwrap();
        } else {
            // Multi-element: print as flat list or nested.
            print_dense_elements_array(&de.shape, sc, &de.data, out);
        }
    } else {
        write!(out, "?").unwrap();
    }

    write!(out, ">").unwrap();

    // Print `: tile_type`.
    let tile_ty = if !result_types.is_empty() {
        format_type(&result_types[0])
    } else {
        format_type(&de.element_type)
    };
    write!(out, " : {tile_ty}").unwrap();
}

/// Decode a single scalar value from bytes at the given byte offset.
fn decode_scalar(sc: ScalarType, data: &[u8], byte_offset: usize) -> String {
    match sc {
        ScalarType::I1 => {
            if byte_offset < data.len() {
                format!("{}", data[byte_offset] & 1)
            } else {
                "0".into()
            }
        }
        ScalarType::I8 => {
            if byte_offset < data.len() {
                format!("{}", data[byte_offset] as i8)
            } else {
                "0".into()
            }
        }
        ScalarType::I16 => {
            if byte_offset + 2 <= data.len() {
                let v = i16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]);
                format!("{v}")
            } else {
                "0".into()
            }
        }
        ScalarType::I32 => {
            if byte_offset + 4 <= data.len() {
                let v = i32::from_le_bytes([
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                ]);
                format!("{v}")
            } else {
                "0".into()
            }
        }
        ScalarType::I64 => {
            if byte_offset + 8 <= data.len() {
                let v = i64::from_le_bytes([
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                    data[byte_offset + 4],
                    data[byte_offset + 5],
                    data[byte_offset + 6],
                    data[byte_offset + 7],
                ]);
                format!("{v}")
            } else {
                "0".into()
            }
        }
        ScalarType::F16 => {
            if byte_offset + 2 <= data.len() {
                let bits = u16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]);
                let v = half::f16::from_bits(bits);
                format_float(f64::from(v.to_f32()))
            } else {
                "0.0".into()
            }
        }
        ScalarType::BF16 => {
            if byte_offset + 2 <= data.len() {
                let bits = u16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]);
                let v = half::bf16::from_bits(bits);
                format_float(f64::from(v.to_f32()))
            } else {
                "0.0".into()
            }
        }
        ScalarType::F32 | ScalarType::TF32 => {
            if byte_offset + 4 <= data.len() {
                let v = f32::from_le_bytes([
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                ]);
                format_float(f64::from(v))
            } else {
                "0.0".into()
            }
        }
        ScalarType::F64 => {
            if byte_offset + 8 <= data.len() {
                let v = f64::from_le_bytes([
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                    data[byte_offset + 4],
                    data[byte_offset + 5],
                    data[byte_offset + 6],
                    data[byte_offset + 7],
                ]);
                format_float(v)
            } else {
                "0.0".into()
            }
        }
        ScalarType::I4
        | ScalarType::F4E2M1FN
        | ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E5M3FNU
        | ScalarType::F8E8M0FNU => {
            if byte_offset < data.len() {
                format!("{}", data[byte_offset])
            } else {
                "0".into()
            }
        }
    }
}

/// Format a float value, ensuring a decimal point is present.
fn format_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Print a (possibly nested) array of dense elements.
fn print_dense_elements_array(shape: &[i64], sc: ScalarType, data: &[u8], out: &mut String) {
    let bw = sc.byte_width();
    let total: i64 = shape.iter().product();

    if shape.len() <= 1 {
        // 1-D or scalar: flat list.
        let count = if shape.is_empty() {
            1
        } else {
            shape[0] as usize
        };
        // Check if all elements are the same (splat).
        if count > 1 && is_splat(sc, data, count) {
            write!(out, "{}", decode_scalar(sc, data, 0)).unwrap();
        } else {
            for i in 0..count {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{}", decode_scalar(sc, data, i * bw)).unwrap();
            }
        }
    } else {
        // Multi-dimensional: recurse.
        let inner_size: i64 = shape[1..].iter().product();
        let inner_bytes = inner_size as usize * bw;
        let n = shape[0] as usize;

        // Check splat at top level.
        if n > 1 && total > 1 && is_splat(sc, data, total as usize) {
            write!(out, "{}", decode_scalar(sc, data, 0)).unwrap();
        } else {
            write!(out, "[").unwrap();
            for i in 0..n {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                let start = i * inner_bytes;
                let end = start + inner_bytes;
                let slice = &data[start..end.min(data.len())];
                if shape.len() == 2 {
                    write!(out, "[").unwrap();
                    let count = shape[1] as usize;
                    for j in 0..count {
                        if j > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "{}", decode_scalar(sc, slice, j * bw)).unwrap();
                    }
                    write!(out, "]").unwrap();
                } else {
                    print_dense_elements_array(&shape[1..], sc, slice, out);
                }
            }
            write!(out, "]").unwrap();
        }
    }
}

/// Check if all elements are the same value.
fn is_splat(sc: ScalarType, data: &[u8], count: usize) -> bool {
    let bw = sc.byte_width();
    if data.len() < count * bw {
        return false;
    }
    let first = &data[..bw];
    for i in 1..count {
        if &data[i * bw..(i + 1) * bw] != first {
            return false;
        }
    }
    true
}

/// Format a DenseI32ArrayAttr (stored as an Array of Integers).
fn format_dense_i32_array(attr: &Attribute) -> String {
    match attr {
        Attribute::Array(arr) => {
            let elems: Vec<_> = arr
                .iter()
                .map(|a| match a {
                    Attribute::Integer(v, _) => v.to_string(),
                    _ => "?".into(),
                })
                .collect();
            format!("[{}]", elems.join(", "))
        }
        _ => format_attr(attr).to_string(),
    }
}

// =========================================================================
// Value result lookup
// =========================================================================

/// Find all Values that are results of a given operation.
fn find_result_values(module: &Module, op_id: OpId) -> Vec<Value> {
    use super::value::ValueProducer;
    let num_results = module.op(op_id).result_types.len();
    let mut results = Vec::with_capacity(num_results);
    for (i, vd) in module.values.iter().enumerate() {
        if let ValueProducer::OpResult { op, .. } = vd.producer {
            if op == op_id {
                results.push(Value(i as u32));
            }
        }
    }
    results.sort_by_key(|v| {
        if let ValueProducer::OpResult { result_index, .. } = module.value_data(*v).producer {
            result_index
        } else {
            0
        }
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{append_op, build_single_block_region, OpBuilder};
    use crate::bytecode::Opcode;
    use crate::ir::location::Location;
    use crate::ir::{FuncType, ScalarType, TileElementType, TileType};

    #[test]
    fn mlir_text_contains_op_names() {
        let mut module = Module::new("test_module");
        let _f32_ty = Type::Scalar(ScalarType::F32);
        let tile_ty = Type::Tile(TileType {
            shape: vec![128],
            element_type: TileElementType::Scalar(ScalarType::F32),
        });
        let func_ty = Type::Func(FuncType {
            inputs: vec![tile_ty.clone()],
            results: vec![],
        });

        // Build a function with addf and return.
        let (region_id, block_id, args) =
            build_single_block_region(&mut module, std::slice::from_ref(&tile_ty));

        let (add_id, add_results) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[0])
            .result(tile_ty.clone())
            .attr("rounding_mode", crate::ir::Attribute::i32(0))
            .build(&mut module);
        append_op(&mut module, block_id, add_id);

        let (sqrt_id, _) = OpBuilder::new(Opcode::Sqrt, Location::Unknown)
            .operand(add_results[0])
            .result(tile_ty.clone())
            .build(&mut module);
        append_op(&mut module, block_id, sqrt_id);

        let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
        append_op(&mut module, block_id, ret_id);

        let (entry_id, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", crate::ir::Attribute::String("my_kernel".into()))
            .attr("function_type", crate::ir::Attribute::Type(func_ty))
            .region(region_id)
            .build(&mut module);
        module.functions.push(entry_id);

        let text = module.to_mlir_text();
        println!("{text}");

        // These are the same kinds of substring checks the cutile tests use.
        assert!(text.contains("addf"), "expected addf in output");
        assert!(text.contains("sqrt"), "expected sqrt in output");
        assert!(text.contains("return"), "expected return in output");
        assert!(text.contains("entry"), "expected entry in output");
        assert!(text.contains("@my_kernel"), "expected kernel name");
        assert!(text.contains("@test_module"), "expected module name");
        assert!(text.contains("%"), "expected value references");
        assert!(text.contains("f32"), "expected f32 type");
    }
}
