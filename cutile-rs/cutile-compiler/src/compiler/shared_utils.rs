/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure-Rust utility functions and types used by compiler2.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use crate::syn_utils::get_ident_from_path_expr;
use half::{bf16, f16};
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::{BTreeSet, HashSet};
use std::fmt::{Debug, LowerHex};
use std::hash::Hash;
use syn::{Expr, Pat, Stmt};

use super::_value::{CompilerContext, TileRustValue};

// ---------------------------------------------------------------------------
// Stack management constants
// ---------------------------------------------------------------------------

/// Minimum remaining stack space before growing (4 MiB).
///
/// Large kernels with nested control flow and inlined DSL helpers can enter
/// non-trivial work just below a `stacker::maybe_grow` boundary. Growing before
/// the final MiB avoids finite deep lowering paths exhausting the native stack.
pub(crate) const STACK_RED_ZONE: usize = 4 * 1024 * 1024;
/// Size of each new stack segment when growth is needed (10 MiB).
pub(crate) const STACK_GROW_SIZE: usize = 10 * 1024 * 1024;

/// Maximum rank of a mapped partition (`MappedPartitionMut` / `iter_indices`).
/// Matches the DSL's `variadic_struct(N = 6)` expansion ceiling.
pub(crate) const MAX_MAPPED_PARTITION_RANK: usize = 6;

/// Map-shape sentinel marking an owned axis (mirrors `cutile::tensor::OWNED`).
///
/// An owned axis is not traversed by the mapped work-item stream: each stream
/// item owns the axis's full extent (subtensor-per-CTA exclusivity), and
/// in-kernel loops traverse it with bounds-proven indices.
pub(crate) const OWNED_MAP_DIM: i32 = 0;

// ---------------------------------------------------------------------------
// AtomicMode
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
/// Supported atomic read-modify-write modes.
pub enum AtomicMode {
    And = 0,
    Or = 1,
    Xor = 2,
    Add = 3,
    AddF = 4,
    Max = 5,
    Min = 6,
    UMax = 7,
    UMin = 8,
    XChg = 9,
}

// ---------------------------------------------------------------------------
// ElementTypePrefix
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
/// Whether an element type is floating-point or integer.
pub enum ElementTypePrefix {
    Float,
    Integer,
}

impl ElementTypePrefix {
    /// Determines the prefix from a CUDA Tile element type string (e.g. `"f32"` -> `Float`).
    pub fn new(cuda_elem_ty_str: &str) -> Result<Self, JITError> {
        match super::_type::scalar_from_name(cuda_elem_ty_str) {
            Some(s) if s.is_float() => Ok(ElementTypePrefix::Float),
            Some(_) => Ok(ElementTypePrefix::Integer),
            None => SourceLocation::unknown()
                .jit_error_result(&format!("unsupported element type `{cuda_elem_ty_str}`")),
        }
    }
}

impl AtomicMode {
    /// Parses an atomic mode from the ZST type identifier (e.g. `AddF`),
    /// validating compatibility with the element type.
    pub fn new(mode: &str, elem_ty_prefix: ElementTypePrefix) -> Result<Self, JITError> {
        let result = match mode {
            "And" => AtomicMode::And,
            "Or" => AtomicMode::Or,
            "Xor" => AtomicMode::Xor,
            "Add" => AtomicMode::Add,
            "AddF" => AtomicMode::AddF,
            "Max" => AtomicMode::Max,
            "Min" => AtomicMode::Min,
            "Umax" => AtomicMode::UMax,
            "Umin" => AtomicMode::UMin,
            "Xchg" => AtomicMode::XChg,
            _ => return SourceLocation::unknown().jit_error_result(
                &format!("invalid atomic mode `{mode}`; valid modes are: And, Or, Xor, Add, AddF, Max, Min, Umax, Umin, Xchg"),
            ),
        };
        if elem_ty_prefix == ElementTypePrefix::Float
            && ![AtomicMode::XChg, AtomicMode::AddF].contains(&result)
        {
            return SourceLocation::unknown().jit_error_result(&format!(
                "float types only support `Xchg` and `AddF` atomic modes, got `{:?}`",
                result
            ));
        }
        Ok(result)
    }
}

// Re-export from bounds.rs (canonical location shared with old compiler).
pub use crate::bounds::{get_binary_op_from_op_str, get_tile_bop_from_rust_bop, TileBinaryOp};

// ---------------------------------------------------------------------------
// Constant hex encoding
// ---------------------------------------------------------------------------

fn format_hex<T: LowerHex>(val: T) -> String {
    format!("0x{:x}", val)
}

trait Float {
    fn to_hex(&self) -> String;
    fn zero() -> Self;
    fn one() -> Self;
    fn negative_infinity() -> Self;
    fn positive_infinity() -> Self;
    fn e() -> Self;
}

impl Float for f16 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f16 {
        f16::ZERO
    }
    fn one() -> f16 {
        f16::ONE
    }
    fn negative_infinity() -> f16 {
        f16::NEG_INFINITY
    }
    fn positive_infinity() -> f16 {
        f16::INFINITY
    }
    fn e() -> f16 {
        f16::E
    }
}

impl Float for bf16 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> bf16 {
        bf16::ZERO
    }
    fn one() -> bf16 {
        bf16::ONE
    }
    fn negative_infinity() -> bf16 {
        bf16::NEG_INFINITY
    }
    fn positive_infinity() -> bf16 {
        bf16::INFINITY
    }
    fn e() -> bf16 {
        bf16::E
    }
}

impl Float for f32 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f32 {
        0.0f32
    }
    fn one() -> f32 {
        1.0f32
    }
    fn negative_infinity() -> f32 {
        f32::NEG_INFINITY
    }
    fn positive_infinity() -> f32 {
        f32::INFINITY
    }
    fn e() -> f32 {
        std::f32::consts::E
    }
}

impl Float for f64 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f64 {
        0.0f64
    }
    fn one() -> f64 {
        1.0f64
    }
    fn negative_infinity() -> f64 {
        f64::NEG_INFINITY
    }
    fn positive_infinity() -> f64 {
        f64::INFINITY
    }
    fn e() -> f64 {
        std::f64::consts::E
    }
}

trait Integer
where
    Self: LowerHex,
{
    fn to_hex(&self) -> String {
        format_hex(self)
    }
    fn zero() -> Self;
    fn one() -> Self;
    fn min() -> Self;
    fn max() -> Self;
}

impl Integer for i32 {
    fn zero() -> i32 {
        0i32
    }
    fn one() -> i32 {
        1i32
    }
    fn min() -> i32 {
        i32::MIN
    }
    fn max() -> i32 {
        i32::MAX
    }
}
impl Integer for i64 {
    fn zero() -> i64 {
        0i64
    }
    fn one() -> i64 {
        1i64
    }
    fn min() -> i64 {
        i64::MIN
    }
    fn max() -> i64 {
        i64::MAX
    }
}
impl Integer for u32 {
    fn zero() -> u32 {
        0u32
    }
    fn one() -> u32 {
        1u32
    }
    fn min() -> u32 {
        u32::MIN
    }
    fn max() -> u32 {
        u32::MAX
    }
}
impl Integer for u64 {
    fn zero() -> u64 {
        0u64
    }
    fn one() -> u64 {
        1u64
    }
    fn min() -> u64 {
        u64::MIN
    }
    fn max() -> u64 {
        u64::MAX
    }
}

fn get_float_const<T: Float>(const_str: &str) -> Result<String, JITError> {
    match const_str {
        "zero" => Ok(T::zero().to_hex()),
        "one" => Ok(T::one().to_hex()),
        "min" => Ok(T::negative_infinity().to_hex()),
        "max" => Ok(T::positive_infinity().to_hex()),
        "e" => Ok(T::e().to_hex()),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("Unsupported float constant type {}.", const_str)),
    }
}

fn get_integer_const<T: Integer>(const_str: &str) -> Result<String, JITError> {
    match const_str {
        "zero" => Ok(T::zero().to_hex()),
        "one" => Ok(T::one().to_hex()),
        "min" => Ok(T::min().to_hex()),
        "max" => Ok(T::max().to_hex()),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("Unsupported integer constant type {}.", const_str)),
    }
}

/// Returns the hex-encoded constant string for a typed constant name (e.g. `"zero"`, `"one"`).
pub fn get_const_hex(rust_element_type_str: &str, const_str: &str) -> Result<String, JITError> {
    // Storage-only scalar wrappers use their raw byte encodings. In particular,
    // scale formats must not pass through the f32 constant encoding path.
    if const_str == "zero"
        && matches!(
            rust_element_type_str,
            "i4" | "f4e2m1fn" | "f4e2m1fnx2" | "f8e4m3fn" | "f8e5m2" | "f8e8m0fnu" | "f8e5m3fnu"
        )
    {
        return Ok("0".into());
    }
    if const_str == "one" {
        let bits = match rust_element_type_str {
            "i4" => Some(1),
            "f4e2m1fn" => Some(2),
            "f4e2m1fnx2" => Some(0x22),
            "f8e4m3fn" => Some(0x38),
            "f8e5m2" => Some(0x3c),
            "f8e8m0fnu" => Some(0x7f),
            "f8e5m3fnu" => Some(0x78),
            _ => None,
        };
        if let Some(bits) = bits {
            return Ok(bits.to_string());
        }
    }
    match rust_element_type_str {
        "bf16" => get_float_const::<bf16>(const_str),
        "f16" => get_float_const::<f16>(const_str),
        "f32" => get_float_const::<f32>(const_str),
        "f64" => get_float_const::<f64>(const_str),
        "i32" => get_integer_const::<i32>(const_str),
        "i64" => get_integer_const::<i64>(const_str),
        "u32" => get_integer_const::<u32>(const_str),
        "u64" => get_integer_const::<u64>(const_str),
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "Unsupported constant type {} {}.",
            rust_element_type_str, const_str
        )),
    }
}

// ---------------------------------------------------------------------------
// String literal / option arg extraction
// ---------------------------------------------------------------------------

/// Reads the last path-segment ident from a ZST type-as-value expression.
///
/// Examples: `atomic::AddF` -> `"AddF"`, `Enabled` -> `"Enabled"`.
/// Used to resolve attribute selectors that the DSL surfaces as ZST type
/// arguments (`mode: atomic::Mode`, `memory_ordering: ordering::Mode`, etc.).
pub fn extract_zst_type_name(expr: &syn::Expr, param_name: &str) -> Result<String, JITError> {
    use syn::Expr;
    match expr {
        Expr::Path(path) => Ok(path.path.segments.last().unwrap().ident.to_string()),
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "`{param_name}` must be a unit-struct type-as-value path, got `{}`",
            expr.to_token_stream()
        )),
    }
}

pub fn padding_zst_value(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    let ident = path.path.segments.last()?.ident.to_string();
    match ident.as_str() {
        "Zero" => Some("zero".to_string()),
        "NegZero" => Some("neg_zero".to_string()),
        "Nan" => Some("nan".to_string()),
        "PosInf" => Some("pos_inf".to_string()),
        "NegInf" => Some("neg_inf".to_string()),
        _ => None,
    }
}

pub fn zst_type_name(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    Some(path.path.segments.last()?.ident.to_string())
}

/// Extracts a string literal value from an expression, handling both direct literals
/// and variables that were bound from string literals.
pub fn extract_string_literal(
    expr: &syn::Expr,
    param_name: &str,
    ctx: &CompilerContext,
) -> Result<String, JITError> {
    use syn::{Expr, ExprLit, Lit};

    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        Expr::Path(path_expr) => {
            let var_name = path_expr.path.segments.last().unwrap().ident.to_string();
            if let Some(val) = ctx.vars.get(&var_name) {
                if let Some(Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                })) = &val.string_literal
                {
                    return Ok(s.value());
                }
            }
            SourceLocation::unknown().jit_error_result(&format!(
                "`{param_name}` must be a string literal, but got variable `{var_name}`; \
                     ensure string literals are passed directly",
            ))
        }
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "`{param_name}` must be a string literal, got `{}`",
            expr.to_token_stream()
        )),
    }
}

/// Helper to resolve compile-time optional argument.
/// Returns the inner expression if it is Some(expr), or None if it is None.
pub fn resolve_option_arg(expr: &syn::Expr, ctx: &CompilerContext) -> Option<syn::Expr> {
    use syn::Expr;
    if let Expr::Call(call) = expr {
        if let Expr::Path(path) = &*call.func {
            if path.path.segments.last().unwrap().ident == "Some" {
                return call.args.first().cloned();
            }
        }
    } else if let Expr::Path(path) = expr {
        if path.path.segments.len() == 1 && path.path.segments.last().unwrap().ident == "None" {
            return None;
        }
        let var_name = path.path.segments.last().unwrap().ident.to_string();
        if let Some(val) = ctx.vars.get(&var_name) {
            if let Some(variant) = &val.enum_variant {
                return match variant.as_str() {
                    "Some" => val.enum_payload.as_deref().cloned(),
                    "None" => None,
                    _ => None,
                };
            }
            if let Some(ast) = &val.string_literal {
                if let Expr::Call(call) = ast {
                    if let Expr::Path(path) = &*call.func {
                        if path.path.segments.last().unwrap().ident == "Some" {
                            return call.args.first().cloned();
                        }
                    }
                } else if let Expr::Path(path) = ast {
                    if path.path.segments.len() == 1
                        && path.path.segments.last().unwrap().ident == "None"
                    {
                        return None;
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Variable mutation analysis
// ---------------------------------------------------------------------------

pub(super) fn collect_pattern_bindings(pat: &Pat, names: &mut Vec<String>) -> Result<(), JITError> {
    match pat {
        Pat::Ident(ident) => {
            names.push(ident.ident.to_string());
            if let Some((_at, subpat)) = &ident.subpat {
                collect_pattern_bindings(subpat, names)?;
            }
            Ok(())
        }
        Pat::Type(pat_type) => collect_pattern_bindings(&pat_type.pat, names),
        Pat::Paren(paren) => collect_pattern_bindings(&paren.pat, names),
        Pat::Reference(reference) => collect_pattern_bindings(&reference.pat, names),
        Pat::Tuple(tuple) => {
            for elem in &tuple.elems {
                collect_pattern_bindings(elem, names)?;
            }
            Ok(())
        }
        Pat::Slice(slice) => {
            for elem in &slice.elems {
                collect_pattern_bindings(elem, names)?;
            }
            Ok(())
        }
        Pat::Struct(pat_struct) => {
            for field in &pat_struct.fields {
                collect_pattern_bindings(&field.pat, names)?;
            }
            Ok(())
        }
        Pat::TupleStruct(tuple_struct) => {
            for elem in &tuple_struct.elems {
                collect_pattern_bindings(elem, names)?;
            }
            Ok(())
        }
        Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                collect_pattern_bindings(case, names)?;
            }
            Ok(())
        }
        Pat::Wild(_) | Pat::Rest(_) => Ok(()),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("Local pattern type not supported {:#?}", pat)),
    }
}

/// Collects the names of variables assigned (mutated) in a block that were defined outside it.
pub fn collect_mutated_variables_from_block(
    block: &syn::Block,
) -> Result<BTreeSet<String>, JITError> {
    let mut local_vars: HashSet<String> = HashSet::new();
    let mut result: BTreeSet<String> = BTreeSet::new();
    for statement in block.stmts.iter() {
        match statement {
            Stmt::Local(local) => {
                let mut var_names: Vec<String> = vec![];
                collect_pattern_bindings(&local.pat, &mut var_names)?;
                local_vars.extend(var_names);
            }
            Stmt::Expr(Expr::Assign(assign_expr), _) => {
                let var_name: String = match &*assign_expr.left {
                    Expr::Path(path_expr) => get_ident_from_path_expr(path_expr).to_string(),
                    _ => {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "LHS assign expression not implemented {:#?}.",
                            assign_expr.left
                        ));
                    }
                };
                if !local_vars.contains(&var_name) {
                    result.insert(var_name);
                }
            }
            // Recurse into control-flow expressions to find nested mutations.
            Stmt::Expr(Expr::ForLoop(for_expr), _) => {
                let inner = collect_mutated_variables(for_expr)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(Expr::While(while_expr), _) => {
                let inner = collect_mutated_variables_from_block(&while_expr.body)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(Expr::Loop(loop_expr), _) => {
                let inner = collect_mutated_variables_from_block(&loop_expr.body)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(expr, _) => {
                let inner = collect_mutated_variables_from_expr(expr)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            _ => continue,
        }
    }
    Ok(result)
}

/// Collects mutated outer-scope variables from an expression.
pub fn collect_mutated_variables_from_expr(expr: &Expr) -> Result<BTreeSet<String>, JITError> {
    match expr {
        Expr::Assign(assign_expr) => {
            let var_name: String = match &*assign_expr.left {
                Expr::Path(path_expr) => get_ident_from_path_expr(path_expr).to_string(),
                _ => {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "LHS assign expression not implemented {:#?}.",
                        assign_expr.left
                    ));
                }
            };
            Ok(BTreeSet::from([var_name]))
        }
        Expr::Block(block_expr) => collect_mutated_variables_from_block(&block_expr.block),
        Expr::ForLoop(for_expr) => collect_mutated_variables(for_expr),
        Expr::While(while_expr) => collect_mutated_variables_from_block(&while_expr.body),
        Expr::Loop(loop_expr) => collect_mutated_variables_from_block(&loop_expr.body),
        Expr::If(if_expr) => {
            let mut result = collect_mutated_variables_from_block(&if_expr.then_branch)?;
            if let Some((_else, else_expr)) = &if_expr.else_branch {
                result.extend(collect_mutated_variables_from_expr(else_expr)?);
            }
            Ok(result)
        }
        _ => Ok(BTreeSet::new()),
    }
}

/// Does a loop body contain an early exit — `continue`, `break`, or `return`
/// — that targets *this* loop (or the enclosing function)? Nested loops are
/// not descended into: their `continue`/`break` only shortens their own
/// iteration. Closures are opaque for the same reason.
///
/// Used to mark [`super::_value::LoopFrame::has_early_exit`]: a body that can
/// skip the rest of an iteration does not execute every access on every
/// iteration, so no bounds check may be hoisted out of it.
pub fn block_has_early_exit(block: &syn::Block) -> bool {
    use syn::visit::Visit;
    struct Finder {
        found: bool,
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_expr_continue(&mut self, _: &'ast syn::ExprContinue) {
            self.found = true;
        }
        fn visit_expr_break(&mut self, _: &'ast syn::ExprBreak) {
            self.found = true;
        }
        fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
            self.found = true;
        }
        fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {}
        fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {}
        fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {}
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut finder = Finder { found: false };
    finder.visit_block(block);
    finder.found
}

/// Collects mutated outer-scope variables from a for-loop body.
pub fn collect_mutated_variables(
    for_expr: &syn::ExprForLoop,
) -> Result<BTreeSet<String>, JITError> {
    let mut result = collect_mutated_variables_from_block(&for_expr.body)?;
    let mut loop_vars = Vec::new();
    collect_pattern_bindings(&for_expr.pat, &mut loop_vars)?;
    for loop_var in loop_vars {
        result.remove(&loop_var);
    }
    Ok(result)
}

/// Set a variable's ordering token directly. A no-op if the variable is absent
/// or carries no token field. Used to publish a loop's accumulated token to the
/// tensor and to views written in the loop body.
pub fn set_view_token(var: &str, token: cutile_ir::ir::Value, ctx: &mut CompilerContext) {
    let Some(value) = ctx.vars.get(var) else {
        return;
    };
    let mut new_value = value.clone();
    let Some(meta) = new_value.type_meta.as_mut() else {
        return;
    };
    let Some(field) = meta.fields.get_mut("token") else {
        return;
    };
    field.value = Some(token);
    ctx.vars.insert(var.to_string(), new_value);
}

/// Method names that write to a mutable partition/tensor view (advance its
/// ordering token). Used to find which resources a loop body stores to, so the
/// loop can thread and publish their tokens (token-ordering across the loop
/// scope boundary).
const STORE_METHOD_NAMES: &[&str] = &["store", "store_index"];

/// A `.store(...)` call found in a loop body: the receiver view and whether its
/// index varies with the loop variable (distinct per iteration → the writes are
/// disjoint and may fork; otherwise the same address is written each iteration
/// and the writes must be serialized).
pub struct StoreCall {
    pub receiver: String,
    pub index_distinct: bool,
}

/// Collects the `.store(...)` / `.store_index(...)` calls anywhere in `block`
/// (descending through nested control flow and `unsafe` blocks), with the
/// receiver view and whether the store's index references `loop_var`. When
/// `loop_var` is `None` (no simple loop variable), indices are treated as
/// non-distinct (conservative: serialize).
pub fn collect_store_calls(block: &syn::Block, loop_var: Option<&str>) -> Vec<StoreCall> {
    let mut out = Vec::new();
    for stmt in &block.stmts {
        collect_store_calls_stmt(stmt, loop_var, &mut out);
    }
    out
}

fn collect_store_calls_stmt(stmt: &Stmt, loop_var: Option<&str>, out: &mut Vec<StoreCall>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                collect_store_calls_expr(&init.expr, loop_var, out);
            }
        }
        Stmt::Expr(expr, _) => collect_store_calls_expr(expr, loop_var, out),
        _ => {}
    }
}

fn collect_store_calls_expr(expr: &Expr, loop_var: Option<&str>, out: &mut Vec<StoreCall>) {
    match expr {
        Expr::MethodCall(mc) => {
            if STORE_METHOD_NAMES.contains(&mc.method.to_string().as_str()) {
                if let Expr::Path(path) = &*mc.receiver {
                    if let Some(ident) = path.path.get_ident() {
                        // The index is the second argument: `store(tile, index)`.
                        let index_distinct = match (loop_var, mc.args.iter().nth(1)) {
                            (Some(lv), Some(index)) => expr_references_ident(index, lv),
                            _ => false,
                        };
                        out.push(StoreCall {
                            receiver: ident.to_string(),
                            index_distinct,
                        });
                    }
                }
            }
            collect_store_calls_expr(&mc.receiver, loop_var, out);
            for arg in &mc.args {
                collect_store_calls_expr(arg, loop_var, out);
            }
        }
        Expr::Block(b) => collect_store_calls_block(&b.block, loop_var, out),
        Expr::Unsafe(u) => collect_store_calls_block(&u.block, loop_var, out),
        Expr::ForLoop(f) => collect_store_calls_block(&f.body, loop_var, out),
        Expr::While(w) => collect_store_calls_block(&w.body, loop_var, out),
        Expr::Loop(l) => collect_store_calls_block(&l.body, loop_var, out),
        Expr::If(i) => {
            collect_store_calls_block(&i.then_branch, loop_var, out);
            if let Some((_, else_expr)) = &i.else_branch {
                collect_store_calls_expr(else_expr, loop_var, out);
            }
        }
        Expr::Call(c) => {
            for arg in &c.args {
                collect_store_calls_expr(arg, loop_var, out);
            }
        }
        _ => {}
    }
}

fn collect_store_calls_block(block: &syn::Block, loop_var: Option<&str>, out: &mut Vec<StoreCall>) {
    for stmt in &block.stmts {
        collect_store_calls_stmt(stmt, loop_var, out);
    }
}

/// Whether `expr` syntactically references the identifier `name` anywhere. Used
/// to decide whether a store's index varies with the loop variable (fork —
/// distinct per iteration) or not (serialize — a constant/repeated address must
/// be ordered).
pub fn expr_references_ident(expr: &Expr, name: &str) -> bool {
    let mut found = false;
    references_ident_expr(expr, name, &mut found);
    found
}

fn references_ident_expr(expr: &Expr, name: &str, found: &mut bool) {
    if *found {
        return;
    }
    if let Expr::Path(path) = expr {
        if path.path.is_ident(name) {
            *found = true;
            return;
        }
    }
    // Descend into the common sub-expression carriers for index expressions.
    match expr {
        Expr::MethodCall(mc) => {
            references_ident_expr(&mc.receiver, name, found);
            for arg in &mc.args {
                references_ident_expr(arg, name, found);
            }
        }
        Expr::Call(c) => {
            references_ident_expr(&c.func, name, found);
            for arg in &c.args {
                references_ident_expr(arg, name, found);
            }
        }
        Expr::Binary(b) => {
            references_ident_expr(&b.left, name, found);
            references_ident_expr(&b.right, name, found);
        }
        Expr::Unary(u) => references_ident_expr(&u.expr, name, found),
        Expr::Paren(p) => references_ident_expr(&p.expr, name, found),
        Expr::Group(g) => references_ident_expr(&g.expr, name, found),
        Expr::Reference(r) => references_ident_expr(&r.expr, name, found),
        Expr::Tuple(t) => {
            for e in &t.elems {
                references_ident_expr(e, name, found);
            }
        }
        Expr::Array(a) => {
            for e in &a.elems {
                references_ident_expr(e, name, found);
            }
        }
        Expr::Cast(c) => references_ident_expr(&c.expr, name, found),
        Expr::Index(i) => {
            references_ident_expr(&i.expr, name, found);
            references_ident_expr(&i.index, name, found);
        }
        Expr::Field(f) => references_ident_expr(&f.base, name, found),
        _ => {}
    }
}

/// Collects mutated outer-scope variables from a while-loop body.
pub fn collect_mutated_variables_while(
    while_expr: &syn::ExprWhile,
) -> Result<BTreeSet<String>, JITError> {
    collect_mutated_variables_from_block(&while_expr.body)
}

/// Collects mutated outer-scope variables from a loop body.
pub fn collect_mutated_variables_loop(
    loop_expr: &syn::ExprLoop,
) -> Result<BTreeSet<String>, JITError> {
    collect_mutated_variables_from_block(&loop_expr.body)
}

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------

/// Removes duplicate elements from a vector while preserving order.
pub fn dedup<T: Hash + Eq + Clone>(v: &mut Vec<T>) {
    let mut set = HashSet::new();
    v.retain(|x| set.insert(x.clone()));
}

/// Parses a comma-separated token stream into a list of `syn::Expr`.
pub fn parse_list_of_expr(tokens: TokenStream) -> Result<Vec<Expr>, JITError> {
    let mut args: Vec<Expr> = vec![];
    let mut arg_expr: Vec<TokenTree> = vec![];
    for token in tokens.clone().into_iter() {
        match &token {
            TokenTree::Literal(_lit) => {
                arg_expr.push(token.clone());
            }
            TokenTree::Ident(_ident) => {
                arg_expr.push(token.clone());
            }
            TokenTree::Punct(punct) => {
                if punct.as_char() == ',' {
                    if !arg_expr.is_empty() {
                        let expr =
                            syn::parse2::<syn::Expr>(arg_expr.into_iter().collect()).unwrap();
                        args.push(expr);
                    }
                    arg_expr = vec![];
                } else {
                    arg_expr.push(token.clone());
                }
            }
            _ => {
                return SourceLocation::unknown()
                    .jit_error_result(&format!("unexpected token `{}` in expression list", token));
            }
        }
    }
    if !arg_expr.is_empty() {
        let expr = syn::parse2::<syn::Expr>(arg_expr.into_iter().collect()).unwrap();
        args.push(expr);
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Token / type_meta helpers
// ---------------------------------------------------------------------------

/// Updates the ordering token in a variable's type metadata.
pub fn update_token(
    var_arg: &Expr,
    new_token: cutile_ir::ir::Value,
    ctx: &mut CompilerContext,
) -> Result<Option<TileRustValue>, JITError> {
    use crate::syn_utils::get_ident_from_expr;
    let Some(var_arg_ident) = get_ident_from_expr(var_arg) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "expected a variable name, got `{}`",
            var_arg.to_token_stream()
        ));
    };
    let var_name = var_arg_ident.to_string();
    let Some(old_value) = ctx.vars.get(var_name.as_str()) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "undefined variable `{var_name}` when updating token"
        ));
    };
    let mut new_value = old_value.clone();
    let Some(new_type_meta) = &mut new_value.type_meta else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` does not have associated type metadata (expected a view type)"
        ));
    };
    let Some(new_token_value) = new_type_meta.fields.get_mut("token") else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` is missing a `token` field (expected a view with an ordering token)"
        ));
    };
    new_token_value.value = Some(new_token);
    Ok(ctx.vars.insert(var_name, new_value))
}

/// Propagate a resource's current ordering token up its borrow link to the root
/// tensor it views. A view roots at its tensor (`tensor_origin`), so advancing
/// the view's token must advance the tensor's: a *later* view of the same tensor
/// seeds its token from `get_tensor_token`, and this is what makes that later
/// view happen-after this one's writes (the epoch boundary of the token model).
///
/// This runs at scope boundaries (where the view and its root tensor are both in
/// scope), not at the store site — inside a method frame the root tensor is out
/// of scope, so the propagation is deferred to the boundary that reconciles the
/// view back to its caller. A no-op when the resource has no root, roots at
/// itself (a tensor), or the root is not a live variable here.
pub fn propagate_token_to_root(resource_var: &str, ctx: &mut CompilerContext) {
    let Some(resource) = ctx.vars.get(resource_var) else {
        return;
    };
    let Some(root) = resource.tensor_origin.clone() else {
        return;
    };
    if root == resource_var {
        return; // A tensor roots at itself; there is nothing above it to walk to.
    }
    let token = resource
        .type_meta
        .as_ref()
        .and_then(|meta| meta.fields.get("token"))
        .and_then(|field| field.value);
    let Some(token) = token else {
        return;
    };
    let Some(root_value) = ctx.vars.get(&root) else {
        return;
    };
    let mut new_root = root_value.clone();
    let Some(meta) = new_root.type_meta.as_mut() else {
        return;
    };
    let Some(token_field) = meta.fields.get_mut("token") else {
        return;
    };
    token_field.value = Some(token);
    ctx.vars.insert(root, new_root);
}

/// Retrieves the ordering token from a variable expression's type metadata.
pub fn get_token_from_expr(
    var_arg: &Expr,
    ctx: &mut CompilerContext,
) -> Result<TileRustValue, JITError> {
    use crate::syn_utils::get_ident_from_expr;
    let Some(var_arg_ident) = get_ident_from_expr(var_arg) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "expected a variable name, got `{}`",
            var_arg.to_token_stream()
        ));
    };
    let var_name = var_arg_ident.to_string();
    get_token(var_name.as_str(), ctx)
}

/// Retrieves the ordering token from a named variable's type metadata.
pub fn get_token(var_name: &str, ctx: &mut CompilerContext) -> Result<TileRustValue, JITError> {
    let Some(value) = ctx.vars.get(var_name) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "undefined variable `{var_name}` when reading token"
        ));
    };
    let Some(value_type_meta) = &value.type_meta else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` does not have associated type metadata (expected a view type)"
        ));
    };
    let Some(value_token_value) = value_type_meta.fields.get("token") else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` is missing a `token` field (expected a view with an ordering token)"
        ));
    };
    Ok(value_token_value.clone())
}

/// Propagates type metadata changes from an inner block context to the outer block context.
pub fn update_outer_block_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    field_name: String,
) {
    let mut var_map = std::collections::HashMap::new();
    for var_name in outer_block_vars.var_keys() {
        var_map.insert(var_name.clone(), var_name.clone());
    }
    update_type_meta(inner_block_vars, outer_block_vars, &var_map, field_name);
}

/// Copies mutable type metadata fields from inner to outer context using a variable name mapping.
pub fn update_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    outer2inner_vars: &std::collections::HashMap<String, String>,
    _field_name: String,
) {
    use super::shared_types::Mutability;
    let outer_keys_ = outer_block_vars.var_keys();
    let outer_keys = outer_keys_
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    for outer_key in &outer_keys {
        let Some(outer_val) = outer_block_vars.vars.get(outer_key).cloned() else {
            continue;
        };
        let Some(inner_key) = outer2inner_vars.get(outer_key) else {
            continue;
        };
        let Some(inner_val) = inner_block_vars.vars.get(inner_key) else {
            continue;
        };
        // Ordering metadata can change through `&Tensor`, even when the
        // caller used `let mut`: an explicit `set_token` /
        // `set_tensor_token` installs an external ordering point (a PDL
        // wait) that every later view must start from. Compare view
        // identity, not names or mutability, and carry *only* explicit
        // installs. The completion token a load writes into its own view
        // (inside the inlined `Partition::load`) must not escape an
        // immutable binding: reads need no ordering among themselves, and
        // chaining a read-only partition's loads serializes them (a 35%
        // slowdown in a load-bound kernel). Writable bindings keep the
        // full metadata copy below, which carries their access tokens.
        if inner_val.value.is_some()
            && inner_val.value == outer_val.value
            && inner_block_vars.explicit_token_updates.contains(inner_key)
        {
            let inner_token = inner_val
                .type_meta
                .as_ref()
                .and_then(|meta| meta.fields.get("token").cloned());
            if let Some(inner_token) = inner_token {
                let mut new_val = outer_val.clone();
                if let Some(meta) = new_val.type_meta.as_mut() {
                    if meta.fields.contains_key("token") {
                        meta.fields.insert("token".to_string(), inner_token);
                        outer_block_vars.vars.insert(outer_key.clone(), new_val);
                        outer_block_vars
                            .explicit_token_updates
                            .insert(outer_key.clone());
                    }
                }
            }
        }
        if outer_val.mutability == Mutability::Mutable
            && inner_val.mutability == Mutability::Mutable
        {
            let mut new_val = outer_val;
            new_val.type_meta = inner_val.type_meta.clone();
            outer_block_vars.vars.insert(outer_key.clone(), new_val);
        }
    }
}
