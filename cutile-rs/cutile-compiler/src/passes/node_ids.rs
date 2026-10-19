/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable node ids for semantic expressions in compiler-owned syn bodies.
//!
//! The current compiler still emits from `syn`, so type-check side tables need
//! a way to refer back to individual source nodes without relying on token
//! strings. These ids are stable only within one cloned function body and pass
//! pipeline.
//!
//! IDs are assigned to source expressions that can produce a value or carry
//! semantic lowering metadata, such as call results, method selections, and
//! dispatch-wrapper rewrites. Syntax-only positions are left untagged:
//! expressions embedded in type syntax, call callees, assignment destinations,
//! and attribute metadata.

use syn::spanned::Spanned;
use syn::visit_mut::{self, VisitMut};
use syn::{Attribute, Expr, ExprLit, ItemFn, Lit, Meta};

const NODE_ID_ATTR: &str = "__cutile_node_id";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

pub fn assign_expr_ids(fn_item: &mut ItemFn) {
    let mut assigner = NodeIdAssigner { next: 0 };
    assigner.visit_block_mut(&mut fn_item.block);
}

pub fn assign_block_expr_ids(block: &mut syn::Block) {
    let mut assigner = NodeIdAssigner { next: 0 };
    assigner.visit_block_mut(block);
}

pub fn expr_id(expr: &Expr) -> Option<NodeId> {
    expr_attrs(expr)?.iter().find_map(node_id_from_attr)
}

pub fn set_expr_id(expr: &mut Expr, id: NodeId) {
    let span = expr.span();
    set_expr_id_spanned(expr, id, span);
}

/// Like [`set_expr_id`], with an explicit span for the marker attribute.
///
/// The attribute is the expression's first token, so `expr.span()` starts at
/// it from then on. For an expression assembled from tokens of more than one
/// source text (a lowered dispatch call: user-written receiver and arguments,
/// wrapper-body method name) `expr.span()` is not a usable anchor, because
/// `Span::join` across source texts fails; callers pass the span of the
/// expression being replaced instead.
pub(crate) fn set_expr_id_spanned(expr: &mut Expr, id: NodeId, span: proc_macro2::Span) {
    let Some(attrs) = expr_attrs_mut(expr) else {
        return;
    };
    attrs.retain(|attr| !attr.path().is_ident(NODE_ID_ATTR));
    let raw_id = syn::LitInt::new(&id.0.to_string(), span);
    attrs.push(syn::parse_quote_spanned!(span=> #[__cutile_node_id = #raw_id]));
}

struct NodeIdAssigner {
    next: u32,
}

impl NodeIdAssigner {
    fn fresh(&mut self) -> NodeId {
        let id = NodeId(self.next);
        self.next += 1;
        id
    }
}

impl VisitMut for NodeIdAssigner {
    fn visit_attribute_mut(&mut self, _attr: &mut Attribute) {
        // Internal expression ids must not recurse into attribute meta syntax.
    }

    fn visit_type_mut(&mut self, _ty: &mut syn::Type) {
        // Type syntax can contain expressions in const-generic arguments.
        // Mutating those expressions changes type token streams used by generic
        // inference and instantiation.
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if semantic_expr_attrs(expr).is_some() {
            let id = self.fresh();
            set_expr_id(expr, id);
        }

        match expr {
            Expr::Assign(assign) => {
                // The destination is binding syntax, not a value expression.
                self.visit_expr_mut(&mut assign.right);
            }
            Expr::Call(call) => {
                // The callee is name-resolution syntax in this DSL.
                for arg in &mut call.args {
                    self.visit_expr_mut(arg);
                }
            }
            _ => visit_mut::visit_expr_mut(self, expr),
        }
    }
}

fn node_id_from_attr(attr: &Attribute) -> Option<NodeId> {
    if !attr.path().is_ident(NODE_ID_ATTR) {
        return None;
    }
    let Meta::NameValue(name_value) = &attr.meta else {
        return None;
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Int(raw_id),
        ..
    }) = &name_value.value
    else {
        return None;
    };
    Some(NodeId(raw_id.base10_parse().ok()?))
}

fn semantic_expr_attrs(expr: &Expr) -> Option<&Vec<Attribute>> {
    if !has_semantic_node_id(expr) {
        return None;
    }
    expr_attrs(expr)
}

fn has_semantic_node_id(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Array(_)
            | Expr::Binary(_)
            | Expr::Block(_)
            | Expr::Call(_)
            | Expr::Cast(_)
            | Expr::Field(_)
            | Expr::Group(_)
            | Expr::If(_)
            | Expr::Index(_)
            | Expr::Lit(_)
            | Expr::Macro(_)
            | Expr::MethodCall(_)
            | Expr::Paren(_)
            | Expr::Path(_)
            | Expr::Reference(_)
            | Expr::Repeat(_)
            | Expr::Struct(_)
            | Expr::Tuple(_)
            | Expr::Unary(_)
            | Expr::Unsafe(_)
    )
}

fn expr_attrs(expr: &Expr) -> Option<&Vec<Attribute>> {
    match expr {
        Expr::Array(expr) => Some(&expr.attrs),
        Expr::Assign(expr) => Some(&expr.attrs),
        Expr::Async(expr) => Some(&expr.attrs),
        Expr::Await(expr) => Some(&expr.attrs),
        Expr::Binary(expr) => Some(&expr.attrs),
        Expr::Block(expr) => Some(&expr.attrs),
        Expr::Break(expr) => Some(&expr.attrs),
        Expr::Call(expr) => Some(&expr.attrs),
        Expr::Cast(expr) => Some(&expr.attrs),
        Expr::Closure(expr) => Some(&expr.attrs),
        Expr::Const(expr) => Some(&expr.attrs),
        Expr::Continue(expr) => Some(&expr.attrs),
        Expr::Field(expr) => Some(&expr.attrs),
        Expr::ForLoop(expr) => Some(&expr.attrs),
        Expr::Group(expr) => Some(&expr.attrs),
        Expr::If(expr) => Some(&expr.attrs),
        Expr::Index(expr) => Some(&expr.attrs),
        Expr::Infer(expr) => Some(&expr.attrs),
        Expr::Let(expr) => Some(&expr.attrs),
        Expr::Lit(expr) => Some(&expr.attrs),
        Expr::Loop(expr) => Some(&expr.attrs),
        Expr::Macro(expr) => Some(&expr.attrs),
        Expr::Match(expr) => Some(&expr.attrs),
        Expr::MethodCall(expr) => Some(&expr.attrs),
        Expr::Paren(expr) => Some(&expr.attrs),
        Expr::Path(expr) => Some(&expr.attrs),
        Expr::Range(expr) => Some(&expr.attrs),
        Expr::RawAddr(expr) => Some(&expr.attrs),
        Expr::Reference(expr) => Some(&expr.attrs),
        Expr::Repeat(expr) => Some(&expr.attrs),
        Expr::Return(expr) => Some(&expr.attrs),
        Expr::Struct(expr) => Some(&expr.attrs),
        Expr::Try(expr) => Some(&expr.attrs),
        Expr::TryBlock(expr) => Some(&expr.attrs),
        Expr::Tuple(expr) => Some(&expr.attrs),
        Expr::Unary(expr) => Some(&expr.attrs),
        Expr::Unsafe(expr) => Some(&expr.attrs),
        Expr::While(expr) => Some(&expr.attrs),
        Expr::Yield(expr) => Some(&expr.attrs),
        Expr::Verbatim(_) => None,
        _ => None,
    }
}

fn expr_attrs_mut(expr: &mut Expr) -> Option<&mut Vec<Attribute>> {
    match expr {
        Expr::Array(expr) => Some(&mut expr.attrs),
        Expr::Assign(expr) => Some(&mut expr.attrs),
        Expr::Async(expr) => Some(&mut expr.attrs),
        Expr::Await(expr) => Some(&mut expr.attrs),
        Expr::Binary(expr) => Some(&mut expr.attrs),
        Expr::Block(expr) => Some(&mut expr.attrs),
        Expr::Break(expr) => Some(&mut expr.attrs),
        Expr::Call(expr) => Some(&mut expr.attrs),
        Expr::Cast(expr) => Some(&mut expr.attrs),
        Expr::Closure(expr) => Some(&mut expr.attrs),
        Expr::Const(expr) => Some(&mut expr.attrs),
        Expr::Continue(expr) => Some(&mut expr.attrs),
        Expr::Field(expr) => Some(&mut expr.attrs),
        Expr::ForLoop(expr) => Some(&mut expr.attrs),
        Expr::Group(expr) => Some(&mut expr.attrs),
        Expr::If(expr) => Some(&mut expr.attrs),
        Expr::Index(expr) => Some(&mut expr.attrs),
        Expr::Infer(expr) => Some(&mut expr.attrs),
        Expr::Let(expr) => Some(&mut expr.attrs),
        Expr::Lit(expr) => Some(&mut expr.attrs),
        Expr::Loop(expr) => Some(&mut expr.attrs),
        Expr::Macro(expr) => Some(&mut expr.attrs),
        Expr::Match(expr) => Some(&mut expr.attrs),
        Expr::MethodCall(expr) => Some(&mut expr.attrs),
        Expr::Paren(expr) => Some(&mut expr.attrs),
        Expr::Path(expr) => Some(&mut expr.attrs),
        Expr::Range(expr) => Some(&mut expr.attrs),
        Expr::RawAddr(expr) => Some(&mut expr.attrs),
        Expr::Reference(expr) => Some(&mut expr.attrs),
        Expr::Repeat(expr) => Some(&mut expr.attrs),
        Expr::Return(expr) => Some(&mut expr.attrs),
        Expr::Struct(expr) => Some(&mut expr.attrs),
        Expr::Try(expr) => Some(&mut expr.attrs),
        Expr::TryBlock(expr) => Some(&mut expr.attrs),
        Expr::Tuple(expr) => Some(&mut expr.attrs),
        Expr::Unary(expr) => Some(&mut expr.attrs),
        Expr::Unsafe(expr) => Some(&mut expr.attrs),
        Expr::While(expr) => Some(&mut expr.attrs),
        Expr::Yield(expr) => Some(&mut expr.attrs),
        Expr::Verbatim(_) => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::visit::{self, Visit};
    use syn::Stmt;

    #[test]
    fn assign_expr_ids_marks_supported_expressions() {
        let mut fn_item: ItemFn = syn::parse_quote! {
            fn kernel() {
                let x = foo(1) + 2;
                let y = &x;
            }
        };

        assign_expr_ids(&mut fn_item);

        struct Counter {
            expressions: usize,
            with_ids: usize,
        }

        impl<'ast> Visit<'ast> for Counter {
            fn visit_attribute(&mut self, _attr: &'ast Attribute) {
                // Internal node-id attributes contain literal expressions; they are
                // metadata, not source expressions.
            }

            fn visit_type(&mut self, _ty: &'ast syn::Type) {
                // Expressions inside type syntax are not lowered source expressions.
            }

            fn visit_expr(&mut self, expr: &'ast Expr) {
                self.expressions += 1;
                if expr_id(expr).is_some() {
                    self.with_ids += 1;
                }
                if let Expr::Call(call) = expr {
                    for arg in &call.args {
                        self.visit_expr(arg);
                    }
                } else {
                    visit::visit_expr(self, expr);
                }
            }
        }

        let mut counter = Counter {
            expressions: 0,
            with_ids: 0,
        };
        counter.visit_item_fn(&fn_item);

        assert!(counter.expressions > 2);
        assert_eq!(counter.expressions, counter.with_ids);
    }

    #[test]
    fn assign_expr_ids_leaves_call_callee_untagged() {
        let mut fn_item: ItemFn = syn::parse_quote! {
            fn kernel() {
                let token = new_token_unordered();
            }
        };

        assign_expr_ids(&mut fn_item);

        let Stmt::Local(local) = &fn_item.block.stmts[0] else {
            panic!("expected local statement");
        };
        let Some(init) = &local.init else {
            panic!("expected local initializer");
        };
        let Expr::Call(call) = &*init.expr else {
            panic!("expected call expression");
        };

        assert!(expr_id(&init.expr).is_some());
        assert!(matches!(&*call.func, Expr::Path(_)));
        assert!(expr_id(&call.func).is_none());
    }

    #[test]
    fn assign_expr_ids_leaves_assignment_destination_untagged() {
        let mut fn_item: ItemFn = syn::parse_quote! {
            fn kernel() {
                let mut x = 0i32;
                x = next_value();
            }
        };

        assign_expr_ids(&mut fn_item);

        let Stmt::Expr(Expr::Assign(assign), _) = &fn_item.block.stmts[1] else {
            panic!("expected assignment statement");
        };
        let Expr::Path(_) = &*assign.left else {
            panic!("expected simple assignment destination");
        };
        let Expr::Call(call) = &*assign.right else {
            panic!("expected call assignment value");
        };

        assert!(expr_id(&Expr::Assign(assign.clone())).is_none());
        assert!(expr_id(&assign.left).is_none());
        assert!(expr_id(&assign.right).is_some());
        assert!(expr_id(&call.func).is_none());
    }

    #[test]
    fn assign_expr_ids_does_not_mutate_type_syntax() {
        let mut fn_item: ItemFn = syn::parse_quote! {
            fn kernel<const N: i32>(
                tensor: Tensor<f32, { [N] }>,
            ) {
                let local: Tensor<f32, { [N] }> = make_tensor();
                consume(local);
            }
        };
        let sig = &fn_item.sig;
        let rendered_sig_before = quote::quote!(#sig).to_string();

        assign_expr_ids(&mut fn_item);

        let rendered = quote::quote!(#fn_item).to_string();
        assert!(rendered.contains("__cutile_node_id"));

        let sig = &fn_item.sig;
        let rendered_sig = quote::quote!(#sig).to_string();
        assert_eq!(rendered_sig, rendered_sig_before);

        let Stmt::Local(local) = &fn_item.block.stmts[0] else {
            panic!("expected local statement");
        };
        let pat = &local.pat;
        let rendered_pat = quote::quote!(#pat).to_string();
        assert!(!rendered_pat.contains("__cutile_node_id"));
    }

    #[test]
    fn assign_expr_ids_preserves_literal_span_start() {
        let source = r#"
fn kernel() {
    let _x = 42;
}
"#;
        let mut fn_item: ItemFn = syn::parse_str(source).unwrap();

        fn find_literal(expr: &Expr) -> Option<&ExprLit> {
            match expr {
                Expr::Lit(lit) => Some(lit),
                _ => None,
            }
        }

        let Stmt::Local(local) = &fn_item.block.stmts[0] else {
            panic!("expected local statement");
        };
        let Some(init) = &local.init else {
            panic!("expected local initializer");
        };
        let lit_before = find_literal(&init.expr).unwrap();
        let span_before = lit_before.span().start();

        assign_expr_ids(&mut fn_item);

        let Stmt::Local(local) = &fn_item.block.stmts[0] else {
            panic!("expected local statement");
        };
        let Some(init) = &local.init else {
            panic!("expected local initializer");
        };
        let lit_after = find_literal(&init.expr).unwrap();
        let span_after = lit_after.span().start();

        assert_eq!(span_after, span_before);
    }
}
