/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed dispatch lowering helpers.
//!
//! This is the compiler1-compatible bridge for the pass-manager plan: when a
//! free function is only a trait-dispatch wrapper, lower the call site to the
//! method call before the current inline compiler sees it. Trait impl selection
//! still happens from the typed call arguments in method lookup.

use crate::passes::node_ids;
use crate::passes::type_inference::TypeckResults;
use crate::syn_utils::{get_ident_from_expr, get_sig_param_names};
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprCall, ExprMethodCall, Ident, ItemFn, Stmt, Token};

/// If `fn_item` is a simple wrapper like `fn f(x, y) { x.method(y) }`, rewrite
/// `f(a, b)` to `a.method(b)`.
pub fn lower_dispatch_wrapper_call(
    fn_item: &ItemFn,
    call_expr: &ExprCall,
) -> Option<ExprMethodCall> {
    let [Stmt::Expr(Expr::MethodCall(method_call), _)] = fn_item.block.stmts.as_slice() else {
        return None;
    };

    let param_names = get_sig_param_names(&fn_item.sig);
    if param_names.len() != call_expr.args.len() {
        return None;
    }

    let arg_by_param = param_names
        .iter()
        .cloned()
        .zip(call_expr.args.iter().cloned())
        .collect::<HashMap<_, _>>();

    let receiver_ident = get_ident_from_expr(&method_call.receiver)?;
    let receiver = arg_by_param.get(&receiver_ident.to_string())?;

    let mut lowered_args = Vec::with_capacity(method_call.args.len());
    for arg in &method_call.args {
        let arg_ident = get_ident_from_expr(arg)?;
        lowered_args.push(arg_by_param.get(&arg_ident.to_string())?);
    }

    // Every synthesized token takes the user's call-site span. Tokens copied
    // from the wrapper body would carry core-source spans: `Span::join`
    // across source texts fails, `Spanned::span()` on the lowered expression
    // then degrades to `Span::call_site()`, and the inlined body's debug
    // locations resolve to the enclosing module's first line instead of the
    // call. A defaulted `dot_token` degrades the same way.
    let call_span = call_expr.func.span();
    Some(ExprMethodCall {
        attrs: Vec::new(),
        receiver: Box::new(receiver.clone()),
        dot_token: Token![.](call_span),
        method: Ident::new(&method_call.method.to_string(), call_span),
        turbofish: method_call.turbofish.clone(),
        paren_token: call_expr.paren_token,
        args: lowered_args
            .into_iter()
            .cloned()
            .collect::<Punctuated<_, Token![,]>>(),
    })
}

pub fn lower_function(fn_item: &ItemFn, typeck_results: &TypeckResults) -> ItemFn {
    let mut lowered = fn_item.clone();
    let mut pass = TypedDispatchLowering { typeck_results };
    pass.visit_item_fn_mut(&mut lowered);
    lowered
}

struct TypedDispatchLowering<'a> {
    typeck_results: &'a TypeckResults,
}

impl VisitMut for TypedDispatchLowering<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        let Some(call_id) = node_ids::expr_id(expr) else {
            return;
        };
        let Expr::Call(call) = expr else {
            return;
        };
        let Some(method_call) = self.typeck_results.lowered_method_call(call).cloned() else {
            return;
        };
        // The marker attribute leads the lowered expression's tokens, so it
        // must carry the user's call span: the inline compiler attributes the
        // whole inlined dispatch body to `method_call.span()`.
        let call_span = call.span();
        let mut lowered = Expr::MethodCall(method_call);
        node_ids::set_expr_id_spanned(&mut lowered, call_id, call_span);
        *expr = lowered;
    }
}

#[cfg(test)]
mod tests {
    use super::lower_dispatch_wrapper_call;
    use syn::spanned::Spanned;

    /// The lowered method call must keep the user's call-site position: the
    /// inline compiler attributes every op of the inlined dispatch body to
    /// `method_call.span()`, and debug info resolves that span against the
    /// user's module.
    #[test]
    fn lowered_dispatch_call_keeps_call_site_span() {
        // Wrapper parsed from a different source text, like the core module.
        let wrapper: syn::ItemFn = syn::parse_str(
            "pub fn load_tile_like<X, Y>(x: &X, y: &Y) -> <X as LoadTileLike<Y>>::Out\n\
             where X: LoadTileLike<Y> { x.load_tile_like(y) }",
        )
        .unwrap();
        let user_fn: syn::ItemFn =
            syn::parse_str("fn kernel() {\n    let t = load_tile_like(x, out);\n}").unwrap();
        let syn::Stmt::Local(local) = &user_fn.block.stmts[0] else {
            panic!("expected let statement");
        };
        let syn::Expr::Call(call) = &*local.init.as_ref().unwrap().expr else {
            panic!("expected call expression");
        };
        let lowered = lower_dispatch_wrapper_call(&wrapper, call).expect("wrapper lowers");
        let func = call.func.span().start();
        assert_eq!(lowered.dot_token.span.start(), func, "dot token");
        assert_eq!(lowered.method.span().start(), func, "method ident");
        assert_eq!(
            lowered.paren_token.span.join().start(),
            call.paren_token.span.join().start(),
            "paren token"
        );
        assert_eq!(lowered.method.to_string(), "load_tile_like");
        // Every token of the lowered call sits on the user's line, so the
        // expression span (which starts at the receiver or the paren, whichever
        // comes first in the source) is on the call's line as well.
        assert_eq!(lowered.span().start().line, call.span().start().line);

        // With the id marker in place, as the compiler sees it: the marker
        // leads the tokens and must anchor the expression at the call.
        let mut tagged = syn::Expr::MethodCall(lowered);
        crate::passes::node_ids::set_expr_id_spanned(
            &mut tagged,
            crate::passes::node_ids::NodeId(7),
            call.span(),
        );
        assert_eq!(
            tagged.span().start(),
            call.span().start(),
            "tagged expression span"
        );
        assert_eq!(
            crate::passes::node_ids::expr_id(&tagged),
            Some(crate::passes::node_ids::NodeId(7))
        );
    }
}
