/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only error-quality tests.

use cutile_compiler::ast::Module;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::error::JITError;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

mod common;

const FORBIDDEN_INTERNALS: &[&str] = &[
    "TileRustValue",
    "TileRustType",
    "TypeMeta",
    "Kind::Compound",
    "Kind::Struct",
    "Kind::PrimitiveType",
    "Kind::StructuredType",
    "Kind::String",
    "get_concrete_op_ident_from_types",
];

fn assert_no_internal_leaks(text: &str, context: &str) {
    for &forbidden in FORBIDDEN_INTERNALS {
        assert!(
            !text.contains(forbidden),
            "{context}: error message must not expose internal name `{forbidden}`.\n  \
             Full message: {text}"
        );
    }
}

fn assert_single_error_prefix(text: &str, context: &str) {
    assert!(
        text.starts_with("error: "),
        "{context}: missing outer error prefix"
    );
    assert!(
        !text.starts_with("error: error: "),
        "{context}: 'error: ' prefix is doubled.\n  Full message: {text}"
    );
}

fn assert_jit_error_has_no_prefix(err: &JITError, context: &str) {
    let output = format!("{err}");
    assert!(
        !output.starts_with("error: "),
        "{context}: JITError must NOT start with 'error: '.\n  Got: {output}"
    );
}

fn assert_display_eq_debug_jit(err: &JITError, context: &str) {
    let display = format!("{err}");
    let debug = format!("{err:?}");
    assert_eq!(
        display, debug,
        "{context}: Display and Debug must be identical.\n  Display: {display}\n  Debug:   {debug}"
    );
}

fn unsupported_dsl_call() -> i32 {
    0
}

#[cutile::module]
mod error_quality_untyped_literal {
    use cutile::core::*;

    #[cutile::entry()]
    fn untyped_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let _x = super::unsupported_dsl_call();
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

#[cutile::module]
mod error_quality_same_module_inline {
    use cutile::core::*;

    fn same_module_bad_helper() {
        let _same_module_literal = super::unsupported_dsl_call();
    }

    #[cutile::entry()]
    fn caller<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        same_module_bad_helper();
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

#[cutile::module]
mod error_quality_linked_helper {
    pub fn linked_bad_helper() {
        let _linked_module_literal = super::unsupported_dsl_call();
    }
}

#[cutile::module]
mod error_quality_linked_caller {
    use crate::error_quality_linked_helper::linked_bad_helper;
    use cutile::core::*;

    #[cutile::entry()]
    fn caller<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        linked_bad_helper();
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

fn compile_and_get_error(kernel: Module, module_name: &str, function_name: &str) -> JITError {
    common::compile_to_ir(
        move || kernel.clone(),
        module_name,
        function_name,
        &[128.to_string()],
        &[("output", &[1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .err()
    .unwrap_or_else(|| {
        panic!("Expected compilation of {module_name}::{function_name} to fail, but it succeeded.")
    })
}

fn line_containing(needle: &str) -> usize {
    include_str!("compile_error_quality.rs")
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains(needle))
        .map(|(idx, _)| idx + 1)
        .unwrap_or_else(|| panic!("could not find `{needle}` in compile_error_quality.rs"))
}

fn assert_located_line(err: &JITError, expected_line: usize, context: &str) {
    match err {
        JITError::Located(_msg, loc) => {
            assert!(
                loc.is_known(),
                "{context}: expected a known source location"
            );
            assert!(
                loc.file.ends_with("compile_error_quality.rs"),
                "{context}: expected compile_error_quality.rs, got {}",
                loc.file
            );
            assert_eq!(loc.line, expected_line, "{context}: {err}");
            assert!(loc.column > 0, "{context}: expected non-zero column");
        }
        _ => panic!("{context}: expected a Located JIT error, got: {err}"),
    }
}

fn call_line_in_module(kernel: &Module, call_name: &str) -> Option<usize> {
    struct CallFinder<'a> {
        call_name: &'a str,
        line: Option<usize>,
        kernel: &'a Module,
    }

    impl<'ast> Visit<'ast> for CallFinder<'_> {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if let syn::Expr::Path(path) = &*call.func {
                if path.path.is_ident(self.call_name) {
                    self.line = Some(self.kernel.resolve_span(&call.span()).line);
                    return;
                }
            }
            visit::visit_expr_call(self, call);
        }
    }

    let mut finder = CallFinder {
        call_name,
        line: None,
        kernel,
    };
    finder.visit_item_mod(kernel.ast());
    finder.line
}

#[test]
fn untyped_literal_error_message_quality() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::__module_ast_self(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        let display = format!("{err}");
        let debug = format!("{err:?}");

        assert_no_internal_leaks(&display, "untyped literal (Display)");
        assert_no_internal_leaks(&debug, "untyped literal (Debug)");
        assert_display_eq_debug_jit(&err, "untyped literal");
        assert_jit_error_has_no_prefix(&err, "untyped literal");
        assert!(
            display.contains("42")
                || display.contains("type")
                || display.contains("annotation")
                || display.contains("literal")
                || display.contains("unsupported")
        );

        match &err {
            JITError::Located(msg, loc) => {
                assert!(loc.is_known());
                assert!(loc.file.ends_with("compile_error_quality.rs"));
                assert!(display.contains("-->"));
                assert_no_internal_leaks(msg, "untyped literal (Located msg)");
            }
            JITError::Generic(msg) => {
                assert_no_internal_leaks(msg, "untyped literal (Generic msg)");
            }
            _ => {
                assert_no_internal_leaks(&display, "untyped literal (other variant)");
            }
        }

        let outer: cutile::error::Error = err.into();
        let outer_display = format!("{outer}");
        assert_single_error_prefix(&outer_display, "untyped literal (outer)");
    });
}

#[test]
fn linked_caller_ast_call_span_points_to_call_line() {
    let kernel = error_quality_linked_caller::__module_ast_self();
    assert_eq!(
        call_line_in_module(&kernel, "linked_bad_helper"),
        Some(line_containing("linked_bad_helper();"))
    );
}

#[test]
fn same_module_inline_error_location_points_to_helper_body() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_same_module_inline::__module_ast_self(),
            "error_quality_same_module_inline",
            "caller",
        );

        assert_located_line(
            &err,
            line_containing("let _same_module_literal = super::unsupported_dsl_call();"),
            "same-module inline helper",
        );
    });
}

#[test]
fn linked_inline_error_location_points_to_helper_body() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_linked_caller::__module_ast_self(),
            "error_quality_linked_caller",
            "caller",
        );

        assert_located_line(
            &err,
            line_containing("let _linked_module_literal = super::unsupported_dsl_call();"),
            "linked inline helper",
        );
    });
}

#[test]
fn untyped_literal_error_location_points_to_this_file() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::__module_ast_self(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        match &err {
            JITError::Located(_msg, loc) => {
                assert!(loc.is_known(), "Error should have a known source location");
                assert!(
                    loc.file.ends_with("compile_error_quality.rs"),
                    "Error location file should end with 'compile_error_quality.rs', got: '{}'",
                    loc.file
                );

                let source = include_str!("compile_error_quality.rs");
                let target_line = source
                    .lines()
                    .enumerate()
                    .find(|(_, line)| {
                        let trimmed = line.trim_start();
                        trimmed.starts_with("let _x = super::unsupported_dsl_call();")
                    })
                    .map(|(idx, _)| idx + 1);

                if let Some(expected_line) = target_line {
                    assert_eq!(loc.line, expected_line);
                }

                assert!(loc.column > 0);
            }
            _ => {
                panic!("Expected a Located JIT error, got: {err}");
            }
        }
    });
}
