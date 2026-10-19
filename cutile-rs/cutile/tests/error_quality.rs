/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU integration tests for error message quality, formatting consistency,
//! and source-location utilities that do not need CUDA runtime access.

use cutile_compiler::error::JITError;

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
        "{context}: outer Error output must start with 'error: '.\n  Got: {text}"
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
        "{context}: JITError must NOT start with 'error: ' — that prefix \
         belongs to the outer Error type.\n  Got: {output}"
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

fn assert_display_eq_debug_outer(err: &cutile::error::Error, context: &str) {
    let display = format!("{err}");
    let debug = format!("{err:?}");
    assert_eq!(
        display, debug,
        "{context}: Display and Debug must be identical.\n  Display: {display}\n  Debug:   {debug}"
    );
}

#[test]
fn outer_error_wrapping_jit_error_formatting() {
    common::with_test_stack(|| {
        use cutile_compiler::ast::SourceLocation;

        let jit_generic = JITError::Generic("something went wrong".into());
        assert_jit_error_has_no_prefix(&jit_generic, "JIT(Generic) bare");
        let jit_display = format!("{jit_generic}");

        let outer: cutile::error::Error = jit_generic.into();
        let outer_display = format!("{outer}");
        let outer_debug = format!("{outer:?}");
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Generic)");
        assert_eq!(outer_display, outer_debug);
        assert_eq!(outer_display, format!("error: {jit_display}"));
        assert_no_internal_leaks(&outer_display, "outer Error::JIT(Generic)");

        let loc = SourceLocation::new("test.rs".into(), 10, 5);
        let jit_located = JITError::Located("type mismatch".into(), loc);
        assert_jit_error_has_no_prefix(&jit_located, "JIT(Located known) bare");
        let jit_display = format!("{jit_located}");

        let outer: cutile::error::Error = jit_located.into();
        let outer_display = format!("{outer}");
        let outer_debug = format!("{outer:?}");
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Located known)");
        assert_eq!(outer_display, format!("error: {jit_display}"));
        assert_eq!(outer_display, outer_debug);
        assert!(outer_display.contains("-->"));

        let loc_unknown = SourceLocation::unknown();
        let jit_located_unknown = JITError::Located("some problem".into(), loc_unknown);
        assert_jit_error_has_no_prefix(&jit_located_unknown, "JIT(Located unknown) bare");
        let jit_display_unknown = format!("{jit_located_unknown}");

        let outer: cutile::error::Error = jit_located_unknown.into();
        let outer_display = format!("{outer}");
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Located unknown)");
        assert_eq!(outer_display, format!("error: {jit_display_unknown}"));
        assert!(!outer_display.contains("-->"));

        let tensor_err = cutile::error::tensor_error("shape mismatch: expected [128], got [64]");
        let tensor_display = format!("{tensor_err}");
        let tensor_debug = format!("{tensor_err:?}");
        assert_display_eq_debug_outer(&tensor_err, "Tensor");
        assert_single_error_prefix(&tensor_display, "outer Error::Tensor");
        assert_eq!(tensor_display, tensor_debug);

        let launch_err =
            cutile::error::kernel_launch_error("grid dimensions exceed hardware limits");
        let launch_display = format!("{launch_err}");
        let launch_debug = format!("{launch_err:?}");
        assert_display_eq_debug_outer(&launch_err, "KernelLaunch");
        assert_single_error_prefix(&launch_display, "outer Error::KernelLaunch");
        assert_eq!(launch_display, launch_debug);
    });
}

#[test]
fn located_error_always_shows_file_line_column() {
    use cutile_compiler::ast::SourceLocation;

    let loc = SourceLocation::new("my/module.rs".into(), 42, 7);
    let err = JITError::Located("unexpected token".into(), loc);
    let output = format!("{err}");

    assert_eq!(output, "unexpected token\n  --> my/module.rs:42:7");

    let outer: cutile::error::Error = err.into();
    let outer_output = format!("{outer}");
    assert_eq!(
        outer_output,
        "error: unexpected token\n  --> my/module.rs:42:7"
    );
}

#[test]
fn value_verify_error_messages_are_user_facing() {
    let verify_messages = [
        "internal: string value has inconsistent fields set",
        "internal: primitive value has inconsistent fields set",
        "internal: structured type value has inconsistent fields set",
        "internal: compound value has inconsistent fields set",
        "internal: struct value has inconsistent fields set",
        "internal: compound value missing its element list",
        "internal: struct value missing its fields",
    ];

    for msg in verify_messages {
        assert_no_internal_leaks(msg, &format!("verify message: '{msg}'"));

        let err = JITError::Generic(msg.to_string());
        let jit_output = format!("{err}");
        assert_eq!(jit_output, msg);
        assert_no_internal_leaks(&jit_output, &format!("formatted verify error: '{msg}'"));

        let outer: cutile::error::Error = err.into();
        let outer_output = format!("{outer}");
        assert_single_error_prefix(&outer_output, &format!("verify error prefix: '{msg}'"));
    }
}

#[test]
fn utility_error_messages_are_user_facing() {
    let utility_messages = [
        "failed to parse attribute `foo` with value `bar`",
        "all shape dimensions must be positive, got [-1, 2]",
        "type `Bogus` cannot be used as a tile type",
        "unsupported element type `q16`; expected an integer (`i...`) or float (`f...`) type",
        "invalid atomic mode `bogus`; valid modes are: and, or, xor, add, addf, max, min, umax, umin, xchg",
        "float types only support `xchg` and `addf` atomic modes, got `And`",
        "unrecognized arithmetic operation `bogus`",
        "this binary operator is not supported",
        "expected a variable name, got `1 + 2`",
        "undefined variable `x` when updating token",
        "variable `v` does not have associated type metadata (expected a view type)",
        "variable `v` is missing a `token` field (expected a view with an ordering token)",
        "unexpected token `@` in expression list",
    ];

    for msg in utility_messages {
        assert_no_internal_leaks(msg, &format!("utility message: '{msg}'"));
    }
}

#[test]
fn literal_error_messages_are_user_facing() {
    let literal_messages = [
        "unable to determine type for numeric literal; add a type annotation",
        "failed to compile the type of this literal",
        "expected a scalar type for this literal, got a non-scalar type",
        "repeat length must be a literal or const generic",
        "repeat length must be an integer literal",
    ];

    for msg in literal_messages {
        assert_no_internal_leaks(msg, &format!("literal message: '{msg}'"));
    }
}

#[test]
fn error_to_device_error_preserves_message() {
    use cuda_async::error::DeviceError;
    use cutile_compiler::ast::SourceLocation;

    let jit_err = JITError::Generic("compilation failed".into());
    let outer: cutile::error::Error = jit_err.into();
    let device_err: DeviceError = outer.into();
    let device_display = format!("{device_err}");
    assert!(device_display.contains("compilation failed"));

    let loc = SourceLocation::new("k.rs".into(), 5, 3);
    let jit_err = JITError::Located("type mismatch".into(), loc);
    let outer: cutile::error::Error = jit_err.into();
    let device_err: DeviceError = outer.into();
    let device_display = format!("{device_err}");
    assert!(device_display.contains("type mismatch"));
    assert!(device_display.contains("k.rs"));
}

#[test]
fn no_double_error_prefix_even_with_embedded_error_word() {
    let err = JITError::Generic("something failed".into());
    let jit_output = format!("{err}");
    assert_eq!(jit_output, "something failed");

    let outer: cutile::error::Error = err.into();
    let outer_output = format!("{outer}");
    assert_single_error_prefix(&outer_output, "outer with embedded 'error' word");
}

#[test]
fn display_debug_consistency_for_all_jit_error_variants() {
    use cutile_compiler::ast::SourceLocation;

    let cases: Vec<(&str, JITError)> = vec![
        ("Generic", JITError::Generic("generic problem".into())),
        (
            "Located(known)",
            JITError::Located(
                "located problem".into(),
                SourceLocation::new("f.rs".into(), 1, 0),
            ),
        ),
        (
            "Located(unknown)",
            JITError::Located("located unknown".into(), SourceLocation::unknown()),
        ),
        (
            "Anyhow",
            JITError::Anyhow(anyhow::anyhow!("anyhow problem")),
        ),
    ];

    for (name, err) in &cases {
        assert_display_eq_debug_jit(err, &format!("JITError::{name}"));
        assert_jit_error_has_no_prefix(err, &format!("JITError::{name}"));
    }
}

#[test]
fn spanned_jit_error_produces_located_variant_integration() {
    use cutile_compiler::ast::SourceLocation;
    use cutile_compiler::error::SpannedJITError;

    let loc = SourceLocation::new("my_kernel.rs".into(), 25, 8);
    let err = loc.jit_error("cannot borrow as mutable");

    match &err {
        JITError::Located(msg, eloc) => {
            assert_eq!(msg, "cannot borrow as mutable");
            assert!(eloc.is_known());
            assert_eq!(eloc.file, "my_kernel.rs");
            assert_eq!(eloc.line, 25);
            assert_eq!(eloc.column, 8);

            let output = format!("{err}");
            assert_eq!(output, "cannot borrow as mutable\n  --> my_kernel.rs:25:8");

            let outer: cutile::error::Error = JITError::Located(msg.clone(), eloc.clone()).into();
            let outer_output = format!("{outer}");
            assert_single_error_prefix(&outer_output, "SpannedJITError → outer");
            assert!(outer_output.contains("  --> my_kernel.rs:25:8"));
        }
        other => panic!("expected Located variant, got: {other}"),
    }
}

#[test]
fn compile_cuda_tile_op_error_messages_regression() {
    let op_messages = [
        "Expected some TypeMeta for view",
        "Expected token value in TypeMeta for view",
    ];

    for msg in op_messages {
        let err = JITError::Generic(msg.to_string());
        let jit_output = format!("{err}");
        assert!(!jit_output.starts_with("error: "));

        let outer: cutile::error::Error = err.into();
        let outer_output = format!("{outer}");
        assert_single_error_prefix(&outer_output, &format!("op message outer: '{msg}'"));
    }
}

#[test]
fn all_outer_error_variants_get_uniform_prefix() {
    use cutile_compiler::ast::SourceLocation;

    let err: cutile::error::Error = JITError::Generic("jit generic".into()).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Generic)");

    let err: cutile::error::Error = JITError::Located(
        "jit located".into(),
        SourceLocation::new("f.rs".into(), 1, 0),
    )
    .into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Located known)");

    let err: cutile::error::Error =
        JITError::Located("jit located unknown".into(), SourceLocation::unknown()).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Located unknown)");

    let err: cutile::error::Error = JITError::Anyhow(anyhow::anyhow!("jit anyhow")).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Anyhow)");

    let err = cutile::error::tensor_error("tensor problem");
    assert_single_error_prefix(&format!("{err}"), "Error::Tensor");

    let err = cutile::error::kernel_launch_error("launch problem");
    assert_single_error_prefix(&format!("{err}"), "Error::KernelLaunch");

    let err: cutile::error::Error = anyhow::anyhow!("anyhow problem").into();
    assert_single_error_prefix(&format!("{err}"), "Error::Anyhow");
}
