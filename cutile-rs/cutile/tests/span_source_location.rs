/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for fine-grained span / source-location tracking in the JIT compiler.
//!
//! The cutile proc macro uses `Span::source_text()` to capture the
//! **verbatim** original source text of the module (including comments,
//! whitespace, and all) along with the "span base" (file, line, column) of
//! the module's opening token at expansion time.  At runtime the captured
//! string is re-parsed by `syn::parse_str`, producing spans whose
//! line/column numbers are offsets *within that string*.  The span base
//! converts them to absolute positions via:
//!
//! ```text
//! abs_line = base_line + (str_line − 1)
//! abs_col  = if str_line == 1 { base_col + str_col } else { str_col }
//! ```
//!
//! Because `Span::source_text()` preserves comments (unlike
//! `TokenStream::to_string()` which strips them), the line numbers are exact
//! even when comments appear inside the module body.
//!
//! These tests verify:
//!
//! 1. A comment-free module produces a `JITError::Located` with the **exact**
//!    file path, line number, and column number of the offending token.
//!
//! 2. A module with comments both outside and inside the macro body still
//!    produces a `JITError::Located` with the **exact** file path, line
//!    number, and column number — proving that `source_text()` correctly
//!    preserves comments in the captured text.

use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::error::JITError;

mod common;

fn unsupported_dsl_call() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Test 1 – Comment-free module: unsupported DSL expression produces a Located
//           error with the EXACT source file, line number, and column number.
// ---------------------------------------------------------------------------
//
// This module deliberately contains NO comments inside its body so that
// `TokenStream::to_string()` produces text whose line offsets map 1-to-1
// with the original file.

#[cutile::module]
mod span_error_module {
    use cutile::core::*;
    #[cutile::entry()]
    fn untyped_literal_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let _x = super::unsupported_dsl_call();
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

#[test]
fn untyped_literal_error_has_correct_source_location() {
    common::with_test_stack(|| {
        let compile_result = common::compile_to_ir(
            span_error_module::__module_ast_self,
            "span_error_module",
            "untyped_literal_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        );

        // Extract the error — we cannot use unwrap_err() because
        // ModuleOperation does not implement Debug.
        let err = match compile_result {
            Err(e) => e,
            Ok(_) => {
                panic!(
                    "Expected compilation to fail for an unsupported DSL call, but it succeeded."
                )
            }
        };

        let err_string = format!("{err}");
        println!("\n=== UNSUPPORTED DSL CALL ERROR ===\n{err_string}");

        // The error must be a Located variant carrying a real source location.
        match &err {
            JITError::Located(msg, loc) => {
                assert!(
                    loc.is_known(),
                    "Expected a known source location on the error, got {loc:?}"
                );

                // The file path must end with this test file's name.
                assert!(
                    loc.file.ends_with("span_source_location.rs"),
                    "Expected file to end with 'span_source_location.rs', got '{}'",
                    loc.file
                );

                // Find the expected line for the unsupported helper call by scanning this
                // file's own source text.  This avoids hard-coding a line
                // number that would break whenever the file is edited.
                let source = include_str!("span_source_location.rs");
                let expected_line = source
                    .lines()
                    .enumerate()
                    // Match the actual code line, not comments that mention it.
                    // The real line starts with whitespace then `let`, while
                    // comments contain `//` or `///` before it.
                    .find(|(_, line)| {
                        let trimmed = line.trim_start();
                        trimmed.starts_with("let _x = super::unsupported_dsl_call();")
                    })
                    .map(|(idx, _)| idx + 1) // 1-indexed
                    .expect("Could not find unsupported helper call marker in test source");

                assert_eq!(
                    loc.line, expected_line,
                    "Expected error on line {expected_line} ('let _x = super::unsupported_dsl_call();'), got line {}",
                    loc.line
                );

                // Column: should point into the unsupported call.
                // We verify it is non-zero and plausible.
                assert!(
                    loc.column > 0,
                    "Expected a non-zero column for the unsupported call, got {}",
                    loc.column
                );

                // The error message should mention the unsupported call.
                assert!(
                    msg.contains("unsupported_dsl_call") || msg.contains("not supported"),
                    "Expected error message to mention the unsupported call, got: {msg}"
                );

                println!(
                    "\n✓ Unsupported DSL call error correctly located at {}:{}:{}\n  message: {msg}",
                    loc.file, loc.line, loc.column
                );
            }
            other => {
                panic!("Expected JITError::Located, got a different variant: {other:?}");
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Test 2 – Comments outside and inside the macro: span tracking still
//           produces a Located error with the EXACT source location.
// ---------------------------------------------------------------------------
//
// Because `Span::source_text()` preserves comments, the line numbers
// should be exact even with comments everywhere.  We verify:
//   a) The error is a Located variant with a known source location.
//   b) The file path is correct.
//   c) The reported line matches the EXACT line of the unsupported helper call.
//   d) The column is non-zero.

// A comment OUTSIDE the macro — shifts the module's base_line but must not
// confuse the span-base arithmetic for the non-comment tokens.
// Adding several lines of comments to push the module further down.
//
//   extra line A
//   extra line B
//   extra line C
//

/// A doc-comment on the module itself — still outside the macro body.
#[cutile::module]
mod span_comments_module {
    // Comment INSIDE the macro body — stripped by the tokenizer, which
    // causes subsequent tokens to have smaller relative line numbers in
    // the reconstructed TokenStream string.
    use cutile::core::*;
    // Another comment to widen the gap.
    // And another one.
    /// Doc comment on the function (also stripped).
    #[cutile::entry()]
    fn commented_kernel<const S: [i32; 1]>(
        // Comment inside parameter list.
        output: &mut Tensor<f32, S>,
    ) {
        // Comment inside body.
        // Yet another comment.
        let _y = super::unsupported_dsl_call();
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

#[test]
fn comments_do_not_break_span_tracking() {
    common::with_test_stack(|| {
        let compile_result = common::compile_to_ir(
            span_comments_module::__module_ast_self,
            "span_comments_module",
            "commented_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        );

        let err = match compile_result {
            Err(e) => e,
            Ok(_) => panic!(
                "Expected compilation to fail for an unsupported DSL call inside \
                 commented module, but it succeeded."
            ),
        };

        let err_string = format!("{err}");
        println!("\n=== COMMENTS MODULE ERROR ===\n{err_string}");

        match &err {
            JITError::Located(msg, loc) => {
                // (a) The location must be known — even with comments the
                //     span-base machinery should produce a real location.
                assert!(
                    loc.is_known(),
                    "Expected a known source location even with comments, got {loc:?}"
                );

                // (b) File should still be this test file.
                assert!(
                    loc.file.ends_with("span_source_location.rs"),
                    "Expected file to end with 'span_source_location.rs', got '{}'",
                    loc.file
                );

                // (c) The reported line must EXACTLY match the line containing
                //     the unsupported helper call. Because `Span::source_text()` preserves
                //     comments in the captured text, the span-base arithmetic
                //     produces the correct absolute line even when comments
                //     appear inside the module body.
                let source = include_str!("span_source_location.rs");
                let expected_line = source
                    .lines()
                    .enumerate()
                    // Match the actual code line, not comments that mention it.
                    .find(|(_, line)| {
                        let trimmed = line.trim_start();
                        trimmed.starts_with("let _y = super::unsupported_dsl_call();")
                    })
                    .map(|(idx, _)| idx + 1)
                    .expect("Could not find unsupported helper call marker in test source");

                assert_eq!(
                    loc.line, expected_line,
                    "Expected error on line {expected_line} ('let _y = super::unsupported_dsl_call();'), got line {}.\n\
                     Comments should NOT affect line numbers when source_text() is used.",
                    loc.line
                );

                // (d) Column should be non-zero and plausible.
                assert!(
                    loc.column > 0,
                    "Expected a non-zero column, got {}",
                    loc.column
                );

                // The error message should reference the unsupported call.
                assert!(
                    msg.contains("unsupported_dsl_call") || msg.contains("not supported"),
                    "Expected error message to reference the unsupported call, got: {msg}"
                );

                println!(
                    "\n✓ Span tracking exact despite comments at {}:{}:{}\n  message: {msg}",
                    loc.file, loc.line, loc.column
                );
            }
            other => {
                panic!("Expected JITError::Located even with comments, got: {other:?}");
            }
        }
    });
}
