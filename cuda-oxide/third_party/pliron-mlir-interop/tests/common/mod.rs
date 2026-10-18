// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

#![allow(dead_code)]

//! Shared test helpers:
//! 1. pliron -> MLIR text
//! 2. Verify the text from (1) using `mlir-opt`.

use std::{
    env,
    io::Write as _,
    process::{Command, Stdio},
};

use pliron::{context::Context, printable::Printable};
use pliron_mlir_interop::{MlirPrinter, MlirPrinterT};

/// Environment variable naming the `mlir-opt` binary to verify test output with.
pub const MLIR_OPT_ENV: &str = "MLIR_OPT";

/// Translate `entity` to MLIR text.
///
/// Panics with the underlying pliron error if the translation fails.
pub fn to_mlir<T: MlirPrinterT + ?Sized>(ctx: &Context, entity: &T) -> String {
    String::try_from(MlirPrinter::new(ctx, entity))
        .unwrap_or_else(|err| panic!("MLIR translation failed: {}", err.disp(ctx)))
}

/// The `mlir-opt` to verify with: `$MLIR_OPT` if set, else `mlir-opt` on `PATH`.
fn mlir_opt_binary() -> String {
    env::var(MLIR_OPT_ENV).unwrap_or_else(|_| "mlir-opt".to_string())
}

/// Parse and verify `mlir` with `mlir-opt`, returning back what it printed.
///
/// Panics if `mlir-opt` rejects the input.
/// Warns if `$MLIR_OPT` is unset and no `mlir-opt` is in `PATH`.
pub fn verify_with_mlir_opt(mlir: &str) -> Option<String> {
    let binary = mlir_opt_binary();
    let explicit = env::var(MLIR_OPT_ENV).is_ok();

    let mut child = match Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !explicit => {
            eprintln!(
                "warning: skipping mlir-opt verification: no `mlir-opt` on PATH. \
                 Set ${MLIR_OPT_ENV} to enable it."
            );
            return None;
        }
        Err(e) => panic!("failed to run `{binary}`: {e}"),
    };

    child
        .stdin
        .as_mut()
        .expect("stdin was piped")
        .write_all(mlir.as_bytes())
        .expect("failed to write to mlir-opt");
    let output = child
        .wait_with_output()
        .expect("failed to wait for mlir-opt");

    assert!(
        output.status.success(),
        "`{binary}` rejected the translated MLIR:\n\
         --- stderr ---\n{}\n\
         --- input ---\n{mlir}\n",
        String::from_utf8_lossy(&output.stderr),
    );

    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Wrap `body`, which contains one or more MLIR operations in generic syntax, in a module.
fn wrap_in_module(body: &str) -> String {
    format!("\"builtin.module\"() ({{\n^entry:\n{body}\n}}) : () -> ()\n")
}

/// Verify `types` by making them the parameter types of an MLIR function
/// declaration, since `mlir-opt` parses modules, not bare types.
pub fn verify_types(types: &[String]) {
    let func_type = format!("!llvm.func<!llvm.void ({})>", types.join(", "));
    verify_with_mlir_opt(&wrap_in_module(&format!(
        r#"  "llvm.func"() <{{sym_name = "types", function_type = {func_type}}}> ({{}}) : () -> ()"#
    )));
}

/// Verify `attrs` by attaching them to an MLIR module as a discardable
/// attribute, since `mlir-opt` parses modules, not bare attributes.
pub fn verify_attrs(attrs: &[String]) {
    verify_with_mlir_opt(&format!(
        "\"builtin.module\"() ({{\n^entry:\n}}) {{test.attrs = [{}]}} : () -> ()\n",
        attrs.join(", ")
    ));
}

/// Verify `mlir` with `mlir-opt` and snapshot what it prints back.
///
/// Skipped when `mlir-opt` is unavailable.
pub fn expect_mlir_opt_output(mlir: &str, expected: expect_test::Expect) {
    if let Some(printed) = verify_with_mlir_opt(mlir) {
        expected.assert_eq(&printed);
    }
}
