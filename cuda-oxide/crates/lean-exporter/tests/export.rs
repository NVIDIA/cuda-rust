/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::Parser;
use lean_exporter::{export_function, export_named_source};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::Operation,
    parsable::parse_from_str,
};

const FIXTURE: &str = include_str!("../fixtures/arithmetic.mir");

fn export(source: &str) -> String {
    export_named_source(source, "global_index").expect("valid scalar MIR")
}

fn reject(source: &str, expected: &str) {
    let error = export_named_source(source, "global_index").expect_err("must reject input");
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

fn constant(width: u32, literal: &str) -> String {
    format!(
        "mir.func @constant: builtin.function <() -> (builtin.integer ui{width})> [] {{
           ^entry():
             c = mir.constant () [] [value: builtin.integer <{literal}: ui{width}>]: <() -> (builtin.integer ui{width})>;
             mir.return (c) [] []: <(builtin.integer ui{width}) -> ()>
         }}"
    )
}

fn parsed() -> (Context, Ptr<Operation>) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let source = FIXTURE
        .lines()
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let function = parse_from_str(
        spaced(Operation::top_level_parser()).skip(combine::eof()),
        &mut ctx,
        &source,
    )
    .expect("parse fixture");
    (ctx, function)
}

#[test]
fn exports_actual_ssa_and_preserves_operand_order() {
    let baseline = export(FIXTURE);
    assert!(baseline.contains("def program : CudaOxide.Model.Scalar.Expr 64 3"));
    assert!(baseline.contains(".mul arg0 arg1"));
    assert!(baseline.contains(".add v0 arg2"));
    assert!(
        baseline.contains(
            "def evaluate (inputs : Fin 3 → BitVec 64) : BitVec 64 := program.eval inputs"
        )
    );

    let operator_mutation = export(&FIXTURE.replace("mir.mul", "mir.sub"));
    assert_ne!(baseline, operator_mutation);
    assert!(operator_mutation.contains(".sub arg0 arg1"));

    let operand_mutation = export(&FIXTURE.replace("(product, lane)", "(lane, product)"));
    assert_ne!(baseline, operand_mutation);
    assert!(operand_mutation.contains(".add arg2 v0"));

    let return_mutation = export(&FIXTURE.replace("mir.return (index)", "mir.return (block)"));
    assert_ne!(baseline, return_mutation);
    assert!(return_mutation.contains("\n  arg0\n"));
}

#[test]
fn supports_all_four_unsigned_widths() {
    for width in [8, 16, 32, 64] {
        let output = export(&FIXTURE.replace("ui64", &format!("ui{width}")));
        assert!(output.contains(&format!("Expr {width} 3")));
    }
}

#[test]
fn preserves_high_bit_unsigned_constants_and_zero_arity() {
    for (width, value) in [
        (8, "255"),
        (16, "65535"),
        (32, "4294967295"),
        (64, "18446744073709551615"),
    ] {
        let output = export_named_source(&constant(width, value), "constant").unwrap();
        assert!(output.contains(&format!(".literal (BitVec.ofNat {width} {value})")));
        assert!(output.contains(&format!("Expr {width} 0")));
    }
    let changed = export_named_source(&constant(8, "254"), "constant").unwrap();
    assert_ne!(
        changed,
        export_named_source(&constant(8, "255"), "constant").unwrap()
    );
    assert!(export_named_source(&constant(8, "256"), "constant").is_err());
}

#[test]
fn accepts_only_integer_builtin_constants() {
    let input = "mir.func @constant: builtin.function <() -> (builtin.integer ui8)> [] {
      ^entry():
        c = builtin.constant <builtin.integer <255: ui8>> : builtin.integer ui8;
        mir.return (c) [] []: <(builtin.integer ui8) -> ()>
    }";
    assert!(
        export_named_source(input, "constant")
            .unwrap()
            .contains("BitVec.ofNat 8 255")
    );
    let bad = input.replace("builtin.integer <255: ui8>", "builtin.string \"255\"");
    assert!(
        export_named_source(&bad, "constant")
            .unwrap_err()
            .to_string()
            .contains("malformed attribute")
    );
}

#[test]
fn names_do_not_become_lean_identifiers_or_source() {
    let renamed = FIXTURE
        .replace("global_index", "theorem")
        .replace("product", "namespace")
        .replace("lane", "axiom");
    assert_eq!(
        export(FIXTURE),
        export_named_source(&renamed, "theorem").unwrap()
    );
    assert_eq!(export(FIXTURE), export(FIXTURE));
}

#[test]
fn rejects_signed_signless_float_and_unsupported_width() {
    for ty in ["si64", "i64", "ui128", "ui1"] {
        reject(&FIXTURE.replace("ui64", ty), "unsigned integer of width");
    }
    reject(
        &FIXTURE.replace("builtin.integer ui64", "builtin.fp32"),
        "only unsigned integer types",
    );
    reject(
        &FIXTURE.replace(
            "builtin.integer ui64",
            "mir.ptr <builtin.integer ui64, mutable: false, addrspace: 0, kind: RawConst>",
        ),
        "only unsigned integer types",
    );
}

#[test]
fn rejects_unknown_memory_checked_and_dead_operations() {
    reject(&FIXTURE.replace("mir.mul", "mir.not_registered"), "parse");
    for unsupported in ["mir.checked_add", "mir.load", "mir.call", "mir.assert"] {
        reject(
            &FIXTURE.replace("mir.mul", unsupported),
            "unsupported operation",
        );
    }
    let with_dead = FIXTURE.replace(
        "    mir.return",
        "    ignored = mir.neg (block) [] []: <(builtin.integer ui64) -> (builtin.integer ui64)>;\n    mir.return",
    );
    reject(&with_dead, "unsupported operation");
}

#[test]
fn rejects_attrs_regions_successors_and_extra_blocks() {
    let attributes = FIXTURE.replacen("[] []:", "[] [unexpected: builtin.bool true]:", 1);
    reject(&attributes, "unsupported attribute `unexpected`");
    let block_attribute = FIXTURE.replace(
        "lane: builtin.integer ui64):",
        "lane: builtin.integer ui64) [unexpected: builtin.bool true]:",
    );
    reject(&block_attribute, "unsupported attribute");
    let successors = FIXTURE.replacen("[] []:", "[^entry] []:", 1);
    reject(&successors, "successors are unsupported");
    let with_region = FIXTURE.replacen(
        "-> (builtin.integer ui64)>;",
        "-> (builtin.integer ui64)> {};",
        1,
    );
    reject(&with_region, "regions and successors are unsupported");
    let extra_block = FIXTURE.replace(
        "\n}",
        "\n ^second(other: builtin.integer ui64):\n mir.return (other) [] []: <(builtin.integer ui64) -> ()>\n}",
    );
    reject(&extra_block, "exactly one nonempty basic block");
}

#[test]
fn rejects_forward_undefined_mixed_and_malformed_ssa() {
    reject(
        &FIXTURE.replace("(block, block_size)", "(index, block_size)"),
        "not an earlier result",
    );
    reject(
        &FIXTURE.replace("(block, block_size)", "(undefined, block_size)"),
        "parse",
    );
    reject(&FIXTURE.replace("(block, block_size)", "(block)"), "parse");
    let mixed = FIXTURE.replacen("builtin.integer ui64", "builtin.integer ui32", 1);
    reject(&mixed, "all arguments must have the same type");
    let no_return = FIXTURE
        .lines()
        .filter(|line| !line.contains("mir.return"))
        .collect::<Vec<_>>()
        .join("\n");
    reject(&no_return.replace(">;\n}", ">\n}"), "missing mir.return");
}

#[test]
fn api_rejects_missing_or_wrong_function_type_without_panicking() {
    let (ctx, function) = parsed();
    function
        .deref_mut(&ctx)
        .attributes
        .0
        .retain(|key, _| key.to_string() != "mir_func_type");
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("malformed attribute")
    );

    let (ctx, function) = parsed();
    let integer = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    function.deref_mut(&ctx).attributes.set(
        Identifier::try_new("mir_func_type".into()).unwrap(),
        TypeAttr::new(integer.into()),
    );
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("builtin.function type")
    );
}

#[test]
fn api_rejects_wrong_result_count_without_panicking() {
    let (ctx, function) = parsed();
    let block = function
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_head()
        .unwrap();
    let add = block.deref(&ctx).iter(&ctx).nth(1).unwrap();
    let integer = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    Operation::push_result(add, &ctx, integer.into());
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("incorrect operand or result count")
    );
}

#[test]
fn selects_exact_function_and_rejects_trailing_input_and_duplicate_symbols() {
    assert!(
        export_named_source(FIXTURE, "missing")
            .unwrap_err()
            .to_string()
            .contains("was not found")
    );
    reject(&format!("{FIXTURE}\ntrailing"), "parse");
    let body = FIXTURE
        .lines()
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let module = format!("builtin.module @test {{ ^entry(): {body} }}");
    assert_eq!(export(&module), export(FIXTURE));
    let duplicate = format!("builtin.module @test {{ ^entry(): {body}; {body} }}");
    reject(&duplicate, "duplicate function symbol");
}

#[test]
fn cli_reports_errors_only_on_stderr_and_returns_failure() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_lean-exporter"))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/arithmetic.mir"
        ))
        .arg("missing")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("was not found"));
}

#[test]
fn cli_inspect_shows_the_exact_input_and_export_only_after_success() {
    let run = |function: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_lean-exporter"))
            .arg("--inspect")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/arithmetic.mir"
            ))
            .arg(function)
            .output()
            .unwrap()
    };
    let output = run("global_index");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        text,
        format!(
            "=== MIR input ===\n{FIXTURE}\n=== Generated Lean ===\n{}",
            export(FIXTURE)
        )
    );

    let failed = run("missing");
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("was not found"));
}
