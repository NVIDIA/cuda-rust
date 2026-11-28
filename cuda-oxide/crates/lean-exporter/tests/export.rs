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
const VECADD_FIXTURE: &str = include_str!("../fixtures/vecadd.mir");
const SGEMM_TILED_FIXTURE: &str = include_str!("../fixtures/sgemm_tiled.mir");
const WARP_REDUCE_FIXTURE: &str = include_str!("../fixtures/warp_reduce.mir");
const SUBWARP_MULTIBUF_FIXTURE: &str = include_str!("../fixtures/subwarp_multibuf.mir");
const ASYNC_COPY_FIXTURE: &str = include_str!("../fixtures/async_copy_stage.mir");
const TCGEN05_MATMUL_FIXTURE: &str = include_str!("../fixtures/tcgen05_matmul.mir");
const WARP_RING_FIXTURE: &str = include_str!("../fixtures/warp_ring_stage.mir");
const CLUSTER_CLC_FIXTURE: &str = include_str!("../fixtures/cluster_clc_stage.mir");
const DIVERGENT_UNIFORMITY_FIXTURE: &str = include_str!("../fixtures/divergent_uniformity.mir");

fn export(source: &str) -> String {
    export_named_source(source, "global_index").expect("valid scalar MIR")
}

/// Parse an entire MIR source file, locate `name`, and translate its `mir.func` into
/// [`lean_exporter::translate::program::MirProgramData`] for structural AST assertions.
fn translate_named_mir_program(
    source: &str,
    name: &str,
) -> lean_exporter::translate::program::MirProgramData {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let headers = combine::skip_many(combine::attempt(
        combine::parser::char::spaces()
            .with(combine::parser::char::string("//"))
            .skip(combine::skip_many(combine::satisfy(|c| c != '\n'))),
    ));
    let root = parse_from_str(
        headers
            .with(spaced(Operation::top_level_parser()))
            .skip(combine::eof()),
        &mut ctx,
        source,
    )
    .expect("valid test MIR");

    let find_sym = |op: Ptr<Operation>| -> String {
        op.deref(&ctx)
            .attributes
            .0
            .iter()
            .find(|(k, _)| k.to_string() == "builtin_sym_name")
            .and_then(|(_, v)| v.downcast_ref::<pliron::builtin::attributes::IdentifierAttr>())
            .expect("function symbol name")
            .as_ref()
            .to_string()
    };

    let func = if dialect_mir::ops::MirFuncOp::wrap(&ctx, root).is_some() {
        assert_eq!(find_sym(root), name, "unexpected standalone function name");
        root
    } else {
        let module = root.deref(&ctx);
        let region = module.get_region(0).deref(&ctx);
        let block = region.get_head().expect("module block");
        block
            .deref(&ctx)
            .iter(&ctx)
            .find(|&op| {
                dialect_mir::ops::MirFuncOp::wrap(&ctx, op).is_some() && find_sym(op) == name
            })
            .unwrap_or_else(|| panic!("function `{name}` not found"))
    };

    match lean_exporter::translate::translate_function(&ctx, func)
        .unwrap_or_else(|e| panic!("failed to translate `{name}`: {e}"))
    {
        lean_exporter::translate::TranslatedFunction::MirProgram { program, .. } => *program,
        lean_exporter::translate::TranslatedFunction::Scalar { .. } => {
            panic!("expected MirProgram for `{name}`, got Scalar")
        }
    }
}

/// Resolve the `cuda-oxide/oxide-proofs-preview` package directory when `lake` is installed,
/// panicking if `CUDA_OXIDE_REQUIRE_LEAN=1` is set and Lean is unavailable.
fn lean_package_dir_if_available() -> Option<std::path::PathBuf> {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();
    if !lean_exporter::runner::is_lean_available(&lean_dir) {
        assert!(
            std::env::var("CUDA_OXIDE_REQUIRE_LEAN").as_deref() != Ok("1"),
            "`CUDA_OXIDE_REQUIRE_LEAN=1` set, but `lake` failed in {}",
            lean_dir.display()
        );
        return None;
    }
    Some(lean_dir)
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
    assert!(baseline.contains("def program : CudaOxide.Mir.Scalar.Expr 64 3"));
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

#[test]
fn exports_safe_vecadd_kernel_and_rejects_unguarded_accesses() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    let lean = export_named_source(VECADD_FIXTURE, "vecadd").expect("valid verified vecadd MIR");
    assert!(lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(lean.contains("theorem verify_no_oob_idx_param0"));
    assert!(lean.contains("theorem verify_no_oob_idx_param1"));
    assert!(lean.contains("theorem verify_no_oob_idx_param2"));
    assert!(lean.contains("theorem verify_disjoint_writes_param2"));
    assert!(lean.contains("theorem verify_writes_all_elems_param2"));

    // Scenario 4a: Moving a load into the unguarded exit block exports line by line into `mirFunc`
    // and is rejected by Lean 4 (`MirNoOobGlobal` fails on the unguarded read!).
    let unguarded_read = VECADD_FIXTURE.replace(
        "^exit():\n    mir.return",
        "^exit():\n    bad_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_elem = mir.ptr_offset (bad_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_val = mir.load (bad_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;\n    mir.return",
    );
    let unguarded_read_lean =
        export_named_source(&unguarded_read, "vecadd").expect("transcribes unguarded read 1-to-1");
    lean_exporter::runner::check_generated_lean(&unguarded_read_lean, &lean_dir)
        .expect_err("Lean 4 must reject unguarded read in exit block");

    // Scenario 4b: Moving a store into the unguarded exit block exports line by line into `mirFunc`
    // and is rejected by Lean 4 (`MirNoOobGlobal` fails on the unguarded write!).
    let unguarded_write = VECADD_FIXTURE
        .replace("builtin.fp32", "builtin.integer ui64")
        .replace(
            "^exit():\n    mir.return",
            "^exit():\n    bad_c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.disjoint_slice <builtin.integer ui64 ,[]>) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    bad_c_elem = mir.ptr_offset (bad_c_ptr, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    mir.store (bad_c_elem, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> ()>;\n    mir.return",
        );
    let unguarded_write_lean = export_named_source(&unguarded_write, "vecadd")
        .expect("transcribes unguarded write 1-to-1");
    lean_exporter::runner::check_generated_lean(&unguarded_write_lean, &lean_dir)
        .expect_err("Lean 4 must reject unguarded write in exit block");

    // Scenario 3: Inclusive branch guard (`idx <= c.len()` via `mir.le`) instead of `<` (`mir.lt`).
    let inclusive_guard = VECADD_FIXTURE.replace("mir.lt (idx, c_len)", "mir.le (idx, c_len)");
    let err = export_named_source(&inclusive_guard, "vecadd").unwrap_err();
    assert!(
        err.to_string().contains("writes_all_elems"),
        "unexpected error for inclusive guard: {err}"
    );

    // Scenario 5a: Claiming `disjoint_writes` on read-only parameter 0 (`a`).
    let bad_disjoint = VECADD_FIXTURE.replace(
        "verify_disjoint_writes: builtin.string \"2\"",
        "verify_disjoint_writes: builtin.string \"0\"",
    );
    let err = export_named_source(&bad_disjoint, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 0 is read-only and cannot satisfy a write obligation"),
        "unexpected error: {err}"
    );

    // Scenario 5b: Claiming `no_oob_idx` / `disjoint_writes` on parameter 2 (`c`) when `c` is never written.
    let unwritten_c = VECADD_FIXTURE
        .lines()
        .filter(|line| !line.contains("mir.store (c_elem, sum)"))
        .collect::<Vec<_>>()
        .join("\n");
    let err = export_named_source(&unwritten_c, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 2 is listed in `no_oob_idx` but never accessed"),
        "unexpected error: {err}"
    );
    let unwritten_c_disjoint = unwritten_c.replace(
        "verify_no_oob_idx: builtin.string \"0,1,2\"",
        "verify_no_oob_idx: builtin.string \"0,1\"",
    );
    let err = export_named_source(&unwritten_c_disjoint, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 2 is listed in `disjoint_writes` but never written"),
        "unexpected error: {err}"
    );

    // Scenario 5c: Indexing `a` by `c_len` (`a[c.len()]`) exports 1-to-1 and is rejected by Lean 4 (`c.len() == a.len()` OOB!).
    let bad_index_expr = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "a_elem = mir.ptr_offset (a_ptr, c_len)",
    );
    let bad_idx_lean =
        export_named_source(&bad_index_expr, "vecadd").expect("transcribes a[c_len] 1-to-1");
    lean_exporter::runner::check_generated_lean(&bad_idx_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB `a[c.len()]` access");

    // Scenario 5d: Conditional inner branch inside `^active` that can skip
    // writing `c` when `idx < c.len()` is rejected by Lean 4's `MirWritesAllElems`!
    let conditional_skip_write = VECADD_FIXTURE.replace(
        "    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.fp32) -> ()>;\n    mir.goto () [^exit] []: <() -> ()>",
        "    skip_flag = mir.lt (c_len, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n    mir.cond_br (skip_flag) [^do_write, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^do_write():\n    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.fp32) -> ()>;\n    mir.goto () [^exit] []: <() -> ()>",
    );
    let skip_lean = export_named_source(&conditional_skip_write, "vecadd")
        .expect("transcribes conditional skip write into MirFunc");
    assert!(skip_lean.contains("CudaOxide.Contracts.Program.MirWritesAllElems mirFunc"));
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    if let Err(err) = lean_exporter::runner::check_generated_lean(&skip_lean, &lean_dir) {
        if !err.to_string().contains("could not locate `lake`") {
            assert!(
                err.to_string().contains("lean proof check failed"),
                "unexpected Lean error for conditional skip write: {err}"
            );
        }
    } else {
        panic!("Lean 4 must reject conditional branch skipping `c` write under `writes_all_elems`");
    }
}

#[test]
fn lean_typechecker_accepts_valid_vecadd_and_rejects_false_kernel_claims() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    let baseline = export_named_source(VECADD_FIXTURE, "vecadd").expect("valid vecadd MIR");

    // If `lake` is not installed in a minimal container environment, skip the live
    // Lean subprocess assertions (covered by the dedicated `lean-verification` CI job).
    if let Err(err) = lean_exporter::runner::check_generated_lean(&baseline, &lean_dir) {
        if err.to_string().contains("could not locate `lake`") {
            return;
        }
        panic!("valid vecadd MIR must pass Lean 4 verification, got: {err}");
    }

    // Scenario 1: Missing `a.len() >= c.len()` (`0>=2` dropped from `contract_requires_len_ge`).
    // Stage 1 exports the theorem without `h_req`, and Stage 2 (Lean `omega`) rejects it.
    let missing_requires = VECADD_FIXTURE.replace("0>=2,1>=2", "1>=2");
    let missing_requires_lean =
        export_named_source(&missing_requires, "vecadd").expect("exports theorem obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_requires_lean, &lean_dir)
        .expect_err("Lean must reject `no_oob_idx(a)` when `a.len() >= c.len()` is missing");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing requires: {err}"
    );

    // Scenario 2: Off-by-one neighbor access `a[idx + 1]` under `if idx < c.len()`.
    // Stage 1 extracts `indexOffset := 1`, and Stage 2 (Lean `omega`) rejects it because
    // when `a.len() == c.len()` and `idx == c.len() - 1`, `idx + 1 == a.len()` is OOB!
    let off_by_one = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "one = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    idx_plus_one = mir.add (idx, one) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    a_elem = mir.ptr_offset (a_ptr, idx_plus_one)",
    );
    let off_by_one_lean =
        export_named_source(&off_by_one, "vecadd").expect("exports offset=1 obligation");
    assert!(off_by_one_lean.contains("CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 0"));
    let err = lean_exporter::runner::check_generated_lean(&off_by_one_lean, &lean_dir)
        .expect_err("Lean must reject off-by-one `a[idx + 1]` access");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one access: {err}"
    );

    // Scenario 6: Admitted proof (`sorry`) must be caught by the `#print axioms` gate.
    let spoofed_sorry = baseline.replacen("mir_vc mirFunc", "sorry", 1);
    let err = lean_exporter::runner::check_generated_lean(&spoofed_sorry, &lean_dir)
        .expect_err("Lean runner must reject `sorry` proofs");
    assert!(
        err.to_string().contains("sorry"),
        "unexpected error for sorry proof: {err}"
    );
}

#[test]
fn verifies_2d_sgemm_tiled_views_and_rejects_shared_races_oob_and_divergent_barriers() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    let baseline = export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_views")
        .expect("valid sgemm_tiled_views MIR");
    assert!(baseline.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(baseline.contains("theorem verify_no_oob_idx_shared0"));
    assert!(baseline.contains("theorem verify_no_data_race_shared0"));
    assert!(baseline.contains("theorem verify_uniform_barriers"));

    // Negative check: indexing SharedArray `TILE_A` by unbounded `k` instead of `smem_idx`
    // exports 1-to-1 into `mirFunc` and is rejected by Lean 4 (`MirNoOobShared` fails for `k >= 256`!).
    let bad_shared_offset = SGEMM_TILED_FIXTURE.replace(
        "slot_a = mir.ptr_offset (tile_a_mut, smem_idx)",
        "k_u64 = mir.cast (k) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    slot_a = mir.ptr_offset (tile_a_mut, k_u64)",
    );
    let bad_shared_lean = export_named_source(&bad_shared_offset, "sgemm_tiled_views")
        .expect("transcribes k_u64 shared offset 1-to-1");
    lean_exporter::runner::check_generated_lean(&bad_shared_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB `TILE_A[k]` shared-memory write");

    if let Err(err) = lean_exporter::runner::check_generated_lean(&baseline, &lean_dir) {
        if err.to_string().contains("could not locate `lake`") {
            return;
        }
        panic!("valid sgemm_tiled_views MIR must pass Lean 4 verification, got: {err}");
    }

    // Stage 2 Negative Scenario A: Missing first `sync_threads()` (`nvvm.barrier0`) between
    // Phase 0 writes (`TILE_A[ty * 16 + tx]`) and Phase 1 reads (`TILE_A[ty * 16 + i]`).
    // Both accesses land in Phase 0, and Lean 4 rejects `NoDataRace` with a counterexample!
    let missing_barrier =
        SGEMM_TILED_FIXTURE.replacen("    nvvm.barrier0 () [] []: <() -> ()>;\n", "", 1);
    let missing_barrier_lean = export_named_source(&missing_barrier, "sgemm_tiled_views")
        .expect("exports phase-collapsed obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_barrier_lean, &lean_dir)
        .expect_err("Lean must reject missing `sync_threads()` shared-memory data race");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing sync_threads: {err}"
    );

    // Stage 2 Negative Scenario B: Shrinking `TILE_A` from 256 to 128 elements under a 16x16 block.
    // Lean 4 rejects `NoOobTileAccess` because `ty * 16 + tx` can reach 255 >= 128.
    let small_tile = SGEMM_TILED_FIXTURE.replacen(
        "size: builtin.integer <256: i64>",
        "size: builtin.integer <128: i64>",
        1,
    );
    let small_tile_lean =
        export_named_source(&small_tile, "sgemm_tiled_views").expect("exports OOB tile obligation");
    let err = lean_exporter::runner::check_generated_lean(&small_tile_lean, &lean_dir)
        .expect_err("Lean must reject undersized SharedArray<f32, 128> for 16x16 block");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for undersized shared tile: {err}"
    );

    // Stage 2 Negative Scenario C: Non-uniform / divergent barrier (early conditional return
    // before the barrier loop). Transcribed into `controlFlowCfg` and rejected by Lean 4!
    let divergent_barrier = SGEMM_TILED_FIXTURE.replace(
        "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        "early_cond = mir.lt (tx, k) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;\n    mir.cond_br (early_cond) [^exit, ^enter_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^enter_loop():\n    mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
    );
    let divergent_lean = export_named_source(&divergent_barrier, "sgemm_tiled_views")
        .expect("exports controlFlowCfg for divergent barrier obligation");
    assert!(
        divergent_lean
            .contains("CudaOxide.Contracts.ControlFlow.UniformBarriers mirFunc controlFlowCfg")
    );
    let err = lean_exporter::runner::check_generated_lean(&divergent_lean, &lean_dir)
        .expect_err("Lean must reject divergent barrier when early return skips barrier loop");
    assert!(
        err.to_string().contains("lean proof check failed"),
        "unexpected Lean error for divergent barrier: {err}"
    );

    // Stage 2 (`sgemm_tiled_raw`): Verify 2D global slices `a` (param 4) and `b` (param 5)
    // directly from line-by-line MIR (`MirNoOobGlobal`) with zero strip pattern-matching in Rust!
    let raw_baseline =
        export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_raw").expect("valid sgemm_tiled_raw");
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param4 :\n    CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 4"
    ));
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param5 :\n    CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 5"
    ));
    lean_exporter::runner::check_generated_lean(&raw_baseline, &lean_dir)
        .expect("valid sgemm_tiled_raw MIR must pass Lean 4 verification");

    // Reversing the attribute list (`"5,4"` instead of `"4,5"`) still emits both obligations cleanly.
    let swapped_order = SGEMM_TILED_FIXTURE.replace(
        "verify_no_oob_idx: builtin.string \"4,5\"",
        "verify_no_oob_idx: builtin.string \"5,4\"",
    );
    let swapped_lean =
        export_named_source(&swapped_order, "sgemm_tiled_raw").expect("swapped order still valid");
    assert!(swapped_lean.contains("CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 4"));
    assert!(swapped_lean.contains("CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 5"));

    // Negative check A: indexing global slice `a` by `a_len` (`a[a.len()]`) exports 1-to-1
    // and is rejected by Lean 4 (`MirNoOobGlobal`).
    let bad_global_offset = SGEMM_TILED_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, a_idx)",
        "a_elem = mir.ptr_offset (a_ptr, a_len)",
    );
    let bad_global_lean = export_named_source(&bad_global_offset, "sgemm_tiled_raw")
        .expect("transcribes a[a_len] 1-to-1");
    lean_exporter::runner::check_generated_lean(&bad_global_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB `a[a.len()]` access");

    // Negative check B (Hole #2): missing Check 2 (`in_a = a_col < k` instead of `< a_eff`,
    // ignoring `a_start == u64::MAX`) or missing Check 1 (`a_fits` not checking `< a_len`)
    // is rejected by Lean 4 (`MirNoOobGlobal`)!
    let unguarded_row_strip =
        SGEMM_TILED_FIXTURE.replace("in_a = mir.lt (a_col, a_eff)", "in_a = mir.lt (a_col, k)");
    let unguarded_strip_lean = export_named_source(&unguarded_row_strip, "sgemm_tiled_raw")
        .expect("transcribes unguarded row access 1-to-1");
    lean_exporter::runner::check_generated_lean(&unguarded_strip_lean, &lean_dir)
        .expect_err("Lean 4 must reject missing `a_col < a_eff` guard");

    let missing_band_check = SGEMM_TILED_FIXTURE.replace(
        "a_fits = mir.lt (a_last, a_len)",
        "a_fits = mir.lt (zero_u32, k)",
    );
    let missing_band_lean = export_named_source(&missing_band_check, "sgemm_tiled_raw")
        .expect("transcribes missing band check 1-to-1");
    lean_exporter::runner::check_generated_lean(&missing_band_lean, &lean_dir)
        .expect_err("Lean 4 must reject missing `a_last < a_len` band check");

    // Negative check C (Hole #2): wrong stride (`m` instead of `n`) on `b`'s column access
    // is rejected by Lean 4 (`MirNoOobGlobal` fails when `m > n`)!
    let wrong_col_stride = SGEMM_TILED_FIXTURE.replace(
        "n_u64 = mir.cast (n) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
        "n_u64 = mir.cast (m) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
    );
    let wrong_stride_lean = export_named_source(&wrong_col_stride, "sgemm_tiled_raw")
        .expect("transcribes mismatched stride 1-to-1");
    lean_exporter::runner::check_generated_lean(&wrong_stride_lean, &lean_dir)
        .expect_err("Lean 4 must reject mismatched column stride `m` instead of `n`");

    // Negative check D (Hole #2): off-by-one `a[a_start + a_col + 1]` under valid guards
    // is rejected by Lean 4's `omega`!
    let off_by_one_strip = SGEMM_TILED_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, a_idx)",
        "a_idx_p1 = mir.add (a_idx, one_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        a_elem = mir.ptr_offset (a_ptr, a_idx_p1)",
    );
    let off_by_one_strip_lean =
        export_named_source(&off_by_one_strip, "sgemm_tiled_raw").expect("exports +1 access");
    assert!(off_by_one_strip_lean.contains("CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc 4"));
    let err = lean_exporter::runner::check_generated_lean(&off_by_one_strip_lean, &lean_dir)
        .expect_err("Lean must reject off-by-one `a[a_start + a_col + 1]` access");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one 2D access: {err}"
    );
}

/// Reject `no_data_race` on shared memory when `uniform_barriers` is omitted from the
/// kernel verification contract.
#[test]
fn rejects_no_data_race_without_uniform_barriers() {
    // `MirNoSharedRace` orders shared accesses by per-thread `syncThreads` counts. Without
    // `UniformBarriers`, `if tx < 32 { sync_threads() }` puts threads in different phases
    // and a cross-phase race passes.
    let no_uniform =
        SGEMM_TILED_FIXTURE.replacen(", verify_uniform_barriers: builtin.string \"true\"", "", 1);
    assert_ne!(no_uniform, SGEMM_TILED_FIXTURE);
    let err = export_named_source(&no_uniform, "sgemm_tiled_views")
        .expect_err("no_data_race without uniform_barriers must be rejected");
    assert!(
        err.to_string()
            .contains("`no_data_race` on shared memory requires `uniform_barriers`"),
        "unexpected error: {err}"
    );
}

#[test]
fn verifies_stage3_warp_and_block_reductions_and_rejects_missing_leader_guard() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. 1D warp butterfly reduction (`warp_reduce_sum`)
    let warp_lean = export_named_source(WARP_REDUCE_FIXTURE, "warp_reduce_sum")
        .expect("valid warp_reduce_sum MIR");
    assert!(warp_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(warp_lean.contains("theorem verify_no_oob_idx_param0"));
    assert!(warp_lean.contains("theorem verify_disjoint_writes_param1"));
    assert!(warp_lean.contains("theorem verify_writes_all_elems_param1"));
    lean_exporter::runner::check_generated_lean(&warp_lean, &lean_dir)
        .expect("valid warp_reduce_sum MIR must pass Lean 4 verification");

    // 2. 2-phase shared-memory block reduction (`block_reduce_sum`, migrated onto `MirFunc`!)
    let block_lean = export_named_source(WARP_REDUCE_FIXTURE, "block_reduce_sum")
        .expect("valid block_reduce_sum MIR");
    assert!(block_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(block_lean.contains("theorem verify_no_oob_idx_shared0"));
    assert!(block_lean.contains("theorem verify_no_data_race_shared0"));
    assert!(block_lean.contains("theorem verify_uniform_barriers"));
    assert_eq!(
        count_proven_asserts(WARP_REDUCE_FIXTURE, "block_reduce_sum"),
        2
    );
    lean_exporter::runner::check_generated_lean(&block_lean, &lean_dir)
        .expect("valid block_reduce_sum MIR must pass Lean 4 verification");

    // 3. Negative check: forgetting `if lane == 0` (`mir.goto () [^leader_stage]` unconditionally)
    // causes all 32 threads in a warp to write `WARP_SUMS[tid / 32]` in Phase 0,
    // which Lean 4 rejects in `MirNoSharedRace` with a counterexample!
    let unguarded_leader = WARP_REDUCE_FIXTURE.replace(
        "mir.cond_br (is_leader) [^leader_stage, ^barrier_sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.goto () [^leader_stage] []: <() -> ()>",
    );
    let unguarded_lean = export_named_source(&unguarded_leader, "block_reduce_sum")
        .expect("transcribes unguarded group-slot write 1-to-1 into mirFunc");
    let err = lean_exporter::runner::check_generated_lean(&unguarded_lean, &lean_dir)
        .expect_err("Lean must reject unguarded `WARP_SUMS[tid / 32]` intra-warp write race");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing `lane == 0` guard: {err}"
    );
}

#[test]
fn verifies_stage4_subwarp_bit_splits_and_multibuffer_shared_staging() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Sub-warp 16x16 bit-split staging `(tid >> 4) * 16 + (tid & 15)` + lane scan `(tid & 15) * 16 + j` on `MirFunc`
    let split_lean = export_named_source(SUBWARP_MULTIBUF_FIXTURE, "subwarp_bit_split_stage")
        .expect("valid subwarp_bit_split_stage MIR");
    assert!(split_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(split_lean.contains("(.binOp .shr"));
    assert!(split_lean.contains("(.binOp .bitAnd"));
    assert!(split_lean.contains("theorem verify_no_oob_idx_shared0"));
    assert!(split_lean.contains("theorem verify_no_data_race_shared0"));
    assert_eq!(
        count_proven_asserts(SUBWARP_MULTIBUF_FIXTURE, "subwarp_bit_split_stage"),
        2
    );
    lean_exporter::runner::check_generated_lean(&split_lean, &lean_dir)
        .expect("valid subwarp_bit_split_stage MIR must pass Lean 4 verification");

    // 2. 3-phase, 3-buffer (`S_ACT_DIRECT`, `S_XS_DIRECT`, `S_X_DIRECT`) sub-warp staging on `MirFunc`
    let multi_lean = export_named_source(SUBWARP_MULTIBUF_FIXTURE, "multibuf_subwarp_stage")
        .expect("valid multibuf_subwarp_stage MIR");
    assert!(multi_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(multi_lean.contains("theorem verify_no_oob_idx_shared0"));
    assert!(multi_lean.contains("theorem verify_no_oob_idx_shared1"));
    assert!(multi_lean.contains("theorem verify_no_oob_idx_shared2"));
    assert!(multi_lean.contains("theorem verify_no_data_race_shared0"));
    assert!(multi_lean.contains("theorem verify_no_data_race_shared1"));
    assert!(multi_lean.contains("theorem verify_no_data_race_shared2"));
    assert_eq!(
        count_proven_asserts(SUBWARP_MULTIBUF_FIXTURE, "multibuf_subwarp_stage"),
        6
    );
    lean_exporter::runner::check_generated_lean(&multi_lean, &lean_dir)
        .expect("valid multibuf_subwarp_stage MIR must pass Lean 4 verification");

    // 3. Negative check A: Widening `if tid < 64` to `if tid < 128` in Phase 0 causes
    // `S_XS_DIRECT[tid]` (`size = 64`) and `S_X_DIRECT[tid * 4 + w]` (`size = 256`) to go OOB!
    let bad_active_guard = SUBWARP_MULTIBUF_FIXTURE.replacen(
        "c64 = mir.constant () [] [value: builtin.integer <64: ui64>]",
        "c64 = mir.constant () [] [value: builtin.integer <128: ui64>]",
        1,
    );
    let bad_guard_lean = export_named_source(&bad_active_guard, "multibuf_subwarp_stage")
        .expect("exports widened active-thread guard obligation");
    let err = lean_exporter::runner::check_generated_lean(&bad_guard_lean, &lean_dir)
        .expect_err("Lean must reject `if tid < 128` OOB access on 64-element S_XS_DIRECT and 256-element S_X_DIRECT");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for widened `if tid < 128` guard: {err}"
    );

    // 4. Negative check B: Removing the second `nvvm.barrier0` (`^barrier_1`) between Phase 1
    // leader write `S_ACT_DIRECT[m_local]` and Phase 2 read `S_ACT_DIRECT[tid]` collapses
    // both into Phase 1 and fails `NoDataRace`!
    let missing_second_barrier = SUBWARP_MULTIBUF_FIXTURE.replace(
        "      ^barrier_1():\n        nvvm.barrier0 () [] []: <() -> ()>;",
        "      ^barrier_1():",
    );
    let missing_bar1_lean = export_named_source(&missing_second_barrier, "multibuf_subwarp_stage")
        .expect("exports phase-collapsed obligation on S_ACT_DIRECT");
    let err = lean_exporter::runner::check_generated_lean(&missing_bar1_lean, &lean_dir)
        .expect_err(
            "Lean must reject missing second `sync_threads()` before Phase 2 read of S_ACT_DIRECT",
        );
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing second barrier: {err}"
    );

    // 5. Negative check C: Overlapping packed loop bound (`w < 8` instead of `w < 4` on `tid * 4 + w`)
    // causes thread `tid = 0, w = 4` and thread `tid = 1, w = 0` to race on `S_X_DIRECT[4]`!
    let overlapping_pack = SUBWARP_MULTIBUF_FIXTURE.replace(
        "in_pack = mir.lt (w, c4)",
        "c8_bad = mir.constant () [] [value: builtin.integer <8: ui64>]: <() -> (builtin.integer ui64)>;\n        in_pack = mir.lt (w, c8_bad)",
    );
    let overlapping_lean = export_named_source(&overlapping_pack, "multibuf_subwarp_stage")
        .expect("exports overlapping packed-slot obligation");
    let err = lean_exporter::runner::check_generated_lean(&overlapping_lean, &lean_dir)
        .expect_err("Lean must reject overlapping `tid * 4 + w` with `w < 8`");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for overlapping packed write: {err}"
    );

    // 6. General `MirFunc` leader check: Non-zero leader lane (`if k_lane == 7` instead of `0`)
    // transcribes 1-to-1 into `mirFunc` and is proved race-free by `mir_vc` (`omega`)!
    let lane_seven_leader = SUBWARP_MULTIBUF_FIXTURE.replace(
        "zero_lane = mir.constant () [] [value: builtin.integer <0: ui64>]",
        "zero_lane = mir.constant () [] [value: builtin.integer <7: ui64>]",
    );
    let lane_seven_lean = export_named_source(&lane_seven_leader, "multibuf_subwarp_stage")
        .expect("transcribes `k_lane == 7` leader guard into mirFunc");
    assert!(lane_seven_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    lean_exporter::runner::check_generated_lean(&lane_seven_lean, &lean_dir)
        .expect("Lean must prove `MirNoSharedRace` for arbitrary leader lane `k_lane == 7`");

    // 7. General `MirFunc` strided index: Multi-level `(w * 64 + tid) + 0` transcribes 1-to-1
    // into `mirFunc` and is proved in-bounds and race-free by `mir_vc` (`omega`)!
    let coalesced_strided = SUBWARP_MULTIBUF_FIXTURE.replace(
        "tid_x4 = mir.mul (tid_u64, c4) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        pack_idx = mir.add (tid_x4, w)",
        "w_x64 = mir.mul (w, c64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        w_tid = mir.add (w_x64, tid_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        pack_idx = mir.add (w_tid, zero_w)",
    );
    let coalesced_lean = export_named_source(&coalesced_strided, "multibuf_subwarp_stage")
        .expect("transcribes coalesced `(w * 64 + tid) + 0` into mirFunc");
    assert!(coalesced_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    lean_exporter::runner::check_generated_lean(&coalesced_lean, &lean_dir)
        .expect("Lean must prove `MirNoOobShared` and `MirNoSharedRace` for `(w * 64 + tid) + 0`");
}

#[test]
fn verifies_stage5_async_copy_and_mbarrier_token_ordering() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Valid Stage 5 async TMA copy + `mbarrier_arrive_expect_tx` + `mbarrier_try_wait_parity` on `MirFunc`
    let async_lean = export_named_source(ASYNC_COPY_FIXTURE, "async_copy_stage")
        .expect("valid async_copy_stage MIR");
    assert!(async_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(async_lean.contains("(.tmaCopyG2S"));
    assert!(async_lean.contains("(.mbarTryWait 0"));
    assert!(async_lean.contains("theorem verify_no_oob_idx_shared0"));
    assert!(async_lean.contains("theorem verify_no_data_race_shared0"));
    assert_eq!(
        count_proven_asserts(ASYNC_COPY_FIXTURE, "async_copy_stage"),
        1
    );
    lean_exporter::runner::check_generated_lean(&async_lean, &lean_dir)
        .expect("valid async_copy_stage MIR must pass Lean 4 verification");

    // 2. Negative check A: Bypassing `while !mbarrier_try_wait_parity(&BAR, 0) {}`
    // (jumping straight from `^sync_after_init` / `^issue_tma` to `^consume_tile` without waiting)
    // leaves `StageReady` false, so the read stays in Phase 1 with `cp_async_bulk` and fails `NoDataRace`!
    let missing_wait = ASYNC_COPY_FIXTURE.replace(
        "mir.cond_br (not_ready) [^wait_spin, ^consume_tile] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.goto () [^consume_tile] []: <() -> ()>",
    );
    let missing_wait_lean = export_named_source(&missing_wait, "async_copy_stage")
        .expect("exports unawaited async copy obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_wait_lean, &lean_dir)
        .expect_err("Lean must reject reading TILE without waiting on mbarrier_try_wait_parity");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing mbarrier wait: {err}"
    );

    // 3. Negative check B: Omitting `mbarrier_arrive_expect_tx_shared` leaves the barrier
    // unarmed (`StageProduced` incomplete), so `StageReady` never unlocks and `NoDataRace` fails!
    let missing_arrive = ASYNC_COPY_FIXTURE
        .lines()
        .filter(|line| !line.contains("mbarrier_arrive_expect_tx_shared"))
        .collect::<Vec<_>>()
        .join("\n");
    let missing_arrive_lean = export_named_source(&missing_arrive, "async_copy_stage")
        .expect("exports unarmed mbarrier obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_arrive_lean, &lean_dir)
        .expect_err("Lean must reject mbarrier wait when mbarrier_arrive_expect_tx was omitted");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing mbarrier_arrive_expect_tx: {err}"
    );

    // 4. Negative check C: Waiting on a DIFFERENT initialized barrier (`BAR_OTHER` instead of `BAR`)
    // does not unlock `TILE` (which was bound to `BAR`), so `NoDataRace` fails in Lean!
    let wrong_barrier = ASYNC_COPY_FIXTURE
        .replace(
            "nvvm.mbarrier_init_shared (bar_init_raw, c1_u32) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;",
            "bar_other_init = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"BAR_OTHER\", source_name: builtin.string \"safe_async_copy::kernels::async_copy_stage::BAR_OTHER\"]: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;\n        nvvm.mbarrier_init_shared (bar_init_raw, c1_u32) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;\n        nvvm.mbarrier_init_shared (bar_other_init, c1_u32) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;",
        )
        .replace(
            "bar_wait_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"BAR\", source_name: builtin.string \"safe_async_copy::kernels::async_copy_stage::BAR\"]",
            "bar_wait_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"BAR_OTHER\", source_name: builtin.string \"safe_async_copy::kernels::async_copy_stage::BAR_OTHER\"]",
        );
    let wrong_bar_lean = export_named_source(&wrong_barrier, "async_copy_stage")
        .expect("exports mismatched mbarrier obligation");
    let err = lean_exporter::runner::check_generated_lean(&wrong_bar_lean, &lean_dir)
        .expect_err("Lean must reject waiting on BAR_OTHER when cp_async_bulk used BAR");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for wrong mbarrier wait: {err}"
    );

    // 5. Negative check D (Hole #4): TMA `tx_bytes` (`32768` bytes = `8192` floats) exceeding
    // `TILE`'s declared `4096`-float capacity must emit `loopBound := 8192` vs `bufferLen := 4096`
    // and fail `NoOobTileAccess` in Lean 4's `omega`!
    let oversized_tma = ASYNC_COPY_FIXTURE.replace(
        "c16384_u32 = mir.constant () [] [value: builtin.integer <16384: ui32>]",
        "c16384_u32 = mir.constant () [] [value: builtin.integer <32768: ui32>]",
    );
    let oversized_lean =
        export_named_source(&oversized_tma, "async_copy_stage").expect("exports oversized TMA OOB");
    assert!(oversized_lean.contains("8192)"));
    let err = lean_exporter::runner::check_generated_lean(&oversized_lean, &lean_dir)
        .expect_err("Lean must reject 32 KB TMA transfer into 16 KB (4096-float) TILE");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for oversized TMA transfer: {err}"
    );

    // 6. Negative check E (Hole #4): Unmodeled `s2g` TMA copy (`nvvm.cp_async_bulk_tensor_s2g_tile_2d`)
    // must be rejected fail-closed rather than silently ignored!
    let unmodeled_s2g = ASYNC_COPY_FIXTURE.replace(
        "mir.goto () [^exit] []: <() -> ()>",
        "nvvm.cp_async_bulk_tensor_s2g_tile_2d (tile_r_raw, tmap, tile_x, tile_y) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, mir.ptr <builtin.integer ui8,mutable:false,addrspace:0,kind:SharedRef>, builtin.integer si32, builtin.integer si32) -> ()>;\n        mir.goto () [^exit] []: <() -> ()>",
    );
    let err = export_named_source(&unmodeled_s2g, "async_copy_stage")
        .expect_err("unmodeled `cp_async_bulk_tensor_s2g` must be rejected fail-closed");
    assert!(
        err.to_string()
            .contains("unsupported async bulk operation `nvvm.cp_async_bulk_tensor_s2g_tile_2d`"),
        "unexpected error for s2g TMA copy: {err}"
    );
}

/// Parse `source` and return the number of `mir.assert` operations marked proven
/// redundant by `translate_function` on `fn_name`.
fn count_proven_asserts(source: &str, fn_name: &str) -> usize {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let stripped = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let module = parse_from_str(
        spaced(Operation::top_level_parser()).skip(combine::eof()),
        &mut ctx,
        &stripped,
    )
    .expect("parse module fixture");
    let block = module
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_head()
        .expect("module block");
    let func = block
        .deref(&ctx)
        .iter(&ctx)
        .find(|&op| {
            dialect_mir::ops::MirFuncOp::wrap(&ctx, op).is_some_and(|f| {
                use pliron::builtin::op_interfaces::SymbolOpInterface;
                f.get_symbol_name(&ctx).to_string() == fn_name
            })
        })
        .expect("target function");
    match lean_exporter::translate::translate_function(&ctx, func).expect("translate") {
        lean_exporter::translate::TranslatedFunction::MirProgram {
            proven_assert_ops, ..
        } => proven_assert_ops.len(),
        lean_exporter::translate::TranslatedFunction::Scalar { .. } => 0,
    }
}

#[test]
fn elides_only_proven_shared_array_bounds_checks() {
    // Baseline: `sgemm_tiled_views` verifies both `TILE_A` (0) and `TILE_B` (1),
    // eliding all 4 `SharedArray` bounds checks (2 writes + 2 reads).
    assert_eq!(
        count_proven_asserts(SGEMM_TILED_FIXTURE, "sgemm_tiled_views"),
        4
    );

    // 1. Unrelated `mir.assert(tx < 4)` (e.g. from indexing a 4-element local array
    // `tmp[tx]` or `assert!(tx < 4)`) must NEVER be elided just because `TILE_A` is verified.
    let with_local_array_check = SGEMM_TILED_FIXTURE.replace(
        "^exit():\n        mir.return () [] []: <() -> ()>",
        "^exit():\n        c4_tmp = mir.constant () [] [value: builtin.integer <4: ui32>]: <() -> (builtin.integer ui32)>;\n        ok_tmp = mir.lt (tx, c4_tmp) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;\n        mir.assert (ok_tmp) [] []: <(builtin.integer i1) -> ()>;\n        mir.return () [] []: <() -> ()>",
    );
    assert_eq!(
        count_proven_asserts(&with_local_array_check, "sgemm_tiled_views"),
        4,
        "must not elide unverified `tx < 4` bounds check"
    );

    // 2. Even a same-bound `mir.assert(smem_idx < 256)` in the SAME basic block that is NOT
    // paired with a verified `SharedArray` `mir.ptr_offset` must be preserved.
    let with_extra_256_check = SGEMM_TILED_FIXTURE.replace(
        "mir.store (slot_b, beta) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
        "mir.store (slot_b, beta) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;\n        c256_extra = mir.constant () [] [value: builtin.integer <256: ui64>]: <() -> (builtin.integer ui64)>;\n        ok_extra = mir.lt (smem_idx, c256_extra) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n        mir.assert (ok_extra) [] []: <(builtin.integer i1) -> ()>;",
    );
    assert_eq!(
        count_proven_asserts(&with_extra_256_check, "sgemm_tiled_views"),
        4,
        "must not elide standalone `smem_idx < 256` assert without a paired verified SharedArray access"
    );

    // 3. When only `TILE_A` (`0`) is listed in `verify_no_oob_shared` (and `TILE_B` (`1`) is NOT),
    // only `TILE_A`'s 2 bounds checks may be elided; `TILE_B`'s 2 bounds checks must stay intact.
    let only_tile_a_oob = SGEMM_TILED_FIXTURE.replacen(
        "verify_no_oob_shared: builtin.string \"0,1\"",
        "verify_no_oob_shared: builtin.string \"0\"",
        1,
    );
    assert_eq!(
        count_proven_asserts(&only_tile_a_oob, "sgemm_tiled_views"),
        2,
        "must not elide `TILE_B`'s bounds checks when only `TILE_A` is in `no_oob_idx`"
    );

    // 4. An unused `&TILE_A[bad_idx]` pointer offset (never loaded or stored, so never
    // emitted into Lean's `NoOobTileAccess` obligations) must NOT have its `mir.assert` elided.
    let with_unused_oob_ref = SGEMM_TILED_FIXTURE.replace(
        "mir.store (slot_b, beta) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
        "mir.store (slot_b, beta) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;\n        c100 = mir.constant () [] [value: builtin.integer <100: ui64>]: <() -> (builtin.integer ui64)>;\n        bad_idx = mir.mul (smem_idx, c100) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        ok_unused = mir.lt (bad_idx, c256_a) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n        mir.assert (ok_unused) [] []: <(builtin.integer i1) -> ()>;\n        _unused_slot = mir.ptr_offset (tile_a_mut, bad_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>)>;",
    );
    assert_eq!(
        count_proven_asserts(&with_unused_oob_ref, "sgemm_tiled_views"),
        4,
        "must not elide `mir.assert` on a SharedArray `ptr_offset` that is never loaded or stored"
    );

    // 5. Real `rustc` call-terminator layout: `mir.assert` + `mir.ptr_offset` in the call block,
    // followed by `mir.goto` into the continuation block that executes `mir.store`.
    let split_call_block = SGEMM_TILED_FIXTURE.replace(
        "slot_a = mir.ptr_offset (tile_a_mut, smem_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>)>;\n        mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
        "slot_a = mir.ptr_offset (tile_a_mut, smem_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>)>;\n        mir.goto () [^phase0_store_a] []: <() -> ()>\n\n      ^phase0_store_a():\n        mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
    );
    assert_eq!(
        count_proven_asserts(&split_call_block, "sgemm_tiled_views"),
        4,
        "must elide `mir.assert` across `SharedArray::index` call-terminator block split"
    );
}

#[test]
fn verifies_disjoint_bitor_and_rejects_overlapping_bitor_and_narrowing_cast_collisions() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive disjoint `mir.bitor`: `(m_local * 16) | (tid & 15)` has disjoint bit
    // ranges (`row_off % 16 == 0`, `k_lane < 16`), so `or_eq_add_of_mod_two_pow` proves it
    // equal to `row_off + k_lane` in `MirFunc` and passes Lean 4 verification!
    let disjoint_bitor = SUBWARP_MULTIBUF_FIXTURE.replacen(
        "split_idx = mir.add (row_off, k_lane)",
        "split_idx = mir.bitor (row_off, k_lane)",
        1,
    );
    let disjoint_lean = export_named_source(&disjoint_bitor, "subwarp_bit_split_stage")
        .expect("disjoint `(m_local * 16) | (tid & 15)` must export cleanly");
    assert!(disjoint_lean.contains("(.binOp .bitOr"));
    lean_exporter::runner::check_generated_lean(&disjoint_lean, &lean_dir)
        .expect("Lean must prove disjoint `(m_local * 16) | (tid & 15)` safe");

    // 2. Negative overlapping `mir.bitor`: `SUBWARP_TILE[tid | 1] = 0.0` collides between
    // threads 0 and 1 (`0 | 1 == 1 | 1 == 1`), transcribes 1-to-1 into `mirFunc`, and fails
    // `MirNoSharedRace` in Lean 4!
    let overlapping_bitor = SUBWARP_MULTIBUF_FIXTURE.replacen(
        "split_idx = mir.add (row_off, k_lane)",
        "c1_bad = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        split_idx = mir.bitor (tid, c1_bad)",
        1,
    );
    let overlapping_bitor_lean = export_named_source(&overlapping_bitor, "subwarp_bit_split_stage")
        .expect("transcribes overlapping `tid | 1` 1-to-1 into mirFunc");
    let err = lean_exporter::runner::check_generated_lean(&overlapping_bitor_lean, &lean_dir)
        .expect_err("Lean 4 must reject overlapping `tid | 1` write collision in MirNoSharedRace");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for overlapping `tid | 1`: {err}"
    );

    // 3. Negative narrowing integer cast (`((tid * 2) as u8) as u64`): wraps modulo 256 so
    // thread `0` and thread `128` both write `SUBWARP_TILE[0]`. Transcribes `.intCast 8`
    // and fails `MirNoSharedRace` in Lean 4's `omega`!
    let narrowing_cast_race = SUBWARP_MULTIBUF_FIXTURE
        .replacen(
            "size: builtin.integer <256: i64>",
            "size: builtin.integer <512: i64>",
            2,
        )
        .replacen(
            "split_idx = mir.add (row_off, k_lane) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;",
            "c2_u64 = mir.constant () [] [value: builtin.integer <2: ui64>]: <() -> (builtin.integer ui64)>;\n        tid_x2 = mir.mul (tid, c2_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        trunc_u8 = mir.cast (tid_x2) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui64) -> (builtin.integer ui8)>;\n        split_idx = mir.cast (trunc_u8) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui8) -> (builtin.integer ui64)>;",
            1,
        );
    let narrowing_lean = export_named_source(&narrowing_cast_race, "subwarp_bit_split_stage")
        .expect("narrowing `as u8` exports `.intCast 8` in mirFunc");
    assert!(narrowing_lean.contains("(.intCast 8"));
    let err = lean_exporter::runner::check_generated_lean(&narrowing_lean, &lean_dir)
        .expect_err("Lean must reject `((tid * 2) as u8) as usize` wraparound collision");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for narrowing `as u8` collision: {err}"
    );

    // 4. Negative 1D narrowing cast (`c[(idx as u8) as u64]` in `vecadd`): wraps modulo 256
    // across blocks (`bx=0` and `bx=1` both write `c[tx]`) and is rejected by Lean 4 (`MirDisjointWrites`)!
    let vecadd_narrow_cast = VECADD_FIXTURE.replace(
        "c_elem = mir.ptr_offset (c_ptr, idx)",
        "idx_u8 = mir.cast (idx) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui64) -> (builtin.integer ui8)>;\n    idx_wrapped = mir.cast (idx_u8) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui8) -> (builtin.integer ui64)>;\n    c_elem = mir.ptr_offset (c_ptr, idx_wrapped)",
    );
    let narrow_lean = export_named_source(&vecadd_narrow_cast, "vecadd")
        .expect("transcribes `(idx as u8) as u64` 1-to-1");
    lean_exporter::runner::check_generated_lean(&narrow_lean, &lean_dir)
        .expect_err("Lean 4 must reject 1D narrowing `(idx as u8) as usize` write collision");
}

#[test]
fn verifies_cyclic_loop_phases_with_init_barrier_and_reordered_blocks() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Adding an acyclic init `nvvm.barrier0` BEFORE `^tile_header` while keeping BOTH
    // in-loop barriers (`^phase0_stage` + `^phase1_end`) advances in-loop phases to `1 mod 2`
    // and `2 mod 2` and passes Lean 4 verification.
    let init_plus_two_loop_bars = SGEMM_TILED_FIXTURE.replacen(
        "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        "nvvm.barrier0 () [] []: <() -> ()>;\n        mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        1,
    );
    let valid_init_lean = export_named_source(&init_plus_two_loop_bars, "sgemm_tiled_views")
        .expect("init barrier + 2-barrier loop must export cleanly");
    assert!(valid_init_lean.contains(".syncThreads"));
    lean_exporter::runner::check_generated_lean(&valid_init_lean, &lean_dir)
        .expect("init barrier + 2-barrier loop must pass Lean 4 verification");

    // 2. Negative check (Hole #5): Init `nvvm.barrier0` BEFORE `^tile_header`, but removing the
    // second in-loop `nvvm.barrier0` (`^phase1_end`) leaves ONLY 1 barrier inside the loop.
    // `passed % 1 == 0` wraps the post-barrier read into the pre-barrier write's phase (`1`),
    // and Lean 4's `omega` rejects the loop-carried WAR/WAW data race!
    let init_plus_one_loop_bar = init_plus_two_loop_bars.replacen(
        "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;",
        "      ^phase1_end():",
        1,
    );
    let bad_loop_lean = export_named_source(&init_plus_one_loop_bar, "sgemm_tiled_views")
        .expect("exports wrapped 1-barrier loop phase obligation");
    let err = lean_exporter::runner::check_generated_lean(&bad_loop_lean, &lean_dir)
        .expect_err("Lean must reject 1-barrier loop even when preceded by an init sync_threads()");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for 1-barrier loop after init sync: {err}"
    );

    // 3. Reordering MIR basic blocks so `^phase1_end` (the 2nd barrier) appears BEFORE
    // `^phase0_stage` (the 1st barrier) in the block list must still sort barriers by
    // CFG dominance and pass Lean 4 verification!
    let reordered_blocks = SGEMM_TILED_FIXTURE
        .replacen(
            "      ^phase0_stage():",
            "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        one_u32 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;\n        next_tile = mir.add (tile, one_u32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n        mir.goto (sum_inner, next_tile) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>\n\n      ^phase0_stage():",
            1,
        )
        .replacen(
            "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        one_u32 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;\n        next_tile = mir.add (tile, one_u32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n        mir.goto (sum_inner, next_tile) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>\n\n      ^exit():",
            "      ^exit():",
            1,
        );
    // Move `^phase1_end` after `^scan_header` so `sum_inner` still dominates `^phase1_end` while
    // `^phase1_end` still sits before `^phase1_read` (and swap `^phase0_stage` after `^phase1_end`).
    let _ = reordered_blocks;
    let reordered_valid = SGEMM_TILED_FIXTURE
        .replacen(
            "      ^phase1_read():",
            "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        one_u32 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;\n        next_tile = mir.add (tile, one_u32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n        mir.goto (sum_inner, next_tile) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>\n\n      ^phase1_read():",
            1,
        )
        .replacen(
            "      ^phase1_end():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        one_u32 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;\n        next_tile = mir.add (tile, one_u32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n        mir.goto (sum_inner, next_tile) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>\n\n      ^exit():",
            "      ^exit():",
            1,
        );
    let reordered_lean = export_named_source(&reordered_valid, "sgemm_tiled_views")
        .expect("reordered barrier blocks must export cleanly");
    lean_exporter::runner::check_generated_lean(&reordered_lean, &lean_dir)
        .expect("reordered barrier blocks must pass Lean 4 verification");
}

#[test]
fn rejects_post_loop_loopvar_leak_conflicting_loop_counters_and_escaping_shared_ptrs() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Post-loop `w` leak (Hole #6): Indexing `WARP_SUMS[w]` in `^exit` AFTER
    // `while w < 8` has terminated (`!(w < 8)`) transcribes 1-to-1 into `mirFunc`
    // and is rejected by Lean 4's `MirNoOobShared`!
    let post_loop_oob = WARP_REDUCE_FIXTURE.replace(
        "      ^exit():\n        mir.return () [] []: <() -> ()>\n    }\n}",
        "      ^exit():\n        bad_ws_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string \"WARP_SUMS\", source_name: builtin.string \"safe_warp_reduce::kernels::block_reduce_sum::WARP_SUMS\"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;\n        bad_ws_ref = mir.cast (bad_ws_raw) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>)>;\n        bad_post_ptr = mir.ptr_offset (bad_ws_ref, w) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>)>;\n        _bad_post_val = mir.load (bad_post_ptr) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.fp32)>;\n        mir.return () [] []: <() -> ()>\n    }\n}",
    );
    let post_loop_lean = export_named_source(&post_loop_oob, "block_reduce_sum")
        .expect("transcribes post-loop `WARP_SUMS[w]` access 1-to-1 into mirFunc");
    let err = lean_exporter::runner::check_generated_lean(&post_loop_lean, &lean_dir)
        .expect_err("Lean 4 must reject post-loop `WARP_SUMS[w]` access (`!(w < 8)`)");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for post-loop `w` access: {err}"
    );

    // 2a. Valid nested two-loop indexing (`g < 4, g2 < 4 -> k_lane * 16 + g + g2 <= 246 < 256`):
    // `MirFunc` assigns distinct `.havoc` variables to each loop header, proving safe nested loops!
    let nested_valid = SUBWARP_MULTIBUF_FIXTURE.replace(
        "      ^phase1_body():\n        c16_x = mir.constant () [] [value: builtin.integer <16: ui64>]: <() -> (builtin.integer ui64)>;",
        "      ^phase1_body():\n        c16_x = mir.constant () [] [value: builtin.integer <16: ui64>]: <() -> (builtin.integer ui64)>;\n        mir.goto (zero_g) [^inner_h] []: <(builtin.integer ui64) -> ()>\n\n      ^inner_h(g2: builtin.integer ui64):\n        in_g2 = mir.lt (g2, c4_g) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n        mir.cond_br (in_g2) [^inner_b, ^phase1_leader_check] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^inner_b():",
    ).replace(
        "x_read_idx = mir.add (x_lane_base, g)",
        "mixed_loops = mir.add (g, g2) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        x_read_idx = mir.add (x_lane_base, mixed_loops)",
    ).replace(
        "mir.goto (next_acc, next_g) [^phase1_header] []: <(builtin.fp32, builtin.integer ui64) -> ()>",
        "one_g2 = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        next_g2 = mir.add (g2, one_g2) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        mir.goto (next_g2) [^inner_h] []: <(builtin.integer ui64) -> ()>",
    ).replace(
        "mir.cond_br (in_g2) [^inner_b, ^phase1_leader_check] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.cond_br (in_g2) [^inner_b, ^inner_exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^inner_exit():\n        step_g1 = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        adv_g1 = mir.add (g, step_g1) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        mir.goto (acc, adv_g1) [^phase1_header] []: <(builtin.fp32, builtin.integer ui64) -> ()>",
    );
    let nested_valid_lean = export_named_source(&nested_valid, "multibuf_subwarp_stage")
        .expect("transcribes valid 2-level nested loop into mirFunc");
    lean_exporter::runner::check_generated_lean(&nested_valid_lean, &lean_dir)
        .expect("Lean 4 proves in-bounds `k_lane * 16 + g + g2 < 256` across nested loops");

    // 2b. Out-of-bounds nested two-loop indexing (`g < 4, g2 < 16 -> 15 * 16 + 3 + 15 = 258 >= 256`):
    // rejected by Lean 4's `MirNoOobShared`!
    let nested_oob =
        nested_valid.replace("in_g2 = mir.lt (g2, c4_g)", "in_g2 = mir.lt (g2, c16_x)");
    let nested_oob_lean = export_named_source(&nested_oob, "multibuf_subwarp_stage")
        .expect("transcribes OOB nested loop into mirFunc");
    let err = lean_exporter::runner::check_generated_lean(&nested_oob_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB nested loop `k_lane * 16 + g + g2` (`258 >= 256`)");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for OOB nested loop counters: {err}"
    );

    // 3. Escaping verified `SharedArray` pointer into an unmodeled `mir.call`:
    // must be rejected fail-closed!
    let escaping_shared = SGEMM_TILED_FIXTURE
        .replacen(
            "  ^entry():\n",
            "  ^entry():\n    mir.func @external_clobber_tile: builtin.function <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>) -> ()> [] { ^entry(p: mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>): mir.return () [] []: <() -> ()> };\n",
            1,
        )
        .replacen(
            "mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
            "mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;\n        mir.call (tile_a_mut) [] [callee: builtin.string \"external_clobber_tile\"]: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>) -> ()>;",
            1,
        );
    let err = export_named_source(&escaping_shared, "sgemm_tiled_views")
        .expect_err("verified SharedArray pointer escaping into `mir.call` must fail closed");
    assert!(
        err.to_string().contains(
            "verified shared-memory pointer cannot escape into unmodeled operation `mir.call`"
        ),
        "unexpected error for escaping shared pointer: {err}"
    );
}

#[test]
fn verifies_stage6_tcgen05_tmem_proxy_and_mbarrier_lifecycle() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Valid Stage 6 Blackwell `tcgen05` + `TMEM` + TMA + proxy + `mbarrier` kernel passes
    //    `lean-exporter` and Lean 4 (`0 sorryAx`) on `MirFunc`, eliding both `TMEM_ADDR[0]` and `SMEM_OUT` bounds checks!
    let valid_lean = export_named_source(TCGEN05_MATMUL_FIXTURE, "tcgen05_matmul_verified")
        .expect("valid Stage 6 tcgen05_matmul_verified MIR must export cleanly");
    assert!(valid_lean.contains("def mirFunc : CudaOxide.Mir.Program.MirFunc"));
    assert!(valid_lean.contains("theorem verify_no_oob_tmem"));
    assert!(valid_lean.contains("theorem verify_tmem_lifecycle"));
    lean_exporter::runner::check_generated_lean(&valid_lean, &lean_dir)
        .expect("valid Stage 6 tcgen05_matmul_verified MIR must pass Lean 4 verification");
    assert_eq!(
        count_proven_asserts(TCGEN05_MATMUL_FIXTURE, "tcgen05_matmul_verified"),
        2,
        "both `TMEM_ADDR[0]` and `SMEM_OUT[tid * 128 + col]` mir.assert checks must be proven redundant"
    );

    // 2. Negative check (PTX Proxy Coherence in Lean 4): Omitting `fence_proxy_async_shared_cta()`
    //    after `mbarrier_init` before `cp_async_bulk` / `tcgen05_commit` is transcribed into
    //    `controlFlowCfg` and rejected by Lean 4's `TmemLifecycle`!
    let missing_proxy_fence = TCGEN05_MATMUL_FIXTURE
        .lines()
        .filter(|line| !line.contains("fence_proxy_async_shared_cta"))
        .collect::<Vec<_>>()
        .join("\n");
    let missing_proxy_lean = export_named_source(&missing_proxy_fence, "tcgen05_matmul_verified")
        .expect("transcribes missing proxy fence into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_proxy_lean, &lean_dir)
        .expect_err("Lean 4 must reject missing fence_proxy_async_shared_cta after mbarrier_init");

    // 3. Negative check (`mbarrier` Init Lifecycle in Lean 4): Omitting `sync_threads()`
    //    between `mbarrier_init` and `mbarrier_arrive` / `mbarrier_try_wait` fails in Lean 4!
    let missing_init_sync = ASYNC_COPY_FIXTURE.replacen(
        "      ^sync_after_init():\n        nvvm.barrier0 () [] []: <() -> ()>;",
        "      ^sync_after_init():",
        1,
    );
    let missing_init_sync_lean = export_named_source(&missing_init_sync, "async_copy_stage")
        .expect("transcribes missing sync_threads after mbarrier_init into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_init_sync_lean, &lean_dir)
        .expect_err("Lean 4 must reject missing sync_threads() after mbarrier_init");

    // 4. Negative check (Uninitialized Barrier in Lean 4): Waiting on an uninitialized `BAR_UNINIT`!
    let uninit_bar = ASYNC_COPY_FIXTURE.replace(
        "bar_wait_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"BAR\", source_name: builtin.string \"safe_async_copy::kernels::async_copy_stage::BAR\"]",
        "bar_wait_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"BAR_UNINIT\", source_name: builtin.string \"safe_async_copy::kernels::async_copy_stage::BAR_UNINIT\"]",
    );
    let uninit_bar_lean = export_named_source(&uninit_bar, "async_copy_stage")
        .expect("transcribes uninitialized Barrier wait into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&uninit_bar_lean, &lean_dir)
        .expect_err("Lean 4 must reject waiting on an uninitialized Barrier");

    // 5. Negative check (Loop Back-Pressure `EMPTY_BAR` in Lean 4): Looping back from consumer
    //    `^consume_tile` into async TMA producer `^issue_tma` with no `EMPTY_BAR` or `sync_threads()`!
    let missing_backpressure = ASYNC_COPY_FIXTURE.replace(
        "        _val = mir.load (elem_ptr) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.fp32)>;\n        mir.goto () [^exit] []: <() -> ()>",
        "        _val = mir.load (elem_ptr) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.fp32)>;\n        mir.cond_br (is_leader) [^issue_tma, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
    );
    let missing_bp_lean = export_named_source(&missing_backpressure, "async_copy_stage")
        .expect("transcribes un-backpressured cyclic TMA loop into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_bp_lean, &lean_dir).expect_err(
        "Lean 4 must reject cyclic async TMA loop without EMPTY_BAR or sync_threads back-pressure",
    );

    // 6. Negative check (Missing `tmem_lifecycle` clause): `tcgen05` op without `verify_tmem_lifecycle`!
    let missing_clause =
        TCGEN05_MATMUL_FIXTURE.replace(", verify_tmem_lifecycle: builtin.string \"true\"", "");
    let err = export_named_source(&missing_clause, "tcgen05_matmul_verified")
        .expect_err("tcgen05 ops without `tmem_lifecycle` must fail closed");
    assert!(
        err.to_string()
            .contains("tcgen05 operations in a #[verify] kernel require `tmem_lifecycle`"),
        "unexpected error for missing tmem_lifecycle clause: {err}"
    );

    // 7. Negative check (Single-lane `tcgen05_alloc` instead of 32-lane warp collective in Lean 4):
    let lane_divergent_alloc = TCGEN05_MATMUL_FIXTURE.replacen(
        "mir.cond_br (is_warp0) [^alloc_tmem, ^sync_after_alloc]",
        "mir.cond_br (is_leader) [^alloc_tmem, ^sync_after_alloc]",
        1,
    );
    let lane_div_lean = export_named_source(&lane_divergent_alloc, "tcgen05_matmul_verified")
        .expect("transcribes single-lane tcgen05_alloc guard into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&lane_div_lean, &lean_dir).expect_err(
        "Lean 4 must reject single-lane `tid == 0` guard on warp-synchronous `tcgen05_alloc`",
    );

    // 8. Negative check (Missing `mbarrier_try_wait` on `MMA_BAR` before `tcgen05_ld` in Lean 4):
    let missing_mma_wait = TCGEN05_MATMUL_FIXTURE.replacen(
        "mir.cond_br (not_ready_mma) [^wait_mma_spin, ^epilogue_init]",
        "mir.goto () [^epilogue_init]",
        1,
    ).replace(
        " [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^wait_mma_spin():",
        " []: <() -> ()>\n\n      ^wait_mma_spin():",
    );
    let missing_mma_wait_lean = export_named_source(&missing_mma_wait, "tcgen05_matmul_verified")
        .expect("transcribes unawaited MMA_BAR into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_mma_wait_lean, &lean_dir).expect_err(
        "Lean 4 must reject tcgen05_ld without dominating mbarrier_try_wait on MMA_BAR",
    );

    // 9. Negative check (Missing `tcgen05_load_wait()` after `tcgen05_ld` in Lean 4):
    let missing_ld_wait = TCGEN05_MATMUL_FIXTURE
        .lines()
        .filter(|line| !line.contains("tcgen05_load_wait"))
        .collect::<Vec<_>>()
        .join("\n");
    let missing_ld_wait_lean = export_named_source(&missing_ld_wait, "tcgen05_matmul_verified")
        .expect("transcribes missing tcgen05_load_wait into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_ld_wait_lean, &lean_dir)
        .expect_err("Lean 4 must reject tcgen05_ld without tcgen05_load_wait()");

    // 10. Negative check (`TMEM` leak: early-return branch `tile_x == tile_y` bypassing
    //     `tcgen05_dealloc` to `^exit`, rejected by Lean 4):
    let leaked_tmem = TCGEN05_MATMUL_FIXTURE.replacen(
        "mir.cond_br (in_col) [^col_body, ^sync_before_dealloc]",
        "mir.cond_br (in_col) [^col_body, ^maybe_skip_dealloc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^maybe_skip_dealloc():\n        early_cond = mir.eq (tile_x, tile_y) [] []: <(builtin.integer si32, builtin.integer si32) -> (builtin.integer i1)>;\n        mir.cond_br (early_cond) [^exit, ^sync_before_dealloc]",
        1,
    );
    let leaked_tmem_lean = export_named_source(&leaked_tmem, "tcgen05_matmul_verified")
        .expect("transcribes early-return TMEM leak into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&leaked_tmem_lean, &lean_dir)
        .expect_err("Lean 4 must reject early return bypassing tcgen05_dealloc");

    // 11. Negative check (Lean 4 `NoOobTmemAccess` column overflow: allocating only `64` TMEM
    //     columns while `M128_N128` MMA + `col < 128` epilogue access `128` columns!):
    let tmem_col_overflow = TCGEN05_MATMUL_FIXTURE
        .replace(
            "c128_cols = mir.constant () [] [value: builtin.integer <128: ui32>]",
            "c128_cols = mir.constant () [] [value: builtin.integer <64: ui32>]",
        )
        .replace(
            "c128_dealloc = mir.constant () [] [value: builtin.integer <128: ui32>]",
            "c128_dealloc = mir.constant () [] [value: builtin.integer <64: ui32>]",
        );
    let overflow_lean = export_named_source(&tmem_col_overflow, "tcgen05_matmul_verified")
        .expect("exports 64-col TMEM allocation against 128-col MMA/ld obligations");
    let err = lean_exporter::runner::check_generated_lean(&overflow_lean, &lean_dir)
        .expect_err("Lean 4 must reject 128-column MMA/ld on a 64-column TMEM allocation");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for TMEM column overflow: {err}"
    );

    // 12. Negative check (Lean 4 warpgroup lane-slice violation: reading `tmem_addr + col`
    //     without `(warp_id * 32) << 16` across 4 warps!):
    let bad_warp_slice = TCGEN05_MATMUL_FIXTURE.replace(
        "ld_addr = mir.add (tmem_row_base, col_u32)",
        "ld_addr = mir.add (tmem_addr, col_u32)",
    );
    let bad_slice_lean = export_named_source(&bad_warp_slice, "tcgen05_matmul_verified")
        .expect("exports warpgroup lane-slice violation to Lean");
    let err = lean_exporter::runner::check_generated_lean(&bad_slice_lean, &lean_dir)
        .expect_err("Lean 4 must reject warps 1..3 reading TMEM lane 0 via tcgen05_ld");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for TMEM warpgroup slice violation: {err}"
    );

    // 13. Negative check (3-level nested non-coordinate `if`s inside `if warp_id == 0` around
    //     `tcgen05_dealloc`: transcribed as `.unknown` edges and rejected by Lean 4!):
    let nested_3_ifs_dealloc = TCGEN05_MATMUL_FIXTURE.replacen(
        "      ^dealloc_tmem():\n        c128_dealloc = mir.constant () [] [value: builtin.integer <128: ui32>]: <() -> (builtin.integer ui32)>;\n        nvvm.tcgen05_dealloc (tmem_addr, c128_dealloc) [] []: <(builtin.integer ui32, builtin.integer ui32) -> ()>;\n        mir.goto () [^exit] []: <() -> ()>",
        "      ^dealloc_tmem():\n        cond_l1 = mir.eq (tile_x, tile_y) [] []: <(builtin.integer si32, builtin.integer si32) -> (builtin.integer i1)>;\n        mir.cond_br (cond_l1) [^dealloc_l2, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^dealloc_l2():\n        cond_l2 = mir.lt (tile_x, tile_y) [] []: <(builtin.integer si32, builtin.integer si32) -> (builtin.integer i1)>;\n        mir.cond_br (cond_l2) [^dealloc_l3, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^dealloc_l3():\n        cond_l3 = mir.ne (tile_x, tile_y) [] []: <(builtin.integer si32, builtin.integer si32) -> (builtin.integer i1)>;\n        mir.cond_br (cond_l3) [^dealloc_inner, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^dealloc_inner():\n        c128_dealloc = mir.constant () [] [value: builtin.integer <128: ui32>]: <() -> (builtin.integer ui32)>;\n        nvvm.tcgen05_dealloc (tmem_addr, c128_dealloc) [] []: <(builtin.integer ui32, builtin.integer ui32) -> ()>;\n        mir.goto () [^exit] []: <() -> ()>",
        1,
    );
    let nested_3_lean = export_named_source(&nested_3_ifs_dealloc, "tcgen05_matmul_verified")
        .expect("transcribes 3-deep nested unknown ifs around tcgen05_dealloc into controlFlowCfg");
    assert!(nested_3_lean.contains(".warpUniform"));
    lean_exporter::runner::check_generated_lean(&nested_3_lean, &lean_dir).expect_err(
        "Lean 4 must reject 3-level nested runtime `if`s around `tcgen05_dealloc` inside `if warp_id == 0`",
    );

    // 14. Negative check (Out-of-order `tcgen05_commit` placed BEFORE `tcgen05_mma_f16`):
    let commit_before_mma = TCGEN05_MATMUL_FIXTURE.replacen(
        "        nvvm.tcgen05_mma_f16 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;\n        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;",
        "        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        nvvm.tcgen05_mma_f16 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;",
        1,
    );
    let commit_before_mma_lean = export_named_source(&commit_before_mma, "tcgen05_matmul_verified")
        .expect("transcribes out-of-order commit-before-mma into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&commit_before_mma_lean, &lean_dir)
        .expect_err("Lean 4 must reject `tcgen05_commit` placed before `tcgen05_mma`");

    // 15. Negative check (Out-of-order `tcgen05_load_wait()` placed AFTER `raw_val` is stored!):
    let use_before_ld_wait = TCGEN05_MATMUL_FIXTURE.replacen(
        "        raw_val = nvvm.tcgen05_ld_32x32b_x1_raw (ld_addr) [] []: <(builtin.integer ui32) -> (builtin.integer ui32)>;\n        nvvm.tcgen05_load_wait () [] []: <() -> ()>;",
        "        raw_val = nvvm.tcgen05_ld_32x32b_x1_raw (ld_addr) [] []: <(builtin.integer ui32) -> (builtin.integer ui32)>;",
        1,
    ).replacen(
        "        mir.store (out_ptr, raw_val) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui32) -> ()>;",
        "        mir.store (out_ptr, raw_val) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui32) -> ()>;\n        nvvm.tcgen05_load_wait () [] []: <() -> ()>;",
        1,
    );
    let use_before_wait_lean = export_named_source(&use_before_ld_wait, "tcgen05_matmul_verified")
        .expect("transcribes tmemLoadUse before tmemLoadWait into controlFlowCfg");
    assert!(
        use_before_wait_lean
            .contains("(.tmemLoad 0), (.tmemLoadUse 0), (.sharedWrite 2), .tmemLoadWait")
    );
    lean_exporter::runner::check_generated_lean(&use_before_wait_lean, &lean_dir).expect_err(
        "Lean 4 must reject reading `tcgen05_ld` register before `tcgen05_load_wait()`",
    );

    // 16. Negative check (`smem_a_off + 2048` interior byte offset overflowing 4096-byte `SMEM_A`):
    let smem_desc_byte_oob = TCGEN05_MATMUL_FIXTURE.replacen(
        "        a_shr = mir.shr (a_off, sh4) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;",
        "        c2048_off = mir.constant () [] [value: builtin.integer <2048: ui64>]: <() -> (builtin.integer ui64)>;\n        a_off_shifted = mir.add (a_off, c2048_off) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        a_shr = mir.shr (a_off_shifted, sh4) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;",
        1,
    );
    let smem_oob_lean = export_named_source(&smem_desc_byte_oob, "tcgen05_matmul_verified")
        .expect("exports byte_offset + tile_bytes = 6144 against 4096-byte SMEM_A");
    assert!(smem_oob_lean.contains("(.smemDescRead"));
    let err = lean_exporter::runner::check_generated_lean(&smem_oob_lean, &lean_dir)
        .expect_err("Lean 4 must reject `smem_a_off + 2048` overflowing 4096-byte SMEM_A");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for SMEM descriptor byte offset overflow: {err}"
    );
}

#[test]
fn verifies_plain_mut_slice_and_rejects_unverified_wraparound_and_cross_thread_races() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive: plain `c: &mut [f32]` (`MirSliceType` with `mutable: true`) + 64-bit hardware
    //    `(blockIdx_x as usize) * (blockDim_x as usize) + (threadIdx_x as usize)` + in-place `c[idx] += a[idx]`
    //    proves `NoOobIdx`, `DisjointWrites`, `DisjointSliceAccessPair`, and `WritesAllElems`, AND elides
    //    both `a[idx]` and `c[idx]` `mir.assert` checks!
    let mut_inplace_mir = r#"mir.func @inplace_add: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=1", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "1", verify_no_oob_idx: builtin.string "0,1", verify_writes_all_elems: builtin.string "1"]
{
  ^entry(a: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, c: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    bd32 = nvvm.read_ptx_sreg_ntid_x () [] []: <() -> (builtin.integer ui32)>;
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    c_len = mir.extract_field (c) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_bounds = mir.lt (idx, c_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_bounds) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    a_len = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    a_chk = mir.lt (idx, a_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (a_chk) [] []: <(builtin.integer i1) -> ()>;
    a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_elem = mir.ptr_offset (a_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_val = mir.load (a_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    c_chk = mir.lt (idx, c_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (c_chk) [] []: <(builtin.integer i1) -> ()>;
    c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    c_elem = mir.ptr_offset (c_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    c_prev = mir.load (c_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;
    c_next = mir.add (c_prev, a_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;
    mir.store (c_elem, c_next) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}"#;
    let inplace_lean =
        export_named_source(mut_inplace_mir, "inplace_add").expect("exports &mut [f32] kernel");
    assert!(inplace_lean.contains("CudaOxide.Contracts.Program.MirDisjointWrites mirFunc 1"));
    lean_exporter::runner::check_generated_lean(&inplace_lean, &lean_dir)
        .expect("Lean 4 proves in-place c[idx] += a[idx] on &mut [f32]");

    // Verify both `a[idx]` and `c[idx]` `mir.assert` ops are proven redundant and elided:
    assert_eq!(
        count_proven_asserts(
            &format!("builtin.module @test {{\n  ^entry_mod():\n{mut_inplace_mir}\n}}"),
            "inplace_add"
        ),
        2
    );

    // 2. Negative: `&mut [f32]` without `disjoint_writes` must fail closed in `spec.rs`:
    let missing_disjoint =
        mut_inplace_mir.replacen("verify_disjoint_writes: builtin.string \"1\", ", "", 1);
    let err = export_named_source(&missing_disjoint, "inplace_add")
        .expect_err("&mut [f32] without disjoint_writes must be rejected");
    assert!(
        err.to_string()
            .contains("requires both `no_oob_idx` and `disjoint_writes`"),
        "unexpected error: {err}"
    );

    // 3. Negative: 32-bit `blockIdx_x() * blockDim_x()` without `u32_coordinates` wraps `2^32`
    //    (`bx = 0` and `bx = 16_777_216` collide!) and is rejected by Lean 4 (`MirDisjointWrites`)!
    let wrap32_mir = mut_inplace_mir.replacen(
        "    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;",
        "    base32 = mir.mul (bx32, bd32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n    idx32 = mir.add (base32, tx32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n    idx = mir.cast (idx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
        1,
    );
    let wrap32_lean = export_named_source(&wrap32_mir, "inplace_add")
        .expect("transcribes 32-bit coordinate math 1-to-1");
    lean_exporter::runner::check_generated_lean(&wrap32_lean, &lean_dir).expect_err(
        "Lean 4 must reject 32-bit blockIdx_x * blockDim_x wraparound without u32_coordinates",
    );

    // 4. Positive: with `launch_contract_u32_coords: "true"`, 32-bit coordinates ARE sound and accepted:
    let u32_coords_ok = wrap32_mir.replacen(
        "gpu_kernel: builtin.string \"true\", ",
        "gpu_kernel: builtin.string \"true\", launch_contract_u32_coords: builtin.string \"true\", ",
        1,
    );
    let u32_lean = export_named_source(&u32_coords_ok, "inplace_add")
        .expect("32-bit coordinates succeed when u32_coordinates is active");
    lean_exporter::runner::check_generated_lean(&u32_lean, &lean_dir)
        .expect("Lean 4 proves u32_coordinates inplace_add");

    // 5. Negative: racy cross-thread read `c[idx + 1]` paired with write `c[idx]` on `&mut [f32]`
    //    must be rejected by Lean 4 (`MirDisjointWrites`):
    let racy_stencil = mut_inplace_mir.replacen(
        "    c_prev = mir.load (c_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;",
        "    c1_off = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    idx_plus1 = mir.add (idx, c1_off) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    c_next_elem = mir.ptr_offset (c_ptr, idx_plus1) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;\n    c_prev = mir.load (c_next_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;",
        1,
    );
    let racy_lean = export_named_source(&racy_stencil, "inplace_add")
        .expect("exports MirDisjointWrites obligation");
    assert!(racy_lean.contains("CudaOxide.Contracts.Program.MirDisjointWrites mirFunc 1"));
    lean_exporter::runner::check_generated_lean(&racy_lean, &lean_dir)
        .expect_err("Lean 4 must reject cross-thread c[idx + 1] read + c[idx] write race");
}

/// Verify that `mir.checked_add`, `mir.checked_sub`, and `mir.checked_mul` on a
/// slice-loaded `u32` keep their overflow arm reachable in Lean 4 (`add`/`mul` fail
/// bounds and disjointness; `sub(1)` overflows at `x = 0` and fails `MirDisjointWrites`),
/// while `x.checked_mul(1)` proves its overflow arm unreachable and a safe handler
/// on both arms verifies and elides all 3 bounds assertions.
#[test]
fn checked_arithmetic_overflow_arm_is_reachable() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // `match xs[idx].checked_OP(K) { Some(_) => out[idx] = out[idx], None => out[x as usize] = out[x as usize] }`.
    // The flag depends on a loaded value. Lean has to reason about it symbolically.
    let mir = r#"mir.func @ovf_probe: builtin.function <(mir.slice <builtin.integer ui32,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=1", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "1", verify_no_oob_idx: builtin.string "0,1"]
{
  ^entry(xs: mir.slice <builtin.integer ui32,mutable:false,kind:SharedRef>, out: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    bd32 = nvvm.read_ptx_sreg_ntid_x () [] []: <() -> (builtin.integer ui32)>;
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    out_len = mir.extract_field (out) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_bounds = mir.lt (idx, out_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_bounds) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    xs_len = mir.extract_field (xs) [] [index: mir.field_index 1]: <(mir.slice <builtin.integer ui32,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    xs_chk = mir.lt (idx, xs_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (xs_chk) [] []: <(builtin.integer i1) -> ()>;
    xs_ptr = mir.extract_field (xs) [] [index: mir.field_index 0]: <(mir.slice <builtin.integer ui32,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.integer ui32,mutable:false,addrspace:0,kind:Erased>)>;
    xs_elem = mir.ptr_offset (xs_ptr, idx) [] []: <(mir.ptr <builtin.integer ui32,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32,mutable:false,addrspace:0,kind:Erased>)>;
    x = mir.load (xs_elem) [] []: <(mir.ptr <builtin.integer ui32,mutable:false,addrspace:0,kind:Erased>) -> (builtin.integer ui32)>;
    k = mir.constant () [] [value: builtin.integer <K: ui32>]: <() -> (builtin.integer ui32)>;
    pair = CHECKED_OP (x, k) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (mir.tuple <[builtin.integer ui32, builtin.integer i1],[0, 1],[0, 4],8,4>)>;
    ovf = mir.extract_field (pair) [] [index: mir.field_index 1]: <(mir.tuple <[builtin.integer ui32, builtin.integer i1],[0, 1],[0, 4],8,4>) -> (builtin.integer i1)>;
    mir.cond_br (ovf) [^overflow, ^ok] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^overflow():
    x64 = mir.cast (x) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    x_chk = mir.lt (x64, out_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (x_chk) [] []: <(builtin.integer i1) -> ()>;
    out_ptr_o = mir.extract_field (out) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    out_elem_o = mir.ptr_offset (out_ptr_o, x64) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    v_o = mir.load (out_elem_o) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;
    mir.store (out_elem_o, v_o) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^ok():
    idx_chk = mir.lt (idx, out_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (idx_chk) [] []: <(builtin.integer i1) -> ()>;
    out_ptr = mir.extract_field (out) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    out_elem = mir.ptr_offset (out_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    v = mir.load (out_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>) -> (builtin.fp32)>;
    mir.store (out_elem, v) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}"#;
    let module = |src: &str| format!("builtin.module @test {{\n  ^entry_mod():\n{src}\n}}");

    let probe = |op: &str, k: &str| {
        mir.replacen("CHECKED_OP", op, 1)
            .replacen("<K:", &format!("<{k}:"), 1)
    };

    // Overflow points: `x + 1` at `u32::MAX`, `x - 1` at `0`, `x * 2` at `2^31`.
    // Lean must reject `out[x]` in the reachable overflow arm.
    for (op, k) in [
        ("mir.checked_add", "1"),
        ("mir.checked_sub", "1"),
        ("mir.checked_mul", "2"),
    ] {
        let lean =
            export_named_source(&probe(op, k), "ovf_probe").expect("exports checked arithmetic");
        lean_exporter::runner::check_generated_lean(&lean, &lean_dir)
            .expect_err("Lean 4 must reject out[x] in the reachable overflow arm");
    }

    // `x * 1` never overflows. The overflow arm is dead and the kernel verifies.
    let no_ovf =
        export_named_source(&probe("mir.checked_mul", "1"), "ovf_probe").expect("exports x * 1");
    lean_exporter::runner::check_generated_lean(&no_ovf, &lean_dir)
        .expect("Lean 4 proves the overflow arm of x * 1 unreachable");

    // Positive: when both arms index `out[idx]`, the kernel verifies and all 3 checks go.
    let safe = probe("mir.checked_add", "1")
        .replacen(
            "x_chk = mir.lt (x64, out_len)",
            "x_chk = mir.lt (idx, out_len)",
            1,
        )
        .replacen(
            "mir.ptr_offset (out_ptr_o, x64)",
            "mir.ptr_offset (out_ptr_o, idx)",
            1,
        );
    let safe_lean = export_named_source(&safe, "ovf_probe").expect("exports safe probe");
    lean_exporter::runner::check_generated_lean(&safe_lean, &lean_dir)
        .expect("Lean 4 proves both arms of a checked_add branch");
    assert_eq!(count_proven_asserts(&module(&safe), "ovf_probe"), 3);
}

#[test]
fn verifies_stage8_warp_ring_and_rejects_slot_mismatch_wrap_oob_and_missing_backpressure() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive: 4-stage warp-specialized TMA + metadata + `SharedArray<Barrier, 4, 8>` ring
    //    proves `MirNoOobShared`, `MirNoSharedRace`, `UniformBarriers`, and `MirAssertRedundant`!
    let ring_lean =
        export_named_source(WARP_RING_FIXTURE, "warp_ring_stage").expect("exports warp_ring_stage");
    assert!(ring_lean.contains("(.barrierSlotCheck"));
    assert!(ring_lean.contains("(.mbarReady 0 true .ctaAcquire)"));
    assert!(ring_lean.contains("(.posLoopExit"));
    lean_exporter::runner::check_generated_lean(&ring_lean, &lean_dir)
        .expect("Lean 4 proves warp_ring_stage with 0 sorryAx");

    // 2. Negative (Exporter fail-closed): omitting `FULL_BAR` (`shared 1`) from `verify_no_oob_shared`
    //    is rejected fail-closed!
    let unlisted_bar = WARP_RING_FIXTURE.replacen(
        "verify_no_oob_shared: builtin.string \"0,1,2,3,4\"",
        "verify_no_oob_shared: builtin.string \"0,2,3,4\"",
        1,
    );
    let err = export_named_source(&unlisted_bar, "warp_ring_stage")
        .expect_err("SharedArray<Barrier, ...> omitted from no_oob_idx must fail closed");
    assert!(
        err.to_string().contains("must be listed in `no_oob_idx"),
        "unexpected error for unlisted SharedArray<Barrier, ...>: {err}"
    );

    // 3. Negative (Lean 4 `MirNoOobShared` inductive ring wrap off-by-one):
    //    Mutating `p_wrap = (p_inc == 4)` to `p_inc == 5` on a 4-stage ring fails in Lean 4!
    let wrap_oob = WARP_RING_FIXTURE.replacen(
        "p_wrap = mir.eq (p_inc, c4_u64)",
        "c5_u64 = mir.constant () [] [value: builtin.integer <5: ui64>]: <() -> (builtin.integer ui64)>;\n        p_wrap = mir.eq (p_inc, c5_u64)",
        1,
    );
    let wrap_oob_lean = export_named_source(&wrap_oob, "warp_ring_stage")
        .expect("transcribes off-by-one ring wrap 1-to-1 into MirFunc");
    let err = lean_exporter::runner::check_generated_lean(&wrap_oob_lean, &lean_dir)
        .expect_err("Lean 4 must reject `if stage + 1 == 5` wrap on a 4-stage ring");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one ring wrap: {err}"
    );

    // 4. Negative (Lean 4 `MirNoSharedRace` barrier/data stage-slot mismatch):
    //    Consumer awaits `FULL_BAR[c_stage]` then reads `RING_BUF[((c_stage + 1) & 3) * 1024 + tid]`!
    let slot_mismatch = WARP_RING_FIXTURE.replacen(
        "c_off = mir.mul (c_stage, c1024_u64)",
        "c3_mask = mir.constant () [] [value: builtin.integer <3: ui64>]: <() -> (builtin.integer ui64)>;\n        c_plus1 = mir.add (c_stage, one_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        c_wrong = mir.bitand (c_plus1, c3_mask) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        c_off = mir.mul (c_wrong, c1024_u64)",
        1,
    );
    let slot_mismatch_lean = export_named_source(&slot_mismatch, "warp_ring_stage")
        .expect("transcribes mismatched barrier/data stage index 1-to-1 into MirFunc");
    let err = lean_exporter::runner::check_generated_lean(&slot_mismatch_lean, &lean_dir)
        .expect_err("Lean 4 must reject reading RING_BUF[(c_stage + 1) & 3] after awaiting FULL_BAR[c_stage]");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for mismatched barrier/data stage slot: {err}"
    );

    // 5. Negative (Lean 4 `UniformBarriers` / `NoUnbackpressuredLoop` missing `EMPTY_BAR` wait):
    //    Producer warp skips spinning on `empty_ready` and jumps straight to `^prod_issue`!
    let missing_empty_wait = WARP_RING_FIXTURE.replacen(
        "mir.cond_br (empty_not_ready) [^prod_wait_spin, ^prod_issue] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.goto () [^prod_issue] []: <() -> ()>",
        1,
    );
    let missing_empty_lean = export_named_source(&missing_empty_wait, "warp_ring_stage")
        .expect("transcribes missing producer EMPTY_BAR wait into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&missing_empty_lean, &lean_dir).expect_err(
        "Lean 4 must reject warp-specialized TMA producer ring missing EMPTY_BAR back-pressure wait",
    );
}

#[test]
fn verifies_stage9_cluster_dsmem_clc_and_cg2_tmem_lifecycle() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive (`cluster_clc_stage`): 2-CTA cluster + DSMEM `map_shared_rank` + remote
    //    `mbarrier_arrive_cluster` + Blackwell `clc_try_cancel_multicast` verifies in Lean 4!
    let clc_lean = export_named_source(CLUSTER_CLC_FIXTURE, "cluster_clc_stage")
        .expect("exports cluster_clc_stage");
    assert!(clc_lean.contains("clusterDimX := 2"));
    assert!(clc_lean.contains("(.clusterRankCheck"));
    assert!(clc_lean.contains("(.dsmemAccess 1)"));
    assert!(clc_lean.contains("(.clcTryCancel 0 0 true)"));
    assert!(clc_lean.contains("(.mbarReady 0 false .clusterAcquire)"));
    assert!(clc_lean.contains("CudaOxide.Contracts.Program.MirNoOobClusterRank mirFunc"));
    lean_exporter::runner::check_generated_lean(&clc_lean, &lean_dir)
        .expect("Lean 4 proves cluster_clc_stage with 0 sorryAx");

    // 1b. Negative (`.ctaAcquire` wait on remote `PEER_DONE` or multicast `CLC_BAR`):
    let cta_wait_peer = CLUSTER_CLC_FIXTURE.replacen(
        "peer_ready = nvvm.mbarrier_try_wait_parity_cluster (peer_bar_w, zero_u32)",
        "peer_ready = nvvm.mbarrier_try_wait_parity_shared (peer_bar_w, zero_u32)",
        1,
    );
    let cta_wait_peer_lean = export_named_source(&cta_wait_peer, "cluster_clc_stage")
        .expect("transcribes .cta wait on remote PEER_DONE");
    lean_exporter::runner::check_generated_lean(&cta_wait_peer_lean, &lean_dir)
        .expect_err("Lean 4 must reject .cta try_wait on remote cluster-signaled PEER_DONE");

    let relaxed_wait_clc = CLUSTER_CLC_FIXTURE.replacen(
        "clc_ready = nvvm.mbarrier_try_wait_parity_cluster (clc_bar_w, zero_u32)",
        "clc_ready = nvvm.mbarrier_try_wait_parity_cluster_relaxed (clc_bar_w, zero_u32)",
        1,
    );
    let relaxed_wait_clc_lean = export_named_source(&relaxed_wait_clc, "cluster_clc_stage")
        .expect("transcribes .relaxed.cluster wait on CLC_BAR");
    lean_exporter::runner::check_generated_lean(&relaxed_wait_clc_lean, &lean_dir)
        .expect_err("Lean 4 must reject .relaxed wait on CLC_BAR before reading CLC_RESPONSE");

    // 2. Negative (`MirNoOobClusterRank` in Lean 4): `my_rank + 1` overflows `clusterDimX == 2` on CTA 1!
    let oob_rank = CLUSTER_CLC_FIXTURE.replacen(
        "peer_rank = mir.bitxor (my_rank, one_u32)",
        "peer_rank = mir.add (my_rank, one_u32)",
        1,
    );
    let oob_rank_lean = export_named_source(&oob_rank, "cluster_clc_stage")
        .expect("transcribes peer_rank = my_rank + 1 into MirFunc");
    let err = lean_exporter::runner::check_generated_lean(&oob_rank_lean, &lean_dir)
        .expect_err("Lean 4 must reject map_shared_rank(my_rank + 1) on a 2-CTA cluster");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for OOB cluster rank: {err}"
    );

    // 3. Negative (Cross-CTA DSMEM race in Lean 4): replacing `cluster_sync()` with CTA-local
    //    `nvvm.barrier0` before `map_shared_rank` DSMEM read fails `UniformBarriers`!
    let local_sync_only = CLUSTER_CLC_FIXTURE.replacen(
        "nvvm.cluster_sync () [] []: <() -> ()>;",
        "nvvm.barrier0 () [] []: <() -> ()>;",
        1,
    );
    let local_sync_lean = export_named_source(&local_sync_only, "cluster_clc_stage")
        .expect("transcribes nvvm.barrier0 before dsmemAccess into controlFlowCfg");
    lean_exporter::runner::check_generated_lean(&local_sync_lean, &lean_dir).expect_err(
        "Lean 4 must reject cross-CTA DSMEM read after CTA-local sync_threads() without cluster_sync()",
    );

    // 3b. Negative (Exiting CTA before peer DSMEM reads complete): omitting the trailing
    //     `cluster_sync()` before `mir.return` after DSMEM access fails `UniformBarriers`!
    let missing_exit_cluster_sync = CLUSTER_CLC_FIXTURE.replace(
        "        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.return () [] []: <() -> ()>",
        "        mir.return () [] []: <() -> ()>",
    );
    let missing_exit_lean = export_named_source(&missing_exit_cluster_sync, "cluster_clc_stage")
        .expect("transcribes missing exit cluster_sync after DSMEM read");
    lean_exporter::runner::check_generated_lean(&missing_exit_lean, &lean_dir).expect_err(
        "Lean 4 must reject returning after DSMEM access without a trailing cluster_sync()",
    );

    // 4. Negative (`CLC_RESPONSE` undersized in Lean 4): 8-byte `SharedArray<u64, 1>` for 16-byte CLC response!
    let undersized_clc = CLUSTER_CLC_FIXTURE.replace(
        "size: builtin.integer <2: i64>, alloc_key: builtin.string \"CLC_RESPONSE\"",
        "size: builtin.integer <1: i64>, alloc_key: builtin.string \"CLC_RESPONSE\"",
    );
    let undersized_clc_lean = export_named_source(&undersized_clc, "cluster_clc_stage")
        .expect("transcribes 8-byte CLC_RESPONSE against 16-byte clc_try_cancel");
    let err = lean_exporter::runner::check_generated_lean(&undersized_clc_lean, &lean_dir)
        .expect_err("Lean 4 must reject 16-byte clc_try_cancel into 8-byte CLC_RESPONSE");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for undersized CLC_RESPONSE: {err}"
    );

    // 5. Negative (`CLC_RESPONSE` read without `CLC_BAR` wait in Lean 4):
    let unawaited_clc = CLUSTER_CLC_FIXTURE.replacen(
        "mir.cond_br (clc_not_ready) [^wait_clc_spin, ^consume_clc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.goto () [^consume_clc] []: <() -> ()>",
        1,
    );
    let unawaited_clc_lean = export_named_source(&unawaited_clc, "cluster_clc_stage")
        .expect("transcribes unawaited CLC_BAR read into MirFunc");
    let err = lean_exporter::runner::check_generated_lean(&unawaited_clc_lean, &lean_dir)
        .expect_err("Lean 4 must reject reading CLC_RESPONSE without awaiting CLC_BAR");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for unawaited CLC_BAR: {err}"
    );

    // 6. Positive + 5 Negative 2-CTA `_cg2` `tcgen05` / `TMEM` lifecycle checks:
    //    Upgrade `tcgen05_matmul` to 2-CTA `cluster_dim_x = 2` (`M256_N128` -> `idesc` with `M=256`,
    //    `cluster_sync` before `tcgen05_alloc_cg2`, `cluster_sync`, `tcgen05_mma_f16_cg2`,
    //    `tcgen05_commit_shared_cluster_cg2`, `tcgen05_relinquish_alloc_permit_cg2`,
    //    `cluster_sync`, `tcgen05_dealloc_cg2`, trailing `cluster_sync` before return):
    let cg2_valid = TCGEN05_MATMUL_FIXTURE
        .replacen(
            "[gpu_kernel: builtin.string \"true\", ",
            "[cluster_dim_x: builtin.integer <2: ui32>, cluster_dim_y: builtin.integer <1: ui32>, cluster_dim_z: builtin.integer <1: ui32>, gpu_kernel: builtin.string \"true\", ",
            1,
        )
        .replacen(
            "      ^sync_after_init():\n        nvvm.barrier0 () [] []: <() -> ()>;",
            "      ^sync_after_init():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
            1,
        )
        .replacen("nvvm.tcgen05_alloc (", "nvvm.tcgen05_alloc_cg2 (", 1)
        .replacen(
            "      ^sync_after_alloc():\n        nvvm.barrier0 () [] []: <() -> ()>;",
            "      ^sync_after_alloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
            1,
        )
        .replacen("nvvm.tcgen05_mma_f16 (", "nvvm.tcgen05_mma_f16_cg2 (", 1)
        .replacen(
            "nvvm.tcgen05_commit_shared_cluster (",
            "nvvm.tcgen05_commit_shared_cluster_cg2 (",
            1,
        )
        .replacen(
            "ready_mma = nvvm.mbarrier_try_wait_parity_shared (mma_bar_w, zero_u32)",
            "ready_mma = nvvm.mbarrier_try_wait_parity_cluster (mma_bar_w, zero_u32)",
            1,
        )
        .replacen(
            "      ^sync_before_dealloc():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
            "      ^sync_before_dealloc():\n        mir.cond_br (is_warp0) [^relinquish_cg2, ^cluster_sync_dealloc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^relinquish_cg2():\n        nvvm.tcgen05_relinquish_alloc_permit_cg2 () [] []: <() -> ()>;\n        mir.goto () [^cluster_sync_dealloc] []: <() -> ()>\n\n      ^cluster_sync_dealloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
            1,
        )
        .replacen("nvvm.tcgen05_dealloc (", "nvvm.tcgen05_dealloc_cg2 (", 1)
        .replacen(
            "      ^exit():\n        mir.return () [] []: <() -> ()>",
            "      ^exit():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.return () [] []: <() -> ()>",
            1,
        );
    let cg2_lean = export_named_source(&cg2_valid, "tcgen05_matmul_verified")
        .expect("exports valid 2-CTA _cg2 tcgen05 kernel");
    assert!(cg2_lean.contains("(.tmemRelinquishPermit true)"));
    lean_exporter::runner::check_generated_lean(&cg2_lean, &lean_dir)
        .expect("Lean 4 proves valid 2-CTA _cg2 TmemLifecycle");

    // 7. Negative (`_cg2` missing `tcgen05_relinquish_alloc_permit_cg2`):
    let cg2_no_relinquish = cg2_valid.replacen(
        "        nvvm.tcgen05_relinquish_alloc_permit_cg2 () [] []: <() -> ()>;\n",
        "",
        1,
    );
    let cg2_no_rel_lean = export_named_source(&cg2_no_relinquish, "tcgen05_matmul_verified")
        .expect("transcribes _cg2 missing relinquish_alloc_permit_cg2");
    lean_exporter::runner::check_generated_lean(&cg2_no_rel_lean, &lean_dir).expect_err(
        "Lean 4 must reject tcgen05_dealloc_cg2 without tcgen05_relinquish_alloc_permit_cg2",
    );

    // 8. Negative (`_cg2` without `#[cluster_launch(2, 1, 1)]` -> `clusterDimX == 1`):
    let cg2_non_cluster = cg2_valid.replacen(
        "cluster_dim_x: builtin.integer <2: ui32>, cluster_dim_y: builtin.integer <1: ui32>, cluster_dim_z: builtin.integer <1: ui32>, ",
        "",
        1,
    );
    let cg2_non_cluster_lean = export_named_source(&cg2_non_cluster, "tcgen05_matmul_verified")
        .expect("transcribes _cg2 with clusterDimX = 1");
    lean_exporter::runner::check_generated_lean(&cg2_non_cluster_lean, &lean_dir)
        .expect_err("Lean 4 must reject _cg2 tcgen05 ops when clusterDimX != 2");

    // 9. Negative (`_cg2` using CTA-local `sync_threads()` instead of `cluster_sync()` before dealloc):
    let cg2_local_sync_dealloc = cg2_valid.replacen(
        "      ^cluster_sync_dealloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
        "      ^cluster_sync_dealloc():\n        nvvm.barrier0 () [] []: <() -> ()>;",
        1,
    );
    let cg2_local_dealloc_lean =
        export_named_source(&cg2_local_sync_dealloc, "tcgen05_matmul_verified")
            .expect("transcribes _cg2 dealloc preceded only by CTA-local barrier0");
    lean_exporter::runner::check_generated_lean(&cg2_local_dealloc_lean, &lean_dir).expect_err(
        "Lean 4 must reject tcgen05_dealloc_cg2 preceded only by CTA-local sync_threads()",
    );

    // 10. Negative (`_cg2` `mbarrier_init` before `tcgen05_alloc_cg2` with only CTA-local `barrier0`):
    let cg2_local_sync_init = cg2_valid.replacen(
        "      ^sync_after_init():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
        "      ^sync_after_init():\n        nvvm.barrier0 () [] []: <() -> ()>;",
        1,
    );
    let cg2_local_init_lean = export_named_source(&cg2_local_sync_init, "tcgen05_matmul_verified")
        .expect("transcribes _cg2 alloc preceded only by CTA-local barrier0 after mbarrier_init");
    lean_exporter::runner::check_generated_lean(&cg2_local_init_lean, &lean_dir).expect_err(
        "Lean 4 must reject tcgen05_alloc_cg2 after mbarrier_init without cluster_sync()",
    );

    // 11. Negative (`_cg2` omitting trailing `cluster_sync()` after `tcgen05_dealloc_cg2` before return):
    let cg2_no_exit_sync = cg2_valid.replacen(
        "      ^exit():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.return () [] []: <() -> ()>",
        "      ^exit():\n        mir.return () [] []: <() -> ()>",
        1,
    );
    let cg2_no_exit_lean = export_named_source(&cg2_no_exit_sync, "tcgen05_matmul_verified")
        .expect("transcribes _cg2 dealloc returning without trailing cluster_sync()");
    lean_exporter::runner::check_generated_lean(&cg2_no_exit_lean, &lean_dir).expect_err(
        "Lean 4 must reject returning after tcgen05_dealloc_cg2 without trailing cluster_sync()",
    );
}

#[test]
fn verifies_stage10_dual_commit_cg2_multicast_and_dynamic_tile_kind_bounds() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // Build a 2-CTA `_cg2` GEMM stage exercising Stage 10's dual-commit (`MMA_BAR` + `ACCUM_FULL`)
    // and leader-only `mbarrier_arrive_expect_tx` + `_cg2` multicast TMA:
    let base_cg2 = TCGEN05_MATMUL_FIXTURE
        .replacen(
            "[gpu_kernel: builtin.string \"true\", ",
            "[cluster_dim_x: builtin.integer <2: ui32>, cluster_dim_y: builtin.integer <1: ui32>, cluster_dim_z: builtin.integer <1: ui32>, gpu_kernel: builtin.string \"true\", ",
            1,
        )
        .replacen(
            "      ^sync_after_init():\n        nvvm.barrier0 () [] []: <() -> ()>;",
            "      ^sync_after_init():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
            1,
        )
        .replacen("nvvm.tcgen05_alloc (", "nvvm.tcgen05_alloc_cg2 (", 1)
        .replacen(
            "      ^sync_after_alloc():\n        nvvm.barrier0 () [] []: <() -> ()>;",
            "      ^sync_after_alloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;",
            1,
        )
        .replacen("nvvm.tcgen05_mma_f16 (", "nvvm.tcgen05_mma_f16_cg2 (", 1)
        .replacen(
            "        nvvm.tcgen05_commit_shared_cluster (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^wait_mma_hdr] []: <() -> ()>",
            "        nvvm.tcgen05_commit_shared_cluster_cg2 (mma_bar_p) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^commit_accum] []: <() -> ()>\n\n      ^commit_accum():\n        mma_bar_p2 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui64, size: builtin.integer <1: i64>, alloc_key: builtin.string \"MMA_BAR\", source_name: builtin.string \"safe_tcgen05_matmul::kernels::tcgen05_matmul_verified::MMA_BAR\"]: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;\n        nvvm.tcgen05_commit_shared_cluster_cg2 (mma_bar_p2) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        mir.goto () [^wait_mma_hdr] []: <() -> ()>",
            1,
        )
        .replacen(
            "ready_mma = nvvm.mbarrier_try_wait_parity_shared (mma_bar_w, zero_u32)",
            "ready_mma = nvvm.mbarrier_try_wait_parity_cluster (mma_bar_w, zero_u32)",
            1,
        )
        .replacen(
            "      ^sync_before_dealloc():\n        nvvm.barrier0 () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
            "      ^sync_before_dealloc():\n        mir.cond_br (is_warp0) [^relinquish_cg2, ^cluster_sync_dealloc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^relinquish_cg2():\n        nvvm.tcgen05_relinquish_alloc_permit_cg2 () [] []: <() -> ()>;\n        mir.goto () [^cluster_sync_dealloc] []: <() -> ()>\n\n      ^cluster_sync_dealloc():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.cond_br (is_warp0) [^dealloc_tmem, ^exit]",
            1,
        )
        .replacen("nvvm.tcgen05_dealloc (", "nvvm.tcgen05_dealloc_cg2 (", 1)
        .replacen(
            "      ^exit():\n        mir.return () [] []: <() -> ()>",
            "      ^exit():\n        nvvm.cluster_sync () [] []: <() -> ()>;\n        mir.return () [] []: <() -> ()>",
            1,
        );

    // 1. Positive: Dual `tcgen05_commit_shared_cluster_cg2` (`MMA_BAR` then post-loop `ACCUM_FULL`) passes Lean 4!
    let dual_commit_lean = export_named_source(&base_cg2, "tcgen05_matmul_verified")
        .expect("exports Stage 10 dual-commit _cg2 kernel");
    lean_exporter::runner::check_generated_lean(&dual_commit_lean, &lean_dir)
        .expect("Lean 4 proves Stage 10 dual-commit _cg2 TmemLifecycle");

    // 1b. Negative: `.ctaAcquire` wait on `_cg2` `MMA_BAR` fails `TmemLifecycle`!
    let cg2_cta_wait = base_cg2.replacen(
        "ready_mma = nvvm.mbarrier_try_wait_parity_cluster (mma_bar_w, zero_u32)",
        "ready_mma = nvvm.mbarrier_try_wait_parity_shared (mma_bar_w, zero_u32)",
        1,
    );
    let cg2_cta_wait_lean = export_named_source(&cg2_cta_wait, "tcgen05_matmul_verified")
        .expect("transcribes .cta wait on _cg2 MMA_BAR");
    lean_exporter::runner::check_generated_lean(&cg2_cta_wait_lean, &lean_dir)
        .expect_err("Lean 4 must reject .cta wait on _cg2 committed MMA_BAR");

    // 1c. Negative: Missing `tcgen05_fence_after_thread_sync` between `MMA_BAR` wait and `tcgen05_ld` fails!
    let cg2_missing_after_fence = base_cg2.replacen(
        "      ^epilogue_init():\n        nvvm.tcgen05_fence_after_thread_sync () [] []: <() -> ()>;\n",
        "      ^epilogue_init():\n",
        1,
    );
    let cg2_no_fence_lean =
        export_named_source(&cg2_missing_after_fence, "tcgen05_matmul_verified")
            .expect("transcribes missing tcgen05_fence_after_thread_sync before tcgen05_ld");
    lean_exporter::runner::check_generated_lean(&cg2_no_fence_lean, &lean_dir).expect_err(
        "Lean 4 must reject tcgen05_ld after mbarrier wait without tcgen05_fence_after_thread_sync",
    );

    // 2. Negative: `tcgen05_commit_shared_cluster_cg2` with zero `tcgen05_mma` anywhere in the kernel fails `TmemLifecycle`!
    let commit_without_any_mma = base_cg2.replacen(
        "        nvvm.tcgen05_mma_f16_cg2 (tmem_addr, a_desc, b_desc, idesc, false_i1) [] []: <(builtin.integer ui32, builtin.integer ui64, builtin.integer ui64, builtin.integer ui32, builtin.integer i1) -> ()>;\n",
        "",
        1,
    );
    let no_mma_lean = export_named_source(&commit_without_any_mma, "tcgen05_matmul_verified")
        .expect("transcribes tcgen05_commit without any tcgen05_mma");
    lean_exporter::runner::check_generated_lean(&no_mma_lean, &lean_dir)
        .expect_err("Lean 4 must reject tcgen05_commit when no tcgen05_mma exists on the path");

    // 3. Negative: 2-CTA `_cg2` TMA copy on a barrier that is NEVER armed by `mbarrier_arrive_expect_tx` fails!
    let cg2_tma_never_armed = base_cg2
        .replacen(
            "        _tok_a = nvvm.mbarrier_arrive_expect_tx_shared (tma_a_bar_p, c4096_u32) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> (builtin.integer ui64)>;\n",
            "",
            1,
        )
        .replacen(
            "nvvm.cp_async_bulk_tensor_g2s_cta_tile_2d (smem_a_tma, tma_a_bar_p, a_tmap, tile_x, tile_y, ch_zero) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>, mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, mir.ptr <builtin.integer ui8,mutable:false,addrspace:0,kind:SharedRef>, builtin.integer si32, builtin.integer si32, builtin.integer ui64) -> ()>;",
            "self_mask = mir.constant () [] [value: builtin.integer <3: ui16>]: <() -> (builtin.integer ui16)>;\n        nvvm.cp_async_bulk_tensor_g2s_tile_2d_multicast_cg2 (smem_a_tma, tma_a_bar_p, a_tmap, tile_x, tile_y, self_mask, ch_zero) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>, mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, mir.ptr <builtin.integer ui8,mutable:false,addrspace:0,kind:SharedRef>, builtin.integer si32, builtin.integer si32, builtin.integer ui16, builtin.integer ui64) -> ()>;",
            1,
        );
    let unarmed_cg2_lean = export_named_source(&cg2_tma_never_armed, "tcgen05_matmul_verified")
        .expect("transcribes _cg2 multicast TMA without any mbarrier_arrive_expect_tx");
    lean_exporter::runner::check_generated_lean(&unarmed_cg2_lean, &lean_dir).expect_err(
        "Lean 4 must reject _cg2 multicast TMA when barrier has zero expect_tx arrivals",
    );

    // 4. Negative: Data-dependent `u32` load from `SharedArray` (`tile_kind`) selecting OOB `192` columns
    //    on `128`-column `TMEM` (`if tile_kind == 1 { 64 } else { 192 }`) fails `MirNoOobTmem` in Lean 4!
    let oob_dyn_tmem = base_cg2.replacen(
        "      ^col_hdr(col: builtin.integer ui64):\n        c128_u64 = mir.constant () [] [value: builtin.integer <128: ui64>]: <() -> (builtin.integer ui64)>;\n        in_col = mir.lt (col, c128_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;",
        "      ^col_hdr(col: builtin.integer ui64):\n        c128_u64 = mir.constant () [] [value: builtin.integer <128: ui64>]: <() -> (builtin.integer ui64)>;\n        kind_raw = mir.load (tmem_elem_ptr) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.integer ui64)>;\n        one_k = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        is_n192 = mir.eq (kind_raw, one_k) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n        mir.cond_br (is_n192) [^bound_64, ^bound_192] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^bound_64():\n        b64 = mir.constant () [] [value: builtin.integer <64: ui64>]: <() -> (builtin.integer ui64)>;\n        mir.goto (b64) [^col_cmp] []: <(builtin.integer ui64) -> ()>\n\n      ^bound_192():\n        b192 = mir.constant () [] [value: builtin.integer <192: ui64>]: <() -> (builtin.integer ui64)>;\n        mir.goto (b192) [^col_cmp] []: <(builtin.integer ui64) -> ()>\n\n      ^col_cmp(dyn_cols: builtin.integer ui64):\n        in_col = mir.lt (col, dyn_cols) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;",
        1,
    );
    // Note `tmem_elem_ptr` is the `TMEM_ADDR` slot (`tmem_slot_cols`); use `SMEM_A` for a plain `SharedArray` load:
    let oob_dyn_tmem = oob_dyn_tmem.replacen(
        "kind_raw = mir.load (tmem_elem_ptr) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:3,kind:SharedRef>) -> (builtin.integer ui64)>;",
        "smem_k_raw = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui8, size: builtin.integer <4096: i64>, alloc_key: builtin.string \"SMEM_A\", source_name: builtin.string \"safe_tcgen05_matmul::kernels::tcgen05_matmul_verified::SMEM_A\"]: <() -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>)>;\n        smem_k_elem = mir.ptr_offset (smem_k_raw, zero_u64) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>)>;\n        kind_u8 = mir.load (smem_k_elem) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.integer ui8)>;\n        kind_raw = mir.cast (kind_u8) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui8) -> (builtin.integer ui64)>;",
        1,
    );
    let oob_dyn_lean = export_named_source(&oob_dyn_tmem, "tcgen05_matmul_verified")
        .expect("transcribes dynamic tile_kind 64/192 TMEM bound");
    let err = lean_exporter::runner::check_generated_lean(&oob_dyn_lean, &lean_dir)
        .expect_err("Lean 4 must reject dynamic tile_kind else-branch 192 > 128 TMEM columns");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for OOB dynamic tile_kind TMEM bound: {err}"
    );
}

#[test]
fn proof_cache_records_valid_proofs_and_invalidates_on_mutation() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    let cache_dir = std::env::temp_dir().join(format!(
        "cuda_oxide_lean_cache_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&cache_dir);

    let valid_lean = export_named_source(VECADD_FIXTURE, "vecadd").expect("exports vecadd");
    lean_exporter::runner::check_generated_lean_in_cache(&valid_lean, &lean_dir, Some(&cache_dir))
        .expect("cold proof check succeeds and populates cache");
    let entries_after_cold = std::fs::read_dir(&cache_dir)
        .expect("cache dir created")
        .count();
    assert_eq!(entries_after_cold, 1, "expected 1 SHA-256 .ok cache marker");

    let t0 = std::time::Instant::now();
    lean_exporter::runner::check_generated_lean_in_cache(&valid_lean, &lean_dir, Some(&cache_dir))
        .expect("warm cached proof check succeeds");
    assert!(
        t0.elapsed().as_millis() < 250,
        "cached proof check must skip spawning Lean"
    );

    let broken_lean = valid_lean.replacen(
        "requires := [{ cmp := .ge, lhs := (.sliceLen 0), rhs := (.sliceLen 2) }, { cmp := .ge, lhs := (.sliceLen 1), rhs := (.sliceLen 2) }]",
        "requires := []",
        1,
    );
    assert_ne!(
        broken_lean, valid_lean,
        "requires mutation must modify Lean source"
    );
    lean_exporter::runner::check_generated_lean_in_cache(&broken_lean, &lean_dir, Some(&cache_dir))
        .expect_err("mutated Lean source must miss cache and fail proof check");

    let _ = std::fs::remove_dir_all(&cache_dir);
}

#[test]
fn audit_hardening_rejects_early_return_uninit_smem_mbar_inval_checked_ovf_2d_lane_ptr_cast_and_calls()
 {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Pre-guard early return `if idx == 0 { return; }` before `if idx < c.len()` must fail `MirWritesAllElems`:
    let early_ret_vecadd = VECADD_FIXTURE.replacen(
        "    in_bounds = mir.lt (idx, c_len)",
        "    zero_idx = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;\n    is_t0 = mir.eq (idx, zero_idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n    mir.cond_br (is_t0) [^exit, ^chk_bounds] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^chk_bounds():\n    in_bounds = mir.lt (idx, c_len)",
        1,
    );
    let early_ret_lean = export_named_source(&early_ret_vecadd, "vecadd")
        .expect("transcribes pre-guard early return");
    lean_exporter::runner::check_generated_lean(&early_ret_lean, &lean_dir)
        .expect_err("Lean 4 must reject `if idx == 0 { return; }` under writes_all_elems");

    // 2. Writing constant `c[0]` instead of `c[idx]` must fail `MirWritesAllElems`:
    let const_slot_vecadd = VECADD_FIXTURE.replacen(
        "c_elem = mir.ptr_offset (c_ptr, idx)",
        "zero_slot = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;\n    c_elem = mir.ptr_offset (c_ptr, zero_slot)",
        1,
    );
    let const_slot_lean =
        export_named_source(&const_slot_vecadd, "vecadd").expect("transcribes constant c[0] write");
    lean_exporter::runner::check_generated_lean(&const_slot_lean, &lean_dir)
        .expect_err("Lean 4 must reject constant `c[0]` write under writes_all_elems");

    // 3. Reading uninitialized `SharedArray` (`TILE_A` never written before `mir.load`) fails `UniformBarriers`:
    let uninit_smem = SGEMM_TILED_FIXTURE.replacen(
        "        mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;\n",
        "",
        1,
    );
    let uninit_smem_lean = export_named_source(&uninit_smem, "sgemm_tiled_views")
        .expect("transcribes uninitialized TILE_A read");
    lean_exporter::runner::check_generated_lean(&uninit_smem_lean, &lean_dir)
        .expect_err("Lean 4 must reject reading uninitialized SharedArray TILE_A");

    // 4. `mbarrier_inval` before `mbarrier_arrive_expect_tx` resets `BAR` to `.uninit` and fails `UniformBarriers`:
    let inval_bar = ASYNC_COPY_FIXTURE.replacen(
        "        _tok = nvvm.mbarrier_arrive_expect_tx_shared (bar_prod_raw, c16384_u32)",
        "        nvvm.mbarrier_inval_shared (bar_prod_raw) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> ()>;\n        _tok = nvvm.mbarrier_arrive_expect_tx_shared (bar_prod_raw, c16384_u32)",
        1,
    );
    let inval_bar_lean = export_named_source(&inval_bar, "async_copy_stage")
        .expect("transcribes mbarrier_inval before expect_tx");
    assert!(inval_bar_lean.contains("(.mbarInval 0)"));
    lean_exporter::runner::check_generated_lean(&inval_bar_lean, &lean_dir)
        .expect_err("Lean 4 must reject arriving on an invalidated Barrier");

    // 5. `mbarrier_init(&BAR, 2)` arrived only once by a single `tid == 0` thread fails `validMbarArrivalCounts`:
    let bad_init_count = ASYNC_COPY_FIXTURE.replacen(
        "c1_u32 = mir.constant () [] [value: builtin.integer <1: ui32>]",
        "c1_u32 = mir.constant () [] [value: builtin.integer <2: ui32>]",
        1,
    );
    let bad_cnt_lean = export_named_source(&bad_init_count, "async_copy_stage")
        .expect("transcribes count = 2 mbarrier_init");
    lean_exporter::runner::check_generated_lean(&bad_cnt_lean, &lean_dir).expect_err(
        "Lean 4 must reject mbarrier_init(count = 2) with single non-cyclic leader arrival",
    );

    // 6. Multi-warp `lane_id() == 0` (8 threads in a 256-thread block) is NOT a single-thread leader:
    let lane_only_leader = ASYNC_COPY_FIXTURE.replacen(
        "        tx_u32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;\n        tid = mir.cast (tx_u32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
        "        tx_u32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;\n        lane_u32 = nvvm.read_ptx_sreg_laneid () [] []: <() -> (builtin.integer ui32)>;\n        tid = mir.cast (lane_u32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
        1,
    );
    let lane_leader_lean = export_named_source(&lane_only_leader, "async_copy_stage")
        .expect("transcribes lane == 0 multi-warp TMA producer");
    lean_exporter::runner::check_generated_lean(&lane_leader_lean, &lean_dir).expect_err(
        "Lean 4 must reject `lane_id() == 0` as a single-thread TMA leader in a 256-thread CTA",
    );

    // 7. `tcgen05_dealloc(tmem_addr + 16, 128)` (non-zero offset) fails `MirNoOobTmem`,
    //    and `tcgen05_fence_before_thread_sync` emits `.tmemFenceBeforeSync`:
    let dealloc_nonzero = TCGEN05_MATMUL_FIXTURE
        .replacen(
            "        nvvm.tcgen05_dealloc (tmem_addr, c128_dealloc)",
            "        nvvm.tcgen05_fence_before_thread_sync () [] []: <() -> ()>;\n        off16 = mir.constant () [] [value: builtin.integer <16: ui32>]: <() -> (builtin.integer ui32)>;\n        bad_dealloc_addr = mir.add (tmem_addr, off16) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;\n        nvvm.tcgen05_dealloc (bad_dealloc_addr, c128_dealloc)",
            1,
        );
    let dealloc_nonzero_lean = export_named_source(&dealloc_nonzero, "tcgen05_matmul_verified")
        .expect("transcribes tcgen05_dealloc with non-zero offset + tcgen05_fence");
    assert!(dealloc_nonzero_lean.contains(".tmemFenceBeforeSync"));
    lean_exporter::runner::check_generated_lean(&dealloc_nonzero_lean, &lean_dir)
        .expect_err("Lean 4 must reject tcgen05_dealloc on non-zero TMEM offset");

    // 8. `tcgen05_ld` inside a sub-warp lane-divergent `if tid == 0` branch fails `TmemLifecycle`:
    let divergent_tmem_ld = TCGEN05_MATMUL_FIXTURE.replacen(
        "      ^col_body():\n        col_u32 = mir.cast (col)",
        "      ^col_body():\n        z64_d = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;\n        is_t0_d = mir.eq (tid, z64_d) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;\n        mir.cond_br (is_t0_d) [^ld_t0, ^sync_before_dealloc] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n      ^ld_t0():\n        col_u32 = mir.cast (col)",
        1,
    );
    let div_ld_lean = export_named_source(&divergent_tmem_ld, "tcgen05_matmul_verified")
        .expect("transcribes lane-divergent tcgen05_ld");
    lean_exporter::runner::check_generated_lean(&div_ld_lean, &lean_dir)
        .expect_err("Lean 4 must reject sub-warp lane-divergent tcgen05_ld");

    // 9. `mir.checked_add` on `255u8 + 1u8` sets `.1` (`checkedOvf`) to `true` (not `false`!):
    let checked_ovf_mir = r#"
builtin.module @checked_ovf_test {
  ^entry():
    mir.func @ovf_k: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_idx: builtin.string "0"]
    {
      ^entry(a: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>):
        c255 = mir.constant () [] [value: builtin.integer <255: ui8>]: <() -> (builtin.integer ui8)>;
        c1 = mir.constant () [] [value: builtin.integer <1: ui8>]: <() -> (builtin.integer ui8)>;
        chk = mir.checked_add (c255, c1) [] []: <(builtin.integer ui8, builtin.integer ui8) -> (mir.tuple <[builtin.integer ui8, builtin.integer i1], [], [], 2, 1>)>;
        ovf = mir.extract_field (chk) [] [index: mir.field_index 1]: <(mir.tuple <[builtin.integer ui8, builtin.integer i1], [], [], 2, 1>) -> (builtin.integer i1)>;
        mir.cond_br (ovf) [^bad_oob, ^safe_ret] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^bad_oob():
        a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
        a_len = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
        oob_p = mir.ptr_offset (a_ptr, a_len) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
        _v = mir.load (oob_p) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
        mir.return () [] []: <() -> ()>

      ^safe_ret():
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let ovf_lean = export_named_source(checked_ovf_mir, "ovf_k")
        .expect("transcribes mir.checked_add with real CheckedOvf");
    assert!(ovf_lean.contains(".checkedOvf .add 8"));
    lean_exporter::runner::check_generated_lean(&ovf_lean, &lean_dir)
        .expect_err("Lean 4 must see `255u8 + 1u8` overflow = true and reject OOB branch");

    // 10. 2D `laneId` (`16x16` block): `(tx = 0, ty = 1)` has `laneId = 16`, colliding with `TILE_A[16]`:
    let lane_2d_race = r#"
builtin.module @lane_2d_test {
  ^entry():
    mir.func @lane2d_k: builtin.function <(builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <2: ui32>, reqntid_x: builtin.integer <16: ui32>, reqntid_y: builtin.integer <16: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_shared: builtin.string "0", verify_no_data_race: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(_n: builtin.integer ui32):
        tx = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
        ty = nvvm.read_ptx_sreg_tid_y () [] []: <() -> (builtin.integer ui32)>;
        lane = nvvm.read_ptx_sreg_laneid () [] []: <() -> (builtin.integer ui32)>;
        z32 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
        one32 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
        is_row1_col0 = mir.eq (tx, z32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (is_row1_col0) [^write_by_lane, ^other] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^write_by_lane():
        is_ty1 = mir.eq (ty, one32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (is_ty1) [^do_w1, ^sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^do_w1():
        smem0 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui32, size: builtin.integer <32: i64>, alloc_key: builtin.string "S", source_name: builtin.string "k::S"]: <() -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>)>;
        lane64 = mir.cast (lane) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
        p1 = mir.ptr_offset (smem0, lane64) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>)>;
        mir.store (p1, one32) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;
        mir.goto () [^sync] []: <() -> ()>

      ^other():
        is_ty0 = mir.eq (ty, z32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (is_ty0) [^do_w2, ^sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^do_w2():
        smem1 = mir.shared_alloc () [] [elem_type: builtin.type builtin.integer ui32, size: builtin.integer <32: i64>, alloc_key: builtin.string "S", source_name: builtin.string "k::S"]: <() -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>)>;
        c16 = mir.constant () [] [value: builtin.integer <16: ui64>]: <() -> (builtin.integer ui64)>;
        p2 = mir.ptr_offset (smem1, c16) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>)>;
        mir.store (p2, one32) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;
        mir.goto () [^sync] []: <() -> ()>

      ^sync():
        nvvm.barrier0 () [] []: <() -> ()>;
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let lane_2d_lean =
        export_named_source(lane_2d_race, "lane2d_k").expect("transcribes 2D laneId kernel");
    lean_exporter::runner::check_generated_lean(&lane_2d_lean, &lean_dir)
        .expect_err("Lean 4 must catch 2D `(tx=0, ty=1)` laneId == 16 race against S[16]");

    // 11. `PtrToPtr` cast changing pointee byte width (`*mut fp32` -> `*mut u8`) before `mir.ptr_offset` fails closed:
    let bad_ptr_stride = SGEMM_TILED_FIXTURE.replacen(
        "        slot_a = mir.ptr_offset (tile_a_mut, smem_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>)>;\n        mir.store (slot_a, alpha) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>, builtin.fp32) -> ()>;",
        "        tile_a_u8 = mir.cast (tile_a_mut) [] [cast_kind: mir.cast_kind PtrToPtr]: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:UniqueRef>) -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>)>;\n        slot_a = mir.ptr_offset (tile_a_u8, smem_idx) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>)>;\n        z_u8 = mir.constant () [] [value: builtin.integer <0: ui8>]: <() -> (builtin.integer ui8)>;\n        mir.store (slot_a, z_u8) [] []: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui8) -> ()>;",
        1,
    );
    let stride_err = export_named_source(&bad_ptr_stride, "sgemm_tiled_views").expect_err(
        "PtrToPtr cast changing pointee element size before ptr_offset must fail closed",
    );
    assert!(
        stride_err
            .to_string()
            .contains("element size from 4B to 1B"),
        "unexpected PtrToPtr stride error: {stride_err}"
    );

    // 12. Uninlined 0-result `mir.call` (`helper_with_barrier`) and non-`cuda_device` `fake_index_1d` both fail closed:
    let uninlined_zero_call = VECADD_FIXTURE.replacen(
        "    in_bounds = mir.lt (idx, c_len)",
        "    mir.call (idx) [] [callee: builtin.string \"user_crate::helper_with_barrier\"]: <(builtin.integer ui64) -> ()>;\n    in_bounds = mir.lt (idx, c_len)",
        1,
    );
    let call_err = export_named_source(&uninlined_zero_call, "vecadd")
        .expect_err("uninlined 0-result helper mir.call must fail closed");
    assert!(
        call_err.to_string().contains("unmodeled function call"),
        "unexpected mir.call error: {call_err}"
    );

    let fake_idx1d = VECADD_FIXTURE.replacen(
        "_RINvNtNtCs11cuda_device6thread10___internal8index_1d",
        "user_crate::fake_index_1d",
        1,
    );
    let fake_err = export_named_source(&fake_idx1d, "vecadd")
        .expect_err("non-cuda_device function named fake_index_1d must fail closed");
    assert!(fake_err.to_string().contains("unmodeled function call"));
}

#[test]
fn verifies_reverse_writes_all_elems_and_rejects_trapping_asserts_off_by_one_and_2d_coverage() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive: reverse-indexed `c[c.len() - 1 - idx] = a[idx] + b[idx]` with flipped `mir.gt (c_len, idx)`
    //    and uninlined `ThreadIndex::get` (`mir.alloca` + `mir.store` + `mir.cast` + `mir.call`) proves all obligations:
    let rev_mir = r#"
mir.func @vecadd_rev: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=2,1>=2", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "2", verify_no_oob_idx: builtin.string "0,1,2", verify_writes_all_elems: builtin.string "2"]
{
  ^entry(a: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, b: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, c: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    lc = mir.alloca () [] []: <() -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>)>;
    idx_slot = mir.alloca () [] []: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:Erased>)>;
    lc_ref = mir.cast (lc) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>) -> (mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>)>;
    tid_wrapper = mir.call (lc_ref) [] [callee: builtin.string "_RINvNtNtCs11cuda_device6thread10___internal8index_1d"]: <(mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>) -> (builtin.integer ui64)>;
    mir.store (idx_slot, tid_wrapper) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:Erased>, builtin.integer ui64) -> ()>;
    mir.goto () [^get_blk] []: <() -> ()>

  ^get_blk():
    slot_ref = mir.cast (idx_slot) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:Erased>) -> (mir.ptr <builtin.integer ui64 ,mutable:false,addrspace:0,kind:SharedRef>)>;
    idx = mir.call (slot_ref) [] [callee: builtin.string "_RNvMs8_NtCs11cuda_device6threadNtB5_11ThreadIndex3get"]: <(mir.ptr <builtin.integer ui64 ,mutable:false,addrspace:0,kind:SharedRef>) -> (builtin.integer ui64)>;
    c_len = mir.extract_field (c) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_bounds = mir.gt (c_len, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_bounds) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    a_len = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    a_ok = mir.lt (idx, a_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (a_ok) [] []: <(builtin.integer i1) -> ()>;
    a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_elem = mir.ptr_offset (a_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    a_val = mir.load (a_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    b_ptr = mir.extract_field (b) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    b_elem = mir.ptr_offset (b_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    b_val = mir.load (b_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    one = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;
    last = mir.sub (c_len, one) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    rev_idx = mir.sub (last, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    c_ok = mir.lt (rev_idx, c_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (c_ok) [] []: <(builtin.integer i1) -> ()>;
    sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;
    c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    c_elem = mir.ptr_offset (c_ptr, rev_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (c_elem, sum) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"#;
    let rev_lean = export_named_source(rev_mir, "vecadd_rev")
        .expect("transcribes reverse-indexed vecadd with ThreadIndex::get");
    assert!(rev_lean.contains("theorem verify_writes_all_elems_param2"));
    assert!(rev_lean.contains("theorem verify_assert_redundant_1"));
    lean_exporter::runner::check_generated_lean(&rev_lean, &lean_dir)
        .expect("Lean 4 must verify reverse-indexed writes_all_elems and bounds elision");

    // 2. Negative: off-by-one reverse index `c[c.len() - 2 - idx]` fails in Lean 4:
    let rev_off_by_two = rev_mir.replacen(
        "one = mir.constant () [] [value: builtin.integer <1: ui64>]",
        "one = mir.constant () [] [value: builtin.integer <2: ui64>]",
        1,
    );
    let rev_oob_lean = export_named_source(&rev_off_by_two, "vecadd_rev")
        .expect("transcribes off-by-one reverse write");
    lean_exporter::runner::check_generated_lean(&rev_oob_lean, &lean_dir)
        .expect_err("Lean 4 must reject off-by-one `c[c.len() - 2 - idx]`");

    // 3. Negative: guard `idx + 1 < c.len()` skips `idx = c.len() - 1` (leaving `c[0]` unwritten) and fails `MirWritesAllElems`:
    let rev_skip_last = rev_mir.replacen(
        "    in_bounds = mir.gt (c_len, idx)",
        "    one_g = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    idx_p1 = mir.add (idx, one_g) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    in_bounds = mir.gt (c_len, idx_p1)",
        1,
    );
    let rev_skip_lean = export_named_source(&rev_skip_last, "vecadd_rev")
        .expect("transcribes `idx + 1 < c.len()` reverse write");
    lean_exporter::runner::check_generated_lean(&rev_skip_lean, &lean_dir)
        .expect_err("Lean 4 must reject `idx + 1 < c.len()` missing `c[0]` writer");

    // 4. Negative (soundness): unverified `a[idx]` retaining `mir.assert(idx < a.len())` without `a.len() >= c.len()`
    //    traps when `a.len() == 0 < c.len()` and must fail `MirWritesAllElems`:
    let trapping_partial = rev_mir
        .replacen(
            "contract_requires_len_ge: builtin.string \"0>=2,1>=2\", ",
            "contract_requires_len_ge: builtin.string \"1>=2\", ",
            1,
        )
        .replacen(
            "verify_no_oob_idx: builtin.string \"0,1,2\"",
            "verify_no_oob_idx: builtin.string \"1,2\"",
            1,
        );
    let trapping_lean = export_named_source(&trapping_partial, "vecadd_rev")
        .expect("transcribes kernel with unproven mir.assert on `a`");
    lean_exporter::runner::check_generated_lean(&trapping_lean, &lean_dir).expect_err(
        "Lean 4 must reject `writes_all_elems(c)` when an earlier `mir.assert` on `a` can trap",
    );

    // 5. Negative (soundness): `writes_all_elems` on `launch_contract_domain = 2` is rejected fail-closed:
    let dom2_wae = rev_mir.replacen(
        "launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>",
        "launch_contract_domain: builtin.integer <2: ui32>, reqntid_x: builtin.integer <16: ui32>, reqntid_y: builtin.integer <16: ui32>",
        1,
    );
    let dom2_err = export_named_source(&dom2_wae, "vecadd_rev")
        .expect_err("writes_all_elems on domain = 2 must fail closed");
    assert!(
        dom2_err
            .to_string()
            .contains("`writes_all_elems` requires a 1D launch contract (`domain = 1`)"),
        "unexpected domain=2 error: {dom2_err}"
    );
}

/// `a.as_ptr().add(idx + 1_000_000)` on a slice parameter named in `#[verify]`: slice -> raw slice
/// -> thin pointer casts, then `mir.ptr_offset` and `mir.load`. The model tracks no access on it.
fn with_raw_far_read(fixture: &str) -> String {
    fixture
        .replacen(
            "    b_ptr = mir.extract_field (b)",
            concat!(
                "    raw_slice = mir.cast (a) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RawAddress]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>)>;\n",
                "    raw_ptr = mir.cast (raw_slice) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>)>;\n",
                "    far = mir.constant () [] [value: builtin.integer <1000000: ui64>]: <() -> (builtin.integer ui64)>;\n",
                "    far_idx = mir.add (idx, far) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n",
                "    far_elem = mir.ptr_offset (raw_ptr, far_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>)>;\n",
                "    far_val = mir.load (far_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>) -> (builtin.fp32)>;\n",
                "    b_ptr = mir.extract_field (b)",
            ),
            1,
        )
        .replacen(
            "    mir.store (c_elem, sum) [] []:",
            "    sum_far = mir.add (sum, far_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;\n    mir.store (c_elem, sum_far) [] []:",
            1,
        )
}

/// Verify that `as_ptr` / `as_mut_ptr` casts on verified slice parameters transcribe
/// 1-to-1 into `MirRhs::SlicePtr`, so Lean 4 proves in-bounds raw pointer accesses and
/// rejects out-of-bounds raw pointer reads and stores.
#[test]
fn verifies_safe_raw_slice_pointers_and_rejects_oob_immutable_and_exposed_accesses() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };

    // 1. Positive: in-bounds `a.as_ptr().add(idx)` under `idx < c.len()` + `a.len() >= c.len()`
    //    transcribes to `MirRhs::SlicePtr(0)` + `MirRhs::PtrOffset` and verifies in Lean 4.
    let safe_raw_read = VECADD_FIXTURE.replacen(
        "    a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    a_elem = mir.ptr_offset (a_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    a_val = mir.load (a_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;",
        concat!(
            "    a_raw_slice = mir.cast (a) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RawAddress]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>)>;\n",
            "    a_ptr = mir.cast (a_raw_slice) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>)>;\n",
            "    a_elem = mir.ptr_offset (a_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>)>;\n",
            "    a_val = mir.load (a_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>) -> (builtin.fp32)>;",
        ),
        1,
    );
    let safe_lean = export_named_source(&safe_raw_read, "vecadd")
        .expect("safe raw-pointer read from verified slice exports cleanly");
    assert!(safe_lean.contains("(.slicePtr 0)"));
    lean_exporter::runner::check_generated_lean(&safe_lean, &lean_dir)
        .expect("Lean 4 proves in-bounds `a.as_ptr().add(idx)` read");

    // 2. Negative: `a.as_ptr().add(idx + 1_000_000)` exports and fails `verify_no_oob_idx_param0` in Lean 4.
    let raw_far = with_raw_far_read(VECADD_FIXTURE);
    let raw_far_lean =
        export_named_source(&raw_far, "vecadd").expect("exports OOB raw-pointer read for Lean VC");
    let far_err = lean_exporter::runner::check_generated_lean(&raw_far_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB `a.as_ptr().add(idx + 1_000_000)`");
    assert!(
        far_err.to_string().contains(
            "'CudaOxide.Generated.verify_no_oob_idx_param0' depends on axioms: [propext, sorryAx"
        ),
        "expected `verify_no_oob_idx_param0` failure on OOB raw read, got: {far_err}"
    );

    // 3. Negative: OOB `c.as_mut_ptr().add(1_000_000)` on `&mut [u32]` fails `verify_no_oob_idx_param0` in Lean 4.
    let raw_store = r#"
builtin.module @raw_store_test {
  ^entry():
    mir.func @raw_store_k: builtin.function <(mir.slice <builtin.integer ui32 ,mutable:true,kind:UniqueRef>) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_idx: builtin.string "0", verify_disjoint_writes: builtin.string "0"]
    {
      ^entry(c: mir.slice <builtin.integer ui32 ,mutable:true,kind:UniqueRef>):
        one = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
        raw_slice = mir.cast (c) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RawAddress]: <(mir.slice <builtin.integer ui32 ,mutable:true,kind:UniqueRef>) -> (mir.slice <builtin.integer ui32 ,mutable:true,kind:RawMut>)>;
        raw_ptr = mir.cast (raw_slice) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.slice <builtin.integer ui32 ,mutable:true,kind:RawMut>) -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:0,kind:RawMut>)>;
        far = mir.constant () [] [value: builtin.integer <1000000: ui64>]: <() -> (builtin.integer ui64)>;
        far_elem = mir.ptr_offset (raw_ptr, far) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:0,kind:RawMut>)>;
        mir.store (far_elem, one) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui32) -> ()>;
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let raw_store_lean = export_named_source(raw_store, "raw_store_k")
        .expect("exports OOB raw-pointer store on `&mut [u32]`");
    let store_err = lean_exporter::runner::check_generated_lean(&raw_store_lean, &lean_dir)
        .expect_err("Lean 4 must reject OOB `c.as_mut_ptr().add(1_000_000)` store");
    assert!(
        store_err
            .to_string()
            .contains("'CudaOxide.Generated.verify_no_oob_idx_param0' depends on axioms"),
        "expected `verify_no_oob_idx_param0` failure on OOB raw store, got: {store_err}"
    );

    // 4. Fail-closed guards: mutating through `a: &[f32]` and `expose_provenance` on slice pointers.
    let mut_through_shared = VECADD_FIXTURE.replacen(
        "    b_ptr = mir.extract_field (b)",
        concat!(
            "    a_rs = mir.cast (a) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RawAddress]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>)>;\n",
            "    a_rp = mir.cast (a_rs) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.slice <builtin.fp32 ,mutable:false,kind:RawConst>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>)>;\n",
            "    a_mut = mir.cast (a_rp) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:RawConst>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>)>;\n",
            "    mir.store (a_mut, a_val) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:RawMut>, builtin.fp32) -> ()>;\n",
            "    b_ptr = mir.extract_field (b)",
        ),
        1,
    );
    let ro_err = export_named_source(&mut_through_shared, "vecadd")
        .expect_err("writing through `*const T as *mut T` on `&[T]` must fail closed");
    assert!(ro_err.to_string().contains("read-only slice parameter 0"));

    let exposed = VECADD_FIXTURE.replacen(
        "    b_ptr = mir.extract_field (b)",
        "    _exp = mir.cast (a_elem) [] [cast_kind: mir.cast_kind PointerExposeAddress]: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.integer ui64)>;\n    b_ptr = mir.extract_field (b)",
        1,
    );
    let exp_err = export_named_source(&exposed, "vecadd")
        .expect_err("exposing provenance of a verified slice pointer must fail closed");
    assert!(exp_err.to_string().contains("pointer provenance exposure"));

    // 5. Block-argument provenance lattice: merging a verified `GlobalPtr(0)` with `UnverifiedPtr`
    //    (`core::ptr::null()`) saturates to `ConflictingPtr` and fails closed on dereference.
    let mixed_phi = r#"
builtin.module @mixed_phi_test {
  ^entry():
    mir.func @mixed_phi_k: builtin.function <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>, builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_idx: builtin.string "0"]
    {
      ^entry(a: mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>, flag: builtin.integer ui32):
        zero32 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
        cond = mir.eq (flag, zero32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        a_len = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
        zero64 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        ok0 = mir.lt (zero64, a_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
        mir.cond_br (ok0) [^guard_ok, ^done] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>
      ^guard_ok():
        a_ptr0 = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>)>;
        _v0 = mir.load (a_ptr0) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.integer ui32)>;
        mir.cond_br (cond) [^arm_verified, ^arm_null] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>
      ^arm_verified():
        rs = mir.cast (a) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RawAddress]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> (mir.slice <builtin.integer ui32 ,mutable:false,kind:RawConst>)>;
        rp = mir.cast (rs) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:RawConst>) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>)>;
        mir.goto (rp) [^join] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>) -> ()>
      ^arm_null():
        null_p = mir.cast (zero64) [] [cast_kind: mir.cast_kind PointerWithExposedProvenance, pointer_kind_authority: mir.pointer_kind_authority RustCast]: <(builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>)>;
        mir.goto (null_p) [^join] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>) -> ()>
      ^join(merged: mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>):
        far = mir.constant () [] [value: builtin.integer <1000000: ui64>]: <() -> (builtin.integer ui64)>;
        bad_p = mir.ptr_offset (merged, far) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>)>;
        _bad_v = mir.load (bad_p) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:RawConst>) -> (builtin.integer ui32)>;
        mir.goto () [^done] []: <() -> ()>
      ^done():
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let mixed_err = export_named_source(mixed_phi, "mixed_phi_k")
        .expect_err("merging verified slice pointer with null/unverified pointer must fail closed");
    assert!(mixed_err.to_string().contains("unmodeled raw-pointer load"));

    // 6. Concurrent temp-file isolation: 8 simultaneous uncached `check_generated_lean_in_cache`
    //    invocations in the same process never collide on `/tmp/cuda_oxide_verified_*.lean`.
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let lean_src = &safe_lean;
            let dir = &lean_dir;
            handles.push(s.spawn(move || {
                lean_exporter::runner::check_generated_lean_in_cache(lean_src, dir, None)
                    .expect("concurrent uncached Lean check must not collide on temp file");
            }));
        }
        for h in handles {
            h.join().expect("worker did not panic");
        }
    });
}

/// Allow raw pointer reads on a slice parameter that appears only in `requires` and
/// is not named in any `#[verify(...)]` safety clause.
#[test]
fn accepts_raw_pointer_read_of_a_slice_parameter_no_verify_clause_names() {
    // `a` appears only in `requires`, so no clause makes a bounds claim about it and a raw read of it
    // stays accepted: the rejection is for the parameters a clause names.
    let unnamed = with_raw_far_read(VECADD_FIXTURE).replacen(
        "verify_no_oob_idx: builtin.string \"0,1,2\"",
        "verify_no_oob_idx: builtin.string \"1,2\"",
        1,
    );
    export_named_source(&unnamed, "vecadd")
        .expect("raw read of a slice parameter that no verify clause names stays accepted");
}

/// Collect all `(id, width)` pairs from [`lean_exporter::translate::program::MirRhs::OpaqueUint`]
/// assignments in RPO block and statement emission order.
fn collect_program_opaque_uints(
    program: &lean_exporter::translate::program::MirProgramData,
) -> Vec<(usize, u32)> {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    let mut out = Vec::new();
    for blk in &program.blocks {
        for stmt in &blk.stmts {
            if let MirStmt::Assign(
                _,
                MirRhs::OpaqueUint(id, width) | MirRhs::CtaUint1(id, width, _),
            ) = stmt
            {
                out.push((*id, *width));
            }
        }
    }
    out
}

/// Ensure `nvvm.clc_query_*`, `mir.extract_array_element`, and `mir.load` allocate
/// globally distinct, strictly monotonic `OpaqueUint` IDs from `opaque_ids` across both a
/// minimal multi-block synthetic kernel and all shipped `MirFunc` exporter fixtures.
#[test]
fn clc_query_and_array_element_use_distinct_opaque_ids() {
    let collision_mir = r#"
builtin.module @opaque_id_collision {
  ^entry():
    mir.func @opaque_id_collision_k: builtin.function <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_idx: builtin.string "0"]
    {
      ^entry(a: mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>):
        z64 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        a_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>)>;
        a_len = mir.extract_field (a) [] [index: mir.field_index 1]: <(mir.slice <builtin.integer ui32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
        ok = mir.lt (z64, a_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
        mir.cond_br (ok) [^blk0, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^blk0():
        p0 = mir.ptr_offset (a_ptr, z64) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>)>;
        ld0 = mir.load (p0) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.integer ui32)>;
        p1 = mir.ptr_offset (a_ptr, z64) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>)>;
        ld1 = mir.load (p1) [] []: <(mir.ptr <builtin.integer ui32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.integer ui32)>;
        mir.goto () [^blk1] []: <() -> ()>

      ^blk1():
        c_arr = mir.construct_array (ld0, ld1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (mir.array <builtin.integer ui32, 2>)>;
        _elem_v = mir.extract_array_element (c_arr, z64) [] []: <(mir.array <builtin.integer ui32, 2>, builtin.integer ui64) -> (builtin.integer ui32)>;
        _q_x = nvvm.clc_query_get_first_ctaid_x (z64, z64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui32)>;
        _q_c = nvvm.clc_query_is_canceled (z64, z64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui32)>;
        mir.return () [] []: <() -> ()>

      ^exit():
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let prog = translate_named_mir_program(collision_mir, "opaque_id_collision_k");
    let opaques = collect_program_opaque_uints(&prog);
    assert_eq!(
        opaques.len(),
        5,
        "expected 5 OpaqueUint values: {opaques:?}"
    );
    let unique: std::collections::HashSet<usize> = opaques.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        unique.len(),
        5,
        "load, extract_array_element, and clc_query_* must have distinct OpaqueUint IDs: {opaques:?}"
    );
    assert!(
        opaques.windows(2).all(|w| w[0].0 < w[1].0),
        "OpaqueUint IDs must be strictly increasing in RPO emission order: {opaques:?}"
    );
    let lean = export_named_source(collision_mir, "opaque_id_collision_k")
        .expect("transcribes opaque_id_collision_k");
    if let Some(lean_dir) = lean_package_dir_if_available() {
        lean_exporter::runner::check_generated_lean(&lean, &lean_dir)
            .expect("Lean 4 accepts kernel with distinct OpaqueUint IDs");
    }

    for (name, src, fn_name) in [
        ("vecadd", VECADD_FIXTURE, "vecadd"),
        (
            "sgemm_tiled_views",
            SGEMM_TILED_FIXTURE,
            "sgemm_tiled_views",
        ),
        ("sgemm_tiled_raw", SGEMM_TILED_FIXTURE, "sgemm_tiled_raw"),
        ("warp_reduce_sum", WARP_REDUCE_FIXTURE, "warp_reduce_sum"),
        ("block_reduce_sum", WARP_REDUCE_FIXTURE, "block_reduce_sum"),
        (
            "subwarp_bit_split_stage",
            SUBWARP_MULTIBUF_FIXTURE,
            "subwarp_bit_split_stage",
        ),
        (
            "multibuf_subwarp_stage",
            SUBWARP_MULTIBUF_FIXTURE,
            "multibuf_subwarp_stage",
        ),
        ("async_copy_stage", ASYNC_COPY_FIXTURE, "async_copy_stage"),
        (
            "tcgen05_matmul_verified",
            TCGEN05_MATMUL_FIXTURE,
            "tcgen05_matmul_verified",
        ),
        ("warp_ring_stage", WARP_RING_FIXTURE, "warp_ring_stage"),
        (
            "cluster_clc_stage",
            CLUSTER_CLC_FIXTURE,
            "cluster_clc_stage",
        ),
    ] {
        let fx_prog = translate_named_mir_program(src, fn_name);
        let fx_opaques = collect_program_opaque_uints(&fx_prog);
        let fx_unique: std::collections::HashSet<usize> =
            fx_opaques.iter().map(|&(id, _)| id).collect();
        assert_eq!(
            fx_unique.len(),
            fx_opaques.len(),
            "{name}: OpaqueUint IDs must all be globally distinct: {fx_opaques:?}"
        );
        assert!(
            fx_opaques.windows(2).all(|w| w[0].0 < w[1].0),
            "{name}: OpaqueUint IDs must be strictly increasing in RPO emission order: {fx_opaques:?}"
        );
        if name == "cluster_clc_stage" {
            assert!(
                fx_opaques.len() >= 4,
                "cluster_clc_stage must emit loads + clc_query_* OpaqueUint IDs: {fx_opaques:?}"
            );
        }
    }
}

/// Allow pure compiler float/integer math intrinsic placeholders (`.sin()`, `.cos()`, `ctpop`)
/// while rejecting un-inlined user float/zero-arg functions and unchecked UB intrinsics.
#[test]
fn accepts_pure_math_intrinsic_placeholders_and_rejects_unmodeled_float_helpers() {
    // 1. Positive: `a[idx].sin() + b[idx].cos()` via compiler math placeholders exports as `.unk`
    //    and proves all safety obligations in Lean 4.
    let sin_cos_mir = VECADD_FIXTURE.replace(
        "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
        "a_sin = mir.call (a_val) [] [callee: builtin.string \"__cuda_oxide_rust_intrinsic_sinf32\"]: <(builtin.fp32) -> (builtin.fp32)>;\n    b_cos = mir.call (b_val) [] [callee: builtin.string \"__cuda_oxide_rust_intrinsic_cosf32\"]: <(builtin.fp32) -> (builtin.fp32)>;\n    sum = mir.add (a_sin, b_cos) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
    );
    let sin_cos_lean = export_named_source(&sin_cos_mir, "vecadd")
        .expect("pure float math intrinsics must export");
    if let Some(lean_dir) = lean_package_dir_if_available() {
        lean_exporter::runner::check_generated_lean(&sin_cos_lean, &lean_dir)
            .expect("Lean 4 must accept verified vecadd using sinf32/cosf32");
    }

    // 2. Negative: un-inlined user helper taking only `f32` could hide `sync_threads()` or shared
    //    writes, so it must fail closed even though all arguments are floating-point.
    let sneaky_float_helper = VECADD_FIXTURE.replace(
        "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
        "a_tmp = mir.call (a_val) [] [callee: builtin.string \"user_crate::sneaky_sync_helper\"]: <(builtin.fp32) -> (builtin.fp32)>;\n    sum = mir.add (a_tmp, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
    );
    let err = export_named_source(&sneaky_float_helper, "vecadd")
        .expect_err("un-inlined user float helper must be rejected");
    assert!(
        err.to_string().contains("unmodeled function call"),
        "unexpected error for user float helper: {err}"
    );

    // 3. Negative: 0-argument un-inlined user helper returning `f32` must also fail closed.
    let zero_arg_helper = VECADD_FIXTURE.replace(
        "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
        "z_tmp = mir.call () [] [callee: builtin.string \"user_crate::zero_arg_helper\"]: <() -> (builtin.fp32)>;\n    sum = mir.add (z_tmp, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
    );
    let err_zero = export_named_source(&zero_arg_helper, "vecadd")
        .expect_err("0-argument user helper must be rejected");
    assert!(
        err_zero.to_string().contains("unmodeled function call"),
        "unexpected error for 0-argument user helper: {err_zero}"
    );

    // 4. Negative: unchecked intrinsic with UB preconditions (`float_to_int_unchecked`) must fail closed.
    let unchecked_intrinsic = VECADD_FIXTURE.replace(
        "sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
        "bad_int = mir.call (a_val) [] [callee: builtin.string \"__cuda_oxide_rust_intrinsic_float_to_int_unchecked\"]: <(builtin.fp32) -> (builtin.integer ui32)>;\n    sum = mir.add (a_val, b_val) [] []: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;",
    );
    let err_unchecked = export_named_source(&unchecked_intrinsic, "vecadd")
        .expect_err("unchecked intrinsic with UB preconditions must be rejected");
    assert!(
        err_unchecked
            .to_string()
            .contains("unmodeled function call"),
        "unexpected error for float_to_int_unchecked: {err_unchecked}"
    );

    // 5. Negative (Lean 4 proof check): total integer intrinsic (`ctpop`) exports as unconstrained
    //    `OpaqueUint`, so unguarded `a[idx.count_ones() as usize]` fails `MirNoOobGlobal` in Lean 4!
    let unguarded_ctpop_idx = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "pop = mir.call (idx) [] [callee: builtin.string \"__cuda_oxide_rust_intrinsic_ctpop\"]: <(builtin.integer ui64) -> (builtin.integer ui32)>;\n    pop64 = mir.cast (pop) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    a_elem = mir.ptr_offset (a_ptr, pop64)",
    );
    let unguarded_ctpop_lean = export_named_source(&unguarded_ctpop_idx, "vecadd")
        .expect("ctpop exports 1-to-1 as OpaqueUint");
    if let Some(lean_dir) = lean_package_dir_if_available() {
        let err_oob = lean_exporter::runner::check_generated_lean(&unguarded_ctpop_lean, &lean_dir)
            .expect_err("Lean 4 must reject unguarded `a[ctpop(idx)]` when `a.len()` can be 0/1");
        assert!(
            err_oob
                .to_string()
                .contains("omega could not prove the goal"),
            "unexpected Lean error for unguarded ctpop index: {err_oob}"
        );
    }
}

/// Ensure missing `.olean` artifacts trigger `lake build` and broken Lean package configs fail closed.
#[test]
fn ensure_lean_library_built_succeeds_on_valid_package_and_fails_closed_on_invalid_lakefile() {
    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    lean_exporter::runner::ensure_lean_library_built(&lean_dir)
        .expect("real oxide-proofs-preview package must build or already have CudaOxide.olean");

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let bad_pkg = std::env::temp_dir().join(format!(
        "cuda_oxide_bad_lake_pkg_{}_{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&bad_pkg).unwrap();
    std::fs::copy(
        lean_dir.join("lean-toolchain"),
        bad_pkg.join("lean-toolchain"),
    )
    .unwrap();
    std::fs::write(bad_pkg.join("lakefile.toml"), "not = [valid toml\n").unwrap();
    let err = lean_exporter::runner::ensure_lean_library_built(&bad_pkg)
        .expect_err("broken lakefile.toml without CudaOxide.olean must fail closed");
    let _ = std::fs::remove_dir_all(&bad_pkg);
    assert!(
        err.to_string().contains("`lake build` failed"),
        "unexpected error from broken lakefile.toml: {err}"
    );
}

/// Verify that Lean 4's `Split -> Assign -> Meet` (`MirValidUniSet`) check rejects barrier
/// loops bounded by divergent `if/else` joins, `mbarrier` spin counts, and `continue` loops
/// jumping back up to a loop header, while accepting CTA-uniform controls and divergent
/// non-barrier loops followed by a uniform barrier.
#[test]
fn rejects_barrier_loops_bounded_by_divergent_joins_and_spin_counts() {
    let join_lean = export_named_source(DIVERGENT_UNIFORMITY_FIXTURE, "sync_dep_barrier")
        .expect("exports the sync-dependent barrier loop");

    let uniform_join = DIVERGENT_UNIFORMITY_FIXTURE.replacen(
        "tx = nvvm.read_ptx_sreg_tid_x () [] []",
        "tx = nvvm.read_ptx_sreg_ctaid_x () [] []",
        1,
    );
    let uniform_join_lean = export_named_source(&uniform_join, "sync_dep_barrier")
        .expect("exports the CTA-uniform barrier loop");

    let spin_lean = export_named_source(DIVERGENT_UNIFORMITY_FIXTURE, "spin_count_barrier")
        .expect("exports the spin-count barrier loop");

    let uniform_spin = DIVERGENT_UNIFORMITY_FIXTURE
        .replacen("more = mir.lt (k, spins) [] []", "more = mir.lt (k, k_bound) [] []", 1)
        .replacen(
            "      ^bar_head(k: builtin.integer ui32):\n",
            "      ^bar_head(k: builtin.integer ui32):\n        k_bound = mir.constant () [] [value: builtin.integer <2: ui32>]: <() -> (builtin.integer ui32)>;\n",
            1,
        );
    let uniform_spin_lean = export_named_source(&uniform_spin, "spin_count_barrier")
        .expect("exports the constant-trip barrier loop");

    // Two `continue`/loop-end edges jumping UP into `^outer_head(k, n)` with `n = 1` vs `n = 3`.
    let continue_loop = r#"
builtin.module @continue_split {
  ^entry():
    mir.func @continue_split_barrier: builtin.function <(mir.disjoint_slice <builtin.integer ui32,[]>) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <64: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(out: mir.disjoint_slice <builtin.integer ui32,[]>):
        tx = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
        k0 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
        n0 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
        mir.goto (k0, n0) [^outer_head] []: <(builtin.integer ui32, builtin.integer ui32) -> ()>

      ^outer_head(k: builtin.integer ui32, n: builtin.integer ui32):
        two = mir.constant () [] [value: builtin.integer <2: ui32>]: <() -> (builtin.integer ui32)>;
        outer_more = mir.lt (k, two) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (outer_more) [^outer_body, ^after_outer] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^outer_body():
        one = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
        k_next = mir.add (k, one) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        c16 = mir.constant () [] [value: builtin.integer <16: ui32>]: <() -> (builtin.integer ui32)>;
        lo = mir.lt (tx, c16) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (lo) [^cont_arm, ^else_arm] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^cont_arm():
        c1 = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
        mir.goto (k_next, c1) [^outer_head] []: <(builtin.integer ui32, builtin.integer ui32) -> ()>

      ^else_arm():
        c3 = mir.constant () [] [value: builtin.integer <3: ui32>]: <() -> (builtin.integer ui32)>;
        mir.goto (k_next, c3) [^outer_head] []: <(builtin.integer ui32, builtin.integer ui32) -> ()>

      ^after_outer():
        i0 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
        mir.goto (i0) [^bar_head] []: <(builtin.integer ui32) -> ()>

      ^bar_head(i: builtin.integer ui32):
        more = mir.lt (i, n) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
        mir.cond_br (more) [^bar_body, ^tail] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

      ^bar_body():
        nvvm.barrier0 () [] []: <() -> ()>;
        step = mir.constant () [] [value: builtin.integer <1: ui32>]: <() -> (builtin.integer ui32)>;
        i_next = mir.add (i, step) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        mir.goto (i_next) [^bar_head] []: <(builtin.integer ui32) -> ()>

      ^tail():
        nvvm.barrier0 () [] []: <() -> ()>;
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let continue_lean = export_named_source(continue_loop, "continue_split_barrier")
        .expect("exports two-backedge continue split loop");

    // Safe control: an unrelated divergent `n = if tx < 16 { 1 } else { 3 }` alongside a
    // constant-bounded `while i < 2 { sync_threads() }` loop keeps `i < 2` uniform and verifies.
    let safe_unrelated_div = DIVERGENT_UNIFORMITY_FIXTURE
        .replacen(
            "more = mir.lt (i, n) [] []",
            "more = mir.lt (i, bound2) [] []",
            1,
        )
        .replacen(
            "      ^head(i: builtin.integer ui32):\n",
            "      ^head(i: builtin.integer ui32):\n        bound2 = mir.constant () [] [value: builtin.integer <2: ui32>]: <() -> (builtin.integer ui32)>;\n",
            1,
        );
    let safe_unrelated_lean = export_named_source(&safe_unrelated_div, "sync_dep_barrier")
        .expect("exports constant-trip barrier loop after unrelated divergent join");

    let Some(lean_dir) = lean_package_dir_if_available() else {
        return;
    };
    lean_exporter::runner::check_generated_lean(&join_lean, &lean_dir)
        .expect_err("Lean 4 must reject a barrier loop bounded by a divergent join");
    lean_exporter::runner::check_generated_lean(&spin_lean, &lean_dir)
        .expect_err("Lean 4 must reject a barrier loop bounded by a spin count");
    lean_exporter::runner::check_generated_lean(&continue_lean, &lean_dir)
        .expect_err("Lean 4 must reject a barrier loop bounded by a `continue` split join");
    lean_exporter::runner::check_generated_lean(&safe_unrelated_lean, &lean_dir)
        .expect("Lean 4 accepts constant-trip barrier loop after unrelated divergent join");
    lean_exporter::runner::check_generated_lean(&uniform_join_lean, &lean_dir)
        .expect("Lean 4 accepts the CTA-uniform barrier loop");
    lean_exporter::runner::check_generated_lean(&uniform_spin_lean, &lean_dir)
        .expect("Lean 4 accepts the constant-trip barrier loop");
}

/// Collect all source register IDs read by `stmt` via an exhaustive match over
/// [`lean_exporter::translate::program::MirStmt`] and [`lean_exporter::translate::program::MirRhs`].
fn mir_stmt_read_regs(stmt: &lean_exporter::translate::program::MirStmt) -> Vec<usize> {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    match stmt {
        MirStmt::Assign(_, rhs) => match rhs {
            MirRhs::Copy(src)
            | MirRhs::NotBool(src)
            | MirRhs::IntCast(_, src)
            | MirRhs::Load(src)
            | MirRhs::CvtaShared(src)
            | MirRhs::LoadTmemBase(src, _)
            | MirRhs::CtaUint1(_, _, src)
            | MirRhs::CtaUnary(src) => vec![*src],
            MirRhs::BinOp(_, _, l, r)
            | MirRhs::CheckedOvf(_, _, l, r)
            | MirRhs::CmpOp(_, l, r)
            | MirRhs::PtrOffset(l, r)
            | MirRhs::CtaBin(l, r) => vec![*l, *r],
            MirRhs::MbarTryWait(_, slot, _) => vec![*slot],
            MirRhs::ConstInt(_, _)
            | MirRhs::ConstBool(_)
            | MirRhs::Sreg(_)
            | MirRhs::Index1D
            | MirRhs::Coord2DU32
            | MirRhs::SlicePtr(_)
            | MirRhs::SliceLen(_)
            | MirRhs::SharedBase(_)
            | MirRhs::OpaqueUint(_, _)
            | MirRhs::CtaConst
            | MirRhs::OpaqueVal => Vec::new(),
        },
        MirStmt::Store(p, v) | MirStmt::TmaCopyG2S(p, _, v, _) => vec![*p, *v],
        MirStmt::AssertCond(_, r)
        | MirStmt::BarrierSlotCheck(r)
        | MirStmt::ClusterRankCheck(r)
        | MirStmt::MbarArrive(_, r, _)
        | MirStmt::MbarArriveExpectTx(_, r, _)
        | MirStmt::TmemAlloc(r, _)
        | MirStmt::SmemDescRead(r, _)
        | MirStmt::TmemAccess(r, _, _, _) => vec![*r],
        MirStmt::SyncThreads => Vec::new(),
    }
}

/// Collect all source register IDs read by `term` via an exhaustive match over
/// [`lean_exporter::translate::program::MirTerminator`].
fn mir_term_read_regs(term: &lean_exporter::translate::program::MirTerminator) -> Vec<usize> {
    use lean_exporter::translate::program::MirTerminator;

    match term {
        MirTerminator::Ret | MirTerminator::Unreachable => Vec::new(),
        MirTerminator::Goto(_, args) => args.clone(),
        MirTerminator::CondBr(cond, _, t_args, _, f_args) => {
            let mut out = Vec::with_capacity(1 + t_args.len() + f_args.len());
            out.push(*cond);
            out.extend_from_slice(t_args);
            out.extend_from_slice(f_args);
            out
        }
    }
}

/// Compute exact CFG dominator sets `dom[b]` over reachable RPO basic blocks in `program`
/// (with block `0` as entry) so cross-block register reads can be checked for strict dominance.
fn compute_cfg_dominators(
    program: &lean_exporter::translate::program::MirProgramData,
) -> Vec<std::collections::HashSet<usize>> {
    use lean_exporter::translate::program::MirTerminator;
    use std::collections::HashSet;

    let n = program.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b_idx, blk) in program.blocks.iter().enumerate() {
        match blk.term {
            MirTerminator::Goto(dst, _) => {
                if dst < n && !preds[dst].contains(&b_idx) {
                    preds[dst].push(b_idx);
                }
            }
            MirTerminator::CondBr(_, t_dst, _, f_dst, _) => {
                for dst in [t_dst, f_dst] {
                    if dst < n && !preds[dst].contains(&b_idx) {
                        preds[dst].push(b_idx);
                    }
                }
            }
            MirTerminator::Ret | MirTerminator::Unreachable => {}
        }
    }

    let all: HashSet<usize> = (0..n).collect();
    let mut dom: Vec<HashSet<usize>> = vec![all; n];
    dom[0] = HashSet::from([0]);
    loop {
        let mut changed = false;
        for b in 1..n {
            if preds[b].is_empty() {
                let singleton = HashSet::from([b]);
                if dom[b] != singleton {
                    dom[b] = singleton;
                    changed = true;
                }
                continue;
            }
            let mut inter: Option<HashSet<usize>> = None;
            for &p in &preds[b] {
                inter = Some(match inter {
                    None => dom[p].clone(),
                    Some(cur) => cur.intersection(&dom[p]).copied().collect(),
                });
            }
            let mut new_set = inter.unwrap_or_default();
            new_set.insert(b);
            if new_set != dom[b] {
                dom[b] = new_set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dom
}

/// Assert on the typed [`lean_exporter::translate::program::MirProgramData`] AST that every
/// register read before local definition in any block is uniquely defined across blocks (zero
/// Tier-2 slot reuse clobbers) in a block that strictly dominates the reader.
fn assert_valid_cross_block_registers(
    program: &lean_exporter::translate::program::MirProgramData,
    label: &str,
) {
    use lean_exporter::translate::program::MirStmt;
    use std::collections::{HashMap, HashSet};

    assert!(
        !program.blocks.is_empty(),
        "{label}: expected non-empty MirFunc blocks"
    );
    let mut def_blocks: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (b_idx, blk) in program.blocks.iter().enumerate() {
        for &(aid, _) in &blk.args {
            def_blocks.entry(aid).or_default().insert(b_idx);
        }
        for stmt in &blk.stmts {
            if let MirStmt::Assign(dst, _) = stmt {
                def_blocks.entry(*dst).or_default().insert(b_idx);
            }
        }
    }

    let dom = compute_cfg_dominators(program);
    let check_read = |b_idx: usize, reg: usize, defined_here: &HashSet<usize>, ctx_desc: &str| {
        let defs = def_blocks.get(&reg).unwrap_or_else(|| {
            panic!("{label}: block {b_idx} reads undefined register {reg} in {ctx_desc}")
        });
        if !defined_here.contains(&reg) {
            assert_eq!(
                defs.len(),
                1,
                "{label}: block {b_idx} reads cross-block register {reg} before local definition in {ctx_desc}, while blocks {defs:?} also write {reg} (Tier-2 clobber)"
            );
            let def_b = *defs.iter().next().expect("unique writer");
            assert!(
                def_b != b_idx && dom[b_idx].contains(&def_b),
                "{label}: block {b_idx} reads register {reg} defined in non-dominating block {def_b} in {ctx_desc}"
            );
        }
    };

    for (b_idx, blk) in program.blocks.iter().enumerate() {
        let mut defined_here: HashSet<usize> = blk.args.iter().map(|&(aid, _)| aid).collect();
        for stmt in &blk.stmts {
            for reg in mir_stmt_read_regs(stmt) {
                check_read(b_idx, reg, &defined_here, &format!("{stmt:?}"));
            }
            if let MirStmt::Assign(dst, _) = stmt {
                defined_here.insert(*dst);
            }
        }
        for reg in mir_term_read_regs(&blk.term) {
            check_read(b_idx, reg, &defined_here, &format!("{:?}", blk.term));
        }
    }
}

/// Extract the register ID assigned by the unique `mir.add` (`MirRhs::BinOp("add", ...)`)
/// statement inside `blk` for structural regression assertions.
fn find_add_assign_reg(blk: &lean_exporter::translate::program::MirBlockData) -> usize {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    blk.stmts
        .iter()
        .find_map(|stmt| match stmt {
            MirStmt::Assign(dst, MirRhs::BinOp("add", _, _, _)) => Some(*dst),
            _ => None,
        })
        .expect("block must contain an add assignment")
}

/// A register captured by an aggregate must still identify the original field when
/// the successor extracts it. This fixture only constructs and reads value aggregates.
#[test]
fn aggregate_and_inserted_fields_preserve_register_identity() {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    let tuple = "mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>";
    for (op, ty, operands, types, field) in [
        (
            "mir.construct_struct",
            "mir.struct <\"Wrap\", [\"field\"], [builtin.integer ui64], [], [], 8, 8, Aggregate>",
            "sum",
            "builtin.integer ui64",
            0,
        ),
        (
            "mir.construct_tuple",
            tuple,
            "sum, seed",
            "builtin.integer ui64, builtin.integer ui64",
            0,
        ),
        (
            "mir.construct_array",
            "mir.array <builtin.integer ui64, 2>",
            "sum, seed",
            "builtin.integer ui64, builtin.integer ui64",
            0,
        ),
        ("mir.insert_field", tuple, "base, sum", "", 1),
    ] {
        let construct = if op == "mir.insert_field" {
            format!(
                "base = mir.construct_tuple (seed, seed) [] []: <(builtin.integer ui64, builtin.integer ui64) -> ({ty})>;\n        \
                 value = mir.insert_field (base, sum) [] [insert_index: mir.field_index 1]: <({ty}, builtin.integer ui64) -> ({ty})>;"
            )
        } else {
            format!("value = {op} ({operands}) [] []: <({types}) -> ({ty})>;")
        };
        let source = format!(
            r#"
mir.func @field_lifetime: builtin.function <(builtin.integer ui32) -> ()>
  [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_uniform_barriers: builtin.string "true"]
{{
  ^entry(_n: builtin.integer ui32):
    seed = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;
    sum = mir.add (seed, seed) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    {construct}
    mir.goto () [^unpack] []: <() -> ()>
  ^unpack():
    local_a = mir.constant () [] [value: builtin.integer <2: ui64>]: <() -> (builtin.integer ui64)>;
    local_b = mir.constant () [] [value: builtin.integer <3: ui64>]: <() -> (builtin.integer ui64)>;
    field_value = mir.extract_field (value) [] [index: mir.field_index {field}]: <({ty}) -> (builtin.integer ui64)>;
    mir.return () [] []: <() -> ()>
}}"#
        );
        let program = translate_named_mir_program(&source, "field_lifetime");
        assert_valid_cross_block_registers(&program, op);
        let captured = find_add_assign_reg(&program.blocks[0]);
        let successor = &program.blocks[1].stmts;
        assert!(
            successor
                .iter()
                .all(|stmt| !matches!(stmt, MirStmt::Assign(dst, _) if *dst == captured)),
            "{op}: successor redefined the captured register"
        );
        assert!(
            successor.iter().any(
                |stmt| matches!(stmt, MirStmt::Assign(_, MirRhs::Copy(src)) if *src == captured)
            ),
            "{op}: extraction must read the captured register"
        );
    }
}

/// A valid shared barrier slot computed in the entry block must retain its register
/// when the successor emits the matching arrival.
#[test]
fn bar_slot_vars_do_not_clobber_tier2_slot_register_across_blocks() {
    use lean_exporter::translate::program::MirStmt;

    let valid_mir = r#"
builtin.module @bar_slot_tier2 {
  ^entry():
    mir.func @bar_slot_tier2_k: builtin.function <(builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_oob_shared: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(_n: builtin.integer ui32):
        lhs_c = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;
        rhs_c = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;
        slot = mir.add (lhs_c, rhs_c) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
        bar_base = mir.shared_alloc () [] [alignment: builtin.integer <8: i64>, elem_type: builtin.type builtin.integer ui64, size: builtin.integer <4: i64>, alloc_key: builtin.string "BARS", source_name: builtin.string "k::BARS"]: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;
        bar_p = mir.ptr_offset (bar_base, slot) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;
        bar_cast = mir.cast (bar_p) [] [cast_kind: mir.cast_kind PtrToPtr]: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;
        cnt32 = mir.constant () [] [value: builtin.integer <32: ui32>]: <() -> (builtin.integer ui32)>;
        nvvm.mbarrier_init_shared (bar_cast, cnt32) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui32) -> ()>;
        nvvm.fence_proxy_async_shared_cta () [] []: <() -> ()>;
        mir.goto () [^use_bar] []: <() -> ()>

      ^use_bar():
        z0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        z1 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        z2 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        z3 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        nvvm.barrier0 () [] []: <() -> ()>;
        _tok = nvvm.mbarrier_arrive_shared (bar_cast) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.integer ui64)>;
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let valid_prog = translate_named_mir_program(valid_mir, "bar_slot_tier2_k");
    assert_valid_cross_block_registers(&valid_prog, "bar_slot_tier2_k_valid");
    let slot_reg = find_add_assign_reg(&valid_prog.blocks[0]);
    assert!(
        !valid_prog.blocks[1]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::Assign(dst, _) if *dst == slot_reg)),
        "^use_bar must not clobber barrier slot register {slot_reg}"
    );
    assert!(
        valid_prog.blocks[1]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::MbarArrive(0, sreg, _) if *sreg == slot_reg)),
        "^use_bar MbarArrive must reference block 0's slot register {slot_reg}"
    );

    if let Some(lean_dir) = lean_package_dir_if_available() {
        let valid_lean =
            export_named_source(valid_mir, "bar_slot_tier2_k").expect("transcribes valid bar slot");
        lean_exporter::runner::check_generated_lean(&valid_lean, &lean_dir)
            .expect("Lean 4 must accept valid in-bounds SharedArray<Barrier, 4> slot `1 + 1`");
    }
}

/// Ensure wrapper `LocalTag` (`Index1DWrapper`) and aggregate `struct_fields` (`coord_2d_u32`)
/// captured indirectly through `mir.construct_enum` (`enum_payload_tags`) or `mir.insert_field`
/// `base_s` (`o_idx == 0`) allocate Tier-1 IDs and survive across blocks.
#[test]
fn wrapper_tags_and_fields_carried_through_enum_and_insert_field_survive_across_blocks() {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    // Subcase 1: `mir.construct_enum` operand (`idx_w`) carries `LocalTag::Index1DWrapper(idx_vid)`
    // into `enum_payload_tags` across blocks even when `idx_w` is only used inside `^blk0`.
    let enum_mir = r#"
builtin.module @enum_wrapper_tier2 {
  ^entry():
    mir.func @_RINvNtNtCs11cuda_device6thread10___internal8index_1d: builtin.function <(mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>) -> (builtin.integer ui64)>
      []
    {
      ^entry(_lc: mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>):
        _z = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        mir.return (_z) [] []: <(builtin.integer ui64) -> ()>
    };
    mir.func @enum_idx_wrapper_k: builtin.function <(builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(_n: builtin.integer ui32):
        z0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        mir.goto () [^blk0] []: <() -> ()>

      ^blk0():
        lc = mir.alloca () [] []: <() -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>)>;
        lc_ref = mir.cast (lc) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>) -> (mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>)>;
        idx_w = mir.call (lc_ref) [] [callee: builtin.string "_RINvNtNtCs11cuda_device6thread10___internal8index_1d"]: <(mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>) -> (builtin.integer ui64)>;
        opt = mir.construct_enum (idx_w) [] [construct_enum_variant_index: mir.variant_index 1]: <(builtin.integer ui64) -> (mir.enum <"Option", builtin.integer ui8, ["None", "Some"], [0, 1], [0, 1], [builtin.integer ui64], [8], [8], 0, 16, 8, Direct, Integer, 8, 0, 0, 0, 0, 0, 0, [1, 1]>)>;
        mir.goto () [^blk1] []: <() -> ()>

      ^blk1():
        clobber0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber1 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber2 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber3 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        pay = mir.enum_payload (opt) [] [payload_variant_index: mir.variant_index 1, payload_field_index: mir.field_index 0]: <(mir.enum <"Option", builtin.integer ui8, ["None", "Some"], [0, 1], [0, 1], [builtin.integer ui64], [8], [8], 0, 16, 8, Direct, Integer, 8, 0, 0, 0, 0, 0, 0, [1, 1]>) -> (builtin.integer ui64)>;
        ext_idx = mir.extract_field (pay) [] [index: mir.field_index 0]: <(builtin.integer ui64) -> (builtin.integer ui64)>;
        mir.return () [] []: <() -> ()>

    }
}
"#;
    let enum_prog = translate_named_mir_program(enum_mir, "enum_idx_wrapper_k");
    assert_valid_cross_block_registers(&enum_prog, "enum_idx_wrapper_k");
    let idx_reg = enum_prog.blocks[1]
        .stmts
        .iter()
        .find_map(|s| match s {
            MirStmt::Assign(dst, MirRhs::Index1D) => Some(*dst),
            _ => None,
        })
        .expect("^blk0 must assign MirRhs::Index1D");
    assert!(
        !enum_prog.blocks[2]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::Assign(dst, _) if *dst == idx_reg)),
        "^blk1 must not clobber enum-wrapped Index1D register {idx_reg}"
    );
    assert!(
        enum_prog.blocks[2]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::Assign(_, MirRhs::Copy(src)) if *src == idx_reg)),
        "^blk1 extract_field on enum payload must copy Index1D register {idx_reg}"
    );

    // Subcase 2: `mir.insert_field` `base_s` (`o_idx == 0`) retains unreplaced slots in
    // `struct_fields` even when `base_s` is a tuple whose own SSA value is used only in `^blk0`.
    let insert_base_mir = r#"
builtin.module @insert_base_coord_tier2 {
  ^entry():
    mir.func @_RINvNtCs11cuda_device6thread11coord_2d_u32: builtin.function <(mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>) -> (mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>)>
      []
    {
      ^entry(_lc: mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>):
        _z = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        _tup = mir.construct_tuple (_z, _z) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>)>;
        mir.return (_tup) [] []: <(mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>) -> ()>
    };
    mir.func @insert_base_coord_k: builtin.function <(builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(_n: builtin.integer ui32):
        c0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        mir.goto () [^blk0] []: <() -> ()>

      ^blk0():
        lc = mir.alloca () [] []: <() -> (mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>)>;
        lc_ref = mir.cast (lc) [] [cast_kind: mir.cast_kind PtrToPtr, pointer_kind_authority: mir.pointer_kind_authority Reborrow]: <(mir.ptr <builtin.integer ui8 ,mutable:true,addrspace:0,kind:Erased>) -> (mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>)>;
        coord_w = mir.call (lc_ref) [] [callee: builtin.string "_RINvNtCs11cuda_device6thread11coord_2d_u32"]: <(mir.ptr <builtin.integer ui8 ,mutable:false,addrspace:0,kind:SharedRef>) -> (mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>)>;
        t1 = mir.insert_field (coord_w, c0) [] [insert_index: mir.field_index 1]: <(mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>, builtin.integer ui64) -> (mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>)>;
        mir.goto () [^blk1] []: <() -> ()>

      ^blk1():
        clobber0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber1 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber2 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        clobber3 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        ext_c = mir.extract_field (t1) [] [index: mir.field_index 0]: <(mir.tuple <[builtin.integer ui64, builtin.integer ui64], [], [], 16, 8>) -> (builtin.integer ui64)>;
        mir.return () [] []: <() -> ()>

    }
}
"#;
    let base_prog = translate_named_mir_program(insert_base_mir, "insert_base_coord_k");
    assert_valid_cross_block_registers(&base_prog, "insert_base_coord_k");
    let coord_reg = base_prog.blocks[1]
        .stmts
        .iter()
        .find_map(|s| match s {
            MirStmt::Assign(dst, MirRhs::Coord2DU32) => Some(*dst),
            _ => None,
        })
        .expect("^blk0 must assign MirRhs::Coord2DU32");
    assert!(
        !base_prog.blocks[2]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::Assign(dst, _) if *dst == coord_reg)),
        "^blk1 must not clobber insert_field base_s Coord2DU32 register {coord_reg}"
    );
    assert!(
        base_prog.blocks[2]
            .stmts
            .iter()
            .any(|s| matches!(s, MirStmt::Assign(_, MirRhs::Copy(src)) if *src == coord_reg)),
        "^blk1 extract_field on t1 field 0 must copy Coord2DU32 register {coord_reg}"
    );
}

/// Ensure block-local Tier-2 register IDs inserted into `smem_desc_ready` in an earlier block are
/// evicted at block boundaries so recycled Tier-2 slots in a later block still emit the `>> 4`
/// shift on `Tcgen05SmemDescriptor`.
#[test]
fn smem_desc_ready_evicts_tier2_ids_at_block_boundary() {
    use lean_exporter::translate::program::{MirRhs, MirStmt};

    let mir_src = r#"
builtin.module @smem_desc_ready_evict {
  ^entry():
    mir.func @_RINvNtCs11cuda_device7tcgen0521Tcgen05SmemDescriptor3new: builtin.function <(builtin.integer ui64) -> (builtin.integer ui64)>
      []
    {
      ^entry(_raw: builtin.integer ui64):
        mir.return (_raw) [] []: <(builtin.integer ui64) -> ()>
    };
    mir.func @smem_desc_ready_evict_k: builtin.function <(builtin.integer ui32) -> ()>
      [gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <32: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_no_data_race: builtin.string "0", verify_no_oob_shared: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
    {
      ^entry(_n: builtin.integer ui32):
        z0 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
        c4 = mir.constant () [] [value: builtin.integer <4: ui64>]: <() -> (builtin.integer ui64)>;
        s = mir.shared_alloc () [] [alignment: builtin.integer <16: i64>, elem_type: builtin.type builtin.integer ui64, size: builtin.integer <16: i64>, alloc_key: builtin.string "SMEM", source_name: builtin.string "k::SMEM"]: <() -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;
        sp = mir.ptr_offset (s, z0) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>)>;
        _lv = mir.load (sp) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.integer ui64)>;
        mir.goto () [^blk0] []: <() -> ()>

      ^blk0():
        raw0 = nvvm.cvta_generic_to_shared_offset (s) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.integer ui64)>;
        _shr0 = mir.shr (raw0, c4) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
        mir.goto () [^blk1] []: <() -> ()>

      ^blk1():
        pad0 = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;
        raw1 = nvvm.cvta_generic_to_shared_offset (s) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.integer ui64)>;
        _desc = mir.call (raw1) [] [callee: builtin.string "_RINvNtCs11cuda_device7tcgen0521Tcgen05SmemDescriptor3new"]: <(builtin.integer ui64) -> (builtin.integer ui64)>;
        mir.return () [] []: <() -> ()>
    }
}
"#;
    let prog = translate_named_mir_program(mir_src, "smem_desc_ready_evict_k");
    assert_valid_cross_block_registers(&prog, "smem_desc_ready_evict_k");
    assert_eq!(prog.blocks.len(), 3);
    let blk0_shr_dst = prog.blocks[1]
        .stmts
        .iter()
        .find_map(|s| match s {
            MirStmt::Assign(dst, MirRhs::BinOp("shr", 64, _, _)) => Some(*dst),
            _ => None,
        })
        .expect("block 1 computes `shr0`");
    let blk1_raw_dst = prog.blocks[2]
        .stmts
        .iter()
        .find_map(|s| match s {
            MirStmt::Assign(dst, MirRhs::CvtaShared(_)) => Some(*dst),
            _ => None,
        })
        .expect("block 2 computes `raw1`");
    assert_eq!(
        blk0_shr_dst, blk1_raw_dst,
        "test requires `shr0` and `raw1` to share the same recycled Tier-2 slot"
    );
    let has_blk1_shr = prog.blocks[2]
        .stmts
        .iter()
        .any(|s| matches!(s, MirStmt::Assign(_, MirRhs::BinOp("shr", 64, src, _)) if *src == blk1_raw_dst));
    assert!(
        has_blk1_shr,
        "block 2 Tcgen05SmemDescriptor must emit `>> 4` after Tier-2 smem_desc_ready eviction: {:?}",
        prog.blocks[2].stmts
    );
}

/// Verify across all shipped `MirFunc` exporter fixtures that no block reads an undefined,
/// non-dominated, or clobbered cross-block register ID.
#[test]
fn all_mir_program_fixtures_preserve_cross_block_register_disjointness() {
    for (name, src, fn_name) in [
        ("vecadd", VECADD_FIXTURE, "vecadd"),
        (
            "sgemm_tiled_views",
            SGEMM_TILED_FIXTURE,
            "sgemm_tiled_views",
        ),
        ("sgemm_tiled_raw", SGEMM_TILED_FIXTURE, "sgemm_tiled_raw"),
        ("warp_reduce_sum", WARP_REDUCE_FIXTURE, "warp_reduce_sum"),
        ("block_reduce_sum", WARP_REDUCE_FIXTURE, "block_reduce_sum"),
        (
            "subwarp_bit_split_stage",
            SUBWARP_MULTIBUF_FIXTURE,
            "subwarp_bit_split_stage",
        ),
        (
            "multibuf_subwarp_stage",
            SUBWARP_MULTIBUF_FIXTURE,
            "multibuf_subwarp_stage",
        ),
        ("async_copy_stage", ASYNC_COPY_FIXTURE, "async_copy_stage"),
        (
            "tcgen05_matmul_verified",
            TCGEN05_MATMUL_FIXTURE,
            "tcgen05_matmul_verified",
        ),
        ("warp_ring_stage", WARP_RING_FIXTURE, "warp_ring_stage"),
        (
            "cluster_clc_stage",
            CLUSTER_CLC_FIXTURE,
            "cluster_clc_stage",
        ),
    ] {
        let prog = translate_named_mir_program(src, fn_name);
        assert_valid_cross_block_registers(&prog, name);
    }
    let scalar_lean = export(FIXTURE);
    assert!(scalar_lean.contains("CudaOxide.Mir.Scalar.Expr"));
}

#[test]
fn verifies_requires_arith_and_nvvm_scalar_arithmetic_and_rejects_insufficient_or_malformed() {
    // 1. Fail-closed `spec.rs` validation on invalid parameter kinds and syntax in `contract_requires`:
    for (bad_attr, expected_msg) in [
        (
            "s2>=1",
            "parameter 2 in `s2>=1` is not a slice or DisjointSlice",
        ),
        (
            "p0>=1",
            "parameter 0 in `p0>=1` is not an unsigned integer scalar",
        ),
        (
            "p3>=1",
            "parameter 3 in `p3>=1` is not an unsigned integer scalar",
        ),
        ("s0>=p9", "parameter index 9 in `s0>=p9` out of range"),
        ("s0>=4*", "unexpected end of expression"),
    ] {
        let bad_mir = format!(
            r#"
mir.func @bad_req: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, builtin.integer ui32, builtin.fp32) -> ()>
  [contract_requires: builtin.string "{bad_attr}", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "1", verify_no_oob_idx: builtin.string "0,1"]
{{
  ^entry(_x: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, _y: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, _n: builtin.integer ui32, _s: builtin.fp32):
    mir.return () [] []: <() -> ()>
}}
"#
        );
        let err = export_named_source(&bad_mir, "bad_req")
            .expect_err(&format!("expected `{bad_attr}` to be rejected"));
        assert!(
            err.to_string().contains(expected_msg),
            "expected `{expected_msg}` for `{bad_attr}`, got: {err}"
        );
    }

    // 2. Positive & negative end-to-end Lean 4 checks on scalar/scaled/2D `requires` + `nvvm.scalar_arithmetic` / `nvvm.scalar_math`:
    let scale_quad_mir = r#"
mir.func @scale_quad: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, builtin.integer ui32, builtin.fp32) -> ()>
  [contract_requires: builtin.string "s0>=4*s1,s1>=p2,p2!=0", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "1", verify_no_oob_idx: builtin.string "0,1"]
{
  ^entry(x: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, out: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>, n: builtin.integer ui32, scale: builtin.fp32):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    bd32 = nvvm.read_ptx_sreg_ntid_x () [] []: <() -> (builtin.integer ui32)>;
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    bx = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    bd = mir.cast (bd32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    tx = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    base = mir.mul (bx, bd) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    idx = mir.add (base, tx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    n64 = mir.cast (n) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    in_n = mir.lt (idx, n64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_n) [^active, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^active():
    c4 = mir.constant () [] [value: builtin.integer <4: ui64>]: <() -> (builtin.integer ui64)>;
    c3 = mir.constant () [] [value: builtin.integer <3: ui64>]: <() -> (builtin.integer ui64)>;
    idx4_pair = mir.checked_mul (c4, idx) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>)>;
    idx4 = mir.extract_field (idx4_pair) [] [index: mir.field_index 0]: <(mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>) -> (builtin.integer ui64)>;
    idx4_ovf = mir.extract_field (idx4_pair) [] [index: mir.field_index 1]: <(mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>) -> (builtin.integer i1)>;
    idx4_ok = mir.not (idx4_ovf) [] []: <(builtin.integer i1) -> (builtin.integer i1)>;
    mir.assert (idx4_ok) [] []: <(builtin.integer i1) -> ()>;
    src_pair = mir.checked_add (idx4, c3) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>)>;
    src_idx = mir.extract_field (src_pair) [] [index: mir.field_index 0]: <(mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>) -> (builtin.integer ui64)>;
    src_ovf = mir.extract_field (src_pair) [] [index: mir.field_index 1]: <(mir.tuple <[builtin.integer ui64, builtin.integer i1], [], [], 16, 8>) -> (builtin.integer i1)>;
    src_ok = mir.not (src_ovf) [] []: <(builtin.integer i1) -> (builtin.integer i1)>;
    mir.assert (src_ok) [] []: <(builtin.integer i1) -> ()>;
    x_len = mir.extract_field (x) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    x_ok = mir.lt (src_idx, x_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.assert (x_ok) [] []: <(builtin.integer i1) -> ()>;
    x_ptr = mir.extract_field (x) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    x_elem = mir.ptr_offset (x_ptr, src_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    v = mir.load (x_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    mul_rn = nvvm.scalar_arithmetic (v, scale) [] [nvvm_scalar_arithmetic_format: nvvm.scalar_arithmetic_format F32, nvvm_scalar_arithmetic_operation: nvvm.scalar_arithmetic_operation Mul, nvvm_scalar_arithmetic_rounding: nvvm.scalar_arithmetic_rounding Rn, nvvm_scalar_arithmetic_saturation: nvvm.scalar_arithmetic_saturation None, nvvm_scalar_arithmetic_subnormal: nvvm.scalar_arithmetic_subnormal Preserve]: <(builtin.fp32, builtin.fp32) -> (builtin.fp32)>;
    soft = nvvm.scalar_math (mul_rn) [] [nvvm_scalar_math_format: nvvm.scalar_math_format F32, nvvm_scalar_math_operation: nvvm.scalar_math_operation Tanh, nvvm_scalar_math_precision: nvvm.scalar_math_precision Approx, nvvm_scalar_math_subnormal: nvvm.scalar_math_subnormal Preserve]: <(builtin.fp32) -> (builtin.fp32)>;
    out_ptr = mir.extract_field (out) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    out_elem = mir.ptr_offset (out_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (out_elem, soft) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"#;
    let valid_lean =
        export_named_source(scale_quad_mir, "scale_quad").expect("exports scale_quad cleanly");

    // Negative: unmodeled warp vote / atomic ops still fail closed:
    let vote_mir = scale_quad_mir.replacen(
        "    mir.goto () [^exit] []: <() -> ()>",
        "    m32 = mir.constant () [] [value: builtin.integer <4294967295: ui32>]: <() -> (builtin.integer ui32)>;\n    _bal = nvvm.vote_sync_ballot (m32, in_n) [] []: <(builtin.integer ui32, builtin.integer i1) -> (builtin.integer ui32)>;\n    mir.goto () [^exit] []: <() -> ()>",
        1,
    );
    let vote_err = export_named_source(&vote_mir, "scale_quad")
        .expect_err("unmodeled vote_sync_ballot must fail closed");
    assert!(
        vote_err.to_string().contains("nvvm.vote_sync_ballot"),
        "unexpected error: {vote_err}"
    );

    if let Some(lean_dir) = lean_package_dir_if_available() {
        lean_exporter::runner::check_generated_lean(&valid_lean, &lean_dir).expect(
            "Lean 4 must accept `s0>=4*s1, s1>=p2` + `nvvm.scalar_arithmetic` / `nvvm.scalar_math`",
        );

        // Negative Lean check 1: off-by-one `4 * idx + 4` under `s0 >= 4 * s1` must fail:
        let oob_mir = scale_quad_mir.replacen(
            "c3 = mir.constant () [] [value: builtin.integer <3: ui64>]",
            "c3 = mir.constant () [] [value: builtin.integer <4: ui64>]",
            1,
        );
        let oob_lean = export_named_source(&oob_mir, "scale_quad").expect("exports oob_mir");
        lean_exporter::runner::check_generated_lean(&oob_lean, &lean_dir).expect_err(
            "Lean 4 must reject off-by-one `4 * idx + 4` under `x.len() >= 4 * out.len()`",
        );

        // Negative Lean check 2: inclusive `idx <= n` under `out.len() >= n` must fail:
        let inc_mir =
            scale_quad_mir.replacen("in_n = mir.lt (idx, n64)", "in_n = mir.le (idx, n64)", 1);
        let inc_lean = export_named_source(&inc_mir, "scale_quad").expect("exports inc_mir");
        lean_exporter::runner::check_generated_lean(&inc_lean, &lean_dir)
            .expect_err("Lean 4 must reject inclusive `idx <= n` under `out.len() >= n`");
    }
}

#[test]
fn verifies_cta_uniform_early_return_and_strided_bool_loops_and_rejects_divergent_returns() {
    let valid_mir = r#"
mir.func @strided_reduce_early_ret: builtin.function <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, builtin.integer ui32, mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> ()>
  [contract_requires_len_ge: builtin.string "0>=2", gpu_kernel: builtin.string "true", launch_contract_domain: builtin.integer <1: ui32>, reqntid_x: builtin.integer <256: ui32>, reqntid_y: builtin.integer <1: ui32>, reqntid_z: builtin.integer <1: ui32>, verify_disjoint_writes: builtin.string "2", verify_no_data_race: builtin.string "0", verify_no_oob_idx: builtin.string "0,2", verify_no_oob_shared: builtin.string "0", verify_uniform_barriers: builtin.string "true"]
{
  ^entry(x: mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>, m: builtin.integer ui32, y: mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>):
    bx32 = nvvm.read_ptx_sreg_ctaid_x () [] []: <() -> (builtin.integer ui32)>;
    skip_blk = mir.ge (bx32, m) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
    mir.cond_br (skip_blk) [^exit, ^init] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^init():
    tx32 = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;
    tid = mir.cast (tx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    k = mir.extract_field (x) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (builtin.integer ui64)>;
    t_bool = mir.constant () [] [value: builtin.integer <1: i1>]: <() -> (builtin.integer i1)>;
    mir.goto (tid, t_bool) [^loop_hdr] []: <(builtin.integer ui64, builtin.integer i1) -> ()>

  ^loop_hdr(it: builtin.integer ui64, flag: builtin.integer i1):
    c768 = mir.constant () [] [value: builtin.integer <768: ui64>]: <() -> (builtin.integer ui64)>;
    it_end = mir.add (it, c768) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    in_loop = mir.lt (it_end, k) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_loop) [^loop_body, ^post_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^loop_body():
    x_ptr = mir.extract_field (x) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    p0 = mir.ptr_offset (x_ptr, it) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    _v0 = mir.load (p0) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    p3 = mir.ptr_offset (x_ptr, it_end) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;
    _v3 = mir.load (p3) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;
    c1024 = mir.constant () [] [value: builtin.integer <1024: ui64>]: <() -> (builtin.integer ui64)>;
    it_next = mir.add (it, c1024) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    f_next = mir.not (flag) [] []: <(builtin.integer i1) -> (builtin.integer i1)>;
    mir.goto (it_next, f_next) [^loop_hdr] []: <(builtin.integer ui64, builtin.integer i1) -> ()>

  ^post_loop():
    lane = nvvm.read_ptx_sreg_laneid () [] []: <() -> (builtin.integer ui32)>;
    z32 = mir.constant () [] [value: builtin.integer <0: ui32>]: <() -> (builtin.integer ui32)>;
    is_l0 = mir.eq (lane, z32) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;
    mir.cond_br (is_l0) [^stage, ^sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^stage():
    c32 = mir.constant () [] [value: builtin.integer <32: ui64>]: <() -> (builtin.integer ui64)>;
    w_idx = mir.div (tid, c32) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    ws_base = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string "WSUM", source_name: builtin.string "k::WSUM"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    ws_slot = mir.ptr_offset (ws_base, w_idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    zf = mir.float_constant () [] [float_value: builtin.single 0]: <() -> (builtin.fp32)>;
    mir.store (ws_slot, zf) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.fp32) -> ()>;
    mir.goto () [^sync] []: <() -> ()>

  ^sync():
    nvvm.barrier0 () [] []: <() -> ()>;
    ws_r = mir.shared_alloc () [] [elem_type: builtin.type builtin.fp32, size: builtin.integer <8: i64>, alloc_key: builtin.string "WSUM", source_name: builtin.string "k::WSUM"]: <() -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    z64 = mir.constant () [] [value: builtin.integer <0: ui64>]: <() -> (builtin.integer ui64)>;
    ws0 = mir.ptr_offset (ws_r, z64) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>)>;
    s_val = mir.load (ws0) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:3,kind:Erased>) -> (builtin.fp32)>;
    bx64 = mir.cast (bx32) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;
    c256 = mir.constant () [] [value: builtin.integer <256: ui64>]: <() -> (builtin.integer ui64)>;
    b_off = mir.mul (bx64, c256) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    gid = mir.add (b_off, tid) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;
    y_len = mir.extract_field (y) [] [index: mir.field_index 1]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (builtin.integer ui64)>;
    in_y = mir.lt (gid, y_len) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer i1)>;
    mir.cond_br (in_y) [^write_y, ^exit] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

  ^write_y():
    y_ptr = mir.extract_field (y) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:true,kind:UniqueRef>) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    y_elem = mir.ptr_offset (y_ptr, gid) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>)>;
    mir.store (y_elem, s_val) [] []: <(mir.ptr <builtin.fp32 ,mutable:true,addrspace:0,kind:UniqueRef>, builtin.fp32) -> ()>;
    mir.goto () [^exit] []: <() -> ()>

  ^exit():
    mir.return () [] []: <() -> ()>
}
"#;
    let valid_lean = export_named_source(valid_mir, "strided_reduce_early_ret")
        .expect("transcribes strided_reduce_early_ret");
    if let Some(lean_dir) = lean_package_dir_if_available() {
        lean_exporter::runner::check_generated_lean(&valid_lean, &lean_dir)
            .expect("Lean 4 must accept CTA-uniform early return + strided bool loop");

        // Negative 1: thread-divergent early return (`tid_x >= m`) before `sync_threads()` must fail:
        let div_ret_mir = valid_mir.replacen(
            "skip_blk = mir.ge (bx32, m)",
            "tx_early = nvvm.read_ptx_sreg_tid_x () [] []: <() -> (builtin.integer ui32)>;\n    skip_blk = mir.ge (tx_early, m)",
            1,
        );
        let div_ret_lean = export_named_source(&div_ret_mir, "strided_reduce_early_ret")
            .expect("transcribes div_ret");
        lean_exporter::runner::check_generated_lean(&div_ret_lean, &lean_dir)
            .expect_err("Lean 4 must reject thread-divergent early return before sync_threads");

        // Negative 2: mid-loop data-dependent early return before `sync_threads()` must fail:
        let mid_loop_ret_mir = valid_mir.replacen(
            "    mir.goto (it_next, f_next) [^loop_hdr]",
            "    mir.cond_br (f_next) [^exit, ^cont_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^cont_loop():\n    mir.goto (it_next, f_next) [^loop_hdr]",
            1,
        );
        let mid_ret_lean = export_named_source(&mid_loop_ret_mir, "strided_reduce_early_ret")
            .expect("transcribes mid_loop_ret");
        lean_exporter::runner::check_generated_lean(&mid_ret_lean, &lean_dir)
            .expect_err("Lean 4 must reject divergent mid-loop early return before sync_threads");
    }
}
