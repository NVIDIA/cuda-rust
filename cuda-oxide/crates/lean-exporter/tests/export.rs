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
    let lean = export_named_source(VECADD_FIXTURE, "vecadd").expect("valid verified vecadd MIR");
    assert!(lean.contains("def kernelSummary : CudaOxide.Mir.Kernel.KernelSummary1D 3"));
    assert!(lean.contains("theorem verify_no_oob_idx_param0"));
    assert!(lean.contains("theorem verify_no_oob_idx_param1"));
    assert!(lean.contains("theorem verify_no_oob_idx_param2"));
    assert!(lean.contains("theorem verify_disjoint_writes_param2"));
    assert!(lean.contains("theorem verify_writes_all_elems_param2"));

    // Scenario 4a: Moving a load into the unguarded exit block must be rejected fail-closed.
    let unguarded_read = VECADD_FIXTURE.replace(
        "^exit():\n    mir.return",
        "^exit():\n    bad_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_elem = mir.ptr_offset (bad_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_val = mir.load (bad_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;\n    mir.return",
    );
    let err = export_named_source(&unguarded_read, "vecadd").unwrap_err();
    assert!(err.to_string().contains("unguarded read"));

    // Scenario 4b: Moving a store into the unguarded exit block must be rejected fail-closed.
    let unguarded_write = VECADD_FIXTURE
        .replace("builtin.fp32", "builtin.integer ui64")
        .replace(
            "^exit():\n    mir.return",
            "^exit():\n    bad_c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.disjoint_slice <builtin.integer ui64 ,[]>) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    bad_c_elem = mir.ptr_offset (bad_c_ptr, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    mir.store (bad_c_elem, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> ()>;\n    mir.return",
        );
    let err = export_named_source(&unguarded_write, "vecadd").unwrap_err();
    assert!(
        err.to_string().contains("unguarded write"),
        "unexpected error for unguarded write: {err}"
    );

    // Scenario 3: Inclusive branch guard (`idx <= c.len()` via `mir.le`) instead of `<` (`mir.lt`).
    let inclusive_guard = VECADD_FIXTURE.replace("mir.lt (idx, c_len)", "mir.le (idx, c_len)");
    let err = export_named_source(&inclusive_guard, "vecadd").unwrap_err();
    assert!(
        err.to_string().contains("unguarded read")
            || err.to_string().contains("unsupported operation `mir.le`"),
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

    // Scenario 5c: Indexing a slice by a non-thread-index value (`c_len` instead of `idx`).
    let bad_index_expr = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "a_elem = mir.ptr_offset (a_ptr, c_len)",
    );
    let err = export_named_source(&bad_index_expr, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("pointer offset must index a kernel slice parameter by the 1D thread index"),
        "unexpected error: {err}"
    );
}

#[test]
fn lean_typechecker_accepts_valid_vecadd_and_rejects_false_kernel_claims() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();
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
    assert!(off_by_one_lean.contains("indexOffset := 1"));
    assert!(off_by_one_lean.contains("CudaOxide.Contracts.NoOobIdx 256 1"));
    let err = lean_exporter::runner::check_generated_lean(&off_by_one_lean, &lean_dir)
        .expect_err("Lean must reject off-by-one `a[idx + 1]` access");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one access: {err}"
    );

    // Scenario 6: Admitted proof (`sorry`) must be caught by the `#print axioms` gate.
    let spoofed_sorry = baseline.replace(
        "exact CudaOxide.Proofs.Bounds.no_oob_idx_of_guard_le h_req",
        "sorry",
    );
    let err = lean_exporter::runner::check_generated_lean(&spoofed_sorry, &lean_dir)
        .expect_err("Lean runner must reject `sorry` proofs");
    assert!(
        err.to_string().contains("sorry"),
        "unexpected error for sorry proof: {err}"
    );
}

#[test]
fn verifies_2d_sgemm_tiled_views_and_rejects_shared_races_oob_and_divergent_barriers() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();
    let baseline = export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_views")
        .expect("valid sgemm_tiled_views MIR");
    assert!(baseline.contains("def kernelSummary : CudaOxide.Mir.Kernel.TiledKernelSummary2D"));
    assert!(baseline.contains("theorem verify_no_oob_idx_shared0_0"));
    assert!(baseline.contains("theorem verify_no_data_race_shared0_0_1"));
    assert!(baseline.contains("theorem verify_uniform_barriers"));

    // Stage 1 negative check: indexing SharedArray by a non-2D-tile expression (`k` instead of `smem_idx`).
    let bad_shared_offset = SGEMM_TILED_FIXTURE.replace(
        "slot_a = mir.ptr_offset (tile_a_mut, smem_idx)",
        "k_u64 = mir.cast (k) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    slot_a = mir.ptr_offset (tile_a_mut, k_u64)",
    );
    let err = export_named_source(&bad_shared_offset, "sgemm_tiled_views").unwrap_err();
    assert!(
        err.to_string()
            .contains("shared memory pointer offset must use a recognized 2D tile coordinate"),
        "unexpected error: {err}"
    );

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
    // before the barrier loop). Lean 4 rejects `UniformBarriers false ...`.
    let divergent_barrier = SGEMM_TILED_FIXTURE.replace(
        "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        "early_cond = mir.lt (tx, k) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;\n    mir.cond_br (early_cond) [^exit, ^enter_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^enter_loop():\n    mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
    );
    let divergent_lean = export_named_source(&divergent_barrier, "sgemm_tiled_views")
        .expect("exports UniformBarriers false obligation");
    assert!(divergent_lean.contains("barriersUniform := false"));
    let err = lean_exporter::runner::check_generated_lean(&divergent_lean, &lean_dir)
        .expect_err("Lean must reject divergent barrier when early return skips barrier loop");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for divergent barrier: {err}"
    );

    // Stage 2 (`sgemm_tiled_raw`): Verify 2D global slice `RowStrip` (`a`) and `ColStrip` (`b`)
    // are extracted from MIR pointer-offset math rather than parameter list order.
    let raw_baseline =
        export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_raw").expect("valid sgemm_tiled_raw");
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param4 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobRowStrip bufferLen"
    ));
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param5 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobColStrip bufferLen"
    ));
    lean_exporter::runner::check_generated_lean(&raw_baseline, &lean_dir)
        .expect("valid sgemm_tiled_raw MIR must pass Lean 4 verification");

    // Reversing the attribute list (`"5,4"` instead of `"4,5"`) must still map param 5 (`b`)
    // to `NoOobColStrip` and param 4 (`a`) to `NoOobRowStrip`.
    let swapped_order = SGEMM_TILED_FIXTURE.replace(
        "verify_no_oob_idx: builtin.string \"4,5\"",
        "verify_no_oob_idx: builtin.string \"5,4\"",
    );
    let swapped_lean =
        export_named_source(&swapped_order, "sgemm_tiled_raw").expect("swapped order still valid");
    assert!(swapped_lean.contains(
        "theorem verify_no_oob_idx_param5 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobColStrip bufferLen"
    ));
    assert!(swapped_lean.contains(
        "theorem verify_no_oob_idx_param4 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobRowStrip bufferLen"
    ));

    // Negative check A: indexing global slice `a` by a non-strip index (`a_len` instead of `a_idx`)
    // must be rejected fail-closed by `translate.rs`.
    let bad_global_offset = SGEMM_TILED_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, a_idx)",
        "a_elem = mir.ptr_offset (a_ptr, a_len)",
    );
    let err = export_named_source(&bad_global_offset, "sgemm_tiled_raw").unwrap_err();
    assert!(
        err.to_string()
            .contains("2D global slice pointer offset must use a row-strip"),
        "unexpected error for bad 2D global slice offset: {err}"
    );

    // Negative check B (Hole #2): missing Check 2 (`in_a = a_col < k` instead of `< a_eff`,
    // ignoring `a_start == u64::MAX`) or missing Check 1 (`a_fits` not checking `< a_len`)
    // must be rejected fail-closed before Lean!
    let unguarded_row_strip =
        SGEMM_TILED_FIXTURE.replace("in_a = mir.lt (a_col, a_eff)", "in_a = mir.lt (a_col, k)");
    let err = export_named_source(&unguarded_row_strip, "sgemm_tiled_raw").unwrap_err();
    assert!(
        err.to_string()
            .contains("2D global slice pointer offset must use a row-strip"),
        "missing `if a_col < a_eff` guard must be rejected fail-closed, got: {err}"
    );

    let missing_band_check = SGEMM_TILED_FIXTURE.replace(
        "a_fits = mir.lt (a_last, a_len)",
        "a_fits = mir.lt (zero_u32, k)",
    );
    let err = export_named_source(&missing_band_check, "sgemm_tiled_raw").unwrap_err();
    assert!(
        err.to_string()
            .contains("2D global slice pointer offset must use a row-strip"),
        "missing `a_start + (k - 1) < a.len()` band check must be rejected fail-closed, got: {err}"
    );

    // Negative check C (Hole #2): wrong stride (`m` instead of `n`) on `b`'s column strip
    // must be rejected fail-closed!
    let wrong_col_stride = SGEMM_TILED_FIXTURE.replace(
        "n_u64 = mir.cast (n) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
        "n_u64 = mir.cast (m) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;",
    );
    let err = export_named_source(&wrong_col_stride, "sgemm_tiled_raw").unwrap_err();
    assert!(
        err.to_string()
            .contains("2D global slice pointer offset must use a row-strip"),
        "mismatched column-strip stride must be rejected fail-closed, got: {err}"
    );

    // Negative check D (Hole #2): off-by-one `a[a_start + a_col + 1]` under valid strip guards
    // extracts `offset = 1` (`NoOobRowStrip bufferLen 1`) and is rejected by Lean 4's `omega`!
    let off_by_one_strip = SGEMM_TILED_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, a_idx)",
        "a_idx_p1 = mir.add (a_idx, one_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        a_elem = mir.ptr_offset (a_ptr, a_idx_p1)",
    );
    let off_by_one_strip_lean =
        export_named_source(&off_by_one_strip, "sgemm_tiled_raw").expect("exports offset=1 strip");
    assert!(off_by_one_strip_lean.contains("CudaOxide.Contracts.NoOobRowStrip bufferLen 1"));
    let err = lean_exporter::runner::check_generated_lean(&off_by_one_strip_lean, &lean_dir)
        .expect_err("Lean must reject off-by-one `a[a_start + a_col + 1]` row-strip access");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one 2D row strip: {err}"
    );
}

#[test]
fn verifies_stage3_warp_and_block_reductions_and_rejects_missing_leader_guard() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();

    // 1. 1D warp butterfly reduction (`warp_reduce_sum`)
    let warp_lean = export_named_source(WARP_REDUCE_FIXTURE, "warp_reduce_sum")
        .expect("valid warp_reduce_sum MIR");
    assert!(warp_lean.contains("def kernelSummary : CudaOxide.Mir.Kernel.KernelSummary1D 2"));
    assert!(warp_lean.contains("theorem verify_no_oob_idx_param0"));
    assert!(warp_lean.contains("theorem verify_disjoint_writes_param1"));
    assert!(warp_lean.contains("theorem verify_writes_all_elems_param1"));
    lean_exporter::runner::check_generated_lean(&warp_lean, &lean_dir)
        .expect("valid warp_reduce_sum MIR must pass Lean 4 verification");

    // 2. 2-phase shared-memory block reduction (`block_reduce_sum`)
    let block_lean = export_named_source(WARP_REDUCE_FIXTURE, "block_reduce_sum")
        .expect("valid block_reduce_sum MIR");
    assert!(block_lean.contains(".groupSlot 32"));
    assert!(block_lean.contains("leaderOnly := true"));
    assert!(block_lean.contains(".linearScan"));
    assert!(block_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_group_slot"));
    assert!(block_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_linear_scan"));
    assert!(block_lean.contains("CudaOxide.Proofs.Launch.no_data_race_group_leader_slot"));
    lean_exporter::runner::check_generated_lean(&block_lean, &lean_dir)
        .expect("valid block_reduce_sum MIR must pass Lean 4 verification");

    // 3. Negative check: forgetting `if lane == 0` (`mir.goto () [^leader_stage]` unconditionally)
    // causes all 32 threads in a warp to write `WARP_SUMS[tid / 32]` in Phase 0 (`leaderOnly := false`),
    // which Lean 4 rejects in `NoDataRace` with a counterexample!
    let unguarded_leader = WARP_REDUCE_FIXTURE.replace(
        "mir.cond_br (is_leader) [^leader_stage, ^barrier_sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
        "mir.goto () [^leader_stage] []: <() -> ()>",
    );
    let unguarded_lean = export_named_source(&unguarded_leader, "block_reduce_sum")
        .expect("exports unguarded group-slot obligation");
    assert!(!unguarded_lean.contains("leaderOnly := true"));
    let err = lean_exporter::runner::check_generated_lean(&unguarded_lean, &lean_dir)
        .expect_err("Lean must reject unguarded `WARP_SUMS[tid / 32]` intra-warp write race");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing `lane == 0` guard: {err}"
    );
}

#[test]
fn verifies_stage4_subwarp_bit_splits_and_multibuffer_shared_staging() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();

    // 1. Sub-warp 16x16 bit-split staging `(tid >> 4) * 16 + (tid & 15)` + lane scan `(tid & 15) * 16 + j`
    let split_lean = export_named_source(SUBWARP_MULTIBUF_FIXTURE, "subwarp_bit_split_stage")
        .expect("valid subwarp_bit_split_stage MIR");
    assert!(split_lean.contains(".bitSplitSlot 16 16"));
    assert!(split_lean.contains(".groupLaneScan 16 16"));
    assert!(split_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_bit_split_slot"));
    assert!(split_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_group_lane_scan"));
    assert!(split_lean.contains("CudaOxide.Proofs.Launch.no_data_race_bit_split_slot"));
    lean_exporter::runner::check_generated_lean(&split_lean, &lean_dir)
        .expect("valid subwarp_bit_split_stage MIR must pass Lean 4 verification");

    // 2. 3-phase, 3-buffer (`S_ACT_DIRECT`, `S_XS_DIRECT`, `S_X_DIRECT`) sub-warp staging
    let multi_lean = export_named_source(SUBWARP_MULTIBUF_FIXTURE, "multibuf_subwarp_stage")
        .expect("valid multibuf_subwarp_stage MIR");
    assert!(multi_lean.contains("numBarriers := 2"));
    assert!(multi_lean.contains(".packedSlot 4"));
    assert!(multi_lean.contains(".groupLaneScan 16 4"));
    assert!(multi_lean.contains(".groupLaneScan 16 16"));
    assert!(multi_lean.contains(".groupSlot 16"));
    assert!(multi_lean.contains("phase := 2"));
    assert!(multi_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_1d_slot"));
    assert!(multi_lean.contains("CudaOxide.Proofs.Bounds.no_oob_tile_packed_slot"));
    assert!(multi_lean.contains("CudaOxide.Proofs.Launch.no_data_race_packed_slot"));
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

    // 6. General `GuardExpr` test: Non-zero leader lane (`if k_lane == 7` instead of `0`)
    // emits `guard := (.eq (.mod .tx 16) (.const 7))` and is proved race-free by `GuardExpr.eval` + `omega`!
    let lane_seven_leader = SUBWARP_MULTIBUF_FIXTURE.replace(
        "zero_lane = mir.constant () [] [value: builtin.integer <0: ui64>]",
        "zero_lane = mir.constant () [] [value: builtin.integer <7: ui64>]",
    );
    let lane_seven_lean = export_named_source(&lane_seven_leader, "multibuf_subwarp_stage")
        .expect("exports GuardExpr for `k_lane == 7`");
    assert!(lane_seven_lean.contains("guard := (.eq (.mod .tx 16) (.const 7))"));
    lean_exporter::runner::check_generated_lean(&lane_seven_lean, &lean_dir)
        .expect("Lean must prove `NoDataRace` for arbitrary leader lane `k_lane == 7`");

    // 7. General `IdxExpr` test: Multi-level index expression `(w * 64 + tid) + 0` (outside the
    // 8 named templates) emits `.expr (.add (.add (.mul .loopVar (.const 64)) .tx) (.const 0))`
    // with `guard := (.lt .tx (.const 64))` and is proved in-bounds and race-free by
    // `IdxExpr.eval` + `GuardExpr.eval` + `omega`!
    let coalesced_strided = SUBWARP_MULTIBUF_FIXTURE.replace(
        "tid_x4 = mir.mul (tid_u64, c4) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        pack_idx = mir.add (tid_x4, w)",
        "w_x64 = mir.mul (w, c64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        w_tid = mir.add (w_x64, tid_u64) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n        pack_idx = mir.add (w_tid, zero_w)",
    );
    let coalesced_lean = export_named_source(&coalesced_strided, "multibuf_subwarp_stage")
        .expect("exports general `.expr` for coalesced `(w * 64 + tid) + 0`");
    assert!(
        coalesced_lean.contains(".expr (.add (.add (.mul .loopVar (.const 64)) .tx) (.const 0))")
    );
    lean_exporter::runner::check_generated_lean(&coalesced_lean, &lean_dir)
        .expect("Lean must prove `NoOobTileAccess` and `NoDataRace` for general `.expr` `(w * 64 + tid) + 0`");
}

#[test]
fn verifies_stage5_async_copy_and_mbarrier_token_ordering() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();

    // 1. Valid Stage 5 async TMA copy + `mbarrier_arrive_expect_tx` + `mbarrier_try_wait_parity`
    let async_lean = export_named_source(ASYNC_COPY_FIXTURE, "async_copy_stage")
        .expect("valid async_copy_stage MIR");
    assert!(async_lean.contains("numBarriers := 2"));
    assert!(async_lean.contains("CudaOxide.Proofs.Launch.no_data_race_single_leader"));
    assert!(async_lean.contains("CudaOxide.Proofs.Launch.no_data_race_diff_phase"));
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

    // 4. Negative check C: Waiting on a DIFFERENT barrier (`BAR_OTHER` instead of `BAR`)
    // does not unlock `TILE` (which was bound to `BAR`), so `NoDataRace` fails in Lean!
    let wrong_barrier = ASYNC_COPY_FIXTURE.replace(
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
    assert!(oversized_lean.contains("loopBound := 8192"));
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
        lean_exporter::translate::TranslatedFunction::Kernel {
            proven_assert_ops, ..
        }
        | lean_exporter::translate::TranslatedFunction::TiledKernel2D {
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
    let lean_dir = lean_exporter::runner::default_lean_package_dir();

    // 1. Positive disjoint `mir.bitor`: `(m_local * 16) | (tid & 15)` has disjoint bit
    // ranges (`row_off` aligned to 16, `k_lane < 16`), so it lowers to `.bitSplitSlot 16 16`
    // and passes Lean 4 verification.
    let disjoint_bitor = SUBWARP_MULTIBUF_FIXTURE.replacen(
        "split_idx = mir.add (row_off, k_lane)",
        "split_idx = mir.bitor (row_off, k_lane)",
        1,
    );
    let disjoint_lean = export_named_source(&disjoint_bitor, "subwarp_bit_split_stage")
        .expect("disjoint `(m_local * 16) | (tid & 15)` must export cleanly");
    assert!(disjoint_lean.contains(".bitSplitSlot 16 16"));
    lean_exporter::runner::check_generated_lean(&disjoint_lean, &lean_dir)
        .expect("Lean must prove disjoint `(m_local * 16) | (tid & 15)` safe");

    // 2. Negative overlapping `mir.bitor`: `SUBWARP_TILE[tid | 1] = 0.0` collides between
    // threads 0 and 1 (`0 | 1 == 1 | 1 == 1`) and must be rejected fail-closed!
    let overlapping_bitor = SUBWARP_MULTIBUF_FIXTURE.replacen(
        "split_idx = mir.add (row_off, k_lane)",
        "c1_bad = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n        split_idx = mir.bitor (tid, c1_bad)",
        1,
    );
    let err = export_named_source(&overlapping_bitor, "subwarp_bit_split_stage")
        .expect_err("overlapping `tid | 1` must be rejected fail-closed");
    assert!(
        err.to_string()
            .contains("shared memory pointer offset must use a recognized 2D tile coordinate"),
        "unexpected error for overlapping `tid | 1`: {err}"
    );

    // 3. Negative narrowing integer cast (`((tid * 2) as u8) as u64`): wraps modulo 256 so
    // thread `0` and thread `128` both write `SUBWARP_TILE[0]`. Must emit `Mod(..., 256)`
    // and fail `NoDataRace` in Lean 4's `omega`!
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
        .expect("narrowing `as u8` exports modulo-256 index expression");
    assert!(narrowing_lean.contains(".mod (.mul .tx (.const 2)) 256"));
    let err = lean_exporter::runner::check_generated_lean(&narrowing_lean, &lean_dir)
        .expect_err("Lean must reject `((tid * 2) as u8) as usize` wraparound collision");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for narrowing `as u8` collision: {err}"
    );

    // 4. Negative 1D narrowing cast (`a[(idx as u8) as u64]` in `vecadd`): must be rejected
    // fail-closed because `idx` spans dynamic blocks and wraps modulo 256.
    let vecadd_narrow_cast = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "idx_u8 = mir.cast (idx) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui64) -> (builtin.integer ui8)>;\n    idx_wrapped = mir.cast (idx_u8) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui8) -> (builtin.integer ui64)>;\n    a_elem = mir.ptr_offset (a_ptr, idx_wrapped)",
    );
    let err = export_named_source(&vecadd_narrow_cast, "vecadd")
        .expect_err("1D narrowing `(idx as u8) as usize` must be rejected fail-closed");
    assert!(
        err.to_string()
            .contains("pointer offset must index a kernel slice parameter by the 1D thread index"),
        "unexpected error for 1D narrowing cast: {err}"
    );
}

#[test]
fn verifies_cyclic_loop_phases_with_init_barrier_and_reordered_blocks() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();

    // 1. Adding an acyclic init `nvvm.barrier0` BEFORE `^tile_header` while keeping BOTH
    // in-loop barriers (`^phase0_stage` + `^phase1_end`) advances in-loop phases to `1` and `2`
    // and passes Lean 4 verification.
    let init_plus_two_loop_bars = SGEMM_TILED_FIXTURE.replacen(
        "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        "nvvm.barrier0 () [] []: <() -> ()>;\n        mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        1,
    );
    let valid_init_lean = export_named_source(&init_plus_two_loop_bars, "sgemm_tiled_views")
        .expect("init barrier + 2-barrier loop must export cleanly");
    assert!(valid_init_lean.contains("numBarriers := 3"));
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
