/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unit tests for the libdevice math shard parser, evidence validator, and code generator.

use super::*;

/// Return the `cuda-oxide` workspace root directory.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Build an in-memory schema-2 `libdevice_math` shard fixture from function TOML blocks.
fn fixture(functions: &str) -> Shard {
    parse(format!("schema = 2\nfamily = \"libdevice_math\"\n[libdevice_math]\nevidence = \"evidence/test.json\"\nlibdevice_sha256 = \"{}\"\nwidths = [\"f32\", \"f64\"]\n{functions}", "a".repeat(64)).as_bytes()).unwrap()
}

/// Verify that all 124 shard rows (120 libm exports plus core-only routes) match `dialect-mir`.
#[test]
fn generated_metadata_covers_120_exports_and_core_only_routes() {
    let shard = read(&root()).unwrap();
    let rows = mappings(&shard).unwrap();
    assert_eq!(rows.len(), 124);
    assert_eq!(rows.iter().filter(|row| row.libm.is_some()).count(), 120);
    assert_eq!(expected_symbols(&rows).unwrap().len(), 102);
    for row in &rows {
        if let Some(variant) = &row.variant {
            let callee = format!("{}{}", compiled::PLACEHOLDER_PREFIX, row.placeholders[0]);
            let intrinsic =
                compiled::RustFloatMathIntrinsic::from_placeholder_callee(&callee).unwrap();
            assert_eq!(format!("{intrinsic:?}"), *variant);
            assert_eq!(intrinsic.placeholder_callee(), callee);
            if let Some((arguments, _)) = intrinsic.scalar_signature(row.width == "f64") {
                assert_eq!(arguments.len(), intrinsic.arg_count());
            }
            assert_eq!(
                intrinsic.libdevice_symbol(row.width == "f64"),
                row.libdevice.as_deref()
            );
            assert_eq!(
                intrinsic.llvm_symbol(row.width == "f64"),
                row.llvm.as_deref()
            );
            match row.lowering.as_deref() {
                None => assert!(matches!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::ScalarCall { arity, .. }
                        if arity == intrinsic.arg_count()
                )),
                Some("handwritten:lower_bessel_j1") => assert!(matches!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::BesselJ1 { symbol }
                        if Some(symbol) == row.libdevice.as_deref()
                )),
                Some("handwritten:emit_frexp") => assert!(matches!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::Frexp { symbol }
                        if Some(symbol) == row.libdevice.as_deref()
                )),
                Some("handwritten:emit_modf") => assert!(matches!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::Modf { symbol }
                        if Some(symbol) == row.libdevice.as_deref()
                )),
                Some("handwritten:emit_remquo") => assert!(matches!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::Remquo { symbol }
                        if Some(symbol) == row.libdevice.as_deref()
                )),
                Some("handwritten:emit_lgamma_r") => assert_eq!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::LgammaR
                ),
                Some("handwritten:lower_float_minmax_nsz") => assert_eq!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::MinMaxNsz {
                        is_max: variant.starts_with("Max")
                    }
                ),
                Some("handwritten:lower_float_ieee_minmax") => assert_eq!(
                    intrinsic.lowering(),
                    compiled::RustFloatMathLowering::IeeeMinMax {
                        is_max: variant.starts_with("Maximum"),
                        propagate_nan: !variant.contains("Number"),
                    }
                ),
                Some(other) => panic!("unexpected handwritten lowering {other}"),
            }
            for path in &row.paths {
                assert_eq!(
                    compiled::RustFloatMathIntrinsic::from_core_path(path),
                    Some(intrinsic)
                );
            }
            if let Some(alias) = &row.libm {
                assert_eq!(
                    compiled::RustFloatMathIntrinsic::from_libm_path(&format!("libm::{alias}")),
                    Some(intrinsic)
                );
            }
        }
    }
}

/// Compiled `dialect-mir::rust_intrinsics` consumer used to verify generated dispatch.
#[path = "../../../dialect-mir/src/rust_intrinsics.rs"]
mod compiled;

/// Pre-migration baseline route recorded from the handwritten math tables.
#[derive(Deserialize)]
struct LegacyRoute {
    path: String,
    variant: String,
    placeholder: String,
    libdevice: Option<String>,
    llvm: Option<String>,
    link: String,
}

/// Verify that every pre-migration compiler path, symbol, and link route is preserved.
#[test]
fn compiler_dispatch_preserves_every_pre_migration_path_symbol_and_link_route() {
    let expected: Vec<LegacyRoute> =
        serde_json::from_str(include_str!("migration-baseline.json")).unwrap();
    assert_eq!(expected.len(), 214);
    let shard = read(&root()).unwrap();
    let rows = mappings(&shard).unwrap();
    let mut actual_paths: BTreeSet<_> = rows
        .iter()
        .flat_map(|row| row.paths.iter().cloned())
        .collect();
    actual_paths.extend(
        rows.iter()
            .filter(|row| row.variant.is_some())
            .filter_map(|row| row.libm.as_ref().map(|name| format!("libm::{name}"))),
    );
    for (_, _, name) in compiler::FAST {
        actual_paths.extend([
            format!("core::intrinsics::{name}"),
            format!("std::intrinsics::{name}"),
        ]);
    }
    let legacy_paths: BTreeSet<_> = expected.iter().map(|row| row.path.clone()).collect();
    assert!(legacy_paths.is_subset(&actual_paths));
    for row in expected {
        let intrinsic = compiled::RustFloatMathIntrinsic::from_core_path(&row.path).unwrap();
        assert_eq!(format!("{intrinsic:?}"), row.variant, "{}", row.path);
        let placeholder = format!("{}{}", compiled::PLACEHOLDER_PREFIX, row.placeholder);
        assert_eq!(intrinsic.placeholder_callee(), placeholder, "{}", row.path);
        assert_eq!(
            compiled::RustFloatMathIntrinsic::from_placeholder_callee(&placeholder),
            Some(intrinsic)
        );
        let is_f64 = row.variant == "Fabs" && row.path.ends_with("fabs");
        assert_eq!(
            intrinsic.libdevice_symbol(is_f64),
            row.libdevice.as_deref(),
            "{}",
            row.path
        );
        assert_eq!(
            intrinsic.llvm_symbol(is_f64),
            row.llvm.as_deref(),
            "{}",
            row.path
        );
        assert_eq!(
            compiled::is_libdevice_backed_placeholder(&placeholder),
            row.link == "always",
            "{}",
            row.path
        );
        assert_eq!(
            compiled::is_backend_dependent_libdevice_placeholder(&placeholder),
            row.link == "libnvvm",
            "{}",
            row.path
        );
        assert!(compiled::is_known_placeholder(&placeholder));
        assert_eq!(
            compiled::is_intrinsic_lowered_cmath_shim(&row.path),
            row.path.starts_with("std::sys::cmath::")
        );
        if let Some(alias) = row.path.strip_prefix("libm::") {
            assert_eq!(
                compiled::RustFloatMathIntrinsic::from_core_path(&format!(
                    "libm::math::module::{alias}"
                )),
                Some(intrinsic)
            );
            assert_eq!(
                compiled::RustFloatMathIntrinsic::from_core_path(&format!("user::libm::{alias}")),
                None
            );
        }
    }
}

/// Ensure unknown paths, unsupported widths, and unmapped placeholders are rejected.
#[test]
fn unregistered_aliases_and_unknown_placeholders_remain_rejected() {
    for path in [
        "libm::unknown_future_math",
        "libm::fmaximumf128",
        "libm::sqrtf128",
        "user::std::sys::cmath::tgamma",
        "libmath::sqrtf",
    ] {
        assert_eq!(
            compiled::RustFloatMathIntrinsic::from_core_path(path),
            None,
            "{path}"
        );
        assert!(!compiled::is_intrinsic_lowered_cmath_shim(path));
    }
    let unknown = format!("{}future_math_f32", compiled::PLACEHOLDER_PREFIX);
    assert_eq!(
        compiled::RustFloatMathIntrinsic::from_placeholder_callee(&unknown),
        None
    );
    assert!(!compiled::is_known_placeholder(&unknown));
    assert!(!compiled::is_libdevice_backed_placeholder(&unknown));
    assert!(!compiled::is_backend_dependent_libdevice_placeholder(
        &unknown
    ));
    assert_eq!(
        compiled::RustFloatMathIntrinsic::from_libm_path("libm::sincosf"),
        None
    );
    assert_eq!(compiled::libm_sincos_is_f64("libm::sincosf"), Some(false));
    assert_eq!(
        compiled::libm_sincos_is_f64("libm::math::sincos::sincos"),
        Some(true)
    );
    assert_eq!(compiled::libm_sincos_is_f64("user::libm::sincos"), None);
    assert_eq!(compiled::RustFloatMathIntrinsic::LdexpF64.arg_count(), 2);
    assert_eq!(compiled::RustFloatMathIntrinsic::FmaF32.arg_count(), 3);
    assert_eq!(compiled::RustFloatMathIntrinsic::FrexpF64.arg_count(), 1);
}

/// Check that shared aliases and handwritten adapters preserve their link and ABI rules.
#[test]
fn aliases_and_handwritten_controls_keep_their_semantics() {
    let rows = mappings(&read(&root()).unwrap()).unwrap();
    let find = |name: &str| {
        rows.iter()
            .find(|row| row.libm.as_deref() == Some(name))
            .unwrap()
    };
    for (alias, original) in [("roundeven", "rint"), ("roundevenf", "rintf")] {
        assert_eq!(find(alias).placeholders, find(original).placeholders);
        assert_eq!(find(alias).libdevice, find(original).libdevice);
        assert_eq!(find(alias).llvm, find(original).llvm);
    }
    for name in ["fmax", "fmaxf", "fmin", "fminf"] {
        assert_eq!(find(name).link, Link::Never);
        assert!(find(name).libdevice.is_none());
        assert!(find(name).llvm.is_none());
    }
    for name in [
        "fmaximum",
        "fmaximumf",
        "fminimum",
        "fminimumf",
        "fmaximum_num",
        "fmaximum_numf",
        "fminimum_num",
        "fminimum_numf",
    ] {
        let row = find(name);
        assert_eq!(row.link, Link::Never);
        assert_eq!(
            row.lowering.as_deref(),
            Some("handwritten:lower_float_ieee_minmax")
        );
        assert_eq!(
            row.signature,
            if name.ends_with('f') {
                "(f32, f32) -> f32"
            } else {
                "(f64, f64) -> f64"
            }
        );
        assert!(row.libdevice.is_none());
        assert!(row.llvm.is_none());
    }
    assert_eq!(find("fabs").placeholders, find("fabsf").placeholders);
    assert_eq!(find("floor").link, Link::Libnvvm);
    assert_eq!(find("sqrt").link, Link::Always);
    for name in ["lgamma_r", "lgammaf_r"] {
        let row = find(name);
        assert_eq!(row.link, Link::Always);
        assert_eq!(row.lowering.as_deref(), Some("handwritten:emit_lgamma_r"));
        assert_eq!(
            row.signature,
            if name == "lgamma_r" {
                "(f64) -> (f64, i32)"
            } else {
                "(f32) -> (f32, i32)"
            }
        );
        assert!(row.libdevice.is_none());
    }
    for (name, symbol) in [("j1", "__nv_j1"), ("j1f", "__nv_j1f")] {
        let row = find(name);
        assert_eq!(row.link, Link::Always);
        assert_eq!(row.libdevice.as_deref(), Some(symbol));
        assert_eq!(row.lowering.as_deref(), Some("handwritten:lower_bessel_j1"));
        assert_eq!(
            row.signature,
            if name == "j1" {
                "(f64) -> f64"
            } else {
                "(f32) -> f32"
            }
        );
    }
}

/// Verify that invalid variant names, constants, and handwritten handlers fail closed.
#[test]
fn compiler_identities_and_handwritten_handlers_fail_closed() {
    let base = "[[libdevice_math.function]]\nname = \"demo\"\nsignature = \"(F) -> F\"\n";
    for (suffix, expected) in [
        ("libm = []\n", "no compiler entry point"),
        ("variant = \"demo{width}\"\n", "CamelCase"),
        ("constant = \"callee_demo\"\n", "CALLEE_"),
        ("variant = \"FaddFast\"\n", "collides with fast arithmetic"),
        ("core = [\"fadd_fast\"]\n", "collides with fast arithmetic"),
        ("variant = \"Demo\"\n", "duplicate math variant"),
        (
            "variant = \"Demo\"\nconstant = \"CALLEE_DEMO\"\nplaceholders = [\"demo\"]\n",
            "generic-width math identity",
        ),
        ("variant = \"Frexp{Width}\"\n", "frexp identity"),
        ("variant = \"LgammaR{Width}\"\n", "lgamma_r identity"),
        ("variant = \"MaxNumNsz{Width}\"\n", "min/max identity"),
        ("variant = \"Maximum{Width}\"\n", "C23 min/max identity"),
        ("variant = \"J1{Width}\"\n", "j1 identity"),
        (
            "lowering = \"handwritten:lower_bessel_j1\"\nlibdevice = \"__nv_demo{suffix}\"\n",
            "j1 requires its exact unary",
        ),
        (
            "lowering = \"handwritten:lower_float_ieee_minmax\"\nlink = \"never\"\n",
            "C23 min/max requires its exact binary",
        ),
        (
            "variant = \"Fabs\"\nconstant = \"CALLEE_FABS\"\nplaceholders = [\"fabs\"]\n",
            "generic fabs",
        ),
        (
            "core = [\"demo{width}\", \"demo{width}\"]\n",
            "duplicate compiler math path",
        ),
        (
            "lowering = \"handwritten:unknown_adapter\"\n",
            "unsupported handwritten",
        ),
        (
            "lowering = \"handwritten:emit_frexp\"\nlibdevice = \"__nv_demo{suffix}\"\n",
            "float/i32 tuple ABI",
        ),
        (
            "lowering = \"handwritten:emit_lgamma_r\"\n",
            "composed float/i32 tuple ABI",
        ),
        (
            "lowering = \"handwritten:emit_modf\"\nlibdevice = \"__nv_demo{suffix}\"\n",
            "exact tuple ABI",
        ),
        (
            "lowering = \"handwritten:emit_remquo\"\nlibdevice = \"__nv_demo{suffix}\"\n",
            "exact tuple ABI",
        ),
        (
            "[[libdevice_math.function]]\nname = \"other\"\nsignature = \"(F) -> F\"\nvariant = \"Demo{Width}\"\n",
            "duplicate math variant",
        ),
        (
            "[[libdevice_math.function]]\nname = \"other\"\nsignature = \"(F) -> F\"\nconstant = \"CALLEE_DEMO_{WIDTH}\"\n",
            "duplicate math constant",
        ),
    ] {
        let error = mappings(&fixture(&format!("{base}{suffix}")))
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{suffix}: {error}");
    }
}

/// Confirm deterministic output ordering and drift detection on generated artifacts.
#[test]
fn render_is_independent_of_row_order_and_check_detects_drift() {
    let mut shard = read(&root()).unwrap();
    let before = render::outputs(
        &mappings(&shard).unwrap(),
        &shard.libdevice_math.libdevice_sha256,
    );
    shard.libdevice_math.functions.reverse();
    let after = render::outputs(
        &mappings(&shard).unwrap(),
        &shard.libdevice_math.libdevice_sha256,
    );
    assert_eq!(before, after);
    for (path, source) in before {
        let source = if path.extension().is_some_and(|extension| extension == "rs") {
            crate::util::rustfmt_source(&source).unwrap()
        } else {
            source
        };
        crate::util::check_contents(&root().join(&path), &source).unwrap();
        let mut stale = source.clone();
        stale.push_str("// drift\n");
        assert!(crate::util::check_contents(&root().join(&path), &stale).is_err());
    }
}

/// Verify C ABI signature rendering for scalar, integer, and out-parameter classes.
#[test]
fn signature_classes_distinguish_integer_results_arguments_and_out_parameters() {
    for (signature, c32, c64) in [
        (Signature::Unary, "float(float)", "double(double)"),
        (
            Signature::Binary,
            "float(float, float)",
            "double(double, double)",
        ),
        (
            Signature::Ternary,
            "float(float, float, float)",
            "double(double, double, double)",
        ),
        (Signature::IntegerResult, "i32(float)", "i32(double)"),
        (
            Signature::IntegerArgument,
            "float(float, i32)",
            "double(double, i32)",
        ),
        (
            Signature::FloatIntegerPair,
            "float(float, ptr)",
            "double(double, ptr)",
        ),
        (
            Signature::FloatPair,
            "float(float, ptr)",
            "double(double, ptr)",
        ),
        (
            Signature::RemainderPair,
            "float(float, float, ptr)",
            "double(double, double, ptr)",
        ),
    ] {
        assert_eq!(signature.c_abi("f32"), c32);
        assert_eq!(signature.c_abi("f64"), c64);
    }
    let adapter = fixture(
        "[[libdevice_math.function]]\nname = \"frexp\"\nsignature = \"(F) -> (F, i32)\"\nlowering = \"handwritten:emit_frexp\"\nlibdevice = \"__nv_frexp{suffix}\"\n",
    );
    let rows = mappings(&adapter).unwrap();
    assert_eq!(
        expected_symbols(&rows).unwrap()["__nv_frexpf"],
        "float(float, ptr)"
    );
    assert_eq!(rows[0].signature, "(f64) -> (f64, i32)");
    for (name, signature, adapter, abi) in [
        ("modf", "(F) -> (F, F)", "emit_modf", "float(float, ptr)"),
        (
            "remquo",
            "(F, F) -> (F, i32)",
            "emit_remquo",
            "float(float, float, ptr)",
        ),
    ] {
        let input = fixture(&format!(
            "[[libdevice_math.function]]\nname = \"{name}\"\nsignature = \"{signature}\"\nlowering = \"handwritten:{adapter}\"\nlibdevice = \"__nv_{name}{{suffix}}\"\n"
        ));
        let rows = mappings(&input).unwrap();
        assert_eq!(
            expected_symbols(&rows).unwrap()[&format!("__nv_{name}f")],
            abi
        );
    }
}

/// Ensure duplicate functions, malformed templates, and conflicting symbols fail closed.
#[test]
fn malformed_rows_fail_closed() {
    let base = "[[libdevice_math.function]]\nname = \"demo\"\nsignature = \"(F) -> F\"\n";
    for (suffix, expected) in [
        (
            "[[libdevice_math.function]]\nname = \"demo\"\nsignature = \"(F) -> F\"\n",
            "duplicate math function",
        ),
        ("libm = [\"demo\", \"demo\"]\n", "duplicate libm spelling"),
        (
            "placeholders = [\"bad{unknown}\"]\n",
            "invalid math name/template",
        ),
        ("link = \"never\"\n", "never-linked math"),
        ("link = \"libnvvm\"\n", "backend-dependent math"),
        ("llvm = \"llvm_demo_{width}\"\n", "backend-dependent math"),
        (
            "placeholders = [\"left{width}\", \"right{width}\"]\n",
            "one placeholder",
        ),
        (
            "lowering = \"handwritten:emit_demo\"\nllvm = \"llvm_demo_{width}\"\nlink = \"libnvvm\"\n",
            "unsupported handwritten",
        ),
        (
            "[[libdevice_math.function]]\nname = \"other\"\nsignature = \"(F) -> F\"\nplaceholders = [\"demo{width}\"]\n",
            "duplicate scalar math placeholder",
        ),
        (
            "libdevice = \"user_demo{suffix}\"\n",
            "must start with __nv_",
        ),
        (
            "lowering = \"arbitrary_rust()\"\n",
            "invalid handwritten math lowering",
        ),
    ] {
        let shard = fixture(&format!("{base}{suffix}"));
        let error = mappings(&shard).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
    let tuple = fixture(&base.replace("(F) -> F", "(F) -> (F, i32)"));
    assert!(
        mappings(&tuple)
            .unwrap_err()
            .to_string()
            .contains("handwritten adapter")
    );
    let cmath = fixture(&format!(
        "{base}cmath = \"demo{{suffix}}\"\n[[libdevice_math.function]]\nname = \"other\"\nsignature = \"(F) -> F\"\ncmath = \"demo{{suffix}}\"\n"
    ));
    assert!(
        mappings(&cmath)
            .unwrap_err()
            .to_string()
            .contains("duplicate cmath")
    );
    let conflict = fixture(&format!(
        "{base}libdevice = \"__nv_shared{{suffix}}\"\n[[libdevice_math.function]]\nname = \"other\"\nsignature = \"(F, F) -> F\"\nlibdevice = \"__nv_shared{{suffix}}\"\n"
    ));
    assert!(
        expected_symbols(&mappings(&conflict).unwrap())
            .unwrap_err()
            .to_string()
            .contains("conflicting libdevice signatures")
    );
}

/// Reject schema mismatches, path traversal in evidence, and bitcode hash drift.
#[test]
fn schema_and_evidence_pins_are_strict() {
    let source = fs::read_to_string(root().join("intrinsics").join(SHARD)).unwrap();
    for changed in [
        source.replacen("schema = 2", "schema = 3", 1),
        source.replacen("family = \"libdevice_math\"", "family = \"scalar_math\"", 1),
        source.replacen("[\"f32\", \"f64\"]", "[\"f16\"]", 1),
        source.replacen(
            "evidence/libdevice_math/cuda-13.0.json",
            "evidence/../outside.json",
            1,
        ),
        source.replacen("schema = 2", "schema = 2\nunknown = true", 1),
        source.replacen("(F) -> F", "(F) -> bool", 1),
    ] {
        assert!(parse(changed.as_bytes()).is_err());
    }
    let shard = read(&root()).unwrap();
    let rows = mappings(&shard).unwrap();
    let mut evidence: Evidence = read_json(
        &root()
            .join("intrinsics")
            .join(&shard.libdevice_math.evidence),
    )
    .unwrap();
    validate_evidence(&shard, &rows, &evidence).unwrap();
    evidence
        .symbols
        .insert("__nv_sinf".into(), "double(double)".into());
    assert!(
        validate_evidence(&shard, &rows, &evidence)
            .unwrap_err()
            .to_string()
            .contains("signature mismatch")
    );
    evidence.bitcode_sha256 = "0".repeat(64);
    assert!(
        validate_evidence(&shard, &rows, &evidence)
            .unwrap_err()
            .to_string()
            .contains("hash")
    );
}

/// Require exported LLVM definitions with matching return and parameter types.
#[test]
fn llvm_evidence_requires_exported_definitions_and_exact_signatures() {
    let expected = BTreeMap::from([
        ("__nv_demo".into(), "double(double, i32)".into()),
        ("__nv_pairf".into(), "float(float, ptr)".into()),
    ]);
    let ir = "declare float @__nv_unrelated(float)\ndefine void @control() { ret void }\ndefine double @__nv_demo(\n double %x, i32 noundef %n) { ret double %x }\ndefine float @__nv_pairf(float %x, ptr nocapture %out) { ret float %x }\n";
    assert_eq!(evidence::signatures(ir, &expected).unwrap(), expected);
    for changed in [
        ir.replace("define double @__nv_demo", "declare double @__nv_demo"),
        ir.replace("i32 noundef %n", "i64 %n"),
        ir.replace(
            "define double @__nv_demo",
            "define internal double @__nv_demo",
        ),
        ir.replace("ptr nocapture", "ptr addrspace(1) nocapture"),
        format!("{ir}\ndefine double @__nv_demo(double %x, i32 %n) {{ ret double %x }}"),
    ] {
        assert!(evidence::signatures(&changed, &expected).is_err());
    }
    let typed = ir.replace("ptr nocapture", "i32* nocapture");
    assert_eq!(evidence::signatures(&typed, &expected).unwrap(), expected);
}

/// Verify that the shard covers 120 of the 124 public f32/f64 `libm` 0.2.16 exports.
#[test]
fn libm_0_2_16_coverage_leaves_only_semantic_functions() {
    /// Snapshot of public `f32`/`f64` exports from `libm` 0.2.16.
    #[derive(Deserialize)]
    struct PublicExports {
        #[serde(rename = "crate")]
        name: String,
        version: String,
        crate_sha256: String,
        exports: BTreeSet<String>,
    }
    // Public f32/f64 exports from the published crate, independently of our shard.
    let upstream: PublicExports =
        serde_json::from_str(include_str!("libm-0.2.16-exports.json")).unwrap();
    assert_eq!(upstream.name, "libm");
    assert_eq!(upstream.version, "0.2.16");
    assert_eq!(
        upstream.crate_sha256,
        "b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981"
    );
    assert_eq!(upstream.exports.len(), 124);
    let rows = mappings(&read(&root()).unwrap()).unwrap();
    let mapped: BTreeSet<_> = rows.iter().filter_map(|row| row.libm.clone()).collect();
    assert!(mapped.is_subset(&upstream.exports));
    let missing: BTreeSet<_> = upstream
        .exports
        .difference(&mapped)
        .map(String::as_str)
        .collect();
    assert_eq!(missing, BTreeSet::from(["jn", "jnf", "yn", "ynf"]));
    for name in upstream.exports {
        let path = format!("libm::{name}");
        let recognized = compiled::RustFloatMathIntrinsic::from_libm_path(&path).is_some()
            || compiled::libm_sincos_is_f64(&path).is_some();
        assert_eq!(recognized, mapped.contains(&name), "{path}");
    }
}

/// Check that the 22 plain `libm` aliases preserve width and namespace isolation.
#[test]
fn new_plain_aliases_preserve_width_and_namespace_controls() {
    use compiled::MathScalarType::{F32, F64};
    for (name, arity) in [
        ("erf", 1),
        ("erfc", 1),
        ("exp10", 1),
        ("fdim", 2),
        ("fmod", 2),
        ("j0", 1),
        ("lgamma", 1),
        ("nextafter", 2),
        ("remainder", 2),
        ("y0", 1),
        ("y1", 1),
    ] {
        for wide in [false, true] {
            let alias = format!("{name}{}", if wide { "" } else { "f" });
            let float = if wide { F64 } else { F32 };
            let intrinsic =
                compiled::RustFloatMathIntrinsic::from_libm_path(&format!("libm::{alias}"))
                    .unwrap();
            let symbol = format!("__nv_{alias}");
            assert_eq!(intrinsic.libdevice_symbol(wide), Some(symbol.as_str()));
            assert_eq!(intrinsic.llvm_symbol(wide), None);
            assert_eq!(intrinsic.arg_count(), arity);
            let arguments = vec![float; arity];
            assert_eq!(
                intrinsic.scalar_signature(wide),
                Some((arguments.as_slice(), float))
            );
            assert!(compiled::is_libdevice_backed_placeholder(
                intrinsic.placeholder_callee()
            ));
            assert!(!compiled::is_backend_dependent_libdevice_placeholder(
                intrinsic.placeholder_callee()
            ));
            assert_eq!(
                compiled::RustFloatMathIntrinsic::from_core_path(&format!(
                    "libm::math::module::{alias}"
                )),
                Some(intrinsic)
            );
            for root in ["user::libm", "libm_lookalike", "libmath"] {
                assert_eq!(
                    compiled::RustFloatMathIntrinsic::from_core_path(&format!("{root}::{alias}")),
                    None
                );
            }
            assert_eq!(
                compiled::RustFloatMathIntrinsic::from_core_path(&format!("libm::{name}f128")),
                None
            );
            assert!(!compiled::is_intrinsic_lowered_cmath_shim(&format!(
                "std::sys::cmath::{alias}"
            )));
        }
    }
    // These names compile in libm, but direct libdevice substitution changes
    // their behavior for negative orders or signed zero at infinity.
    for alias in ["jn", "jnf", "yn", "ynf"] {
        assert_eq!(
            compiled::RustFloatMathIntrinsic::from_libm_path(&format!("libm::{alias}")),
            None,
            "{alias} requires a semantic adapter"
        );
    }
}
