/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../intrinsics/generated/libdevice_math.rs"
    ));
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(functions: &str) -> Shard {
    parse(format!("schema = 1\nfamily = \"libdevice_math\"\n[libdevice_math]\nevidence = \"evidence/test.json\"\nlibdevice_sha256 = \"{}\"\nwidths = [\"f32\", \"f64\"]\n{functions}", "a".repeat(64)).as_bytes()).unwrap()
}

#[test]
fn staged_snapshot_has_exactly_the_original_72_exports() {
    let shard = read(&root()).unwrap();
    let rows = mappings(&shard).unwrap();
    assert_eq!(rows.len(), 72);
    assert_eq!(expected_symbols(&rows).unwrap().len(), 64);
    assert_eq!(generated::MAPPINGS.len(), rows.len());
    for (actual, row) in generated::MAPPINGS.iter().zip(&rows) {
        assert_eq!(actual.libm, row.libm);
        assert_eq!(actual.signature, row.signature);
        assert_eq!(
            actual.placeholders,
            row.placeholders
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(actual.cmath, row.cmath.as_deref());
        assert_eq!(actual.libdevice, row.libdevice.as_deref());
        assert_eq!(actual.llvm, row.llvm.as_deref());
        assert_eq!(actual.lowering, row.lowering.as_deref());
        assert_eq!(actual.link, row.link.as_str());
    }
}

fn quoted(text: &str) -> impl Iterator<Item = &str> {
    text.split('"').skip(1).step_by(2)
}

fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let rest = &source[start..];
    let end = [rest.find("\n    fn "), rest.find("\n}\n")]
        .into_iter()
        .flatten()
        .min()
        .unwrap();
    &rest[..end]
}

fn variants(source: &str, name: &str, target: &str) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for line in function_body(source, name).lines() {
        let Some((left, right)) = line.split_once(target) else {
            continue;
        };
        for variant in left.split('|') {
            if let Some(variant) = variant.trim().strip_prefix("Self::") {
                result.insert(
                    variant.into(),
                    right.split([')', ',']).next().unwrap().trim().into(),
                );
            }
        }
    }
    result
}

/// Compare staged rows against the live handwritten routes, not another copy
/// of an expected mapping list. Additional hand-landed functions may coexist
/// until the compiler-table migration, without making this PR depend on them.
#[test]
fn staged_rows_preserve_existing_import_placeholder_symbol_and_link_routes() {
    let importer =
        include_str!("../../../mir-importer/src/translator/terminator/intrinsics/float_math.rs");
    let dialect = include_str!("../../../dialect-mir/src/rust_intrinsics.rs");
    let lower = include_str!("../../../mir-lower/src/convert/ops/call.rs");
    let start = importer.find("fn from_libm_path(").unwrap();
    let end = start
        + importer[start..]
            .find("fn from_fast_intrinsic_path(")
            .unwrap();
    let mut imports = BTreeMap::new();
    for line in importer[start..end].lines() {
        if let Some((left, right)) = line.split_once("=> Some(Self::") {
            for name in quoted(left) {
                imports.insert(name, right.split(')').next().unwrap());
            }
        }
    }
    let placeholders = variants(importer, "placeholder_callee", "=> rust_intrinsics::");
    let constants: BTreeMap<_, _> = dialect
        .lines()
        .filter_map(|line| {
            let line = line.strip_prefix("pub const ")?;
            let (name, _) = line.split_once(':')?;
            Some((name, quoted(line).next()?))
        })
        .collect();
    let symbols = variants(lower, "libdevice_name", "=> Ok(");
    let native = variants(lower, "llvm_intrinsic_name", "=> Ok(Some(");
    let always = function_body(dialect, "is_libdevice_backed_placeholder");
    let conditional = function_body(dialect, "is_backend_dependent_libdevice_placeholder");
    let contains_constant = |body: &str, name: &str| {
        body.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
            .any(|token| token == name)
    };
    for row in mappings(&read(&root()).unwrap()).unwrap() {
        if row.libm == "sincos" || row.libm == "sincosf" {
            assert!(importer.contains(&format!("\"{}\" => Some(", row.libm)));
            assert_eq!(row.placeholders.len(), 2);
            assert_eq!(row.lowering.as_deref(), Some("handwritten:emit_sincos"));
            continue;
        }
        let variant = imports[row.libm.as_str()];
        let constant = &placeholders[variant];
        assert_eq!(
            row.placeholders,
            [constants[constant.as_str()]],
            "{}",
            row.libm
        );
        assert_eq!(
            contains_constant(always, constant),
            row.link == Link::Always,
            "{}",
            row.libm
        );
        assert_eq!(
            contains_constant(conditional, constant),
            row.link == Link::Libnvvm,
            "{}",
            row.libm
        );
        if variant == "Fabs" {
            for symbol in [&row.libdevice, &row.llvm].into_iter().flatten() {
                assert!(lower.contains(&format!("\"{symbol}\"")));
            }
        } else {
            assert_eq!(
                symbols.get(variant).map(|name| name.trim_matches('"')),
                row.libdevice.as_deref(),
                "{}",
                row.libm
            );
            assert_eq!(
                native.get(variant).map(|name| name.trim_matches('"')),
                row.llvm.as_deref(),
                "{}",
                row.libm
            );
        }
        if let Some(path) = &row.cmath {
            assert!(importer.contains(&format!("\"{path}\" => Some(Self::{variant})")));
        }
    }
}

#[test]
fn aliases_and_handwritten_controls_keep_their_semantics() {
    let rows = mappings(&read(&root()).unwrap()).unwrap();
    let find = |name: &str| rows.iter().find(|row| row.libm == name).unwrap();
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
    assert_eq!(find("fabs").placeholders, find("fabsf").placeholders);
    assert_eq!(find("floor").link, Link::Libnvvm);
    assert_eq!(find("sqrt").link, Link::Always);
}

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
}

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
            "libdevice ABI",
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

#[test]
fn schema_and_evidence_pins_are_strict() {
    let source = fs::read_to_string(root().join("intrinsics").join(SHARD)).unwrap();
    for changed in [
        source.replacen("schema = 1", "schema = 2", 1),
        source.replacen("family = \"libdevice_math\"", "family = \"scalar_math\"", 1),
        source.replacen("[\"f32\", \"f64\"]", "[\"f16\"]", 1),
        source.replacen(
            "evidence/libdevice_math/cuda-13.0.json",
            "evidence/../outside.json",
            1,
        ),
        source.replacen("schema = 1", "schema = 1\nunknown = true", 1),
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
