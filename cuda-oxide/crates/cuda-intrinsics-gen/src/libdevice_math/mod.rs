/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler math mappings. This family has bitcode evidence, not hardware ABI IDs.

mod compiler;
mod evidence;
mod render;
#[cfg(test)]
mod tests;

use crate::util::{pretty_json, read_json, write_if_changed};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Read separately from the hardware shard index so hardware catalog hashes and
/// ABI IDs remain unchanged. `generate` reconciles this input explicitly.
pub const SHARD: &str = "overlay/libdevice_math.toml";

/// Top-level schema-2 `libdevice_math` shard file (`intrinsics/overlay/libdevice_math.toml`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shard {
    schema: u32,
    family: String,
    libdevice_math: Math,
}

/// Metadata and function entries inside the `[libdevice_math]` section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Math {
    evidence: String,
    libdevice_sha256: String,
    widths: Vec<String>,
    #[serde(rename = "function")]
    functions: Vec<Function>,
}

/// One width-parameterized math function entry in `overlay/libdevice_math.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Function {
    name: String,
    signature: Signature,
    #[serde(default)]
    libm: Option<Vec<String>>,
    #[serde(default)]
    core: Vec<String>,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    variant: Option<String>,
    #[serde(default)]
    constant: Option<String>,
    #[serde(default)]
    cmath: Option<String>,
    #[serde(default)]
    placeholders: Vec<String>,
    #[serde(default)]
    libdevice: Option<String>,
    #[serde(default)]
    llvm: Option<String>,
    #[serde(default)]
    lowering: Option<String>,
    #[serde(default)]
    link: Link,
}

/// Supported Rust signature classes for width-parameterized `f32`/`f64` math entries.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum Signature {
    /// Unary floating-point signature `(F) -> F`.
    #[serde(rename = "(F) -> F")]
    Unary,
    /// Binary floating-point signature `(F, F) -> F`.
    #[serde(rename = "(F, F) -> F")]
    Binary,
    /// Ternary floating-point signature `(F, F, F) -> F`.
    #[serde(rename = "(F, F, F) -> F")]
    Ternary,
    /// Unary float-to-integer signature `(F) -> i32` (`ilogb`).
    #[serde(rename = "(F) -> i32")]
    IntegerResult,
    /// Float-and-integer binary signature `(F, i32) -> F` (`ldexp`, `scalbn`, `powi`).
    #[serde(rename = "(F, i32) -> F")]
    IntegerArgument,
    /// Float-and-integer pair return `(F) -> (F, i32)` (`frexp`, `lgamma_r`).
    #[serde(rename = "(F) -> (F, i32)")]
    FloatIntegerPair,
    /// Two-float pair return `(F) -> (F, F)` (`modf`, `sincos`).
    #[serde(rename = "(F) -> (F, F)")]
    FloatPair,
    /// Remainder-and-quotient pair return `(F, F) -> (F, i32)` (`remquo`).
    #[serde(rename = "(F, F) -> (F, i32)")]
    RemainderPair,
}

impl Signature {
    /// Expand generic `F` placeholders to the concrete Rust float type (`f32` or `f64`).
    fn rust(self, width: &str) -> String {
        let generic = match self {
            Self::Unary => "(F) -> F",
            Self::Binary => "(F, F) -> F",
            Self::Ternary => "(F, F, F) -> F",
            Self::IntegerResult => "(F) -> i32",
            Self::IntegerArgument => "(F, i32) -> F",
            Self::FloatIntegerPair => "(F) -> (F, i32)",
            Self::FloatPair => "(F) -> (F, F)",
            Self::RemainderPair => "(F, F) -> (F, i32)",
        };
        generic.replace('F', width)
    }

    /// Render the C/LLVM bitcode signature expected in `libdevice.10.bc`.
    fn c_abi(self, width: &str) -> String {
        let float = if width == "f32" { "float" } else { "double" };
        match self {
            Self::Unary => format!("{float}({float})"),
            Self::Binary => format!("{float}({float}, {float})"),
            Self::Ternary => format!("{float}({float}, {float}, {float})"),
            Self::IntegerResult => format!("i32({float})"),
            Self::IntegerArgument => format!("{float}({float}, i32)"),
            Self::FloatIntegerPair | Self::FloatPair => format!("{float}({float}, ptr)"),
            Self::RemainderPair => format!("{float}({float}, {float}, ptr)"),
        }
    }
}

/// Libdevice link requirement across the LLVM NVPTX and libNVVM backends.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Link {
    /// Links `libdevice.10.bc` on both intrinsic backends.
    #[default]
    Always,
    /// Links `libdevice.10.bc` only on the libNVVM backend (NVPTX uses an LLVM intrinsic).
    Libnvvm,
    /// Expands inline in LLVM IR without linking any libdevice symbol.
    Never,
}

impl Link {
    /// Return the snake_case string representation used in Markdown and JSON fixtures.
    fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Libnvvm => "libnvvm",
            Self::Never => "never",
        }
    }
}

/// One concrete (`f32` or `f64`) expanded math mapping row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Mapping {
    libm: Option<String>,
    width: String,
    variant: Option<String>,
    constant: Option<String>,
    paths: Vec<String>,
    signature: String,
    placeholders: Vec<String>,
    cmath: Option<String>,
    libdevice: Option<String>,
    llvm: Option<String>,
    lowering: Option<String>,
    link: Link,
    c_abi: String,
}

/// Captured `libdevice.10.bc` provenance and exported function signatures.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema: u32,
    bitcode_sha256: String,
    llvm_dis_version: String,
    symbols: BTreeMap<String, String>,
}

/// Check whether `value` is a valid non-empty ASCII Rust/C identifier.
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && (value.as_bytes()[0].is_ascii_alphabetic() || value.starts_with('_'))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// Expand `{width}`, `{Width}`, `{WIDTH}`, and `{suffix}` placeholders for `width`.
fn expand(template: &str, width: &str) -> Result<String> {
    let value = template
        .replace("{width}", width)
        .replace("{Width}", if width == "f32" { "F32" } else { "F64" })
        .replace("{WIDTH}", if width == "f32" { "F32" } else { "F64" })
        .replace("{suffix}", if width == "f32" { "f" } else { "" });
    ensure!(
        identifier(&value),
        "invalid math name/template {template:?}"
    );
    Ok(value)
}

/// Parse and validate the schema-2 `libdevice_math` TOML shard from raw bytes.
fn parse(bytes: &[u8]) -> Result<Shard> {
    let shard: Shard = toml::from_slice(bytes).context("parse libdevice_math shard")?;
    ensure!(shard.schema == 2, "unsupported libdevice_math schema");
    ensure!(
        shard.family == "libdevice_math",
        "wrong libdevice_math family"
    );
    let math = &shard.libdevice_math;
    ensure!(
        math.widths == ["f32", "f64"],
        "libdevice_math widths must be f32, f64"
    );
    ensure!(
        !math.functions.is_empty(),
        "libdevice_math has no functions"
    );
    ensure!(
        math.libdevice_sha256.len() == 64
            && math
                .libdevice_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "invalid libdevice bitcode SHA-256"
    );
    let path = Path::new(&math.evidence);
    ensure!(
        path.starts_with("evidence")
            && path.extension().and_then(|part| part.to_str()) == Some("json")
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "libdevice evidence must be a relative JSON path under evidence/"
    );
    Ok(shard)
}

/// Read and parse `intrinsics/overlay/libdevice_math.toml` from `root`.
fn read(root: &Path) -> Result<Shard> {
    let path = root.join("intrinsics").join(SHARD);
    parse(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
}

/// Expand and validate all width-specific mapping rows from the parsed shard.
fn mappings(shard: &Shard) -> Result<Vec<Mapping>> {
    let mut names = BTreeSet::new();
    let mut exports = BTreeSet::new();
    let mut cmath_paths = BTreeSet::new();
    let mut scalar_placeholders = BTreeSet::new();
    let mut compiler_paths = BTreeMap::new();
    let mut variants = BTreeMap::new();
    let mut constants = BTreeMap::new();
    let mut registered_placeholders = BTreeMap::new();
    let mut result = Vec::new();
    for function in &shard.libdevice_math.functions {
        ensure!(identifier(&function.name), "invalid math function name");
        ensure!(
            names.insert(&function.name),
            "duplicate math function {}",
            function.name
        );
        if let Some(lowering) = &function.lowering {
            ensure!(
                lowering
                    .strip_prefix("handwritten:")
                    .is_some_and(identifier),
                "invalid handwritten math lowering {lowering:?}"
            );
            ensure!(
                matches!(
                    lowering.as_str(),
                    "handwritten:emit_sincos"
                        | "handwritten:emit_frexp"
                        | "handwritten:emit_modf"
                        | "handwritten:emit_remquo"
                        | "handwritten:emit_lgamma_r"
                        | "handwritten:lower_float_minmax_nsz"
                        | "handwritten:lower_float_ieee_minmax"
                        | "handwritten:lower_bessel_j1"
                ),
                "unsupported handwritten math lowering {lowering:?}"
            );
        } else {
            ensure!(
                !matches!(
                    function.signature,
                    Signature::FloatPair | Signature::FloatIntegerPair | Signature::RemainderPair
                ),
                "tuple-returning math requires an explicit handwritten adapter"
            );
        }
        let aliases = function
            .libm
            .clone()
            .unwrap_or_else(|| vec![function.name.clone()]);
        let sincos = function.lowering.as_deref() == Some("handwritten:emit_sincos");
        if sincos {
            ensure!(
                function.signature == Signature::FloatPair
                    && function.libdevice.is_none()
                    && function.link == Link::Always,
                "sincos must expand a float pair without a direct libdevice call"
            );
            ensure!(
                !aliases.is_empty()
                    && function.variant.is_none()
                    && function.constant.is_none()
                    && function.core.is_empty()
                    && function.paths.is_empty()
                    && function.cmath.is_none(),
                "sincos must use its handwritten libm tuple route"
            );
        }
        if function.lowering.as_deref() == Some("handwritten:emit_frexp") {
            ensure!(
                function.signature == Signature::FloatIntegerPair
                    && function.link == Link::Always
                    && function.libdevice.is_some(),
                "frexp requires its float/i32 tuple ABI and libdevice symbol"
            );
        }
        if function.lowering.as_deref() == Some("handwritten:emit_lgamma_r") {
            ensure!(
                function.name == "lgamma_r"
                    && function.signature == Signature::FloatIntegerPair
                    && function.link == Link::Always
                    && function.libdevice.is_none()
                    && function.llvm.is_none()
                    && function.core.is_empty()
                    && function.paths.is_empty()
                    && function.cmath.is_none(),
                "lgamma_r requires its composed float/i32 tuple ABI without a direct symbol"
            );
        }
        for (adapter, signature) in [
            ("handwritten:emit_modf", Signature::FloatPair),
            ("handwritten:emit_remquo", Signature::RemainderPair),
        ] {
            if function.lowering.as_deref() == Some(adapter) {
                ensure!(
                    function.name == adapter.trim_start_matches("handwritten:emit_")
                        && function.signature == signature
                        && function.link == Link::Always
                        && function.libdevice.is_some(),
                    "{adapter} requires its exact tuple ABI and libdevice symbol"
                );
            }
        }
        if function.lowering.as_deref() == Some("handwritten:lower_float_minmax_nsz") {
            ensure!(
                function.signature == Signature::Binary && function.link == Link::Never,
                "min/max must use the handwritten binary never-linked route"
            );
        }
        if function.lowering.as_deref() == Some("handwritten:lower_float_ieee_minmax") {
            ensure!(
                matches!(
                    function.name.as_str(),
                    "fmaximum" | "fminimum" | "fmaximum_num" | "fminimum_num"
                ) && function.signature == Signature::Binary
                    && function.link == Link::Never
                    && function.libdevice.is_none()
                    && function.llvm.is_none()
                    && function.libm.is_none()
                    && function.core.is_empty()
                    && function.paths.is_empty()
                    && function.cmath.is_none(),
                "C23 min/max requires its exact binary never-linked libm route"
            );
        }
        if function.lowering.as_deref() == Some("handwritten:lower_bessel_j1") {
            ensure!(
                function.name == "j1"
                    && function.signature == Signature::Unary
                    && function.link == Link::Always
                    && function.libdevice.as_deref() == Some("__nv_j1{suffix}")
                    && function.llvm.is_none()
                    && function.libm.is_none()
                    && function.core.is_empty()
                    && function.paths.is_empty()
                    && function.cmath.is_none(),
                "j1 requires its exact unary libdevice ABI with infinity correction"
            );
        }
        for width in &shard.libdevice_math.widths {
            let symbol = if let Some(symbol) = &function.libdevice {
                Some(expand(symbol, width)?)
            } else if function.lowering.is_none() {
                Some(format!(
                    "__nv_{}{}",
                    function.name,
                    if width == "f32" { "f" } else { "" }
                ))
            } else {
                None
            };
            if let Some(symbol) = &symbol {
                ensure!(
                    symbol.starts_with("__nv_"),
                    "math symbol must start with __nv_"
                );
            }
            let llvm = function
                .llvm
                .as_deref()
                .map(|name| expand(name, width))
                .transpose()?;
            ensure!(
                (function.link == Link::Libnvvm) == llvm.is_some(),
                "backend-dependent math must name its LLVM intrinsic"
            );
            ensure!(
                function.link != Link::Libnvvm || symbol.is_some(),
                "backend-dependent math must name its libdevice ABI"
            );
            ensure!(
                function.link != Link::Never || (symbol.is_none() && function.lowering.is_some()),
                "never-linked math must use a handwritten lowering without a libdevice call"
            );
            let templates = if function.placeholders.is_empty() {
                vec![format!("{}{{width}}", function.name)]
            } else {
                function.placeholders.clone()
            };
            let placeholders = templates
                .iter()
                .map(|name| expand(name, width))
                .collect::<Result<Vec<_>>>()?;
            ensure!(!placeholders.is_empty(), "math requires a placeholder");
            if sincos {
                ensure!(
                    placeholders.len() == 2,
                    "sincos requires two scalar placeholders"
                );
            } else {
                ensure!(
                    placeholders.len() == 1,
                    "plain math requires one placeholder"
                );
            }
            if function.lowering.is_none() {
                ensure!(
                    placeholders.len() == 1,
                    "plain math requires one placeholder"
                );
                ensure!(
                    scalar_placeholders.insert((width, placeholders[0].clone())),
                    "duplicate scalar math placeholder {} for {width}",
                    placeholders[0]
                );
            }
            let cmath = function
                .cmath
                .as_deref()
                .map(|name| expand(name, width).map(|name| format!("std::sys::cmath::{name}")))
                .transpose()?;
            if let Some(path) = &cmath {
                ensure!(
                    cmath_paths.insert(path.clone()),
                    "duplicate cmath spelling {path}"
                );
            }
            let mut paths = Vec::new();
            for core in &function.core {
                let name = expand(core, width)?;
                paths.extend([
                    format!("core::intrinsics::{name}"),
                    format!("std::intrinsics::{name}"),
                ]);
            }
            for path in &function.paths {
                paths.push(
                    path.split("::")
                        .map(|segment| expand(segment, width))
                        .collect::<Result<Vec<_>>>()?
                        .join("::"),
                );
            }
            paths.extend(cmath.clone());
            ensure!(
                !aliases.is_empty() || !paths.is_empty(),
                "math row has no compiler entry point"
            );
            let (variant, constant) = if sincos {
                (None, None)
            } else {
                let title = format!(
                    "{}{}",
                    function.name[..1].to_ascii_uppercase(),
                    &function.name[1..]
                );
                let variant = expand(
                    function
                        .variant
                        .as_deref()
                        .unwrap_or(&format!("{title}{{Width}}")),
                    width,
                )?;
                let constant = expand(
                    function.constant.as_deref().unwrap_or(&format!(
                        "CALLEE_{}_{{WIDTH}}",
                        function.name.to_ascii_uppercase()
                    )),
                    width,
                )?;
                ensure!(
                    variant.as_bytes()[0].is_ascii_uppercase() && !variant.contains('_'),
                    "math variant must be CamelCase"
                );
                ensure!(
                    !compiler::FAST
                        .iter()
                        .any(|(name, callee, suffix)| *name == variant
                            || *callee == constant
                            || *suffix == placeholders[0]),
                    "math row collides with fast arithmetic"
                );
                ensure!(
                    constant.starts_with("CALLEE_")
                        && constant.bytes().all(|byte| byte.is_ascii_uppercase()
                            || byte.is_ascii_digit()
                            || byte == b'_'),
                    "math constant must start with CALLEE_ and use uppercase identifiers"
                );
                let role = if width == "f32" { "F32" } else { "F64" };
                ensure!(
                    (variant == format!("Frexp{role}"))
                        == (function.lowering.as_deref() == Some("handwritten:emit_frexp")),
                    "frexp identity must use the matching handwritten adapter"
                );
                ensure!(
                    (variant == format!("LgammaR{role}"))
                        == (function.lowering.as_deref() == Some("handwritten:emit_lgamma_r")),
                    "lgamma_r identity must use the matching handwritten adapter"
                );
                for adapter in ["Modf", "Remquo"] {
                    let handler = format!("handwritten:emit_{}", adapter.to_ascii_lowercase());
                    ensure!(
                        (variant == format!("{adapter}{role}"))
                            == (function.lowering.as_deref() == Some(handler.as_str())),
                        "{adapter} identity must use the matching handwritten adapter"
                    );
                }
                ensure!(
                    (variant == format!("MaxNumNsz{role}")
                        || variant == format!("MinNumNsz{role}"))
                        == (function.lowering.as_deref()
                            == Some("handwritten:lower_float_minmax_nsz")),
                    "min/max identity must use the matching handwritten expansion"
                );
                let ieee_variant = match function.name.as_str() {
                    "fmaximum" => Some(format!("Maximum{role}")),
                    "fminimum" => Some(format!("Minimum{role}")),
                    "fmaximum_num" => Some(format!("MaximumNumber{role}")),
                    "fminimum_num" => Some(format!("MinimumNumber{role}")),
                    _ => None,
                };
                ensure!(
                    ieee_variant.is_some()
                        == (function.lowering.as_deref()
                            == Some("handwritten:lower_float_ieee_minmax"))
                        && ieee_variant
                            .as_deref()
                            .is_none_or(|expected| variant == expected)
                        && (ieee_variant.is_some()
                            || ![
                                format!("Maximum{role}"),
                                format!("Minimum{role}"),
                                format!("MaximumNumber{role}"),
                                format!("MinimumNumber{role}"),
                            ]
                            .contains(&variant)),
                    "C23 min/max identity must match its libm operation"
                );
                ensure!(
                    (variant == format!("J1{role}"))
                        == (function.lowering.as_deref() == Some("handwritten:lower_bessel_j1")),
                    "j1 identity must use its infinity correction"
                );
                if variant == "Fabs" {
                    ensure!(
                        function.signature == Signature::Unary && function.link == Link::Libnvvm,
                        "generic fabs requires its unary backend-dependent route"
                    );
                }
                let identity = (
                    function.name.clone(),
                    constant.clone(),
                    placeholders[0].clone(),
                );
                if let Some(previous) = variants.insert(variant.clone(), identity.clone()) {
                    ensure!(previous == identity, "duplicate math variant {variant}");
                    ensure!(
                        variant == "Fabs",
                        "only fabs uses a generic-width math identity"
                    );
                }
                let identity = (
                    function.name.clone(),
                    variant.clone(),
                    placeholders[0].clone(),
                );
                if let Some(previous) = constants.insert(constant.clone(), identity.clone()) {
                    ensure!(previous == identity, "duplicate math constant {constant}");
                }
                if let Some(previous) =
                    registered_placeholders.insert(placeholders[0].clone(), variant.clone())
                {
                    ensure!(
                        previous == variant,
                        "duplicate compiler math placeholder {}",
                        placeholders[0]
                    );
                }
                for path in &paths {
                    ensure!(
                        !compiler::FAST.iter().any(|(_, _, suffix)| path
                            == &format!("core::intrinsics::{suffix}")
                            || path == &format!("std::intrinsics::{suffix}")),
                        "compiler math path collides with fast arithmetic"
                    );
                }
                (Some(variant), Some(constant))
            };
            for path in &paths {
                if let Some((previous_variant, previous_width)) =
                    compiler_paths.insert(path.clone(), (variant.clone(), width.clone()))
                {
                    ensure!(
                        previous_variant == variant && previous_width != *width,
                        "duplicate compiler math path {path}"
                    );
                }
            }
            let libm_names = if aliases.is_empty() {
                vec![None]
            } else {
                aliases
                    .iter()
                    .map(|alias| {
                        // libm's lgammaf_r puts the precision suffix before _r.
                        // Plain aliases keep their existing suffix-at-end spelling.
                        let libm = if alias.contains('{') {
                            expand(alias, width)?
                        } else {
                            ensure!(identifier(alias), "invalid libm alias {alias:?}");
                            format!("{alias}{}", if width == "f32" { "f" } else { "" })
                        };
                        ensure!(
                            exports.insert(libm.clone()),
                            "duplicate libm spelling {libm}"
                        );
                        Ok(Some(libm))
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            for libm in libm_names {
                result.push(Mapping {
                    libm,
                    width: width.clone(),
                    variant: variant.clone(),
                    constant: constant.clone(),
                    paths: paths.clone(),
                    signature: function.signature.rust(width),
                    placeholders: placeholders.clone(),
                    cmath: cmath.clone(),
                    libdevice: symbol.clone(),
                    llvm: llvm.clone(),
                    lowering: function.lowering.clone(),
                    link: function.link,
                    c_abi: function.signature.c_abi(width),
                });
            }
        }
    }
    result.sort_by(|left, right| {
        (&left.libm, &left.variant, &left.width).cmp(&(&right.libm, &right.variant, &right.width))
    });
    Ok(result)
}

/// Collect all unique libdevice symbols and their expected C ABI signatures.
fn expected_symbols(rows: &[Mapping]) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for row in rows {
        if let Some(symbol) = &row.libdevice
            && let Some(previous) = result.insert(symbol.clone(), row.c_abi.clone())
        {
            ensure!(
                previous == row.c_abi,
                "conflicting libdevice signatures for {symbol}"
            );
        }
    }
    Ok(result)
}

/// Validate the captured bitcode evidence JSON against the shard pin and expected symbols.
fn validate_evidence(shard: &Shard, rows: &[Mapping], evidence: &Evidence) -> Result<()> {
    ensure!(
        evidence.schema == 1,
        "unsupported libdevice evidence schema"
    );
    ensure!(
        evidence.bitcode_sha256 == shard.libdevice_math.libdevice_sha256,
        "libdevice bitcode hash does not match the shard pin"
    );
    ensure!(
        !evidence.llvm_dis_version.trim().is_empty(),
        "missing llvm-dis provenance"
    );
    for (symbol, signature) in expected_symbols(rows)? {
        ensure!(
            evidence.symbols.get(&symbol) == Some(&signature),
            "libdevice evidence missing or signature mismatch for {symbol}: expected {signature}"
        );
    }
    Ok(())
}

/// Validate the `libdevice_math` shard and render its generated repository outputs.
pub fn outputs(root: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let shard = read(root)?;
    let rows = mappings(&shard)?;
    let evidence: Evidence =
        read_json(&root.join("intrinsics").join(&shard.libdevice_math.evidence))?;
    validate_evidence(&shard, &rows, &evidence)?;
    Ok(render::outputs(&rows, &evidence.bitcode_sha256))
}

/// Capture reviewed signatures, or compare a local toolkit against the same
/// rows. Verification accepts a compatible newer toolkit without changing pins.
pub fn inspect(root: &Path, bitcode: &Path, llvm_dis: &Path, capture: bool) -> Result<()> {
    let shard = read(root)?;
    let rows = mappings(&shard)?;
    let expected = expected_symbols(&rows)?;
    let evidence = evidence::capture(bitcode, llvm_dis, &expected)?;
    if capture {
        validate_evidence(&shard, &rows, &evidence)?;
        let path = root.join("intrinsics").join(&shard.libdevice_math.evidence);
        write_if_changed(&path, &pretty_json(&evidence)?)?;
        println!(
            "captured {} libdevice signatures in {}",
            expected.len(),
            path.display()
        );
    } else {
        println!(
            "verified {} libdevice signatures; bitcode SHA-256 {}",
            expected.len(),
            evidence.bitcode_sha256
        );
    }
    Ok(())
}
