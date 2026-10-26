/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Staged libm mappings. This family has bitcode evidence, not hardware ABI IDs.

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shard {
    schema: u32,
    family: String,
    libdevice_math: Math,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Math {
    evidence: String,
    libdevice_sha256: String,
    widths: Vec<String>,
    #[serde(rename = "function")]
    functions: Vec<Function>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Function {
    name: String,
    signature: Signature,
    #[serde(default)]
    libm: Vec<String>,
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum Signature {
    #[serde(rename = "(F) -> F")]
    Unary,
    #[serde(rename = "(F, F) -> F")]
    Binary,
    #[serde(rename = "(F, F, F) -> F")]
    Ternary,
    #[serde(rename = "(F) -> i32")]
    IntegerResult,
    #[serde(rename = "(F, i32) -> F")]
    IntegerArgument,
    #[serde(rename = "(F) -> (F, i32)")]
    FloatIntegerPair,
    #[serde(rename = "(F) -> (F, F)")]
    FloatPair,
    #[serde(rename = "(F, F) -> (F, i32)")]
    RemainderPair,
}

impl Signature {
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

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Link {
    #[default]
    Always,
    Libnvvm,
    Never,
}

impl Link {
    fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Libnvvm => "libnvvm",
            Self::Never => "never",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Mapping {
    libm: String,
    signature: String,
    placeholders: Vec<String>,
    cmath: Option<String>,
    libdevice: Option<String>,
    llvm: Option<String>,
    lowering: Option<String>,
    link: Link,
    c_abi: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema: u32,
    bitcode_sha256: String,
    llvm_dis_version: String,
    symbols: BTreeMap<String, String>,
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && (value.as_bytes()[0].is_ascii_alphabetic() || value.starts_with('_'))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn expand(template: &str, width: &str) -> Result<String> {
    let value = template
        .replace("{width}", width)
        .replace("{suffix}", if width == "f32" { "f" } else { "" });
    ensure!(
        identifier(&value),
        "invalid math name/template {template:?}"
    );
    Ok(value)
}

fn parse(bytes: &[u8]) -> Result<Shard> {
    let shard: Shard = toml::from_slice(bytes).context("parse libdevice_math shard")?;
    ensure!(shard.schema == 1, "unsupported libdevice_math schema");
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

fn read(root: &Path) -> Result<Shard> {
    let path = root.join("intrinsics").join(SHARD);
    parse(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
}

fn mappings(shard: &Shard) -> Result<Vec<Mapping>> {
    let mut names = BTreeSet::new();
    let mut exports = BTreeSet::new();
    let mut cmath_paths = BTreeSet::new();
    let mut scalar_placeholders = BTreeSet::new();
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
        } else {
            ensure!(
                !matches!(
                    function.signature,
                    Signature::FloatPair | Signature::FloatIntegerPair | Signature::RemainderPair
                ),
                "tuple-returning math requires an explicit handwritten adapter"
            );
        }
        let aliases = if function.libm.is_empty() {
            vec![function.name.clone()]
        } else {
            function.libm.clone()
        };
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
            for alias in &aliases {
                ensure!(identifier(alias), "invalid libm alias {alias:?}");
                let libm = format!("{alias}{}", if width == "f32" { "f" } else { "" });
                ensure!(
                    exports.insert(libm.clone()),
                    "duplicate libm spelling {libm}"
                );
                result.push(Mapping {
                    libm,
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
    result.sort_by(|left, right| left.libm.cmp(&right.libm));
    Ok(result)
}

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
