/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Disassemble libdevice bitcode and extract verified C ABI signatures.

use super::Evidence;
use crate::util::sha256_bytes;
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Disassemble libdevice bitcode with `llvm-dis` and capture expected symbol signatures.
pub(super) fn capture(
    bitcode: &Path,
    llvm_dis: &Path,
    expected: &BTreeMap<String, String>,
) -> Result<Evidence> {
    let bytes = fs::read(bitcode).with_context(|| format!("read {}", bitcode.display()))?;
    ensure!(
        bytes.starts_with(b"BC\xc0\xde"),
        "input is not LLVM bitcode"
    );
    let version = Command::new(llvm_dis)
        .arg("--version")
        .output()
        .with_context(|| format!("run {} --version", llvm_dis.display()))?;
    ensure!(version.status.success(), "llvm-dis --version failed");
    let version = String::from_utf8(version.stdout).context("llvm-dis version is not UTF-8")?;
    let mut child = Command::new(llvm_dis)
        .args(["-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("start {}", llvm_dis.display()))?;
    // Hash and disassemble the same bytes, rather than reopening a mutable path.
    child
        .stdin
        .take()
        .context("llvm-dis stdin unavailable")?
        .write_all(&bytes)?;
    let output = child.wait_with_output().context("wait for llvm-dis")?;
    ensure!(
        output.status.success(),
        "llvm-dis failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ir = String::from_utf8(output.stdout).context("llvm-dis output is not UTF-8")?;
    Ok(Evidence {
        schema: 1,
        bitcode_sha256: sha256_bytes(&bytes),
        llvm_dis_version: version.trim().to_string(),
        symbols: signatures(&ir, expected)?,
    })
}

/// Normalize an LLVM IR parameter or return type token into its canonical C ABI spelling.
fn llvm_type(text: &str) -> Result<String> {
    let mut words = text.split_whitespace();
    let first = words.next().context("missing LLVM parameter type")?;
    if first == "ptr" || first.ends_with('*') {
        if let Some(space) = words.next().filter(|word| word.starts_with("addrspace(")) {
            return Ok(format!("ptr {}", space.trim_end_matches('*')));
        }
        return Ok("ptr".into());
    }
    ensure!(
        matches!(first, "float" | "double" | "i32" | "i64" | "void"),
        "unsupported libdevice ABI type {first}"
    );
    Ok(first.into())
}

/// Parse only requested exported definitions. Unrelated intrinsics and helper
/// functions are controls; declarations alone are insufficient evidence.
pub(super) fn signatures(
    ir: &str,
    expected: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let mut found = BTreeMap::new();
    let mut header = String::new();
    for line in ir.lines() {
        let line = line.trim();
        if line.starts_with("define ") {
            ensure!(header.is_empty(), "unterminated LLVM function definition");
            header.push_str(line);
        } else if !header.is_empty() {
            header.push(' ');
            header.push_str(line);
        } else {
            continue;
        }
        if !header.contains('{') {
            continue;
        }
        let definition = std::mem::take(&mut header);
        let Some((prefix, tail)) = definition.split_once('@') else {
            continue;
        };
        let Some((name, parameters)) = tail.split_once('(') else {
            continue;
        };
        let name = name.trim().trim_matches('"');
        let Some(expected_signature) = expected.get(name) else {
            continue;
        };
        ensure!(
            !prefix
                .split_whitespace()
                .any(|word| matches!(word, "internal" | "private" | "available_externally")),
            "libdevice symbol {name} is not an exported definition"
        );
        let return_type = prefix
            .split_whitespace()
            .last()
            .context("missing LLVM return type")?;
        let return_type = llvm_type(return_type)?;
        // Address-space parentheses can occur inside the parameter list.
        let mut depth = 1;
        let mut end = None;
        for (index, byte) in parameters.bytes().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                end = Some(index);
                break;
            }
        }
        let end = end.context("unterminated LLVM parameter list")?;
        let parameters = &parameters[..end];
        let arguments = if parameters.trim().is_empty() {
            Vec::new()
        } else {
            parameters
                .split(',')
                .map(llvm_type)
                .collect::<Result<Vec<_>>>()?
        };
        let signature = format!("{return_type}({})", arguments.join(", "));
        ensure!(
            &signature == expected_signature,
            "libdevice signature mismatch for {name}: expected {expected_signature}, found {signature}"
        );
        ensure!(
            found.insert(name.into(), signature).is_none(),
            "duplicate libdevice definition {name}"
        );
    }
    for symbol in expected.keys() {
        if !found.contains_key(symbol) {
            bail!("libdevice definition is missing: {symbol}");
        }
    }
    Ok(found)
}
