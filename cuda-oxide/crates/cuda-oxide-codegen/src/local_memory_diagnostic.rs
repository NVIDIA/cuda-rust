/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Post-optimization diagnostics for Rust locals that remain stack-backed.
//!
//! Rust MIR locals are tagged before lowering and exported with an encoded SSA
//! name. The ordinary LLVM `opt -O2` pipeline removes locals that mem2reg/SROA
//! can promote. This module inspects the exact LLVM IR subsequently passed to
//! `llc` and reports only tagged allocas that still exist there.
//!
//! The analysis is intentionally conservative. Dynamic GEP indices and obvious
//! address escapes get specific reasons; every other surviving tagged alloca
//! receives a generic "survived scalar replacement" explanation rather than a
//! guessed cause.

use std::collections::BTreeSet;
use std::path::Path;

const LOCAL_MEMORY_VALUE_PREFIX: &str = "%__cuda_oxide_local_x";
const LOCAL_MEMORY_ALLOCA_PREFIX: &str = "%__cuda_oxide_local_alloca_x";

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalProvenance {
    local_index: usize,
    size_bytes: u64,
    name: String,
    type_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromotionBlocker {
    DynamicIndex,
    AddressEscapes,
    SurvivedScalarReplacement,
}

#[derive(Clone, Debug)]
struct FunctionBody<'a> {
    name: String,
    lines: Vec<&'a str>,
}

#[derive(Clone, Debug)]
struct SurvivingAlloca {
    value: String,
    provenance: LocalProvenance,
}

/// Inspect optimized LLVM IR and return user-facing warning diagnostics.
pub(crate) fn diagnose_file(path: &Path) -> std::io::Result<Vec<String>> {
    let llvm = std::fs::read_to_string(path)?;
    Ok(diagnose_text(&llvm))
}

fn diagnose_text(llvm: &str) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for function in function_bodies(llvm) {
        let function_name = demangle_function_name(&function.name);
        for alloca in surviving_allocas(&function) {
            let blocker = classify_promotion_blocker(&function.lines, &alloca.value);
            diagnostics.push(format_diagnostic(
                &function_name,
                &alloca.provenance,
                blocker,
            ));
        }
    }
    diagnostics
}

/// Render the containing function readably in the warning.
///
/// Kernel entry points already carry their exported source name, but a
/// non-inlined device function survives optimization under its mangled Rust
/// symbol. Demangle those (dropping the disambiguating hash) so the warning
/// points at the source function; anything unmangled passes through untouched.
fn demangle_function_name(name: &str) -> String {
    match rustc_demangle::try_demangle(name) {
        Ok(demangled) => format!("{demangled:#}"),
        Err(_) => name.to_string(),
    }
}

fn function_bodies(llvm: &str) -> Vec<FunctionBody<'_>> {
    let mut functions = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_lines = Vec::new();

    for line in llvm.lines() {
        if current_name.is_none() {
            if let Some(name) = defined_function_name(line) {
                current_name = Some(name);
                current_lines.push(line);
            }
            continue;
        }

        current_lines.push(line);
        if line.trim() == "}" {
            functions.push(FunctionBody {
                name: current_name.take().unwrap(),
                lines: std::mem::take(&mut current_lines),
            });
        }
    }

    functions
}

fn defined_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("define ") {
        return None;
    }
    let at = trimmed.find('@')? + 1;
    let rest = &trimmed[at..];
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some(quoted[..end].to_string());
    }
    let end = rest.find('(')?;
    Some(rest[..end].trim().to_string())
}

fn surviving_allocas(function: &FunctionBody<'_>) -> Vec<SurvivingAlloca> {
    function
        .lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if !trimmed.contains(" = alloca ") {
                return None;
            }
            let value = trimmed.split_once(" = ")?.0.trim();
            let provenance = decode_provenance_name(value)?;
            Some(SurvivingAlloca {
                value: value.to_string(),
                provenance,
            })
        })
        .collect()
}

fn decode_provenance_name(value: &str) -> Option<LocalProvenance> {
    let encoded = value
        .strip_prefix(LOCAL_MEMORY_VALUE_PREFIX)
        .or_else(|| value.strip_prefix(LOCAL_MEMORY_ALLOCA_PREFIX))?;
    // The exporter terminates the hex payload with a `_` sentinel. LLVM
    // uniques colliding value names by appending bare digits (a second inlined
    // copy of the same local becomes `<name>1`), and digits are valid hex, so
    // without the sentinel a uniquing suffix would silently extend the payload
    // and garble the report. Key the decode on the sentinel: hex digits up to
    // the first `_`, and reject names that lack it.
    let hex_len = encoded
        .bytes()
        .take_while(|byte| byte.is_ascii_hexdigit())
        .count();
    if encoded.as_bytes().get(hex_len) != Some(&b'_') {
        return None;
    }
    let encoded = &encoded[..hex_len];
    if encoded.is_empty() || encoded.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for index in (0..encoded.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&encoded[index..index + 2], 16).ok()?);
    }
    let decoded = String::from_utf8(bytes).ok()?;
    let mut fields = decoded.splitn(4, '\t');
    Some(LocalProvenance {
        local_index: fields.next()?.parse().ok()?,
        size_bytes: fields.next()?.parse().ok()?,
        name: fields.next()?.to_string(),
        type_name: fields.next()?.to_string(),
    })
}

fn classify_promotion_blocker(lines: &[&str], root: &str) -> PromotionBlocker {
    let pointer_values = derived_pointer_values(lines, root);

    if lines
        .iter()
        .any(|line| gep_has_dynamic_index(line, &pointer_values))
    {
        return PromotionBlocker::DynamicIndex;
    }

    if lines
        .iter()
        .any(|line| address_escapes(line, &pointer_values))
    {
        return PromotionBlocker::AddressEscapes;
    }

    PromotionBlocker::SurvivedScalarReplacement
}

fn derived_pointer_values(lines: &[&str], root: &str) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    values.insert(root.to_string());

    loop {
        let mut changed = false;
        for line in lines {
            let Some((lhs, rhs)) = assignment(line) else {
                continue;
            };
            if !is_pointer_identity_or_derivation(rhs) {
                continue;
            }
            if values.iter().any(|value| contains_value_token(rhs, value))
                && values.insert(lhs.to_string())
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    values
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let (lhs, rhs) = trimmed.split_once(" = ")?;
    lhs.starts_with('%').then_some((lhs.trim(), rhs.trim()))
}

fn is_pointer_identity_or_derivation(rhs: &str) -> bool {
    rhs.starts_with("getelementptr ")
        || rhs.starts_with("bitcast ")
        || rhs.starts_with("addrspacecast ")
        || rhs.starts_with("freeze ")
        || rhs.starts_with("phi ")
        || rhs.starts_with("select ")
}

fn gep_has_dynamic_index(line: &str, pointers: &BTreeSet<String>) -> bool {
    let Some((_, rhs)) = assignment(line) else {
        return false;
    };
    if !rhs.starts_with("getelementptr ") {
        return false;
    }

    let Some(root) = pointers
        .iter()
        .find(|value| contains_value_token(rhs, value))
    else {
        return false;
    };

    let fields: Vec<&str> = rhs.split(',').collect();
    let Some(base_field) = fields
        .iter()
        .position(|field| contains_value_token(field, root))
    else {
        return false;
    };

    fields[base_field + 1..]
        .iter()
        .any(|field| value_tokens(field).next().is_some())
}

fn address_escapes(line: &str, pointers: &BTreeSet<String>) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with(';') || trimmed.contains(" = getelementptr ") {
        return false;
    }

    if (trimmed.starts_with("call ") || trimmed.contains(" = call "))
        && !trimmed.contains("@llvm.")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if trimmed.starts_with("ret ")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if trimmed.contains(" = ptrtoint ")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if let Some(store) = trimmed.strip_prefix("store ")
        && let Some(stored_value) = store.split(',').next()
        && pointers
            .iter()
            .any(|value| contains_value_token(stored_value, value))
    {
        return true;
    }

    false
}

fn contains_value_token(text: &str, value: &str) -> bool {
    let mut search_from = 0usize;
    while let Some(relative) = text[search_from..].find(value) {
        let start = search_from + relative;
        let end = start + value.len();
        let before_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|ch| !llvm_name_char(ch));
        let after_ok = text[end..]
            .chars()
            .next()
            .is_none_or(|ch| !llvm_name_char(ch));
        if before_ok && after_ok {
            return true;
        }
        search_from = start + 1;
    }
    false
}

fn llvm_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-')
}

fn value_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '(' | ')' | '[' | ']'))
        .filter(|token| token.starts_with('%'))
}

fn format_diagnostic(
    function: &str,
    provenance: &LocalProvenance,
    blocker: PromotionBlocker,
) -> String {
    let binding = if provenance.name.is_empty() {
        format!("_{}", provenance.local_index)
    } else {
        provenance.name.clone()
    };
    let bytes = if provenance.size_bytes == 1 {
        "1 byte".to_string()
    } else {
        format!("{} bytes", provenance.size_bytes)
    };
    let mut message = format!(
        "warning: local `{binding}` (`{}`, {bytes}) in `{function}` could not be promoted to registers",
        provenance.type_name
    );

    match blocker {
        PromotionBlocker::DynamicIndex => {
            message.push_str(
                "\n  = note: indexed by a non-constant value; the allocation survives LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: use constant indexing or a small match/select when the index range is bounded",
            );
        }
        PromotionBlocker::AddressEscapes => {
            message.push_str(
                "\n  = note: the allocation's address escapes as a first-class pointer, preventing scalar replacement",
            );
            message.push_str(
                "\n  = help: keep accesses local to the value when possible so LLVM can promote the storage",
            );
        }
        PromotionBlocker::SurvivedScalarReplacement => {
            message.push_str(
                "\n  = note: the allocation survives mem2reg and LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: inspect dynamic indexing, pointer escapes, volatile accesses, or aliasing around this local; large aggregates are also harder to promote because scalar replacement must split every access into independent scalars",
            );
        }
    }

    message
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(provenance: &str) -> String {
        let mut output = String::new();
        for byte in provenance.as_bytes() {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output.push('_');
        output
    }

    fn alloca_name(provenance: &str) -> String {
        format!("{LOCAL_MEMORY_VALUE_PREFIX}{}", encoded(provenance))
    }

    #[test]
    fn dynamic_index_reports_the_rust_local() {
        let name = alloca_name("3\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %idx) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  %value = load i32, ptr %elt, align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `scratch` (`[u32; 4]`, 16 bytes)"));
        assert!(diagnostics[0].contains("indexed by a non-constant value"));
    }

    #[test]
    fn optimizer_suffixes_do_not_hide_surviving_allocas() {
        let provenance = "8\t64\tvalues\t[u64; 8]";
        let name = format!("{}.sroa.0", alloca_name(provenance));
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [8 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `values`"));
    }

    #[test]
    fn name_uniquing_suffixes_do_not_garble_the_report() {
        // LLVM uniques colliding value names by appending bare digits: the
        // same local inlined twice into one kernel yields `<name>` and
        // `<name>1`. The `_` sentinel ends the hex payload, so the digit must
        // not leak into the decoded fields.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let base = alloca_name(provenance);
        let uniqued = format!("{base}1");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {base} = alloca [4 x i32], align 4\n  {uniqued} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 2);
        for diagnostic in &diagnostics {
            assert!(
                diagnostic.contains("local `scratch` (`[u32; 4]`, 16 bytes)"),
                "uniquing digit corrupted the decoded provenance: {diagnostic}"
            );
        }
    }

    #[test]
    fn capped_payloads_still_attribute_the_local() {
        // The exporter caps the payload length before hex-encoding; a
        // truncated trailing type spelling must not lose the report.
        let full = format!("7\t4096\tstate\t{}", "VeryLongTypeName".repeat(8));
        let truncated = &full[..40];
        let name = alloca_name(truncated);
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [512 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `state`"));
        assert!(diagnostics[0].contains("4096 bytes"));
    }

    #[test]
    fn payload_without_the_sentinel_is_not_trusted() {
        // Without the terminating `_` a trailing uniquing digit is
        // indistinguishable from payload, so such names must be ignored
        // rather than half-decoded.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let name = alloca_name(provenance);
        let unterminated = name.trim_end_matches('_');
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {unterminated} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm).is_empty());
    }

    #[test]
    fn address_escape_gets_a_specific_reason() {
        let name = alloca_name("4\t32\tstate\tState");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca %State, align 8\n  call void @consume(ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn llvm_lifetime_intrinsics_are_not_reported_as_address_escapes() {
        let name = alloca_name("5\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  call void @llvm.lifetime.start.p0(i64 16, ptr {name})\n  call void @llvm.lifetime.end.p0(i64 16, ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("survives mem2reg and LLVM scalar replacement"));
        assert!(!diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn mangled_containing_functions_are_demangled() {
        let name = alloca_name("2\t8\tacc\t[f32; 2]");
        let llvm = format!(
            "define void @_ZN6kernel5inner17h0123456789abcdefE() {{\nentry:\n  {name} = alloca [2 x float], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm);
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("in `kernel::inner`"),
            "expected demangled function name without hash: {}",
            diagnostics[0]
        );
    }

    #[test]
    fn untagged_codegen_allocas_are_ignored() {
        let llvm = "define void @kernel(i64 %idx) {\nentry:\n  %tmp = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr %tmp, i64 0, i64 %idx\n  ret void\n}\n";
        assert!(diagnose_text(llvm).is_empty());
    }

    #[test]
    fn promoted_local_is_silent_when_the_alloca_is_absent() {
        let llvm = "define i32 @kernel(i32 %value) {\nentry:\n  ret i32 %value\n}\n";
        assert!(diagnose_text(llvm).is_empty());
    }
}
