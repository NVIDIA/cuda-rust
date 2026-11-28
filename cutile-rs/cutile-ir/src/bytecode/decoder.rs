/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode decoder — reads Tile IR bytecode and produces a human-readable
//! text dump for debugging.
//!
//! This is not a full IR reconstructor (no round-trip to `Module`). It reads
//! the raw bytecode sections and prints their contents in a structured format.
//!
//! Ported from `BytecodeReader.cpp` in the `cuda-tile` submodule.

use crate::bytecode::encoding::EncodingReader;
use crate::bytecode::enums::{BytecodeVersion, FunctionFlag, Section};
use crate::bytecode::reader::parse_string_section;
use crate::ir::Type;
use crate::{Error, Result};
use std::fmt::Write;

// =========================================================================
// Public API
// =========================================================================

/// Decode a bytecode buffer into a human-readable string.
pub fn decode_bytecode(data: &[u8]) -> Result<String> {
    let mut r = EncodingReader::new(data);
    let mut out = String::new();

    // Header
    let header_version = r.read_header()?;
    let version = header_version.without_preview();
    if header_version.is_preview() {
        writeln!(out, "TileIR bytecode v{version} (preview)").unwrap();
    } else {
        writeln!(out, "TileIR bytecode v{version}").unwrap();
    }
    writeln!(out).unwrap();

    // Collect raw sections first, then parse in dependency order.
    let mut sections = super::reader::SectionTable::new();
    loop {
        let (id, len, _aligned) = r.read_section_header()?;
        if id == Section::EndOfBytecode as u8 {
            break;
        }
        let payload = r.read_bytes(len)?;
        sections.insert(id, payload);
    }

    // Parse string table first (other sections reference strings).
    let strings = parse_string_section(sections.get(Section::String as u8))?;
    if !strings.is_empty() {
        writeln!(out, "=== Strings ({}) ===", strings.len()).unwrap();
        for (i, s) in strings.iter().enumerate() {
            writeln!(out, "  [{i}] {s:?}").unwrap();
        }
        writeln!(out).unwrap();
    }

    // Parse type table.
    if let Some(payload) = sections.get(Section::Producer as u8) {
        let mut producer = EncodingReader::new(payload);
        let index = producer.read_varint()? as usize;
        let name = strings
            .get(index)
            .ok_or_else(|| Error::BytecodeWrite("producer string index out of range".into()))?;
        writeln!(out, "=== Producer ===\n  {name:?}\n").unwrap();
    }

    let types = super::reader::parse_type_section(sections.get(Section::Type as u8), version)?;
    if !types.is_empty() {
        writeln!(out, "=== Types ({}) ===", types.len()).unwrap();
        for (i, t) in types.iter().enumerate() {
            writeln!(out, "  [{i}] {}", crate::ir::fmt::format_type(t)).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Constant section.
    let constants = super::reader::parse_constant_section(sections.get(Section::Constant as u8))?;
    if !constants.is_empty() {
        writeln!(out, "=== Constants ({}) ===", constants.len()).unwrap();
        for (i, c) in constants.iter().enumerate() {
            writeln!(out, "  [{i}] {} bytes", c.len()).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Global section.
    if let Some(payload) = sections.get(Section::Global as u8) {
        let globals = parse_global_section(payload, &strings, &types, version)?;
        if !globals.is_empty() {
            writeln!(out, "=== Globals ({}) ===", globals.len()).unwrap();
            for g in &globals {
                writeln!(out, "  {g}").unwrap();
            }
            writeln!(out).unwrap();
        }
    }

    // Function section.
    if let Some(payload) = sections.get(Section::Func as u8) {
        let funcs = parse_func_section(payload, &strings, &types, &constants, version)?;
        writeln!(out, "=== Functions ({}) ===", funcs.len()).unwrap();
        for f in &funcs {
            writeln!(out, "{f}").unwrap();
        }
    }

    // Debug section.
    if let Some(payload) = sections.get(Section::Debug as u8) {
        parse_debug_section(payload, &strings, version, &mut out)?;
    }

    Ok(out)
}

/// Convenience: decode bytecode from a file.
pub fn decode_bytecode_file(path: &str) -> Result<String> {
    let data = std::fs::read(path)
        .map_err(|e| Error::BytecodeWrite(format!("failed to read {path}: {e}")))?;
    decode_bytecode(&data)
}

// =========================================================================
// Debug section parser
// =========================================================================

/// Parses and pretty-prints the Debug section: per-function attribute-id
/// lists plus the interned attribute table (tag byte + varint fields; see
/// the writer's `write_debug_section` for the layout).
fn parse_debug_section(
    payload: &[u8],
    strings: &[String],
    version: BytecodeVersion,
    out: &mut String,
) -> Result<()> {
    use super::enums::DebugTag;

    let mut r = EncodingReader::new(payload);
    // Counts come from attacker-controllable varints: cap pre-allocation by
    // what the payload could physically hold, so a huge count fails at the
    // read below instead of aborting on allocation.
    let cap = |n: usize| n.min(payload.len());
    let num_functions = r.read_varint()? as usize;
    r.skip_padding(4)?;
    let mut index_offsets = Vec::with_capacity(cap(num_functions));
    for _ in 0..num_functions {
        index_offsets.push(r.read_le_u32()? as usize);
    }
    let num_indices = r.read_varint()? as usize;
    r.skip_padding(8)?;
    let mut attr_ids = Vec::with_capacity(cap(num_indices));
    for _ in 0..num_indices {
        let bytes = r.read_bytes(8)?;
        attr_ids.push(u64::from_le_bytes(bytes.try_into().unwrap()));
    }

    let attr_count = r.read_varint()? as usize;
    r.skip_padding(4)?;
    let mut attr_offsets = Vec::with_capacity(cap(attr_count));
    for _ in 0..attr_count {
        attr_offsets.push(r.read_le_u32()? as usize);
    }
    let attr_data = r.read_bytes(r.remaining())?;

    writeln!(out, "=== Debug ({num_functions} functions) ===").unwrap();
    let s = |idx: u64| -> &str {
        strings
            .get(idx as usize)
            .map(|s| s.as_str())
            .unwrap_or("<bad string index>")
    };
    for i in 0..attr_count {
        let start = attr_offsets[i];
        let end = if i + 1 < attr_count {
            attr_offsets[i + 1]
        } else {
            attr_data.len()
        };
        if start >= end || end > attr_data.len() {
            return Err(err("debug attribute offsets out of range"));
        }
        let mut a = EncodingReader::new(&attr_data[start..end]);
        let tag = a.read_byte()?;
        write!(out, "  di[{}] = ", i + 1).unwrap();
        match tag {
            t if t == DebugTag::DIFile as u8 => {
                let name = a.read_varint()?;
                let dir = a.read_varint()?;
                writeln!(out, "DIFile(name={:?}, dir={:?})", s(name), s(dir)).unwrap();
            }
            t if t == DebugTag::DICompileUnit as u8 => {
                writeln!(out, "DICompileUnit(file=di[{}])", a.read_varint()?).unwrap();
            }
            t if t == DebugTag::DILexicalBlock as u8 => {
                let scope = a.read_varint()?;
                let file = a.read_varint()?;
                let line = a.read_varint()?;
                let col = a.read_varint()?;
                writeln!(
                    out,
                    "DILexicalBlock(scope=di[{scope}], file=di[{file}], {line}:{col})"
                )
                .unwrap();
            }
            t if t == DebugTag::DILoc as u8 => {
                let scope = a.read_varint()?;
                let file = a.read_varint()?;
                let line = a.read_varint()?;
                let col = a.read_varint()?;
                writeln!(out, "DILoc(scope=di[{scope}], {:?}:{line}:{col})", s(file)).unwrap();
            }
            t if t == DebugTag::DISubprogram as u8 => {
                let file = a.read_varint()?;
                let line = a.read_varint()?;
                let name = a.read_varint()?;
                let linkage = a.read_varint()?;
                let cu = a.read_varint()?;
                let scope_line = a.read_varint()?;
                let flags = if version >= BytecodeVersion::V13_5 {
                    a.read_varint()?
                } else {
                    0
                };
                writeln!(
                    out,
                    "DISubprogram(file=di[{file}], line={line}, name={:?}, linkage={:?}, cu=di[{cu}], scope_line={scope_line}, flags={flags})",
                    s(name),
                    s(linkage),
                )
                .unwrap();
            }
            t if t == DebugTag::CallSite as u8 => {
                let callee = a.read_varint()?;
                let caller = a.read_varint()?;
                writeln!(out, "CallSite(callee=di[{callee}], caller=di[{caller}])").unwrap();
            }
            t if t == DebugTag::Unknown as u8 => {
                writeln!(out, "Unknown").unwrap();
            }
            t => return Err(err(&format!("unknown debug attribute tag {t}"))),
        }
    }
    for (i, &start) in index_offsets.iter().enumerate() {
        let end = if i + 1 < num_functions {
            index_offsets[i + 1]
        } else {
            num_indices
        };
        if start > end || end > attr_ids.len() {
            return Err(err("debug index offsets out of range"));
        }
        let ids: Vec<String> = attr_ids[start..end]
            .iter()
            .map(|id| format!("{id}"))
            .collect();
        writeln!(out, "  fn {i}: [{}]", ids.join(", ")).unwrap();
    }
    Ok(())
}
// =========================================================================
// Global section parser
// =========================================================================

fn parse_global_section(
    data: &[u8],
    strings: &[String],
    types: &[Type],
    version: BytecodeVersion,
) -> Result<Vec<String>> {
    let mut r = EncodingReader::new(data);
    let count = r.read_varint()? as usize;
    let mut globals = Vec::with_capacity(count);
    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let type_idx = r.read_varint()? as usize;
        let const_idx = r.read_varint()? as usize;
        let alignment = r.read_varint()?;
        let mut visibility = None;
        let mut constant = None;
        if version >= BytecodeVersion::V13_3 {
            visibility = Some(match r.read_byte()? {
                0 => "public",
                1 => "private",
                _ => "unknown",
            });
            constant = Some(r.read_varint()? != 0);
        }
        let name = strings.get(name_idx).cloned().unwrap_or("?".into());
        let ty = types
            .get(type_idx)
            .map(crate::ir::fmt::format_type)
            .unwrap_or("?".into());
        let suffix = match (visibility, constant) {
            (Some(vis), Some(is_constant)) => format!(", {vis}, constant={is_constant}"),
            _ => String::new(),
        };
        globals.push(format!(
            "@{name} : {ty} = const[{const_idx}], align {alignment}{suffix}"
        ));
    }
    Ok(globals)
}

// =========================================================================
// Function section parser
// =========================================================================

fn parse_func_section(
    data: &[u8],
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
    version: BytecodeVersion,
) -> Result<Vec<String>> {
    let mut r = EncodingReader::new(data);
    let count = r.read_varint()? as usize;
    let mut funcs = Vec::with_capacity(count);

    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let sig_idx = r.read_varint()? as usize;
        let flags_byte = r.read_byte()?;
        let _loc_idx = r.read_varint()?;

        let name = strings.get(name_idx).cloned().unwrap_or("?".into());
        let sig = types
            .get(sig_idx)
            .map(crate::ir::fmt::format_type)
            .unwrap_or("?".into());

        let is_kernel = flags_byte & FunctionFlag::KindKernel as u8 != 0;
        // 13.5 drops the hints flag (a kernel always carries the attribute)
        // and adds argument attributes as a positional field.
        let has_hints = if version >= BytecodeVersion::V13_5 {
            is_kernel
        } else {
            flags_byte & FunctionFlag::HasOptimizationHints as u8 != 0
        };
        let kind = if is_kernel { "entry" } else { "func" };

        // Decode the optimization hints (self-contained attribute) if
        // present. Consuming the attribute keeps the stream in sync with
        // the function body that follows.
        let hints = if has_hints {
            Some(
                super::reader::read_self_contained_attribute(&mut r, strings, types, constants)
                    .map_err(|e| err(&format!("optimization hints: {e}")))?,
            )
        } else {
            None
        };
        if version >= BytecodeVersion::V13_5 {
            super::reader::read_self_contained_attribute(&mut r, strings, types, constants)
                .map_err(|e| err(&format!("argument attributes: {e}")))?;
        }

        let body_len = r.read_varint()? as usize;
        let body_data = r.read_bytes(body_len)?;
        let op_count = count_ops_in_body(body_data);

        let mut out = String::new();
        writeln!(out, "  {kind} @{name} : {sig}").unwrap();
        writeln!(out, "    body: {body_len} bytes, ~{op_count} ops").unwrap();
        if let Some(h) = hints {
            writeln!(
                out,
                "    optimization_hints = {}",
                crate::ir::fmt::format_attr(&h)
            )
            .unwrap();
        }
        funcs.push(out);
    }
    Ok(funcs)
}

/// Quick heuristic: count opcodes in a function body by scanning varints.
fn count_ops_in_body(data: &[u8]) -> usize {
    // This is an approximation — a proper count requires full per-op parsing.
    // For now just report the body byte size.
    // TODO: implement full per-op decoding for function bodies.
    data.len() // placeholder: return byte count, not op count
}

// =========================================================================
// Helpers
// =========================================================================

fn err(msg: &str) -> Error {
    Error::BytecodeWrite(format!("decode: {msg}"))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::encoding::EncodingWriter;
    use crate::bytecode::enums::MAGIC;

    /// Build a minimal valid bytecode (header + end marker, no sections).
    fn minimal_bytecode() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(13); // major
        buf.push(1); // minor
        buf.extend_from_slice(&0u16.to_le_bytes()); // tag
        buf.push(Section::EndOfBytecode as u8);
        buf
    }

    #[test]
    fn decode_minimal() {
        let data = minimal_bytecode();
        let out = decode_bytecode(&data).unwrap();
        assert!(out.contains("TileIR bytecode v13.1"));
    }

    #[test]
    fn decode_bad_magic() {
        let mut data = minimal_bytecode();
        data[1] = b'X'; // corrupt magic
        assert!(decode_bytecode(&data).is_err());
    }

    #[test]
    fn varint_u64_boundaries() {
        for value in [0, 127, 128, 1 << 63, u64::MAX] {
            let mut writer = EncodingWriter::new();
            writer.write_varint(value);
            let mut reader = EncodingReader::new(writer.as_bytes());
            assert_eq!(reader.read_varint().unwrap(), value);
            assert_eq!(reader.remaining(), 0);
        }
    }

    #[test]
    fn roundtrip_string_section() {
        // Build a bytecode with just a string section.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(13);
        buf.push(1);
        buf.extend_from_slice(&0u16.to_le_bytes());

        // String section: 2 strings "hello" and "world"
        let mut section = EncodingWriter::new();
        section.write_varint(2); // count
        section.align_to(4);
        let offsets_pos = section.tell();
        section.write_le_u32(0);
        section.write_le_u32(0);
        let s1 = b"hello";
        let s2 = b"world";
        // Patch offsets
        let buf_ref = section.buf_mut();
        let o1: u32 = 0;
        let o2: u32 = s1.len() as u32;
        buf_ref[offsets_pos..offsets_pos + 4].copy_from_slice(&o1.to_le_bytes());
        buf_ref[offsets_pos + 4..offsets_pos + 8].copy_from_slice(&o2.to_le_bytes());
        section.write_bytes(s1);
        section.write_bytes(s2);

        let section_bytes = section.into_bytes();
        // Write section header: String section, no alignment needed externally.
        let mut header = EncodingWriter::new();
        header.write_byte((Section::String as u8) | 0x80); // has alignment
        header.write_varint(section_bytes.len() as u64);
        header.write_varint(4); // alignment
        header.align_to(4);
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&section_bytes);

        buf.push(Section::EndOfBytecode as u8);

        let out = decode_bytecode(&buf).unwrap();
        assert!(out.contains("\"hello\""));
        assert!(out.contains("\"world\""));
    }
}
