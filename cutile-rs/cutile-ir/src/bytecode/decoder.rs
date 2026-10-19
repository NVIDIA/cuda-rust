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

use super::enums::{
    BytecodeVersion, FunctionFlag, Section, TypeTag, ALIGNMENT_BYTE, MAGIC, NUM_SECTIONS,
};
use crate::ir::DYNAMIC;
use crate::{Error, Result};
use std::fmt::Write;

// =========================================================================
// Public API
// =========================================================================

/// Decode a bytecode buffer into a human-readable string.
pub fn decode_bytecode(data: &[u8]) -> Result<String> {
    let mut r = Reader::new(data);
    let mut out = String::new();

    // Header
    let version = r.read_header()?;
    writeln!(out, "TileIR bytecode v{version}").unwrap();
    writeln!(out).unwrap();

    // Collect raw sections first, then parse in dependency order.
    let mut sections = RawSections::default();
    loop {
        let sh = r.read_section_header()?;
        if sh.id == Section::EndOfBytecode as u8 {
            break;
        }
        let payload = r.read_bytes(sh.data_len)?;
        sections.insert(sh.id, payload);
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
        let mut producer = Reader::new(payload);
        let index = producer.read_varint()? as usize;
        let name = strings
            .get(index)
            .ok_or_else(|| Error::BytecodeWrite("producer string index out of range".into()))?;
        writeln!(out, "=== Producer ===\n  {name:?}\n").unwrap();
    }

    let types = parse_type_section(sections.get(Section::Type as u8), version)?;
    if !types.is_empty() {
        writeln!(out, "=== Types ({}) ===", types.len()).unwrap();
        for (i, t) in types.iter().enumerate() {
            writeln!(out, "  [{i}] {t}").unwrap();
        }
        writeln!(out).unwrap();
    }

    // Constant section.
    let constants = parse_constant_section(sections.get(Section::Constant as u8))?;
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
        let funcs = parse_func_section(payload, &strings, &types)?;
        writeln!(out, "=== Functions ({}) ===", funcs.len()).unwrap();
        for f in &funcs {
            writeln!(out, "{f}").unwrap();
        }
    }

    // Debug section.
    if let Some(payload) = sections.get(Section::Debug as u8) {
        parse_debug_section(payload, &strings, &mut out)?;
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
// Raw section collector
// =========================================================================

#[derive(Default)]
struct RawSections<'a> {
    data: [Option<&'a [u8]>; NUM_SECTIONS as usize],
}

impl<'a> RawSections<'a> {
    fn insert(&mut self, id: u8, payload: &'a [u8]) {
        if (id as usize) < self.data.len() {
            self.data[id as usize] = Some(payload);
        }
    }
    fn get(&self, id: u8) -> Option<&'a [u8]> {
        self.data.get(id as usize).copied().flatten()
    }
}

// =========================================================================
// Section header
// =========================================================================

struct SectionHeader {
    id: u8,
    data_len: usize,
}

// =========================================================================
// Low-level reader
// =========================================================================

/// Low-level bytecode reader (mirror of EncodingWriter).
#[allow(dead_code)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(err("unexpected end of data"));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(err("unexpected end of data"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.read_byte()?;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(err("varint overflow"));
            }
        }
        Ok(result)
    }

    fn read_signed_varint(&mut self) -> Result<i64> {
        let v = self.read_varint()?;
        // zigzag decode
        Ok(((v >> 1) as i64) ^ (-((v & 1) as i64)))
    }

    fn read_le_u8(&mut self) -> Result<u8> {
        self.read_byte()
    }

    fn read_le_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_le_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_le_i64(&mut self) -> Result<i64> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_le_i32(&mut self) -> Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn skip_padding(&mut self, alignment: u64) -> Result<()> {
        if alignment < 2 {
            return Ok(());
        }
        let padding = (alignment - (self.pos as u64 % alignment)) % alignment;
        for _ in 0..padding {
            let b = self.read_byte()?;
            if b != ALIGNMENT_BYTE {
                return Err(err(&format!(
                    "expected padding byte 0x{ALIGNMENT_BYTE:02X}, got 0x{b:02X}"
                )));
            }
        }
        Ok(())
    }

    fn read_header(&mut self) -> Result<BytecodeVersion> {
        let magic = self.read_bytes(8)?;
        if magic != MAGIC {
            return Err(err("invalid magic number"));
        }
        let major = self.read_le_u8()?;
        let minor = self.read_le_u8()?;
        let tag = self.read_le_u16()?;
        Ok(BytecodeVersion { major, minor, tag })
    }

    fn read_section_header(&mut self) -> Result<SectionHeader> {
        let id_and_align = self.read_byte()?;
        let id = id_and_align & 0x7F;
        let has_alignment = id_and_align & 0x80 != 0;

        if id == Section::EndOfBytecode as u8 {
            return Ok(SectionHeader { id, data_len: 0 });
        }

        let length = self.read_varint()? as usize;
        if has_alignment {
            let alignment = self.read_varint()?;
            self.skip_padding(alignment)?;
        }

        Ok(SectionHeader {
            id,
            data_len: length,
        })
    }
}

// =========================================================================
// String section parser
// =========================================================================

/// Parses and pretty-prints the Debug section: per-function attribute-id
/// lists plus the interned attribute table (tag byte + varint fields; see
/// the writer's `write_debug_section` for the layout).
fn parse_debug_section(payload: &[u8], strings: &[String], out: &mut String) -> Result<()> {
    use super::enums::DebugTag;

    let mut r = Reader::new(payload);
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
        let mut a = Reader::new(&attr_data[start..end]);
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
                writeln!(
                    out,
                    "DISubprogram(file=di[{file}], line={line}, name={:?}, linkage={:?}, cu=di[{cu}], scope_line={scope_line})",
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

fn parse_string_section(payload: Option<&[u8]>) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;

    // Read offset table.
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(r.read_le_u32()? as usize);
    }

    // Remaining bytes are concatenated string data.
    let string_data = r.read_bytes(r.remaining())?;

    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            string_data.len()
        };
        let s = std::str::from_utf8(&string_data[start..end])
            .unwrap_or("<invalid utf8>")
            .to_owned();
        strings.push(s);
    }
    Ok(strings)
}

// =========================================================================
// Type section parser
// =========================================================================

fn parse_type_section(payload: Option<&[u8]>, version: BytecodeVersion) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;

    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(r.read_le_u32()? as usize);
    }

    let type_data = r.read_bytes(r.remaining())?;
    let mut types = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            type_data.len()
        };
        let desc = decode_type_entry(&type_data[start..end], &types, version);
        types.push(desc);
    }
    Ok(types)
}

fn decode_type_entry(data: &[u8], prev_types: &[String], version: BytecodeVersion) -> String {
    let mut r = Reader::new(data);
    let Ok(tag_val) = r.read_varint() else {
        return "<read error>".into();
    };
    let tag = tag_val as u8;
    match tag {
        t if t == TypeTag::I1 as u8 => "i1".into(),
        t if t == TypeTag::I4 as u8 => "i4".into(),
        t if t == TypeTag::I8 as u8 => "i8".into(),
        t if t == TypeTag::I16 as u8 => "i16".into(),
        t if t == TypeTag::I32 as u8 => "i32".into(),
        t if t == TypeTag::I64 as u8 => "i64".into(),
        t if t == TypeTag::F16 as u8 => "f16".into(),
        t if t == TypeTag::BF16 as u8 => "bf16".into(),
        t if t == TypeTag::F32 as u8 => "f32".into(),
        t if t == TypeTag::TF32 as u8 => "tf32".into(),
        t if t == TypeTag::F64 as u8 => "f64".into(),
        t if t == TypeTag::F8E4M3FN as u8 => "f8e4m3fn".into(),
        t if t == TypeTag::F8E5M2 as u8 => "f8e5m2".into(),
        t if t == TypeTag::F8E8M0FNU as u8 => "f8e8m0fnu".into(),
        t if t == TypeTag::F4E2M1FN as u8 => "f4e2m1fn".into(),
        t if t == TypeTag::F8E5M3FNU as u8 => "f8e5m3fnu".into(),
        t if t == TypeTag::Token as u8 => "token".into(),
        t if t == TypeTag::Pointer as u8 => {
            let flags = if version >= BytecodeVersion::V13_4 {
                r.read_varint().unwrap_or(0)
            } else {
                0
            };
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            if flags & 1 != 0 {
                let attr = r.read_byte().unwrap_or(0);
                format!("ptr<{elem_name}, ptr_attr={attr}>")
            } else {
                format!("ptr<{elem_name}>")
            }
        }
        t if t == TypeTag::Tile as u8 => {
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            let rank = r.read_varint().unwrap_or(0) as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(r.read_le_i64().unwrap_or(0));
            }
            let shape_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            if shape_str.is_empty() {
                format!("tile<{elem_name}>")
            } else {
                format!("tile<{shape_str}x{elem_name}>")
            }
        }
        t if t == TypeTag::TensorView as u8 => {
            let flags = if version >= BytecodeVersion::V13_4 {
                r.read_varint().unwrap_or(0)
            } else {
                0
            };
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            let rank = r.read_varint().unwrap_or(0) as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(r.read_le_i64().unwrap_or(0));
            }
            let stride_count = r.read_varint().unwrap_or(0) as usize;
            let mut strides = Vec::with_capacity(stride_count);
            for _ in 0..stride_count {
                strides.push(r.read_le_i64().unwrap_or(0));
            }
            let shape_str = shape
                .iter()
                .map(|d| {
                    if *d == DYNAMIC {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("x");
            let stride_str = strides
                .iter()
                .map(|d| {
                    if *d == DYNAMIC {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            let ptr_attr = if flags & 1 != 0 {
                format!(", ptr_attr={}", r.read_byte().unwrap_or(0))
            } else {
                String::new()
            };
            format!("tensor_view<{shape_str}x{elem_name}, strides=[{stride_str}]{ptr_attr}>")
        }
        t if t == TypeTag::PartitionView as u8 => {
            let mut padding_flag_from_bitfield = None;
            if version >= BytecodeVersion::V13_3 {
                let flags = r.read_varint().unwrap_or(0);
                padding_flag_from_bitfield = Some(flags & 1 != 0);
            }
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size);
            for _ in 0..dim_map_size {
                dim_map.push(r.read_le_i32().unwrap_or(0));
            }
            let has_padding =
                padding_flag_from_bitfield.unwrap_or_else(|| r.read_byte().unwrap_or(0) != 0);
            let padding = if has_padding {
                let p = if version >= BytecodeVersion::V13_3 {
                    r.read_byte().unwrap_or(0) as u64
                } else {
                    r.read_varint().unwrap_or(0)
                };
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!("partition_view<tile=({ts}), tv={tv_name}{padding}>")
        }
        t if t == TypeTag::GatherScatterView as u8 => {
            let flags = r.read_varint().unwrap_or(0);
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let sparse_dim = r.read_varint().unwrap_or(0);
            let padding = if flags & 1 != 0 {
                let p = r.read_byte().unwrap_or(0);
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!(
                "gather_scatter_view<tile=({ts}), tv={tv_name}, sparse_dim={sparse_dim}{padding}>"
            )
        }
        t if t == TypeTag::StridedView as u8 => {
            let flags = r.read_varint().unwrap_or(0);
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let stride_rank = r.read_varint().unwrap_or(0) as usize;
            let mut traversal_strides = Vec::with_capacity(stride_rank);
            for _ in 0..stride_rank {
                traversal_strides.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size);
            for _ in 0..dim_map_size {
                dim_map.push(r.read_le_i32().unwrap_or(0));
            }
            let padding = if flags & 1 != 0 {
                let p = r.read_byte().unwrap_or(0);
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let strides = traversal_strides
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let dim_map_str = dim_map
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "strided_view<tile=({ts}), traversal_strides=[{strides}], tv={tv_name}, dim_map=[{dim_map_str}]{padding}>"
            )
        }
        t if t == TypeTag::Func as u8 => {
            let num_inputs = r.read_varint().unwrap_or(0) as usize;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                let idx = r.read_varint().unwrap_or(0) as usize;
                inputs.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            let num_results = r.read_varint().unwrap_or(0) as usize;
            let mut results = Vec::with_capacity(num_results);
            for _ in 0..num_results {
                let idx = r.read_varint().unwrap_or(0) as usize;
                results.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            format!("({}) -> ({})", inputs.join(", "), results.join(", "))
        }
        _ => format!("<unknown type tag {tag}>"),
    }
}

// =========================================================================
// Constant section parser
// =========================================================================

fn parse_constant_section(payload: Option<&[u8]>) -> Result<Vec<Vec<u8>>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(8)?;

    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        // Offsets are u64 LE (raw array), not varints.
        let bytes = r.read_bytes(8)?;
        let v = u64::from_le_bytes(bytes.try_into().unwrap());
        offsets.push(v as usize);
    }

    let const_data = r.read_bytes(r.remaining())?;
    let mut constants = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            const_data.len()
        };
        constants.push(const_data[start..end].to_vec());
    }
    Ok(constants)
}

// =========================================================================
// Global section parser
// =========================================================================

fn parse_global_section(
    data: &[u8],
    strings: &[String],
    types: &[String],
    version: BytecodeVersion,
) -> Result<Vec<String>> {
    let mut r = Reader::new(data);
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
        let ty = types.get(type_idx).cloned().unwrap_or("?".into());
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

fn parse_func_section(data: &[u8], strings: &[String], types: &[String]) -> Result<Vec<String>> {
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    let mut funcs = Vec::with_capacity(count);

    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let sig_idx = r.read_varint()? as usize;
        let flags_byte = r.read_byte()?;
        let _loc_idx = r.read_varint()?;

        let name = strings.get(name_idx).cloned().unwrap_or("?".into());
        let sig = types.get(sig_idx).cloned().unwrap_or("?".into());

        let is_kernel = flags_byte & FunctionFlag::KindKernel as u8 != 0;
        let has_hints = flags_byte & FunctionFlag::HasOptimizationHints as u8 != 0;
        let kind = if is_kernel { "entry" } else { "func" };

        // Skip optimization hints if present (self-contained attribute).
        if has_hints {
            // Skip the self-contained attribute — we'd need full attribute
            // decoding to display it. For now just note it.
            // TODO: decode optimization hints attribute.
        }

        let body_len = r.read_varint()? as usize;
        let body_data = r.read_bytes(body_len)?;
        let op_count = count_ops_in_body(body_data);

        let mut out = String::new();
        writeln!(out, "  {kind} @{name} : {sig}").unwrap();
        writeln!(out, "    body: {body_len} bytes, ~{op_count} ops").unwrap();
        if has_hints {
            writeln!(out, "    [has optimization_hints]").unwrap();
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
