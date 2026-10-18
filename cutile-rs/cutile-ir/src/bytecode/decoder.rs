/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode decoder — reads Tile IR bytecode and produces a human-readable
//! text dump for debugging.
//!
//! This is not a full IR reconstructor (no round-trip to `Module`). It reads
//! the raw bytecode sections and prints their contents in a structured format.
//! Attributes are reconstructed as [`Attribute`] values and printed with the
//! formatter, so the dump spells them the way `to_mlir_text` does.
//!
//! Ported from `BytecodeReader.cpp` in the `cuda-tile` submodule.

use super::enums::{
    AttributeTag, BytecodeVersion, FunctionFlag, Section, TypeTag, ALIGNMENT_BYTE,
    FUNCTION_FORM_EXTERN_IMPL, FUNCTION_FORM_MASK, FUNCTION_FORM_RESERVED, FUNCTION_FORM_TESTING,
    MAGIC, NUM_SECTIONS, VERSION_TAG_CONTAINS_PREVIEW,
};
use crate::ir::fmt::{format_attr, format_layout, format_sym_expr};
use crate::ir::{
    Access, Attribute, Bounded, ConstParam, DenseElements, DivBy, Layout, MemorySpace,
    OptimizationHints, SameElements, ScalarType, ScratchRequest, SymExpr, Synchronization, Type,
    DYNAMIC,
};
use crate::{Error, Result};
use std::fmt::Write;

// =========================================================================
// Public API
// =========================================================================

/// Decode a bytecode buffer into a human-readable string.
pub fn decode_bytecode(data: &[u8]) -> Result<String> {
    let mut r = Reader::new(data);
    let mut out = String::new();

    // Header. Bit 0 of the tag marks public-preview content; the remaining
    // bits are the version proper, which the layout decisions below use.
    let header = r.read_header()?;
    let contains_preview = header.tag & VERSION_TAG_CONTAINS_PREVIEW != 0;
    let version = BytecodeVersion {
        tag: header.tag & !VERSION_TAG_CONTAINS_PREVIEW,
        ..header
    };
    writeln!(out, "TileIR bytecode v{header}").unwrap();
    if contains_preview {
        writeln!(out, "  contains public-preview content").unwrap();
    }
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

    let types = parse_type_section(sections.get(Section::Type as u8), version, &strings)?;
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
        let cx = AttrContext {
            strings: &strings,
            types: &types,
            constants: &constants,
        };
        let funcs = parse_func_section(payload, &cx, version)?;
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

    /// `varint(count)` then `count` little-endian i64 values.
    fn read_le_var_size_i64(&mut self) -> Result<Vec<i64>> {
        let count = self.read_varint()? as usize;
        if count > self.remaining() / 8 {
            return Err(err("i64 array count exceeds the payload"));
        }
        (0..count).map(|_| self.read_le_i64()).collect()
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
    let mut offsets = Vec::with_capacity(count.min(data.len()));
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
        if start > end || end > string_data.len() {
            return Err(err("string offsets out of range"));
        }
        let s = std::str::from_utf8(&string_data[start..end])
            .unwrap_or("<invalid utf8>")
            .to_owned();
        strings.push(s);
    }
    Ok(strings)
}

// =========================================================================
// Attribute reader
// =========================================================================

/// Tables a self-contained attribute can refer to.
struct AttrContext<'a> {
    strings: &'a [String],
    /// Decoded type descriptions, indexed by type-table position.
    types: &'a [String],
    constants: &'a [Vec<u8>],
}

impl AttrContext<'_> {
    fn string(&self, r: &mut Reader) -> Result<String> {
        let idx = r.read_varint()? as usize;
        self.strings
            .get(idx)
            .cloned()
            .ok_or_else(|| err(&format!("string index {idx} out of range")))
    }

    fn type_name(&self, r: &mut Reader) -> Result<String> {
        let idx = r.read_varint()? as usize;
        self.types
            .get(idx)
            .cloned()
            .ok_or_else(|| err(&format!("type index {idx} out of range")))
    }

    /// Types embedded in attributes are simple (scalars, tiles, `index`,
    /// parameter placeholders), so the decoded description parses back.
    fn parse_type(&self, r: &mut Reader) -> Result<Type> {
        let name = self.type_name(r)?;
        Type::parse(&name).ok_or_else(|| {
            err(&format!(
                "cannot reconstruct type '{name}' referenced from an attribute"
            ))
        })
    }
}

/// Read `varint(tag)` then the payload.
fn read_tagged_attribute(r: &mut Reader, cx: &AttrContext) -> Result<Attribute> {
    let tag = r.read_varint()?;
    read_attribute_payload(tag, r, cx)
}

/// Every attribute tag the writer can produce, builtin and preview alike
/// (`BytecodeAttrOpcodes.td`).
fn read_attribute_payload(tag: u64, r: &mut Reader, cx: &AttrContext) -> Result<Attribute> {
    let tag_is = |t: AttributeTag| tag == t as u64;
    if tag_is(AttributeTag::Integer) {
        let ty = cx.parse_type(r)?;
        let value = r.read_varint()?;
        return Ok(Attribute::Integer(value as i64, ty));
    }
    if tag_is(AttributeTag::Float) {
        let ty = cx.parse_type(r)?;
        return Ok(Attribute::Float(read_ap_float(r, &ty)?, ty));
    }
    if tag_is(AttributeTag::Bool) {
        return Ok(Attribute::Bool(r.read_byte()? != 0));
    }
    if tag_is(AttributeTag::Type) {
        return Ok(Attribute::Type(cx.parse_type(r)?));
    }
    if tag_is(AttributeTag::String) {
        return Ok(Attribute::String(cx.string(r)?));
    }
    if tag_is(AttributeTag::Array) {
        return Ok(Attribute::Array(read_attribute_array(r, cx)?));
    }
    if tag_is(AttributeTag::DenseElements) {
        let ty = cx.parse_type(r)?;
        let idx = r.read_varint()? as usize;
        let entry = cx
            .constants
            .get(idx)
            .ok_or_else(|| err(&format!("constant index {idx} out of range")))?;
        // Pool entries carry their byte length first.
        let mut c = Reader::new(entry);
        let len = c.read_varint()? as usize;
        let data = c.read_bytes(len.min(c.remaining()))?.to_vec();
        return Ok(Attribute::DenseElements(DenseElements {
            element_type: ty,
            shape: Vec::new(),
            data,
        }));
    }
    if tag_is(AttributeTag::DivBy) {
        let divisor = r.read_varint()?;
        let flags = r.read_byte()?;
        let every = (flags & 1 != 0)
            .then(|| r.read_signed_varint())
            .transpose()?;
        let along = (flags & 2 != 0)
            .then(|| r.read_signed_varint())
            .transpose()?;
        return Ok(Attribute::DivBy(DivBy {
            divisor,
            every,
            along,
        }));
    }
    if tag_is(AttributeTag::SameElements) {
        return Ok(Attribute::SameElements(SameElements {
            values: r.read_le_var_size_i64()?,
        }));
    }
    if tag_is(AttributeTag::Dictionary) {
        return Ok(Attribute::Dictionary(read_dictionary(r, cx)?));
    }
    if tag_is(AttributeTag::OptimizationHints) {
        let entries = read_dictionary(r, cx)?
            .into_iter()
            .map(|(arch, value)| match value {
                Attribute::Dictionary(hints) => Ok((arch, hints)),
                _ => Err(err("optimization hint value must be a dictionary")),
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Attribute::OptimizationHints(OptimizationHints { entries }));
    }
    if tag_is(AttributeTag::Bounded) {
        let flags = r.read_byte()?;
        let lb = (flags & 1 != 0)
            .then(|| r.read_signed_varint())
            .transpose()?;
        let ub = (flags & 2 != 0)
            .then(|| r.read_signed_varint())
            .transpose()?;
        return Ok(Attribute::Bounded(Bounded { lb, ub }));
    }
    if tag_is(AttributeTag::Access) {
        let value = r.read_varint()?;
        return Access::from_u64(value)
            .map(Attribute::Access)
            .ok_or_else(|| err(&format!("unknown access value {value}")));
    }
    if tag_is(AttributeTag::MemorySpace) {
        let value = r.read_varint()?;
        return MemorySpace::from_u64(value)
            .map(Attribute::MemorySpace)
            .ok_or_else(|| err(&format!("unknown memory space value {value}")));
    }
    if tag_is(AttributeTag::Synchronization) {
        let value = r.read_varint()?;
        return Synchronization::from_u64(value)
            .map(Attribute::Synchronization)
            .ok_or_else(|| err(&format!("unknown synchronization value {value}")));
    }
    if tag_is(AttributeTag::Layout) {
        return Ok(Attribute::Layout(read_layout_payload(r, cx)?));
    }
    if tag_is(AttributeTag::ScratchRequest) {
        return Ok(Attribute::ScratchRequest(read_scratch_payload(r, cx)?));
    }
    if tag_is(AttributeTag::ConstParam) {
        return Ok(Attribute::ConstParam(read_const_param_payload(r, cx)?));
    }
    if is_sym_expr_tag(tag) {
        return Ok(Attribute::SymExpr(read_sym_expr_payload(tag, r, cx)?));
    }
    Err(err(&format!("unknown attribute tag 0x{tag:X}")))
}

/// `varint(count)` then tagged elements.
fn read_attribute_array(r: &mut Reader, cx: &AttrContext) -> Result<Vec<Attribute>> {
    let count = r.read_varint()? as usize;
    if count > r.remaining() {
        return Err(err("array count exceeds the payload"));
    }
    (0..count).map(|_| read_tagged_attribute(r, cx)).collect()
}

/// `varint(count)` then `(string index, tagged value)` pairs.
fn read_dictionary(r: &mut Reader, cx: &AttrContext) -> Result<Vec<(String, Attribute)>> {
    let count = r.read_varint()? as usize;
    if count > r.remaining() {
        return Err(err("dictionary count exceeds the payload"));
    }
    (0..count)
        .map(|_| {
            let key = cx.string(r)?;
            let value = read_tagged_attribute(r, cx)?;
            Ok((key, value))
        })
        .collect()
}

/// `writeAPFloatRepresentation`: a byte for 8-bit floats, otherwise the
/// zigzag-encoded bit pattern.
fn read_ap_float(r: &mut Reader, ty: &Type) -> Result<f64> {
    let scalar = match ty {
        Type::Scalar(s) => *s,
        _ => ScalarType::F64,
    };
    Ok(match scalar {
        ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F8E5M3FNU
        | ScalarType::F4E2M1FN => f64::from(r.read_byte()?),
        ScalarType::F16 => f64::from(half::f16::from_bits(r.read_signed_varint()? as u16).to_f32()),
        ScalarType::BF16 => {
            f64::from(half::bf16::from_bits(r.read_signed_varint()? as u16).to_f32())
        }
        ScalarType::F32 | ScalarType::TF32 => {
            f64::from(f32::from_bits(r.read_signed_varint()? as u32))
        }
        _ => f64::from_bits(r.read_signed_varint()? as u64),
    })
}

fn is_sym_expr_tag(tag: u64) -> bool {
    [
        AttributeTag::SymbolicExprInt,
        AttributeTag::SymbolicExprVar,
        AttributeTag::SymbolicExprMul,
        AttributeTag::SymbolicExprDiv,
        AttributeTag::SymbolicExprAdd,
        AttributeTag::SymbolicExprSub,
        AttributeTag::SymbolicExprFloorDiv,
        AttributeTag::SymbolicTuple,
        AttributeTag::SymbolicExprCeilDiv,
        AttributeTag::ConcreteType,
        AttributeTag::SymbolicExprNeg,
        AttributeTag::SymbolicExprBuiltin,
        AttributeTag::SymbolicExprConstStr,
    ]
    .iter()
    .any(|t| *t as u64 == tag)
}

/// A tagged symbolic expression (spec 4.3).
fn read_sym_expr(r: &mut Reader, cx: &AttrContext) -> Result<SymExpr> {
    let tag = r.read_varint()?;
    if !is_sym_expr_tag(tag) {
        return Err(err(&format!(
            "expected a symbolic expression attribute, got tag 0x{tag:X}"
        )));
    }
    read_sym_expr_payload(tag, r, cx)
}

fn read_sym_expr_payload(tag: u64, r: &mut Reader, cx: &AttrContext) -> Result<SymExpr> {
    let tag_is = |t: AttributeTag| tag == t as u64;
    let binary = |make: fn(Box<SymExpr>, Box<SymExpr>) -> SymExpr, r: &mut Reader| {
        let lhs = read_sym_expr(r, cx)?;
        let rhs = read_sym_expr(r, cx)?;
        Ok::<_, Error>(make(Box::new(lhs), Box::new(rhs)))
    };
    if tag_is(AttributeTag::SymbolicExprInt) {
        return Ok(SymExpr::Int(r.read_signed_varint()?));
    }
    if tag_is(AttributeTag::SymbolicExprVar) {
        return Ok(SymExpr::Var(cx.string(r)?));
    }
    if tag_is(AttributeTag::SymbolicExprConstStr) {
        return Ok(SymExpr::ConstStr(cx.string(r)?));
    }
    if tag_is(AttributeTag::SymbolicExprBuiltin) {
        let name = cx.string(r)?;
        let ty = cx.parse_type(r)?;
        return Ok(SymExpr::Builtin {
            name,
            ty: Box::new(ty),
        });
    }
    if tag_is(AttributeTag::ConcreteType) {
        return Ok(SymExpr::ConcreteType(Box::new(cx.parse_type(r)?)));
    }
    if tag_is(AttributeTag::SymbolicTuple) {
        return Ok(SymExpr::Tuple(read_sym_expr_array(r, cx)?));
    }
    if tag_is(AttributeTag::SymbolicExprNeg) {
        return Ok(SymExpr::Neg(Box::new(read_sym_expr(r, cx)?)));
    }
    if tag_is(AttributeTag::SymbolicExprMul) {
        return binary(SymExpr::Mul, r);
    }
    if tag_is(AttributeTag::SymbolicExprDiv) {
        return binary(SymExpr::Div, r);
    }
    if tag_is(AttributeTag::SymbolicExprAdd) {
        return binary(SymExpr::Add, r);
    }
    if tag_is(AttributeTag::SymbolicExprSub) {
        return binary(SymExpr::Sub, r);
    }
    if tag_is(AttributeTag::SymbolicExprFloorDiv) {
        return binary(SymExpr::FloorDiv, r);
    }
    if tag_is(AttributeTag::SymbolicExprCeilDiv) {
        return binary(SymExpr::CeilDiv, r);
    }
    Err(err(&format!("unknown symbolic expression tag 0x{tag:X}")))
}

fn read_sym_expr_array(r: &mut Reader, cx: &AttrContext) -> Result<Vec<SymExpr>> {
    let count = r.read_varint()? as usize;
    if count > r.remaining() {
        return Err(err("expression array count exceeds the payload"));
    }
    (0..count).map(|_| read_sym_expr(r, cx)).collect()
}

/// Layout payload: inline shapes tuple, inline strides tuple.
fn read_layout_payload(r: &mut Reader, cx: &AttrContext) -> Result<Layout> {
    let shapes = read_sym_expr_array(r, cx)?;
    let strides = read_sym_expr_array(r, cx)?;
    Ok(Layout { shapes, strides })
}

fn read_layout_tagged(r: &mut Reader, cx: &AttrContext) -> Result<Layout> {
    let tag = r.read_varint()?;
    if tag != AttributeTag::Layout as u64 {
        return Err(err(&format!(
            "expected a layout attribute, got tag 0x{tag:X}"
        )));
    }
    read_layout_payload(r, cx)
}

fn read_scratch_payload(r: &mut Reader, cx: &AttrContext) -> Result<ScratchRequest> {
    let size = read_sym_expr(r, cx)?;
    let alignment = read_sym_expr(r, cx)?;
    Ok(ScratchRequest { size, alignment })
}

fn read_const_param_payload(r: &mut Reader, cx: &AttrContext) -> Result<ConstParam> {
    let name = cx.string(r)?;
    let ty = cx.parse_type(r)?;
    Ok(ConstParam { name, ty })
}

// =========================================================================
// Type section parser
// =========================================================================

fn parse_type_section(
    payload: Option<&[u8]>,
    version: BytecodeVersion,
    strings: &[String],
) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;

    let mut offsets = Vec::with_capacity(count.min(data.len()));
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
        if start > end || end > type_data.len() {
            return Err(err("type offsets out of range"));
        }
        let desc = decode_type_entry(&type_data[start..end], &types, version, strings);
        types.push(desc);
    }
    Ok(types)
}

fn decode_type_entry(
    data: &[u8],
    prev_types: &[String],
    version: BytecodeVersion,
    strings: &[String],
) -> String {
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
            let mut dims = Vec::with_capacity(rank.min(data.len()));
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
            let mut shape = Vec::with_capacity(rank.min(data.len()));
            for _ in 0..rank {
                shape.push(r.read_le_i64().unwrap_or(0));
            }
            let stride_count = r.read_varint().unwrap_or(0) as usize;
            let mut strides = Vec::with_capacity(stride_count.min(data.len()));
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
            let mut tile_shape = Vec::with_capacity(tile_rank.min(data.len()));
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size.min(data.len()));
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
            let mut tile_shape = Vec::with_capacity(tile_rank.min(data.len()));
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
            let mut tile_shape = Vec::with_capacity(tile_rank.min(data.len()));
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let stride_rank = r.read_varint().unwrap_or(0) as usize;
            let mut traversal_strides = Vec::with_capacity(stride_rank.min(data.len()));
            for _ in 0..stride_rank {
                traversal_strides.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size.min(data.len()));
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
            let mut inputs = Vec::with_capacity(num_inputs.min(data.len()));
            for _ in 0..num_inputs {
                let idx = r.read_varint().unwrap_or(0) as usize;
                inputs.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            let num_results = r.read_varint().unwrap_or(0) as usize;
            let mut results = Vec::with_capacity(num_results.min(data.len()));
            for _ in 0..num_results {
                let idx = r.read_varint().unwrap_or(0) as usize;
                results.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            format!("({}) -> ({})", inputs.join(", "), results.join(", "))
        }
        // Public-preview types (spec 4.4). The parameter placeholders and
        // builtin singletons are tag-only.
        t if t == TypeTag::Index as u8 => "index".into(),
        t if t == TypeTag::None as u8 => "none".into(),
        t if t == TypeTag::DTypeParam as u8 => "!cuda_tile.preview$Datatype".into(),
        t if t == TypeTag::ShapeParam as u8 => "!cuda_tile.preview$shape".into(),
        t if t == TypeTag::StringParam as u8 => "!cuda_tile.preview$string".into(),
        t if t == TypeTag::SymbolicType as u8 || t == TypeTag::Fragment as u8 => {
            let cx = AttrContext {
                strings,
                types: prev_types,
                constants: &[],
            };
            let decoded = if t == TypeTag::Fragment as u8 {
                decode_fragment_type(&mut r, &cx)
            } else {
                read_sym_expr(&mut r, &cx)
                    .map(|expr| format!("preview$sym_t<{}>", format_sym_expr(&expr)))
            };
            decoded.unwrap_or_else(|e| format!("<preview type: {e}>"))
        }
        _ => format!("<unknown type tag {tag}>"),
    }
}

/// Fragment entry: tagged symbolic shape, static shape, element type index,
/// memory space, tagged layouts, tagged alignment.
fn decode_fragment_type(r: &mut Reader, cx: &AttrContext) -> Result<String> {
    let symbolic = match read_sym_expr(r, cx)? {
        SymExpr::Tuple(values) => values,
        _ => return Err(err("fragment symbolic shape must be a tuple")),
    };
    let shape = r.read_le_var_size_i64()?;
    let elem_name = cx.type_name(r)?;
    let memory_space = r.read_varint()?;
    let memory_space = MemorySpace::from_u64(memory_space)
        .ok_or_else(|| err(&format!("unknown memory space value {memory_space}")))?;
    let mem_layout = read_layout_tagged(r, cx)?;
    let fragmap = read_layout_tagged(r, cx)?;
    let alignment = read_sym_expr(r, cx)?;

    let mut symbolic = symbolic.into_iter();
    let mut dims = String::new();
    for dim in &shape {
        if *dim == DYNAMIC {
            let expr = symbolic
                .next()
                .ok_or_else(|| err("fragment has more dynamic dimensions than expressions"))?;
            write!(dims, "[{}]x", format_sym_expr(&expr)).unwrap();
        } else {
            write!(dims, "{dim}x").unwrap();
        }
    }
    let mut text = format!(
        "preview$fragment<{dims}{elem_name}, {}, mem_layout = {}, fragmap = {}",
        memory_space.name(),
        format_layout(&mem_layout),
        format_layout(&fragmap)
    );
    if alignment != SymExpr::Int(1) {
        write!(text, ", alignment = {}", format_sym_expr(&alignment)).unwrap();
    }
    text.push('>');
    Ok(text)
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

    let mut offsets = Vec::with_capacity(count.min(data.len()));
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
        if start > end || end > const_data.len() {
            return Err(err("constant offsets out of range"));
        }
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
    let mut globals = Vec::with_capacity(count.min(data.len()));
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

/// One entry per function in the 13.5 form layout or the earlier flagged
/// layout (`BytecodeReader.cpp`, `parseFunctionTableSection`).
fn parse_func_section(
    data: &[u8],
    cx: &AttrContext,
    version: BytecodeVersion,
) -> Result<Vec<String>> {
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    let mut funcs = Vec::with_capacity(count.min(data.len()));
    let form_layout = version >= BytecodeVersion::FUNCTION_FORM_LAYOUT;

    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let sig_idx = r.read_varint()? as usize;
        let flags_byte = r.read_byte()?;
        let _loc_idx = r.read_varint()?;

        let name = cx.strings.get(name_idx).cloned().unwrap_or("?".into());
        let sig = cx.types.get(sig_idx).cloned().unwrap_or("?".into());

        let is_kernel = flags_byte & FunctionFlag::KindKernel as u8 != 0;
        let mut is_extern_impl = false;
        let mut read_hints = false;
        if form_layout {
            if flags_byte
                & !(FunctionFlag::VisibilityPrivate as u8
                    | FunctionFlag::KindKernel as u8
                    | FUNCTION_FORM_MASK)
                != 0
            {
                return Err(err(&format!(
                    "function entry has unrecognized flag bits (0x{:02X})",
                    flags_byte
                )));
            }
            match flags_byte & FUNCTION_FORM_MASK {
                FUNCTION_FORM_RESERVED => {
                    return Err(err("function entry has reserved form value"));
                }
                FUNCTION_FORM_TESTING => {
                    return Err(err("function entry has the test-only form"));
                }
                FUNCTION_FORM_EXTERN_IMPL => is_extern_impl = true,
                _ => {}
            }
            if is_extern_impl && is_kernel {
                return Err(err("function entry cannot be both kernel and extern impl"));
            }
            read_hints = is_kernel;
        } else if flags_byte & FunctionFlag::HasOptimizationHints as u8 != 0 {
            read_hints = is_kernel;
        }

        let mut out = String::new();
        if is_extern_impl {
            let payload = read_extern_impl_payload(&mut r, cx)?;
            let body_len = r.read_varint()?;
            if body_len != 0 {
                return Err(err(&format!(
                    "extern implementation entry has non-zero body length ({body_len})"
                )));
            }
            writeln!(
                out,
                "  preview$extern_impl @{name} implements @{} : {sig}",
                payload.implements
            )
            .unwrap();
            for (key, value) in payload.fields {
                writeln!(out, "    {key}: {value}").unwrap();
            }
            funcs.push(out);
            continue;
        }

        let hints = if read_hints {
            Some(read_tagged_attribute(&mut r, cx)?)
        } else {
            None
        };
        let arg_attrs = if form_layout {
            Some(read_tagged_attribute(&mut r, cx)?)
        } else {
            None
        };
        let body_len = r.read_varint()? as usize;

        // A normal-form, non-kernel entry without a body is an extern
        // declaration; its constant parameters follow.
        if form_layout && !is_kernel && body_len == 0 {
            let const_params = read_tagged_attribute(&mut r, cx)?;
            writeln!(out, "  preview$func extern @{name} : {sig}").unwrap();
            if let Some(arg_attrs) = arg_attrs {
                writeln!(out, "    arg_attrs: {}", format_attr(&arg_attrs)).unwrap();
            }
            writeln!(out, "    const_params: {}", format_attr(&const_params)).unwrap();
            funcs.push(out);
            continue;
        }

        let body_data = r.read_bytes(body_len)?;
        let op_count = count_ops_in_body(body_data);
        let kind = if is_kernel { "entry" } else { "func" };
        writeln!(out, "  {kind} @{name} : {sig}").unwrap();
        if let Some(hints) = hints {
            writeln!(out, "    optimization_hints: {}", format_attr(&hints)).unwrap();
        }
        if let Some(arg_attrs) = arg_attrs {
            writeln!(out, "    arg_attrs: {}", format_attr(&arg_attrs)).unwrap();
        }
        writeln!(out, "    body: {body_len} bytes, ~{op_count} ops").unwrap();
        funcs.push(out);
    }
    Ok(funcs)
}

/// The decoded `preview$extern_impl` payload: the declaration it implements
/// and every other field as `(name, formatted value)` in declaration order.
struct ExternImplPayload {
    implements: String,
    fields: Vec<(&'static str, String)>,
}

/// Spec 4.6: flags, sym_name, implements, optional arrays in declaration
/// order around the function type, enum and integer tails, one empty region.
fn read_extern_impl_payload(r: &mut Reader, cx: &AttrContext) -> Result<ExternImplPayload> {
    let flags = r.read_varint()?;
    let present = |bit: u32| flags & (1 << bit) != 0;
    let _sym_name = cx.string(r)?;
    let implements = cx.string(r)?;
    let mut fields = Vec::new();
    let array = |r: &mut Reader, name: &'static str, fields: &mut Vec<_>| -> Result<()> {
        let items = read_attribute_array(r, cx)?;
        fields.push((name, format_attr(&Attribute::Array(items))));
        Ok(())
    };
    if present(0) {
        array(r, "const_params", &mut fields)?;
    }
    if present(1) {
        array(r, "const_args", &mut fields)?;
    }
    let _function_type = cx.type_name(r)?;
    if present(2) {
        array(r, "arg_attrs", &mut fields)?;
    }
    if present(3) {
        array(r, "res_attrs", &mut fields)?;
    }
    if present(4) {
        let scratch = read_scratch_payload(r, cx)?;
        fields.push((
            "scratch_smem",
            format_attr(&Attribute::ScratchRequest(scratch)),
        ));
    }
    if present(5) {
        let config = read_dictionary(r, cx)?;
        fields.push(("config", format_attr(&Attribute::Dictionary(config))));
    }
    if present(6) {
        array(r, "arch", &mut fields)?;
    }
    if present(7) {
        let abi = read_dictionary(r, cx)?;
        fields.push(("frontendABI", format_attr(&Attribute::Dictionary(abi))));
    }
    if present(8) {
        let value = r.read_varint()?;
        let sync = Synchronization::from_u64(value)
            .ok_or_else(|| err(&format!("unknown synchronization value {value}")))?;
        fields.push((
            "synchronization",
            format_attr(&Attribute::Synchronization(sync)),
        ));
    }
    if present(9) {
        // Unsigned LEB128 of the zero-extended i64.
        let score = r.read_varint()? as i64;
        fields.push(("provider_score", score.to_string()));
    }
    if flags >> 10 != 0 {
        return Err(err(&format!(
            "extern implementation has unrecognized flag bits (0x{:X})",
            flags >> 10
        )));
    }
    let regions = r.read_varint()?;
    let blocks = if regions == 1 { r.read_varint()? } else { 1 };
    if regions != 1 || blocks != 0 {
        return Err(err(
            "extern implementation must have one empty region and no other",
        ));
    }
    Ok(ExternImplPayload { implements, fields })
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

    /// Every self-contained attribute tag reads back to the value that was
    /// written, through the same string and type tables.
    #[test]
    fn attribute_reader_covers_every_tag() {
        use crate::bytecode::writer::{
            write_self_contained_attribute, ConstantManager, StringManager, TypeManager,
        };
        use crate::ir::{DivBy, SameElements, Type};

        let f32_ty = Type::Scalar(ScalarType::F32);
        let i32_ty = Type::Scalar(ScalarType::I32);
        let samples = vec![
            Attribute::Integer(7, i32_ty.clone()),
            Attribute::Float(1.5, f32_ty.clone()),
            Attribute::Bool(true),
            Attribute::Type(Type::Index),
            Attribute::String("name".into()),
            Attribute::Array(vec![Attribute::Bool(false), Attribute::String("x".into())]),
            Attribute::DenseElements(DenseElements {
                element_type: f32_ty.clone(),
                shape: vec![],
                data: 2.0f32.to_le_bytes().to_vec(),
            }),
            Attribute::DivBy(DivBy {
                divisor: 16,
                every: Some(2),
                along: Some(0),
            }),
            Attribute::SameElements(SameElements { values: vec![4, 2] }),
            Attribute::Dictionary(vec![("k".into(), Attribute::Integer(1, i32_ty.clone()))]),
            Attribute::OptimizationHints(OptimizationHints {
                entries: vec![(
                    "sm_120".into(),
                    vec![("occupancy".into(), Attribute::Integer(2, i32_ty.clone()))],
                )],
            }),
            Attribute::Bounded(Bounded {
                lb: Some(-3),
                ub: None,
            }),
            Attribute::Access(Access::Out),
            Attribute::MemorySpace(MemorySpace::Smem),
            Attribute::Synchronization(Synchronization::Warpgroup),
            Attribute::Layout(Layout {
                shapes: vec![
                    SymExpr::Int(8),
                    SymExpr::tuple([SymExpr::Int(2), SymExpr::var("M")]),
                ],
                strides: vec![
                    SymExpr::Int(0),
                    SymExpr::tuple([SymExpr::Int(1), SymExpr::Int(2)]),
                ],
            }),
            Attribute::SymExpr(SymExpr::negate(SymExpr::ceil_div(
                SymExpr::floor_div(SymExpr::var("N"), SymExpr::Int(4)),
                SymExpr::minus(
                    SymExpr::plus(SymExpr::builtin("context.num_threads"), SymExpr::Int(1)),
                    SymExpr::times(
                        SymExpr::Int(2),
                        SymExpr::Div(Box::new(SymExpr::Int(9)), Box::new(SymExpr::Int(3))),
                    ),
                ),
            ))),
            Attribute::SymExpr(SymExpr::const_str("nvrtc")),
            Attribute::SymExpr(SymExpr::concrete_type(f32_ty.clone())),
            Attribute::ScratchRequest(ScratchRequest::bytes(512)),
            Attribute::ConstParam(ConstParam::wildcard()),
            Attribute::ConstParam(ConstParam::new(
                "T",
                Type::PreviewParam(crate::ir::PreviewParamType::DType),
            )),
        ];

        let mut strings = StringManager::new();
        let mut types = TypeManager::new();
        let mut constants = ConstantManager::new();
        let mut w = EncodingWriter::new();
        for attr in &samples {
            write_self_contained_attribute(attr, &mut w, &mut strings, &mut types, &mut constants)
                .unwrap();
        }
        let string_table = strings.entries();
        let type_table: Vec<String> = types.entries().iter().map(crate::ir::format_type).collect();
        let constant_table = constants.entries().to_vec();
        let cx = AttrContext {
            strings: &string_table,
            types: &type_table,
            constants: &constant_table,
        };
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        for expected in &samples {
            let decoded = read_tagged_attribute(&mut r, &cx).unwrap();
            // The dense element shape is not encoded; everything else reads
            // back as written.
            assert_eq!(format_attr(&decoded), format_attr(expected));
        }
        assert_eq!(r.remaining(), 0);
    }
}
