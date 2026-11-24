/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode reader — reconstructs a [`Module`](crate::ir::Module) from Tile
//! IR bytecode.
//!
//! Ported from the top-level `readBytecode` function and its helper
//! managers in `BytecodeReader.cpp`.

use std::collections::HashMap;

use super::encoding::EncodingReader;
use super::enums::{AttributeTag, BytecodeVersion, Section, TypeTag};
use super::opcode::Opcode;
use crate::ir::{
    Attribute, Block, BlockId, DIFile, DILexicalBlock, DISubprogram, DebugInfoLoc, DebugScope,
    DenseElements, Location, Module, OpId, Operation, PaddingValue, Region, RegionId, ScalarType,
    SymbolVisibility, TileElementType, TileType, Type, Value, ValueProducer,
};
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Deserialize Tile IR bytecode into a [`Module`].
///
/// The bytecode version is read from the header. Versions outside
/// [`BytecodeVersion::MIN_SUPPORTED`]..=[`BytecodeVersion::CURRENT`] are
/// rejected.
pub fn read_bytecode(data: &[u8]) -> Result<Module> {
    Ok(read_bytecode_versioned(data)?.0)
}

/// Deserialize Tile IR bytecode into [`Module`] with a specific bytecode version.
pub fn read_bytecode_versioned(data: &[u8]) -> Result<(Module, BytecodeVersion)> {
    let mut r = EncodingReader::new(data);
    // The preview bit rides in the header's tag field; it selects op
    // vocabulary, not layout, so strip it before any version comparison.
    let version = r.read_header()?.without_preview();
    if version < BytecodeVersion::MIN_SUPPORTED || version > BytecodeVersion::CURRENT {
        return Err(Error::BytecodeRead(format!(
            "unsupported bytecode version {version}; supported range is {}..={}",
            BytecodeVersion::MIN_SUPPORTED,
            BytecodeVersion::CURRENT
        )));
    }

    // Collect raw sections first, then parse in dependency order (strings
    // and types are referenced by every later section).
    let mut sections = SectionTable::new();
    loop {
        let (id, len, _aligned) = r.read_section_header()?;
        if id == Section::EndOfBytecode as u8 {
            break;
        }
        if (id as usize) >= super::enums::NUM_SECTIONS as usize {
            return Err(Error::BytecodeRead(format!(
                "unknown section id {id} (valid range 1..={})",
                super::enums::NUM_SECTIONS - 1
            )));
        }
        sections.insert(id, r.read_bytes(len)?);
    }

    let mut reader = ReaderState {
        module: Module::new("module"),
        version,
        strings: parse_string_section(sections.get(Section::String as u8))?,
        types: parse_type_section(sections.get(Section::Type as u8), version)?,
        constants: parse_constant_section(sections.get(Section::Constant as u8))?,
        debug: parse_debug_section(sections.get(Section::Debug as u8))?,
        value_map: HashMap::new(),
        next_idx: 0,
        debug_cursor: None,
    };

    reader.parse_global_section(sections.get(Section::Global as u8))?;
    if let Some(payload) = sections.get(Section::Producer as u8) {
        let mut r = EncodingReader::new(payload);
        let idx = r.read_varint()?;
        reader.module.producer = Some(table_get(&reader.strings, idx, "producer string")?);
    }
    if let Some(payload) = sections.get(Section::Func as u8) {
        let mut r = EncodingReader::new(payload);
        reader.parse_func_section(&mut r)?;
    }

    Ok((reader.module, version))
}

// ---------------------------------------------------------------------------
// Section framing
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct SectionTable<'a> {
    data: [Option<&'a [u8]>; super::enums::NUM_SECTIONS as usize],
}

impl<'a> SectionTable<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn insert(&mut self, id: u8, payload: &'a [u8]) {
        if (id as usize) < self.data.len() {
            self.data[id as usize] = Some(payload);
        }
    }
    pub(crate) fn get(&self, id: u8) -> Option<&'a [u8]> {
        self.data.get(id as usize).copied().flatten()
    }
}

impl EncodingReader<'_> {
    /// Read the 10-byte file header (magic + version triple).
    pub(crate) fn read_header(&mut self) -> Result<BytecodeVersion> {
        let magic = self.read_bytes(8)?;
        if magic != super::enums::MAGIC {
            return Err(Error::BytecodeRead("invalid magic number".into()));
        }
        let major = self.read_byte()?;
        let minor = self.read_byte()?;
        let tag = self.read_le_u16()?;
        Ok(BytecodeVersion { major, minor, tag })
    }

    /// Returns `(section_id, data_len, has_alignment)`. The alignment
    /// varint and its padding are consumed when `has_alignment` is set.
    pub(crate) fn read_section_header(&mut self) -> Result<(u8, usize, bool)> {
        let id_and_align = self.read_byte()?;
        let id = id_and_align & 0x7F;
        let has_alignment = id_and_align & 0x80 != 0;
        if id == Section::EndOfBytecode as u8 {
            return Ok((id, 0, false));
        }
        let length = self.read_varint()? as usize;
        if has_alignment {
            let alignment = self.read_varint()?;
            self.skip_padding(alignment)?;
        }
        Ok((id, length, has_alignment))
    }
}

// ---------------------------------------------------------------------------
// Offset-table framing (shared by the string, type, and constant sections)
// ---------------------------------------------------------------------------

/// A parsed offset table: the payload bytes plus one validated `(start, end)`
/// range per entry. Entries are slices of [`Self::data`], so no per-entry
/// copy is made.
pub(crate) struct OffsetTable<'a> {
    data: &'a [u8],
    ranges: Vec<(usize, usize)>,
}

impl<'a> OffsetTable<'a> {
    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }
    /// The validated payload slice for entry `i`.
    pub(crate) fn entry(&self, i: usize) -> &'a [u8] {
        let (start, end) = self.ranges[i];
        &self.data[start..end]
    }
}

/// Parse the "count -> align -> offset array -> payload" offset table.
pub(crate) fn read_offset_table<'a>(
    data: &'a [u8],
    what: &str,
    offset_bytes: u32,
) -> Result<OffsetTable<'a>> {
    let mut r = EncodingReader::new(data);
    let count = super::encoding::cap_count(r.read_varint()?, what)?;
    if count > 0 {
        r.skip_padding(offset_bytes as u64)?;
    }
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        let off = if offset_bytes == 4 {
            r.read_le_u32()? as usize
        } else {
            r.read_le_u64()? as usize
        };
        offsets.push(off);
    }
    let payload = r.read_bytes(r.remaining())?;
    let mut ranges = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            payload.len()
        };
        if start > end || end > payload.len() {
            return Err(Error::BytecodeRead(format!(
                "{what} table offset {i} out of range"
            )));
        }
        ranges.push((start, end));
    }
    Ok(OffsetTable {
        data: payload,
        ranges,
    })
}

// ---------------------------------------------------------------------------
// String section
// ---------------------------------------------------------------------------

pub(crate) fn parse_string_section(payload: Option<&[u8]>) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let table = read_offset_table(data, "string", 4)?;
    let mut strings = Vec::with_capacity(table.len());
    for i in 0..table.len() {
        let s = std::str::from_utf8(table.entry(i))
            .map_err(|_| Error::BytecodeRead(format!("string table entry {i} is not utf8")))?
            .to_owned();
        strings.push(s);
    }
    Ok(strings)
}

// ---------------------------------------------------------------------------
// Type section
// ---------------------------------------------------------------------------

pub(crate) fn parse_type_section(
    payload: Option<&[u8]>,
    version: BytecodeVersion,
) -> Result<Vec<Type>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let table = read_offset_table(data, "type", 4)?;
    let mut types: Vec<Type> = Vec::with_capacity(table.len());
    for i in 0..table.len() {
        // Dependency types always precede their dependents (the writer
        // registers them first), so decoding in table order is well-defined.
        let mut er = EncodingReader::new(table.entry(i));
        let ty = read_type_entry(&mut er, &types, version)?;
        if er.remaining() != 0 {
            return Err(Error::BytecodeRead(format!(
                "type table entry {i} has trailing bytes"
            )));
        }
        types.push(ty);
    }
    Ok(types)
}

/// Decode one type-table entry into a [`Type`]. `prev` holds the already
/// decoded entries (referenced by index from this one).
pub(crate) fn read_type_entry(
    r: &mut EncodingReader,
    prev: &[Type],
    version: BytecodeVersion,
) -> Result<Type> {
    let v13_3 = version >= BytecodeVersion::V13_3;
    let v13_4 = version >= BytecodeVersion::V13_4;
    let tag = r.read_varint()? as u8;
    let index = |r: &mut EncodingReader, what: &str| -> Result<Type> {
        table_get(prev, r.read_varint()?, what)
    };
    match tag {
        t if t == TypeTag::I1 as u8 => Ok(Type::Scalar(ScalarType::I1)),
        t if t == TypeTag::I4 as u8 => Ok(Type::Scalar(ScalarType::I4)),
        t if t == TypeTag::I8 as u8 => Ok(Type::Scalar(ScalarType::I8)),
        t if t == TypeTag::I16 as u8 => Ok(Type::Scalar(ScalarType::I16)),
        t if t == TypeTag::I32 as u8 => Ok(Type::Scalar(ScalarType::I32)),
        t if t == TypeTag::I64 as u8 => Ok(Type::Scalar(ScalarType::I64)),
        t if t == TypeTag::F16 as u8 => Ok(Type::Scalar(ScalarType::F16)),
        t if t == TypeTag::BF16 as u8 => Ok(Type::Scalar(ScalarType::BF16)),
        t if t == TypeTag::F32 as u8 => Ok(Type::Scalar(ScalarType::F32)),
        t if t == TypeTag::TF32 as u8 => Ok(Type::Scalar(ScalarType::TF32)),
        t if t == TypeTag::F64 as u8 => Ok(Type::Scalar(ScalarType::F64)),
        t if t == TypeTag::F8E4M3FN as u8 => Ok(Type::Scalar(ScalarType::F8E4M3FN)),
        t if t == TypeTag::F8E5M2 as u8 => Ok(Type::Scalar(ScalarType::F8E5M2)),
        t if t == TypeTag::F8E8M0FNU as u8 => Ok(Type::Scalar(ScalarType::F8E8M0FNU)),
        t if t == TypeTag::F4E2M1FN as u8 => Ok(Type::Scalar(ScalarType::F4E2M1FN)),
        t if t == TypeTag::F8E5M3FNU as u8 => Ok(Type::Scalar(ScalarType::F8E5M3FNU)),
        t if t == TypeTag::Token as u8 => Ok(Type::Token),
        t if t == TypeTag::Pointer as u8 => {
            let has_attr = v13_4 && r.read_varint()? != 0;
            let elem = index(r, "pointee")?;
            let base = match elem {
                Type::Scalar(s) => Type::Pointer(crate::ir::PointerType { pointee: s }),
                other => {
                    return Err(Error::BytecodeRead(format!(
                        "pointer pointee must be a scalar, got {other:?}"
                    )))
                }
            };
            if has_attr {
                let attr = read_pointer_attribute(r, version)?;
                Ok(Type::WithPointerAttribute(Box::new(base), attr))
            } else {
                Ok(base)
            }
        }
        t if t == TypeTag::Tile as u8 => {
            let elem = index(r, "tile element")?;
            let element_type = match elem {
                Type::Scalar(s) => TileElementType::Scalar(s),
                Type::Pointer(p) => TileElementType::Pointer(Box::new(p)),
                // An attributed pointer here means the tile was serialized
                // through WithPointerAttribute; the tile element type cannot
                // carry the attribute, so this is not representable.
                Type::WithPointerAttribute(base, _) if matches!(*base, Type::Pointer(_)) => {
                    return Err(Error::BytecodeRead(
                        "attributed pointer tile elements are not supported by the reader".into(),
                    ))
                }
                other => {
                    return Err(Error::BytecodeRead(format!(
                        "tile element must be a scalar or pointer, got {other:?}"
                    )))
                }
            };
            let shape = r.read_le_var_size_i64()?;
            Ok(Type::Tile(TileType {
                shape,
                element_type,
            }))
        }
        t if t == TypeTag::TensorView as u8 => {
            let has_attr = v13_4 && r.read_varint()? != 0;
            let elem = index(r, "tensor_view element")?;
            let Type::Scalar(element_type) = elem else {
                return Err(Error::BytecodeRead(
                    "tensor_view element must be a scalar".into(),
                ));
            };
            let shape = r.read_le_var_size_i64()?;
            let strides = r.read_le_var_size_i64()?;
            let base = Type::TensorView(crate::ir::TensorViewType {
                element_type,
                shape,
                strides,
            });
            if has_attr {
                let attr = read_pointer_attribute(r, version)?;
                Ok(Type::WithPointerAttribute(Box::new(base), attr))
            } else {
                Ok(base)
            }
        }
        t if t == TypeTag::PartitionView as u8 => {
            // 13.3+ stores flags first.
            // 13.2 stores flags after the dim_map.
            let flags = if v13_3 {
                Some(r.read_varint()? & 1 != 0)
            } else {
                None
            };
            let tile_shape = r.read_le_var_size_i32()?;
            let (tensor_view, pointer_attribute) =
                split_view_tensor_view(index(r, "partition_view tensor_view")?, "partition_view")?;
            let dim_map = r.read_le_var_size_i32()?;
            let has_padding = match flags {
                Some(has) => has,
                None => r.read_byte()? != 0,
            };
            let padding_value = if has_padding {
                let p = if v13_3 {
                    r.read_byte()?
                } else {
                    r.read_varint()? as u8
                };
                Some(padding_value_from_u8(p)?)
            } else {
                None
            };
            Ok(qualify_view(
                Type::PartitionView(crate::ir::PartitionViewType {
                    tile_shape,
                    tensor_view,
                    dim_map,
                    padding_value,
                }),
                pointer_attribute,
            ))
        }
        t if t == TypeTag::GatherScatterView as u8 => {
            let flags = r.read_varint()?;
            let tile_shape = r.read_le_var_size_i32()?;
            let (tensor_view, pointer_attribute) = split_view_tensor_view(
                index(r, "gather_scatter_view tensor_view")?,
                "gather_scatter_view",
            )?;
            let sparse_dim = r.read_varint()? as i32;
            let padding_value = if flags & 1 != 0 {
                Some(padding_value_from_u8(r.read_byte()?)?)
            } else {
                None
            };
            Ok(qualify_view(
                Type::GatherScatterView(crate::ir::GatherScatterViewType {
                    tile_shape,
                    tensor_view,
                    sparse_dim,
                    padding_value,
                }),
                pointer_attribute,
            ))
        }
        t if t == TypeTag::StridedView as u8 => {
            let flags = r.read_varint()?;
            let tile_shape = r.read_le_var_size_i32()?;
            let traversal_strides = r.read_le_var_size_i32()?;
            let (tensor_view, pointer_attribute) =
                split_view_tensor_view(index(r, "strided_view tensor_view")?, "strided_view")?;
            let dim_map = r.read_le_var_size_i32()?;
            let padding_value = if flags & 1 != 0 {
                Some(padding_value_from_u8(r.read_byte()?)?)
            } else {
                None
            };
            Ok(qualify_view(
                Type::StridedView(crate::ir::StridedViewType {
                    tile_shape,
                    traversal_strides,
                    tensor_view,
                    dim_map,
                    padding_value,
                }),
                pointer_attribute,
            ))
        }
        t if t == TypeTag::Func as u8 => {
            let num_inputs = super::encoding::cap_count(r.read_varint()?, "func input")?;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                inputs.push(index(r, "func input")?);
            }
            let num_results = super::encoding::cap_count(r.read_varint()?, "func result")?;
            let mut results = Vec::with_capacity(num_results);
            for _ in 0..num_results {
                results.push(index(r, "func result")?);
            }
            Ok(Type::Func(crate::ir::FuncType { inputs, results }))
        }
        t => Err(Error::BytecodeRead(format!("unknown type tag {t}"))),
    }
}

/// A view's pointer attribute rides on the tensor_view entry it references,
/// because the writer pushes the qualification down into the dependency.
/// Lift it back off so it can be restored onto the view type itself.
fn split_view_tensor_view(
    ty: Type,
    what: &str,
) -> Result<(
    crate::ir::TensorViewType,
    Option<crate::ir::PointerAttribute>,
)> {
    let (base, attr) = match ty {
        Type::WithPointerAttribute(base, attr) => (*base, Some(attr)),
        other => (other, None),
    };
    match base {
        Type::TensorView(tv) => Ok((tv, attr)),
        _ => Err(Error::BytecodeRead(format!(
            "{what} tensor_view must be a tensor_view"
        ))),
    }
}

fn qualify_view(view: Type, attr: Option<crate::ir::PointerAttribute>) -> Type {
    match attr {
        Some(attr) => view.with_pointer_attribute(attr),
        None => view,
    }
}

/// Read a 13.4 pointer-attribute byte (inverse of the writer's
/// `attr as u8`). The fabric classifications arrived at 13.5.
fn read_pointer_attribute(
    r: &mut EncodingReader,
    version: BytecodeVersion,
) -> Result<crate::ir::PointerAttribute> {
    let b = r.read_byte()?;
    let attr = crate::ir::PointerAttribute::from_u8(b)
        .ok_or_else(|| Error::BytecodeRead(format!("unknown pointer attribute {b}")))?;
    if attr != crate::ir::PointerAttribute::None && version < BytecodeVersion::V13_5 {
        return Err(Error::BytecodeRead(format!(
            "pointer attribute {} requires Tile IR 13.5, bytecode is {version}",
            attr.name()
        )));
    }
    Ok(attr)
}

fn padding_value_from_u8(p: u8) -> Result<PaddingValue> {
    PaddingValue::from_u8(p)
        .ok_or_else(|| Error::BytecodeRead(format!("invalid padding value {p}")))
}

// ---------------------------------------------------------------------------
// Constant section
// ---------------------------------------------------------------------------

pub(crate) fn parse_constant_section(payload: Option<&[u8]>) -> Result<Vec<Vec<u8>>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let table = read_offset_table(data, "constant", 8)?;
    let mut constants = Vec::with_capacity(table.len());
    for i in 0..table.len() {
        constants.push(table.entry(i).to_vec());
    }
    Ok(constants)
}

// ---------------------------------------------------------------------------
// Debug section
// ---------------------------------------------------------------------------

/// The parsed Debug section
#[derive(Default)]
struct DebugSection {
    /// `index_offsets[i]` = start of function `i`'s ids in `attr_ids`.
    index_offsets: Vec<usize>,
    attr_ids: Vec<u64>,
    /// `attr_offsets[i]` = start of attribute id `i+1` in `attr_data`.
    attr_offsets: Vec<usize>,
    attr_data: Vec<u8>,
    /// Lazily decoded table entries, indexed by id-1.
    cache: Vec<Option<DebugAttr>>,
}

impl DebugSection {
    /// The per-function id list for 1-based `func_idx`, or `None` if the
    /// function has no debug info.
    fn func_ids(&self, func_idx: u64) -> Option<&[u64]> {
        // Index 0 means the function carries no debug info at all.
        let i = (func_idx as usize).checked_sub(1)?;
        let start = *self.index_offsets.get(i)?;
        let end = self
            .index_offsets
            .get(i + 1)
            .copied()
            .unwrap_or(self.attr_ids.len());
        self.attr_ids.get(start..end)
    }

    /// Decode (memoized) the table entry for 1-based attribute `id`.
    fn attr(&mut self, id: u64, strings: &[String]) -> Result<&DebugAttr> {
        if id == 0 {
            return Err(Error::BytecodeRead(
                "debug attribute id 0 has no table entry".into(),
            ));
        }
        let i = (id as usize) - 1;
        let slot = self
            .cache
            .get_mut(i)
            .ok_or_else(|| Error::BytecodeRead(format!("debug attribute id {id} out of range")))?;
        if slot.is_none() {
            let start = *self.attr_offsets.get(i).ok_or_else(|| {
                Error::BytecodeRead(format!("debug attribute id {id} out of range"))
            })?;
            let end = self
                .attr_offsets
                .get(i + 1)
                .copied()
                .unwrap_or(self.attr_data.len());
            if start > end || end > self.attr_data.len() {
                return Err(Error::BytecodeRead(format!(
                    "debug attribute {id} offsets out of range"
                )));
            }
            let mut r = EncodingReader::new(&self.attr_data[start..end]);
            let attr = decode_debug_attr(&mut r, strings)?;
            if r.remaining() != 0 {
                return Err(Error::BytecodeRead(format!(
                    "debug attribute {id} has trailing bytes"
                )));
            }
            *slot = Some(attr);
        }
        Ok(self.cache[i].as_ref().unwrap())
    }
}

/// A decoded debug table entry. String/di references are kept as raw
/// indices and resolved when building locations/scopes.
#[derive(Debug, Clone)]
enum DebugAttr {
    Unknown,
    File {
        name: u64,
        dir: u64,
    },
    CompileUnit {
        file: u64,
    },
    LexicalBlock {
        scope: u64,
        file: u64,
        line: u32,
        column: u32,
    },
    Loc {
        scope: u64,
        file: u64,
        line: u32,
        column: u32,
    },
    Subprogram {
        file: u64,
        line: u32,
        name: u64,
        linkage: u64,
        cu: u64,
        scope_line: u32,
        is_declaration: bool,
    },
    CallSite {
        callee: u64,
        caller: u64,
    },
}

fn parse_debug_section(payload: Option<&[u8]>) -> Result<DebugSection> {
    let Some(data) = payload else {
        return Ok(DebugSection::default());
    };
    let mut r = EncodingReader::new(data);
    let cap32 = data.len() / 4;
    let cap64 = data.len() / 8;
    let num_functions = super::encoding::cap_count(r.read_varint()?, "debug function")?;
    if num_functions > cap32 {
        return Err(Error::BytecodeRead(
            "debug section function count exceeds payload".into(),
        ));
    }
    r.skip_padding(4)?;
    let mut index_offsets = Vec::with_capacity(num_functions);
    for _ in 0..num_functions {
        index_offsets.push(r.read_le_u32()? as usize);
    }
    let num_indices = super::encoding::cap_count(r.read_varint()?, "debug index")?;
    if num_indices > cap64 {
        return Err(Error::BytecodeRead(
            "debug section index count exceeds payload".into(),
        ));
    }
    r.skip_padding(8)?;
    let mut attr_ids = Vec::with_capacity(num_indices);
    for _ in 0..num_indices {
        attr_ids.push(r.read_le_u64()?);
    }
    let attr_count = super::encoding::cap_count(r.read_varint()?, "debug attribute")?;
    if attr_count > cap32 {
        return Err(Error::BytecodeRead(
            "debug section attribute count exceeds payload".into(),
        ));
    }
    r.skip_padding(4)?;
    let mut attr_offsets = Vec::with_capacity(attr_count);
    for _ in 0..attr_count {
        attr_offsets.push(r.read_le_u32()? as usize);
    }
    let attr_data = r.read_bytes(r.remaining())?;

    Ok(DebugSection {
        index_offsets,
        attr_ids,
        attr_offsets,
        attr_data: attr_data.to_vec(),
        cache: vec![None; attr_count],
    })
}

fn decode_debug_attr(r: &mut EncodingReader, strings: &[String]) -> Result<DebugAttr> {
    let _ = strings;
    let tag = r.read_byte()?;
    use super::enums::DebugTag;
    match tag {
        t if t == DebugTag::Unknown as u8 => Ok(DebugAttr::Unknown),
        t if t == DebugTag::DIFile as u8 => Ok(DebugAttr::File {
            name: r.read_varint()?,
            dir: r.read_varint()?,
        }),
        t if t == DebugTag::DICompileUnit as u8 => Ok(DebugAttr::CompileUnit {
            file: r.read_varint()?,
        }),
        t if t == DebugTag::DILexicalBlock as u8 => Ok(DebugAttr::LexicalBlock {
            scope: r.read_varint()?,
            file: r.read_varint()?,
            line: r.read_varint()? as u32,
            column: r.read_varint()? as u32,
        }),
        t if t == DebugTag::DILoc as u8 => Ok(DebugAttr::Loc {
            scope: r.read_varint()?,
            file: r.read_varint()?,
            line: r.read_varint()? as u32,
            column: r.read_varint()? as u32,
        }),
        t if t == DebugTag::DISubprogram as u8 => {
            let file = r.read_varint()?;
            let line = r.read_varint()? as u32;
            let name = r.read_varint()?;
            let linkage = r.read_varint()?;
            let cu = r.read_varint()?;
            let scope_line = r.read_varint()? as u32;
            // The 13.5 flags field; writers before it omit the trailing varint.
            let is_declaration = r.remaining() > 0 && r.read_varint()? != 0;
            Ok(DebugAttr::Subprogram {
                file,
                line,
                name,
                linkage,
                cu,
                scope_line,
                is_declaration,
            })
        }
        t if t == DebugTag::CallSite as u8 => Ok(DebugAttr::CallSite {
            callee: r.read_varint()?,
            caller: r.read_varint()?,
        }),
        t => Err(Error::BytecodeRead(format!(
            "unknown debug attribute tag {t}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Reader state
// ---------------------------------------------------------------------------

struct ReaderState {
    module: Module,
    version: BytecodeVersion,
    strings: Vec<String>,
    types: Vec<Type>,
    constants: Vec<Vec<u8>>,
    debug: DebugSection,
    /// Bytecode value index to module value, for the function currently
    /// being parsed (block-scoped, with rollback).
    value_map: HashMap<u64, Value>,
    next_idx: u64,
    /// `(function_index, consumed_position)` into the function's per-op
    /// debug id list, while a body is being parsed.
    debug_cursor: Option<(u64, usize)>,
}

impl ReaderState {
    fn v13_3(&self) -> bool {
        self.version >= BytecodeVersion::V13_3
    }

    fn read_string_idx(&self, r: &mut EncodingReader, what: &str) -> Result<String> {
        table_get(&self.strings, r.read_varint()?, what)
    }

    fn read_type_idx(&self, r: &mut EncodingReader) -> Result<Type> {
        table_get(&self.types, r.read_varint()?, "type")
    }

    /// Read one constant-pool entry: `varint(len) + bytes`.
    fn read_constant_entry(&self, idx: usize) -> Result<Vec<u8>> {
        let entry = self
            .constants
            .get(idx)
            .ok_or_else(|| Error::BytecodeRead(format!("constant index {idx} out of range")))?;
        let mut r = EncodingReader::new(entry);
        let len = super::encoding::cap_count(r.read_varint()?, "constant length")?;
        Ok(r.read_bytes(len)?.to_vec())
    }

    /// Whether `name` is present in the string table.
    fn name_in_string_table(&self, name: &str) -> bool {
        self.strings.iter().any(|s| s.as_str() == name)
    }

    fn read_results_fixed(&self, r: &mut EncodingReader, count: usize) -> Result<Vec<Type>> {
        (0..count).map(|_| self.read_type_idx(r)).collect()
    }

    fn read_results_varint(&self, r: &mut EncodingReader) -> Result<Vec<Type>> {
        let count = super::encoding::cap_count(r.read_varint()?, "result type")?;
        (0..count).map(|_| self.read_type_idx(r)).collect()
    }

    fn read_operand(&self, r: &mut EncodingReader, what: &str) -> Result<Value> {
        let idx = r.read_varint()?;
        self.value_map.get(&idx).copied().ok_or_else(|| {
            Error::BytecodeRead(format!(
                "{what} index {idx} does not reference a defined value (next_idx={})",
                self.next_idx
            ))
        })
    }

    fn read_operands_fixed(&self, r: &mut EncodingReader, count: usize) -> Result<Vec<Value>> {
        (0..count)
            .map(|_| self.read_operand(r, "operand"))
            .collect()
    }

    fn read_operands_sized(&self, r: &mut EncodingReader) -> Result<Vec<Value>> {
        let count = super::encoding::cap_count(r.read_varint()?, "operand")?;
        (0..count)
            .map(|_| self.read_operand(r, "operand"))
            .collect()
    }

    fn read_operand_group(&self, r: &mut EncodingReader, count: usize) -> Result<Vec<Value>> {
        self.read_operands_fixed(r, count)
    }

    /// Read a variadic operand group (size varint + indices).
    fn read_variadic_operand_group(&self, r: &mut EncodingReader) -> Result<Vec<Value>> {
        self.read_operands_sized(r)
    }

    // ------ Debug locations ------

    /// Take the next operation's location from the function's debug list.
    fn take_op_location(&mut self) -> Result<Location> {
        let Some((func_idx, pos)) = self.debug_cursor.as_mut() else {
            return Ok(Location::Unknown);
        };
        let ids = self
            .debug
            .func_ids(*func_idx)
            .ok_or_else(|| Error::BytecodeRead(format!("function {func_idx} has no debug ids")))?;
        // ids[0] is the function's own attribute; ops start at 1.
        let id = *ids.get(*pos).ok_or_else(|| {
            Error::BytecodeRead(format!(
                "function {func_idx} ran out of per-op debug ids before its body ended"
            ))
        })?;
        *pos += 1;
        self.resolve_location(id)
    }

    /// The function's own location, plus the display name from the function's
    /// subprogram attribute.
    fn function_location(
        &mut self,
        func_idx: u64,
        sym_name: &str,
    ) -> Result<(Location, Option<String>)> {
        let Some(ids) = self.debug.func_ids(func_idx) else {
            return Ok((Location::Unknown, None));
        };
        let Some(&id) = ids.first() else {
            return Ok((Location::Unknown, None));
        };
        if id == 0 {
            return Ok((Location::Unknown, None));
        }
        let attr = self.debug.attr(id, &self.strings)?.clone();
        match attr {
            DebugAttr::Loc {
                scope,
                file,
                line,
                column,
            } => {
                let filename = self.di_string(file)?;
                match self.debug.attr(scope, &self.strings)?.clone() {
                    DebugAttr::Subprogram {
                        file: f,
                        line: sp_line,
                        name,
                        linkage,
                        scope_line,
                        ..
                    } => {
                        let di_name = self.di_string(name)?;
                        let subprogram_file = self.di_string(f)?;
                        let linkage = self.di_string(linkage)?;
                        let base = super::debug_info::split_file_path(&filename).1;
                        // If the subprogram is exactly the one the writer
                        // derives from a bare file:line:col, restore that
                        // form. Any other subprogram means the original
                        // location carried an explicit scope.
                        let is_derived = linkage == sym_name
                            && subprogram_file == base
                            && sp_line == line
                            && scope_line == line;
                        if is_derived {
                            Ok((
                                Location::FileLineCol {
                                    filename,
                                    line,
                                    column,
                                },
                                Some(di_name),
                            ))
                        } else {
                            let scope = self.resolve_scope(scope)?;
                            Ok((
                                Location::DebugInfo(DebugInfoLoc {
                                    filename,
                                    line,
                                    column,
                                    scope,
                                }),
                                Some(di_name),
                            ))
                        }
                    }
                    _ => {
                        let scope = self.resolve_scope(scope)?;
                        Ok((
                            Location::DebugInfo(DebugInfoLoc {
                                filename,
                                line,
                                column,
                                scope,
                            }),
                            None,
                        ))
                    }
                }
            }
            DebugAttr::CallSite { callee, caller } => {
                let callee = self.resolve_location(callee)?;
                let caller = self.resolve_location(caller)?;
                Ok((
                    Location::CallSite {
                        callee: Box::new(callee),
                        caller: Box::new(caller),
                    },
                    None,
                ))
            }
            other => Err(Error::BytecodeRead(format!(
                "function debug attribute has unexpected kind {other:?}"
            ))),
        }
    }

    fn resolve_location(&mut self, id: u64) -> Result<Location> {
        if id == 0 {
            return Ok(Location::Unknown);
        }
        match self.debug.attr(id, &self.strings)?.clone() {
            DebugAttr::Loc {
                scope,
                file,
                line,
                column,
            } => {
                let scope = self.resolve_scope(scope)?;
                Ok(Location::DebugInfo(DebugInfoLoc {
                    filename: self.di_string(file)?,
                    line,
                    column,
                    scope,
                }))
            }
            DebugAttr::CallSite { callee, caller } => {
                let callee = self.resolve_location(callee)?;
                let caller = self.resolve_location(caller)?;
                Ok(Location::CallSite {
                    callee: Box::new(callee),
                    caller: Box::new(caller),
                })
            }
            other => Err(Error::BytecodeRead(format!(
                "debug attribute has unexpected location kind {other:?}"
            ))),
        }
    }

    fn resolve_scope(&mut self, id: u64) -> Result<DebugScope> {
        match self.debug.attr(id, &self.strings)?.clone() {
            DebugAttr::Subprogram {
                file,
                line,
                name,
                linkage,
                cu,
                scope_line,
                is_declaration,
            } => {
                let compile_unit = match self.debug.attr(cu, &self.strings)?.clone() {
                    DebugAttr::CompileUnit { file } => crate::ir::DICompileUnit {
                        file: self.di_file(file)?,
                    },
                    other => {
                        return Err(Error::BytecodeRead(format!(
                            "subprogram compile unit has unexpected kind {other:?}"
                        )))
                    }
                };
                Ok(DebugScope::Subprogram(DISubprogram {
                    file: self.di_file(file)?,
                    line,
                    name: self.di_string(name)?,
                    linkage_name: self.di_string(linkage)?,
                    compile_unit,
                    scope_line,
                    is_declaration,
                }))
            }
            DebugAttr::LexicalBlock {
                scope,
                file,
                line,
                column,
            } => Ok(DebugScope::LexicalBlock(DILexicalBlock {
                scope: Box::new(self.resolve_scope(scope)?),
                file: self.di_file(file)?,
                line,
                column,
            })),
            other => Err(Error::BytecodeRead(format!(
                "debug attribute has unexpected scope kind {other:?}"
            ))),
        }
    }

    fn di_string(&self, idx: u64) -> Result<String> {
        table_get(&self.strings, idx, "debug string")
    }

    fn di_file(&mut self, idx: u64) -> Result<DIFile> {
        match self.debug.attr(idx, &self.strings)?.clone() {
            DebugAttr::File { name, dir } => Ok(DIFile {
                name: self.di_string(name)?,
                directory: self.di_string(dir)?,
            }),
            other => Err(Error::BytecodeRead(format!(
                "debug file attribute has unexpected kind {other:?}"
            ))),
        }
    }

    // ------ Sections ------

    fn parse_global_section(&mut self, payload: Option<&[u8]>) -> Result<()> {
        let Some(data) = payload else {
            return Ok(());
        };
        let mut r = EncodingReader::new(data);
        let count = super::encoding::cap_count(r.read_varint()?, "global")?;
        self.module.globals.reserve(count);
        for _ in 0..count {
            let name = self.read_string_idx(&mut r, "global name")?;
            let element_type = self.read_type_idx(&mut r)?;
            let const_idx = super::encoding::cap_count(r.read_varint()?, "global constant")?;
            let alignment = r.read_varint()?;
            let (visibility, constant) = if self.v13_3() {
                let v = r.read_byte()?;
                let vis = match v {
                    0 => SymbolVisibility::Public,
                    1 => SymbolVisibility::Private,
                    v => {
                        return Err(Error::BytecodeRead(format!(
                            "invalid global visibility {v}"
                        )))
                    }
                };
                let c = r.read_varint()? != 0;
                (vis, c)
            } else {
                // Pre-13.3: the writer defaults to public, non-constant.
                (SymbolVisibility::Public, false)
            };
            let data = self.read_constant_entry(const_idx)?;
            // The global section does not carry the dense shape; derive it
            // from the element type (the writer only serializes data).
            let shape = match &element_type {
                Type::Tile(t) => t.shape.clone(),
                _ => vec![],
            };
            self.module.globals.push(crate::ir::Global {
                sym_name: name,
                value: DenseElements {
                    element_type,
                    shape,
                    data,
                },
                alignment,
                constant,
                symbol_visibility: visibility,
            });
        }
        Ok(())
    }

    fn parse_func_section(&mut self, r: &mut EncodingReader) -> Result<()> {
        let count = super::encoding::cap_count(r.read_varint()?, "function")?;
        self.module.functions.reserve(count);
        for _ in 0..count {
            self.parse_function(r)?;
        }
        Ok(())
    }

    fn parse_function(&mut self, r: &mut EncodingReader) -> Result<()> {
        let name = self.read_string_idx(r, "function name")?;
        let sig = self.read_type_idx(r)?;
        let flags = r.read_byte()?;
        let di_idx = r.read_varint()?;
        // 13.5 drops the hints flag (kernels always carry the attribute) and
        // adds argument attributes as a positional field.
        let is_kernel = flags & super::enums::FunctionFlag::KindKernel as u8 != 0;
        let has_hints = if self.version >= BytecodeVersion::V13_5 {
            is_kernel
        } else {
            flags & super::enums::FunctionFlag::HasOptimizationHints as u8 != 0
        };
        let hints = if has_hints {
            Some(read_self_contained_attribute(
                r,
                &self.strings,
                &self.types,
                &self.constants,
            )?)
        } else {
            None
        };
        let arg_attrs = if self.version >= BytecodeVersion::V13_5 {
            Some(read_self_contained_attribute(
                r,
                &self.strings,
                &self.types,
                &self.constants,
            )?)
        } else {
            None
        };
        let body_len = r.read_varint()? as usize;
        let body = r.read_bytes(body_len)?;

        // The writer numbers values from zero per function; do the same.
        self.value_map.clear();
        self.next_idx = 0;

        let func_type = match &sig {
            Type::Func(f) => f.clone(),
            other => {
                return Err(Error::BytecodeRead(format!(
                    "function signature must be a function type, got {other:?}"
                )))
            }
        };
        let (loc, di_name) = self.function_location(di_idx, &name)?;
        // If the file has no debug info for this function (or no Debug
        // section at all), every location degrades to Unknown.
        self.debug_cursor = if self.debug.func_ids(di_idx).is_some() {
            Some((di_idx, 1))
        } else {
            None
        };

        let mut body_reader = EncodingReader::new(body);
        // The function body encodes the entry block's operations directly;
        // the argument types come from the signature (the writer does not
        // repeat them).
        let block_id = self.module.alloc_block(Block {
            args: Vec::new(),
            ops: Vec::new(),
        });
        let mut args = Vec::with_capacity(func_type.inputs.len());
        for (i, ty) in func_type.inputs.iter().enumerate() {
            let v = self.module.alloc_value(
                ty.clone(),
                ValueProducer::BlockArg {
                    block: block_id,
                    arg_index: i as u32,
                },
            );
            self.value_map.insert(self.next_idx, v);
            self.next_idx += 1;
            args.push((v, ty.clone()));
        }
        let mut ops = Vec::new();
        while body_reader.remaining() > 0 {
            ops.push(self.parse_operation(&mut body_reader)?);
        }
        // No rollback: the writer keeps the entry block's values alive.
        let block = self.module.block_mut(block_id);
        block.args = args;
        block.ops = ops;
        let region_id = self.module.alloc_region(Region {
            blocks: vec![block_id],
        });

        // The function's debug ids must be exactly exhausted.
        if let Some((func_idx, pos)) = self.debug_cursor.take() {
            let ids = self.debug.func_ids(func_idx);
            if ids.is_some_and(|ids| pos != ids.len()) {
                return Err(Error::BytecodeRead(format!(
                    "function {func_idx} body consumed {pos} of {} debug ids",
                    ids.map(|i| i.len()).unwrap_or(0)
                )));
            }
        }

        // Rebuild the entry op. Attribute order matches the frontend's
        // construction ([sym_name, di_name, function_type, hints]) so that
        // the prescan string interning order is reproduced on re-write.
        let mut attributes = vec![("sym_name".to_owned(), Attribute::String(name.clone()))];
        // Restore the `di_name` attribute only when the original module
        // carried one for this function.
        if let Some(display) = &di_name {
            if display != &name {
                attributes.push(("di_name".to_owned(), Attribute::String(display.clone())));
            }
        }
        attributes.push(("function_type".to_owned(), Attribute::Type(sig)));
        // At 13.5 the writer synthesizes empty hints for a kernel that has
        // none, so an empty attribute carries no information to restore.
        let hints = hints.filter(|h| {
            self.version < BytecodeVersion::V13_5
                || !matches!(h, Attribute::OptimizationHints(oh) if oh.entries.is_empty())
        });
        if let Some(h) = hints {
            attributes.push(("optimization_hints".to_owned(), h));
        }
        if let Some(a) =
            arg_attrs.filter(|a| !matches!(a, Attribute::Array(items) if items.is_empty()))
        {
            attributes.push(("arg_attrs".to_owned(), a));
        }
        let entry_op = Operation {
            opcode: Opcode::Entry,
            operands: Vec::new(),
            result_types: Vec::new(),
            attributes,
            regions: vec![region_id],
            location: loc,
        };
        let entry_id = self.module.alloc_op(entry_op);
        self.module.functions.push(entry_id);
        Ok(())
    }

    // ------ Body parsing ------

    fn parse_operation(&mut self, r: &mut EncodingReader) -> Result<OpId> {
        let loc = self.take_op_location()?;
        let raw = r.read_varint()?;
        let opcode = Opcode::from_u16(raw as u16)
            .ok_or_else(|| Error::BytecodeRead(format!("unknown opcode 0x{raw:04X}")))?;
        let parsed = self.parse_op_body(r, opcode)?;

        let op_id = self.module.alloc_op(Operation {
            opcode,
            operands: Vec::new(),
            result_types: Vec::new(),
            attributes: Vec::new(),
            regions: Vec::new(),
            location: loc,
        });
        let op = self.module.op_mut(op_id);
        op.operands = parsed.operands;
        op.result_types = parsed.result_types;
        op.attributes = parsed.attributes;
        op.regions = parsed.regions;

        // Register results, mirroring the writer's post-op value map update.
        let result_types = self.module.op(op_id).result_types.clone();
        for (i, ty) in result_types.into_iter().enumerate() {
            let v = self.module.alloc_value(
                ty,
                ValueProducer::OpResult {
                    op: op_id,
                    result_index: i as u32,
                },
            );
            self.value_map.insert(self.next_idx, v);
            self.next_idx += 1;
        }
        Ok(op_id)
    }

    fn parse_region(&mut self, r: &mut EncodingReader) -> Result<RegionId> {
        let num_blocks = super::encoding::cap_count(r.read_varint()?, "block")?;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(self.parse_block(r)?);
        }
        Ok(self.module.alloc_region(Region { blocks }))
    }

    fn parse_block(&mut self, r: &mut EncodingReader) -> Result<BlockId> {
        let num_args = super::encoding::cap_count(r.read_varint()?, "block argument")?;
        let block_id = self.module.alloc_block(Block {
            args: Vec::new(),
            ops: Vec::new(),
        });
        // Block-scoped value numbering. Everything allocated from here on
        // is rolled back at block end.
        let saved_next_idx = self.next_idx;
        let mut args = Vec::with_capacity(num_args);
        for i in 0..num_args {
            let ty = self.read_type_idx(r)?;
            let v = self.module.alloc_value(
                ty.clone(),
                ValueProducer::BlockArg {
                    block: block_id,
                    arg_index: i as u32,
                },
            );
            self.value_map.insert(self.next_idx, v);
            self.next_idx += 1;
            args.push((v, ty));
        }
        let num_ops = super::encoding::cap_count(r.read_varint()?, "block operation")?;
        let mut ops = Vec::with_capacity(num_ops);
        for _ in 0..num_ops {
            ops.push(self.parse_operation(r)?);
        }
        self.value_map.retain(|idx, _| *idx < saved_next_idx);
        self.next_idx = saved_next_idx;
        let block = self.module.block_mut(block_id);
        block.args = args;
        block.ops = ops;
        Ok(block_id)
    }

    // ------ Per-op body dispatch (mirror of op_writer::write_op_body) ------

    fn parse_op_body(&mut self, r: &mut EncodingReader, opcode: Opcode) -> Result<ParsedOp> {
        use Opcode::*;
        match opcode {
            // ----- Simple: result types + operands (no size) -----
            AbsF
            | AbsI
            | AndI
            | Atan2
            | Bitcast
            | Broadcast
            | Ceil
            | Cos
            | CosH
            | FPowI
            | Floor
            | IntToPtr
            | Log
            | Log2
            | MakeGatherScatterView
            | MakeStridedView
            | MemoryFenceAliasTko
            | MmaFScaled
            | MulhiI
            | NegF
            | Offset
            | OrI
            | Pack
            | Pow
            | PtrToInt
            | PtrToPtr
            | RemF
            | Reshape
            | Select
            | Sin
            | SinH
            | Tan
            | Unpack
            | XOrI => {
                let result_types =
                    self.read_results_fixed(r, opcode.fixed_result_count().unwrap())?;
                let operands = self.read_operands_fixed(r, fixed_operand_count(opcode))?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: Vec::new(),
                    regions: Vec::new(),
                })
            }

            Exp => {
                let result_types = self.read_results_fixed(r, 1)?;
                let mut attributes = Vec::new();
                if self.v13_3() {
                    let v = r.read_varint()? as i64;
                    if v != 5 || self.name_in_string_table("rounding_mode") {
                        attributes.push(("rounding_mode".into(), Attribute::i32(v)));
                    }
                }
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            MmaF => {
                let result_types = self.read_results_fixed(r, 1)?;
                let mut attributes = Vec::new();
                if self.v13_3() {
                    let flags = r.read_varint()?;
                    if flags & 1 != 0 {
                        attributes.push(("fast_acc".into(), Attribute::Bool(true)));
                    }
                }
                let operands = self.read_operands_fixed(r, 3)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            Alloca => {
                let result_types = self.read_results_fixed(r, 1)?;
                let flags = r.read_varint()?;
                let num_elem = read_inline_int(r)?;
                let alignment = read_inline_int(r)?;
                let mut attributes = vec![
                    ("num_elem".into(), num_elem),
                    ("alignment".into(), alignment),
                ];
                if flags & 1 != 0 {
                    attributes.push(("global".into(), Attribute::Bool(true)));
                }
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes,
                    regions: Vec::new(),
                })
            }

            NegI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let v = r.read_varint()? as i64;
                let attributes = if v != 0 || self.name_in_string_table("overflow") {
                    vec![("overflow".into(), Attribute::i32(v))]
                } else {
                    Vec::new()
                };
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            TanH => {
                let result_types = self.read_results_fixed(r, 1)?;
                let v = r.read_varint()? as i64;
                let attributes = if v != 5 || self.name_in_string_table("rounding_mode") {
                    vec![("rounding_mode".into(), Attribute::i32(v))]
                } else {
                    Vec::new()
                };
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            // ----- No operands, just result types -----
            Iota | GetNumTileBlocks | GetTileBlockId | MakeToken => {
                let result_types =
                    self.read_results_fixed(r, opcode.fixed_result_count().unwrap())?;
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes: Vec::new(),
                    regions: Vec::new(),
                })
            }

            // ----- Required attributes + fixed operands -----
            AddI | MulI | SubI | ShLI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let overflow = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("overflow".into(), overflow)],
                    regions: Vec::new(),
                })
            }
            TruncI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let overflow = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("overflow".into(), overflow)],
                    regions: Vec::new(),
                })
            }
            ShRI | ExtI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let signedness = read_inline_int(r)?;
                let count = if opcode == ShRI { 2 } else { 1 };
                let operands = self.read_operands_fixed(r, count)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("signedness".into(), signedness)],
                    regions: Vec::new(),
                })
            }
            Cat => {
                let result_types = self.read_results_fixed(r, 1)?;
                let dim = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("dim".into(), dim)],
                    regions: Vec::new(),
                })
            }
            Permute => {
                let result_types = self.read_results_fixed(r, 1)?;
                let permutation = read_inline_dense_i32_array(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("permutation".into(), permutation)],
                    regions: Vec::new(),
                })
            }
            Assert => {
                let result_types = self.read_results_fixed(r, 0)?;
                let message = self.read_string_idx(r, "assert message")?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("message".into(), Attribute::String(message))],
                    regions: Vec::new(),
                })
            }
            Assume => {
                let result_types = self.read_results_fixed(r, 1)?;
                let predicate =
                    read_self_contained_attribute(r, &self.strings, &self.types, &self.constants)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("predicate".into(), predicate)],
                    regions: Vec::new(),
                })
            }
            CmpF => {
                let result_types = self.read_results_fixed(r, 1)?;
                let comparison_predicate = read_inline_int(r)?;
                let comparison_ordering = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![
                        ("comparison_predicate".into(), comparison_predicate),
                        ("comparison_ordering".into(), comparison_ordering),
                    ],
                    regions: Vec::new(),
                })
            }
            CmpI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let comparison_predicate = read_inline_int(r)?;
                let signedness = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![
                        ("comparison_predicate".into(), comparison_predicate),
                        ("signedness".into(), signedness),
                    ],
                    regions: Vec::new(),
                })
            }
            Constant => {
                let result_types = self.read_results_fixed(r, 1)?;
                let const_idx = super::encoding::cap_count(r.read_varint()?, "constant")?;
                let data = self.read_constant_entry(const_idx)?;
                // The constant pool carries only the raw bytes.
                // The element type is the op's result type (see the writer's
                // inline DenseElements for comparison).
                let element_type = result_types[0].clone();
                let shape = match &element_type {
                    Type::Tile(t) => t.shape.clone(),
                    _ => vec![],
                };
                let attributes = vec![(
                    "value".into(),
                    Attribute::DenseElements(DenseElements {
                        element_type,
                        shape,
                        data,
                    }),
                )];
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes,
                    regions: Vec::new(),
                })
            }
            DivI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let signedness = read_inline_int(r)?;
                let rounding = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![
                        ("signedness".into(), signedness),
                        ("rounding".into(), rounding),
                    ],
                    regions: Vec::new(),
                })
            }
            FToF => {
                let result_types = self.read_results_fixed(r, 1)?;
                let rounding_mode = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("rounding_mode".into(), rounding_mode)],
                    regions: Vec::new(),
                })
            }
            FToI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let saturating =
                    self.version >= BytecodeVersion::V13_4 && r.read_varint()? & 1 != 0;
                let signedness = read_inline_int(r)?;
                let rounding_mode = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                let mut attributes = vec![
                    ("signedness".into(), signedness),
                    ("rounding_mode".into(), rounding_mode),
                ];
                if saturating {
                    attributes.push(("saturating".into(), Attribute::Bool(true)));
                }
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            IToF => {
                let result_types = self.read_results_fixed(r, 1)?;
                let signedness = read_inline_int(r)?;
                let rounding_mode = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![
                        ("signedness".into(), signedness),
                        ("rounding_mode".into(), rounding_mode),
                    ],
                    regions: Vec::new(),
                })
            }
            MaxI | MinI | RemI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let signedness = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 2)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![("signedness".into(), signedness)],
                    regions: Vec::new(),
                })
            }
            MmaI => {
                let result_types = self.read_results_fixed(r, 1)?;
                let signedness_lhs = read_inline_int(r)?;
                let signedness_rhs = read_inline_int(r)?;
                let operands = self.read_operands_fixed(r, 3)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: vec![
                        ("signedness_lhs".into(), signedness_lhs),
                        ("signedness_rhs".into(), signedness_rhs),
                    ],
                    regions: Vec::new(),
                })
            }
            MakePartitionView => {
                let result_types = self.read_results_fixed(r, 1)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: Vec::new(),
                    regions: Vec::new(),
                })
            }
            GetGlobal => {
                let result_types = self.read_results_fixed(r, 1)?;
                let name = self.read_string_idx(r, "get_global name")?;
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes: vec![("name".into(), Attribute::String(name))],
                    regions: Vec::new(),
                })
            }

            // ----- Variadic result count + operands -----
            JoinTokens | Break | Continue | Return | Yield | Extract | For | Insert | Loop
            | Reduce | Scan => {
                let result_types = self.read_results_varint(r)?;
                let mut attributes = Vec::new();
                match opcode {
                    Reduce => {
                        let dim = read_inline_int(r)?;
                        let identities =
                            read_inline_array(r, &self.strings, &self.types, &self.constants)?;
                        attributes = vec![("dim".into(), dim), ("identities".into(), identities)];
                    }
                    Scan => {
                        let dim = read_inline_int(r)?;
                        let reverse = read_inline_bool(r)?;
                        let identities =
                            read_inline_array(r, &self.strings, &self.types, &self.constants)?;
                        attributes = vec![
                            ("dim".into(), dim),
                            ("reverse".into(), reverse),
                            ("identities".into(), identities),
                        ];
                    }
                    For => {
                        let flags = r.read_varint()?;
                        if flags & 1 != 0 {
                            attributes = vec![("unsigned_cmp".into(), Attribute::i32(1))];
                        }
                    }
                    _ => {}
                }
                let operands = self.read_operands_sized(r)?;
                let mut regions = Vec::new();
                if matches!(opcode, For | Loop | Reduce | Scan) {
                    // write_regions: region count, then one region each.
                    let num_regions = super::encoding::cap_count(r.read_varint()?, "region")?;
                    for _ in 0..num_regions {
                        regions.push(self.parse_region(r)?);
                    }
                }
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions,
                })
            }

            // ----- GDC PDL: result types + optional token operand -----
            GdcLaunchDependentsTko | GdcWaitTko => {
                let result_types =
                    self.read_results_fixed(r, opcode.fixed_result_count().unwrap())?;
                let operands = if r.read_varint()? != 0 {
                    vec![self.read_operand(r, "operand")?]
                } else {
                    Vec::new()
                };
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: Vec::new(),
                    regions: Vec::new(),
                })
            }

            // ----- MakeTensorView: result count + AttrSizedOperandSegments -----
            MakeTensorView => {
                let result_types = self.read_results_varint(r)?;
                let base = self.read_operand_group(r, 1)?;
                let dynamic_shape = self.read_variadic_operand_group(r)?;
                let dynamic_strides = self.read_variadic_operand_group(r)?;
                let segment_sizes = segment_sizes_attribute(&[
                    1,
                    dynamic_shape.len() as i64,
                    dynamic_strides.len() as i64,
                ]);
                let mut operands = base;
                operands.extend(dynamic_shape);
                operands.extend(dynamic_strides);
                let attributes = vec![("operandSegmentSizes".into(), segment_sizes)];
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            // ----- Flags + optional attributes -----
            AddF | DivF | MulF | SubF | Fma | Sqrt => {
                let result_types = self.read_results_fixed(r, 1)?;
                let flags = r.read_varint()?;
                let rounding_mode = read_inline_int(r)?;
                let count = match opcode {
                    Fma => 3,
                    Sqrt => 1,
                    _ => 2,
                };
                let operands = self.read_operands_fixed(r, count)?;
                let mut attributes = vec![("rounding_mode".into(), rounding_mode)];
                if flags & 1 != 0 {
                    attributes.push(("flush_to_zero".into(), Attribute::Bool(true)));
                }
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            Exp2 | Rsqrt => {
                let result_types = self.read_results_fixed(r, 1)?;
                let flags = r.read_varint()?;
                let operands = self.read_operands_fixed(r, 1)?;
                let attributes = if flags & 1 != 0 {
                    vec![("flush_to_zero".into(), Attribute::Bool(true))]
                } else {
                    Vec::new()
                };
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            MaxF | MinF => {
                let result_types = self.read_results_fixed(r, 1)?;
                let flags = r.read_varint()?;
                let operands = self.read_operands_fixed(r, 2)?;
                let mut attributes = Vec::new();
                if flags & 1 != 0 {
                    attributes.push(("propagate_nan".into(), Attribute::Bool(true)));
                }
                if flags & 2 != 0 {
                    attributes.push(("flush_to_zero".into(), Attribute::Bool(true)));
                }
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            // ----- Variadic results, no attrs -----
            GetIndexSpaceShape | GetTensorShape => {
                let result_types = self.read_results_varint(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: Vec::new(),
                    regions: Vec::new(),
                })
            }

            If => {
                let result_types = self.read_results_varint(r)?;
                let operands = self.read_operands_fixed(r, 1)?;
                let num_regions = super::encoding::cap_count(r.read_varint()?, "region")?;
                let mut regions = Vec::with_capacity(num_regions);
                for _ in 0..num_regions {
                    regions.push(self.parse_region(r)?);
                }
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes: Vec::new(),
                    regions,
                })
            }

            // ----- Global / Module (rarely in bodies, kept in sync with the writer) -----
            Global => {
                let result_types = self.read_results_fixed(r, 0)?;
                let mut attributes = Vec::new();
                if self.v13_3() {
                    let flags = r.read_varint()?;
                    if flags & 1 != 0 {
                        attributes.push(("constant".into(), Attribute::Bool(true)));
                    }
                }
                let sym_name = self.read_string_idx(r, "global sym_name")?;
                let const_idx = super::encoding::cap_count(r.read_varint()?, "global value")?;
                let data = self.read_constant_entry(const_idx)?;
                // The inline form serializes data only. Element type is not
                // recoverable from the body and is not re-serialized.
                attributes.push(("sym_name".into(), Attribute::String(sym_name)));
                attributes.push((
                    "value".into(),
                    Attribute::DenseElements(DenseElements {
                        element_type: Type::Token,
                        shape: vec![],
                        data,
                    }),
                ));
                let alignment = read_inline_int(r)?;
                attributes.push(("alignment".into(), alignment));
                if self.v13_3() {
                    let v = r.read_varint()? as i64;
                    if v != 0 || self.name_in_string_table("symbol_visibility") {
                        attributes.push(("symbol_visibility".into(), Attribute::i32(v)));
                    }
                }
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes,
                    regions: Vec::new(),
                })
            }

            Module => {
                let result_types = self.read_results_fixed(r, 0)?;
                let mut attributes = Vec::new();
                if self.v13_3() {
                    let flags = r.read_varint()?;
                    let has_producer = flags & 1 != 0;
                    let sym_name = self.read_string_idx(r, "module sym_name")?;
                    attributes.push(("sym_name".into(), Attribute::String(sym_name)));
                    if has_producer {
                        let producer = self.read_string_idx(r, "module producer")?;
                        attributes.push(("producer".into(), Attribute::String(producer)));
                    }
                } else {
                    let sym_name = self.read_string_idx(r, "module sym_name")?;
                    attributes.push(("sym_name".into(), Attribute::String(sym_name)));
                }
                let mut regions = Vec::new();
                let num_regions = super::encoding::cap_count(r.read_varint()?, "region")?;
                for _ in 0..num_regions {
                    regions.push(self.parse_region(r)?);
                }
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes,
                    regions,
                })
            }

            // ----- Print -----
            Print => {
                let result_types = self.read_results_varint(r)?;
                let flags = r.read_varint()?;
                let str_attr = self.read_string_idx(r, "print str")?;
                let args = self.read_variadic_operand_group(r)?;
                let has_token = flags & 1 != 0;
                let token = if has_token {
                    Some(self.read_operand(r, "print token")?)
                } else {
                    None
                };
                let segment_sizes =
                    segment_sizes_attribute(&[args.len() as i64, i64::from(has_token)]);
                let mut operands = args;
                if let Some(t) = token {
                    operands.push(t);
                }
                let attributes = vec![
                    ("str".into(), Attribute::String(str_attr)),
                    ("operandSegmentSizes".into(), segment_sizes),
                ];
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            // ----- AttrSizedOperandSegments + flags -----
            AtomicCAS => {
                let result_types = self.read_results_fixed(r, 2)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = read_inline_int(r)?;
                let pointers = self.read_operand_group(r, 1)?;
                let cmp = self.read_operand_group(r, 1)?;
                let val = self.read_operand_group(r, 1)?;
                let mask = if flags & 1 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let token = if flags & 2 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, 1, 1, mask.len() as i64, token.len() as i64]);
                let mut operands = pointers;
                operands.extend(cmp);
                operands.extend(val);
                operands.extend(mask);
                operands.extend(token);
                let attributes = vec![
                    (
                        "memory_ordering_semantics".into(),
                        memory_ordering_semantics,
                    ),
                    ("memory_scope".into(), memory_scope),
                    ("operandSegmentSizes".into(), segment_sizes),
                ];
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            AtomicRMW => {
                let result_types = self.read_results_fixed(r, 2)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = read_inline_int(r)?;
                let mode = read_inline_int(r)?;
                let pointers = self.read_operand_group(r, 1)?;
                let value = self.read_operand_group(r, 1)?;
                let mask = if flags & 1 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let token = if flags & 2 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, 1, mask.len() as i64, token.len() as i64]);
                let mut operands = pointers;
                operands.extend(value);
                operands.extend(mask);
                operands.extend(token);
                let attributes = vec![
                    (
                        "memory_ordering_semantics".into(),
                        memory_ordering_semantics,
                    ),
                    ("memory_scope".into(), memory_scope),
                    ("mode".into(), mode),
                    ("operandSegmentSizes".into(), segment_sizes),
                ];
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            AtomicRedViewTko => {
                let result_types = self.read_results_varint(r)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = read_inline_int(r)?;
                let mode = read_inline_int(r)?;
                // This op joined the inbounds vector at 13.5, not 13.4.
                let inbounds = read_inbounds(r, self.version >= BytecodeVersion::V13_5)?;
                let view = self.read_operand_group(r, 1)?;
                let index = self.read_variadic_operand_group(r)?;
                let value = self.read_operand_group(r, 1)?;
                let token = if flags & 1 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, index.len() as i64, 1, token.len() as i64]);
                let mut operands = view;
                operands.extend(index);
                operands.extend(value);
                operands.extend(token);
                let mut attributes = vec![
                    (
                        "memory_ordering_semantics".into(),
                        memory_ordering_semantics,
                    ),
                    ("memory_scope".into(), memory_scope),
                    ("mode".into(), mode),
                ];
                if inbounds.iter().any(|&v| v) {
                    attributes.push((
                        "inbounds".into(),
                        Attribute::Array(inbounds.into_iter().map(Attribute::Bool).collect()),
                    ));
                }
                attributes.push(("operandSegmentSizes".into(), segment_sizes));
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            LoadPtrTko => {
                let result_types = self.read_results_fixed(r, 2)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = if flags & 1 != 0 {
                    Some(read_inline_int(r)?)
                } else {
                    None
                };
                let hints = if flags & 2 != 0 {
                    Some(read_inline_optimization_hints(
                        r,
                        &self.strings,
                        &self.types,
                        &self.constants,
                    )?)
                } else {
                    None
                };
                let source = self.read_operand_group(r, 1)?;
                let mask = if flags & 4 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let padding = if flags & 8 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let token = if flags & 16 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes = segment_sizes_attribute(&[
                    1,
                    mask.len() as i64,
                    padding.len() as i64,
                    token.len() as i64,
                ]);
                let mut operands = source;
                operands.extend(mask);
                operands.extend(padding);
                operands.extend(token);
                let mut attributes = vec![(
                    "memory_ordering_semantics".into(),
                    memory_ordering_semantics,
                )];
                if let Some(ms) = memory_scope {
                    attributes.push(("memory_scope".into(), ms));
                }
                if let Some(h) = hints {
                    attributes.push(("optimization_hints".into(), h));
                }
                attributes.push(("operandSegmentSizes".into(), segment_sizes));
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            LoadViewTko => {
                let result_types = self.read_results_varint(r)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = if flags & 1 != 0 {
                    Some(read_inline_int(r)?)
                } else {
                    None
                };
                let hints = if flags & 2 != 0 {
                    Some(read_inline_optimization_hints(
                        r,
                        &self.strings,
                        &self.types,
                        &self.constants,
                    )?)
                } else {
                    None
                };
                let inbounds = read_inbounds(r, self.version >= BytecodeVersion::V13_4)?;
                let view = self.read_operand_group(r, 1)?;
                let index = self.read_variadic_operand_group(r)?;
                let token = if flags & 4 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, index.len() as i64, token.len() as i64]);
                let mut operands = view;
                operands.extend(index);
                operands.extend(token);
                // Frontend puts optimization_hints first in the attribute list.
                let mut attributes = Vec::new();
                if let Some(h) = hints {
                    attributes.push(("optimization_hints".into(), h));
                }
                attributes.push((
                    "memory_ordering_semantics".into(),
                    memory_ordering_semantics,
                ));
                if let Some(ms) = memory_scope {
                    attributes.push(("memory_scope".into(), ms));
                }
                if inbounds.iter().any(|&b| b) {
                    attributes.push((
                        "inbounds".into(),
                        Attribute::Array(inbounds.into_iter().map(Attribute::Bool).collect()),
                    ));
                }
                attributes.push(("operandSegmentSizes".into(), segment_sizes));
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            StorePtrTko => {
                let result_types = self.read_results_fixed(r, 1)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = if flags & 1 != 0 {
                    Some(read_inline_int(r)?)
                } else {
                    None
                };
                let hints = if flags & 2 != 0 {
                    Some(read_inline_optimization_hints(
                        r,
                        &self.strings,
                        &self.types,
                        &self.constants,
                    )?)
                } else {
                    None
                };
                let destination = self.read_operand_group(r, 1)?;
                let value = self.read_operand_group(r, 1)?;
                let mask = if flags & 4 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let token = if flags & 8 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, 1, mask.len() as i64, token.len() as i64]);
                let mut operands = destination;
                operands.extend(value);
                operands.extend(mask);
                operands.extend(token);
                let mut attributes = vec![(
                    "memory_ordering_semantics".into(),
                    memory_ordering_semantics,
                )];
                if let Some(ms) = memory_scope {
                    attributes.push(("memory_scope".into(), ms));
                }
                if let Some(h) = hints {
                    attributes.push(("optimization_hints".into(), h));
                }
                attributes.push(("operandSegmentSizes".into(), segment_sizes));
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }
            StoreViewTko => {
                let result_types = self.read_results_varint(r)?;
                let flags = r.read_varint()?;
                let memory_ordering_semantics = read_inline_int(r)?;
                let memory_scope = if flags & 1 != 0 {
                    Some(read_inline_int(r)?)
                } else {
                    None
                };
                let hints = if flags & 2 != 0 {
                    Some(read_inline_optimization_hints(
                        r,
                        &self.strings,
                        &self.types,
                        &self.constants,
                    )?)
                } else {
                    None
                };
                let inbounds = read_inbounds(r, self.version >= BytecodeVersion::V13_4)?;
                let tile = self.read_operand_group(r, 1)?;
                let view = self.read_operand_group(r, 1)?;
                let index = self.read_variadic_operand_group(r)?;
                let token = if flags & 4 != 0 {
                    self.read_operand_group(r, 1)?
                } else {
                    Vec::new()
                };
                let segment_sizes =
                    segment_sizes_attribute(&[1, 1, index.len() as i64, token.len() as i64]);
                let mut operands = tile;
                operands.extend(view);
                operands.extend(index);
                operands.extend(token);
                // Frontend puts optimization_hints first in the attribute list.
                let mut attributes = Vec::new();
                if let Some(h) = hints {
                    attributes.push(("optimization_hints".into(), h));
                }
                attributes.push((
                    "memory_ordering_semantics".into(),
                    memory_ordering_semantics,
                ));
                if let Some(ms) = memory_scope {
                    attributes.push(("memory_scope".into(), ms));
                }
                if inbounds.iter().any(|&b| b) {
                    attributes.push((
                        "inbounds".into(),
                        Attribute::Array(inbounds.into_iter().map(Attribute::Bool).collect()),
                    ));
                }
                attributes.push(("operandSegmentSizes".into(), segment_sizes));
                Ok(ParsedOp {
                    result_types,
                    operands,
                    attributes,
                    regions: Vec::new(),
                })
            }

            // ----- Entry (only reaches here for Entry ops nested in bodies) -----
            Entry => {
                let result_types = self.read_results_fixed(r, 0)?;
                let flags = r.read_varint()?;
                let sym_name = self.read_string_idx(r, "entry sym_name")?;
                let function_type = self.read_type_idx(r)?;
                let mut attributes = vec![
                    ("sym_name".into(), Attribute::String(sym_name)),
                    ("function_type".into(), Attribute::Type(function_type)),
                ];
                if flags & 1 != 0 {
                    let arg_attrs =
                        read_inline_dict(r, &self.strings, &self.types, &self.constants)?;
                    attributes.push(("arg_attrs".into(), arg_attrs));
                }
                if flags & 2 != 0 {
                    let res_attrs =
                        read_inline_dict(r, &self.strings, &self.types, &self.constants)?;
                    attributes.push(("res_attrs".into(), res_attrs));
                }
                if flags & 4 != 0 {
                    let hints = read_inline_optimization_hints(
                        r,
                        &self.strings,
                        &self.types,
                        &self.constants,
                    )?;
                    attributes.push(("optimization_hints".into(), hints));
                }
                let regions = vec![self.parse_region(r)?];
                Ok(ParsedOp {
                    result_types,
                    operands: Vec::new(),
                    attributes,
                    regions,
                })
            }

            // All opcodes are covered above; keep the fallback so that
            // newly added opcodes fail with a clear error.
            #[allow(unreachable_patterns)]
            _ => Err(Error::BytecodeRead(format!(
                "unsupported operation {opcode:?} in bytecode reader"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-op result container
// ---------------------------------------------------------------------------

struct ParsedOp {
    operands: Vec<Value>,
    result_types: Vec<Type>,
    attributes: Vec<(String, Attribute)>,
    regions: Vec<RegionId>,
}

// ---------------------------------------------------------------------------
// Fixed operand counts
// ---------------------------------------------------------------------------

/// Fixed operand counts from the ODS definitions in `Ops.td` (the same
/// source the reference reader's generated parsers derive their arity
/// from). The writer does not use the size prefix, so the count is only
/// known from the opcode.
fn fixed_operand_count(opcode: Opcode) -> usize {
    use Opcode::*;
    match opcode {
        AbsF
        | AbsI
        | Assume
        | Assert
        | Bitcast
        | Broadcast
        | Ceil
        | Cos
        | CosH
        | ExtI
        | Floor
        | FToF
        | FToI
        | GetIndexSpaceShape
        | GetTensorShape
        | If
        | IntToPtr
        | Log
        | Log2
        | MakeGatherScatterView
        | MakePartitionView
        | MakeStridedView
        | MemoryFenceAliasTko
        | NegF
        | NegI
        | Pack
        | Permute
        | PtrToInt
        | PtrToPtr
        | Reshape
        | Rsqrt
        | ShRI
        | Sin
        | SinH
        | Sqrt
        | Tan
        | TanH
        | TruncI
        | Unpack => 1,
        AddF | AddI | AndI | Atan2 | Cat | CmpF | CmpI | DivF | DivI | FPowI | MaxF | MaxI
        | MinF | MinI | MulF | MulI | MulhiI | Offset | OrI | Pow | RemF | RemI | ShLI | SubF
        | SubI | XOrI => 2,
        Fma | MmaI | Select => 3,
        MmaF => 3,
        MmaFScaled => 5,
        // No operands (explicit so fallback below is not reached by mistake).
        Alloca | Constant | GetGlobal | GetNumTileBlocks | GetTileBlockId | Global | Iota
        | MakeToken | Module | Entry => 0,
        // Everything else is sized or grouped and is handled by its own arm.
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Attribute readers
// ---------------------------------------------------------------------------

/// Read the inbounds vector (rank + one byte per index dimension) that
/// the writer emits before view operands according to version.
/// Functionality added in v13.4
fn read_inbounds(r: &mut EncodingReader, v13_4: bool) -> Result<Vec<bool>> {
    if !v13_4 {
        return Ok(Vec::new());
    }
    let rank = super::encoding::cap_count(r.read_varint()?, "inbounds")?;
    let mut values = Vec::with_capacity(rank);
    for _ in 0..rank {
        values.push(r.read_byte()? != 0);
    }
    Ok(values)
}

/// Read an inline integer attribute
/// (`write_attr_value_inline` drops the type for integers).
fn read_inline_int(r: &mut EncodingReader) -> Result<Attribute> {
    let v = r.read_varint()? as i64;
    Ok(Attribute::i32(v))
}

fn read_inline_bool(r: &mut EncodingReader) -> Result<Attribute> {
    Ok(Attribute::Bool(r.read_byte()? != 0))
}

fn read_inline_dense_i32_array(r: &mut EncodingReader) -> Result<Attribute> {
    Ok(Attribute::DenseI32Array(r.read_le_var_size_i32()?))
}

/// Inline Array: element count + self-contained elements (the writer's
/// untagged array form).
fn read_inline_array(
    r: &mut EncodingReader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Attribute> {
    let count = super::encoding::cap_count(r.read_varint()?, "array element")?;
    let mut elems = Vec::with_capacity(count);
    for _ in 0..count {
        elems.push(read_self_contained_attribute(r, strings, types, constants)?);
    }
    Ok(Attribute::Array(elems))
}

/// Inline OptimizationHints: arch count + (arch, self-contained dictionary).
fn read_inline_optimization_hints(
    r: &mut EncodingReader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Attribute> {
    let count = super::encoding::cap_count(r.read_varint()?, "optimization hints arch")?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let arch = read_string_indexed(r, strings)?;
        let Attribute::Dictionary(hints) =
            read_self_contained_attribute(r, strings, types, constants)?
        else {
            return Err(Error::BytecodeRead(
                "optimization hints arch entry must be a dictionary".into(),
            ));
        };
        entries.push((arch, hints));
    }
    Ok(Attribute::OptimizationHints(crate::ir::OptimizationHints {
        entries,
    }))
}

/// Inline Dictionary: entry count + (key, self-contained value).
fn read_inline_dict(
    r: &mut EncodingReader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Attribute> {
    let count = super::encoding::cap_count(r.read_varint()?, "dictionary entry")?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let key = read_string_indexed(r, strings)?;
        let val = read_self_contained_attribute(r, strings, types, constants)?;
        entries.push((key, val));
    }
    Ok(Attribute::Dictionary(entries))
}

fn read_string_indexed(r: &mut EncodingReader, strings: &[String]) -> Result<String> {
    table_get(strings, r.read_varint()?, "string")
}

/// Read a self-contained attribute: tag varint + payload.
pub(crate) fn read_self_contained_attribute(
    r: &mut EncodingReader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Attribute> {
    let tag = r.read_varint()? as u8;
    match tag {
        t if t == AttributeTag::Integer as u8 => {
            let ty = {
                let idx = super::encoding::cap_count(r.read_varint()?, "attribute type")?;
                types.get(idx).cloned().ok_or_else(|| {
                    Error::BytecodeRead(format!("attribute type index {idx} out of range"))
                })?
            };
            let v = r.read_varint()? as i64;
            Ok(Attribute::Integer(v, ty))
        }
        t if t == AttributeTag::Float as u8 => {
            let idx = super::encoding::cap_count(r.read_varint()?, "attribute type")?;
            let ty = types.get(idx).cloned().ok_or_else(|| {
                Error::BytecodeRead(format!("attribute type index {idx} out of range"))
            })?;
            let v = r.read_ap_float(&ty)?;
            Ok(Attribute::Float(v, ty))
        }
        t if t == AttributeTag::Bool as u8 => Ok(Attribute::Bool(r.read_byte()? != 0)),
        t if t == AttributeTag::Type as u8 => {
            let idx = super::encoding::cap_count(r.read_varint()?, "attribute type")?;
            Ok(Attribute::Type(types.get(idx).cloned().ok_or_else(
                || Error::BytecodeRead(format!("attribute type index {idx} out of range")),
            )?))
        }
        t if t == AttributeTag::String as u8 => {
            Ok(Attribute::String(read_string_indexed(r, strings)?))
        }
        t if t == AttributeTag::Array as u8 => read_inline_array(r, strings, types, constants),
        t if t == AttributeTag::DenseElements as u8 => {
            let idx = super::encoding::cap_count(r.read_varint()?, "attribute type")?;
            let element_type = types.get(idx).cloned().ok_or_else(|| {
                Error::BytecodeRead(format!("dense elements type index {idx} out of range"))
            })?;
            let const_idx =
                super::encoding::cap_count(r.read_varint()?, "dense elements constant")?;
            let entry = constants.get(const_idx).ok_or_else(|| {
                Error::BytecodeRead(format!(
                    "dense elements constant index {const_idx} out of range"
                ))
            })?;
            let mut cr = EncodingReader::new(entry);
            let len = super::encoding::cap_count(cr.read_varint()?, "dense elements length")?;
            let data = cr.read_bytes(len)?.to_vec();
            let shape = match &element_type {
                Type::Tile(t) => t.shape.clone(),
                _ => vec![],
            };
            Ok(Attribute::DenseElements(DenseElements {
                element_type,
                shape,
                data,
            }))
        }
        t if t == AttributeTag::DivBy as u8 => {
            let divisor = r.read_varint()?;
            let flags = r.read_byte()?;
            let every = if flags & 1 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            let along = if flags & 2 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            Ok(Attribute::DivBy(crate::ir::DivBy {
                divisor,
                every,
                along,
            }))
        }
        t if t == AttributeTag::SameElements as u8 => {
            Ok(Attribute::SameElements(crate::ir::SameElements {
                values: r.read_le_var_size_i64()?,
            }))
        }
        t if t == AttributeTag::Dictionary as u8 => read_inline_dict(r, strings, types, constants),
        t if t == AttributeTag::OptimizationHints as u8 => {
            read_inline_optimization_hints(r, strings, types, constants)
        }
        t if t == AttributeTag::Bounded as u8 => {
            let flags = r.read_byte()?;
            let lb = if flags & 1 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            let ub = if flags & 2 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            Ok(Attribute::Bounded(crate::ir::Bounded { lb, ub }))
        }
        t => Err(Error::BytecodeRead(format!(
            "unknown self-contained attribute tag {t}"
        ))),
    }
}

/// Build an `operandSegmentSizes` attribute from parsed group sizes.
///
/// The writer does not serialize this, so we have to derive it from the
/// structure. Without this a roundtrip would get one operand per group.
fn segment_sizes_attribute(sizes: &[i64]) -> Attribute {
    Attribute::Array(sizes.iter().map(|&s| Attribute::i32(s)).collect())
}

/// Index table lookup (strings, types, ...) with cap.
fn table_get<T: Clone>(table: &[T], idx: u64, what: &str) -> Result<T> {
    let idx = super::encoding::cap_count(idx, what)?;
    table
        .get(idx)
        .cloned()
        .ok_or_else(|| Error::BytecodeRead(format!("{what} index {idx} out of range")))
}
