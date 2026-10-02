/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Top-level bytecode writer — orchestrates header, section, and
//! operation serialization into a byte buffer or file.
//!
//! Ported from the top-level `writeBytecode` function and its helper
//! managers in `BytecodeWriter.cpp`.

use std::collections::HashMap;
use std::io::Write;

use super::encoding::{patch_u32, patch_u64, EncodingWriter};
use super::enums::{AttributeTag, BytecodeVersion, FunctionFlag, Section, TypeTag, MAGIC};
use super::opcode::Opcode;
use crate::ir::SymbolVisibility;
use crate::ir::{
    Attribute, BlockId, Module, OpId, RegionId, TileElementType, Type, Value, ValueProducer,
};
use crate::requirements::{target_requirement, Feature};
use crate::{Error, Result};

// =========================================================================
// Public entry point
// =========================================================================

/// Serialize a [`Module`] into Tile IR bytecode.
///
/// Returns the bytecode as a `Vec<u8>`.
pub fn write_bytecode(module: &Module) -> Result<Vec<u8>> {
    write_bytecode_version(module, BytecodeVersion::CURRENT)
}

/// Serialize a [`Module`] with a specific bytecode version.
pub fn write_bytecode_version(module: &Module, version: BytecodeVersion) -> Result<Vec<u8>> {
    if version < BytecodeVersion::MIN_SUPPORTED {
        return Err(Error::BytecodeWrite(format!(
            "unsupported version {version}, minimum is {}",
            BytecodeVersion::MIN_SUPPORTED
        )));
    }

    let mut out = Vec::new();

    if !BytecodeVersion::SUPPORTED.contains(&version) {
        return Err(Error::BytecodeWrite(format!(
            "unsupported bytecode version {version}; supported versions: {}",
            BytecodeVersion::SUPPORTED.map(|v| v.to_string()).join(", ")
        )));
    }

    // Initialize writer context with all managers.
    let mut ctx = WriterCtx {
        module,
        value_map: HashMap::new(),
        next_idx: 0,
        strings: StringManager::new(),
        types: TypeManager::new(),
        constants: ConstantManager::new(),
        debug: DebugInfoCollector::new(),
        version,
    };

    // Pre-scan: register all types, strings, constants used by globals and functions.
    if let Some(producer) = &module.producer {
        require_version("module.producer", Feature::Producer.since(), version)?;
        ctx.strings.get_or_insert(producer);
    }
    for global in &module.globals {
        ctx.strings.get_or_insert(&global.sym_name);
        ctx.types.get_or_insert(&global.value.element_type);
    }
    for &func_op in &module.functions {
        check_region_versions(module, func_op, version, false, false)?;
        prescan_function(
            module,
            func_op,
            &mut ctx.strings,
            &mut ctx.types,
            &mut ctx.constants,
        )?;
    }

    // 1. Header
    write_header(&mut out, version);

    // 2. Global section
    write_global_section(&mut out, &mut ctx)?;

    // 3. Function table section
    write_function_section(&mut out, &mut ctx)?;

    // 4. Constant section
    write_constant_section(&mut out, &ctx.constants)?;

    // 5. Debug section (always present, like the reference emitter).
    write_debug_section(&mut out, &mut ctx)?;

    // 6. Type section
    write_type_section(&mut out, &ctx.types, version)?;

    // 7. String section
    write_string_section(&mut out, &ctx.strings)?;

    // 8. End marker
    if let Some(producer) = &module.producer {
        let mut payload = EncodingWriter::new();
        payload.write_varint(ctx.strings.get_or_insert(producer));
        let bytes = payload.into_bytes();
        write_section_header(&mut out, Section::Producer, bytes.len(), 1);
        out.extend_from_slice(&bytes);
    }
    out.push(Section::EndOfBytecode as u8);

    Ok(out)
}

fn check_region_versions(
    module: &Module,
    id: OpId,
    version: BytecodeVersion,
    in_loop: bool,
    in_for: bool,
) -> Result<()> {
    let op = module.op(id);
    if op.opcode == Opcode::Return {
        if in_for {
            return Err(Error::BytecodeWrite(
                "return is not allowed inside for".into(),
            ));
        }
        if in_loop {
            require_version("return inside loop", Feature::LoopReturn.since(), version)?;
        }
    }
    for (_, attr) in &op.attributes {
        if let Attribute::OptimizationHints(hints) = attr {
            for (arch, values) in &hints.entries {
                if let Some(requirement) = arch
                    .strip_prefix("sm_")
                    .and_then(|sm| sm.parse().ok())
                    .and_then(target_requirement)
                {
                    require_version(
                        &format!("optimization hint target {arch}"),
                        requirement.since,
                        version,
                    )?;
                }
                if values
                    .iter()
                    .any(|(key, _)| key == "num_worker_warps_per_cta")
                {
                    require_version(
                        "optimization hint 'num_worker_warps_per_cta'",
                        Feature::WorkerWarps.since(),
                        version,
                    )?;
                }
            }
        }
    }
    for &region in &op.regions {
        for &block in &module.region(region).blocks {
            for &child in &module.block(block).ops {
                check_region_versions(
                    module,
                    child,
                    version,
                    in_loop || op.opcode == Opcode::Loop,
                    in_for || op.opcode == Opcode::For,
                )?;
            }
        }
    }
    Ok(())
}

/// Convenience: write bytecode directly to a file.
pub fn write_bytecode_to_file(module: &Module, path: &str) -> Result<()> {
    let bytes = write_bytecode(module)?;
    let mut f = std::fs::File::create(path)
        .map_err(|e| Error::BytecodeWrite(format!("failed to create {path}: {e}")))?;
    f.write_all(&bytes)
        .map_err(|e| Error::BytecodeWrite(format!("failed to write {path}: {e}")))?;
    Ok(())
}

// =========================================================================
// Header
// =========================================================================

fn write_header(out: &mut Vec<u8>, version: BytecodeVersion) {
    out.extend_from_slice(&MAGIC);
    out.push(version.major);
    out.push(version.minor);
    out.extend_from_slice(&version.tag.to_le_bytes());
}

// =========================================================================
// Section header
// =========================================================================

fn write_section_header(out: &mut Vec<u8>, section_id: Section, length: usize, alignment: u64) {
    // Write directly onto `out` so that alignment padding is relative to the
    // overall stream position (matching the C++ writer and the reader's
    // expectation).
    let mut id_byte = section_id as u8 & 0x7F;
    if alignment > 1 {
        id_byte |= 0x80;
    }
    out.push(id_byte);
    // Write length as varint inline.
    {
        let mut v = length as u64;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
    }
    if alignment > 1 {
        // Write alignment varint.
        {
            let mut v = alignment;
            loop {
                let mut byte = (v & 0x7F) as u8;
                v >>= 7;
                if v != 0 {
                    byte |= 0x80;
                }
                out.push(byte);
                if v == 0 {
                    break;
                }
            }
        }
        // Pad to alignment relative to overall stream position.
        let pos = out.len() as u64;
        let padding = (alignment - (pos % alignment)) % alignment;
        for _ in 0..padding {
            out.push(super::enums::ALIGNMENT_BYTE);
        }
    }
}

// =========================================================================
// String manager
// =========================================================================

pub(super) struct StringManager {
    map: indexmap::IndexMap<String, u64>,
}

impl StringManager {
    pub(super) fn new() -> Self {
        Self {
            map: indexmap::IndexMap::new(),
        }
    }

    pub(super) fn get_or_insert(&mut self, s: &str) -> u64 {
        if let Some(&idx) = self.map.get(s) {
            return idx;
        }
        let idx = self.map.len() as u64;
        self.map.insert(s.to_owned(), idx);
        idx
    }
}

fn write_string_section(out: &mut Vec<u8>, strings: &StringManager) -> Result<()> {
    if strings.map.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(strings.map.len() as u64);
    w.align_to(4);

    // Reserve offset table.
    let offsets_pos = w.tell();
    for _ in 0..strings.map.len() {
        w.write_le_u32(0);
    }

    // Write strings and track offsets.
    let mut running: u32 = 0;
    let mut offsets = Vec::with_capacity(strings.map.len());
    for (s, _) in &strings.map {
        offsets.push(running);
        w.write_bytes(s.as_bytes());
        running += s.len() as u32;
    }

    // Patch offsets.
    for (i, offset) in offsets.iter().enumerate() {
        patch_u32(w.buf_mut(), offsets_pos + i * 4, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::String, buf.len(), 4);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Type manager
// =========================================================================

pub(super) struct TypeManager {
    list: Vec<Type>,
    map: HashMap<Type, u64>,
}

impl TypeManager {
    pub(super) fn new() -> Self {
        Self {
            list: Vec::new(),
            map: HashMap::new(),
        }
    }

    pub(super) fn get_or_insert(&mut self, ty: &Type) -> u64 {
        if let Some(&idx) = self.map.get(ty) {
            return idx;
        }
        // Register dependent types first.
        self.register_deps(ty);
        let idx = self.list.len() as u64;
        self.map.insert(ty.clone(), idx);
        self.list.push(ty.clone());
        idx
    }

    fn register_deps(&mut self, ty: &Type) {
        match ty {
            Type::WithPointerAttribute(base, attr) => match base.as_ref() {
                Type::Pointer(p) => {
                    self.get_or_insert(&Type::Scalar(p.pointee));
                }
                Type::TensorView(tv) => {
                    self.get_or_insert(&Type::Scalar(tv.element_type));
                }
                _ => {
                    if let Some(dependency) = pointer_dependency(base) {
                        self.get_or_insert(&dependency.with_pointer_attribute(*attr));
                    }
                }
            },
            Type::Pointer(p) => {
                self.get_or_insert(&Type::Scalar(p.pointee));
            }
            Type::Tile(t) => match &t.element_type {
                TileElementType::Scalar(s) => {
                    self.get_or_insert(&Type::Scalar(*s));
                }
                TileElementType::Pointer(p) => {
                    self.get_or_insert(&Type::Pointer((**p).clone()));
                }
            },
            Type::TensorView(tv) => {
                self.get_or_insert(&Type::Scalar(tv.element_type));
            }
            Type::PartitionView(pv) => {
                self.get_or_insert(&Type::TensorView(pv.tensor_view.clone()));
            }
            Type::GatherScatterView(gsv) => {
                self.get_or_insert(&Type::TensorView(gsv.tensor_view.clone()));
            }
            Type::StridedView(sv) => {
                self.get_or_insert(&Type::TensorView(sv.tensor_view.clone()));
            }
            Type::Func(f) => {
                for inp in &f.inputs {
                    self.get_or_insert(inp);
                }
                for res in &f.results {
                    self.get_or_insert(res);
                }
            }
            Type::Scalar(_) | Type::Token => {}
        }
    }
}

fn serialize_type(
    ty: &Type,
    types: &mut TypeManager,
    w: &mut EncodingWriter,
    version: BytecodeVersion,
) -> Result<()> {
    serialize_type_with_pointer_attribute(ty, types, w, version, None)
}

fn pointer_dependency(ty: &Type) -> Option<Type> {
    match ty {
        Type::Tile(t) => match &t.element_type {
            TileElementType::Pointer(p) => Some(Type::Pointer((**p).clone())),
            _ => None,
        },
        Type::PartitionView(v) => Some(Type::TensorView(v.tensor_view.clone())),
        Type::GatherScatterView(v) => Some(Type::TensorView(v.tensor_view.clone())),
        Type::StridedView(v) => Some(Type::TensorView(v.tensor_view.clone())),
        _ => None,
    }
}

fn serialize_type_with_pointer_attribute(
    ty: &Type,
    types: &mut TypeManager,
    w: &mut EncodingWriter,
    version: BytecodeVersion,
    pointer_attribute: Option<crate::ir::PointerAttribute>,
) -> Result<()> {
    let qualify = |ty: Type| match pointer_attribute {
        Some(attr) => ty.with_pointer_attribute(attr),
        None => ty,
    };
    match ty {
        Type::WithPointerAttribute(base, attr) => {
            require_version("ptr_attr", Feature::PointerAttribute.since(), version)?;
            if !matches!(base.as_ref(), Type::Pointer(_) | Type::TensorView(_))
                && pointer_dependency(base).is_none()
            {
                return Err(Error::BytecodeWrite(
                    "ptr_attr requires a pointer, pointer tile or view type".into(),
                ));
            }
            serialize_type_with_pointer_attribute(base, types, w, version, Some(*attr))?;
        }
        Type::Scalar(s) => {
            require_version(&format!("{s:?}"), s.minimum_version(), version)?;
            w.write_varint(s.type_tag() as u64);
        }
        Type::Pointer(p) => {
            w.write_varint(TypeTag::Pointer as u64);
            if version >= BytecodeVersion::V13_4 {
                w.write_varint(u64::from(pointer_attribute.is_some()));
            }
            let idx = types.get_or_insert(&Type::Scalar(p.pointee));
            w.write_varint(idx);
            if let Some(attr) = pointer_attribute {
                w.write_byte(attr as u8);
            }
        }
        Type::Tile(t) => {
            w.write_varint(TypeTag::Tile as u64);
            let elem_ty = match &t.element_type {
                TileElementType::Scalar(s) => Type::Scalar(*s),
                TileElementType::Pointer(p) => qualify(Type::Pointer((**p).clone())),
            };
            let idx = types.get_or_insert(&elem_ty);
            w.write_varint(idx);
            w.write_le_var_size_i64(&t.shape);
        }
        Type::TensorView(tv) => {
            w.write_varint(TypeTag::TensorView as u64);
            if version >= BytecodeVersion::V13_4 {
                w.write_varint(u64::from(pointer_attribute.is_some()));
            }
            let idx = types.get_or_insert(&Type::Scalar(tv.element_type));
            w.write_varint(idx);
            w.write_le_var_size_i64(&tv.shape);
            w.write_le_var_size_i64(&tv.strides);
            if let Some(attr) = pointer_attribute {
                w.write_byte(attr as u8);
            }
        }
        Type::PartitionView(pv) => {
            w.write_varint(TypeTag::PartitionView as u64);
            if version >= BytecodeVersion::V13_3 {
                let flags = if pv.padding_value.is_some() {
                    1u64
                } else {
                    0u64
                };
                w.write_varint(flags);
            }
            w.write_le_var_size_i32(&pv.tile_shape);
            let idx = types.get_or_insert(&qualify(Type::TensorView(pv.tensor_view.clone())));
            w.write_varint(idx);
            w.write_le_var_size_i32(&pv.dim_map);
            if version >= BytecodeVersion::V13_3 {
                if let Some(pv_val) = pv.padding_value {
                    w.write_byte(pv_val as u8);
                }
            } else {
                let has_padding = pv.padding_value.is_some();
                w.write_byte(has_padding as u8);
                if let Some(pv_val) = pv.padding_value {
                    w.write_varint(pv_val as u64);
                }
            }
        }
        Type::GatherScatterView(gsv) => {
            require_version(
                "gather_scatter_view",
                Feature::GatherScatterView.since(),
                version,
            )?;
            w.write_varint(TypeTag::GatherScatterView as u64);
            w.write_varint(if gsv.padding_value.is_some() { 1 } else { 0 });
            w.write_le_var_size_i32(&gsv.tile_shape);
            let idx = types.get_or_insert(&qualify(Type::TensorView(gsv.tensor_view.clone())));
            w.write_varint(idx);
            w.write_varint(gsv.sparse_dim as u64);
            if let Some(padding_value) = gsv.padding_value {
                w.write_byte(padding_value as u8);
            }
        }
        Type::StridedView(sv) => {
            require_version("strided_view", Feature::StridedView.since(), version)?;
            w.write_varint(TypeTag::StridedView as u64);
            w.write_varint(if sv.padding_value.is_some() { 1 } else { 0 });
            w.write_le_var_size_i32(&sv.tile_shape);
            w.write_le_var_size_i32(&sv.traversal_strides);
            let idx = types.get_or_insert(&qualify(Type::TensorView(sv.tensor_view.clone())));
            w.write_varint(idx);
            w.write_le_var_size_i32(&sv.dim_map);
            if let Some(padding_value) = sv.padding_value {
                w.write_byte(padding_value as u8);
            }
        }
        Type::Func(f) => {
            w.write_varint(TypeTag::Func as u64);
            w.write_varint(f.inputs.len() as u64);
            for inp in &f.inputs {
                let idx = types.get_or_insert(inp);
                w.write_varint(idx);
            }
            w.write_varint(f.results.len() as u64);
            for res in &f.results {
                let idx = types.get_or_insert(res);
                w.write_varint(idx);
            }
        }
        Type::Token => {
            w.write_varint(TypeTag::Token as u64);
        }
    }
    Ok(())
}

pub(super) fn require_version(
    feature: &str,
    required: BytecodeVersion,
    target: BytecodeVersion,
) -> Result<()> {
    if target < required {
        return Err(Error::BytecodeWrite(format!(
            "{feature} requires bytecode version {required} or newer, requested {target}"
        )));
    }
    Ok(())
}

fn write_type_section(
    out: &mut Vec<u8>,
    types: &TypeManager,
    version: BytecodeVersion,
) -> Result<()> {
    if types.list.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(types.list.len() as u64);
    w.align_to(4);

    // Reserve offset table.
    let offsets_pos = w.tell();
    for _ in 0..types.list.len() {
        w.write_le_u32(0);
    }

    // Serialize each type.
    let mut running: u32 = 0;
    let mut offsets = Vec::with_capacity(types.list.len());
    // Clone the type list to avoid borrow issues — the list is small.
    let type_list: Vec<Type> = types.list.clone();
    let mut types_mut = TypeManager {
        list: types.list.clone(),
        map: types.map.clone(),
    };
    for ty in &type_list {
        offsets.push(running);
        let before = w.tell();
        serialize_type(ty, &mut types_mut, &mut w, version)?;
        running += (w.tell() - before) as u32;
    }

    // Patch offsets.
    for (i, offset) in offsets.iter().enumerate() {
        patch_u32(w.buf_mut(), offsets_pos + i * 4, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Type, buf.len(), 4);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Constant manager
// =========================================================================

pub(super) struct ConstantManager {
    entries: Vec<Vec<u8>>,
    // We don't deduplicate by value for now — ops reference constants by index.
}

impl ConstantManager {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, data: Vec<u8>) -> u64 {
        let idx = self.entries.len() as u64;
        self.entries.push(data);
        idx
    }
}

fn write_constant_section(out: &mut Vec<u8>, constants: &ConstantManager) -> Result<()> {
    if constants.entries.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(constants.entries.len() as u64);
    w.align_to(8);

    // Reserve offset table (u64 per constant).
    let offsets_pos = w.tell();
    for _ in 0..constants.entries.len() {
        w.write_le_u64(0);
    }

    let mut running: u64 = 0;
    let mut offsets = Vec::with_capacity(constants.entries.len());
    for entry in &constants.entries {
        offsets.push(running);
        w.write_bytes(entry);
        running += entry.len() as u64;
    }

    for (i, offset) in offsets.iter().enumerate() {
        patch_u64(w.buf_mut(), offsets_pos + i * 8, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Constant, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Global section
// =========================================================================

fn write_global_section(out: &mut Vec<u8>, ctx: &mut WriterCtx) -> Result<()> {
    if ctx.module.globals.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(ctx.module.globals.len() as u64);

    for global in &ctx.module.globals {
        if global.constant {
            require_version(
                "global.constant",
                Feature::GlobalConstant.since(),
                ctx.version,
            )?;
        }
        if global.symbol_visibility != SymbolVisibility::Public {
            require_version(
                "global.symbol_visibility",
                Feature::GlobalVisibility.since(),
                ctx.version,
            )?;
        }
        // 1. Symbol name index.
        let name_idx = ctx.strings.get_or_insert(&global.sym_name);
        w.write_varint(name_idx);

        // 2. Type index of the global's value.
        let ty_idx = ctx.types.get_or_insert(&global.value.element_type);
        w.write_varint(ty_idx);

        // 3. Constant pool index for the raw data.
        let mut cdata = EncodingWriter::new();
        cdata.write_varint(global.value.data.len() as u64);
        cdata.write_bytes(&global.value.data);
        let const_idx = ctx.constants.add(cdata.into_bytes());
        w.write_varint(const_idx);

        // 4. Alignment.
        w.write_varint(global.alignment);

        if ctx.version >= BytecodeVersion::V13_3 {
            w.write_byte(global.symbol_visibility as u8);
            w.write_varint(global.constant as u64);
        }
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Global, buf.len(), 1);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Debug info section
// =========================================================================

/// Writes the Debug section in the reference layout:
///
/// ```text
///   numFunctions[varint]
///   padding[align 4]
///   indexOffsets[u32 x numFunctions]   // start of each function's ids
///   numIndices[varint]                 // total ids across functions
///   padding[align 8]
///   attrIds[u64 x numIndices]          // [0] = function attr, then per op
///   attrCount[varint]                  // interned attribute table
///   padding[align 4]
///   attrOffsets[u32 x attrCount]
///   attrData[bytes]                    // tag byte + varint fields each
/// ```
///
/// The section is always written (like the reference emitter): a bytecode
/// consumer treats a missing id as "no debug info", and the empty-table
/// workaround keeps decoders that reject zero-length tables working.
fn write_debug_section(out: &mut Vec<u8>, ctx: &mut WriterCtx) -> Result<()> {
    let mut w = EncodingWriter::new();

    let per_function = ctx.debug.per_function();
    w.write_varint(per_function.len() as u64);
    w.align_to(4);

    // Per-function offsets into the flattened id array.
    let mut index_offset: u32 = 0;
    for func_ids in per_function {
        w.write_le_u32(index_offset);
        index_offset += func_ids.len() as u32;
    }

    w.write_varint(index_offset as u64);
    w.align_to(8);

    for func_ids in per_function {
        for &attr_id in func_ids {
            w.write_le_u64(attr_id);
        }
    }

    ctx.debug.attrs.ensure_non_empty();
    let entries = ctx.debug.attrs.entries();

    w.write_varint(entries.len() as u64);
    w.align_to(4);
    let mut offset: u32 = 0;
    for encoded in entries {
        w.write_le_u32(offset);
        offset += encoded.len() as u32;
    }
    for encoded in entries {
        w.write_bytes(encoded);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Debug, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Pre-scan: collect all types, strings, constants from functions
// =========================================================================

fn prescan_function(
    module: &Module,
    func_op: OpId,
    strings: &mut StringManager,
    types: &mut TypeManager,
    _constants: &mut ConstantManager,
) -> Result<()> {
    let op = module.op(func_op);

    // Register function name and type from attributes.
    for (name, attr) in &op.attributes {
        prescan_attribute(name, attr, strings, types);
    }

    // Recurse into regions.
    for &region_id in &op.regions {
        prescan_region(module, region_id, strings, types)?;
    }

    Ok(())
}

fn prescan_region(
    module: &Module,
    region_id: RegionId,
    strings: &mut StringManager,
    types: &mut TypeManager,
) -> Result<()> {
    let region = module.region(region_id);
    for &block_id in &region.blocks {
        let block = module.block(block_id);
        // Block argument types.
        for (_, ty) in &block.args {
            types.get_or_insert(ty);
        }
        for &op_id in &block.ops {
            let op = module.op(op_id);
            // Result types.
            for ty in &op.result_types {
                types.get_or_insert(ty);
            }
            // Attributes.
            for (name, attr) in &op.attributes {
                prescan_attribute(name, attr, strings, types);
            }
            // Nested regions.
            for &rid in &op.regions {
                prescan_region(module, rid, strings, types)?;
            }
        }
    }
    Ok(())
}

fn prescan_attribute(
    name: &str,
    attr: &Attribute,
    strings: &mut StringManager,
    types: &mut TypeManager,
) {
    strings.get_or_insert(name);
    match attr {
        Attribute::String(s) => {
            strings.get_or_insert(s);
        }
        Attribute::Type(ty) => {
            types.get_or_insert(ty);
        }
        Attribute::DenseElements(de) => {
            types.get_or_insert(&de.element_type);
        }
        Attribute::Dictionary(entries) => {
            for (k, v) in entries {
                prescan_attribute(k, v, strings, types);
            }
        }
        Attribute::OptimizationHints(oh) => {
            for (arch, hints) in &oh.entries {
                strings.get_or_insert(arch);
                for (k, v) in hints {
                    prescan_attribute(k, v, strings, types);
                }
            }
        }
        Attribute::Array(elems) => {
            for elem in elems {
                prescan_attribute("", elem, strings, types);
            }
        }
        _ => {}
    }
}

// =========================================================================
// Function section
// =========================================================================

fn write_function_section(out: &mut Vec<u8>, ctx: &mut WriterCtx) -> Result<()> {
    if ctx.module.functions.is_empty() {
        return Ok(());
    }

    let func_ids: Vec<OpId> = ctx.module.functions.clone();
    let mut w = EncodingWriter::new();
    w.write_varint(func_ids.len() as u64);

    for func_op_id in &func_ids {
        let op = ctx.module.op(*func_op_id);

        // Extract name and function type from attributes.
        let name = find_string_attr(&op.attributes, "sym_name")
            .ok_or_else(|| Error::BytecodeWrite("function missing sym_name attribute".into()))?;
        let func_type = find_type_attr(&op.attributes, "function_type").ok_or_else(|| {
            Error::BytecodeWrite("function missing function_type attribute".into())
        })?;

        let name_idx = ctx.strings.get_or_insert(&name);
        let sig_idx = ctx.types.get_or_insert(&func_type);

        w.write_varint(name_idx);
        w.write_varint(sig_idx);

        // Entry flag.
        let is_entry = op.opcode == Opcode::Entry;
        let hints_attr = if is_entry {
            find_attr(&op.attributes, "optimization_hints").cloned()
        } else {
            None
        };
        if ctx.version < BytecodeVersion::V13_3 {
            if let Some(Attribute::OptimizationHints(hints)) = &hints_attr {
                for (_, arch_hints) in &hints.entries {
                    if arch_hints
                        .iter()
                        .any(|(name, _)| name == "num_worker_warps_per_cta")
                    {
                        return Err(Error::BytecodeWrite(format!(
                            "optimization hint 'num_worker_warps_per_cta' requires bytecode version 13.3 or newer, requested {}",
                            ctx.version
                        )));
                    }
                }
            }
        }
        let has_hints = hints_attr.is_some();
        let mut flags: u8 = 0;
        if is_entry {
            flags |= FunctionFlag::KindKernel as u8;
        }
        if is_entry && has_hints {
            flags |= FunctionFlag::HasOptimizationHints as u8;
        }
        w.write_byte(flags);

        // Function debug-info index: 1-based into the Debug section's
        // per-function attribute lists. The DI name is the user-facing
        // kernel name when the frontend provided one; the linkage name is
        // the emitted symbol.
        let di_name = find_string_attr(&op.attributes, "di_name");
        let di_idx =
            ctx.debug
                .begin_function(&mut ctx.strings, &name, di_name.as_deref(), &op.location);
        w.write_varint(di_idx);

        // Write optimization hints if present.
        if let Some(hints_attr) = hints_attr {
            write_self_contained_attribute(
                &hints_attr,
                &mut w,
                &mut ctx.strings,
                &mut ctx.types,
                &mut ctx.constants,
            )?;
        }

        // Write function body.
        // Reset per-function state.
        ctx.value_map.clear();
        ctx.next_idx = 0;
        let body = write_function_body(ctx, *func_op_id)?;
        w.write_varint(body.len() as u64);
        w.write_bytes(&body);
    }

    w.align_to(8);
    let buf = w.into_bytes();
    write_section_header(out, Section::Func, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

fn write_function_body(ctx: &mut WriterCtx, func_op: OpId) -> Result<Vec<u8>> {
    let mut w = EncodingWriter::new();

    let op = ctx.module.op(func_op);
    // Register function argument values (from the entry block of the first region).
    if let Some(&region_id) = op.regions.first() {
        let region = ctx.module.region(region_id);
        if let Some(&entry_block) = region.blocks.first() {
            let block = ctx.module.block(entry_block);
            let args: Vec<_> = block.args.clone();
            let ops: Vec<_> = block.ops.clone();
            for (val, _) in &args {
                ctx.value_map.insert(*val, ctx.next_idx);
                ctx.next_idx += 1;
            }
            for op_id in ops {
                ctx.write_operation(op_id, &mut w)?;
            }
        }
    }

    Ok(w.into_bytes())
}

// =========================================================================
// Writer context — bundles mutable state to avoid borrow conflicts
// =========================================================================

/// Bundles all mutable serialization state so it can be passed as a single
/// `&mut` to per-op writers and recursive region/block writers.
pub(super) struct WriterCtx<'a> {
    pub module: &'a Module,
    pub version: BytecodeVersion,
    pub value_map: HashMap<Value, u64>,
    pub next_idx: u64,
    pub strings: StringManager,
    pub types: TypeManager,
    pub constants: ConstantManager,
    pub debug: DebugInfoCollector,
}

/// Collects debug info during serialization, mirroring the reference
/// frontend emitter: a content-addressed attribute table plus one
/// attribute-id list per function, whose first element is the function's own
/// location attribute and whose remaining elements are one id per operation
/// in body-serialization order.
pub(super) struct DebugInfoCollector {
    pub(super) attrs: super::debug_info::DebugAttrTable,
    /// One list per function, in Func-section order.
    per_function: Vec<Vec<u64>>,
    /// Subprogram attribute of the function currently being serialized;
    /// scopes bare `FileLineCol` op locations. 0 = none.
    current_scope: u64,
}

impl DebugInfoCollector {
    pub(super) fn new() -> Self {
        Self {
            attrs: super::debug_info::DebugAttrTable::default(),
            per_function: Vec::new(),
            current_scope: 0,
        }
    }

    pub fn per_function(&self) -> &[Vec<u64>] {
        &self.per_function
    }

    fn subprogram_attr(
        &mut self,
        strings: &mut StringManager,
        sp: &crate::ir::DISubprogram,
    ) -> u64 {
        let file = self.attrs.file(strings, &sp.file.name, &sp.file.directory);
        let cu_file = self.attrs.file(
            strings,
            &sp.compile_unit.file.name,
            &sp.compile_unit.file.directory,
        );
        let cu = self.attrs.compile_unit(cu_file);
        self.attrs.subprogram(
            strings,
            file,
            sp.line as u64,
            &sp.name,
            &sp.linkage_name,
            cu,
            sp.scope_line as u64,
        )
    }

    fn scope_attr(&mut self, strings: &mut StringManager, scope: &crate::ir::DebugScope) -> u64 {
        match scope {
            crate::ir::DebugScope::Subprogram(sp) => self.subprogram_attr(strings, sp),
            crate::ir::DebugScope::LexicalBlock(lb) => {
                let parent = self.scope_attr(strings, &lb.scope);
                let file = self.attrs.file(strings, &lb.file.name, &lb.file.directory);
                self.attrs
                    .lexical_block(parent, file, lb.line as u64, lb.column as u64)
            }
        }
    }

    /// Converts a location into an interned attribute id (0 = no info).
    ///
    /// Preserve the producer's scopes and call chains. The compiler uses
    /// one compilation unit for a kernel and its inlined helpers, while
    /// keeping each helper's source file on its subprogram. Live cross-file
    /// scopes are exercised through tileiras --device-debug by cutile's
    /// `debug_info` integration test, not just by verifier-only probes.
    fn attr_for(&mut self, strings: &mut StringManager, loc: &crate::ir::Location) -> u64 {
        use crate::ir::Location;
        match loc {
            Location::Unknown => super::debug_info::MISSING_DEBUG_ATTR_ID,
            Location::FileLineCol {
                filename,
                line,
                column,
            } => {
                // A bare file:line:col is scoped to the enclosing function's
                // subprogram. Without one there is nothing valid to emit.
                if self.current_scope == super::debug_info::MISSING_DEBUG_ATTR_ID {
                    return super::debug_info::MISSING_DEBUG_ATTR_ID;
                }
                let scope = self.current_scope;
                self.attrs
                    .loc(strings, scope, filename, *line as u64, *column as u64)
            }
            Location::DebugInfo(di) => {
                let scope = self.scope_attr(strings, &di.scope);
                self.attrs.loc(
                    strings,
                    scope,
                    &di.filename,
                    di.line as u64,
                    di.column as u64,
                )
            }
            Location::CallSite { callee, caller } => {
                let callee_attr = self.attr_for(strings, callee);
                if callee_attr == super::debug_info::MISSING_DEBUG_ATTR_ID {
                    return super::debug_info::MISSING_DEBUG_ATTR_ID;
                }
                let caller_attr = self.attr_for(strings, caller);
                self.attrs.call_site(callee_attr, caller_attr)
            }
        }
    }

    /// Starts a new function's debug list and returns its 1-based index for
    /// the Func-section record. `di_name` is the user-facing kernel name;
    /// `sym_name` is the emitted symbol (the DI linkage name).
    pub fn begin_function(
        &mut self,
        strings: &mut StringManager,
        sym_name: &str,
        di_name: Option<&str>,
        loc: &crate::ir::Location,
    ) -> u64 {
        use crate::ir::Location;
        self.current_scope = match loc {
            Location::FileLineCol { filename, line, .. } => {
                let (dir, base) = split_file_path(filename);
                let file = self.attrs.file(strings, base, dir);
                let cu = self.attrs.compile_unit(file);
                self.attrs.subprogram(
                    strings,
                    file,
                    *line as u64,
                    di_name.unwrap_or(sym_name),
                    sym_name,
                    cu,
                    *line as u64,
                )
            }
            Location::DebugInfo(di) => self.scope_attr(strings, &di.scope.clone()),
            _ => super::debug_info::MISSING_DEBUG_ATTR_ID,
        };
        let func_attr = self.attr_for(strings, loc);
        self.per_function.push(vec![func_attr]);
        self.per_function.len() as u64
    }

    /// Records one operation's attribute id, in serialization order.
    pub fn record_op(&mut self, strings: &mut StringManager, loc: &crate::ir::Location) {
        let attr = self.attr_for(strings, loc);
        // Ops are only ever serialized inside a function; recording one
        // outside would silently desync every subsequent per-op index.
        debug_assert!(
            !self.per_function.is_empty(),
            "op serialized before any begin_function"
        );
        if let Some(list) = self.per_function.last_mut() {
            list.push(attr);
        }
    }
}

/// Splits a path into (directory, basename) for DIFile encoding.
fn split_file_path(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => ("", path),
    }
}

impl<'a> WriterCtx<'a> {
    pub fn write_operation(&mut self, op_id: OpId, w: &mut EncodingWriter) -> Result<()> {
        let op = self.module.op(op_id);

        // Record the op's debug attribute, in serialization order — the
        // Debug section's per-op ids must line up with the ops as decoded.
        let loc = op.location.clone();
        self.debug.record_op(&mut self.strings, &loc);

        let op = self.module.op(op_id);

        // Write opcode.
        w.write_varint(op.opcode.as_u16() as u64);

        // Per-op body serialization.
        super::op_writer::write_op_body(op, w, self)?;

        // Register result values.
        let result_types_len = self.module.op(op_id).result_types.len();
        for i in 0..result_types_len {
            let val = find_result_value(self.module, op_id, i as u32);
            if let Some(v) = val {
                self.value_map.insert(v, self.next_idx);
                self.next_idx += 1;
            }
        }

        Ok(())
    }

    pub fn write_region(&mut self, region_id: RegionId, w: &mut EncodingWriter) -> Result<()> {
        let region = self.module.region(region_id);
        let block_ids: Vec<_> = region.blocks.clone();
        w.write_varint(block_ids.len() as u64);
        for block_id in block_ids {
            self.write_block(block_id, w)?;
        }
        Ok(())
    }

    pub fn write_block(&mut self, block_id: BlockId, w: &mut EncodingWriter) -> Result<()> {
        let saved_next_idx = self.next_idx;
        let block = self.module.block(block_id);
        let args: Vec<_> = block.args.clone();
        let ops: Vec<_> = block.ops.clone();

        // Block arguments.
        w.write_varint(args.len() as u64);
        for (val, ty) in &args {
            let idx = self.types.get_or_insert(ty);
            w.write_varint(idx);
            self.value_map.insert(*val, self.next_idx);
            self.next_idx += 1;
        }

        // Operations.
        w.write_varint(ops.len() as u64);
        for op_id in ops {
            self.write_operation(op_id, w)?;
        }

        // Roll back block-scoped values.
        self.value_map.retain(|_, v| *v < saved_next_idx);
        self.next_idx = saved_next_idx;

        Ok(())
    }
}

/// Find the Value in the module that was produced as result `result_index` of `op_id`.
fn find_result_value(module: &Module, op_id: OpId, result_index: u32) -> Option<Value> {
    for (i, vd) in module.values.iter().enumerate() {
        if let ValueProducer::OpResult {
            op,
            result_index: ri,
        } = vd.producer
        {
            if op == op_id && ri == result_index {
                return Some(Value(i as u32));
            }
        }
    }
    None
}

// =========================================================================
// Attribute serialization
// =========================================================================

pub(super) fn write_self_contained_attribute(
    attr: &Attribute,
    w: &mut EncodingWriter,
    strings: &mut StringManager,
    types: &mut TypeManager,
    constants: &mut ConstantManager,
) -> Result<()> {
    match attr {
        Attribute::Integer(v, ty) => {
            w.write_varint(AttributeTag::Integer as u64);
            let ty_idx = types.get_or_insert(ty);
            w.write_varint(ty_idx);
            w.write_varint(*v as u64);
        }
        Attribute::Float(v, ty) => {
            w.write_varint(AttributeTag::Float as u64);
            let ty_idx = types.get_or_insert(ty);
            w.write_varint(ty_idx);
            w.write_ap_float(*v, ty);
        }
        Attribute::Bool(v) => {
            w.write_varint(AttributeTag::Bool as u64);
            w.write_byte(if *v { 0x01 } else { 0x00 });
        }
        Attribute::Type(ty) => {
            w.write_varint(AttributeTag::Type as u64);
            let idx = types.get_or_insert(ty);
            w.write_varint(idx);
        }
        Attribute::String(s) => {
            w.write_varint(AttributeTag::String as u64);
            let idx = strings.get_or_insert(s);
            w.write_varint(idx);
        }
        Attribute::Array(elems) => {
            w.write_varint(AttributeTag::Array as u64);
            w.write_varint(elems.len() as u64);
            for elem in elems {
                write_self_contained_attribute(elem, w, strings, types, constants)?;
            }
        }
        Attribute::DenseElements(de) => {
            w.write_varint(AttributeTag::DenseElements as u64);
            let ty_idx = types.get_or_insert(&de.element_type);
            w.write_varint(ty_idx);
            // Serialize raw data into constant pool.
            let mut cdata = EncodingWriter::new();
            cdata.write_varint(de.data.len() as u64);
            cdata.write_bytes(&de.data);
            let const_idx = constants.add(cdata.into_bytes());
            w.write_varint(const_idx);
        }
        Attribute::DenseI32Array(arr) => {
            // Self-contained DenseI32Array: no tag needed (not a tagged attribute
            // in the bytecode format — it's always inline).
            w.write_le_var_size_i32(arr);
        }
        Attribute::DivBy(db) => {
            w.write_varint(AttributeTag::DivBy as u64);
            w.write_varint(db.divisor);
            let mut flags: u8 = 0;
            if db.every.is_some() {
                flags |= 0x01;
            }
            if db.along.is_some() {
                flags |= 0x02;
            }
            w.write_byte(flags);
            if let Some(every) = db.every {
                w.write_signed_varint(every);
            }
            if let Some(along) = db.along {
                w.write_signed_varint(along);
            }
        }
        Attribute::SameElements(se) => {
            w.write_varint(AttributeTag::SameElements as u64);
            w.write_le_var_size_i64(&se.values);
        }
        Attribute::Dictionary(entries) => {
            w.write_varint(AttributeTag::Dictionary as u64);
            w.write_varint(entries.len() as u64);
            for (key, val) in entries {
                let key_idx = strings.get_or_insert(key);
                w.write_varint(key_idx);
                write_self_contained_attribute(val, w, strings, types, constants)?;
            }
        }
        Attribute::OptimizationHints(oh) => {
            w.write_varint(AttributeTag::OptimizationHints as u64);
            // Serialize as a dictionary.
            w.write_varint(oh.entries.len() as u64);
            for (arch, hints) in &oh.entries {
                let arch_idx = strings.get_or_insert(arch);
                w.write_varint(arch_idx);
                // Each arch maps to a dictionary of hints.
                w.write_varint(AttributeTag::Dictionary as u64);
                w.write_varint(hints.len() as u64);
                for (k, v) in hints {
                    let k_idx = strings.get_or_insert(k);
                    w.write_varint(k_idx);
                    write_self_contained_attribute(v, w, strings, types, constants)?;
                }
            }
        }
        Attribute::Bounded(b) => {
            w.write_varint(AttributeTag::Bounded as u64);
            let mut flags: u8 = 0;
            if b.lb.is_some() {
                flags |= 0x01;
            }
            if b.ub.is_some() {
                flags |= 0x02;
            }
            w.write_byte(flags);
            if let Some(lb) = b.lb {
                w.write_signed_varint(lb);
            }
            if let Some(ub) = b.ub {
                w.write_signed_varint(ub);
            }
        }
    }
    Ok(())
}

// =========================================================================
// Attribute helpers
// =========================================================================

fn find_string_attr(attrs: &[(String, Attribute)], name: &str) -> Option<String> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::String(s) = v {
                return Some(s.clone());
            }
        }
        None
    })
}

fn find_type_attr(attrs: &[(String, Attribute)], name: &str) -> Option<Type> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::Type(ty) = v {
                return Some(ty.clone());
            }
        }
        None
    })
}

fn find_attr<'a>(attrs: &'a [(String, Attribute)], name: &str) -> Option<&'a Attribute> {
    attrs
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

/// Attributes that are encoded in the function header, not per-operation.
#[allow(dead_code)]
fn is_function_level_attr(name: &str) -> bool {
    matches!(name, "sym_name" | "function_type" | "optimization_hints")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{PointerAttribute, PointerType, ScalarType, TensorViewType};

    fn type_bytes(ty: &Type, version: BytecodeVersion) -> Result<Vec<u8>> {
        let mut types = TypeManager::new();
        types.get_or_insert(ty);
        let mut writer = EncodingWriter::new();
        serialize_type(ty, &mut types, &mut writer, version)?;
        Ok(writer.into_bytes())
    }

    fn compare_python(name: &str, version: BytecodeVersion, present: bool, bytes: &[u8]) {
        let Some(source) = std::env::var_os("CUTILE_PYTHON_SOURCE") else {
            return;
        };
        let output = std::process::Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/support/python_encoding.py"
            ))
            .arg(source)
            .arg(name)
            .arg(version.to_string())
            .arg(present.to_string())
            .output()
            .expect("run Python type encoder");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
    }

    #[test]
    fn pointer_attribute_presence_and_legacy_layouts_match_reference() {
        for (name, base) in [
            (
                "Pointer",
                Type::Pointer(PointerType {
                    pointee: ScalarType::F32,
                }),
            ),
            (
                "TensorView",
                Type::TensorView(TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![4],
                    strides: vec![1],
                }),
            ),
        ] {
            for version in BytecodeVersion::SUPPORTED {
                let absent = type_bytes(&base, version).unwrap();
                compare_python(name, version, false, &absent);
                let explicit = base.clone().with_pointer_attribute(PointerAttribute::None);
                assert_ne!(base, explicit);
                assert_eq!(
                    Type::parse(&crate::ir::format_type(&explicit)),
                    Some(explicit.clone())
                );
                if version < BytecodeVersion::V13_4 {
                    assert!(type_bytes(&explicit, version)
                        .unwrap_err()
                        .to_string()
                        .contains("ptr_attr requires bytecode version 13.4"));
                } else {
                    assert_eq!(absent[1], 0);
                    let present = type_bytes(&explicit, version).unwrap();
                    assert_eq!(present[1], 1);
                    assert_eq!(present.last(), Some(&0));
                    compare_python(name, version, true, &present);
                }
            }
        }
    }

    #[test]
    fn new_scale_type_uses_multibyte_tag_and_refuses_old_versions() {
        let ty = Type::Scalar(ScalarType::F8E5M3FNU);
        assert_eq!(
            type_bytes(&ty, BytecodeVersion::V13_4).unwrap(),
            [0x82, 0x01]
        );
        compare_python("F8E5M3FNU", BytecodeVersion::V13_4, false, &[0x82, 0x01]);
        assert!(type_bytes(&ty, BytecodeVersion::V13_3).is_err());
    }

    #[test]
    fn producer_roundtrips_and_is_not_silently_dropped_on_13_2() {
        let mut module = Module::new("producer");
        module.producer = Some("cutile-rs test".into());
        assert!(write_bytecode_version(&module, BytecodeVersion::V13_2).is_err());
        for version in [BytecodeVersion::V13_3, BytecodeVersion::V13_4] {
            let bytes = write_bytecode_version(&module, version).unwrap();
            assert!(crate::decode_bytecode(&bytes)
                .unwrap()
                .contains("cutile-rs test"));
        }
    }
}
