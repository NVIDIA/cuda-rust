/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Serialize a library's two halves as Tile IR bytecode.
//!
//! The two modules only ever contain bodiless functions, so the file has
//! four sections: the function table (one entry per declaration or
//! implementation), the debug table (one unknown location per function), the
//! type table and the string table. Every encoding below was checked against
//! the reference reader's sources and against an assembler that reads 13.5
//! content, which packs the output into a `.tilelib` with
//! `--preview-create-tile-lib`.
//!
//! ```text
//! header: 7F 'T' 'i' 'l' 'e' 'I' 'R' 00 | major | minor | tag u16 (bit 0: preview)
//! section Func  (aligned 8): varint(n) then entries, padded to 8
//! section Debug (aligned 8): per-function id lists, one unknown entry
//! section Type  (aligned 4): varint(n), offsets, entries
//! section String(aligned 4): varint(n), offsets, bytes
//! end byte 00
//! ```

use std::collections::HashMap;

use crate::Error;
use crate::encoding::{
    CONTAINS_PREVIEW, MAGIC, Section, Writer, begin_offset_table, write_section,
};
use crate::model::{
    Access, BytecodeVersion, Declaration, FragmentType, Implementation, Layout, Library,
    ScalarType, TileType,
};

// Type-table tags.
const TAG_TILE: u64 = 13;
const TAG_FUNC: u64 = 16;
const TAG_FRAGMENT: u64 = 0xD2;

// Attribute tags.
const ATTR_STRING: u64 = 5;
const ATTR_ARRAY: u64 = 6;
const ATTR_DICTIONARY: u64 = 10;
const ATTR_SYMBOLIC_INT: u64 = 0xD1;
const ATTR_SYMBOLIC_TUPLE: u64 = 0xD8;
const ATTR_LAYOUT: u64 = 0xD9;
const ATTR_ACCESS: u64 = 0xDF;

/// The argument attribute that carries an access.
const ACCESS_ATTR: &str = "cuda_tile.preview$access";

// Function-table entry flags: bit 0 private, bit 1 kernel, bits 3:2 form.
const FLAG_PRIVATE: u8 = 0x01;
const FORM_EXTERN_IMPL: u8 = 0x04;

// `preview$extern_impl` payload flag bits, in the op's declaration order.
const IMPL_ARG_ATTRS: u64 = 1 << 2;
const IMPL_ARCH: u64 = 1 << 6;
const IMPL_FRONTEND_ABI: u64 = 1 << 7;
const IMPL_SYNCHRONIZATION: u64 = 1 << 8;

/// A type the type table can hold.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Type {
    Scalar(ScalarType),
    Tile(TileType),
    Fragment(FragmentType),
    Func(Vec<Type>, Vec<Type>),
}

/// Interning tables, filled in first-use order.
struct Tables {
    /// The version the module is written at; the fragment type's symbolic
    /// shape is encoded differently before 13.6.
    version: BytecodeVersion,
    strings: Vec<String>,
    string_index: HashMap<String, u64>,
    types: Vec<Type>,
    type_index: HashMap<Type, u64>,
}

impl Tables {
    fn new(version: BytecodeVersion) -> Self {
        Self {
            version,
            strings: Default::default(),
            string_index: Default::default(),
            types: Default::default(),
            type_index: Default::default(),
        }
    }

    fn string(&mut self, text: &str) -> u64 {
        if let Some(&index) = self.string_index.get(text) {
            return index;
        }
        let index = self.strings.len() as u64;
        self.strings.push(text.to_owned());
        self.string_index.insert(text.to_owned(), index);
        index
    }

    /// Register `ty`, its dependencies first, and return its index.
    fn ty(&mut self, ty: &Type) -> u64 {
        if let Some(&index) = self.type_index.get(ty) {
            return index;
        }
        match ty {
            Type::Scalar(_) => {}
            Type::Tile(tile) => {
                self.ty(&Type::Scalar(tile.element));
            }
            Type::Fragment(fragment) => {
                self.ty(&Type::Scalar(fragment.element));
            }
            Type::Func(inputs, results) => {
                for input in inputs {
                    self.ty(input);
                }
                for result in results {
                    self.ty(result);
                }
            }
        }
        let index = self.types.len() as u64;
        self.types.push(ty.clone());
        self.type_index.insert(ty.clone(), index);
        index
    }
}

/// `varint(count)` then each value as a tagged symbolic integer: the
/// payload of a `SymbolicTuple` and the inline form of a layout's modes.
fn write_int_tuple(w: &mut Writer, values: &[i64]) {
    w.varint(values.len() as u64);
    for &value in values {
        w.varint(ATTR_SYMBOLIC_INT);
        w.signed_varint(value);
    }
}

/// A layout attribute: tag, then the shapes and strides tuples inline.
fn write_layout(w: &mut Writer, layout: &Layout) {
    w.varint(ATTR_LAYOUT);
    write_int_tuple(w, &layout.shapes);
    write_int_tuple(w, &layout.strides);
}

/// The dictionary `{cuda_tile.preview$access = #access<..>}`, tagged.
fn write_access_dictionary(w: &mut Writer, tables: &mut Tables, access: Access) {
    w.varint(ATTR_DICTIONARY);
    w.varint(1);
    w.varint(tables.string(ACCESS_ATTR));
    w.varint(ATTR_ACCESS);
    w.varint(access.value());
}

/// `arg_attrs`: one access dictionary per argument. Tagged as an array for
/// a declaration (a positional field of the function table), inline (count
/// first, no array tag) inside an implementation's payload.
fn write_arg_attrs(w: &mut Writer, tables: &mut Tables, accesses: &[Access], tagged: bool) {
    if tagged {
        w.varint(ATTR_ARRAY);
    }
    w.varint(accesses.len() as u64);
    for &access in accesses {
        write_access_dictionary(w, tables, access);
    }
}

/// One type-table entry.
fn write_type(w: &mut Writer, tables: &mut Tables, ty: &Type) {
    match ty {
        Type::Scalar(scalar) => w.varint(u64::from(scalar.tag())),
        Type::Tile(tile) => {
            w.varint(TAG_TILE);
            w.varint(tables.ty(&Type::Scalar(tile.element)));
            w.le_var_size_i64(&tile.shape);
        }
        Type::Func(inputs, results) => {
            w.varint(TAG_FUNC);
            w.varint(inputs.len() as u64);
            for input in inputs {
                w.varint(tables.ty(input));
            }
            w.varint(results.len() as u64);
            for result in results {
                w.varint(tables.ty(result));
            }
        }
        // Tagged symbolic shape, the static shape, element type, memory
        // space, both layouts tagged, then the alignment as a tagged symbolic
        // integer, always 1.
        //
        // The symbolic shape's contents depend on the version. The 13.5
        // assembler keeps one integer-valued expression per dimension and
        // rejects a rank mismatch ("concrete shape cache rank 1 does not
        // match symbolic shape rank 0"). The compiler trunk switched on
        // 2026-09-08 to holding only the dynamic dimensions, so a static
        // shape writes an empty tuple from 13.6 on.
        Type::Fragment(fragment) => {
            w.varint(TAG_FRAGMENT);
            w.varint(ATTR_SYMBOLIC_TUPLE);
            if tables.version < BytecodeVersion::new(13, 6) {
                w.varint(fragment.shape.len() as u64);
                for &dim in &fragment.shape {
                    w.varint(ATTR_SYMBOLIC_INT);
                    w.signed_varint(dim);
                }
            } else {
                w.varint(0);
            }
            w.le_var_size_i64(&fragment.shape);
            w.varint(tables.ty(&Type::Scalar(fragment.element)));
            w.varint(fragment.memory_space.value());
            write_layout(w, &fragment.mem_layout);
            write_layout(w, &fragment.fragmap);
            w.varint(ATTR_SYMBOLIC_INT);
            w.signed_varint(1);
        }
    }
}

fn declaration_signature(decl: &Declaration) -> Type {
    Type::Func(
        decl.args
            .iter()
            .map(|arg| Type::Tile(arg.ty.clone()))
            .collect(),
        decl.results().map(|ty| Type::Tile(ty.clone())).collect(),
    )
}

fn implementation_signature(imp: &Implementation) -> Type {
    Type::Func(
        imp.args
            .iter()
            .map(|arg| Type::Fragment(arg.fragment.clone()))
            .collect(),
        Vec::new(),
    )
}

/// A `preview$func extern` entry: normal form, private, body length zero,
/// then its (empty) constant parameters.
fn write_declaration(w: &mut Writer, tables: &mut Tables, decl: &Declaration, debug: u64) {
    w.varint(tables.string(&decl.name));
    w.varint(tables.ty(&declaration_signature(decl)));
    w.byte(FLAG_PRIVATE);
    w.varint(debug);
    let accesses: Vec<Access> = decl.args.iter().map(|arg| arg.access).collect();
    write_arg_attrs(w, tables, &accesses, true);
    w.varint(0);
    w.varint(ATTR_ARRAY);
    w.varint(0);
}

/// A `preview$extern_impl` entry: form bits 01, private, then the op
/// payload in place of a body (flags, symbols, type, the present optional
/// clauses, one empty region) and a body length of zero.
fn write_implementation(w: &mut Writer, tables: &mut Tables, imp: &Implementation, debug: u64) {
    w.varint(tables.string(&imp.symbol));
    w.varint(tables.ty(&implementation_signature(imp)));
    w.byte(FLAG_PRIVATE | FORM_EXTERN_IMPL);
    w.varint(debug);

    let mut flags = IMPL_ARG_ATTRS;
    if !imp.arch.is_empty() {
        flags |= IMPL_ARCH;
    }
    if !imp.frontend_abi.is_empty() {
        flags |= IMPL_FRONTEND_ABI;
    }
    if imp.synchronization.is_some() {
        flags |= IMPL_SYNCHRONIZATION;
    }
    w.varint(flags);
    w.varint(tables.string(&imp.symbol));
    w.varint(tables.string(&imp.implements));
    w.varint(tables.ty(&implementation_signature(imp)));
    let accesses: Vec<Access> = imp.args.iter().map(|arg| arg.access).collect();
    write_arg_attrs(w, tables, &accesses, false);
    if !imp.arch.is_empty() {
        w.varint(imp.arch.len() as u64);
        for name in &imp.arch {
            w.varint(ATTR_STRING);
            w.varint(tables.string(name));
        }
    }
    if !imp.frontend_abi.is_empty() {
        w.varint(imp.frontend_abi.len() as u64);
        for (key, value) in &imp.frontend_abi {
            w.varint(tables.string(key));
            w.varint(ATTR_STRING);
            w.varint(tables.string(value));
        }
    }
    if let Some(sync) = imp.synchronization {
        w.varint(sync.value());
    }
    // One region with zero blocks, then the zero body length.
    w.varint(1);
    w.varint(0);
    w.varint(0);
}

/// Per-function id lists (each holds one id 0, an unknown location) and the
/// single unknown attribute the reader expects a non-empty table to hold.
fn debug_section(functions: usize) -> Vec<u8> {
    let mut w = Writer::new();
    w.varint(functions as u64);
    w.align_to(4);
    for index in 0..functions {
        w.le_u32(index as u32);
    }
    w.varint(functions as u64);
    w.align_to(8);
    for _ in 0..functions {
        w.le_u64(0);
    }
    w.varint(1);
    w.align_to(4);
    w.le_u32(0);
    w.byte(0);
    w.into_bytes()
}

fn type_section(tables: &mut Tables) -> Vec<u8> {
    let mut w = Writer::new();
    let count = tables.types.len();
    let offsets = begin_offset_table(&mut w, count);
    let start = w.len();
    // Writing an entry can only reference types registered before it, so
    // the table does not grow while it is serialized.
    let types = tables.types.clone();
    for (index, ty) in types.iter().enumerate() {
        w.patch_u32(offsets + 4 * index, (w.len() - start) as u32);
        write_type(&mut w, tables, ty);
    }
    debug_assert_eq!(tables.types.len(), count);
    w.into_bytes()
}

fn string_section(tables: &Tables) -> Vec<u8> {
    let mut w = Writer::new();
    let offsets = begin_offset_table(&mut w, tables.strings.len());
    let mut running = 0u32;
    for (index, text) in tables.strings.iter().enumerate() {
        w.patch_u32(offsets + 4 * index, running);
        w.bytes(text.as_bytes());
        running += text.len() as u32;
    }
    w.into_bytes()
}

fn write_module(
    version: BytecodeVersion,
    entries: usize,
    mut write_entries: impl FnMut(&mut Writer, &mut Tables),
) -> Result<Vec<u8>, Error> {
    if version < BytecodeVersion::PREVIEW_FIRST {
        return Err(Error::Version(version));
    }
    let mut tables = Tables::new(version);
    let mut functions = Writer::new();
    functions.varint(entries as u64);
    write_entries(&mut functions, &mut tables);
    functions.align_to(8);

    let mut out = MAGIC.to_vec();
    out.push(version.major);
    out.push(version.minor);
    out.extend_from_slice(&CONTAINS_PREVIEW.to_le_bytes());
    write_section(&mut out, Section::Func, &functions.into_bytes(), 8);
    write_section(&mut out, Section::Debug, &debug_section(entries), 8);
    write_section(&mut out, Section::Type, &type_section(&mut tables), 4);
    write_section(&mut out, Section::String, &string_section(&tables), 4);
    out.push(Section::EndOfBytecode as u8);
    Ok(out)
}

impl Library {
    /// The interface half: every declaration as a `preview$func extern`.
    ///
    /// `version` must be the one the consuming assembler calls current;
    /// preview content is readable at no other version. Anything below
    /// 13.5 is an error.
    pub fn interface_bytecode(&self, version: BytecodeVersion) -> Result<Vec<u8>, Error> {
        write_module(version, self.declarations.len(), |w, tables| {
            for (index, decl) in self.declarations.iter().enumerate() {
                write_declaration(w, tables, decl, index as u64 + 1);
            }
        })
    }

    /// The implementation half: every implementation as a
    /// `preview$extern_impl`.
    pub fn implementation_bytecode(&self, version: BytecodeVersion) -> Result<Vec<u8>, Error> {
        write_module(version, self.implementations.len(), |w, tables| {
            for (index, imp) in self.implementations.iter().enumerate() {
                write_implementation(w, tables, imp, index as u64 + 1);
            }
        })
    }
}
