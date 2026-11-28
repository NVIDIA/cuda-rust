/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug-attribute interning for the bytecode Debug section.
//!
//! Attributes are content-addressed: each is encoded as `tag byte + varint
//! fields` and interned by its encoded bytes, so identical files, scopes,
//! and locations share one table entry. Ids are 1-based; id 0 is the
//! reserved "no debug info" value that per-op indices use for
//! [`Location::Unknown`](crate::ir::Location::Unknown). This mirrors the
//! reference frontend emitter (cutile-python's `DebugAttrTable`), which is
//! the format `tileiras` consumes.

use std::collections::HashMap;

use super::encoding::EncodingWriter;
use super::enums::DebugTag;
use super::writer::StringManager;

/// The reserved "no debug info" attribute id.
pub(super) const MISSING_DEBUG_ATTR_ID: u64 = 0;

/// Content-addressed table of encoded debug attributes.
#[derive(Default)]
pub(super) struct DebugAttrTable {
    map: HashMap<Vec<u8>, u64>,
    /// Encoded entries in id order (`entries[0]` is id 1).
    entries: Vec<Vec<u8>>,
}

impl DebugAttrTable {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Encoded entries in id order.
    pub fn entries(&self) -> &[Vec<u8>] {
        &self.entries
    }

    fn intern(&mut self, encoded: Vec<u8>) -> u64 {
        if let Some(&id) = self.map.get(&encoded) {
            return id;
        }
        let id = self.entries.len() as u64 + 1;
        self.map.insert(encoded.clone(), id);
        self.entries.push(encoded);
        id
    }

    pub fn file(&mut self, strings: &mut StringManager, name: &str, directory: &str) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::DIFile as u8);
        w.write_varint(strings.get_or_insert(name));
        w.write_varint(strings.get_or_insert(directory));
        self.intern(w.into_bytes())
    }

    pub fn compile_unit(&mut self, file: u64) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::DICompileUnit as u8);
        w.write_varint(file);
        self.intern(w.into_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn subprogram(
        &mut self,
        strings: &mut StringManager,
        file: u64,
        line: u64,
        name: &str,
        linkage_name: &str,
        compile_unit: u64,
        scope_line: u64,
        is_declaration: bool,
    ) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::DISubprogram as u8);
        w.write_varint(file);
        w.write_varint(line);
        w.write_varint(strings.get_or_insert(name));
        w.write_varint(strings.get_or_insert(linkage_name));
        w.write_varint(compile_unit);
        w.write_varint(scope_line);
        // Canonical interning uses the 13.5 flags field; older writers omit it.
        w.write_varint(u64::from(is_declaration));
        self.intern(w.into_bytes())
    }

    pub fn lexical_block(&mut self, parent_scope: u64, file: u64, line: u64, column: u64) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::DILexicalBlock as u8);
        w.write_varint(parent_scope);
        w.write_varint(file);
        w.write_varint(line);
        w.write_varint(column);
        self.intern(w.into_bytes())
    }

    pub fn loc(
        &mut self,
        strings: &mut StringManager,
        scope: u64,
        filename: &str,
        line: u64,
        column: u64,
    ) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::DILoc as u8);
        w.write_varint(scope);
        w.write_varint(strings.get_or_insert(filename));
        w.write_varint(line);
        w.write_varint(column);
        self.intern(w.into_bytes())
    }

    pub fn call_site(&mut self, callee: u64, caller: u64) -> u64 {
        let mut w = EncodingWriter::new();
        w.write_byte(DebugTag::CallSite as u8);
        w.write_varint(callee);
        w.write_varint(caller);
        self.intern(w.into_bytes())
    }

    /// The decoder in the consuming toolchain fails on an empty table; the
    /// reference emitter interns a single tag-0 entry in that case, and so
    /// do we.
    pub fn ensure_non_empty(&mut self) {
        if self.is_empty() {
            let id = self.intern(vec![DebugTag::Unknown as u8]);
            debug_assert_eq!(id, 1);
        }
    }
}

/// Splits a path into (directory, basename) for DIFile encoding.
pub fn split_file_path(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => ("", path),
    }
}
