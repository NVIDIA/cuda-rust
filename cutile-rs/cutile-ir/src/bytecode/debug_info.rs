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

fn push_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
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
        let mut buf = vec![DebugTag::DIFile as u8];
        push_varint(&mut buf, strings.get_or_insert(name));
        push_varint(&mut buf, strings.get_or_insert(directory));
        self.intern(buf)
    }

    pub fn compile_unit(&mut self, file: u64) -> u64 {
        let mut buf = vec![DebugTag::DICompileUnit as u8];
        push_varint(&mut buf, file);
        self.intern(buf)
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
    ) -> u64 {
        let mut buf = vec![DebugTag::DISubprogram as u8];
        push_varint(&mut buf, file);
        push_varint(&mut buf, line);
        push_varint(&mut buf, strings.get_or_insert(name));
        push_varint(&mut buf, strings.get_or_insert(linkage_name));
        push_varint(&mut buf, compile_unit);
        push_varint(&mut buf, scope_line);
        self.intern(buf)
    }

    pub fn lexical_block(&mut self, parent_scope: u64, file: u64, line: u64, column: u64) -> u64 {
        let mut buf = vec![DebugTag::DILexicalBlock as u8];
        push_varint(&mut buf, parent_scope);
        push_varint(&mut buf, file);
        push_varint(&mut buf, line);
        push_varint(&mut buf, column);
        self.intern(buf)
    }

    pub fn loc(
        &mut self,
        strings: &mut StringManager,
        scope: u64,
        filename: &str,
        line: u64,
        column: u64,
    ) -> u64 {
        let mut buf = vec![DebugTag::DILoc as u8];
        push_varint(&mut buf, scope);
        push_varint(&mut buf, strings.get_or_insert(filename));
        push_varint(&mut buf, line);
        push_varint(&mut buf, column);
        self.intern(buf)
    }

    pub fn call_site(&mut self, callee: u64, caller: u64) -> u64 {
        let mut buf = vec![DebugTag::CallSite as u8];
        push_varint(&mut buf, callee);
        push_varint(&mut buf, caller);
        self.intern(buf)
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
