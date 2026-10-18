/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The byte-level primitives of Tile IR bytecode: LEB128 varints, zigzag
//! signed varints, little-endian words, alignment padding and section
//! framing. These are the same primitives every Tile IR producer uses; the
//! reference reader documents them in its bytecode grammar.

/// The eight magic bytes every Tile IR bytecode file starts with.
pub const MAGIC: [u8; 8] = [0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00];

/// Padding byte used to reach a section's alignment.
pub const ALIGNMENT_BYTE: u8 = 0xCB;

/// Header tag bit set when a module carries public-preview content.
pub const CONTAINS_PREVIEW: u16 = 0x1;

/// Section identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Section {
    /// Marks the end of the file.
    EndOfBytecode = 0x00,
    /// The string table.
    String = 0x01,
    /// The function table.
    Func = 0x02,
    /// Debug locations and attributes.
    Debug = 0x03,
    /// The type table.
    Type = 0x05,
}

/// An append-only byte buffer with the bytecode's integer encodings.
#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// One raw byte.
    pub fn byte(&mut self, value: u8) {
        self.buf.push(value);
    }

    /// Raw bytes.
    pub fn bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// An unsigned LEB128 varint: seven bits per byte, high bit set while
    /// more bytes follow. Tags at or above 0x80 therefore take two bytes.
    pub fn varint(&mut self, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            self.buf.push(byte);
            if value == 0 {
                return;
            }
        }
    }

    /// A zigzag-encoded signed varint, used for `int64_t` parameters.
    pub fn signed_varint(&mut self, value: i64) {
        self.varint(((value << 1) ^ (value >> 63)) as u64);
    }

    /// A little-endian `u32`.
    pub fn le_u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// A little-endian `u64`.
    pub fn le_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// `varint(count)` followed by each value as a little-endian `i64`: the
    /// encoding of a shape.
    pub fn le_var_size_i64(&mut self, values: &[i64]) {
        self.varint(values.len() as u64);
        for &value in values {
            self.buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Pad with [`ALIGNMENT_BYTE`] until the length is a multiple of
    /// `alignment`.
    pub fn align_to(&mut self, alignment: usize) {
        if alignment < 2 {
            return;
        }
        while !self.buf.len().is_multiple_of(alignment) {
            self.buf.push(ALIGNMENT_BYTE);
        }
    }

    /// The current length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Overwrite a little-endian `u32` written earlier (an offset table slot).
    pub fn patch_u32(&mut self, offset: usize, value: u32) {
        self.buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// The bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Append a section to `out`: the id byte (bit 7 set when aligned), the
/// payload length, then for an aligned section the alignment and padding
/// relative to the whole file, then the payload.
pub fn write_section(out: &mut Vec<u8>, section: Section, payload: &[u8], alignment: usize) {
    let mut id = section as u8;
    if alignment > 1 {
        id |= 0x80;
    }
    let mut header = Writer::new();
    header.byte(id);
    header.varint(payload.len() as u64);
    if alignment > 1 {
        header.varint(alignment as u64);
    }
    out.extend_from_slice(&header.into_bytes());
    if alignment > 1 {
        while !out.len().is_multiple_of(alignment) {
            out.push(ALIGNMENT_BYTE);
        }
    }
    out.extend_from_slice(payload);
}

/// Write `count`, align to four, reserve `count` little-endian `u32` offset
/// slots and return the position of the first one. Tables of strings, types
/// and debug entries all start this way.
pub fn begin_offset_table(w: &mut Writer, count: usize) -> usize {
    w.varint(count as u64);
    w.align_to(4);
    let first = w.len();
    for _ in 0..count {
        w.le_u32(0);
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_follow_leb128() {
        let mut w = Writer::new();
        w.varint(0);
        w.varint(127);
        w.varint(128);
        w.varint(0xD2);
        w.varint(300);
        assert_eq!(
            w.into_bytes(),
            [0x00, 0x7F, 0x80, 0x01, 0xD2, 0x01, 0xAC, 0x02]
        );
    }

    #[test]
    fn signed_varints_are_zigzag() {
        let mut w = Writer::new();
        w.signed_varint(0);
        w.signed_varint(1);
        w.signed_varint(-1);
        w.signed_varint(128);
        assert_eq!(w.into_bytes(), [0x00, 0x02, 0x01, 0x80, 0x02]);
    }

    #[test]
    fn aligned_sections_pad_relative_to_the_file() {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[13, 5, 1, 0]);
        write_section(&mut out, Section::Func, &[0xAA; 3], 8);
        // id, length, alignment at 12..15, then padding to 16.
        assert_eq!(&out[12..16], &[0x82, 0x03, 0x08, ALIGNMENT_BYTE]);
        assert_eq!(&out[16..19], &[0xAA; 3]);
        write_section(&mut out, Section::EndOfBytecode, &[], 1);
        assert_eq!(out.last(), Some(&0x00));
    }
}
