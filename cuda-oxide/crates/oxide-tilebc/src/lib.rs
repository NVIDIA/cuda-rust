/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR bytecode for the tile libraries cuda-oxide ships.
//!
//! A tile library lets a Tile kernel call a SIMT function. cuda-oxide is the
//! SIMT provider: it compiles the function to LTOIR and describes it twice,
//! once as the declaration a Tile kernel calls (the interface) and once as
//! the native symbol with the fragment each parameter receives (the
//! implementation). `tileiras --preview-create-tile-lib` packs the three into
//! one `.tilelib`. This crate writes the two descriptions.
//!
//! ```text
//! Library { declarations, implementations }
//!    |-- interface_bytecode(version)       -> <crate>.iface.tilebc
//!    |-- implementation_bytecode(version)  -> <crate>.impl.tilebc
//!    `-- interface_text(), implementation_text()   readable form
//! ```
//!
//! The descriptions are public-preview content (Tile IR 13.5 and newer).
//! An assembler reads such content only at the version it calls current,
//! the newest numbered version its `--list-versions` prints, so the caller
//! passes that version; [`BytecodeVersion::parse`] reads the listing.
//!
//! ```
//! use oxide_tilebc::*;
//!
//! let tile = TileType::new(vec![128], ScalarType::F32);
//! let decl = Declaration::new("relu", vec![
//!     TileArg::new("x", tile.clone(), Access::In),
//!     TileArg::new("relu_out", tile, Access::Out),
//! ]);
//! let by_value = FragmentType::new(vec![1], ScalarType::F32, MemorySpace::Local,
//!     Layout::scalar_mem(), Layout::elementwise(128));
//! let by_pointer = FragmentType::new(vec![1], ScalarType::F32, MemorySpace::Local,
//!     Layout::contiguous_mem(1), Layout::elementwise(128));
//! let imp = Implementation::new("relu_impl", "relu", vec![
//!     FragmentArg::new("x", by_value, Access::In),
//!     FragmentArg::new("relu_out", by_pointer, Access::Out),
//! ]);
//! let library = Library::new("relu", vec![decl], vec![imp]);
//! library.verify().unwrap();
//! let interface = library.interface_bytecode(BytecodeVersion::PREVIEW_FIRST).unwrap();
//! assert_eq!(&interface[8..12], &[13, 5, 1, 0]);
//! ```

mod encoding;
mod model;
mod text;
mod verify;
mod writer;

pub use model::{
    Access, BytecodeVersion, Declaration, FragmentArg, FragmentType, Implementation, Layout,
    Library, MemorySpace, ScalarType, Synchronization, TileArg, TileType,
};

/// What can go wrong when describing a library.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// The library breaks a structural rule; the message names the part.
    #[error("{0}")]
    Verify(String),
    /// Preview content needs bytecode 13.5 or newer.
    #[error("bytecode version {0} is older than 13.5, the first with public-preview content")]
    Version(BytecodeVersion),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(section id, payload)` in file order, with the framing checked.
    fn sections(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
        assert_eq!(&bytes[..8], b"\x7fTileIR\0");
        let mut pos = 12;
        let mut out = Vec::new();
        loop {
            let id_byte = bytes[pos];
            pos += 1;
            if id_byte & 0x7F == 0 {
                assert_eq!(pos, bytes.len());
                return out;
            }
            let len = read_varint(bytes, &mut pos) as usize;
            if id_byte & 0x80 != 0 {
                let alignment = read_varint(bytes, &mut pos) as usize;
                while pos % alignment != 0 {
                    assert_eq!(bytes[pos], 0xCB);
                    pos += 1;
                }
            }
            out.push((id_byte & 0x7F, bytes[pos..pos + len].to_vec()));
            pos += len;
        }
    }

    fn read_varint(bytes: &[u8], pos: &mut usize) -> u64 {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = bytes[*pos];
            *pos += 1;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    }

    fn section(bytes: &[u8], id: u8) -> Vec<u8> {
        sections(bytes)
            .into_iter()
            .find(|(s, _)| *s == id)
            .map(|(_, payload)| payload)
            .unwrap_or_else(|| panic!("no section {id}"))
    }

    fn strings(bytes: &[u8]) -> Vec<String> {
        let payload = section(bytes, 1);
        let mut pos = 0;
        let count = read_varint(&payload, &mut pos) as usize;
        while pos % 4 != 0 {
            pos += 1;
        }
        let offsets: Vec<usize> = (0..count)
            .map(|i| {
                u32::from_le_bytes(payload[pos + 4 * i..pos + 4 * i + 4].try_into().unwrap())
                    as usize
            })
            .collect();
        let data = &payload[pos + 4 * count..];
        (0..count)
            .map(|i| {
                let end = offsets.get(i + 1).copied().unwrap_or(data.len());
                String::from_utf8(data[offsets[i]..end].to_vec()).unwrap()
            })
            .collect()
    }

    fn index(table: &[String], text: &str) -> u8 {
        u8::try_from(table.iter().position(|t| t == text).expect(text)).unwrap()
    }

    fn tile() -> TileType {
        TileType::new(vec![128], ScalarType::F32)
    }

    fn scale_library() -> Library {
        let decl = Declaration::new(
            "tilelib_scale",
            vec![TileArg::new("arg0", tile(), Access::InOut)],
        );
        let fragment = FragmentType::new(
            vec![1],
            ScalarType::F32,
            MemorySpace::Local,
            Layout::contiguous_mem(1),
            Layout::elementwise(128),
        );
        let imp = Implementation::new(
            "tilelib_scale_impl",
            "tilelib_scale",
            vec![FragmentArg::new("arg0", fragment, Access::InOut)],
        );
        Library::new("tilelib", vec![decl], vec![imp])
    }

    #[test]
    fn header_carries_the_version_and_the_preview_tag() {
        let lib = scale_library();
        let bytes = lib.interface_bytecode(BytecodeVersion::new(13, 6)).unwrap();
        assert_eq!(&bytes[8..12], &[13, 6, 1, 0]);
        assert_eq!(
            lib.interface_bytecode(BytecodeVersion::new(13, 4)),
            Err(Error::Version(BytecodeVersion::new(13, 4)))
        );
        let ids: Vec<u8> = sections(&bytes).into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, [2, 3, 5, 1], "function, debug, type, string");
    }

    /// Extern declaration: normal form, private, body length 0, then the
    /// empty constant parameters. The upstream fixture `tilelib_interface`.
    #[test]
    fn declaration_entry_layout() {
        let bytes = scale_library()
            .interface_bytecode(BytecodeVersion::PREVIEW_FIRST)
            .unwrap();
        let table = strings(&bytes);
        let name = index(&table, "tilelib_scale");
        let access = index(&table, "cuda_tile.preview$access");
        assert_eq!(
            section(&bytes, 2),
            vec![
                0x01, name, 0x02, // one function, name, signature type index
                0x01, // flags: private, not a kernel, normal form
                0x01, // debug list index
                0x06, 0x01, 0x0A, 0x01, access, 0xDF, 0x01,
                0x02, // arg_attrs: [{access = inout}]
                0x00, // body length 0: a declaration
                0x06, 0x00, // const_params: empty array
            ]
        );
        // Types: f32, tile<128xf32>, then (tile) -> (tile).
        let types = section(&bytes, 5);
        let mut pos = 0;
        assert_eq!(read_varint(&types, &mut pos), 3);
        let entries = &types[4 + 4 * 3..];
        assert_eq!(entries[0], 0x07, "f32");
        assert_eq!(
            &entries[1..12],
            &[0x0D, 0x00, 0x01, 128, 0, 0, 0, 0, 0, 0, 0],
            "tile<128xf32>"
        );
        assert_eq!(
            &entries[12..],
            &[0x10, 0x01, 0x01, 0x01, 0x01],
            "(tile) -> (tile)"
        );
    }

    /// Implementation: form bits 01, private, the op payload in place of a
    /// body, body length 0. The upstream fixture `tilelib_implementation`.
    #[test]
    fn implementation_entry_layout() {
        let lib = scale_library();
        let bytes = lib
            .implementation_bytecode(BytecodeVersion::PREVIEW_FIRST)
            .unwrap();
        let table = strings(&bytes);
        let symbol = index(&table, "tilelib_scale_impl");
        let implements = index(&table, "tilelib_scale");
        let access = index(&table, "cuda_tile.preview$access");
        let mut expected = vec![
            0x01, symbol, 0x02, // one function, symbol, signature type index
            0x05, // flags: private, form extern_impl
            0x01, // debug list index
            0x04, // payload flags: arg_attrs only
            symbol, implements, // sym_name, implements
            0x02,       // function_type index
            0x01, 0x0A, 0x01, access, 0xDF, 0x01,
            0x02, // arg_attrs inline: [{access = inout}]
            0x01, 0x00, // one region, zero blocks
            0x00, // body length 0
        ];
        expected.resize(24, 0xCB);
        assert_eq!(section(&bytes, 2), expected);

        // The fragment type entry at 13.5: tagged symbolic shape holding one
        // integer per dimension (the 13.5 assembler requires full rank), static shape
        // [1], element f32 (index 0), local, (1):(1), (128, 1, 1):(1, 0, 0),
        // alignment 1.
        let types = section(&bytes, 5);
        let entries = &types[4 + 4 * 3..];
        assert_eq!(
            &entries[1..],
            &[
                0xD2, 0x01, 0xD8, 0x01, 0x01, 0xD1, 0x01, 0x02, 0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0x00,
                0x00, //
                0xD9, 0x01, 0x01, 0xD1, 0x01, 0x02, 0x01, 0xD1, 0x01, 0x02, //
                0xD9, 0x01, 0x03, 0xD1, 0x01, 0x80, 0x02, 0xD1, 0x01, 0x02, 0xD1, 0x01,
                0x02, //
                0x03, 0xD1, 0x01, 0x02, 0xD1, 0x01, 0x00, 0xD1, 0x01, 0x00, //
                0xD1, 0x01, 0x02, //
                0x10, 0x01, 0x01, 0x00, // (fragment) -> ()
            ]
        );
    }

    /// A warp-synchronous implementation sets bit 8 and writes the value
    /// after the argument attributes; arch and frontendABI take bits 6 and 7.
    #[test]
    fn optional_clauses_set_their_flag_bits() {
        let mut lib = scale_library();
        let imp = &mut lib.implementations[0];
        imp.args[0].fragment.fragmap = Layout::warp_lanes(128);
        imp.synchronization = Some(Synchronization::Warp);
        imp.arch = vec!["sm_120".into()];
        imp.frontend_abi = vec![("bf16".into(), "i16".into())];
        lib.verify().unwrap();
        let bytes = lib
            .implementation_bytecode(BytecodeVersion::PREVIEW_FIRST)
            .unwrap();
        let table = strings(&bytes);
        let func = section(&bytes, 2);
        // flags = bits 2, 6, 7, 8 = 0x1C4 -> varint C4 03
        assert_eq!(&func[5..7], &[0xC4, 0x03]);
        let after_attrs = 7 + 3 + 7;
        assert_eq!(
            &func[after_attrs..after_attrs + 10],
            &[
                0x01,
                0x05,
                index(&table, "sm_120"), // arch: one tagged string
                0x01,
                index(&table, "bf16"),
                0x05,
                index(&table, "i16"), // frontendABI
                0x01,                 // synchronization = warp
                0x01,
                0x00, // region
            ]
        );
    }

    #[test]
    fn debug_section_holds_one_unknown_location_per_function() {
        let bytes = scale_library()
            .interface_bytecode(BytecodeVersion::PREVIEW_FIRST)
            .unwrap();
        assert_eq!(
            section(&bytes, 3),
            vec![
                0x01, 0xCB, 0xCB, 0xCB, // one function, aligned
                0, 0, 0, 0, // its id list starts at 0
                0x01, 0xCB, 0xCB, 0xCB, 0xCB, 0xCB, 0xCB, 0xCB, // one id, aligned to 8
                0, 0, 0, 0, 0, 0, 0, 0, // id 0: unknown location
                0x01, 0xCB, 0xCB, 0xCB, // one attribute entry
                0, 0, 0, 0,    // at offset 0
                0x00, // tag 0: unknown
            ]
        );
    }

    #[test]
    fn text_matches_the_reference_form() {
        let lib = scale_library();
        assert_eq!(
            lib.interface_text(),
            "cuda_tile.module @tilelib {\n  cuda_tile.preview$func extern @tilelib_scale(%arg0: \
             !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) \
             -> !cuda_tile.tile<128xf32>\n}\n"
        );
        assert!(lib.implementation_text().contains(
            "@tilelib_scale_impl implements @tilelib_scale(%arg0: !cuda_tile.preview$fragment<1xf32, \
             local, mem_layout = #cuda_tile.preview$layout<(1):(1)>, fragmap = \
             #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {cuda_tile.preview$access = \
             #cuda_tile.preview$access<inout>})\n"
        ));
    }

    #[test]
    fn versions_parse_from_the_assembler_listing() {
        assert_eq!(
            BytecodeVersion::parse("13.5"),
            Some(BytecodeVersion::new(13, 5))
        );
        assert_eq!(
            BytecodeVersion::parse(" 13.6.1 "),
            Some(BytecodeVersion::new(13, 6))
        );
        assert_eq!(BytecodeVersion::parse("internal"), None);
        assert_eq!(BytecodeVersion::parse("13"), None);
        assert!(BytecodeVersion::new(13, 6) > BytecodeVersion::PREVIEW_FIRST);
        assert_eq!(BytecodeVersion::new(13, 6).to_string(), "13.6");
    }
}
