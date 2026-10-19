/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decode the writer's output with the Tile IR disassembler, when one is
//! available: the real reader is the acceptance test of the wire format.
//!
//! Opt in with `OXIDE_TILEBC_TILEIRDISASM=/path/to/tileirdisasm`; the
//! bytecode is written at the version `OXIDE_TILEBC_BYTECODE_VERSION` names
//! (default 13.6), which must be the one that disassembler calls current.
//! Without the variable the test passes without doing anything.

use std::path::PathBuf;
use std::process::Command;

use oxide_tile_export::{TileAccess, TileElem, TileExportDesc, TileParamDesc, TileSync};
use oxide_tilebc::{BytecodeVersion, Library, describe};

fn disassembler() -> Option<PathBuf> {
    std::env::var_os("OXIDE_TILEBC_TILEIRDISASM")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn version() -> BytecodeVersion {
    std::env::var("OXIDE_TILEBC_BYTECODE_VERSION")
        .ok()
        .and_then(|text| BytecodeVersion::parse(&text))
        .unwrap_or(BytecodeVersion::new(13, 6))
}

fn decode(disasm: &PathBuf, name: &str, bytes: &[u8]) -> String {
    let path =
        std::env::temp_dir().join(format!("oxide-tilebc-{}-{name}.tilebc", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let output = Command::new(disasm)
        .arg("--preview=simt")
        .arg(&path)
        .output()
        .expect("tileirdisasm runs");
    let _ = std::fs::remove_file(&path);
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{name}: tileirdisasm failed:\n{}\n{text}",
        String::from_utf8_lossy(&output.stderr)
    );
    text
}

fn scalar(name: &str, access: TileAccess, by_value: bool) -> TileParamDesc {
    TileParamDesc {
        name: name.into(),
        elem: TileElem::F32,
        access,
        footprint: 1,
        by_value,
    }
}

fn exports() -> Vec<TileExportDesc> {
    let relu = TileExportDesc {
        version: oxide_tile_export::CURRENT_VERSION,
        decl: "relu".into(),
        symbol: "relu_impl".into(),
        shape: vec![128],
        params: vec![
            scalar("x", TileAccess::In, true),
            scalar("relu_out", TileAccess::Out, false),
        ],
        sync: TileSync::None,
        arch: vec![],
    };
    let mut warp_sum = relu.clone();
    warp_sum.decl = "warp_sum".into();
    warp_sum.symbol = "warp_sum_impl".into();
    warp_sum.sync = TileSync::Warp;
    warp_sum.params[1].name = "warp_sum_out".into();
    let mut window = relu.clone();
    window.decl = "window4".into();
    window.symbol = "window4_impl".into();
    window.params = vec![TileParamDesc {
        name: "x".into(),
        elem: TileElem::F32,
        access: TileAccess::InOut,
        footprint: 4,
        by_value: false,
    }];
    vec![relu, warp_sum, window]
}

#[test]
fn the_disassembler_reads_a_static_library() {
    let Some(disasm) = disassembler() else {
        return;
    };
    let library: Library = describe::library("lib", &exports()).unwrap();
    let interface = decode(
        &disasm,
        "static-iface",
        &library.interface_bytecode(version()).unwrap(),
    );
    assert!(
        interface.contains("preview$func extern @relu(%arg0: tile<128xf32>"),
        "{interface}"
    );
    let implementation = decode(
        &disasm,
        "static-impl",
        &library.implementation_bytecode(version()).unwrap(),
    );
    assert!(
        implementation.contains("fragmap = <(4, 32, 1):(32, 1, 0)>"),
        "{implementation}"
    );
}

#[test]
fn the_disassembler_reads_a_parametric_library() {
    let Some(disasm) = disassembler() else {
        return;
    };
    let library = describe::parametric_library("lib", &exports()).unwrap();
    let interface = decode(
        &disasm,
        "param-iface",
        &library.interface_bytecode(version()).unwrap(),
    );
    println!("{interface}");
    assert!(
        interface.contains("preview$func extern @relu<[N: index]>(%arg0: tile<[N]xf32>"),
        "{interface}"
    );
    assert!(interface.contains("-> tile<[N]xf32>"), "{interface}");
    let implementation = decode(
        &disasm,
        "param-impl",
        &library.implementation_bytecode(version()).unwrap(),
    );
    println!("{implementation}");
    assert!(
        implementation.contains("preview$extern_impl @relu_impl<[N: index]> implements @relu<[N]>"),
        "{implementation}"
    );
    assert!(
        implementation.contains("fragmap = <(N, 1, 1):(1, 0, 0)>"),
        "{implementation}"
    );
    assert!(
        implementation.contains("@warp_sum_impl<[N: index]> implements @warp_sum<[N]>"),
        "{implementation}"
    );
    assert!(
        implementation.contains(", 32, 1):(32, 1, 0)>"),
        "{implementation}"
    );
    assert!(
        implementation.contains(", 1, 4):(4, 0, 1)>"),
        "{implementation}"
    );
    assert!(
        implementation.contains("synchronization = #cuda_tile<preview$synchronization<warp>>"),
        "{implementation}"
    );
}
