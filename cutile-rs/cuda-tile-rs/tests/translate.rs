/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exercise the translator built by this crate, without CUDA or a GPU.

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn translate(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CUDA_TILE_RS_TRANSLATE_BIN"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the translator built by cuda-tile-rs");
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn successful(output: Output) -> Vec<u8> {
    assert!(
        output.status.success(),
        "translator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn roundtrip(source: &str, version: u8, preview: bool) -> String {
    let version_flag = format!("--bytecode-version=13.{version}");
    let mut encode_args = vec!["--mlir-to-cudatilebc", &version_flag];
    let mut decode_args = vec!["--cudatilebc-to-mlir"];
    if preview {
        encode_args.push("--preview=simt");
        decode_args.push("--preview=simt");
    }
    let bytes = successful(translate(&encode_args, source.as_bytes()));
    assert_eq!(&bytes[..8], b"\x7fTileIR\0");
    assert_eq!(&bytes[8..12], &[13, version, u8::from(preview), 0]);
    let mlir = successful(translate(&decode_args, &bytes));
    // Check that decoding preserves an encodable module and stabilizes after
    // printing, rather than merely accepting the bytecode header.
    let reencoded = successful(translate(&encode_args, &mlir));
    assert_eq!(successful(translate(&decode_args, &reencoded)), mlir);
    String::from_utf8(mlir).unwrap()
}

#[test]
fn ordinary_bytecode_versions_roundtrip() {
    let source = "cuda_tile.module @kernels { entry @plain() { return } }";
    for version in 1..=5 {
        let mlir = roundtrip(source, version, false);
        assert!(mlir.contains("entry @plain"), "{mlir}");
    }
}

#[test]
fn pointer_classifications_roundtrip_in_13_5() {
    let source = r#"
        cuda_tile.module @kernels {
          entry @classified(
              %uc: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
              %mc: tile<ptr<f32, #cuda_tile.ptr_attr<multicast>>>) {
            return
          }
        }
    "#;
    let mlir = roundtrip(source, 5, false);
    assert!(mlir.contains("ptr<i32, <unicast>>"), "{mlir}");
    assert!(mlir.contains("ptr<f32, <multicast>>"), "{mlir}");
}

#[test]
fn tile_library_interface_and_implementation_roundtrip() {
    // Reuse the bundled upstream fixture, including generic declarations,
    // access annotations, fragment layouts and synchronization.
    let mlir = roundtrip(
        include_str!("../cuda-tile/test/Bytecode/preview_ops.mlir"),
        5,
        true,
    );
    assert!(mlir.contains("preview$func extern @foreign"), "{mlir}");
    assert!(mlir.contains("preview$extern_impl @foreign_sm90"), "{mlir}");
    assert!(mlir.contains("preview$fragment"), "{mlir}");
}

#[test]
fn tile_library_call_roundtrips_with_simt_preview() {
    let source = r#"
        cuda_tile.module @kernels {
          preview$func extern @add_one(
              %x: tile<32xf32>
                {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
              -> tile<32xf32>
          entry @caller() {
            %x = constant <f32: 1.0> : tile<32xf32>
            %result = preview$call @add_one(%x)
                : (!cuda_tile.tile<32xf32>) -> !cuda_tile.tile<32xf32>
            return
          }
        }
    "#;
    let mlir = roundtrip(source, 5, true);
    assert!(mlir.contains("preview$call @add_one"), "{mlir}");
    let rejected = translate(
        &["--mlir-to-cudatilebc", "--bytecode-version=13.5"],
        source.as_bytes(),
    );
    assert!(
        !rejected.status.success(),
        "SIMT preview must be explicitly enabled"
    );
}
