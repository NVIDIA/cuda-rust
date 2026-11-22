/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-free coverage for scalar padding in multidimensional pointer loads.

use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[cutile::module]
mod pointer_padding {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn rank_one(out: &mut Tensor<f32, { [8] }>, input: *const f32) {
        let ptr: PointerTile<*const f32, { [] }> = pointer_to_tile(input);
        let ptr: PointerTile<*const f32, { [1] }> = ptr.reshape(shape![1]);
        let ptr: PointerTile<*const f32, { [8] }> = ptr.broadcast(shape![8]);
        let mask: Tile<bool, { [8] }> = constant(false, shape![8]);
        let padding: f32 = 3.0f32;
        // All lanes are masked. No input address is dereferenced.
        let (value, _token): (Tile<f32, { [8] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<0>,
            )
        };
        out.store(value);
    }

    #[cutile::entry()]
    unsafe fn rank_two(out: &mut Tensor<f16, { [2, 4] }>, input: *const f16) {
        let ptr: PointerTile<*const f16, { [] }> = pointer_to_tile(input);
        let ptr: PointerTile<*const f16, { [1, 1] }> = ptr.reshape(shape![1, 1]);
        let ptr: PointerTile<*const f16, { [2, 4] }> = ptr.broadcast(shape![2, 4]);
        let indices: Tile<i32, { [8] }> = iota(shape![8]);
        let limit: Tile<i32, { [8] }> = constant(3i32, shape![8]);
        let mask: Tile<bool, { [8] }> = lt_tile(indices, limit);
        let mask: Tile<bool, { [2, 4] }> = mask.reshape(shape![2, 4]);
        let padding: f16 = convert_scalar(-2.0f32);
        // Three enabled lanes read the one input element; the rest use padding.
        let (value, _token): (Tile<f16, { [2, 4] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<0>,
            )
        };
        out.store(value);
    }

    #[cutile::entry()]
    unsafe fn rank_three(out: &mut Tensor<i32, { [2, 2, 4] }>, input: *const i32) {
        let ptr: PointerTile<*const i32, { [] }> = pointer_to_tile(input);
        let ptr: PointerTile<*const i32, { [1, 1, 1] }> = ptr.reshape(shape![1, 1, 1]);
        let ptr: PointerTile<*const i32, { [2, 2, 4] }> = ptr.broadcast(shape![2, 2, 4]);
        let mask: Tile<bool, { [2, 2, 4] }> = constant(false, shape![2, 2, 4]);
        let padding: i32 = -7i32;
        // All lanes are masked. No input address is dereferenced.
        let (value, _token): (Tile<i32, { [2, 2, 4] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(padding),
                None,
                Latency::<0>,
            )
        };
        out.store(value);
    }
}

fn check_padding(entry: &'static str, strides: &'static [i32], padding_type: &'static str) {
    common::with_test_stack(move || {
        let artifacts =
            KernelCompiler::new(pointer_padding::__module_ast_self, "pointer_padding", entry)
                .strides(&[("out", strides)])
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_4)
                .compile()
                .expect("masked pointer load with scalar padding");
        let ir = artifacts.ir_text();
        let load = ir
            .lines()
            .find(|line| line.contains("load_ptr_tko"))
            .expect("pointer load retained");
        assert!(
            load.contains(&format!(", {padding_type} ->")),
            "padding must be a full tile of the loaded type: {load}"
        );
        assert!(!artifacts
            .bytecode()
            .expect("valid padding broadcast")
            .is_empty());
    });
}

#[test]
fn scalar_padding_rank_one_serializes_valid_bytecode() {
    check_padding("rank_one", &[1], "tile<8xf32>");
}

#[test]
fn scalar_padding_rank_two_serializes_valid_bytecode() {
    check_padding("rank_two", &[4, 1], "tile<2x4xf16>");
}

#[test]
fn scalar_padding_rank_three_serializes_valid_bytecode() {
    check_padding("rank_three", &[8, 4, 1], "tile<2x2x4xi32>");
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn scalar_padding_rank_two_executes_with_mixed_mask() {
    common::with_test_stack(|| {
        use cutile::half::f16;
        use cutile::prelude::*;
        let stream = Context::new(0).unwrap().new_stream().unwrap();
        let input = api::full(f16::from_f32(5.0), &[1])
            .sync_on(&stream)
            .unwrap();
        let output = api::full(f16::NAN, &[2, 4]).sync_on(&stream).unwrap();
        // SAFETY: each enabled lane reads the same live input element; there
        // are no writes to it, and it is retained until this launch completes.
        let result =
            unsafe { pointer_padding::rank_two(output.partition([2, 4]), input.device_pointer()) }
                .sync_on(&stream)
                .unwrap();
        let got = result
            .0
            .unpartition()
            .to_host_vec()
            .sync_on(&stream)
            .unwrap();
        let expected = [5.0, 5.0, 5.0, -2.0, -2.0, -2.0, -2.0, -2.0].map(f16::from_f32);
        assert_eq!(got, expected);
    });
}
