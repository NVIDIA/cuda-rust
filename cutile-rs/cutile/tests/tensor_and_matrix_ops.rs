/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::{
    api, Arc, Device, DeviceOp, DeviceOpReshape, IntoPartition, Tensor, ToHostVec,
};
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_ir::requirements::Feature;

mod common;

#[cutile::module]
mod tensor_and_matrix_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn cat_kernel(output: &mut Tensor<f32, { [8] }>) {
        // Test cat operation - concatenate two tiles
        let source: Tile<f32, { [8] }> = load_tile_mut(output);

        // Split into two halves
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let tile_a: Tile<f32, { [4] }> = extract(source, [idx0]);

        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let tile_b: Tile<f32, { [4] }> = extract(source, [idx1]);

        // Concatenate them back together
        let result: Tile<f32, { [8] }> = cat(tile_a, tile_b, 0i32);

        output.store(result);
    }

    #[cutile::entry()]
    fn extract_kernel(output: &mut Tensor<f32, { [8] }>) {
        // Test extract operation - extract 4-element slices from an 8-element tile
        let source: Tile<f32, { [8] }> = load_tile_mut(output);

        // Extract first half [0:4]
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        // This extract is independent from the second extract below. Each extract slices the source tile independently.
        // The number of slices is determined by the number of indices provided.
        let _slice0: Tile<f32, { [4] }> = extract(source, [idx0]);

        // Extract second half [4:8]
        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let _slice1: Tile<f32, { [4] }> = extract(source, [idx1]);

        // Store original (extract operations will appear in MLIR)
        output.store(source);
    }

    #[cutile::entry()]
    fn mmai_kernel(output: &mut Tensor<i64, { [16, 16] }>) {
        // Test mmai operation - integer matrix multiply-accumulate
        // NOTE: Using i64 tensor because mma output is i32, extended to i64 for storage

        let lhs_shape: Shape<{ [16, 32] }> = Shape::<{ [16, 32] }> {
            dims: &[16i32, 32i32],
        };
        let rhs_shape: Shape<{ [32, 16] }> = Shape::<{ [32, 16] }> {
            dims: &[32i32, 16i32],
        };
        let acc_shape: Shape<{ [16, 16] }> = Shape::<{ [16, 16] }> {
            dims: &[16i32, 16i32],
        };

        let lhs: Tile<i8, { [16, 32] }> = constant(1i8, lhs_shape);
        let rhs: Tile<i8, { [32, 16] }> = constant(1i8, rhs_shape);
        let acc: Tile<i32, { [16, 16] }> = constant(0i32, acc_shape);

        // Perform integer matrix multiply-accumulate
        let result_i32: Tile<i32, { [16, 16] }> = mma(lhs, rhs, acc);

        // Convert to i64 for storage
        let result_i64: Tile<i64, { [16, 16] }> = exti(result_i32);

        output.store(result_i64);
    }

    #[cutile::entry()]
    fn raw_mmai_kernel(output: &mut Tensor<i64, { [16, 16] }>) {
        let lhs: Tile<i8, { [16, 32] }> = constant(1i8, shape![16, 32]);
        let rhs: Tile<i8, { [32, 16] }> = constant(1i8, shape![32, 16]);
        let acc: Tile<i32, { [16, 16] }> = constant(0i32, shape![16, 16]);

        let result_i32: Tile<i32, { [16, 16] }> =
            mmai(lhs, rhs, acc, signedness::Signed, signedness::Unsigned);
        let result_i64: Tile<i64, { [16, 16] }> = exti(result_i32);

        output.store(result_i64);
    }

    #[cutile::entry()]
    fn raw_mmaf_kernel(output: &mut Tensor<f32, { [16, 16] }>) {
        let lhs: Tile<f32, { [16, 8] }> = constant(1.0f32, shape![16, 8]);
        let rhs: Tile<f32, { [8, 16] }> = constant(1.0f32, shape![8, 16]);
        let acc: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);

        let result: Tile<f32, { [16, 16] }> = mmaf(lhs, rhs, acc);

        output.store(result);
    }

    #[cutile::entry()]
    fn transpose_tile_kernel(
        output: &mut Tensor<f32, { [8, 16] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let source: Tile<f32, { [16, 8] }> = load_tile(input, shape![16, 8], [0, 0]);
        let result: Tile<f32, { [8, 16] }> = source.transpose();
        output.store(result);
    }

    #[cutile::entry()]
    fn nvfp4_pack_unpack_kernel(
        output: &mut Tensor<f4e2m1fnx2, { [32] }>,
        input: &Tensor<f4e2m1fnx2, { [-1] }>,
    ) {
        let bytes: Tile<f4e2m1fnx2, { [32] }> = load_tile(input, shape![32], [0]);
        let f4s: Tile<f4e2m1fn, { [64] }> = unpack(bytes);
        let packed: Tile<f4e2m1fnx2, { [32] }> = pack(f4s);
        output.store(packed);
    }

    #[cutile::entry()]
    fn nvfp4_pack_unpack_2d_kernel(
        output: &mut Tensor<f4e2m1fnx2, { [16, 32] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
    ) {
        let bytes_2d: Tile<f4e2m1fnx2, { [16, 32] }> = load_tile(input, shape![16, 32], [0, 0]);
        let f4s: Tile<f4e2m1fn, { [16, 64] }> = bytes_2d.unpack(shape![16, 64]);
        let packed: Tile<f4e2m1fnx2, { [16, 32] }> = f4s.pack(shape![16, 32]);
        output.store(packed);
    }

    #[cutile::entry()]
    fn nvfp4_u8_escape_unpack_kernel(
        output: &mut Tensor<f4e2m1fnx2, { [32] }>,
        input: &Tensor<u8, { [-1] }>,
    ) {
        let bytes: Tile<u8, { [32] }> = load_tile(input, shape![32], [0]);
        let f4s: Tile<f4e2m1fn, { [64] }> = unpack(bytes);
        let packed: Tile<f4e2m1fnx2, { [32] }> = pack(f4s);
        output.store(packed);
    }

    #[cutile::entry()]
    fn pack_unpack_2d_f16_kernel(
        output: &mut Tensor<f16, { [8, 8] }>,
        input: &Tensor<f16, { [-1, -1] }>,
    ) {
        let src: Tile<f16, { [8, 8] }> = load_tile(input, shape![8, 8], [0, 0]);
        let flat: Tile<f16, { [64] }> = reshape(src, shape![64]);
        let packed: Tile<u8, { [128] }> = pack(flat);
        let unpacked: Tile<f16, { [64] }> = unpack(packed);
        let result: Tile<f16, { [8, 8] }> = reshape(unpacked, shape![8, 8]);
        output.store(result);
    }

    #[cutile::entry()]
    fn cross_type_pack_unpack_kernel(
        output: &mut Tensor<i32, { [16] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let src: Tile<f32, { [16] }> = load_tile(input, shape![16], [0]);
        let packed: Tile<u8, { [64] }> = pack(src);
        let result: Tile<i32, { [16] }> = unpack(packed);
        output.store(result);
    }

    #[cutile::entry()]
    fn i32_packed_nibbles_to_i4_roundtrip_kernel(
        output: &mut Tensor<i32, { [16] }>,
        input: &Tensor<i32, { [-1] }>,
    ) {
        let words: Tile<i32, { [16] }> = load_tile(input, shape![16], [0]);
        let bytes: Tile<u8, { [64] }> = pack(words);
        let nibbles: Tile<i4, { [128] }> = unpack(bytes);
        let repacked: Tile<u8, { [64] }> = pack(nibbles);
        let result: Tile<i32, { [16] }> = unpack(repacked);
        output.store(result);
    }

    #[cutile::entry()]
    fn raw_mmaf_scaled_nvfp4_from_i32_words_kernel(
        output: &mut Tensor<f32, { [16, 16] }>,
        lhs_words: &Tensor<i32, { [-1] }>,
        rhs_words: &Tensor<i32, { [-1] }>,
        lhs_scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        rhs_scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let lhs_words_tile: Tile<i32, { [32] }> = load_tile(lhs_words, shape![32], [0]);
        let lhs_bytes: Tile<u8, { [128] }> = pack(lhs_words_tile);
        let lhs_flat: Tile<f4e2m1fn, { [256] }> = unpack(lhs_bytes);
        let lhs: Tile<f4e2m1fn, { [16, 16] }> = lhs_flat.reshape(shape![16, 16]);

        let rhs_words_tile: Tile<i32, { [32] }> = load_tile(rhs_words, shape![32], [0]);
        let rhs_bytes: Tile<u8, { [128] }> = pack(rhs_words_tile);
        let rhs_flat: Tile<f4e2m1fn, { [256] }> = unpack(rhs_bytes);
        let rhs: Tile<f4e2m1fn, { [16, 16] }> = rhs_flat.reshape(shape![16, 16]);

        let lscale: Tile<f8e4m3fn, { [16, 1] }> = load_tile(lhs_scale, shape![16, 1], [0, 0]);
        let rscale: Tile<f8e4m3fn, { [1, 16] }> = load_tile(rhs_scale, shape![1, 16], [0, 0]);
        let acc: Tile<f32, { [16, 16] }> = load_tile_mut(output);

        let result: Tile<f32, { [16, 16] }> = mmaf_scaled(lhs, rhs, acc, lscale, rscale);
        output.store(result);
    }

    #[cutile::entry()]
    fn raw_mmaf_scaled_nvfp4_kernel(
        output: &mut Tensor<f32, { [16, 16] }>,
        lhs_bytes: &Tensor<f4e2m1fnx2, { [-1] }>,
        rhs_bytes: &Tensor<f4e2m1fnx2, { [-1] }>,
        lhs_scale: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        rhs_scale: &Tensor<f8e8m0fnu, { [-1, -1] }>,
    ) {
        let lhs_raw: Tile<f4e2m1fnx2, { [128] }> = load_tile(lhs_bytes, shape![128], [0]);
        let rhs_raw: Tile<f4e2m1fnx2, { [128] }> = load_tile(rhs_bytes, shape![128], [0]);
        let lhs: Tile<f4e2m1fn, { [16, 16] }> = lhs_raw.unpack(shape![16, 16]);
        let rhs: Tile<f4e2m1fn, { [16, 16] }> = rhs_raw.unpack(shape![16, 16]);
        let lscale: Tile<f8e8m0fnu, { [16, 1] }> = load_tile(lhs_scale, shape![16, 1], [0, 0]);
        let rscale: Tile<f8e8m0fnu, { [1, 16] }> = load_tile(rhs_scale, shape![1, 16], [0, 0]);
        let acc: Tile<f32, { [16, 16] }> = load_tile_mut(output);

        let result: Tile<f32, { [16, 16] }> = mmaf_scaled(lhs, rhs, acc, lscale, rscale);
        output.store(result);
    }

    #[cutile::entry()]
    fn raw_mmaf_scaled_nvfp4_e4_scale_kernel(
        output: &mut Tensor<f32, { [16, 16] }>,
        lhs_bytes: &Tensor<f4e2m1fnx2, { [-1] }>,
        rhs_bytes: &Tensor<f4e2m1fnx2, { [-1] }>,
        lhs_scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        rhs_scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let lhs_raw: Tile<f4e2m1fnx2, { [128] }> = load_tile(lhs_bytes, shape![128], [0]);
        let rhs_raw: Tile<f4e2m1fnx2, { [128] }> = load_tile(rhs_bytes, shape![128], [0]);
        let lhs: Tile<f4e2m1fn, { [16, 16] }> = lhs_raw.unpack(shape![16, 16]);
        let rhs: Tile<f4e2m1fn, { [16, 16] }> = rhs_raw.unpack(shape![16, 16]);
        let lscale: Tile<f8e4m3fn, { [16, 1] }> = load_tile(lhs_scale, shape![16, 1], [0, 0]);
        let rscale: Tile<f8e4m3fn, { [1, 16] }> = load_tile(rhs_scale, shape![1, 16], [0, 0]);
        let acc: Tile<f32, { [16, 16] }> = load_tile_mut(output);

        let result: Tile<f32, { [16, 16] }> = mmaf_scaled(lhs, rhs, acc, lscale, rscale);
        output.store(result);
    }

    #[cutile::entry()]
    fn raw_mmaf_scaled_fp8_kernel(
        output: &mut Tensor<f32, { [16, 16] }>,
        lhs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        rhs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        lhs_scale: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        rhs_scale: &Tensor<f8e8m0fnu, { [-1, -1] }>,
    ) {
        let lhs_tile: Tile<f8e4m3fn, { [16, 64] }> = load_tile(lhs, shape![16, 64], [0, 0]);
        let rhs_tile: Tile<f8e4m3fn, { [64, 16] }> = load_tile(rhs, shape![64, 16], [0, 0]);
        let lscale: Tile<f8e8m0fnu, { [16, 2] }> = load_tile(lhs_scale, shape![16, 2], [0, 0]);
        let rscale: Tile<f8e8m0fnu, { [2, 16] }> = load_tile(rhs_scale, shape![2, 16], [0, 0]);
        let acc: Tile<f32, { [16, 16] }> = load_tile_mut(output);

        let result: Tile<f32, { [16, 16] }> = mmaf_scaled(lhs_tile, rhs_tile, acc, lscale, rscale);
        output.store(result);
    }

    #[cutile::entry()]
    fn batch_mmaf_scaled_fp8_kernel(
        output: &mut Tensor<f32, { [2, 16, 16] }>,
        lhs: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        rhs: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        lhs_scale: &Tensor<f8e8m0fnu, { [-1, -1, -1] }>,
        rhs_scale: &Tensor<f8e8m0fnu, { [-1, -1, -1] }>,
    ) {
        let lhs_tile: Tile<f8e4m3fn, { [2, 16, 64] }> =
            load_tile(lhs, shape![2, 16, 64], [0, 0, 0]);
        let rhs_tile: Tile<f8e4m3fn, { [2, 64, 16] }> =
            load_tile(rhs, shape![2, 64, 16], [0, 0, 0]);
        let lscale: Tile<f8e8m0fnu, { [2, 16, 2] }> =
            load_tile(lhs_scale, shape![2, 16, 2], [0, 0, 0]);
        let rscale: Tile<f8e8m0fnu, { [2, 2, 16] }> =
            load_tile(rhs_scale, shape![2, 2, 16], [0, 0, 0]);
        let acc: Tile<f32, { [2, 16, 16] }> = load_tile_mut(output);

        let result: Tile<f32, { [2, 16, 16] }> =
            mmaf_scaled(lhs_tile, rhs_tile, acc, lscale, rscale);
        output.store(result);
    }
}

use tensor_and_matrix_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "tensor_and_matrix_ops_module",
        function_name,
        generics,
        strides,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed.")
}

#[test]
fn compile_cat() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("cat_kernel", &[], &[("output", &[8])]);
        println!("\n=== CAT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= cat"),
            "Expected cat operation in MLIR output"
        );
        assert!(
            module_op_str.contains("dim = 0"),
            "Expected dim=0 attribute in cat operation"
        );

        println!("\n✓ cat operation verified in MLIR output");
    });
}

#[test]
fn compile_extract() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("extract_kernel", &[], &[("output", &[1])]);
        println!("\n=== EXTRACT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= extract"),
            "Expected extract operation in MLIR output"
        );

        println!("\n✓ extract operation verified in MLIR output");
    });
}

#[test]
fn compile_mmai() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("mmai_kernel", &[], &[("output", &[16, 16])]);
        println!("\n=== MMAI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mmai"),
            "Expected mmai operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attributes in mmai operation"
        );
        assert!(
            module_op_str.contains("= exti"),
            "Expected exti for i32->i64 conversion"
        );

        println!("\n✓ mmai operation verified in MLIR output (using i64 tensor workaround)");
    });
}

#[test]
fn compile_raw_mmai() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("raw_mmai_kernel", &[], &[("output", &[16, 16])]);
        println!("\n=== RAW MMAI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mmai"),
            "Expected raw mmai operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed unsigned"),
            "Expected explicit signedness attributes in raw mmai operation"
        );
    });
}

#[test]
fn compile_raw_mmaf() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("raw_mmaf_kernel", &[], &[("output", &[16, 16])]);
        println!("\n=== RAW MMAF MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mmaf"),
            "Expected raw mmaf operation in MLIR output"
        );
    });
}

#[test]
fn compile_transpose_tile() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "transpose_tile_kernel",
            &[],
            &[("output", &[8, 16]), ("input", &[16, 8])],
        );

        assert!(module_op_str.contains("= permute"));
        assert!(module_op_str.contains("tile<16x8xf32>"));
        assert!(module_op_str.contains("tile<8x16xf32>"));
    });
}

#[test]
fn compile_nvfp4_pack_unpack() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "nvfp4_pack_unpack_kernel",
            &[],
            &[("output", &[32]), ("input", &[1])],
        );
        println!("\n=== NVFP4 PACK/UNPACK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= unpack"),
            "Expected raw unpack operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= pack"),
            "Expected raw pack operation in MLIR output"
        );
        assert!(
            module_op_str.contains("f4e2m1fn"),
            "Expected f4e2m1fn tile type in MLIR output"
        );
    });
}

#[test]
fn compile_nvfp4_pack_unpack_2d() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "nvfp4_pack_unpack_2d_kernel",
            &[],
            &[("output", &[32, 1]), ("input", &[32, 1])],
        );

        assert!(module_op_str.contains("= unpack"));
        assert!(module_op_str.contains("= pack"));
        assert!(module_op_str.contains("tile<16x32xi8>"));
        assert!(module_op_str.contains("tile<512xi8>"));
        assert!(module_op_str.contains("tile<1024xf4e2m1fn>"));
    });
}

#[test]
fn compile_nvfp4_u8_escape_unpack() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "nvfp4_u8_escape_unpack_kernel",
            &[],
            &[("output", &[32]), ("input", &[1])],
        );
        println!("\n=== NVFP4 U8 ESCAPE UNPACK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= unpack"),
            "Expected raw unpack operation in MLIR output"
        );
        assert!(
            module_op_str.contains("f4e2m1fn"),
            "Expected f4e2m1fn tile type in MLIR output"
        );
        assert!(
            module_op_str.contains("xi8"),
            "Expected u8 storage tile type in MLIR output"
        );
    });
}

#[test]
fn compile_pack_unpack_2d_f16() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "pack_unpack_2d_f16_kernel",
            &[],
            &[("output", &[8, 8]), ("input", &[8, 8])],
        );

        assert!(module_op_str.contains("= pack"));
        assert!(module_op_str.contains("= unpack"));
        assert!(module_op_str.contains("tile<128xi8>"));
        assert!(module_op_str.contains("tile<64xf16>"));
    });
}

#[test]
fn compile_cross_type_pack_unpack() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "cross_type_pack_unpack_kernel",
            &[],
            &[("output", &[16]), ("input", &[16])],
        );

        assert!(module_op_str.contains("= pack"));
        assert!(module_op_str.contains("= unpack"));
        assert!(module_op_str.contains("tile<16xf32>"));
        assert!(module_op_str.contains("tile<16xi32>"));
    });
}

#[test]
fn compile_i32_packed_nibbles_to_i4_roundtrip() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "i32_packed_nibbles_to_i4_roundtrip_kernel",
            &[],
            &[("output", &[16]), ("input", &[16])],
        );

        assert!(module_op_str.contains("= pack"));
        assert!(module_op_str.contains("= unpack"));
        assert!(module_op_str.contains("tile<16xi32>"));
        assert!(module_op_str.contains("tile<64xi8>"));
        assert!(module_op_str.contains("tile<128xi4>"));
    });
}

#[test]
fn compile_raw_mmaf_scaled_nvfp4_from_i32_words() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "raw_mmaf_scaled_nvfp4_from_i32_words_kernel",
            &[],
            &[
                ("output", &[16, 16]),
                ("lhs_words", &[32]),
                ("rhs_words", &[32]),
                ("lhs_scale", &[16, 1]),
                ("rhs_scale", &[1, 16]),
            ],
        );

        assert!(module_op_str.contains("= pack"));
        assert!(module_op_str.contains("= unpack"));
        assert!(module_op_str.contains("= mmaf_scaled"));
        assert!(module_op_str.contains("tile<32xi32>"));
        assert!(module_op_str.contains("tile<128xi8>"));
        assert!(module_op_str.contains("tile<256xf4e2m1fn>"));
        assert!(module_op_str.contains("tile<16x16xf4e2m1fn>"));
        assert!(module_op_str.contains("tile<16x16xf32>"));
    });
}

#[test]
fn compile_raw_mmaf_scaled_nvfp4() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "raw_mmaf_scaled_nvfp4_kernel",
            &[],
            &[
                ("output", &[16, 16]),
                ("lhs_bytes", &[1]),
                ("rhs_bytes", &[1]),
                ("lhs_scale", &[1, 1]),
                ("rhs_scale", &[16, 1]),
            ],
        );
        println!("\n=== RAW MMAF_SCALED NVFP4 MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mmaf_scaled"),
            "Expected raw mmaf_scaled operation in MLIR output"
        );
        assert!(
            module_op_str.contains("f4e2m1fn"),
            "Expected f4e2m1fn operands in MLIR output"
        );
        assert!(
            module_op_str.contains("f8e8m0fnu"),
            "Expected f8e8m0fnu scales in MLIR output"
        );
    });
}

#[test]
fn compile_raw_mmaf_scaled_nvfp4_e4_scale() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "raw_mmaf_scaled_nvfp4_e4_scale_kernel",
            &[],
            &[
                ("output", &[16, 16]),
                ("lhs_bytes", &[1]),
                ("rhs_bytes", &[1]),
                ("lhs_scale", &[1, 1]),
                ("rhs_scale", &[16, 1]),
            ],
        );

        assert!(module_op_str.contains("= mmaf_scaled"));
        assert!(module_op_str.contains("f4e2m1fn"));
        assert!(module_op_str.contains("f8e4m3fn"));
    });
}

#[test]
fn execute_raw_mmaf_scaled_nvfp4_e4_scale() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[Feature::ScaledMma, Feature::Fp4]) {
            return;
        }

        let device = Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");
        let fp4_one = f4e2m1fnx2::from_nibbles(0x2, 0x2);
        let scale_one = f8e4m3fn(0x38);

        let output = api::zeros::<f32>(&[16, 16])
            .sync_on(&stream)
            .expect("output")
            .partition([16, 16]);
        let lhs: Arc<Tensor<f4e2m1fnx2>> =
            api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; 128]))
                .sync_on(&stream)
                .expect("lhs")
                .into();
        let rhs: Arc<Tensor<f4e2m1fnx2>> =
            api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; 128]))
                .sync_on(&stream)
                .expect("rhs")
                .into();
        let lhs_scale: Arc<Tensor<f8e4m3fn>> =
            api::copy_host_vec_to_device(&Arc::new(vec![scale_one; 16]))
                .reshape(&[16, 1])
                .sync_on(&stream)
                .expect("lhs_scale")
                .into();
        let rhs_scale: Arc<Tensor<f8e4m3fn>> =
            api::copy_host_vec_to_device(&Arc::new(vec![scale_one; 16]))
                .reshape(&[1, 16])
                .sync_on(&stream)
                .expect("rhs_scale")
                .into();

        let (output, _lhs, _rhs, _lhs_scale, _rhs_scale) =
            tensor_and_matrix_ops_module::raw_mmaf_scaled_nvfp4_e4_scale_kernel(
                output, lhs, rhs, lhs_scale, rhs_scale,
            )
            .sync_on(&stream)
            .expect("kernel");

        let host: Vec<f32> = output
            .unpartition()
            .to_host_vec()
            .sync_on(&stream)
            .expect("to_host");
        for (idx, value) in host.iter().enumerate() {
            assert!(
                (*value - 16.0).abs() <= 1e-3,
                "output[{idx}] = {value}, expected 16"
            );
        }
    });
}

#[test]
fn compile_raw_mmaf_scaled_fp8() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "raw_mmaf_scaled_fp8_kernel",
            &[],
            &[
                ("output", &[16, 16]),
                ("lhs", &[16, 64]),
                ("rhs", &[64, 16]),
                ("lhs_scale", &[16, 2]),
                ("rhs_scale", &[2, 16]),
            ],
        );

        assert!(module_op_str.contains("= mmaf_scaled"));
        assert!(module_op_str.contains("f8e4m3fn"));
        assert!(module_op_str.contains("f8e8m0fnu"));
    });
}

#[test]
fn compile_batch_mmaf_scaled_fp8() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "batch_mmaf_scaled_fp8_kernel",
            &[],
            &[
                ("output", &[2, 16, 16]),
                ("lhs", &[2, 16, 64]),
                ("rhs", &[2, 64, 16]),
                ("lhs_scale", &[2, 16, 2]),
                ("rhs_scale", &[2, 2, 16]),
            ],
        );

        assert!(module_op_str.contains("= mmaf_scaled"));
        assert!(module_op_str.contains("tile<2x16x64xf8e4m3fn>"));
        assert!(module_op_str.contains("tile<2x16x16xf32>"));
    });
}
