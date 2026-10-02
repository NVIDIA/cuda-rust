/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct CUDA launch ABI regressions; the pinned compiler fixture is never launched.

use super::*;

const DESCRIPTOR_ATTRIBUTES: &str =
    "{llvm.align = 64 : i64, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant}";

fn descriptor_kind() -> KernelParameterKind {
    KernelParameterKind::GridConstant {
        size: 128,
        alignment: 64,
    }
}

fn signature(attributes: &str) -> String {
    format!(
        r#""func.func"() <{{arg_attrs = [{{}}, {attributes}, {{}}], function_type = (i64, !llvm.ptr, !llvm.ptr) -> (), sym_name = "probe"}}> ({{"#
    )
}

#[test]
fn grid_constant_signature_preserves_descriptor_bytes_and_alignment() {
    let input = signature(DESCRIPTOR_ATTRIBUTES);
    let abis = expected_kernel_abis(&input, &["probe".into()]).unwrap();
    assert_eq!(
        abis["probe"],
        vec![
            KernelParameterKind::Scalar64,
            descriptor_kind(),
            KernelParameterKind::GlobalPointer64,
        ]
    );
    // The renderer may spell a unit attribute explicitly or canonically elide it.
    let explicit = input.replace("nvvm.grid_constant}", "nvvm.grid_constant = unit}");
    assert_eq!(
        expected_kernel_abis(&explicit, &["probe".into()]).unwrap(),
        abis
    );
}

#[test]
fn grid_constant_signature_rejects_incomplete_or_malformed_attributes() {
    for (original, replacement) in [
        (", nvvm.grid_constant", ""),
        ("nvvm.grid_constant", "nvvm.grid_constant = false"),
        ("64 : i64", "0 : i64"),
        ("64 : i64", "3 : i64"),
        ("64 : i64", "64 : i32"),
        ("128 x i8", "0 x i8"),
        ("128 x i8", "16384 x i8"),
        ("128 x i8", "128 x i16"),
        (
            "llvm.align = 64 : i64",
            "llvm.align = 64 : i64, llvm.align = 64 : i64",
        ),
        ("nvvm.grid_constant", "nvvm.grid_constant, extra = unit"),
        ("!llvm.ptr, !llvm.ptr", "i64, !llvm.ptr"),
        ("[{}, {llvm.align", "[{llvm.align"),
    ] {
        let malformed = signature(DESCRIPTOR_ATTRIBUTES).replace(original, replacement);
        assert!(
            expected_kernel_abis(&malformed, &["probe".into()]).is_err(),
            "accepted malformed ABI: {malformed}"
        );
    }
}

/// Explicit layouts avoid deriving fixture offsets from the code under test.
fn parameter_records(records: &[(u16, u16, u16, u32)], total: u16) -> Vec<u8> {
    let mut info = Vec::new();
    for &(ordinal, offset, size, space) in records.iter().rev() {
        info.extend_from_slice(&[4, 0x17, 12, 0]);
        info.extend_from_slice(&0_u32.to_le_bytes());
        info.extend_from_slice(&ordinal.to_le_bytes());
        info.extend_from_slice(&offset.to_le_bytes());
        let flags = (u32::from(size) << 18) | (0x1f << 12) | (space << 8);
        info.extend_from_slice(&flags.to_le_bytes());
    }
    info.extend_from_slice(&[3, 0x19]);
    info.extend_from_slice(&total.to_le_bytes());
    info
}

#[test]
fn cubin_grid_constant_layout_checks_padding_size_space_and_total() {
    let expected = [
        KernelParameterKind::Scalar64,
        descriptor_kind(),
        descriptor_kind(),
        KernelParameterKind::GlobalPointer64,
        KernelParameterKind::Scalar64,
    ];
    let records = [
        (0, 0, 8, 0),
        (1, 64, 128, 0),
        (2, 192, 128, 0),
        (3, 320, 8, 5),
        (4, 328, 8, 0),
    ];
    validate_kernel_parameter_abi("probe", &parameter_records(&records, 336), &expected).unwrap();
    for replacement in [(1, 8, 128, 0), (1, 64, 8, 0), (1, 64, 128, 5)] {
        let mut malformed = records;
        malformed[1] = replacement;
        let error =
            validate_kernel_parameter_abi("probe", &parameter_records(&malformed, 336), &expected)
                .unwrap_err();
        assert!(error.contains("offset/size/space"), "{error}");
    }
    assert!(
        validate_kernel_parameter_abi("probe", &parameter_records(&records, 280), &expected,)
            .unwrap_err()
            .contains("parameter block")
    );
}

/// Run after installing the fingerprint-pinned compiler; no GPU is required:
/// `CUDA_OXIDE_TEST_CUTLASS_COMPILER=/absolute/path/libCutlassCompiler.so
/// cargo test -p cuda-oxide-codegen cutlass_grid_constant_tma_compiles -- --ignored`
#[test]
#[ignore = "requires the installed official CUTLASS 4.7 compiler; does not require a GPU"]
fn cutlass_grid_constant_tma_compiles() {
    let compiler_path = std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
        .expect("set CUDA_OXIDE_TEST_CUTLASS_COMPILER to the pinned compiler library");
    let mlir = format!(
        r#"module attributes {{gpu.container_module}} {{
 gpu.module @kernels {{
  {}
  ^bb0(%prefix: i64, %desc: !llvm.ptr, %out: !llvm.ptr):
   llvm.inline_asm has_side_effects "prefetch.tensormap [$0];", "l" %desc : (!llvm.ptr) -> ()
   llvm.store %prefix, %out : i64, !llvm.ptr
   func.return
  }}) {{cute.kernel, gpu.kernel}} : () -> ()
 }}
}}"#,
        signature(DESCRIPTOR_ATTRIBUTES),
    );
    let temp = tempfile::tempdir().unwrap();
    let cubin_path = temp.path().join("probe.cubin");
    compile_to_cubin(
        &CutlassBackendConfig::new(compiler_path.into()),
        &mlir,
        "sm_100a",
        &["probe".into()],
        &cubin_path,
        None,
    )
    .unwrap_or_else(|error| panic!("grid-constant compilation failed: {error}\n{mlir}"));
    let cubin = std::fs::read(cubin_path).unwrap();
    // Keeping the address live in a TMA instruction causes plain `byval` to
    // make a 128-byte local copy. With grid_constant the frame must stay empty.
    let file = object::File::parse(cubin.as_slice()).unwrap();
    let info = file.section_by_name(".nv.info").unwrap().data().unwrap();
    let mut frame_sizes = Vec::new();
    let mut offset = 0;
    while offset < info.len() {
        let format = info[offset];
        let attribute = info[offset + 1];
        let size = usize::from(read_u16(info, offset + 2).unwrap());
        if attribute == 0x11 {
            assert_eq!((format, size), (4, 8));
            frame_sizes.push(read_u32(info, offset + 8).unwrap());
        }
        offset += 4 + if format == 4 { size } else { 0 };
    }
    assert_eq!(
        frame_sizes,
        [0],
        "grid-constant descriptor escaped into local memory"
    );
}
