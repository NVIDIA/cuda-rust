/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile actual Rust-imported primitives IR through production image validation.
use super::*;
use object::ObjectSymbol;

#[test]
#[ignore = "requires source-built primitives evidence and pinned CUTLASS 4.7; no GPU"]
fn cutlass_primitives_grid_constant_abi() {
    let evidence = std::env::var_os("CUDA_OXIDE_PRIMITIVES_EVIDENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/primitives-api-evidence"));
    let mlir = std::fs::read_to_string(evidence.join("primitives.mlir"))
        .expect("run scripts/check-primitives-api-compile.py first");
    let names: Vec<String> = [
        "primitives_grid_constant",
        "primitives_lifecycle",
        "primitives_tma",
        "primitives_sync",
        "primitives_mma",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let abis = expected_kernel_abis(&mlir, &names).unwrap();
    assert_eq!(
        abis[&names[0]][0],
        KernelParameterKind::GridConstant {
            size: 128,
            alignment: 64
        }
    );
    assert_eq!(
        &abis["primitives_tma"][..4],
        &[KernelParameterKind::GridConstant {
            size: 128,
            alignment: 64
        }; 4]
    );
    assert!(abis["primitives_lifecycle"].is_empty());
    assert!(!mlir.contains("\"cute.") && !mlir.contains("!cute."));
    let mut config = CutlassBackendConfig::new(
        std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
            .expect("set pinned compiler library")
            .into(),
    );
    config.profile = cuda_oxide_mlir_export::CutlassPrimitivesMlir22::NAME.into();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("primitives.cubin");
    compile_to_cubin(&config, &mlir, "sm_100a", &names, &path, None).unwrap();
    let cubin = std::fs::read(path).unwrap();
    let file = object::File::parse(cubin.as_slice()).unwrap();
    let symbol = file
        .symbols()
        .find(|s| s.name().ok() == Some("primitives_grid_constant"))
        .unwrap()
        .index()
        .0 as u32;
    let info = file.section_by_name(".nv.info").unwrap().data().unwrap();
    let mut offset = 0;
    let mut frames = vec![];
    while offset < info.len() {
        let format = info[offset];
        let attribute = info[offset + 1];
        let size = usize::from(read_u16(info, offset + 2).unwrap());
        if attribute == 0x11 && read_u32(info, offset + 4).unwrap() == symbol {
            assert_eq!((format, size), (4, 8));
            frames.push(read_u32(info, offset + 8).unwrap());
        }
        offset += 4 + if format == 4 { size } else { 0 };
    }
    assert_eq!(
        frames,
        [0],
        "typed grid-constant descriptor was copied to a local frame"
    );
}

#[test]
#[ignore = "requires pinned CUTLASS 4.7; no GPU"]
fn cutlass_primitives_kernel_discovery() {
    let mut config = CutlassBackendConfig::new(
        std::env::var_os("CUDA_OXIDE_TEST_CUTLASS_COMPILER")
            .expect("set pinned compiler library")
            .into(),
    );
    config.profile = cuda_oxide_mlir_export::CutlassPrimitivesMlir22::NAME.into();
    let temp = tempfile::tempdir().unwrap();
    for (label, attributes) in [
        ("marked", "cute.kernel, gpu.kernel"),
        ("gpu_only", "gpu.kernel"),
    ] {
        let mlir = format!(
            r#"module attributes {{gpu.container_module}} {{
 gpu.module @kernels {{
  "func.func"() <{{function_type = (!llvm.ptr) -> (), sym_name = "primitives_discovery"}}> ({{
  ^bb0(%out: !llvm.ptr):
   %value = arith.constant 19 : i64
   llvm.store %value, %out : i64, !llvm.ptr
   func.return
  }}) {{{attributes}}} : () -> ()
 }}
}}"#
        );
        let result = compile_to_cubin(
            &config,
            &mlir,
            "sm_100a",
            &["primitives_discovery".into()],
            &temp.path().join(format!("{label}.cubin")),
            None,
        );
        if label == "marked" {
            result.unwrap();
        } else {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("missing kernel code section .text.primitives_discovery"),
                "{error}"
            );
        }
    }
}
