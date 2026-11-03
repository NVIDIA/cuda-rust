/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::api;
use cutile::tile_kernel::{DeviceOp, TileKernel};
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::specialization::{DivHint, SpecializationBits};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Helper: create a DivHint with the default max (16).
fn dh(divisor: i32) -> DivHint {
    DivHint { divisor, max: 16 }
}

static RAW_PTR_DUMP_LOCK: Mutex<()> = Mutex::new(());
const RAW_PTR_SCALAR_DUMP_DIR: &str = "/tmp/cutile_raw_ptr_scalar_mlir";

mod common;

#[cutile::module]
mod spec_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn simple_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    #[cutile::entry(optimization_hints = (sm_120 = (max_divisibility = 8,),))]
    fn capped_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    /// Kernel with a scalar integer param — used to test DivHint on scalars.
    #[cutile::entry(print_ir = true)]
    fn scalar_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, _n: i32) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    /// Kernel with a raw pointer and scalar integer param — used to test that
    /// scalar DivHints still lower when pointer args are present.
    #[cutile::entry(dump_mlir_dir = "/tmp/cutile_raw_ptr_scalar_mlir")]
    unsafe fn raw_ptr_scalar_kernel(_ptr: *mut f32, _n: i32) {}
}

use spec_test_module::{__module_ast_self, raw_ptr_scalar_kernel};

fn compile_with_spec(
    name: &str,
    strides: &[(&str, &[i32])],
    specs: &[(&str, &SpecializationBits)],
) -> String {
    compile_with_spec_and_options(name, strides, specs, &CompileOptions::default())
}

fn compile_with_spec_and_options(
    name: &str,
    strides: &[(&str, &[i32])],
    specs: &[(&str, &SpecializationBits)],
    options: &CompileOptions,
) -> String {
    compile_kernel(name, &[128.to_string()], strides, specs, &[], options)
}

fn compile_kernel(
    name: &str,
    function_generic_args: &[String],
    strides: &[(&str, &[i32])],
    specs: &[(&str, &SpecializationBits)],
    scalar_hints: &[(&str, &DivHint)],
    options: &CompileOptions,
) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "spec_test_module",
        name,
        function_generic_args,
        strides,
        specs,
        scalar_hints,
        None,
        options,
    )
    .expect("Failed to compile")
}

// -- SpecializationBits produces correct assume_div_by in MLIR --

#[test]
fn spec_bits_div_16_produces_div_by_16() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<16>"),
            "Expected div_by<16> for shape divisible by 16.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn spec_bits_div_8_produces_div_by_8() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(8)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(8),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<8>"),
            "Expected div_by<8> for shape divisible by 8.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn no_spec_bits_no_div_by() {
    common::with_test_stack(|| {
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[]);
        println!("{mlir}");
        assert!(
            !mlir.contains("div_by"),
            "Expected no div_by when no spec bits provided.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn spec_bits_div_1_no_div_by() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(1)],
            stride_div: vec![dh(1)],
            stride_one: vec![true],
            base_ptr_div: dh(1),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            !mlir.contains("div_by"),
            "Expected no div_by when all divisors are 1.\nMLIR:\n{mlir}"
        );
    });
}

// -- Cache key differentiation --

#[test]
fn different_spec_bits_different_cache_keys() {
    use cutile::tile_kernel::TileFunctionKey;

    let spec_a = SpecializationBits {
        shape_div: vec![dh(16)],
        stride_div: vec![dh(16)],
        stride_one: vec![true],
        base_ptr_div: dh(16),
        elements_disjoint: true,
    };
    let spec_b = SpecializationBits {
        shape_div: vec![dh(8)],
        stride_div: vec![dh(8)],
        stride_one: vec![true],
        base_ptr_div: dh(8),
        elements_disjoint: true,
    };

    let key_a = TileFunctionKey::builder("m", "f")
        .spec_args(vec![("output".into(), spec_a.clone())])
        .build();
    let key_b = TileFunctionKey::builder("m", "f")
        .spec_args(vec![("output".into(), spec_b.clone())])
        .build();
    let key_a2 = TileFunctionKey::builder("m", "f")
        .spec_args(vec![("output".into(), spec_a)])
        .build();

    assert_ne!(
        key_a, key_b,
        "Different spec bits should produce different cache keys"
    );
    assert_eq!(
        key_a, key_a2,
        "Same spec bits should produce equal cache keys"
    );
}

// -- max_divisibility ceiling --

#[test]
fn entry_max_divisibility_caps_inferred_div() {
    // capped_kernel has max_divisibility=8 in its entry hints.
    // Spec says shape is div by 16, but the hint should cap it to 8.
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(16)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("capped_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<8>"),
            "Expected div_by<8> (capped from 16 by max_divisibility=8).\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<16>"),
            "Should not contain div_by<16> when max_divisibility=8.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn entry_max_divisibility_does_not_inflate() {
    // capped_kernel has max_divisibility=8.
    // Spec says shape is div by 4 — should stay 4 (not inflated to 8).
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(4)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(4),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("capped_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<4>"),
            "Expected div_by<4> (not inflated by max_divisibility=8).\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<8>"),
            "Should not contain div_by<8> when inferred is only 4.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn runtime_max_divisibility_overrides_entry_hint() {
    // simple_kernel has no entry-level max_divisibility.
    // Runtime CompileOptions sets max_divisibility=4, capping spec div=16 to 4.
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(16)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let options = CompileOptions::default().max_divisibility(4);
        let mlir = compile_with_spec_and_options(
            "simple_kernel",
            &[("output", &[1])],
            &[("output", &spec)],
            &options,
        );
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<4>"),
            "Expected div_by<4> from runtime max_divisibility override.\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<16>"),
            "Should not contain div_by<16> when runtime max_divisibility=4.\nMLIR:\n{mlir}"
        );
    });
}

// -- Scalar integer DivHint --

#[test]
fn scalar_int_hint_emits_assume_div_by_in_entry_wrapper() {
    common::with_test_stack(|| {
        let hint = DivHint::from_value(1024);
        let mlir = compile_kernel(
            "scalar_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[("_n", &hint)],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<16>"),
            "Expected scalar hint to emit div_by<16>.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn scalar_int_hint_respects_runtime_max_divisibility() {
    common::with_test_stack(|| {
        let hint = DivHint::from_value(1024);
        let options = CompileOptions::default().max_divisibility(4);
        let mlir = compile_kernel(
            "scalar_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[("_n", &hint)],
            &options,
        );
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<4>"),
            "Expected scalar hint to be capped to div_by<4>.\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<16>"),
            "Should not contain div_by<16> when runtime max_divisibility=4.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn raw_pointer_integer_scalar_hint_emits_assume_div_by() {
    common::with_test_stack(|| {
        let ptr_hint = DivHint::from_ptr(0x1000);
        let scalar_hint = DivHint::from_value(1024);
        let mlir = compile_kernel(
            "raw_ptr_scalar_kernel",
            &[],
            &[],
            &[],
            &[("_ptr", &ptr_hint), ("_n", &scalar_hint)],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert_eq!(
            mlir.matches("assume div_by<16>").count(),
            2,
            "Expected raw pointer and integer scalar hints to both emit div_by<16>.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn raw_pointer_launch_computes_scalar_div_hint() {
    common::with_test_stack(|| {
        let mlir =
            launch_raw_ptr_scalar_kernel_and_read_mlir(12, CompileOptions::default().occupancy(3));

        assert!(
            mlir.contains("entry @raw_ptr_scalar_kernel_entry"),
            "Expected dumped MLIR for raw_ptr_scalar_kernel.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("assume div_by<4>"),
            "Expected launcher-computed scalar hint for n=12 to emit div_by<4>.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("assume div_by<16>"),
            "Expected launcher-computed raw pointer hint to emit div_by<16>.\nMLIR:\n{mlir}"
        );
        assert_eq!(
            mlir.matches("assume div_by<").count(),
            2,
            "Expected raw pointer and scalar argument to receive divisibility assumes.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn raw_pointer_launch_scalar_div_hint_covers_powers_of_two_through_16() {
    common::with_test_stack(|| {
        let cases = [
            (1, None, 4),
            (2, Some(2), 5),
            (4, Some(4), 6),
            (8, Some(8), 7),
            (16, Some(16), 8),
            (32, Some(16), 9),
        ];

        for (n, expected_divisor, occupancy) in cases {
            let mlir = launch_raw_ptr_scalar_kernel_and_read_mlir(
                n,
                CompileOptions::default().occupancy(occupancy),
            );
            assert!(
                mlir.contains("assume div_by<16>"),
                "Expected raw pointer hint for n={n} to emit div_by<16>.\nMLIR:\n{mlir}"
            );
            match expected_divisor {
                Some(divisor) => {
                    let expected = format!("assume div_by<{divisor}>");
                    if divisor == 16 {
                        assert_eq!(
                            mlir.matches(&expected).count(),
                            2,
                            "Expected raw pointer and scalar hints for n={n} to both emit {expected}.\nMLIR:\n{mlir}"
                        );
                    } else {
                        assert!(
                            mlir.contains(&expected),
                            "Expected launcher-computed scalar hint for n={n} to emit {expected}.\nMLIR:\n{mlir}"
                        );
                    }
                    assert_eq!(
                        mlir.matches("assume div_by<").count(),
                        2,
                        "Expected raw pointer and scalar arguments to receive divisibility assumes for n={n}.\nMLIR:\n{mlir}"
                    );
                }
                None => {
                    assert_eq!(
                        mlir.matches("assume div_by<").count(),
                        1,
                        "Expected only the raw pointer argument to receive a divisibility assume for n={n}.\nMLIR:\n{mlir}"
                    );
                }
            }
        }
    });
}

fn launch_raw_ptr_scalar_kernel_and_read_mlir(n: i32, options: CompileOptions) -> String {
    let _lock = RAW_PTR_DUMP_LOCK.lock().expect("lock raw pointer dump dir");
    let dump_dir = Path::new(RAW_PTR_SCALAR_DUMP_DIR);
    let _ = std::fs::remove_dir_all(dump_dir);
    std::fs::create_dir_all(dump_dir).expect("create MLIR dump dir");

    let backing = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; 1]))
        .sync()
        .expect("alloc backing tensor");
    let ptr = backing.device_pointer();

    unsafe { raw_ptr_scalar_kernel(ptr, n) }
        .grid((1, 1, 1))
        .compile_options(options)
        .sync()
        .expect("raw pointer scalar kernel launch");

    let mut mlir = String::new();
    for entry in std::fs::read_dir(dump_dir).expect("read MLIR dump dir") {
        let path = entry.expect("read MLIR dump entry").path();
        if path.extension().is_some_and(|ext| ext == "mlir") {
            mlir.push_str(
                &std::fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("read dumped MLIR {path:?}: {err}")),
            );
        }
    }
    mlir
}
