/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod global_memory_module {
    use cutile::core::*;

    const INITIAL_COUNTER: i32 = 7;

    static COUNTER: Global<AtomicI32, { [] }> = Global::new(INITIAL_COUNTER);
    static FLOAT_ACCUM: Global<AtomicF32, { [] }> = Global::new(0.0f32);

    #[cutile::entry()]
    fn load_global_kernel(out: &mut Tensor<i32, { [1] }>) {
        let (value, _token) = COUNTER.load(ordering::Acquire, scope::Device);
        out.store(value.reshape(shape![1]));
    }

    #[cutile::entry()]
    fn store_global_kernel(out: &mut Tensor<i32, { [1] }>) {
        let value = constant(3i32, shape![]);
        let _token = COUNTER.store(value, ordering::Release, scope::Device);
        out.store(value.reshape(shape![1]));
    }

    #[cutile::entry()]
    fn atomic_add_global_kernel(out: &mut Tensor<f32, { [1] }>) {
        let increment = constant(1.0f32, shape![]);
        let (old, _token) = FLOAT_ACCUM.atomic_add(increment, ordering::AcqRel, scope::Device);
        out.store(old.reshape(shape![1]));
    }
}

#[cutile::module]
mod bad_static_module {
    use cutile::core::*;

    static BAD_STATIC: i32 = 0;

    #[cutile::entry()]
    fn kernel(out: &mut Tensor<i32, { [1] }>) {
        let value = constant(1i32, shape![1]);
        out.store(value);
    }
}

use bad_static_module::__module_ast_self as bad_static_module_ast;
use global_memory_module::__module_ast_self as global_memory_module_ast;

fn compile_global_kernel(name: &str) -> String {
    common::compile_to_ir(
        global_memory_module_ast,
        "global_memory_module",
        name,
        &[],
        &[("out", &[1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed to compile.")
}

#[test]
fn global_load_lowers_to_get_global_and_load_ptr() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("load_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_COUNTER"));
        assert!(mlir.contains("tile<1xi32>"));
        assert!(mlir.contains("get_global @global_memory_module_COUNTER"));
        assert!(mlir.contains("load_ptr_tko"));
        assert!(mlir.contains("load_ptr_tko acquire device"));
    });
}

#[test]
fn global_store_lowers_to_get_global_and_store_ptr() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("store_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_COUNTER"));
        assert!(mlir.contains("tile<1xi32>"));
        assert!(mlir.contains("get_global @global_memory_module_COUNTER"));
        assert!(mlir.contains("store_ptr_tko"));
        assert!(mlir.contains("store_ptr_tko release device"));
    });
}

#[test]
fn global_atomic_add_lowers_to_atomic_rmw() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("atomic_add_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_FLOAT_ACCUM"));
        assert!(mlir.contains("tile<1xf32>"));
        assert!(mlir.contains("get_global @global_memory_module_FLOAT_ACCUM"));
        assert!(mlir.contains("atomic_rmw_tko"));
        assert!(mlir.contains("atomic_rmw_tko acq_rel device"));
    });
}

#[test]
fn non_global_static_is_rejected() {
    common::with_test_stack(|| {
        let err = common::compile_to_ir(
            bad_static_module_ast,
            "bad_static_module",
            "kernel",
            &[],
            &[("out", &[1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect_err("expected static rejection");
        let msg = err.to_string();
        assert!(msg.contains("only `static NAME: Global<A, { [] }>` items are supported"));
    });
}

// Feed deliberately invalid source to the JIT as well as checking rustc's
// public API boundary in ui.rs. The JIT must not silently emit a weak access.
fn compile_probe(source: &str) -> Result<String, cutile_compiler::error::JITError> {
    common::compile_to_ir(
        || cutile_compiler::ast::Module::new("probe", syn::parse_str(source).unwrap()),
        "probe",
        "kernel",
        &[],
        &[("out", &[1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
}

fn probe_source(atomic: &str, value: &str, operation: &str) -> String {
    format!(
        r#"mod probe {{
        use cutile::core::*;
        static VALUE: Global<{atomic}, {{ [] }}> = Global::new(0{value});
        #[cutile::entry()]
        fn kernel(out: &mut Tensor<{value}, {{ [1] }}>) {{
            {operation}
        }}
    }}"#
    )
}

#[test]
fn every_atomic_payload_lowers_with_scalar_storage() {
    common::with_test_stack(|| {
        for (atomic, value, ir_type, alignment) in [
            ("AtomicI32", "i32", "i32", 4),
            ("AtomicU32", "u32", "i32", 4),
            ("AtomicI64", "i64", "i64", 8),
            ("AtomicU64", "u64", "i64", 8),
            ("AtomicF32", "f32", "f32", 4),
            ("AtomicF64", "f64", "f64", 8),
        ] {
            let source = probe_source(atomic, value, &format!(
                "let (old, _) = VALUE.atomic_add(constant(1{value}, shape![]), ordering::Relaxed, scope::Device);
                 out.store(old.reshape(shape![1]));"
            ));
            let ir = compile_probe(&source).unwrap_or_else(|e| panic!("{atomic}: {e}"));
            assert!(ir.contains(&format!("tile<1x{ir_type}>")), "{ir}");
            assert!(ir.contains("atomic_rmw_tko relaxed device"), "{ir}");
            assert!(ir.contains(&format!("alignment = {alignment}")), "{ir}");
        }
    });
}

#[test]
fn jit_rejects_non_atomic_payloads_and_lookalike_markers() {
    common::with_test_stack(|| {
        let op = "out.store(constant(0i32, shape![1]));";
        for atomic in ["i32", "bool", "AtomicI8"] {
            let err = compile_probe(&probe_source(atomic, "i32", op)).unwrap_err();
            assert!(err.to_string().contains("sealed Atomic"), "{err}");
        }
        let source = probe_source("AtomicI32", "i32", op)
            .replace("static VALUE:", "struct AtomicI32; static VALUE:");
        let err = compile_probe(&source).unwrap_err();
        assert!(err.to_string().contains("sealed Atomic"), "{err}");
    });
}

#[test]
fn jit_rejects_weak_and_block_scoped_global_accesses() {
    common::with_test_stack(|| {
        for (operation, expected) in [
            ("let _ = VALUE.load(ordering::Weak, scope::Device);", "load ordering"),
            ("let _ = VALUE.store(constant(1i32, shape![]), ordering::Weak, scope::Device);", "store ordering"),
            ("let _ = VALUE.atomic_add(constant(1i32, shape![]), ordering::Weak, scope::Device);", "atomic ordering"),
            ("let _ = VALUE.load(ordering::Relaxed, scope::TileBlock);", "Global memory scope"),
            ("let _ = VALUE.store(constant(1i32, shape![]), ordering::Release, scope::TileBlock);", "Global memory scope"),
            ("let _ = VALUE.atomic_add(constant(1i32, shape![]), ordering::Relaxed, scope::TileBlock);", "Global memory scope"),
        ] {
            let err = compile_probe(&probe_source("AtomicI32", "i32", operation)).unwrap_err();
            assert!(err.to_string().contains(expected), "{operation}: {err}");
        }
    });
}

#[test]
fn atomic_marker_alias_and_system_scope_are_supported() {
    common::with_test_stack(|| {
        let source = probe_source("Counter", "i32",
            "let (old, _) = VALUE.load(ordering::Relaxed, scope::System); out.store(old.reshape(shape![1]));")
            .replace("static VALUE:", "type Counter = AtomicI32; static VALUE:");
        let ir = compile_probe(&source).unwrap();
        assert!(ir.contains("load_ptr_tko relaxed sys"), "{ir}");
    });
}

#[test]
fn scoped_global_operations_do_not_add_fences() {
    common::with_test_stack(|| {
        for name in [
            "load_global_kernel",
            "store_global_kernel",
            "atomic_add_global_kernel",
        ] {
            let ir = compile_global_kernel(name);
            assert!(!ir.contains("memory_fence"), "{ir}");
        }
    });
}
