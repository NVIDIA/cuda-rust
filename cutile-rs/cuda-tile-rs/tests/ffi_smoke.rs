/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke tests against the raw bindgen'd MLIR and cuda-tile C-API.
//!
//! Exercises: dialect registration, loading, parsing (module / operation /
//! attribute / type), and a few builder APIs (integer type construction).

use cuda_tile_rs::ffi::*;
use cuda_tile_rs::{attribute_parse, module_parse, register_all_passes, string_ref, type_parse};

/// Create an `MlirContext` with cuda-tile dialects registered and loaded.
fn make_context() -> MlirContext {
    unsafe {
        let registry = mlirDialectRegistryCreate();
        assert!(
            !(registry.ptr as *const ()).is_null(),
            "mlirDialectRegistryCreate returned null"
        );
        mlirCudaTileRegisterAllDialects(registry);

        let ctx = mlirContextCreateWithRegistry(registry, /* threadingEnabled */ false);
        assert!(
            !(ctx.ptr as *const ()).is_null(),
            "mlirContextCreateWithRegistry returned null"
        );
        mlirContextLoadAllAvailableDialects(ctx);

        mlirDialectRegistryDestroy(registry);
        ctx
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

#[test]
fn register_all_dialects_and_passes() {
    unsafe {
        let registry = mlirDialectRegistryCreate();
        assert!(!(registry.ptr as *const ()).is_null());
        mlirCudaTileRegisterAllDialects(registry);
        register_all_passes();
        mlirDialectRegistryDestroy(registry);
    }
}

#[test]
fn cuda_tile_dialect_is_loadable() {
    unsafe {
        let ctx = make_context();
        let dialect = mlirContextGetOrLoadDialect(ctx, string_ref(b"cuda_tile"));
        assert!(
            !(dialect.ptr as *const ()).is_null(),
            "cuda_tile dialect failed to load — registration did not happen"
        );
        mlirContextDestroy(ctx);
    }
}

// ---------------------------------------------------------------------------
// Module parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_empty_module() {
    unsafe {
        let ctx = make_context();
        let module = module_parse(ctx, "module { }").expect("empty module should parse");
        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_nested_cuda_tile_module() {
    unsafe {
        let ctx = make_context();
        // Outer builtin module containing a cuda_tile.module — exercises the
        // dialect's top-level construct.
        let src = r#"
            module {
              cuda_tile.module @kernels {
              }
            }
        "#;
        let module = module_parse(ctx, src).expect("cuda_tile module should parse");
        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_bad_module_returns_none() {
    unsafe {
        let ctx = make_context();
        let module = module_parse(ctx, "this is not valid mlir");
        assert!(module.is_none(), "garbage input should not parse");
        mlirContextDestroy(ctx);
    }
}

// ---------------------------------------------------------------------------
// Operation / attribute / type parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_cuda_tile_tensor_view_type() {
    unsafe {
        let ctx = make_context();
        // Cuda-tile tensor_view is a dialect-specific type.
        let ty = type_parse(ctx, "!cuda_tile.tensor_view<32xf32, strides=[1]>");
        assert!(ty.is_some(), "cuda_tile tensor_view type should parse");
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_cuda_tile_pointer_type() {
    unsafe {
        let ctx = make_context();
        let ty = type_parse(ctx, "!cuda_tile.ptr<f32>");
        assert!(ty.is_some(), "cuda_tile pointer type should parse");
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_integer_attribute() {
    unsafe {
        let ctx = make_context();
        let attr = attribute_parse(ctx, "42 : i32").expect("int attr should parse");
        // No explicit destroy for attributes — they're uniqued in context.
        let _ = attr;
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_integer_type() {
    unsafe {
        let ctx = make_context();
        let ty = type_parse(ctx, "i32").expect("i32 should parse");
        assert!(mlirTypeIsAInteger(ty), "parsed type should be integer");
        assert_eq!(mlirIntegerTypeGetWidth(ty), 32);
        mlirContextDestroy(ctx);
    }
}

#[test]
fn parse_cuda_tile_type() {
    unsafe {
        let ctx = make_context();
        // Sanity: the cuda_tile dialect is registered, so its types should
        // be parseable. Exact spelling may be adjusted if upstream syntax
        // evolves.
        let ty = type_parse(ctx, "!cuda_tile.tile<4xf32>");
        assert!(ty.is_some(), "cuda_tile tile type should parse");
        mlirContextDestroy(ctx);
    }
}

// ---------------------------------------------------------------------------
// Builder API
// ---------------------------------------------------------------------------

#[test]
fn build_integer_type_via_c_api() {
    unsafe {
        let ctx = make_context();
        let i32_ty = mlirIntegerTypeGet(ctx, 32);
        assert!(
            !(i32_ty.ptr as *const ()).is_null(),
            "mlirIntegerTypeGet returned null"
        );
        assert!(
            mlirTypeIsAInteger(i32_ty),
            "built type is not an integer type"
        );
        assert_eq!(mlirIntegerTypeGetWidth(i32_ty), 32);
        mlirContextDestroy(ctx);
    }
}
