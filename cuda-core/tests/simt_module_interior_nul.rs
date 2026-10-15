/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-loading APIs return `CUDA_ERROR_INVALID_VALUE` for a name or source
//! with an interior NUL byte. A single NUL in the final byte is a terminator
//! and loads.

use cuda_core::CudaContext;

const INVALID_VALUE: cuda_bindings::CUresult =
    cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE;

const VALID_PTX: &str = "\
.version 7.0
.target sm_50
.address_size 64
.visible .global .align 4 .u32 g;
.visible .entry k() { ret; }
";

const BAD_STRINGS: [&str; 2] = ["a\0b", "a\0\0"];

#[test]
fn load_module_from_ptx_src_rejects_interior_nul() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    for input in BAD_STRINGS {
        let err = ctx
            .load_module_from_ptx_src(input)
            .err()
            .unwrap_or_else(|| panic!("accepted {input:?}"));
        assert_eq!(err.0, INVALID_VALUE, "ptx_src {input:?}");
    }
}

#[test]
fn load_module_from_ptx_src_accepts_a_terminated_source() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let terminated = format!("{VALID_PTX}\0");
    ctx.load_module_from_ptx_src(&terminated)
        .expect("PTX with its terminator failed to load");
}

#[test]
fn load_module_from_file_rejects_interior_nul() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    for input in BAD_STRINGS {
        let err = ctx
            .load_module_from_file(input)
            .err()
            .unwrap_or_else(|| panic!("accepted {input:?}"));
        assert_eq!(err.0, INVALID_VALUE, "filename {input:?}");
    }
}

#[test]
fn function_and_global_lookup_reject_interior_nul() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let module = ctx
        .load_module_from_ptx_src(VALID_PTX)
        .expect("valid PTX failed to load");

    // Both names resolve with and without a terminator. Each rejection below
    // therefore comes from the NUL check.
    for name in ["k", "k\0"] {
        module.load_function(name).expect("entry `k` not found");
    }
    for name in ["g", "g\0"] {
        module.get_global(name).expect("global `g` not found");
    }

    for input in BAD_STRINGS {
        let err = module
            .load_function(input)
            .err()
            .unwrap_or_else(|| panic!("load_function accepted {input:?}"));
        assert_eq!(err.0, INVALID_VALUE, "load_function {input:?}");

        let err = module
            .get_global(input)
            .err()
            .unwrap_or_else(|| panic!("get_global accepted {input:?}"));
        assert_eq!(err.0, INVALID_VALUE, "get_global {input:?}");
    }
}
