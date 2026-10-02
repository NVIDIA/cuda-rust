/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to NVIDIA libmathdx
//!
//! libmathdx is a runtime code-generation library for cuFFTDx, cuBLASDx, and cuSolverDx.
//! It provides a C API to generate optimized LTOIR (Link-Time Optimized IR) for MathDx
//! operations at runtime.
//!
//! # Configuration
//!
//! Set `LIBMATHDX_PATH` to your libmathdx installation directory before building.
//! The recommended method is to edit `.cargo/config.toml`:
//!
//! ```toml
//! LIBMATHDX_PATH = "/path/to/libmathdx/install"
//! ```
//!
//! The directory should contain `include/` and `lib/` subdirectories.
//!
//! # Example
//!
//! ```ignore
//! use libmathdx_sys::*;
//!
//! unsafe {
//!     // Create a cuFFTDx descriptor
//!     let mut desc: cufftdxDescriptor = 0;
//!     cufftdxCreateDescriptor(&mut desc);
//!
//!     // Configure FFT parameters
//!     cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_SIZE, 32);
//!     cufftdxSetOperatorInt64(desc, cufftdxOperatorType_t_CUFFTDX_OPERATOR_DIRECTION,
//!                             cufftdxDirection_t_CUFFTDX_DIRECTION_FORWARD as i64);
//!     // ... more configuration ...
//!
//!     // Create code handle and configure for LTOIR output
//!     let mut code: commondxCode = 0;
//!     commondxCreateCode(&mut code);
//!     commondxSetCodeOptionInt64(code, commondxOption_t_COMMONDX_OPTION_TARGET_SM, 900);
//!     commondxSetCodeOptionInt64(code, commondxOption_t_COMMONDX_OPTION_CODE_CONTAINER,
//!                               commondxCodeContainer_t_COMMONDX_CODE_CONTAINER_LTOIR as i64);
//!
//!     // Finalize descriptor into code handle
//!     cufftdxFinalizeCode(code, desc);
//!
//!     // Extract LTOIR bytes
//!     let mut lto_size: usize = 0;
//!     commondxGetCodeLTOIRSize(code, &mut lto_size);
//!     let mut lto = vec![0u8; lto_size];
//!     commondxGetCodeLTOIR(code, lto_size, lto.as_mut_ptr() as *mut _);
//!
//!     // Cleanup
//!     cufftdxDestroyDescriptor(desc);
//!     commondxDestroyCode(code);
//! }
//! ```
//!
//! # Safety
//!
//! This crate provides raw FFI bindings. All functions are unsafe and require careful
//! handling of pointers and memory management according to the libmathdx documentation.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Include the bindgen-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
