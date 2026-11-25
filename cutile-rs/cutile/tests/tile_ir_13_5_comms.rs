/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The runnable unsafe comms example is also the hardware regression test.
mod common;
#[path = "../examples/unsafe_comms.rs"]
mod comms;
#[path = "support/tile_ir_reference.rs"]
mod reference;
use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::BytecodeVersion;

#[test]
fn compile_concurrent_unicast_and_multicast_kernels() {
    common::with_test_stack(|| {
        for kernel in ["unicast_concurrent", "nvls_concurrent"] {
            let artifacts =
                KernelCompiler::new(comms::kernels::__module_ast_self, "kernels", kernel)
                    .target("sm_100")
                    .bytecode_version(BytecodeVersion::V13_5)
                    .compile()
                    .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
    });
}

#[test]
#[ignore = "requires a CUDA 13.5 assembler/backend; no GPU required"]
fn assemble_concurrent_unicast_and_multicast_kernels() {
    common::with_test_stack(|| {
        assert_eq!(
            std::env::var("CUTILE_BYTECODE_VERSION").as_deref(),
            Ok("13.5")
        );
        for kernel in ["unicast_concurrent", "nvls_concurrent"] {
            let artifacts =
                KernelCompiler::new(comms::kernels::__module_ast_self, "kernels", kernel)
                    .target("sm_100")
                    .bytecode_version(BytecodeVersion::V13_5)
                    .compile()
                    .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            let cubin = cutile_compiler::cuda_tile_runtime_utils::compile_tile_ir_module(
                artifacts.module(),
                "sm_100",
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            assert!(cubin.starts_with(b"\x7fELF"));
        }
    });
}

#[test]
#[ignore = "requires CUDA 13.5 and two NVLink GPUs with peer access"]
fn unicast_concurrent_signaling_and_buffer_reuse() {
    common::with_test_stack(|| comms::run(comms::Mode::Unicast, 256));
}

#[test]
#[ignore = "requires CUDA 13.5 and two GPUs on an NVLink SHARP-capable fabric"]
fn multicast_concurrent_signaling_and_buffer_reuse() {
    common::with_test_stack(|| comms::run(comms::Mode::Multicast, 256));
}
