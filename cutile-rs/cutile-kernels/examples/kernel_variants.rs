/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[path = "support/variants.rs"]
mod variants;

fn main() {
    let kind = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if !["all", "gemm", "attention", "moe"].contains(&kind.as_str()) {
        eprintln!("usage: kernel_variants [all|gemm|attention|moe]");
        std::process::exit(2);
    }
    // The JIT frontend uses recursive visitors; match the test stack size.
    std::thread::Builder::new()
        .stack_size(32_000_000)
        .spawn(move || {
            if kind == "all" || kind == "gemm" {
                variants::gemm().expect("GEMM validation");
            }
            if kind == "all" || kind == "attention" {
                variants::attention().expect("attention validation");
            }
            if kind == "all" || kind == "moe" {
                variants::moe().expect("MoE validation");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
