// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records, at build time, the LLVM major this crate's textual IR is written
//! for.
//!
//! The standalone codegen API writes its own LLVM text through `llvm-export`,
//! so the compiler on the user's `PATH` is not its producer (#1300). What the
//! text is written for is fixed when this crate is compiled: the toolchain
//! building it supplies the `llvm-tools` the exporter's output is validated
//! against, so its LLVM major is the identity the tool resolver compares `llc`
//! to.
//!
//! Recording it here is the point. The resolver finds `llc` through `PATH`
//! too, so probing `rustc -vV` at run time could report the very toolchain
//! the comparison exists to catch: a backend compiled against LLVM 23 and run
//! with an LLVM 22 toolchain first on `PATH` would select LLVM 22 `llc`, probe
//! LLVM 22 `rustc`, and fall silent. A build-time value cannot drift that way.
//!
//! `CUDA_OXIDE_CODEGEN_LLVM_MAJOR` is left unset when the probe cannot answer,
//! which `option_env!` turns into "no stated producer" and the resolver
//! answers with silence rather than a guess.

use std::{ffi::OsString, process::Command};

fn main() {
    // Cargo points RUSTC at the toolchain compiling this crate, which is the
    // one whose `llvm-tools` the exporter's text is validated against. Not
    // `"rustc"`: that would re-read PATH and reintroduce the drift above.
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    println!("cargo:rerun-if-env-changed=RUSTC");

    if let Some(major) = Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let banner = String::from_utf8_lossy(&output.stdout).into_owned();
            cuda_target_spec::parse_rustc_llvm_major(&banner)
        })
    {
        println!("cargo:rustc-env=CUDA_OXIDE_CODEGEN_LLVM_MAJOR={major}");
    }
}
