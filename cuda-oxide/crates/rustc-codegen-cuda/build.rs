// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records, at build time, the LLVM major of the compiler that will run this
//! backend.
//!
//! This crate is a `dylib` loaded by rustc as a codegen backend, and it links
//! `librustc_driver` from the toolchain that compiles it. That toolchain is
//! therefore the compiler that produces the MIR this backend lowers, and its
//! LLVM major is what the tool resolver compares the chosen `llc` to (#1300).
//!
//! It has to be recorded here rather than probed at run time. The resolver
//! finds `llc` through `PATH`, and `rustc -vV` would read that same `PATH`:
//! a backend built against LLVM 23 but run with an LLVM 22 toolchain first on
//! `PATH` would select LLVM 22 `llc`, probe LLVM 22 `rustc`, agree with
//! itself and report nothing -- which is exactly the mismatch #1234 spent a
//! triage round on. A build-time value cannot drift to match the wrong tool.
//!
//! `CUDA_OXIDE_BACKEND_LLVM_MAJOR` is left unset when the probe cannot
//! answer, which `option_env!` turns into "no stated producer" and the
//! resolver answers with silence rather than a guess.

use std::{ffi::OsString, process::Command};

fn main() {
    // Cargo points RUSTC at the toolchain compiling this crate -- the one
    // whose `librustc_driver` this dylib links, hence the one that loads it.
    // Not `"rustc"`: that would re-read PATH and reintroduce the drift above.
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
        println!("cargo:rustc-env=CUDA_OXIDE_BACKEND_LLVM_MAJOR={major}");
    }
}
