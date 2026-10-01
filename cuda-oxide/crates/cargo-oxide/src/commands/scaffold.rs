/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

use super::*;

// =========================================================================
// cargo oxide new -- standalone project scaffolding
// =========================================================================

const GIT_REPO: &str = "https://github.com/NVlabs/cuda-oxide.git";

/// crates.io release of the host-side runtime shared with cutile-rs
/// (`cuda-core`, `cuda-async`). The device and host glue crates above still
/// come from this repository; the runtime is published from NVlabs/cutile-rs
/// and the SIMT surface lives under its `simt` modules.
pub(super) const SHARED_HOST_CRATES_VERSION: &str = "0.3.1";

const RUST_TOOLCHAIN_TOML: &str = r#"[toolchain]
channel = "nightly-2026-08-28"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
"#;

const SCAFFOLD_GITIGNORE_EXTRA: &[&str] = &[
    "/target/",
    "**/*.bc", // bitcode leftovers not in the clean suffix list
    ".DS_Store",
];

pub(super) fn scaffold_gitignore() -> String {
    let mut lines: Vec<String> = SCAFFOLD_GITIGNORE_EXTRA
        .iter()
        .map(|line| (*line).to_string())
        .collect();
    // Keep in lockstep with `GENERATED_ARTIFACT_SUFFIXES` so `cargo oxide new`
    // ignores every artifact `cargo oxide clean` knows how to delete.
    for suffix in GENERATED_ARTIFACT_SUFFIXES {
        let pattern = format!("**/*.{suffix}");
        if !lines.iter().any(|line| line == &pattern) {
            lines.push(pattern);
        }
    }
    // Stable order for readable diffs: keep the three fixed entries first,
    // then sort generated patterns.
    let (fixed, rest) = lines.split_at(SCAFFOLD_GITIGNORE_EXTRA.len());
    let mut rest = rest.to_vec();
    rest.sort();
    let mut out = fixed.to_vec();
    out.append(&mut rest);
    out.push(String::new());
    out.join("\n")
}

/// File contents produced by `cargo oxide new`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ScaffoldFiles {
    pub(super) cargo_toml: String,
    rust_toolchain_toml: String,
    pub(super) gitignore: String,
    pub(super) readme: String,
    pub(super) main_rs: String,
}

fn scaffold_readme(name: &str, async_mode: bool) -> String {
    let mode = if async_mode {
        "async cuda-oxide"
    } else {
        "cuda-oxide"
    };
    let template_notes = if async_mode {
        "The template is a vector-add kernel launched through `cuda-async`:\n\
         `vecadd_async` returns a lazy `DeviceOperation` scheduled on the\n\
         stream pool. See the cuda-oxide book getting-started chapter for the\n\
         next steps."
    } else {
        "The template is a vector-add kernel. It uses `#[launch_contract]` and\n\
         `PreparedLaunch` so geometry is checked before launch. See the\n\
         cuda-oxide book getting-started chapter for the next steps."
    };
    format!(
        r#"# {name}

Scaffolded {mode} project.

## Setup

```bash
cargo oxide doctor
```

Fix anything doctor reports before building.

## Run

```bash
cargo oxide run
```

{template_notes}
"#
    )
}

fn scaffold_cargo_toml(name: &str, async_mode: bool) -> String {
    if async_mode {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}", features = ["async"] }}
# Shared host-side runtime, published from NVlabs/cutile-rs.
cuda-core = "{SHARED_HOST_CRATES_VERSION}"
cuda-async = "{SHARED_HOST_CRATES_VERSION}"
tokio = {{ version = "1", features = ["rt", "rt-multi-thread", "macros"] }}
"#
        )
    } else {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}" }}
# Shared host-side runtime, published from NVlabs/cutile-rs.
cuda-core = "{SHARED_HOST_CRATES_VERSION}"
"#
        )
    }
}

fn scaffold_main_rs(async_mode: bool) -> String {
    if async_mode {
        r#"use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
use cuda_async::simt::device_operation::DeviceOperation;
use cuda_core::simt::LaunchConfig;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_box::DeviceBox;
    use cuda_core::simt::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let (a_dev, b_dev, mut c_dev) = with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        let num_bytes = N * mem::size_of::<f32>();
        unsafe {
            let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            stream.synchronize().unwrap();
            (
                DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
            )
        }
    })?;

    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;

    let mut c_host = vec![0.0f32; N];
    with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
"#
        .to_string()
    } else {
        r#"use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
    module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

    let c_host = c_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
    Ok(())
}
"#
        .to_string()
    }
}

pub(super) fn scaffold_files(name: &str, async_mode: bool) -> ScaffoldFiles {
    ScaffoldFiles {
        cargo_toml: scaffold_cargo_toml(name, async_mode),
        rust_toolchain_toml: RUST_TOOLCHAIN_TOML.to_string(),
        gitignore: scaffold_gitignore(),
        readme: scaffold_readme(name, async_mode),
        main_rs: scaffold_main_rs(async_mode),
    }
}

/// Scaffold a new standalone cuda-oxide project.
pub fn scaffold_new(name: &str, async_mode: bool) {
    let project_dir = PathBuf::from(name);
    if project_dir.exists() {
        eprintln!("Error: directory '{}' already exists.", name);
        std::process::exit(1);
    }

    let src_dir = project_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| {
        eprintln!("Error creating directory: {}", e);
        std::process::exit(1);
    });

    let files = scaffold_files(name, async_mode);
    std::fs::write(project_dir.join("Cargo.toml"), files.cargo_toml)
        .expect("Failed to write Cargo.toml");
    std::fs::write(
        project_dir.join("rust-toolchain.toml"),
        files.rust_toolchain_toml,
    )
    .expect("Failed to write rust-toolchain.toml");
    std::fs::write(project_dir.join(".gitignore"), files.gitignore)
        .expect("Failed to write .gitignore");
    std::fs::write(project_dir.join("README.md"), files.readme).expect("Failed to write README.md");
    std::fs::write(src_dir.join("main.rs"), files.main_rs).expect("Failed to write src/main.rs");

    let mode = if async_mode { " (async)" } else { "" };
    println!("✓ Created cuda-oxide project '{}'{}", name, mode);
    println!();
    println!("  cd {}", name);
    println!("  cargo oxide doctor");
    println!("  cargo oxide run");
}
