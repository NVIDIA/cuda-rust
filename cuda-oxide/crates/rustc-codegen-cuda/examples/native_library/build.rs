// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    std::io::stdin()
        .take(1024 * 1024 + 1)
        .read_to_string(&mut input)?;
    assert!(input.len() <= 1024 * 1024);
    let request: Value = serde_json::from_str(&input)?;
    assert_eq!(request["version"], 1);
    let target = request["target"].as_str().ok_or("missing target")?;
    // nvcc emits PTX for -arch=compute_XX even with --ltoir. Select its
    // matching real architecture to obtain raw LTO-IR for nvJitLink.
    let nvcc_target = format!(
        "sm_{}",
        target.strip_prefix("compute_").ok_or("invalid target")?
    );
    let cuda = request["cuda_root"].as_str().ok_or("missing toolkit")?;
    let mut source = String::new();
    let mut entries = Vec::new();
    let scalar = json!({"kind":"signed", "bits":32});
    for (index, item) in request["requests"]
        .as_array()
        .ok_or("missing requests")?
        .iter()
        .enumerate()
    {
        assert_eq!(item["operation"], "load_add");
        assert_eq!(item["algorithm"], "direct");
        assert_eq!(item["dtype"], scalar);
        let block = item["block"].as_array().ok_or("missing block")?;
        assert_eq!(&block[1..], &[json!(1), json!(1)]);
        let threads = block[0].as_u64().ok_or("invalid block")?;
        let items = item["items_per_thread"].as_u64().ok_or("invalid extent")?;
        assert!((1..=1024).contains(&threads) && (1..=16).contains(&items));
        let symbol = format!("fixture_load_b{threads}_i{items}");
        source.push_str(&format!(
            "extern \"C\" __device__ void {symbol}(const int* memory, int* values, unsigned char*, int add) {{\n#pragma unroll\nfor (int i=0; i<{items}; ++i) values[i]=memory[threadIdx.x*{items}+i]+add;\n}}\n"
        ));
        entries.push(json!({
            "request_index": index,
            "identity": Sha256::digest(symbol.as_bytes()).to_vec(),
            "symbol": symbol,
            "arguments": [
                {"kind":"const_pointer", "scalar":scalar},
                {"kind":"mut_pointer", "scalar":scalar},
                {"kind":"value", "scalar":scalar}
            ],
            "cub_layout": {"size":0,"alignment":1},
            "scratch": {"size":0,"alignment":1},
            "scratch_free": true, "mutates_payload": true
        }));
    }
    // This fixture compiles a tiny native implementation without a CUB dependency.
    // Each invocation owns its temporary files, including simultaneous consumers.
    let directory =
        std::env::temp_dir().join(format!("oxide-library-fixture-{}", std::process::id()));
    std::fs::create_dir(&directory)?;
    let result = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let cpp = directory.join("provider.cu");
        let lto = directory.join("provider.ltoir");
        std::fs::write(&cpp, source)?;
        let mut command = std::process::Command::new(std::path::Path::new(cuda).join("bin/nvcc"));
        command.args(["--ltoir", "-dlto", "-arch", &nvcc_target]);
        if let Some(host) = std::env::var_os("CUDAHOSTCXX") {
            command.arg("-ccbin").arg(host);
        }
        let status = command.arg(&cpp).arg("-o").arg(&lto).status()?;
        if !status.success() {
            return Err("fixture native compilation failed".into());
        }
        Ok(std::fs::read(lto)?)
    })();
    std::fs::remove_dir_all(&directory)?;
    let lto = result?;
    let header = serde_json::to_vec(&json!({
        "version":1, "batch_identity":Sha256::digest(&lto).to_vec(),
        "specializations":entries
    }))?;
    let mut output = std::io::stdout().lock();
    output.write_all(b"OXDLIB01")?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&(lto.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&lto)?;
    Ok(())
}

fn serve_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    assert!(bytes.len() <= 1024 * 1024);
    let request: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(request["version"], 1);
    let cuda = std::path::Path::new(request["cuda_root"].as_str().ok_or("missing toolkit")?);
    let host = std::env::var_os("CUDAHOSTCXX").unwrap_or_else(|| "g++".into());
    let host = std::path::PathBuf::from(host);
    let host = if host.is_absolute() {
        host
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(&host))
            .find(|path| path.is_file())
            .ok_or("cannot locate native host compiler")?
    };
    // The example tracks CUDA tools/headers and the host compiler driver.
    // The host compiler's installed system headers/libraries are assumed stable.
    serde_json::to_writer(
        std::io::stdout().lock(),
        &json!({
            "version":1,
            "paths":[cuda.join("bin"), cuda.join("nvvm"), cuda.join("include"), host],
            "environment":["CUDAHOSTCXX", "PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                           "CPATH", "CPLUS_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH"]
        }),
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref() {
        Some("--oxide-library-request-v1") => return serve(),
        Some("--oxide-library-inputs-v1") => return serve_inputs(),
        _ => {}
    }
    println!("cargo::rerun-if-changed=build.rs");
    let registration = json!({
        "version":1, "provider":"native_library_fixture",
        "executable":std::env::current_exe()?,
        "operation":"load_add", "algorithm":"direct"
    });
    println!("cargo::rustc-env=FIXTURE_REGISTRATION={registration}");
    Ok(())
}
