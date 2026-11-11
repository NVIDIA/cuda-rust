/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generate or stage the linkage provider bindings (`src/bindings.rs`).
//!
//! When `TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR` is set, the build reads
//! `tile_ir/linkage_provider/LinkageProviderAPI.h` from that directory, checks
//! its SHA-256 against `upstream.lock`, and runs `bindgen` with the pre-release
//! warning banner injected via `.raw_line(...)`. When the variable is unset,
//! the build uses the checked-in `src/bindings.rs` generated from the pinned
//! header.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const INCLUDE_DIR_ENV: &str = "TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR";
const UNPINNED_ENV: &str = "TILEIR_LINKAGE_PROVIDER_UNPINNED";
const HEADER_RELATIVE: &str = "tile_ir/linkage_provider/LinkageProviderAPI.h";
const CHECKED_IN_BINDINGS: &str = "src/bindings.rs";

const GENERATED_HEADER_BANNER: &str = "\
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// WARNING: Experimental, undocumented, pre-release ABI (0.0.1).
// This interface is not stabilized and carries no public compatibility guarantees;
// record layouts and callback signatures may change without notice.";

fn main() {
    println!("cargo:rerun-if-env-changed={INCLUDE_DIR_ENV}");
    println!("cargo:rerun-if-env-changed={UNPINNED_ENV}");
    println!("cargo:rerun-if-changed=upstream.lock");
    println!("cargo:rerun-if-changed={CHECKED_IN_BINDINGS}");
    println!("cargo:rerun-if-changed=build.rs");

    let lock = Lock::read(Path::new("upstream.lock"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let out_bindings = out.join("bindings.rs");

    if let Some(header) = locate_header() {
        println!("cargo:rerun-if-changed={}", header.display());
        check_pin(&header, &lock);

        let bindings = bindgen::Builder::default()
            .raw_line(GENERATED_HEADER_BANNER)
            .header(header.to_string_lossy())
            .allowlist_type("TileIR.*")
            .allowlist_function("tileir_.*")
            .allowlist_var("TILEIR_.*")
            // Unknown enum values are protocol errors per the header, so every C
            // enum becomes a newtype whose unknown values stay representable.
            .newtype_enum(".*")
            .derive_debug(true)
            .derive_default(true)
            .derive_copy(true)
            .generate_comments(true)
            .layout_tests(true)
            .generate()
            .expect("bindgen could not parse the linkage provider header");
        bindings
            .write_to_file(&out_bindings)
            .expect("could not write bindings.rs");
        println!("cargo:include={}", header.parent().unwrap().display());
    } else {
        fs::copy(CHECKED_IN_BINDINGS, &out_bindings)
            .expect("could not copy checked-in src/bindings.rs to OUT_DIR");
    }
    println!("cargo:abi_version={}", lock.abi_version);
}

/// The header to bind when `TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR` is set.
fn locate_header() -> Option<PathBuf> {
    let dir = env::var_os(INCLUDE_DIR_ENV).filter(|value| !value.is_empty())?;
    let header = PathBuf::from(dir).join(HEADER_RELATIVE);
    if !header.is_file() {
        panic!(
            "\n{INCLUDE_DIR_ENV} names {}, but `{HEADER_RELATIVE}` is not there.\n",
            header
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .display()
        );
    }
    Some(header)
}

/// The pin recorded in `upstream.lock`: the header's SHA-256 and ABI version.
struct Lock {
    sha256: String,
    abi_version: String,
}

impl Lock {
    fn read(path: &Path) -> Self {
        let text = fs::read_to_string(path).expect("upstream.lock is readable");
        let field = |key: &str| -> String {
            text.lines()
                .filter_map(|line| line.split_once('='))
                .find(|(name, _)| name.trim() == key)
                .map(|(_, value)| value.trim().trim_matches('"').to_string())
                .unwrap_or_else(|| panic!("upstream.lock has no `{key}` entry"))
        };
        Self {
            sha256: field("sha256"),
            abi_version: field("abi_version"),
        }
    }
}

/// Fail (or warn, when unpinned) if the header is not the one the lock names,
/// and fail if its ABI version macros disagree with the lock.
fn check_pin(header: &Path, lock: &Lock) {
    let bytes = fs::read(header).expect("the header is readable");
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != lock.sha256 {
        let message = format!(
            "the linkage provider header at {} has SHA-256 {digest}, but upstream.lock pins \
             {}. The provider ABI is pre-release: a layout change can hide behind the same \
             version number. Re-pin after checking the header against the assembler it will \
             be loaded by, or set {UNPINNED_ENV}=1 to build against it anyway.",
            header.display(),
            lock.sha256
        );
        if env::var_os(UNPINNED_ENV).is_some_and(|value| value == "1") {
            println!("cargo:warning={message}");
        } else {
            panic!("\n{message}\n");
        }
    }
    let text = String::from_utf8_lossy(&bytes);
    let macro_value = |name: &str| -> Option<u32> {
        let prefix = format!("#define TILEIR_LINKAGE_PROVIDER_API_VERSION_{name} ");
        text.lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| {
                line[prefix.len()..]
                    .trim()
                    .trim_end_matches('u')
                    .parse()
                    .ok()
            })
    };
    let version = match (
        macro_value("MAJOR"),
        macro_value("MINOR"),
        macro_value("PATCH"),
    ) {
        (Some(major), Some(minor), Some(patch)) => format!("{major}.{minor}.{patch}"),
        _ => panic!(
            "could not read the ABI version macros from {}",
            header.display()
        ),
    };
    if version != lock.abi_version {
        panic!(
            "\nthe header declares linkage provider ABI {version}, but upstream.lock expects {}\n",
            lock.abi_version
        );
    }
}
