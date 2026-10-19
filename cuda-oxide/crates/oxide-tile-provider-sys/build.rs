/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generate the linkage provider bindings from the Tile IR compiler's header.
//!
//! The header is not part of this repository. The build reads it from the
//! directory named by `TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR`, which holds
//! `tile_ir/linkage_provider/LinkageProviderAPI.h`, and checks the file's
//! SHA-256 against `upstream.lock`. The pin exists because the ABI is a
//! pre-release 0.x interface whose layout may change without a version bump;
//! a provider built from a header the assembler was not built from fails at
//! load time at best. `TILEIR_LINKAGE_PROVIDER_UNPINNED=1` turns the mismatch
//! into a warning for local experiments against a newer header.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const INCLUDE_DIR_ENV: &str = "TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR";
const UNPINNED_ENV: &str = "TILEIR_LINKAGE_PROVIDER_UNPINNED";
const HEADER_RELATIVE: &str = "tile_ir/linkage_provider/LinkageProviderAPI.h";

fn main() {
    println!("cargo:rerun-if-env-changed={INCLUDE_DIR_ENV}");
    println!("cargo:rerun-if-env-changed={UNPINNED_ENV}");
    println!("cargo:rerun-if-changed=upstream.lock");
    println!("cargo:rerun-if-changed=build.rs");

    let header = locate_header();
    println!("cargo:rerun-if-changed={}", header.display());
    let lock = Lock::read(Path::new("upstream.lock"));
    check_pin(&header, &lock);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let bindings = bindgen::Builder::default()
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
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write bindings.rs");
    println!("cargo:include={}", header.parent().unwrap().display());
    println!("cargo:abi_version={}", lock.abi_version);
}

/// The header to bind, from the include directory the environment names.
fn locate_header() -> PathBuf {
    let Some(dir) = env::var_os(INCLUDE_DIR_ENV).filter(|value| !value.is_empty()) else {
        panic!(
            "\n{INCLUDE_DIR_ENV} is not set. oxide-tile-provider-sys needs the Tile IR \
             compiler's linkage provider header, `{HEADER_RELATIVE}`, which is not part of \
             this repository (see README.md). Point the variable at the include directory \
             that contains it.\n"
        );
    };
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
    header
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
