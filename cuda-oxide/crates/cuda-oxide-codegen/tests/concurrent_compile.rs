/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Concurrency contract for one reusable standalone compiler.

#![cfg(unix)]

use cuda_oxide_codegen::experimental::{
    CodegenModule, CompileOptions, Compiler, Optimization, Target, Toolchain,
};
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

#[test]
fn one_compiler_serves_independent_modules_on_eight_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Compiler>();

    let root = std::env::temp_dir().join(format!(
        "cuda_oxide_codegen_concurrency_{}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    let llc = root.join("llc");
    std::fs::write(
        &llc,
        r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
out=""
target="sm_80"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -mcpu=*) target="${1#-mcpu=}" ;;
    -o) shift; out="$1" ;;
  esac
  shift
done
printf '.version 8.0\n.target %s\n.address_size 64\n.visible .entry fake() { ret; }\n' "$target" > "$out"
"#,
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&llc).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&llc, permissions).unwrap();

    let toolchain = Toolchain::from_paths(&llc, None).unwrap();
    let compiler = Arc::new(Compiler::new(toolchain));
    let joins: Vec<_> = (0..8)
        .map(|index| {
            let compiler = Arc::clone(&compiler);
            std::thread::spawn(move || {
                let mut module = CodegenModule::new(&format!("thread_{index}")).unwrap();
                let target = if index % 2 == 0 { "sm_80" } else { "sm_86" };
                let options = CompileOptions::new(Target::parse(target).unwrap())
                    .with_optimization(Optimization::None);
                let result = compiler.compile(&mut module, &options).unwrap();
                assert_eq!(result.target(), &Target::parse(target).unwrap());
                assert!(String::from_utf8_lossy(result.ptx()).contains(target));
            })
        })
        .collect();

    for join in joins {
        join.join()
            .expect("independent compilation thread panicked");
    }

    std::fs::remove_dir_all(root).unwrap();
}
