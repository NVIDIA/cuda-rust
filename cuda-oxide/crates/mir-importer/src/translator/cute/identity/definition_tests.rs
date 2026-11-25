/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Defining identities must survive rustc's alias-aware display names.

use super::canonical_def_path;

#[test]
fn defining_paths_ignore_reexport_names_and_keep_same_named_modules() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cuda_oxide_definition_paths_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&root).unwrap();
    let library = root.join("library.rs");
    std::fs::write(
        &library,
        r#"
pub mod mma {
    pub mod bf16 {
        pub struct Fragment(pub u32);
        #[inline(never)]
        pub fn identity(value: Fragment) -> Fragment { value }
    }
}
pub use mma::bf16 as bf16_mma;
pub use mma::bf16::Fragment as RootFragment;
pub type FragmentAlias = mma::bf16::Fragment;
pub mod cute_rs {
    #[inline(never)]
    pub fn nested(value: u32) -> u32 { value }
}
"#,
    )
    .unwrap();
    let consumer = root.join("consumer.rs");
    std::fs::write(
        &consumer,
        r#"
pub fn exercise(value: cute_rs::FragmentAlias) -> cute_rs::RootFragment {
    let value = cute_rs::bf16_mma::identity(value);
    let _ = cute_rs::cute_rs::nested(value.0);
    value
}
"#,
    )
    .unwrap();

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let sysroot_output = std::process::Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("query rustc sysroot");
    assert!(sysroot_output.status.success());
    let sysroot = String::from_utf8(sysroot_output.stdout).unwrap();
    let library_output = std::process::Command::new(&rustc)
        .args([
            "--edition=2024",
            "--crate-type=rlib",
            "--crate-name=cute_rs",
        ])
        .arg(format!("--out-dir={}", root.display()))
        .arg(&library)
        .output()
        .expect("compile external reexport fixture");
    assert!(
        library_output.status.success(),
        "{}",
        String::from_utf8_lossy(&library_output.stderr)
    );
    let args = vec![
        "rustc".to_string(),
        "--edition=2024".to_string(),
        "--crate-type=rlib".to_string(),
        "--crate-name=definition_consumer".to_string(),
        "--emit=metadata".to_string(),
        "-Zmir-opt-level=0".to_string(),
        format!(
            "--extern=cute_rs={}",
            root.join("libcute_rs.rlib").display()
        ),
        format!("--out-dir={}", root.display()),
        format!("--sysroot={}", sysroot.trim()),
        consumer.display().to_string(),
    ];

    let identities = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            rustc_public::run!(&args, || {
                use rustc_public::CrateDef;
                use rustc_public::mir::{Operand, TerminatorKind, mono::Instance};
                use rustc_public::ty::{RigidTy, TyKind};

                let item = rustc_public::all_local_items()
                    .into_iter()
                    .find(|item| item.name().ends_with("::exercise"))
                    .expect("consumer function");
                let instance = Instance::try_from(item).expect("consumer instance");
                let body = instance.body().expect("consumer body");
                let TyKind::RigidTy(RigidTy::Adt(adt, _)) = body.locals()[1].ty.kind() else {
                    panic!("aliased parameter retains its ADT identity");
                };
                let mut identities = Vec::new();
                let mut record = |def: rustc_public::DefId, expected: &str| {
                    identities.push((
                        def.name().to_string(),
                        canonical_def_path(def, "cute_rs"),
                        expected.to_string(),
                    ));
                };
                record(adt.def_id(), "cute_rs::mma::bf16::Fragment");
                for block in &body.blocks {
                    let TerminatorKind::Call {
                        func: Operand::Constant(constant),
                        ..
                    } = &block.terminator.kind
                    else {
                        continue;
                    };
                    let TyKind::RigidTy(RigidTy::FnDef(function, _)) = constant.const_.ty().kind()
                    else {
                        continue;
                    };
                    let def = function.def_id();
                    match function.name().rsplit("::").next().unwrap() {
                        "identity" => {
                            record(def, "cute_rs::mma::bf16::identity");
                            record(def.parent().unwrap(), "cute_rs::mma::bf16");
                        }
                        "nested" => {
                            record(def, "cute_rs::cute_rs::nested");
                            record(def.parent().unwrap(), "cute_rs::cute_rs");
                        }
                        other => panic!("unexpected fixture call: {other}"),
                    }
                }
                std::ops::ControlFlow::<(), _>::Continue(identities)
            })
        })
        .unwrap()
        .join()
        .unwrap()
        .expect("in-process consumer compilation succeeds");

    std::fs::remove_dir_all(&root).ok();
    assert_eq!(identities.len(), 5, "ADT, both functions and their modules");
    assert!(
        identities
            .iter()
            .any(|(display, _, _)| display.contains("RootFragment") || display.contains("bf16_mma")),
        "fixture must exercise rustc's reexport-aware names: {identities:?}"
    );
    for (display, canonical, expected) in identities {
        assert_eq!(canonical, expected, "display name: {display}");
    }
}
