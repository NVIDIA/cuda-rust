// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(proc_macro_hygiene)]
#![allow(dead_code)]

use cuda_core::CudaStream;
use cuda_core::simt::LaunchConfig;
use cuda_macros::{cuda_module, kernel};

#[derive(Clone, Copy)]
struct Params {
    value: u32,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn root_typed(params: Params) {
        let _ = params.value;
    }

    #[cfg_attr(not(any()), cfg(any()))]
    #[kernel]
    pub fn root_cfg_attr_disabled(value: RootMissingType) {
        let _ = value;
    }

    pub mod child {
        use super::*;

        #[derive(Clone, Copy)]
        pub struct Params {
            pub values: [u32; 4],
        }

        #[kernel]
        pub fn child_typed(params: Params) {
            let _ = params.values;
        }
    }

    #[cfg(any())]
    pub mod cfg_disabled {
        #[kernel]
        pub fn absent(value: TypeThatMustNotResolve) {
            let _ = value;
        }
    }

    #[cfg_attr(not(any()), cfg(any()))]
    pub mod cfg_attr_disabled {
        #[kernel]
        pub fn also_absent(value: AnotherTypeThatMustNotResolve) {
            let _ = value;
        }
    }

    pub mod kernel_cfg_attr {
        #[cfg_attr(not(any()), cfg(any()))]
        #[kernel]
        pub fn gated(value: YetAnotherMissingType) {
            let _ = value;
        }
    }

    pub mod private_generic {
        use super::*;

        #[kernel]
        fn private_map<T: Copy>(value: T) {
            let _ = value;
        }

        pub(super) fn typecheck(
            module: &LoadedModule,
            stream: &CudaStream,
            config: LaunchConfig,
        ) {
            // SAFETY: this compile-pass fixture supplies the raw launch proof.
            let _ = unsafe { module.private_map(stream, config, 7u32) };
        }
    }

    pub mod restricted {
        use super::*;

        #[kernel]
        pub(super) fn scoped(value: u32) {
            let _ = value;
        }
    }

    pub mod bridge {
        pub mod leaf {
            use super::super::*;

            #[kernel]
            pub fn deep(value: u32) {
                let _ = value;
            }
        }
    }

    pub mod r#type {
        use super::*;

        #[kernel]
        pub fn raw_namespace(value: u32) {
            let _ = value;
        }
    }

    // These are ordinary Rust helpers. The attribute macro must preserve both
    // boundaries without trying to crawl their files.
    pub mod file_helper;
    #[path = "path_helper.rs"]
    pub mod path_helper;
    include!("cuda_module_inline_namespaces_include.rs");

    pub fn typecheck_restricted(
        module: &restricted::LoadedModule,
        stream: &CudaStream,
        config: LaunchConfig,
    ) {
        // SAFETY: this compile-pass fixture supplies the raw launch proof.
        let _ = unsafe { module.scoped(stream, config, 11u32) };
    }
}

fn typecheck_namespaces(
    root: &kernels::LoadedModule,
    child: &kernels::child::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
) {
    // SAFETY: this compile-pass fixture supplies the raw launch proofs.
    unsafe {
        let _ = root.root_typed(stream, config, Params { value: 1 });
        let _ = child.child_typed(
            stream,
            config,
            kernels::child::Params { values: [2; 4] },
        );
    }

    let _ = kernels::child::LoadedModule::from_parent(root);
    let bridge = kernels::bridge::LoadedModule::from_parent(root).unwrap();
    let _ = kernels::bridge::leaf::LoadedModule::from_parent(&bridge);
    let _ = kernels::r#type::LoadedModule::from_parent(root);

    let restricted = kernels::restricted::LoadedModule::from_parent(root).unwrap();
    kernels::typecheck_restricted(&restricted, stream, config);
}

fn main() {
    let _ = kernels::INCLUDED_HELPER;
    let _ = kernels::file_helper::FILE_HELPER;
    let _ = kernels::path_helper::PATH_HELPER;
}
