/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device code for the symbols the compiler asks for.
//!
//! The compiler names every instantiation of a parametric implementation
//! (`relu_impl__N256`) and links exactly that name. The body does not depend
//! on the tile extent, so each requested name becomes a thin wrapper that
//! tail-calls the body, appended to the crate's NVVM IR, and the module is
//! compiled for the kernel's architecture with libNVVM:
//!
//! ```text
//! define void @relu_impl__N256(float %a0, ptr %a1) convergent {
//! entry:
//!   tail call void @relu_impl(float %a0, ptr %a1)
//!   ret void
//! }
//! ```
//!
//! libNVVM comes from the toolkit the host process belongs to when there is
//! one (the assembler's own toolkit, found through the running executable),
//! so the LTOIR matches the linker that consumes it; otherwise from the
//! environment, the way every other cuda-oxide tool finds it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use cuda_artifact_finalizer::{CudaArch, FinalizationOptions, NvvmCompiler};
use oxide_tile_export::TileExportDesc;

/// One symbol to define: the name the compiler asked for and the export
/// whose body it stands for.
pub(crate) struct Wrapper<'a> {
    /// The requested symbol.
    pub symbol: String,
    /// The export behind it.
    pub export: &'a TileExportDesc,
}

/// The crate's IR with one wrapper per requested name that is not already
/// a body symbol. Names are defined once however often they are asked for.
pub(crate) fn with_wrappers(ir: &str, wrappers: &[Wrapper<'_>]) -> String {
    let mut text = String::with_capacity(ir.len() + 256 * wrappers.len());
    text.push_str(ir);
    if !ir.ends_with('\n') {
        text.push('\n');
    }
    let mut defined: Vec<&str> = Vec::new();
    for wrapper in wrappers {
        if wrapper.symbol == wrapper.export.symbol || defined.contains(&wrapper.symbol.as_str()) {
            continue;
        }
        defined.push(&wrapper.symbol);
        let params: Vec<String> = wrapper
            .export
            .abi_params()
            .iter()
            .enumerate()
            .map(|(index, param)| format!("{} %a{index}", param.kind.llvm_type()))
            .collect();
        let _ = write!(
            text,
            "\n; {}: the compiler's name for one instantiation of @{}\n\
             define void @{}({}) convergent {{\nentry:\n  tail call void @{}({})\n  ret void\n}}\n",
            wrapper.symbol,
            wrapper.export.decl,
            wrapper.symbol,
            params.join(", "),
            wrapper.export.symbol,
            params.join(", "),
        );
    }
    text
}

/// Where libNVVM was taken from, for the log.
pub(crate) enum ToolkitSource {
    /// The toolkit the running executable belongs to.
    Host(PathBuf),
    /// The environment and the usual install roots.
    Environment,
}

impl std::fmt::Display for ToolkitSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolkitSource::Host(root) => write!(f, "the host's toolkit at {}", root.display()),
            ToolkitSource::Environment => write!(f, "the environment"),
        }
    }
}

/// The toolkit root of the running executable (`<root>/bin/<exe>`), when it
/// carries libNVVM.
pub(crate) fn host_toolkit_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe.parent()?.parent()?;
    root.join("nvvm/lib64/libnvvm.so")
        .is_file()
        .then(|| root.to_path_buf())
}

/// A libNVVM compiler: the host toolkit's if there is one, else discovered.
pub(crate) fn compiler(host_root: Option<&Path>) -> Result<(NvvmCompiler, ToolkitSource), String> {
    if let Some(root) = host_root {
        return NvvmCompiler::discover_in_toolkit(root)
            .map(|compiler| (compiler, ToolkitSource::Host(root.to_path_buf())))
            .map_err(|error| format!("libNVVM of the toolkit at {}: {error}", root.display()));
    }
    NvvmCompiler::discover()
        .map(|compiler| (compiler, ToolkitSource::Environment))
        .map_err(|error| format!("libNVVM: {error}"))
}

/// Compile `ir` to LTOIR for `arch` (`sm_120a` and the like).
pub(crate) fn compile(
    compiler: &NvvmCompiler,
    module_name: &str,
    ir: &str,
    arch: &str,
    fma_contraction: bool,
) -> Result<Vec<u8>, String> {
    let target: CudaArch = arch
        .parse()
        .map_err(|error| format!("the kernel's architecture `{arch}`: {error}"))?;
    let options = FinalizationOptions::new(target).with_fma_contraction(fma_contraction);
    compiler
        .compile_nvvm_ir_to_ltoir(module_name, ir.as_bytes(), &options)
        .map_err(|error| format!("compiling {module_name} for {arch}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::tests::parametric_payload;

    #[test]
    fn wrappers_forward_the_native_signature_once_per_name() {
        let payload = parametric_payload();
        let relu = payload.export_for_decl("relu").unwrap();
        let warp = payload.export_for_decl("warp_sum").unwrap();
        let text = with_wrappers(
            "; body\n",
            &[
                Wrapper {
                    symbol: "relu_impl__N256".into(),
                    export: relu,
                },
                Wrapper {
                    symbol: "relu_impl__N256".into(),
                    export: relu,
                },
                Wrapper {
                    symbol: "relu_impl".into(),
                    export: relu,
                },
                Wrapper {
                    symbol: "warp_sum_impl__N64_1".into(),
                    export: warp,
                },
            ],
        );
        assert_eq!(text.matches("define void @relu_impl__N256(").count(), 1);
        assert!(
            text.contains(
                "define void @relu_impl__N256(float %a0, ptr %a1) convergent {\nentry:\n  tail \
                 call void @relu_impl(float %a0, ptr %a1)\n  ret void\n}"
            ),
            "{text}"
        );
        assert!(
            !text.contains("define void @relu_impl("),
            "a body symbol gets no wrapper"
        );
        assert!(
            text.contains("@warp_sum_impl__N64_1(float %a0, ptr %a1)"),
            "{text}"
        );
    }

    #[test]
    fn the_host_root_needs_libnvvm_beside_the_executable() {
        // A test binary does not live in a toolkit's bin directory.
        assert!(host_toolkit_root().is_none());
    }
}
