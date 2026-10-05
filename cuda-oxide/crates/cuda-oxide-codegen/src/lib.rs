/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rustc-independent cuda-oxide PTX backend.
//!
//! The only supported public surface is [`experimental`]. It accepts a module
//! assembled from cuda-oxide's `dialect-mir` and `dialect-nvvm` operations and
//! produces PTX through the same MIR preparation, lowering, and LLVM tools as
//! the rustc frontend.
//!
//! This crate intentionally has no rustc linkage. It does not require
//! `rustc_private` or a nightly toolchain matched to `rustc_driver`.

#![warn(missing_docs)]

mod api;
mod error;
mod export;
mod generated;
#[allow(dead_code, missing_docs)]
mod generated_intrinsic_targets;
mod iket;
mod llvm_tools;
mod local_memory_diagnostic;
mod lower;
mod mir_pass_registry;
mod options;
mod pipeline;
mod prep;
mod ptx;
mod target;
mod verify;
mod warp_aggregate_constant_fp_atomics;

/// Standalone code-generation API.
///
/// # Version contract
///
/// This v1 API is source-compatible only with the exact
/// cuda-oxide revision that supplies it. Frontends must pin cuda-oxide, Pliron,
/// `dialect-mir`, and `dialect-nvvm` to one revision; their in-memory IR is not
/// a stable interchange format.
///
/// # Accepted input
///
/// A [`CodegenModule`](experimental::CodegenModule) owns the Pliron context
/// that created its module. Its top-level operation may contain `dialect-mir`,
/// `dialect-nvvm`, and builtin operations accepted by cuda-oxide's normal
/// lowering pipeline. Kernel entries are top-level
/// [`dialect_mir::ops::MirFuncOp`] values whose symbols are marked through
/// [`CodegenModule::mark_kernel_entry`](experimental::CodegenModule::mark_kernel_entry).
/// The v1 PTX output is self-contained. By default any libdevice call or
/// other unresolved function returns
/// [`CompileError::UnsupportedLinking`](experimental::CompileError::UnsupportedLinking).
/// [`Linking::Libdevice`](experimental::Linking::Libdevice) opts into
/// resolving `__nv_*` calls against `libdevice.10.bc` at the LLVM IR level,
/// which keeps the output a single self-contained PTX artifact. Unresolved
/// symbols that are not libdevice stay rejected under that option, and a
/// toolchain that cannot perform the link returns
/// [`CompileError::LibdeviceUnavailable`](experimental::CompileError::LibdeviceUnavailable).
///
/// # Minimal flow
///
/// ```no_run
/// use cuda_oxide_codegen::experimental::{
///     CodegenModule, CompileOptions, Compiler, Target,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut module = CodegenModule::new("frontend_module")?;
/// module.edit(|ctx, root| {
///     // Build top-level dialect-mir functions in `root` with their owner `ctx`.
///     let _ = (ctx, root);
/// });
///
/// let compiler = Compiler::discover()?;
/// let options = CompileOptions::new(Target::parse("sm_90")?);
/// let ptx = compiler.compile(&mut module, &options)?.into_ptx();
/// assert!(!ptx.is_empty());
/// # Ok(())
/// # }
/// ```
///
/// # LLVM tools
///
/// Compilation shells out to `llc` and, for
/// [`Optimization::O2`](experimental::Optimization::O2), a matching `opt`.
/// LLVM 21 is the general floor; targets that require PTX 9.0 need LLVM 22.
/// [`Toolchain`](experimental::Toolchain) discovers these programs explicitly
/// or accepts caller-supplied paths.
///
/// # Execution model
///
/// Compilation is synchronous. V1 has no cache, cancellation, or link step.
/// A module is cloned before destructive compiler passes, so the caller's IR
/// remains available and may be compiled repeatedly. Different threads may
/// create and compile independent
/// [`CodegenModule`](experimental::CodegenModule) values in place; the backend
/// has no global compilation lock. A module itself is not promised to be
/// `Send` and cannot be edited while it is being compiled.
pub mod experimental {
    pub use crate::api::{
        CodegenModule, Compilation, CompilationStage, CompileError, CompileOptions, Compiler,
        DebugInfo, Diagnostic, DiagnosticLevel, Linking, Optimization, Target, Toolchain,
    };
}

/// Existing cross-crate implementation hooks for mir-importer.
///
/// This is not part of the standalone frontend contract.
#[doc(hidden)]
pub mod __private {
    #[doc(hidden)]
    pub use crate::error::PipelineError;
    #[doc(hidden)]
    pub use crate::export::{DeviceExternAttrs, DeviceExternDecl};
    #[doc(hidden)]
    pub use crate::lower::append_to_module;
    #[doc(hidden)]
    pub use crate::options::{BackendOptions, DeviceArchHint};
    #[doc(hidden)]
    pub use crate::pipeline::{
        ModuleArtifactKind, ModulePipelineOutput, ModulePipelineRequest, OutputFiles,
        PipelineTrace, compile_translated_module,
    };
    #[doc(hidden)]
    pub use crate::verify::verify_operation;
    #[doc(hidden)]
    pub use llvm_export::export::DeviceExternType;

    /// Compiler-only attribute used to carry the Rust generated-intrinsic ABI marker.
    #[doc(hidden)]
    pub const GENERATED_INTRINSIC_MARKER_ATTR: &str =
        crate::generated_intrinsic_targets::GENERATED_INTRINSIC_MARKER_ATTR;

    /// Return the unique generated ABI marker for a dialect operation name.
    #[doc(hidden)]
    pub fn generated_intrinsic_marker_by_op_name(op_name: &str) -> Option<&'static str> {
        crate::generated_intrinsic_targets::generated_intrinsic_target_by_op_name(op_name)
            .map(|target| target.marker)
    }
}

#[cfg(test)]
mod tests {
    use super::experimental::CodegenModule;

    #[test]
    fn codegen_module_registers_dialects() {
        let mut module = CodegenModule::new("test").unwrap();
        module.edit(|ctx, _| {
            dialect_mir::register(ctx);
            dialect_nvvm::register(ctx);
        });
    }
}
