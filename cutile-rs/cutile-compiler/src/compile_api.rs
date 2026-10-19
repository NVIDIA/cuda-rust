/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only API for producing Tile IR and bytecode without a GPU.
//!
//! This module provides [`KernelCompiler`], a builder for compiling cuTile
//! kernels to IR and bytecode artifacts without requiring a CUDA driver or
//! GPU at runtime. Only the CUDA **headers** are needed at build time.
//!
//! # Example
//!
//! ```rust,ignore
//! use cutile::compile_api::KernelCompiler;
//!
//! let artifacts = KernelCompiler::new(my_module::__module_ast_self, "my_module", "add")
//!     .generics(vec!["32".into()])
//!     .strides(&[("c", &[1])])
//!     .target("sm_120")
//!     .compile()?;
//!
//! println!("{}", artifacts.ir_text());
//! let bc = artifacts.bytecode()?;
//! ```
//!
//! The persistent JIT-cache key can be derived without compiling a cubin:
//!
//! ```rust,ignore
//! let key = KernelCompiler::new(my_module::__module_ast_self, "my_module", "add")
//!     .generics(vec!["32".into()])
//!     .strides(&[("c", &[1])])
//!     .target("sm_120")
//!     .l2_cache_key()?;
//! assert_eq!(key.len(), 64);
//! ```

use crate::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use crate::cuda_tile_runtime_utils::current_l2_key_for_module;
use crate::error::JITError;
use crate::hints::CompileOptions;
use crate::specialization::{DivHint, SpecializationBits};

/// Where each checked partition access's bounds check ended up: proven at
/// compile time (nothing emitted), hoisted to a loop preheader, or emitted
/// in place at the access.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckPlacementCounts {
    pub discharged: u32,
    pub hoisted: u32,
    pub in_place: u32,
}

/// Compiled kernel artifacts: IR and bytecode.
///
/// Produced by [`KernelCompiler::compile`]. All methods are pure Rust and
/// do not require a GPU or CUDA driver.
pub struct CompileArtifacts {
    module: cutile_ir::Module,
    bytecode_version: Option<cutile_ir::bytecode::BytecodeVersion>,
    check_counts: CheckPlacementCounts,
    launch_checks: Vec<cuda_async::predicate::LaunchCheck>,
}

impl CompileArtifacts {
    /// Returns the human-readable Tile IR text (MLIR-like syntax).
    pub fn ir_text(&self) -> String {
        self.module.to_mlir_text()
    }

    /// Bounds-check placement counters for the compiled kernel — the same
    /// numbers reported on the `CUTILE_JIT_TIMING` line.
    pub fn check_counts(&self) -> CheckPlacementCounts {
        self.check_counts
    }

    /// Safety checks the compiler hoisted out of the kernel to launch time. The
    /// host runs these (via `validate_launch_checks`) before each launch; here
    /// they let a compile-only test inspect exactly what was evacuated.
    pub fn launch_checks(&self) -> &[cuda_async::predicate::LaunchCheck] {
        &self.launch_checks
    }

    /// Serializes the compiled module to bytecode.
    ///
    /// This is the JIT's own serializer: the module verifiers run first, and
    /// the image is written at the bytecode version negotiated for the
    /// resolved `tileiras` toolchain (`CUTILE_BYTECODE_VERSION`, the
    /// assembler's supported versions, or a probe of the binary — see
    /// `cuda_tile_runtime_utils`), so the bytes are exactly what a launch
    /// would hand to `tileiras`. Fails when no version can be negotiated,
    /// e.g. no toolkit and no `tileiras` reachable.
    pub fn bytecode(&self) -> Result<Vec<u8>, JITError> {
        if let Some(version) = self.bytecode_version {
            return crate::cuda_tile_runtime_utils::serialize_tile_ir_bytecode_for_version(
                &self.module,
                version,
            )
            .map(|(bytes, _)| bytes);
        }
        crate::cuda_tile_runtime_utils::serialize_tile_ir_bytecode(&self.module)
            .map(|(bytes, _version)| bytes)
    }

    /// Returns a reference to the underlying `cutile_ir::Module`.
    pub fn module(&self) -> &cutile_ir::Module {
        &self.module
    }

    /// Consumes the artifacts and returns the underlying `cutile_ir::Module`.
    pub fn into_module(self) -> cutile_ir::Module {
        self.module
    }
}

/// Builder for compiling a cuTile kernel without a GPU.
///
/// Wraps the existing [`CUDATileFunctionCompiler`] with a streamlined API
/// for compile-only workflows.
///
/// # Example
///
/// ```rust,ignore
/// let artifacts = KernelCompiler::new(my_module::__module_ast_self, "my_module", "tile_math")
///     .generics(vec!["32".into()])
///     .strides(&[("output", &[1])])
///     .target("sm_120")
///     .compile()?;
/// ```
pub struct KernelCompiler<F: Fn() -> crate::ast::Module> {
    module_ast_fn: F,
    module_name: String,
    function_name: String,
    gpu_name: String,
    generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    spec_args: Vec<(String, SpecializationBits)>,
    scalar_hints: Vec<(String, DivHint)>,
    const_grid: Option<(u32, u32, u32)>,
    compile_options: CompileOptions,
    bytecode_version: Option<cutile_ir::bytecode::BytecodeVersion>,
}

impl<F: Fn() -> crate::ast::Module> KernelCompiler<F> {
    /// Creates a new compiler for the given kernel.
    ///
    /// - `module_ast_fn`: The `__module_ast_self` function generated by `#[cutile::module]`.
    /// - `module_name`: Name of the module containing the kernel (e.g. `"my_module"`).
    /// - `function_name`: Name of the `#[entry]` function to compile (e.g. `"add"`).
    pub fn new(module_ast_fn: F, module_name: &str, function_name: &str) -> Self {
        Self {
            module_ast_fn,
            module_name: module_name.to_string(),
            function_name: function_name.to_string(),
            gpu_name: "sm_120".to_string(),
            generics: Vec::new(),
            stride_args: Vec::new(),
            spec_args: Vec::new(),
            scalar_hints: Vec::new(),
            const_grid: None,
            compile_options: CompileOptions::default(),
            bytecode_version: None,
        }
    }

    /// Sets the target GPU architecture (e.g. `"sm_80"`, `"sm_120"`).
    /// Defaults to `"sm_120"`.
    pub fn target(mut self, gpu_name: &str) -> Self {
        self.gpu_name = gpu_name.to_string();
        self
    }

    /// Force a target version for driver-free compilation and capability tests.
    /// This does not override the executable negotiation used for JIT launches.
    pub fn bytecode_version(mut self, version: cutile_ir::bytecode::BytecodeVersion) -> Self {
        self.bytecode_version = Some(version);
        self
    }

    /// Sets the generic arguments for the kernel (e.g. tile sizes).
    pub fn generics(mut self, generics: Vec<String>) -> Self {
        self.generics = generics;
        self
    }

    /// Sets stride arguments for tensor parameters.
    pub fn strides(mut self, strides: &[(&str, &[i32])]) -> Self {
        self.stride_args = strides
            .iter()
            .map(|(name, s)| (name.to_string(), s.to_vec()))
            .collect();
        self
    }

    /// Sets specialization bits for tensor parameters.
    pub fn spec_args(mut self, specs: &[(&str, SpecializationBits)]) -> Self {
        self.spec_args = specs
            .iter()
            .map(|(name, s)| (name.to_string(), s.clone()))
            .collect();
        self
    }

    /// Sets scalar specialization hints for integer scalar and raw pointer parameters.
    pub fn scalar_hints(mut self, hints: &[(&str, DivHint)]) -> Self {
        self.scalar_hints = hints
            .iter()
            .map(|(name, hint)| (name.to_string(), *hint))
            .collect();
        self
    }

    /// Sets a constant grid size for the kernel launch configuration.
    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.const_grid = Some(grid);
        self
    }

    /// Sets compile options (occupancy hints, etc.).
    pub fn options(mut self, options: CompileOptions) -> Self {
        self.compile_options = options;
        self
    }

    /// Returns the persistent JIT-cache key for this specialization.
    ///
    /// This runs the compiler frontend and serializes its Tile IR output using
    /// the bytecode version selected for the currently resolved `tileiras`
    /// toolchain. It does not consult a JIT store, compile a cubin, initialize
    /// the CUDA driver, or require a GPU.
    ///
    /// The returned string is the same 64-character lowercase SHA-256 key that
    /// the runtime's L2 cache lookup would use for this specialization.
    pub fn l2_cache_key(self) -> Result<String, JITError> {
        let gpu_name = self.gpu_name.clone();
        let tileiras_opts = crate::cuda_tile_runtime_utils::TileirasOptions::from_compile_options(
            &self.compile_options,
        );
        let artifacts = self.compile()?;
        if let Some(version) = artifacts.bytecode_version {
            let bytes = artifacts.bytecode()?;
            Ok(crate::cuda_tile_runtime_utils::current_l2_key_for_bytecode(
                &bytes,
                version,
                &gpu_name,
                &tileiras_opts,
            )
            .0)
        } else {
            current_l2_key_for_module(artifacts.module(), &gpu_name, &tileiras_opts)
        }
    }

    /// Compiles the kernel and returns the artifacts.
    ///
    /// This is a pure compilation step — no GPU or CUDA driver is needed.
    pub fn compile(self) -> Result<CompileArtifacts, JITError> {
        let module_ast = (self.module_ast_fn)();
        let modules = CUDATileModules::from_kernel(module_ast)?;

        let stride_refs: Vec<(&str, &[i32])> = self
            .stride_args
            .iter()
            .map(|(name, s)| (name.as_str(), s.as_slice()))
            .collect();

        let spec_refs: Vec<(&str, &SpecializationBits)> = self
            .spec_args
            .iter()
            .map(|(name, s)| (name.as_str(), s))
            .collect();
        let scalar_hint_refs: Vec<(&str, &DivHint)> = self
            .scalar_hints
            .iter()
            .map(|(name, hint)| (name.as_str(), hint))
            .collect();

        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            &self.module_name,
            &self.function_name,
            &self.generics,
            &stride_refs,
            &spec_refs,
            &scalar_hint_refs,
            self.const_grid,
            self.gpu_name.clone(),
            &self.compile_options,
        )?;

        let compiler = if let Some(version) = self.bytecode_version {
            compiler.with_target_capabilities(cutile_ir::capabilities::TargetCapabilities::new(
                version,
                self.gpu_name,
            ))?
        } else {
            compiler
        };
        let module = compiler.compile()?;
        let check_counts = CheckPlacementCounts {
            discharged: compiler.check_stats.discharged.get(),
            hoisted: compiler.check_stats.hoisted.get(),
            in_place: compiler.check_stats.in_place.get(),
        };
        let launch_checks = compiler.launch_checks.borrow().clone();
        Ok(CompileArtifacts {
            module,
            bytecode_version: self.bytecode_version,
            check_counts,
            launch_checks,
        })
    }
}
