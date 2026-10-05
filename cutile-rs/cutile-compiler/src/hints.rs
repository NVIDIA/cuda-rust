/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optimization hint types shared between both compiler backends.
//!
//! Pure Rust — no melior or tile-ir dependency.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::{Expr, Lit};

/// Per-architecture (SM) optimization hints for kernel compilation.
///
/// `allow_tma` and `latency` are per-op hints (passed at load/store call sites),
/// not entry-level hints. Setting them at the entry level is an error.
pub struct SMHints {
    pub gpu_name: String,
    pub num_cta_in_cga: Option<i32>,
    pub occupancy: Option<i32>,
    pub max_divisibility: Option<i32>,
    pub num_worker_warps_per_cta: Option<i32>,
}

impl SMHints {
    pub fn new(gpu_name: String) -> Self {
        Self {
            gpu_name,
            num_cta_in_cga: None,
            occupancy: None,
            max_divisibility: None,
            num_worker_warps_per_cta: None,
        }
    }

    pub fn set_num_cta_in_cga(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.num_cta_in_cga.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("num_cta_in_cga hint has already been set");
        }
        self.num_cta_in_cga = Some(get_int_hint(hint)?);
        Ok(())
    }

    pub fn set_occupancy(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.occupancy.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("occupancy hint has already been set");
        }
        self.occupancy = Some(get_int_hint(hint)?);
        Ok(())
    }

    pub fn set_max_divisibility(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.max_divisibility.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("max_divisibility hint has already been set");
        }
        self.max_divisibility = Some(get_int_hint(hint)?);
        Ok(())
    }

    pub fn set_num_worker_warps_per_cta(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.num_worker_warps_per_cta.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("num_worker_warps_per_cta hint has already been set");
        }
        self.num_worker_warps_per_cta = Some(get_int_hint(hint)?);
        Ok(())
    }
}

fn get_int_hint(expr: &Expr) -> Result<i32, JITError> {
    let Expr::Lit(lit) = expr else {
        return SourceLocation::unknown()
            .jit_error_result("expected a literal value for optimization hint");
    };
    let Lit::Int(int_expr) = &lit.lit else {
        return SourceLocation::unknown()
            .jit_error_result("expected an integer literal for optimization hint");
    };
    int_expr
        .base10_parse()
        .map_err(|e| JITError::Generic(format!("Failed to parse int hint: {e}")))
}

/// Device debug information to request from `tileiras`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugInfoLevel {
    /// No device debug information.
    None,
    /// Source line tables for profiling optimized kernels.
    Line,
    /// Source locations and inline frames for debugging unoptimized kernels.
    /// Source-variable inspection depends on support in the Tile IR toolchain.
    Full,
}

/// Runtime compile options for kernel JIT compilation.
///
/// These options control kernel-level compilation hints that can vary between
/// launches. Different values trigger separate JIT compilations (they are part
/// of the cache key).
///
/// `new()` and `default()` follow Cargo's profile `debug` setting for the
/// target `cutile-compiler` library: disabled selects `None`, enabled selects
/// `Full`. Cargo exposes only on/off to build scripts, so this also selects
/// `Full` for `debug = "line-tables-only"` or `"limited"`, not `Line`.
/// `CUDA_RUST_DEBUG=none|line|full` overrides this default at build time;
/// changing it when running an already-built app has no effect. Use
/// [`Self::debug_info`] to override the default for one compilation.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct CompileOptions {
    pub occupancy: Option<i32>,
    pub num_cta_in_cga: Option<i32>,
    pub max_divisibility: Option<i32>,
    pub num_worker_warps_per_cta: Option<i32>,
    /// `tileiras` optimization level (`--opt-level`). `None` means the
    /// default: 3, or 0 when `device_debug` is set.
    pub opt_level: Option<u8>,
    /// Compile for debugging (`tileiras --device-debug`): the frontend stops
    /// hoisting bounds checks out of loops, so every check that runs on the
    /// device sits at the source line that wrote it, and the backend
    /// generates debug information. Checks the compiler discharged by proof
    /// or moved to launch time never reach device code in any mode. Implies
    /// `--opt-level 0` unless `opt_level` is set explicitly.
    pub device_debug: bool,
    /// Emit line-number information (`tileiras --lineinfo`) for profiler
    /// correlation, without the rest of the debug contract.
    pub lineinfo: bool,
    /// Instrument memory accesses for Compute Sanitizer's memcheck tool
    /// (`tileiras --sanitize=memcheck`).
    pub sanitize_memcheck: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            occupancy: None,
            num_cta_in_cga: None,
            max_divisibility: None,
            num_worker_warps_per_cta: None,
            opt_level: None,
            device_debug: env!("CUTILE_BUILD_DEBUG_INFO") == "full",
            lineinfo: env!("CUTILE_BUILD_DEBUG_INFO") == "line",
            sanitize_memcheck: false,
        }
    }
}

impl CompileOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects exactly one debug-info mode, replacing both debug flags.
    ///
    /// Other options, including an explicitly set optimization level, are
    /// preserved. With no explicit optimization level, `Full` uses level 0
    /// and `None` / `Line` use level 3.
    pub fn debug_info(mut self, level: DebugInfoLevel) -> Self {
        self.device_debug = level == DebugInfoLevel::Full;
        self.lineinfo = level == DebugInfoLevel::Line;
        self
    }

    pub fn occupancy(mut self, occupancy: i32) -> Self {
        self.occupancy = Some(occupancy);
        self
    }

    pub fn num_cta_in_cga(mut self, num_cta_in_cga: i32) -> Self {
        self.num_cta_in_cga = Some(num_cta_in_cga);
        self
    }

    pub fn max_divisibility(mut self, max_divisibility: i32) -> Self {
        self.max_divisibility = Some(max_divisibility);
        self
    }

    pub fn num_worker_warps_per_cta(mut self, num_worker_warps_per_cta: i32) -> Self {
        self.num_worker_warps_per_cta = Some(num_worker_warps_per_cta);
        self
    }

    pub fn opt_level(mut self, opt_level: u8) -> Self {
        self.opt_level = Some(opt_level);
        self
    }

    pub fn device_debug(mut self, device_debug: bool) -> Self {
        self.device_debug = device_debug;
        self
    }

    pub fn lineinfo(mut self, lineinfo: bool) -> Self {
        self.lineinfo = lineinfo;
        self
    }

    pub fn sanitize_memcheck(mut self, sanitize_memcheck: bool) -> Self {
        self.sanitize_memcheck = sanitize_memcheck;
        self
    }
}

/// Collection of optimization hints for kernel compilation, keyed by SM architecture.
pub struct OptimizationHints {
    pub target_gpu_name: Option<String>,
    pub tile_as_hints: BTreeMap<String, SMHints>,
}

impl OptimizationHints {
    pub fn empty() -> OptimizationHints {
        Self {
            target_gpu_name: None,
            tile_as_hints: BTreeMap::new(),
        }
    }

    fn parse_key_value(expr: &Expr) -> Result<(String, Expr), JITError> {
        let Expr::Assign(key_val) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected an assignment expression in optimization hints");
        };
        let Expr::Path(key_path) = &*key_val.left else {
            return SourceLocation::unknown().jit_error_result(
                "Expected path expression on LHS of optimization hints assignment.",
            );
        };
        if key_path.path.segments.len() != 1 {
            return SourceLocation::unknown().jit_error_result(&format!(
                "Expected single-segment path in optimization hints key, got {} segments.",
                key_path.path.segments.len()
            ));
        }
        let key = key_path.path.segments.last().unwrap().ident.to_string();
        let value = *key_val.right.clone();
        Ok((key, value))
    }

    pub fn parse(expr: &Expr, target_gpu_name: String) -> Result<OptimizationHints, JITError> {
        let Expr::Tuple(opt_hints) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected a tuple expression for optimization hints");
        };
        let mut result = OptimizationHints::empty();
        result.target_gpu_name = Some(target_gpu_name);
        for sm_key_val in &opt_hints.elems {
            let (opt_key, opt_value) = Self::parse_key_value(sm_key_val)?;
            {
                if !opt_key.starts_with("sm_") {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Unexpected optimization hint {}.",
                        sm_key_val.to_token_stream()
                    ));
                }
                let Expr::Tuple(hints_tuple) = opt_value else {
                    return SourceLocation::unknown().jit_error_result(
                        "expected a tuple expression for architecture-specific optimization hints",
                    );
                };
                let mut sm_hints_result = SMHints::new(opt_key.clone());
                for hint_key_val in hints_tuple.elems.iter() {
                    let (key, hints) = Self::parse_key_value(hint_key_val)?;
                    match key.as_str() {
                        "num_cta_in_cga" => sm_hints_result.set_num_cta_in_cga(&hints)?,
                        "occupancy" => sm_hints_result.set_occupancy(&hints)?,
                        "max_divisibility" => sm_hints_result.set_max_divisibility(&hints)?,
                        "num_worker_warps_per_cta" => {
                            sm_hints_result.set_num_worker_warps_per_cta(&hints)?
                        }
                        "allow_tma" | "latency" => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                    "'{key}' is a per-op hint and cannot be set at the entry level. \
                                     Use it as a parameter on individual load/store operations instead."
                                ));
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "Unexpected optimization hint key '{key}'."
                            ));
                        }
                    }
                }
                if result
                    .tile_as_hints
                    .insert(opt_key.clone(), sm_hints_result)
                    .is_some()
                {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Duplicate optimization hint key '{opt_key}'."
                    ));
                }
            }
        }
        Ok(result)
    }

    pub fn get_sm_hints(&self, key: &str) -> Option<&SMHints> {
        self.tile_as_hints.get(key)
    }

    /// Applies runtime compile options, overriding entry-level hints.
    pub fn apply_compile_options(&mut self, options: &CompileOptions) {
        if options.occupancy.is_none()
            && options.num_cta_in_cga.is_none()
            && options.max_divisibility.is_none()
            && options.num_worker_warps_per_cta.is_none()
        {
            return;
        }
        let target_arch = self
            .target_gpu_name
            .clone()
            .unwrap_or_else(|| "sm_100".to_string());
        let sm_hints = self
            .tile_as_hints
            .entry(target_arch.clone())
            .or_insert_with(|| SMHints::new(target_arch));
        if let Some(occupancy) = options.occupancy {
            sm_hints.occupancy = Some(occupancy);
        }
        if let Some(num_cta_in_cga) = options.num_cta_in_cga {
            sm_hints.num_cta_in_cga = Some(num_cta_in_cga);
        }
        if let Some(max_divisibility) = options.max_divisibility {
            sm_hints.max_divisibility = Some(max_divisibility);
        }
        if let Some(num_worker_warps_per_cta) = options.num_worker_warps_per_cta {
            sm_hints.num_worker_warps_per_cta = Some(num_worker_warps_per_cta);
        }
    }
}
