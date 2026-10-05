/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Export LLVM dialect to textual LLVM IR.
//!
//! Two pieces worth knowing about live in this module: the pre-pass that
//! assigns deterministic anonymous-value names so the textual IR is stable
//! across runs, and the block-argument → PHI-node translation that bridges
//! pliron's basic-block argument convention to LLVM's PHI-node convention.
//!
//! # Backend Configuration
//!
//! The export process can be customized via the [`ExportBackendConfig`] trait.
//! Different backends (PTX, etc.) can provide their own configuration for:
//!
//! - Data layout string
//! - Whether to emit `@llvm.used` for kernel preservation
//! - Whether to emit `!nvvmir.version` metadata
//! - Whether to emit `!nvvm.annotations` for all kernels
//!
//! The default [`PtxExportConfig`] uses minimal settings appropriate for standard
//! PTX generation via llc.
//!
//! # Module Structure
//!
//! - `config`: backend configuration trait and built-in implementations
//! - `externs`: device extern declaration types for FFI with external LTOIR
//! - `state`: exporter state and kernel bookkeeping
//! - `names`: value names, block labels, symbol normalization
//! - `module`: module-level export flow
//! - `function`: function and basic block emission
//! - `ops`: operation emission
//! - `types`: LLVM type printing
//! - `literals`: constant/literal formatting
//! - `metadata`: nvvm annotations and version, llvm.used

mod config;
mod debug;
mod externs;
mod function;
mod literals;
mod llvm_attrs;
mod metadata;
mod module;
mod names;
mod ops;
mod state;
mod types;

pub use config::{
    DebugKind, ExportBackendConfig, FunctionLocalStaticPlacement, NvvmExportConfig, NvvmIrDialect,
    PtxExportConfig,
};
pub use externs::{AsDeviceExtern, DeviceExternAttrs, DeviceExternDecl, DeviceExternType};

use pliron::{builtin::ops::ModuleOp, context::Context};

/// Textual LLVM IR plus the exporter-derived symbols consumed outside the
/// module. The optimizer uses this semantic root set instead of reparsing the
/// rendered LLVM syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedModule {
    /// Rendered LLVM IR.
    pub llvm_ir: String,
    /// Kernel entries, standalone device exports, and defined globals whose
    /// external linkage must survive optimization.
    pub public_symbols: Vec<String>,
}

/// Export a module op to a String containing LLVM IR.
///
/// Uses default PTX export mode. For alternate backends, use [`export_module_to_string_with_config`].
pub fn export_module_to_string(ctx: &Context, module: &ModuleOp) -> Result<String, String> {
    module::export_module_to_string_with_config(ctx, module, &config::PtxExportConfig)
}

/// Export a module op with device extern declarations to a String containing LLVM IR.
///
/// This is the primary export function for Device FFI support. It emits:
/// 1. Header (datalayout, target triple)
/// 2. Device extern declarations (`declare` statements)
/// 3. Module functions (from pliron operations)
/// 4. Attribute groups
/// 5. Metadata (nvvm.annotations, etc.)
pub fn export_module_with_externs<T: AsDeviceExtern>(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: &[T],
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    export_module_with_externs_and_roots(ctx, module, device_externs, config)
        .map(|exported| exported.llvm_ir)
}

/// Export a module together with its semantic external-root set.
pub fn export_module_with_externs_and_roots<T: AsDeviceExtern>(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: &[T],
    config: &dyn ExportBackendConfig,
) -> Result<ExportedModule, String> {
    let externs: Vec<DeviceExternDecl> = device_externs
        .iter()
        .map(|e| e.as_device_extern())
        .collect();
    module::export_module_with_externs_impl(ctx, module, &externs, config)
}

/// Export a module op to a String containing LLVM IR with custom backend configuration.
///
/// The `config` parameter controls backend-specific IR generation options like
/// data layout, metadata emission, and symbol preservation.
pub fn export_module_to_string_with_config(
    ctx: &Context,
    module: &ModuleOp,
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    module::export_module_to_string_with_config(ctx, module, config)
}
