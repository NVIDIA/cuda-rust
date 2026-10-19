/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend configuration traits and built-in implementations.

/// Minimal data layout for PTX mode (default behavior).
pub(super) const NVPTX_DATALAYOUT_PTX: &str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

/// Full NVPTX data layout for libNVVM/LTOIR mode (Blackwell+, modern dialect).
///
/// This matches nvcc's output for sm_100+ and is required for full NVVM compatibility.
pub(super) const NVPTX_DATALAYOUT_FULL: &str = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";

/// The only supported 64-bit data layout for the legacy LLVM 7 NVVM dialect.
///
/// Keep this in sync with NVIDIA's NVVM IR specification. In particular, the
/// legacy parser does not accept the modern per-address-space layout entries.
pub(super) const NVPTX_DATALAYOUT_LEGACY: &str = "e-p:64:64:64-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

/// Textual LLVM dialect selected by libNVVM for a concrete GPU target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NvvmIrDialect {
    /// LLVM 7 syntax, including typed pointers. Used for pre-Blackwell targets.
    LegacyLlvm7,
    /// The current toolkit's modern opaque-pointer syntax. Used for Blackwell+.
    #[default]
    Modern,
}

impl NvvmIrDialect {
    pub fn uses_typed_pointers(self) -> bool {
        matches!(self, Self::LegacyLlvm7)
    }
}

/// Device debug metadata to emit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugKind {
    /// Do not emit LLVM debug metadata.
    #[default]
    Off,
    /// Emit enough metadata for source-line breakpoints and stepping.
    LineTables,
    /// Emit line tables plus the supported local-variable/type metadata.
    Full,
}

impl DebugKind {
    pub fn line_tables_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn variables_enabled(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Where a function-local static's `DIGlobalVariableExpression` is retained.
///
/// A `static` declared inside a kernel (a shared-memory tile, say) is a
/// global with a `DISubprogram` scope. LLVM's verifier accepts exactly one
/// home for such a variable, and the accepted home changed in LLVM 23:
///
/// ```text
/// LLVM 21 / 22                          LLVM 23+
///
/// !DICompileUnit(..., globals: !{!E})   !DISubprogram(..., retainedNodes: !{!E})
///   !E = !DIGlobalVariableExpression      !E = !DIGlobalVariableExpression
///        (var: !V ...)                         (var: !V ...)
///   !V = !DIGlobalVariable(scope: !SP)    !V = !DIGlobalVariable(scope: !SP)
///
/// LLVM 23 rejects the left form ("function-local variables are not allowed
/// in a DICompileUnit's global variables list"); LLVM 21/22 reject the right
/// form ("invalid retained nodes, expected DILocalVariable, DILabel or
/// DIImportedEntity"). Either rejection makes llc/opt drop the module's
/// entire debug graph with only a warning.
/// ```
///
/// Module-level globals are unaffected: they always live in the compile
/// unit's `globals:` tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FunctionLocalStaticPlacement {
    /// List the expression in the compile unit's `globals:` tuple, scoped to
    /// its owning `DISubprogram`. Required by LLVM 22 and older, and by
    /// libNVVM's LLVM.
    #[default]
    CompileUnitGlobals,
    /// List the expression in the owning `DISubprogram`'s `retainedNodes:`
    /// tuple. Required by LLVM 23 and newer.
    SubprogramRetainedNodes,
}

impl FunctionLocalStaticPlacement {
    /// The first LLVM major whose verifier requires the retained-nodes form.
    pub const FIRST_RETAINED_NODES_MAJOR: u32 = 23;

    /// The placement the given LLVM major's verifier accepts.
    pub fn for_llvm_major(major: u32) -> Self {
        if major >= Self::FIRST_RETAINED_NODES_MAJOR {
            Self::SubprogramRetainedNodes
        } else {
            Self::CompileUnitGlobals
        }
    }
}

/// Configuration trait for export backends (PTX, LTOIR, etc.).
///
/// This trait allows different backends to customize IR generation without
/// exposing backend-specific details in the public API.
pub trait ExportBackendConfig {
    /// Data layout string for the target.
    fn datalayout(&self) -> &str;

    /// Whether to emit `@llvm.used` for kernel functions.
    /// This prevents the optimizer from removing "unused" kernels.
    fn emit_llvm_used(&self) -> bool;

    /// Whether to emit `!nvvmir.version` metadata.
    fn emit_nvvmir_version(&self) -> bool;

    /// The version tuple for `!nvvmir.version` metadata.
    /// Format: [major, minor, debug_major, debug_minor]
    fn nvvmir_version(&self) -> [i32; 4];

    /// Whether to emit `!nvvm.annotations` for ALL kernels.
    /// When false, only kernels with special attributes get annotations.
    fn emit_all_kernel_annotations(&self) -> bool;

    /// Whether kernel definitions should use the `ptx_kernel` calling convention.
    fn emit_ptx_kernel_keyword(&self) -> bool;

    /// NVVM input dialect, when this is an NVVM export.
    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        None
    }

    /// Which device debug metadata tier to emit.
    fn debug_kind(&self) -> DebugKind {
        DebugKind::Off
    }

    /// Where function-local statics are retained in the debug graph.
    ///
    /// The default is the LLVM 22 form because it is also what libNVVM's
    /// LLVM accepts; a backend feeding `llc` must select the form for the
    /// `llc` major it resolved (see
    /// [`FunctionLocalStaticPlacement::for_llvm_major`]).
    fn function_local_static_placement(&self) -> FunctionLocalStaticPlacement {
        FunctionLocalStaticPlacement::default()
    }
}

/// Default PTX export configuration.
///
/// Uses minimal settings appropriate for standard PTX generation via llc.
#[derive(Clone, Debug, Default)]
pub struct PtxExportConfig;

impl ExportBackendConfig for PtxExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_PTX
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        false
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [0, 0, 0, 0] // Not used in PTX mode
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        false
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        true
    }
}

/// Export configuration for NVVM IR output.
///
/// Emits LLVM IR with full NVVM compatibility:
/// - Full NVPTX datalayout string
/// - `@llvm.used` to prevent kernel optimization
/// - `!nvvm.annotations` for all kernels
/// - `!nvvmir.version` metadata
///
/// This produces IR suitable for consumption by libNVVM (e.g., `nvvmCompileProgram -gen-lto`)
/// or other NVVM-compatible tools.
///
/// Supports both libNVVM input dialects: LLVM 7 typed-pointer IR for
/// pre-Blackwell targets and the current toolkit's opaque-pointer IR for
/// Blackwell and newer targets.
#[derive(Clone, Debug, Default)]
pub struct NvvmExportConfig {
    dialect: NvvmIrDialect,
}

impl NvvmExportConfig {
    pub fn new(dialect: NvvmIrDialect) -> Self {
        Self { dialect }
    }

    pub fn dialect(&self) -> NvvmIrDialect {
        self.dialect
    }
}

impl ExportBackendConfig for NvvmExportConfig {
    fn datalayout(&self) -> &str {
        match self.dialect {
            NvvmIrDialect::LegacyLlvm7 => NVPTX_DATALAYOUT_LEGACY,
            NvvmIrDialect::Modern => NVPTX_DATALAYOUT_FULL,
        }
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        match self.dialect {
            NvvmIrDialect::LegacyLlvm7 => [2, 0, 3, 1],
            NvvmIrDialect::Modern => [2, 0, 3, 2],
        }
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true // Emit annotations for all kernels
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        Some(self.dialect)
    }
}

#[cfg(test)]
mod function_local_static_placement_tests {
    use super::FunctionLocalStaticPlacement;

    #[test]
    fn retained_nodes_start_at_llvm_23() {
        for major in [21, 22] {
            assert_eq!(
                FunctionLocalStaticPlacement::for_llvm_major(major),
                FunctionLocalStaticPlacement::CompileUnitGlobals,
                "LLVM {major}"
            );
        }
        for major in [23, 24] {
            assert_eq!(
                FunctionLocalStaticPlacement::for_llvm_major(major),
                FunctionLocalStaticPlacement::SubprogramRetainedNodes,
                "LLVM {major}"
            );
        }
        assert_eq!(
            FunctionLocalStaticPlacement::default(),
            FunctionLocalStaticPlacement::CompileUnitGlobals
        );
    }
}
