/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug dump tooling for inspecting compiler output, printed to stderr.
//!
//! Controlled by environment variables:
//! - `CUTILE_DUMP` — comma-separated list of stages to dump, or `"all"`.
//!   Two stages are emitted today, both once per compiled module, from the
//!   bytecode serializer (`cuda_tile_runtime_utils::serialize_tile_ir_bytecode`):
//!   `ir` (the cutile-ir module as MLIR-like text) and `bytecode` (alias
//!   `bc`; the encoded image decoded back to text). The pass-level names
//!   `ast`, `resolved`, `typed`, and `instantiated` are accepted but no pass
//!   emits them yet, so they produce no output.
//! - `CUTILE_DUMP_FILTER` — comma-separated list of function names or
//!   module-qualified paths (`my_module::my_kernel`). The emitted stages are
//!   module-level dumps, so a filter narrows them by its *module* part: a
//!   qualified entry must name the module (its function part is not
//!   consulted), and a bare function name matches every module.
//!
//! ```bash
//! CUTILE_DUMP=ir cargo test -p cutile --test my_test
//! CUTILE_DUMP=ir,bytecode CUTILE_DUMP_FILTER=my_module::add cargo test ...
//! ```
//!
//! `TILE_IR_DUMP` (any value) is a legacy alias for `CUTILE_DUMP=ir`.

use std::collections::HashSet;
use std::sync::OnceLock;

/// A stage in the compilation pipeline that can be dumped.
///
/// Only [`Self::Ir`] and [`Self::Bytecode`] have an emitter today; the other
/// names are reserved for per-pass dumps and are parsed but never produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DumpStage {
    /// Raw syn AST before any passes. Reserved: not emitted yet.
    Ast,
    /// After name resolution (paths resolved). Reserved: not emitted yet.
    Resolved,
    /// After type inference (types annotated). Reserved: not emitted yet.
    Typed,
    /// After monomorphization (no generics remain). Reserved: not emitted yet.
    Instantiated,
    /// cutile-ir Module, pretty-printed. Emitted per module.
    Ir,
    /// Encoded bytecode, decoded to human-readable text. Emitted per module.
    Bytecode,
}

impl DumpStage {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "ast" => Some(Self::Ast),
            "resolved" => Some(Self::Resolved),
            "typed" => Some(Self::Typed),
            "instantiated" => Some(Self::Instantiated),
            "ir" => Some(Self::Ir),
            "bytecode" | "bc" => Some(Self::Bytecode),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Ast => "ast",
            Self::Resolved => "resolved",
            Self::Typed => "typed",
            Self::Instantiated => "instantiated",
            Self::Ir => "ir",
            Self::Bytecode => "bytecode",
        }
    }
}

/// A filter entry: either a bare name or a module::name qualified path.
#[derive(Debug, Clone)]
enum FilterEntry {
    /// Matches any function with this name, regardless of module.
    Bare(String),
    /// Matches only module::name exactly.
    Qualified { module: String, name: String },
}

impl FilterEntry {
    fn parse(s: &str) -> Self {
        let s = s.trim();
        if let Some((module, name)) = s.rsplit_once("::") {
            FilterEntry::Qualified {
                module: module.to_string(),
                name: name.to_string(),
            }
        } else {
            FilterEntry::Bare(s.to_string())
        }
    }

    fn matches(&self, fn_name: &str, module_name: &str) -> bool {
        match self {
            FilterEntry::Bare(name) => fn_name == name,
            FilterEntry::Qualified { module, name } => fn_name == name && module_name == module,
        }
    }
}

/// Parsed configuration from environment variables.
struct DumpConfig {
    /// Which stages are enabled. Empty = nothing enabled.
    stages: HashSet<DumpStage>,
    /// Function filter. None = dump all functions. Some = only matching.
    filter: Option<Vec<FilterEntry>>,
}

impl DumpConfig {
    fn from_env() -> Self {
        let stages = match std::env::var("CUTILE_DUMP") {
            Ok(val) => {
                let val = val.trim();
                if val.eq_ignore_ascii_case("all") {
                    [
                        DumpStage::Ast,
                        DumpStage::Resolved,
                        DumpStage::Typed,
                        DumpStage::Instantiated,
                        DumpStage::Ir,
                        DumpStage::Bytecode,
                    ]
                    .into_iter()
                    .collect()
                } else {
                    val.split(',').filter_map(DumpStage::from_str).collect()
                }
            }
            Err(_) => {
                // Backward compat: TILE_IR_DUMP enables the IR stage.
                if std::env::var("TILE_IR_DUMP").is_ok() {
                    [DumpStage::Ir].into_iter().collect()
                } else {
                    HashSet::new()
                }
            }
        };

        let filter = std::env::var("CUTILE_DUMP_FILTER")
            .ok()
            .map(|val| val.split(',').map(FilterEntry::parse).collect());

        DumpConfig { stages, filter }
    }
}

static CONFIG: OnceLock<DumpConfig> = OnceLock::new();

fn config() -> &'static DumpConfig {
    CONFIG.get_or_init(DumpConfig::from_env)
}

/// Check if dumping is enabled for this stage.
pub fn should_dump(stage: DumpStage) -> bool {
    config().stages.contains(&stage)
}

/// Check if a function matches the dump filter.
///
/// - `fn_name`: the function name (e.g. `"my_kernel"`)
/// - `module_name`: the module name (e.g. `"my_module"`)
///
/// Returns true if no filter is set (dump all) or if the function matches.
pub fn matches_filter(fn_name: &str, module_name: &str) -> bool {
    match &config().filter {
        None => true,
        Some(entries) => entries.iter().any(|e| e.matches(fn_name, module_name)),
    }
}

/// Dump output for a compilation stage.
///
/// Prints to stderr with a labeled header. No-op if the stage isn't
/// enabled or the function doesn't match the filter.
pub fn dump(stage: DumpStage, fn_name: &str, module_name: &str, content: &str) {
    if !should_dump(stage) || !matches_filter(fn_name, module_name) {
        return;
    }
    let label = stage.label();
    let qualified = if module_name.is_empty() {
        fn_name.to_string()
    } else {
        format!("{module_name}::{fn_name}")
    };
    eprintln!("=== CUTILE DUMP: {label} ({qualified}) ===");
    eprintln!("{content}");
    eprintln!("=== END {label} ===\n");
}

/// Dump without a function context (e.g. module-level dumps).
pub fn dump_module(stage: DumpStage, module_name: &str, content: &str) {
    if !should_dump(stage) {
        return;
    }
    // Module-level dump: only filter if a filter is set AND no entry matches.
    if let Some(entries) = &config().filter {
        let any_match = entries.iter().any(|e| match e {
            FilterEntry::Qualified { module, .. } => module == module_name,
            FilterEntry::Bare(_) => true, // bare names can't exclude modules
        });
        if !any_match {
            return;
        }
    }
    let label = stage.label();
    eprintln!("=== CUTILE DUMP: {label} ({module_name}) ===");
    eprintln!("{content}");
    eprintln!("=== END {label} ===\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stage_names() {
        assert_eq!(DumpStage::from_str("ast"), Some(DumpStage::Ast));
        assert_eq!(DumpStage::from_str("IR"), Some(DumpStage::Ir));
        assert_eq!(
            DumpStage::from_str("  Bytecode  "),
            Some(DumpStage::Bytecode)
        );
        assert_eq!(DumpStage::from_str("bc"), Some(DumpStage::Bytecode));
        assert_eq!(DumpStage::from_str("nonsense"), None);
    }

    #[test]
    fn filter_bare_name() {
        let entry = FilterEntry::parse("my_kernel");
        assert!(entry.matches("my_kernel", "any_module"));
        assert!(entry.matches("my_kernel", "other_module"));
        assert!(!entry.matches("other_kernel", "any_module"));
    }

    #[test]
    fn filter_qualified_name() {
        let entry = FilterEntry::parse("my_module::my_kernel");
        assert!(entry.matches("my_kernel", "my_module"));
        assert!(!entry.matches("my_kernel", "other_module"));
        assert!(!entry.matches("other_kernel", "my_module"));
    }

    #[test]
    fn filter_nested_path() {
        let entry = FilterEntry::parse("cutile::core::reshape");
        // rsplit_once on "::" splits at the last ::
        assert!(entry.matches("reshape", "cutile::core"));
    }
}
