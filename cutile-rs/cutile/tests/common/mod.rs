/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Common test utilities and constants shared across all test modules.

use std::sync::{Mutex, MutexGuard, OnceLock};

use cutile::compile_api::{CheckPlacementCounts, KernelCompiler};
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::error::JITError;
use cutile_compiler::specialization::{DivHint, SpecializationBits};

/// Runtime prerequisites only. Failed discovery is a test failure; an
/// unsupported feature is a visible skip. Never catch JIT/execution errors.
#[allow(dead_code)]
pub fn supports_tile_ir(features: &[cutile_ir::requirements::Feature]) -> bool {
    use cutile_compiler::cuda_tile_runtime_utils::ToolkitCapabilities;
    use cutile_ir::ir::{Location, Module};
    let caps = ToolkitCapabilities::for_device(0).expect("selected toolkit capabilities");
    let result = caps
        .target
        .validate_module(&Module::new("test_prerequisites"))
        .and_then(|()| {
            features
                .iter()
                .try_for_each(|feature| feature.check(&caps.target, &Location::Unknown))
        });
    if let Err(error) = result {
        eprintln!("SKIP Tile IR execution: {error}");
        return false;
    }
    true
}

/// Process-wide lock serializing tests that assert on global kernel-cache
/// state (presence of a key) or the global JIT compile counter.
///
/// The in-memory kernel cache and `cutile::tile_kernel::jit_compile_count()`
/// are process-global. A test that measures "did this call compile or hit the
/// cache" must hold this lock for the whole measured window so no other test's
/// concurrent compile can move the counter between snapshots. All cache-state
/// tests (in `warmup.rs` and `warmup_bench.rs`) share this single lock.
#[allow(dead_code)]
pub fn cache_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Stack size for test threads.
///
/// Tests require larger stack sizes due to:
/// - Deep MLIR AST structures during compilation
/// - Multiple unary operations in single test kernels
/// - Nested function calls in the compiler
///
/// Binary search determined minimum requirements:
/// - Basic tests: ~2.121 MB
/// - With assume variants: ~2.612 MB
/// - With reduce/scan operations: ~2.7 MB
/// - With all unary math operations: ~5 MB (after adding absf, negf, negi, floor)
/// - tensor_views module tests require a bit more headroom.
///
/// Using 8 MB provides an adequate safety margin for all tests.
pub const TEST_STACK_SIZE: usize = 8_000_000; // 8 MB

/// Helper to run a test with the required stack size.
///
/// # Example
///
/// ```rust,ignore
/// #[test]
/// fn my_test() {
///     common::with_test_stack(|| {
///         // Your test code here
///     });
/// }
/// ```
pub fn with_test_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(f)
        .expect("Failed to spawn test thread")
        .join()
        .expect("Test thread panicked")
}

#[allow(clippy::too_many_arguments, dead_code)]
pub fn compile_to_ir<F>(
    module_ast_fn: F,
    module_name: &str,
    function_name: &str,
    generics: &[String],
    strides: &[(&str, &[i32])],
    spec_args: &[(&str, &SpecializationBits)],
    scalar_hints: &[(&str, &DivHint)],
    const_grid: Option<(u32, u32, u32)>,
    options: &CompileOptions,
) -> Result<String, JITError>
where
    F: Fn() -> Module,
{
    compile_to_ir_with_counts(
        module_ast_fn,
        module_name,
        function_name,
        generics,
        strides,
        spec_args,
        scalar_hints,
        const_grid,
        options,
    )
    .map(|(ir, _)| ir)
}

/// `compile_to_ir` that also returns the bounds-check placement counters.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn compile_to_ir_with_counts<F>(
    module_ast_fn: F,
    module_name: &str,
    function_name: &str,
    generics: &[String],
    strides: &[(&str, &[i32])],
    spec_args: &[(&str, &SpecializationBits)],
    scalar_hints: &[(&str, &DivHint)],
    const_grid: Option<(u32, u32, u32)>,
    options: &CompileOptions,
) -> Result<(String, CheckPlacementCounts), JITError>
where
    F: Fn() -> Module,
{
    let spec_args = spec_args
        .iter()
        .map(|(name, spec)| (*name, (*spec).clone()))
        .collect::<Vec<_>>();
    let scalar_hints = scalar_hints
        .iter()
        .map(|(name, hint)| (*name, **hint))
        .collect::<Vec<_>>();

    let mut compiler = KernelCompiler::new(module_ast_fn, module_name, function_name)
        .target("sm_120")
        .generics(generics.to_vec())
        .strides(strides)
        .spec_args(&spec_args)
        .scalar_hints(&scalar_hints)
        .options(options.clone());

    if let Some(grid) = const_grid {
        compiler = compiler.grid(grid);
    }

    compiler
        .compile()
        .map(|artifacts| (artifacts.ir_text(), artifacts.check_counts()))
}
