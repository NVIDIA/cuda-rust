/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Foreign tile-library calls executed on the GPU, against a static test
//! tile library built from the sources in `gpu/tilelib/`:
//! `cuda-tile-translate` assembles the interface and the implementation
//! catalog, `nvcc` compiles the SIMT code to LTO-IR, and `tileiras
//! --preview-create-tile-lib` packages them.
//!
//! `nvcc` and `tileiras` come from the toolkit. The public toolkit has no
//! `cuda-tile-translate`: `CUTILE_TILE_TRANSLATE` names the executable, and
//! without it the one next to `tileiras` is used. `cargo build --release -p
//! cuda-tile-rs` builds one (see that crate's README).
//!
//! Skipped without a Tile IR 13.5 `tileiras`, without `nvcc` next to it, or
//! without `cuda-tile-translate`. With `CUTILE_REQUIRE_TILELIB=1` a skip is a
//! failure.
//!
//! A static tile library carries precompiled SIMT code and needs no
//! implementation-provider shared library.

use crate::common::supports_tile_ir as supports;
use cutile::prelude::*;
use cutile::tile_kernel::{
    contains_cuda_function, CompileOptions, DebugInfoLevel, TileLibrarySpec,
};
use cutile_ir::requirements::Feature;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The built `test_lib.tilelib`, once [`test_tilelib_available`] built it.
static TILELIB: OnceLock<PathBuf> = OnceLock::new();

/// The `cuda-tile-translate` that built it (see [`tile_translate`]).
static TRANSLATE: OnceLock<PathBuf> = OnceLock::new();

/// The test library, as the host configures it.
fn tilelib() -> TileLibrarySpec {
    TileLibrarySpec::new(TILELIB.get().expect("test library built"))
}

/// Options linking the test library into one launch.
fn with_tilelib() -> CompileOptions {
    CompileOptions::new().tile_library(tilelib())
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::foreign]
    pub unsafe fn add_one(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;

    #[cutile::foreign(name = "add_one")]
    pub unsafe fn add_one_tuple(#[inout] x: Tile<f32, { [32] }>) -> (Tile<f32, { [32] }>,);

    /// The library implements `add_n` only for `N = 2`.
    #[cutile::foreign]
    pub unsafe fn add_n<const N: i32>(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;

    /// `ids` is output-only, so it is a result, not a parameter.
    #[cutile::foreign]
    pub unsafe fn offset_ids(base: Tile<i32, { [32] }>) -> Tile<i32, { [32] }>;

    /// A scalar operand, which the call passes as a rank-0 tile.
    #[cutile::foreign]
    pub unsafe fn add_scalar(#[inout] x: Tile<f32, { [32] }>, s: f32) -> Tile<f32, { [32] }>;

    /// A rank-0 operand and a rank-0 output-only result.
    #[cutile::foreign]
    pub unsafe fn twice(s: Tile<i32, { [] }>) -> Tile<i32, { [] }>;

    /// No operand: the only argument the call passes is the output-only `ids`.
    #[cutile::foreign]
    pub unsafe fn lane_ids() -> Tile<i32, { [32] }>;

    #[cutile::entry()]
    fn add_one_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
    }

    #[cutile::entry()]
    fn tuple_kernel(x: &mut Tensor<f32, { [32] }>) {
        let (first,) = unsafe { add_one_tuple(x.load()) };
        let second = unsafe { add_one_tuple(first) };
        x.store(second.0);
    }

    /// Compiled only by the cache-key test, so no other test has already put
    /// it in the in-memory kernel cache.
    #[cutile::entry()]
    fn cache_key_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
    }

    /// The caller's tile is unchanged by an `#[inout]` call: `y` gets the
    /// tile passed in, `x` the result.
    #[cutile::entry()]
    fn inout_keeps_input(x: &mut Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        let u = unsafe { add_one(t) };
        y.store(t);
        x.store(u);
    }

    /// The same, with the call in a branch.
    #[cutile::entry()]
    fn inout_in_branch(x: &mut Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>, flag: i32) {
        let t = x.load();
        let mut u = t;
        if flag > 0 {
            u = unsafe { add_one(t) };
        }
        y.store(t);
        x.store(u);
    }

    /// Two calls with the same input each see the original tile.
    #[cutile::entry()]
    fn inout_repeated(x: &mut Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        let a = unsafe { add_one(t) };
        let b = unsafe { add_one(t) };
        y.store(t);
        x.store(a + b);
    }

    /// Calls chained in a loop update the carried tile only.
    #[cutile::entry()]
    fn inout_in_loop(x: &mut Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>, n: i32) {
        let t = x.load();
        let mut acc = t;
        for _ in 0..n {
            acc = unsafe { add_one(acc) };
        }
        y.store(t);
        x.store(acc);
    }

    /// Launched only by the rebuilt-library test, so its launch site holds
    /// that test's last launch.
    #[cutile::entry()]
    fn rebuilt_library_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
    }

    /// Calls no foreign function, so it links no library.
    #[cutile::entry()]
    fn no_library_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(t + t);
    }

    /// Calls no foreign function; launched only by the test that rewrites
    /// its library while it compiles, so that launch compiles it.
    #[cutile::entry()]
    fn library_rewritten_while_compiling_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(t + t);
    }

    /// Calls a foreign function; launched only by the test that rewrites its
    /// library while it compiles, so that launch compiles it.
    #[cutile::entry()]
    fn library_rewritten_while_linking_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
    }

    /// Launched without a library of its own: it links the defaults.
    #[cutile::entry()]
    fn default_library_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
    }

    /// Launched only by the `Global` test, so it counts that test's
    /// launches alone.
    static CALLS: Global<AtomicI32, { [] }> = Global::new(0i32);

    /// `Global` accesses after a foreign call wait on its token: an atomic,
    /// then a load and a store.
    #[cutile::entry()]
    fn global_after_call_kernel(
        x: &mut Tensor<f32, { [32] }>,
        before_add: &mut Tensor<i32, { [1] }>,
        after_add: &mut Tensor<i32, { [1] }>,
    ) {
        let t = x.load();
        x.store(unsafe { add_one(t) });
        let one: Tile<i32, { [] }> = constant(1i32, shape![]);
        let (old, _) = CALLS.atomic_add(one, ordering::AcqRel, scope::Device);
        let (loaded, _) = CALLS.load(ordering::Acquire, scope::Device);
        let _ = CALLS.store(loaded + one, ordering::Release, scope::Device);
        before_add.store(old.reshape(shape![1]));
        after_add.store(loaded.reshape(shape![1]));
    }

    #[cutile::entry()]
    fn add_two_kernel(x: &mut Tensor<f32, { [32] }>) {
        let t = x.load();
        x.store(unsafe { add_n::<2>(t) });
    }

    #[cutile::entry()]
    fn offset_ids_kernel(ids: &mut Tensor<i32, { [32] }>) {
        let base: Tile<i32, { [32] }> = constant(100i32, const_shape![32]);
        ids.store(unsafe { offset_ids(base) });
    }

    #[cutile::entry()]
    fn add_scalar_kernel(x: &mut Tensor<f32, { [32] }>, s: f32) {
        let t = x.load();
        x.store(unsafe { add_scalar(t, s) });
    }

    #[cutile::entry()]
    fn twice_kernel(out: &mut Tensor<i32, { [1] }>) {
        let s: Tile<i32, { [] }> = constant(21i32, shape![]);
        let r = unsafe { twice(s) };
        out.store(r.reshape(shape![1]));
    }

    #[cutile::entry()]
    fn lane_ids_kernel(ids: &mut Tensor<i32, { [32] }>) {
        ids.store(unsafe { lane_ids() });
    }
}

/// Runs a toolkit tool and panics with its output when it fails.
fn run(tool: &Path, args: &[&std::ffi::OsStr]) {
    let output = Command::new(tool)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {tool:?}: {e}"));
    assert!(
        output.status.success(),
        "{tool:?} {args:?} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The sources of the test library.
fn tilelib_sources() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gpu/tilelib")
}

/// The `cuda-tile-translate` that assembles the library's `.mlir` sources:
/// `CUTILE_TILE_TRANSLATE` when it is set, otherwise the one next to
/// `tileiras` in `bin`, if there is one.
fn tile_translate(bin: &Path) -> Option<PathBuf> {
    match std::env::var_os("CUTILE_TILE_TRANSLATE").filter(|path| !path.is_empty()) {
        Some(path) => {
            let path = PathBuf::from(path);
            assert!(
                path.is_file(),
                "CUTILE_TILE_TRANSLATE names {path:?}, which is not a file"
            );
            Some(path)
        }
        None => Some(bin.join("cuda-tile-translate")).filter(|path| path.is_file()),
    }
}

/// Builds `test_lib.tilelib` for the device's architecture into `dir`, with
/// the SIMT implementation in `simt_source`. `translate` assembles the
/// `.mlir` sources; `nvcc` and `tileiras` are taken from `bin`. Intermediate
/// files go to a per-process directory, and the library is renamed into
/// place, so concurrent test processes never see a partial file.
fn build_test_tilelib(translate: &Path, bin: &Path, sm: &str, simt_source: &Path, dir: &Path) {
    let src = tilelib_sources();
    let work = dir.join(format!("build-{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("create the test tile-library directory");
    for name in ["interface", "implementation"] {
        run(
            translate,
            &[
                src.join(format!("{name}.mlir")).as_os_str(),
                "--mlir-to-cudatilebc".as_ref(),
                "--preview=simt".as_ref(),
                "--bytecode-version=13.5".as_ref(),
                "-o".as_ref(),
                work.join(format!("{name}.tilebc")).as_os_str(),
            ],
        );
    }
    let arch = sm.trim_start_matches("sm_");
    let fatbin = work.join("test_lib.fatbin");
    run(
        &bin.join("nvcc"),
        &[
            "-O3".as_ref(),
            "-rdc=true".as_ref(),
            "-fatbin".as_ref(),
            "-gencode".as_ref(),
            format!("arch=compute_{arch},code=lto_{arch}").as_ref(),
            simt_source.as_os_str(),
            "-o".as_ref(),
            fatbin.as_os_str(),
        ],
    );
    let library = work.join("test_lib.tilelib");
    run(
        &bin.join("tileiras"),
        &[
            "--preview-create-tile-lib".as_ref(),
            format!(
                "--preview-interface={}",
                work.join("interface.tilebc").display()
            )
            .as_ref(),
            format!(
                "--preview-implementation={}",
                work.join("implementation.tilebc").display()
            )
            .as_ref(),
            format!("--preview-implementation={}", fatbin.display()).as_ref(),
            "-o".as_ref(),
            library.as_os_str(),
        ],
    );
    std::fs::rename(&library, dir.join("test_lib.tilelib")).expect("publish test_lib.tilelib");
    std::fs::remove_dir_all(&work).ok();
}

/// A skipped test passes, and the harness reports it as passed. The CUDA 13.5
/// GPU job sets `CUTILE_REQUIRE_TILELIB=1`: there these tests must run, so a
/// skip fails them instead.
fn skip_tilelib(reason: &str) -> bool {
    if std::env::var("CUTILE_REQUIRE_TILELIB").as_deref() == Ok("1") {
        panic!("CUTILE_REQUIRE_TILELIB=1 but the tile library tests cannot run: {reason}");
    }
    eprintln!("SKIP static tile library: {reason}");
    false
}

/// Builds the test library once per process (see [`tilelib`]). Returns
/// false (skip) without Tile IR 13.5 foreign-call support, without `nvcc`
/// next to `tileiras`, or without `cuda-tile-translate` (see
/// [`tile_translate`]); a tool that is present but fails is a test failure.
/// With `CUTILE_REQUIRE_TILELIB=1` a skip is a failure too (see
/// [`skip_tilelib`]).
fn test_tilelib_available() -> bool {
    static BUILT: OnceLock<bool> = OnceLock::new();
    *BUILT.get_or_init(|| {
        if !supports(&[Feature::ForeignCall]) {
            return skip_tilelib("the selected toolkit has no Tile IR 13.5 foreign-call support");
        }
        let tileiras = cutile_compiler::cuda_tile_runtime_utils::tileiras_binary();
        let bin = tileiras.parent().map(PathBuf::from).unwrap_or_default();
        if !bin.join("nvcc").is_file() {
            return skip_tilelib(&format!("no nvcc next to {tileiras:?}"));
        }
        let Some(translate) = tile_translate(&bin) else {
            return skip_tilelib(&format!(
                "no cuda-tile-translate: CUTILE_TILE_TRANSLATE is not set and none is next to \
                 {tileiras:?}"
            ));
        };
        let sm = cutile_compiler::cuda_tile_runtime_utils::get_gpu_name(0);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("cutile-test-tilelib")
            .join(&sm);
        build_test_tilelib(
            &translate,
            &bin,
            &sm,
            &tilelib_sources().join("test_lib.cu"),
            &dir,
        );
        TRANSLATE.set(translate).ok();
        TILELIB.set(dir.join("test_lib.tilelib")).ok();
        true
    })
}

fn add_one(options: CompileOptions) -> Result<Vec<f32>, Error> {
    let host: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(host))
        .sync()?
        .partition([32]);
    let (x,) = kernels::add_one_kernel(x)
        .compile_options(options.tile_library(tilelib()))
        .sync()?;
    Ok(x.unpartition().to_host_vec().sync()?)
}

fn assert_added_one(values: &[f32]) {
    let expected: Vec<f32> = (0..32).map(|i| i as f32 + 1.0).collect();
    assert_eq!(values, expected);
}

#[test]
fn inout_operand_is_updated_by_the_library() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    assert_added_one(&add_one(CompileOptions::new())?);
    Ok(())
}

#[test]
fn singleton_tuple_results_execute() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let x = api::zeros::<f32>(&[32]).sync()?.partition([32]);
    let (x,) = kernels::tuple_kernel(x)
        .compile_options(with_tilelib())
        .sync()?;
    assert_eq!(x.unpartition().to_host_vec().sync()?, vec![2.0; 32]);
    Ok(())
}

#[test]
fn constant_arguments_select_the_implementation() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let host: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(host))
        .sync()?
        .partition([32]);
    let (x,) = kernels::add_two_kernel(x)
        .compile_options(with_tilelib())
        .sync()?;
    let expected: Vec<f32> = (0..32).map(|i| i as f32 + 2.0).collect();
    assert_eq!(x.unpartition().to_host_vec().sync()?, expected);
    Ok(())
}

#[test]
fn output_only_operands_are_written_by_the_library() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let ids = api::full::<i32>(-7, &[32]).sync()?.partition([32]);
    let (ids,) = kernels::offset_ids_kernel(ids)
        .compile_options(with_tilelib())
        .sync()?;
    let expected: Vec<i32> = (100..132).collect();
    assert_eq!(ids.unpartition().to_host_vec().sync()?, expected);
    Ok(())
}

#[test]
fn scalar_operands_reach_the_library() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let x = api::copy_host_vec_to_device(&Arc::new(iota()))
        .sync()?
        .partition([32]);
    let (x, _) = kernels::add_scalar_kernel(x, 2.5f32)
        .compile_options(with_tilelib())
        .sync()?;
    assert_eq!(x.unpartition().to_host_vec().sync()?, plus(2.5));
    Ok(())
}

#[test]
fn rank_0_operands_and_results_reach_the_library() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let out = api::full::<i32>(-7, &[1]).sync()?.partition([1]);
    let (out,) = kernels::twice_kernel(out)
        .compile_options(with_tilelib())
        .sync()?;
    assert_eq!(out.unpartition().to_host_vec().sync()?, vec![42]);
    Ok(())
}

#[test]
fn a_function_without_operands_returns_its_output_only_result() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let ids = api::full::<i32>(-7, &[32]).sync()?.partition([32]);
    let (ids,) = kernels::lane_ids_kernel(ids)
        .compile_options(with_tilelib())
        .sync()?;
    let expected: Vec<i32> = (0..32).collect();
    assert_eq!(ids.unpartition().to_host_vec().sync()?, expected);
    Ok(())
}

#[test]
fn library_links_with_instrumented_and_line_info_builds() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    for options in [
        CompileOptions::new().sanitize_memcheck(true),
        CompileOptions::new().debug_info(DebugInfoLevel::Line),
    ] {
        assert_added_one(&add_one(options)?);
    }
    Ok(())
}

/// `l2_cache_key()` and the runtime compute the disk-cache key in separate
/// code; for a kernel that links a tile library both must include it.
#[test]
fn l2_cache_key_matches_the_runtime_key_with_a_tile_library() -> Result<(), Error> {
    use cutile::jit_cache::{self, FileSystemJitStore, JitStore};
    if !test_tilelib_available() {
        return Ok(());
    }
    crate::common::with_test_stack(|| {
        let _guard = crate::common::cache_test_lock();
        let launcher = || {
            kernels::cache_key_kernel(api::meta::<f32>(&[32]).partition([32]))
                .compile_options(with_tilelib())
        };
        let l2_key = launcher().l2_cache_key().expect("L2 cache-key helper");
        let dir = std::env::temp_dir().join(format!(
            "cutile_static_tilelib_cache_key_{}",
            std::process::id()
        ));
        let store = Arc::new(FileSystemJitStore::new(&dir).expect("open temporary JIT store"));
        jit_cache::enable(Arc::clone(&store) as Arc<dyn JitStore>);
        let compiled = launcher().compile();
        jit_cache::disable();
        compiled.expect("compile through the runtime path");
        assert!(
            store.contains(&l2_key).expect("query temporary JIT store"),
            "the runtime must store the exact L2 key the helper returns"
        );
        std::fs::remove_dir_all(dir).expect("remove temporary JIT store");
    });
    Ok(())
}

/// Puts the process-wide default tile libraries back when dropped. The
/// defaults are part of every kernel's cache key, so a test that changes
/// them holds [`crate::common::cache_test_lock`] and drops this guard before
/// the lock.
struct RestoreDefaultTileLibraries(Vec<TileLibrarySpec>);

impl RestoreDefaultTileLibraries {
    fn save() -> Self {
        Self(cutile::tile_kernel::default_tile_libraries())
    }
}

impl Drop for RestoreDefaultTileLibraries {
    fn drop(&mut self) {
        cutile::tile_kernel::set_default_tile_libraries(std::mem::take(&mut self.0));
    }
}

/// A process-wide default library links into kernels whose launch names
/// none. The default names the same file as the other tests' launches, so
/// it merges with them while they run in parallel.
#[test]
fn default_tile_libraries_link_every_kernel() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let _lock = crate::common::cache_test_lock();
    let _restore = RestoreDefaultTileLibraries::save();
    cutile::tile_kernel::set_default_tile_libraries([tilelib()]);
    let host: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(host))
        .sync()?
        .partition([32]);
    let (x,) = kernels::default_library_kernel(x).sync()?;
    assert_added_one(&x.unpartition().to_host_vec().sync()?);
    Ok(())
}

/// A library rebuilt at the same path between two launches through one
/// launch site: the site notices, and the second launch runs the new build
/// and caches it under the new key.
#[test]
fn a_rebuilt_library_recompiles_through_a_hot_launch_site() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    // Checks the in-memory cache, which another test clears.
    let _lock = crate::common::cache_test_lock();
    let dir = std::env::temp_dir().join(format!("cutile_tilelib_rebuilt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the library's folder");
    let library = dir.join("test_lib.tilelib");
    std::fs::copy(TILELIB.get().unwrap(), &library).expect("copy the test library");
    let options = || CompileOptions::new().tile_library(TileLibrarySpec::new(&library));
    let launch = || -> Result<Vec<f32>, Error> {
        let x = api::copy_host_vec_to_device(&Arc::new(iota()))
            .sync()?
            .partition([32]);
        let (x,) = kernels::rebuilt_library_kernel(x)
            .compile_options(options())
            .sync()?;
        Ok(x.unpartition().to_host_vec().sync()?)
    };
    assert_eq!(launch()?, plus(1.0));

    // Rebuild it in place with `add_one` adding 11.
    let simt_source = dir.join("test_lib_plus_11.cu");
    let source = std::fs::read_to_string(tilelib_sources().join("test_lib.cu")).unwrap();
    assert!(source.contains("x[0] += 1.0f;"));
    std::fs::write(
        &simt_source,
        source.replace("x[0] += 1.0f;", "x[0] += 11.0f;"),
    )
    .unwrap();
    let bin = cutile_compiler::cuda_tile_runtime_utils::tileiras_binary()
        .parent()
        .map(PathBuf::from)
        .unwrap_or_default();
    let sm = cutile_compiler::cuda_tile_runtime_utils::get_gpu_name(0);
    build_test_tilelib(TRANSLATE.get().unwrap(), &bin, &sm, &simt_source, &dir);

    assert_eq!(launch()?, plus(11.0), "the launch runs the rebuilt library");
    let key = kernels::rebuilt_library_kernel(api::meta::<f32>(&[32]).partition([32]))
        .compile_options(options())
        .l1_cache_key()
        .expect("L1 key");
    assert!(
        contains_cuda_function(&key),
        "the kernel linked against the rebuilt library is cached under its key"
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// A kernel that calls no foreign function links no library, so its launch
/// site does not check the library files (a `stat` each, on every launch),
/// and a library rebuilt between two launches does not compile it again.
#[test]
fn a_kernel_without_foreign_calls_ignores_library_files() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    // Checks the in-memory cache, which another test clears.
    let _lock = crate::common::cache_test_lock();
    let dir = std::env::temp_dir().join(format!("cutile_tilelib_unlinked_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the library's folder");
    let library = dir.join("test_lib.tilelib");
    std::fs::copy(TILELIB.get().unwrap(), &library).expect("copy the test library");
    let options = || CompileOptions::new().tile_library(TileLibrarySpec::new(&library));
    let launch = || -> Result<Vec<f32>, Error> {
        let x = api::copy_host_vec_to_device(&Arc::new(iota()))
            .sync()?
            .partition([32]);
        let (x,) = kernels::no_library_kernel(x)
            .compile_options(options())
            .sync()?;
        Ok(x.unpartition().to_host_vec().sync()?)
    };
    let doubled: Vec<f32> = iota().iter().map(|v| 2.0 * v).collect();
    assert_eq!(launch()?, doubled);
    let mut bytes = std::fs::read(&library).unwrap();
    bytes.push(0);
    std::fs::write(&library, bytes).unwrap();
    assert_eq!(launch()?, doubled);
    let rebuilt_key = kernels::no_library_kernel(api::meta::<f32>(&[32]).partition([32]))
        .compile_options(options())
        .l1_cache_key()
        .expect("L1 key");
    assert!(
        !contains_cuda_function(&rebuilt_key),
        "a kernel that links no library is not compiled again for a rebuilt library"
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// Runs `launch` while another thread replaces `library` every millisecond
/// with `contents(n)` for the `n`th write; the rename keeps each file whole.
/// Returns its result and asserts that the library was replaced during it.
fn while_rewriting<R>(
    library: &Path,
    contents: impl Fn(usize) -> Vec<u8> + Sync,
    launch: impl FnOnce() -> R,
) -> R {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    let stop = AtomicBool::new(false);
    let rewrites = AtomicU32::new(0);
    std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            let staged = library.with_extension("tilelib.tmp");
            let mut n = 0usize;
            while !stop.load(Ordering::Relaxed) {
                n += 1;
                std::fs::write(&staged, contents(n)).expect("stage the rewritten library");
                std::fs::rename(&staged, library).expect("replace the library");
                rewrites.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        while rewrites.load(Ordering::Relaxed) == 0 {
            std::thread::yield_now();
        }
        let before = rewrites.load(Ordering::Relaxed);
        let result = launch();
        let during = rewrites.load(Ordering::Relaxed) - before;
        stop.store(true, Ordering::Relaxed);
        writer.join().expect("the writer thread");
        assert!(during > 1, "the library was rewritten during the launch");
        result
    })
}

/// A kernel that links no library launches while a library configured for
/// it is rewritten throughout its compile: it cannot be stale, so the launch
/// succeeds.
#[test]
fn a_kernel_without_foreign_calls_launches_while_its_library_is_rewritten() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    // Another test turns on the disk cache, whose hit would skip `tileiras`.
    let _lock = crate::common::cache_test_lock();
    let dir = std::env::temp_dir().join(format!("cutile_tilelib_rewritten_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the library's folder");
    let library = dir.join("test_lib.tilelib");
    let original = std::fs::read(TILELIB.get().unwrap()).expect("read the test library");
    std::fs::write(&library, &original).expect("copy the test library");
    // A new length on every write, so every compile sees it change.
    let lengthened = |n: usize| {
        let mut bytes = original.clone();
        bytes.resize(original.len() + 1 + n % 64, 0);
        bytes
    };
    let doubled = while_rewriting(&library, lengthened, || -> Result<Vec<f32>, Error> {
        let x = api::copy_host_vec_to_device(&Arc::new(iota()))
            .sync()?
            .partition([32]);
        let (x,) = kernels::library_rewritten_while_compiling_kernel(x)
            .compile_options(CompileOptions::new().tile_library(TileLibrarySpec::new(&library)))
            .sync()?;
        Ok(x.unpartition().to_host_vec().sync()?)
    })?;
    assert_eq!(doubled, iota().iter().map(|v| 2.0 * v).collect::<Vec<_>>());
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// A kernel that links a library rewritten throughout its compile may have
/// linked any of its builds, which its key does not describe: the launch
/// fails and asks to launch again, and caches nothing, so the next launch
/// compiles and runs.
#[test]
fn a_library_rewritten_while_a_kernel_links_it_fails_the_launch() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    // Checks the in-memory cache, which another test clears, and another
    // test turns on the disk cache, whose hit would skip `tileiras`.
    let _lock = crate::common::cache_test_lock();
    let dir = std::env::temp_dir().join(format!("cutile_tilelib_relinked_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the library's folder");
    let library = dir.join("test_lib.tilelib");
    let original = std::fs::read(TILELIB.get().unwrap()).expect("read the test library");
    std::fs::write(&library, &original).expect("copy the test library");
    let options = || CompileOptions::new().tile_library(TileLibrarySpec::new(&library));
    let launch = || -> Result<Vec<f32>, Error> {
        let x = api::copy_host_vec_to_device(&Arc::new(iota()))
            .sync()?
            .partition([32]);
        let (x,) = kernels::library_rewritten_while_linking_kernel(x)
            .compile_options(options())
            .sync()?;
        Ok(x.unpartition().to_host_vec().sync()?)
    };
    // The same bytes in a new file every time: the key, built from the
    // contents, stays the same, while each file gets a new stamp.
    let error = while_rewriting(&library, |_| original.clone(), launch)
        .expect_err("a library rewritten while it is linked fails the launch")
        .to_string();
    assert!(
        error.contains("changed while it compiled") && error.contains("launch it again"),
        "{error}"
    );
    let key =
        kernels::library_rewritten_while_linking_kernel(api::meta::<f32>(&[32]).partition([32]))
            .compile_options(options())
            .l1_cache_key()
            .expect("L1 key");
    assert!(
        !contains_cuda_function(&key),
        "a kernel linked against a library that changed is not cached"
    );
    assert_eq!(launch()?, plus(1.0), "the next launch compiles and runs");
    assert!(contains_cuda_function(&key));
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

fn iota() -> Vec<f32> {
    (0..32).map(|i| i as f32).collect()
}

fn plus(offset: f32) -> Vec<f32> {
    iota().iter().map(|v| v + offset).collect()
}

/// Two `[32]` tensors holding `0..32`, partitioned for one tile block.
fn two_inputs() -> Result<(Partition<Tensor<f32>>, Partition<Tensor<f32>>), Error> {
    let x = api::copy_host_vec_to_device(&Arc::new(iota()))
        .sync()?
        .partition([32]);
    let y = api::zeros::<f32>(&[32]).sync()?.partition([32]);
    Ok((x, y))
}

/// `x` and `y` after a launch, on the host.
fn outputs(
    x: Partition<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    Ok((
        x.unpartition().to_host_vec().sync()?,
        y.unpartition().to_host_vec().sync()?,
    ))
}

/// `#[inout]` is the library's access mode: the call returns the updated
/// value and the caller's tile keeps its value, in straight-line code,
/// branches, repeated calls with one input, and loops.
#[test]
fn inout_calls_leave_the_callers_tile_unchanged() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    let (x, y) = two_inputs()?;
    let (x, y) = kernels::inout_keeps_input(x, y)
        .compile_options(with_tilelib())
        .sync()?;
    assert_eq!(outputs(x, y)?, (plus(1.0), iota()));

    for (flag, expected) in [(1i32, plus(1.0)), (0i32, iota())] {
        let (x, y) = two_inputs()?;
        let (x, y, _) = kernels::inout_in_branch(x, y, flag)
            .compile_options(with_tilelib())
            .sync()?;
        assert_eq!(outputs(x, y)?, (expected, iota()), "flag = {flag}");
    }

    let (x, y) = two_inputs()?;
    let (x, y) = kernels::inout_repeated(x, y)
        .compile_options(with_tilelib())
        .sync()?;
    let twice_plus_two: Vec<f32> = iota().iter().map(|v| 2.0 * v + 2.0).collect();
    assert_eq!(outputs(x, y)?, (twice_plus_two, iota()));

    for n in [0i32, 1, 3] {
        let (x, y) = two_inputs()?;
        let (x, y, _) = kernels::inout_in_loop(x, y, n)
            .compile_options(with_tilelib())
            .sync()?;
        assert_eq!(outputs(x, y)?, (plus(n as f32), iota()), "n = {n}");
    }
    Ok(())
}

/// A `Global` atomic, load and store after a foreign call take the call's
/// token: `tileiras` accepts them, and they run in order after it. Each
/// launch adds one with the atomic and one with the store.
#[test]
fn global_accesses_after_a_foreign_call_execute() -> Result<(), Error> {
    if !test_tilelib_available() {
        return Ok(());
    }
    for launch in 0..2i32 {
        let x = api::copy_host_vec_to_device(&Arc::new(iota()))
            .sync()?
            .partition([32]);
        let before_add = api::full::<i32>(-1, &[1]).sync()?.partition([1]);
        let after_add = api::full::<i32>(-1, &[1]).sync()?.partition([1]);
        let (x, before_add, after_add) =
            kernels::global_after_call_kernel(x, before_add, after_add)
                .compile_options(with_tilelib())
                .sync()?;
        assert_eq!(x.unpartition().to_host_vec().sync()?, plus(1.0));
        assert_eq!(
            before_add.unpartition().to_host_vec().sync()?,
            vec![2 * launch],
            "launch {launch}: the atomic's old value"
        );
        assert_eq!(
            after_add.unpartition().to_host_vec().sync()?,
            vec![2 * launch + 1],
            "launch {launch}: the load after the atomic"
        );
    }
    Ok(())
}
