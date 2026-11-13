/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU tests for the on-disk cubin cache and the specialization entries that
//! let a launch skip the compiler front end.
//!
//! The cross-process test is the key acceptance condition: a kernel compiled
//! once must be served from disk to a *fresh process* without spawning
//! `tileiras`.
//!
//! Each test uses a distinct tile size so it is an in-memory cache miss even
//! when the other tests in this binary already compiled the kernel — only an
//! L1 miss reaches the disk layer.

use crate::common;
use cutile::api::{copy_host_vec_to_device, meta, ones, zeros};
use cutile::jit_cache::{
    self, jit_backend_compile_count, jit_disk_hit_count, jit_front_end_skip_count,
    FileSystemJitStore,
};
use cutile::prelude::*;
use cutile::tile_kernel::jit_compile_count;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cutile::module]
mod jit_disk_cache_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true)]
    fn add_printed<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

/// Launches `add` with the given tile size and asserts the numeric result, so
/// a disk-served cubin is verified end to end, not just loaded.
fn launch_and_check(tile: usize) {
    let x: Arc<Tensor<f32>> = ones(&[256]).sync().expect("ones").into();
    let y: Arc<Tensor<f32>> = ones(&[256]).sync().expect("ones").into();
    let z = zeros(&[256]).sync().expect("zeros").partition([tile]);
    let z_host = jit_disk_cache_test_module::add(z, x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add kernel");
    assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
}

fn fresh_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutile_disk_cache_gpu_{label}_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn enable_at(dir: &Path) {
    jit_cache::enable(Arc::new(FileSystemJitStore::new(dir).expect("open store")));
}

/// The store's entries whose header starts with `magic`.
fn entries_with_magic(dir: &Path, magic: &[u8]) -> Vec<PathBuf> {
    let mut files = vec![];
    for shard in std::fs::read_dir(dir).unwrap() {
        for entry in std::fs::read_dir(shard.unwrap().path()).unwrap() {
            let path = entry.unwrap().path();
            if std::fs::read(&path).unwrap().starts_with(magic) {
                files.push(path);
            }
        }
    }
    files
}

const CUBIN_MAGIC: &[u8] = b"CUTILECUBIN\0";
const SPECIALIZATION_MAGIC: &[u8] = b"CUTILESPEC\0\0";

/// Env vars the orchestrator uses to drive its child processes. Read only by
/// this test — the library itself has no env switch.
const CHILD_DIR_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_CHILD_DIR";
const CHILD_ROLE_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_ROLE";

/// The cross-process acceptance condition: a kernel compiled by one process is served from
/// disk to another, with no `tileiras` spawn in the second.
///
/// Both the writer and the reader run in **fresh** processes, so their
/// in-memory kernel cache is empty and every kernel reaches the disk layer.
/// The reader also skips the front end for every kernel; a third reader with
/// a compiler trace enabled must run it for every kernel instead.
/// The orchestrator cannot play the writer itself: it shares this test
/// binary's in-memory cache, which earlier tests have already warmed for the
/// `ones`/`zeros`/`to_host_vec` helper kernels, so those would never reach the
/// disk and the reader would then miss them.
#[test]
fn disk_cache_cross_process_hit() {
    if let (Some(dir), Some(role)) = (
        std::env::var_os(CHILD_DIR_ENV),
        std::env::var_os(CHILD_ROLE_ENV),
    ) {
        let role = role.into_string().expect("role is UTF-8");
        common::with_test_stack(move || {
            enable_at(Path::new(&dir));
            launch_and_check(64);
            jit_cache::disable();
            if role != "writer" {
                assert_eq!(
                    jit_backend_compile_count(),
                    0,
                    "reader must not spawn tileiras: the writer stored every kernel it launches"
                );
                assert!(
                    jit_disk_hit_count() >= 1,
                    "reader must serve kernels from the disk cache, got {}",
                    jit_disk_hit_count(),
                );
                let skips = if role == "reader" {
                    jit_compile_count()
                } else {
                    0
                };
                assert_eq!(
                    jit_front_end_skip_count(),
                    skips,
                    "{role}: the writer stored a specialization for every kernel it compiled"
                );
            } else {
                assert_eq!(
                    jit_disk_hit_count(),
                    0,
                    "writer starts cold: nothing is on disk yet"
                );
                assert!(
                    jit_backend_compile_count() >= 1,
                    "writer must compile and store at least one kernel"
                );
                assert_eq!(jit_front_end_skip_count(), 0, "writer starts cold");
            }
        });
        return;
    }

    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("xproc");
        let exe = std::env::current_exe().expect("current_exe");

        let run = |role: &str| {
            let mut child = std::process::Command::new(&exe);
            child
                .args([
                    "jit_disk_cache::disk_cache_cross_process_hit",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_DIR_ENV, &dir)
                .env(CHILD_ROLE_ENV, role);
            if role == "traced" {
                child.env("CUTILE_DEBUG_COMPILER2", "1");
            }
            child.output().expect("spawn child test process")
        };

        for role in ["writer", "reader", "traced"] {
            let out = run(role);
            assert!(
                out.status.success(),
                "{role} process failed.\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            if role == "writer" {
                assert!(
                    std::fs::read_dir(&dir)
                        .expect("read writer cache directory")
                        .next()
                        .transpose()
                        .expect("read writer cache entry")
                        .is_some(),
                    "writer process succeeded without populating the cache; the child test filter may have matched zero tests"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// Regression: with no store enabled, a compile writes nothing to disk.
#[test]
fn disk_cache_default_off_writes_nothing() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        jit_cache::disable();
        assert!(!jit_cache::is_enabled());

        let puts_before = jit_cache::stats().puts;
        launch_and_check(32);
        assert_eq!(
            jit_cache::stats().puts,
            puts_before,
            "no store is installed, so nothing may be written"
        );
    });
}

/// A store that can neither read nor write must not fail the launch — the
/// errors are counted and the compile proceeds.
#[test]
fn disk_cache_degrades_softly_on_io_errors() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("degraded");
        let root = dir.join("store");
        enable_at(&root); // creates `root` as a directory

        // Replace the store root with a regular file. Every store operation
        // then fails with ENOTDIR (`<root>/<shard>/...` traverses a file),
        // which even root cannot bypass — directory permission bits, by
        // contrast, are ignored for root, and this suite runs as root.
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::write(&root, b"not a directory").unwrap();

        let io_errors_before = jit_cache::stats().io_errors;
        launch_and_check(128); // … must still succeed
        jit_cache::disable();
        assert!(
            jit_cache::stats().io_errors > io_errors_before,
            "the broken store must surface as counted soft errors"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// Corrupts the cubin payload of a cache entry while keeping the header valid.
/// The header's `payload_sha256` is recomputed so `decode_entry` passes and the
/// driver is the one that rejects the bytes.
fn corrupt_entry_payload(path: &Path) {
    let mut bytes = std::fs::read(path).expect("read cached cubin");
    assert_eq!(
        &bytes[0..12],
        b"CUTILECUBIN\0",
        "unexpected cache entry magic"
    );

    let gpu_len = u16::from_le_bytes(bytes[80..82].try_into().unwrap()) as usize;
    let fp_len = u16::from_le_bytes(bytes[82..84].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(bytes[88..96].try_into().unwrap()) as usize;
    let payload_start = 96 + gpu_len + fp_len;

    assert_eq!(
        bytes.len(),
        payload_start + payload_len,
        "cache entry size does not match header"
    );

    // Flip a byte in the cubin payload; this keeps the header valid but makes
    // the cubin invalid (e.g., corrupt ELF magic), so cuModuleLoadData rejects it.
    bytes[payload_start] ^= 0xFF;

    // Recompute the payload checksum so the header still validates.
    let new_digest = Sha256::digest(&bytes[payload_start..payload_start + payload_len]);
    bytes[16..48].copy_from_slice(&new_digest);

    std::fs::write(path, bytes).expect("write corrupted cubin");
}

/// Writes `add`'s entries for `tile` to the store with a zero-allocation
/// `meta` compile, then clears the in-memory cache so that the next launch
/// reaches the disk. Returns the cubin entry.
fn store_add_and_clear_memory(dir: &Path, tile: usize) -> PathBuf {
    let z_meta = meta::<f32>(&[256]).partition([tile]);
    let x_meta = meta::<f32>(&[256]);
    let y_meta = meta::<f32>(&[256]);
    jit_disk_cache_test_module::add(z_meta, x_meta, y_meta)
        .compile()
        .expect("meta compile should write the add kernel to disk");
    let cubins = entries_with_magic(dir, CUBIN_MAGIC);
    assert_eq!(cubins.len(), 1, "expected one cubin from the meta compile");
    assert_eq!(
        entries_with_magic(dir, SPECIALIZATION_MAGIC).len(),
        1,
        "expected one specialization entry from the meta compile"
    );
    // Safe here: no launches are in flight at this point in the caller.
    unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
    cubins.into_iter().next().unwrap()
}

/// Launches `add` with `tile` on host-copied inputs (no `ones`/`zeros` JIT
/// compiles to pollute the counters) and asserts the result.
fn launch_add_from_host(tile: usize) {
    let x_vec = Arc::new(vec![1.0f32; 256]);
    let y_vec = Arc::new(vec![1.0f32; 256]);
    let z_vec = Arc::new(vec![0.0f32; 256]);
    let x = Arc::new(copy_host_vec_to_device(&x_vec).sync().expect("copy x"));
    let y = Arc::new(copy_host_vec_to_device(&y_vec).sync().expect("copy y"));
    let z = copy_host_vec_to_device(&z_vec).sync().expect("copy z");
    let z_host = jit_disk_cache_test_module::add(z.partition([tile]), x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add launch");
    assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
}

/// Counter deltas across one launch.
#[derive(Debug, PartialEq)]
struct Counts {
    compiles: u64,
    disk_hits: u64,
    backend: u64,
    skips: u64,
}

fn counts_across(launch: impl FnOnce()) -> Counts {
    let now = || {
        [
            jit_compile_count(),
            jit_disk_hit_count(),
            jit_backend_compile_count(),
            jit_front_end_skip_count(),
        ]
    };
    let before = now();
    launch();
    let after = now();
    Counts {
        compiles: after[0] - before[0],
        disk_hits: after[1] - before[1],
        backend: after[2] - before[2],
        skips: after[3] - before[3],
    }
}

/// A stored specialization serves an in-memory miss without the front end or
/// `tileiras`, and the kernel it loads computes the right result.
#[test]
fn disk_cache_skips_the_front_end_on_a_stored_specialization() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("skip");
        enable_at(&dir);
        let tile = 4;
        store_add_and_clear_memory(&dir, tile);

        let counts = counts_across(|| launch_add_from_host(tile));
        jit_cache::disable();
        assert_eq!(
            counts,
            Counts {
                compiles: 1,
                disk_hits: 1,
                backend: 0,
                skips: 1,
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// A kernel that prints its IR stores no specialization, so every in-memory
/// miss runs the front end and prints it again; its cubin is still cached.
#[test]
fn disk_cache_never_skips_a_kernel_that_prints_its_ir() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("print_ir");
        enable_at(&dir);
        let tile = 8;
        let compile = || {
            let z_meta = meta::<f32>(&[256]).partition([tile]);
            let x_meta = meta::<f32>(&[256]);
            let y_meta = meta::<f32>(&[256]);
            jit_disk_cache_test_module::add_printed(z_meta, x_meta, y_meta)
                .compile()
                .expect("meta compile of add_printed");
        };
        compile();
        assert_eq!(entries_with_magic(&dir, CUBIN_MAGIC).len(), 1);
        assert!(entries_with_magic(&dir, SPECIALIZATION_MAGIC).is_empty());
        // Safe here: no launches are in flight.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };

        let counts = counts_across(compile);
        jit_cache::disable();
        assert_eq!(
            counts,
            Counts {
                compiles: 1,
                disk_hits: 1,
                backend: 0,
                skips: 0,
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// Corrupts the stored cubin of `add` for `tile` so that the driver rejects
/// it, deletes the specialization entry when `through_front_end`, and checks
/// that the launch reads the cubin once, recompiles it once and succeeds.
/// The recompile starts from the bytecode either stage 1 produced, so a
/// launch that skipped the front end does not run it to recover.
fn recompiles_after_driver_rejection(tile: usize, through_front_end: bool) {
    common::with_test_stack(move || {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("driver_reject");
        enable_at(&dir);

        let cubin = store_add_and_clear_memory(&dir, tile);
        // Corrupt the payload while keeping the header valid.
        corrupt_entry_payload(&cubin);
        if through_front_end {
            for spec in entries_with_magic(&dir, SPECIALIZATION_MAGIC) {
                std::fs::remove_file(spec).unwrap();
            }
        }

        // The cached cubin passes header checks but is rejected by the driver,
        // so the library evicts the entry and recompiles once.
        let counts = counts_across(|| launch_add_from_host(tile));
        jit_cache::disable();
        assert_eq!(
            counts,
            Counts {
                compiles: 1,
                disk_hits: 1,
                backend: 1,
                skips: u64::from(!through_front_end),
            }
        );
        assert_eq!(
            entries_with_magic(&dir, SPECIALIZATION_MAGIC).len(),
            1,
            "the front end stores the specialization again; a skip keeps it"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// A disk-served cubin that passes the header check but is rejected by the
/// driver must be evicted and recompiled once, then succeed.
#[test]
fn disk_cache_recompiles_after_driver_rejection() {
    recompiles_after_driver_rejection(16, true);
}

/// The same rejection, met by a launch that skips the front end.
#[test]
fn disk_cache_skip_path_recompiles_after_driver_rejection() {
    recompiles_after_driver_rejection(2, false);
}
