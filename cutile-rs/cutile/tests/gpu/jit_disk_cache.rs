/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU tests for the on-disk cubin cache.
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
use cutile::jit_cache::{self, jit_backend_compile_count, jit_disk_hit_count, FileSystemJitStore};
use cutile::prelude::*;
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

/// Env vars the orchestrator uses to drive its child processes. Read only by
/// this test — the library itself has no env switch.
const CHILD_DIR_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_CHILD_DIR";
const CHILD_ROLE_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_ROLE";

/// The cross-process acceptance condition: a kernel compiled by one process is served from
/// disk to another, with no `tileiras` spawn in the second.
///
/// Both the writer and the reader run in **fresh** processes, so their
/// in-memory kernel cache is empty and every kernel reaches the disk layer.
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
        let is_reader = role.to_str() == Some("reader");
        common::with_test_stack(move || {
            enable_at(Path::new(&dir));
            launch_and_check(64);
            jit_cache::disable();
            if is_reader {
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
            }
        });
        return;
    }

    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("xproc");
        let exe = std::env::current_exe().expect("current_exe");

        let run = |role: &str| {
            std::process::Command::new(&exe)
                .args([
                    "jit_disk_cache::disk_cache_cross_process_hit",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_DIR_ENV, &dir)
                .env(CHILD_ROLE_ENV, role)
                .output()
                .expect("spawn child test process")
        };

        for role in ["writer", "reader"] {
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

/// A disk-served cubin that passes the header check but is rejected by the
/// driver must be evicted and recompiled once, then succeed. Uses `meta` tensors
/// to warm only the `add` kernel, and `copy_host_vec_to_device` to avoid
/// bringing in unrelated `ones`/`zeros` JIT compiles that would pollute counters.
#[test]
fn disk_cache_recompiles_after_driver_rejection() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("driver_reject");
        enable_at(&dir);

        // Use a tile size unique to this test so no other test pollutes L1/disk.
        let tile = 16;

        // Warm the disk cache with a zero-allocation meta compile of just `add`.
        let z_meta = meta::<f32>(&[256]).partition([tile]);
        let x_meta = meta::<f32>(&[256]);
        let y_meta = meta::<f32>(&[256]);
        jit_disk_cache_test_module::add(z_meta, x_meta, y_meta)
            .compile()
            .expect("meta compile should write the add kernel to disk");

        // The store should contain exactly one cubin from the meta compile.
        let mut files = vec![];
        for shard in std::fs::read_dir(&dir).unwrap() {
            let shard = shard.unwrap().path();
            for entry in std::fs::read_dir(&shard).unwrap() {
                files.push(entry.unwrap().path());
            }
        }
        assert_eq!(
            files.len(),
            1,
            "expected one cached cubin from meta compile"
        );

        // Corrupt the payload while keeping the header valid.
        corrupt_entry_payload(&files[0]);

        // Clear the in-memory cache so the real launch must consult the disk
        // store. Safe here: no launches are in flight at this point in the test.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };

        let backend_before = jit_backend_compile_count();
        let hits_before = jit_disk_hit_count();

        // Real launch: the cached cubin passes header checks but is rejected by
        // the driver, so the library evicts the entry and recompiles once.
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
            .expect("real launch after driver rejection should succeed");

        jit_cache::disable();

        assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));

        assert_eq!(
            jit_disk_hit_count() - hits_before,
            1,
            "the corrupted entry must be read from disk once"
        );
        assert_eq!(
            jit_backend_compile_count() - backend_before,
            1,
            "driver rejection must trigger exactly one tileiras recompile"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}
