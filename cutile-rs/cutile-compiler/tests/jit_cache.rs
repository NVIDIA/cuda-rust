/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU tests (no GPU, no tileiras) for the on-disk cubin store: filesystem
//! behavior, write atomicity, and the process-global enable/disable switch.
//!
//! Key derivation and the entry codec are covered by unit tests in
//! `src/jit_cache.rs`; this file exercises the parts that touch the
//! filesystem and global state.

use cutile_compiler::jit_cache::{disable, enable, is_enabled, FileSystemJitStore, JitStore};
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A key-shaped string (64 lowercase hex chars) with a distinguishable prefix.
fn key(tag: u8) -> String {
    format!("{tag:02x}").repeat(32)
}

/// Fresh per-test directory. Hand-rolled: no tempfile dependency in this crate.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cutile_jit_cache_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn put_get_roundtrip_and_overwrite() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let k = key(0xab);

    assert_eq!(store.get(&k).unwrap(), None);
    assert!(!store.contains(&k).unwrap());

    store.put(&k, b"first").unwrap();
    assert_eq!(store.get(&k).unwrap().as_deref(), Some(&b"first"[..]));
    assert!(store.contains(&k).unwrap());

    store.put(&k, b"second").unwrap();
    assert_eq!(store.get(&k).unwrap().as_deref(), Some(&b"second"[..]));
}

#[test]
fn large_value_roundtrip() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let k = key(0x01);
    let value: Vec<u8> = (0..1_000_000u32).map(|i| i as u8).collect();
    store.put(&k, &value).unwrap();
    assert_eq!(store.get(&k).unwrap(), Some(value));
}

#[test]
fn delete_is_idempotent() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let k = key(0x02);
    store.put(&k, b"v").unwrap();
    store.delete(&k).unwrap();
    assert_eq!(store.get(&k).unwrap(), None);
    // Absent key: not an error.
    store.delete(&k).unwrap();
}

#[test]
fn clear_removes_all_entries() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let (a, b) = (key(0x03), key(0xf3));
    store.put(&a, b"va").unwrap();
    store.put(&b, b"vb").unwrap();
    store.clear().unwrap();
    assert_eq!(store.get(&a).unwrap(), None);
    assert_eq!(store.get(&b).unwrap(), None);
    // The root itself survives; the store stays usable.
    store.put(&a, b"va2").unwrap();
    assert_eq!(store.get(&a).unwrap().as_deref(), Some(&b"va2"[..]));
}

#[test]
fn rejects_keys_that_are_not_64_hex_chars() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let bad_keys: Vec<String> = vec![
        String::new(),
        "short".to_string(),
        "../../../../etc/passwd".to_string(),
        key(0xab)[..63].to_string(),
        format!("{}G", &key(0xab)[..63]), // non-hex char
        key(0xab).to_uppercase(),         // we only ever emit lowercase
    ];
    for bad in &bad_keys {
        let err = store.put(bad, b"v").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "key {bad:?}");
        assert!(store.get(bad).is_err(), "key {bad:?}");
    }
    // Nothing escaped the root.
    assert!(!dir.0.join("..").join("passwd").exists());
}

#[test]
fn no_temp_files_left_behind() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    for tag in 0..8u8 {
        store.put(&key(tag), &[tag; 128]).unwrap();
    }
    let mut stack = vec![dir.0.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                stack.push(entry.path());
            } else {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                assert!(
                    !name.contains(".tmp."),
                    "temp file leaked: {}",
                    entry.path().display()
                );
            }
        }
    }
}

/// Concurrent `put`s of the same key from 8 threads: every interleaved `get`
/// must see either nothing or one *complete* value — never a partial write,
/// never a temp file. This is the property `rename` buys.
#[test]
fn concurrent_put_get_is_atomic() {
    let dir = TestDir::new();
    let store = Arc::new(FileSystemJitStore::new(&dir.0).unwrap());
    let k = key(0x77);
    let value_a = vec![0xaa; 64 * 1024];
    let value_b = vec![0xbb; 64 * 1024];

    let mut handles = Vec::new();
    for i in 0..8 {
        let store = Arc::clone(&store);
        let k = k.clone();
        let (va, vb) = (value_a.clone(), value_b.clone());
        handles.push(std::thread::spawn(move || {
            for round in 0..50 {
                if i % 2 == 0 {
                    let v = if (i + round) % 2 == 0 { &va } else { &vb };
                    store.put(&k, v).unwrap();
                } else if let Some(got) = store.get(&k).unwrap() {
                    assert!(
                        got == va || got == vb,
                        "read a value that is neither complete write (len {})",
                        got.len()
                    );
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// `enable`/`disable` must be repeatable (tests install their own stores), and
/// a second `enable` replaces the first — an earlier `OnceLock` version panicked
/// here, which is why the slot is an `RwLock`.
#[test]
fn enable_disable_is_repeatable() {
    // Global state: serialize against any other test that touches the slot.
    static SLOT_LOCK: Mutex<()> = Mutex::new(());
    let _guard = SLOT_LOCK.lock().unwrap();

    struct MockStore;
    impl JitStore for MockStore {
        fn get(&self, _: &str) -> io::Result<Option<Vec<u8>>> {
            Ok(None)
        }
        fn put(&self, _: &str, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        // `contains` deliberately not implemented: the trait's get-based
        // default covers it, which is the "minimal custom backend" surface
        // the trait promises.
        fn delete(&self, _: &str) -> io::Result<()> {
            Ok(())
        }
        fn clear(&self) -> io::Result<()> {
            Ok(())
        }
    }

    assert!(!is_enabled());
    enable(Arc::new(MockStore));
    assert!(is_enabled());
    enable(Arc::new(MockStore)); // replace, no panic
    assert!(is_enabled());
    disable();
    assert!(!is_enabled());
    disable(); // idempotent
    assert!(!is_enabled());
}

// ── Eviction ────────────────────────────────────────────────────────────────

use cutile_compiler::jit_cache::EVICTION_LOCK_FILE_NAME;
use std::time::{Duration, SystemTime};

fn entry_path(root: &std::path::Path, k: &str) -> PathBuf {
    root.join(&k[..2]).join(format!("{k}.cubin"))
}

fn set_mtime(path: &std::path::Path, t: SystemTime) {
    let f = std::fs::File::options().write(true).open(path).unwrap();
    f.set_times(std::fs::FileTimes::new().set_modified(t))
        .unwrap();
}

fn mtime(path: &std::path::Path) -> SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}

fn count_cubin_files(root: &std::path::Path) -> usize {
    let mut n = 0;
    for shard in std::fs::read_dir(root).unwrap().flatten() {
        if shard.file_type().unwrap().is_dir() {
            for f in std::fs::read_dir(shard.path()).unwrap().flatten() {
                if f.file_name().to_string_lossy().ends_with(".cubin") {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Oldest-mtime entries go first, and eviction stops at the low watermark.
/// mtimes are set explicitly (not slept for), so the order is exact.
#[test]
fn lru_evicts_oldest_entries_first() {
    let dir = TestDir::new();

    // Seed over capacity with an unbounded store: capacity 0 never collects.
    let seed = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(0)
        .open()
        .unwrap();
    let now = SystemTime::now();
    for tag in 0..8u8 {
        seed.put(&key(tag), &vec![tag; 2048]).unwrap();
        // Entry `tag` is (100 - tag) minutes old: 0 is oldest, 7 is newest.
        set_mtime(
            &entry_path(&dir.0, &key(tag)),
            now - Duration::from_secs((100 - u64::from(tag)) * 60),
        );
    }

    // capacity 8192, watermarks 1.0/0.8 → collect down to 6553 bytes.
    // Total is 8×2048 + 600 = 16984, so the 6 oldest (0..=5) must go:
    // 16984 − 6×2048 = 4696 ≤ 6553, while stopping one earlier leaves 6744.
    let store = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(8192)
        .open()
        .unwrap();
    // 600 ≥ capacity/16 = 512: this put triggers the collection.
    store.put(&key(0xff), &[0u8; 600]).unwrap();

    for tag in 0..6u8 {
        assert_eq!(
            store.get(&key(tag)).unwrap(),
            None,
            "entry {tag} is among the oldest and must be evicted"
        );
    }
    for tag in 6..8u8 {
        assert!(
            store.get(&key(tag)).unwrap().is_some(),
            "entry {tag} is recent enough to survive"
        );
    }
    assert!(
        store.get(&key(0xff)).unwrap().is_some(),
        "the entry just written is the newest and must survive"
    );
}

/// The capacity bound must hold when every process is short-lived — which is
/// the deployment this cache exists for (#181): a process serves one request,
/// compiles a handful of kernels, exits. Each writes far less than the eviction
/// threshold, so a trigger that sums only the current process's writes would
/// never fire in any of them and the cache would grow without bound. Every
/// iteration here opens a fresh store (a fresh "process") and puts once.
///
/// Entries are 200 B against a 512 B threshold, so each put collects with
/// probability ~0.39; ending above the bound below needs ~68 consecutive
/// misses (~1e-15). A per-process counter instead leaves all 200 entries.
#[test]
fn capacity_holds_across_short_lived_processes() {
    let dir = TestDir::new();
    const ENTRY_BYTES: usize = 200;
    const CAPACITY: u64 = 8192; // eviction threshold = capacity/16 = 512 bytes
    const PUTS: u8 = 200; // 40_000 bytes written in total, ~5× capacity

    for tag in 0..PUTS {
        let store = FileSystemJitStore::builder(&dir.0)
            .capacity_bytes(CAPACITY)
            .open()
            .unwrap();
        store.put(&key(tag), &[tag; ENTRY_BYTES]).unwrap();
    }

    // Collection lands the store on the low watermark (6553 B ≈ 32 entries) and
    // it climbs a little from there before the next one fires.
    let files = count_cubin_files(&dir.0);
    assert!(
        files < 100,
        "cache must stay bounded across short-lived processes, found {files} entries \
         ({} bytes) with a {CAPACITY}-byte capacity; a per-process trigger leaves all \
         {PUTS}",
        files * ENTRY_BYTES,
    );
}

/// An entry larger than the low watermark is declined, not stored: otherwise
/// its own put would trigger an eviction that deletes every other entry and then the
/// oversized entry itself, wiping the whole cache on every write.
#[test]
fn oversized_entry_is_declined_and_preserves_cache() {
    let dir = TestDir::new();

    // Two small entries via an unbounded store (capacity 0 never collects).
    let seed = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(0)
        .open()
        .unwrap();
    seed.put(&key(0x01), &vec![0x01; 500]).unwrap();
    seed.put(&key(0x02), &vec![0x02; 500]).unwrap();

    // capacity 8192, watermarks 1.0/0.8 → low watermark 6553 bytes.
    let store = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(8192)
        .open()
        .unwrap();

    // 7000 > 6553: this entry can never be retained. `put` succeeds (soft) but
    // stores nothing…
    store.put(&key(0xff), &vec![0u8; 7000]).unwrap();
    assert_eq!(
        store.get(&key(0xff)).unwrap(),
        None,
        "an entry above the low watermark must not be stored"
    );

    // …and, crucially, it did not evict the pre-existing entries.
    assert!(
        store.get(&key(0x01)).unwrap().is_some(),
        "the oversized put must not wipe existing entries"
    );
    assert!(
        store.get(&key(0x02)).unwrap().is_some(),
        "the oversized put must not wipe existing entries"
    );
}

/// A `get` hit refreshes the entry's mtime — that is the "recently used" half
/// of LRU. Without it, a hot entry ages like an idle one.
#[test]
fn get_refreshes_entry_mtime() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::new(&dir.0).unwrap();
    let k = key(0x21);
    store.put(&k, b"v").unwrap();

    let path = entry_path(&dir.0, &k);
    set_mtime(&path, SystemTime::now() - Duration::from_secs(7200));
    let aged = mtime(&path);

    store.get(&k).unwrap().unwrap();
    let refreshed = mtime(&path);
    assert!(
        refreshed
            .duration_since(aged)
            .is_ok_and(|d| d > Duration::from_secs(7000)),
        "hit must move mtime from 2h ago to now"
    );
}

/// While another process holds `.eviction.lock`, a triggered collection returns
/// without deleting anything; once the lock is free, the next trigger
/// collects. Deterministic version of "two concurrent evictions, only one runs".
#[test]
fn eviction_skips_while_lock_is_held() {
    let dir = TestDir::new();

    let seed = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(0)
        .open()
        .unwrap();
    let now = SystemTime::now();
    for tag in 0..10u8 {
        seed.put(&key(tag), &vec![tag; 1024]).unwrap();
        set_mtime(
            &entry_path(&dir.0, &key(tag)),
            now - Duration::from_secs((100 - u64::from(tag)) * 60),
        );
    }

    let store = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(4096)
        .open()
        .unwrap();

    // Simulate another process mid-collection.
    let lock = std::fs::File::create(dir.0.join(EVICTION_LOCK_FILE_NAME)).unwrap();
    lock.try_lock().unwrap();

    // 512 ≥ capacity/16 = 256: triggers a collection attempt, which must skip.
    store.put(&key(0xf0), &[0u8; 512]).unwrap();
    assert_eq!(
        count_cubin_files(&dir.0),
        11,
        "with the lock held elsewhere, nothing may be deleted"
    );

    drop(lock); // releases the lock

    store.put(&key(0xf1), &[0u8; 512]).unwrap();
    assert!(
        count_cubin_files(&dir.0) < 12,
        "with the lock free, the over-capacity store must shrink"
    );
}

/// Collection scans also remove temp files old enough to be leftovers of a
/// crashed process — and only those; an in-flight temp (fresh mtime) stays.
#[test]
fn eviction_removes_stale_temp_files_only() {
    let dir = TestDir::new();
    let store = FileSystemJitStore::builder(&dir.0)
        .capacity_bytes(1_000_000)
        .open()
        .unwrap();
    store.put(&key(0x31), b"seed").unwrap();

    let shard = dir.0.join(&key(0x31)[..2]);
    let stale = shard.join(format!("{}.tmp.999.0", key(0x31)));
    let fresh = shard.join(format!("{}.tmp.999.1", key(0x31)));
    std::fs::write(&stale, b"crashed process leftover").unwrap();
    std::fs::write(&fresh, b"in-flight write").unwrap();
    set_mtime(&stale, SystemTime::now() - Duration::from_secs(7200));

    // 62_500 = capacity/16: triggers a scan; total stays far under capacity,
    // so no entry is evicted — but the stale temp goes.
    store.put(&key(0x32), &vec![0u8; 62_500]).unwrap();

    assert!(!stale.exists(), "2h-old temp file must be removed");
    assert!(fresh.exists(), "fresh temp file must be left alone");
    assert!(store.get(&key(0x31)).unwrap().is_some());
}

#[test]
fn builder_rejects_invalid_watermarks() {
    let dir = TestDir::new();
    for (high, low) in [(0.5, 0.8), (1.0, 0.0), (1.0, -0.1), (f64::NAN, 0.8)] {
        let err = FileSystemJitStore::builder(&dir.0)
            .eviction_watermarks(high, low)
            .open()
            .unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidInput,
            "watermarks ({high}, {low}) must be rejected"
        );
    }
}
