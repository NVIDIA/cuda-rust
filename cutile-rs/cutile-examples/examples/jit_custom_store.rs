/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Implementing a custom [`JitStore`] backend.
//!
//! The built-in [`FileSystemJitStore`] covers the common case. Anything else —
//! an object store shared by a fleet, a database, a read-through chain — is a
//! `JitStore` implementation living in *your* crate, not a cutile dependency;
//! the trait is deliberately a minimal byte-oriented interface (`io::Read` /
//! `io::Write` style) so that stays small:
//!
//! - `get`/`put`/`delete`/`clear` are the whole required surface (`contains`
//!   has a `get`-based default);
//! - `key` is 64 lowercase hex chars — safe as a file name, object name, or
//!   primary key;
//! - values are opaque encoded entries. Integrity is checked *above* the
//!   store on every read, so a backend that returns torn or stale bytes
//!   costs one recompile, never a wrong kernel. Errors are soft: they are
//!   counted, logged, and compilation falls back to `tileiras`.
//!
//! The methods map one-to-one onto a remote backend's operations, e.g. for S3:
//! `get` → `GetObject` (`None` on `NoSuchKey`), `put` → `PutObject`,
//! `delete` → `DeleteObject`, `clear` → list + batch delete. A word of
//! caution before pointing one at shared storage: a cache hit executes the
//! stored bytes as device code, and the entry header is an integrity check,
//! not an authenticity check — anyone with write access to the bucket can run
//! code on every machine that reads from it. Restrict writes to trusted CI,
//! or don't share the store across trust boundaries.
//!
//! To keep this example runnable anywhere, the backend here is an in-process
//! `HashMap`. That is useless as a real cache (it dies with the process, and
//! the in-memory kernel cache already covers the process lifetime) — it stands
//! in for your S3/database client so the *shape* is visible end to end: the
//! kernel below compiles once, the store records one `put`, and a `get` of the
//! same key returns the encoded entry.

use cutile::api::{ones, zeros};
use cutile::jit_cache::{self, JitStore};
use cutile::prelude::*;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};

/// Stand-in for a remote client (S3 bucket, database table, …).
struct InMemoryJitStore {
    entries: Mutex<HashMap<String, Vec<u8>>>,
}

impl JitStore for InMemoryJitStore {
    fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        // S3: GetObject, mapping NoSuchKey to Ok(None). SQL: SELECT value.
        Ok(self.entries.lock().unwrap().get(key).cloned())
    }

    fn put(&self, key: &str, value: &[u8]) -> io::Result<()> {
        // S3: PutObject. SQL: INSERT OR REPLACE.
        // Concurrent puts of one key are same-content by construction (the
        // key is content-addressed), so last-writer-wins is fine — but a
        // reader must never see a *partial* value. HashMap replacement is
        // atomic under the lock; on a filesystem this is write-temp + rename.
        self.entries
            .lock()
            .unwrap()
            .insert(key.to_string(), value.to_vec());
        Ok(())
    }

    // `contains` omitted on purpose: the trait's get-based default is fine
    // unless the backend has a cheaper existence check (HEAD request, stat).

    fn delete(&self, key: &str) -> io::Result<()> {
        // Absent keys are not an error — eviction races are expected.
        self.entries.lock().unwrap().remove(key);
        Ok(())
    }

    fn clear(&self) -> io::Result<()> {
        self.entries.lock().unwrap().clear();
        Ok(())
    }
}

#[cutile::module]
mod custom_store_example_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

fn main() {
    if !cutile_examples::requirements::BASELINE
        .check("jit_custom_store", 0)
        .expect("query example capabilities")
    {
        return;
    }

    let store = Arc::new(InMemoryJitStore {
        entries: Mutex::new(HashMap::new()),
    });
    jit_cache::enable(Arc::clone(&store) as Arc<dyn JitStore>);

    let x: Arc<Tensor<f32>> = ones(&[1024]).sync().expect("ones").into();
    let y: Arc<Tensor<f32>> = ones(&[1024]).sync().expect("ones").into();
    let z = zeros(&[1024]).sync().expect("zeros").partition([128]);

    let z_host = custom_store_example_module::add(z, x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add kernel");
    assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));

    let stats = jit_cache::stats();
    let entries = store.entries.lock().unwrap();
    println!(
        "store entries: {}, puts: {}, io errors: {}",
        entries.len(),
        stats.puts,
        stats.io_errors,
    );
    for (key, value) in entries.iter() {
        println!(
            "  {key} → {} bytes (encoded entry: header + cubin)",
            value.len()
        );
    }
}
