/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Concurrency-primitive shim so the completion protocol ([`crate::slot_table`])
//! can be model-checked by `loom`.
//!
//! Under `--cfg loom` (test-only) the sync primitives and `UnsafeCell` resolve
//! to loom's instrumented versions, which `loom::model` uses to explore thread
//! interleavings and weak-memory outcomes exhaustively; otherwise they are the
//! `std` originals with zero overhead. Patterned on tokio's `src/loom` module.
//!
//! The `UnsafeCell` wrapper presents loom's `with`/`with_mut` closure API on
//! both paths so the protocol code is written once. Loom tracks the cell
//! accesses inside those closures to prove the payload handoff is race-free.

// Which of these are used depends on the build config (reactor vs test vs
// loom), so suppress per-config unused-import noise.
#[cfg(not(loom))]
#[allow(unused_imports)]
pub(crate) use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(loom))]
#[allow(unused_imports)]
pub(crate) use std::sync::{Arc, Mutex};

#[cfg(loom)]
#[allow(unused_imports)]
pub(crate) use loom::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
#[allow(unused_imports)]
pub(crate) use loom::sync::{Arc, Mutex};

#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) fn new(data: T) -> Self {
        UnsafeCell(std::cell::UnsafeCell::new(data))
    }
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
