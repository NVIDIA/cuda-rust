/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deferred reclamation of results cancelled while GPU work is in flight.
//!
//! Cancelling a [`DeviceFuture`] cannot cancel GPU work that has already
//! been submitted: kernels run to completion regardless of what the host
//! does. What cancellation *can* do is decide when the host releases the
//! resources the kernel is still using. Dropping them immediately is
//! unsound (the stream-ordered allocator may hand the memory to the next
//! allocation while the kernel still writes to it), and synchronizing the
//! stream inside `Drop` blocks executor threads for an unbounded time.
//!
//! Instead, cancelled in-flight results are *parked* here together with a
//! completion gate (a CUDA event recorded on the assigned stream after the
//! submitted work):
//!
//! ```text
//! drop(future)                 later sweep (poll / drop / drain)
//!   record event on stream       event passed?  -> drop the result
//!   park (event, result)         still running? -> keep it parked
//! ```
//!
//! Parked results are swept opportunistically on subsequent future polls
//! and drops, or deterministically via [`drain`]. Payloads are never
//! dropped inside a CUDA host callback (callbacks must not call CUDA APIs,
//! and dropping a device buffer enqueues an async free), and a sweep never
//! blocks on GPU progress.
//!
//! Entries that are still parked at process exit are leaked; the driver
//! reclaims device memory at process teardown, while dropping them early
//! could corrupt memory that is still in use.
//!
//! [`DeviceFuture`]: crate::simt::device_future::DeviceFuture

use cuda_core::{CudaEvent, DriverError};
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Completion gate guarding a parked result.
///
/// Production code uses a [`CudaEvent`] recorded on the stream that runs
/// the cancelled work. Tests substitute host-side mocks.
pub(crate) trait ReclaimGate: Send {
    /// Returns `true` when the device timeline has passed the gate.
    /// Must never block.
    fn passed(&self) -> Result<bool, DriverError>;

    /// Blocks until the device timeline has passed the gate.
    fn wait(&self) -> Result<(), DriverError>;
}

impl ReclaimGate for CudaEvent {
    fn passed(&self) -> Result<bool, DriverError> {
        self.query()
    }

    fn wait(&self) -> Result<(), DriverError> {
        self.synchronize()
    }
}

/// A parked result and the gate that decides when it may be dropped.
struct LimboEntry {
    /// Completion gate; the payload may only drop after it has passed.
    gate: Box<dyn ReclaimGate>,
    /// Type-erased result kept alive on behalf of the cancelled future.
    payload: Box<dyn Send>,
}

/// Number of parked entries. Mirrors `LIMBO.len()`; updated only while the
/// limbo lock is held, read lock-free as the sweep fast path.
static PENDING: AtomicUsize = AtomicUsize::new(0);

/// All currently parked entries, across every device and thread.
static LIMBO: Mutex<Vec<LimboEntry>> = Mutex::new(Vec::new());

/// Locks the limbo list, recovering from poisoning.
///
/// A poisoned lock only means another thread panicked while holding it;
/// the list itself stays consistent because payloads are always dropped
/// outside the lock.
fn limbo() -> MutexGuard<'static, Vec<LimboEntry>> {
    LIMBO
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Parks `payload` until `gate` reports completion.
pub(crate) fn park(gate: impl ReclaimGate + 'static, payload: Box<dyn Send>) {
    let mut entries = limbo();
    entries.push(LimboEntry {
        gate: Box::new(gate),
        payload,
    });
    PENDING.store(entries.len(), Ordering::Release);
}

/// Number of results currently parked awaiting reclamation.
pub fn pending() -> usize {
    PENDING.load(Ordering::Acquire)
}

/// Drops every parked result whose gate has passed. Never blocks on GPU
/// progress. Returns the number of results reclaimed.
///
/// Called automatically on every [`DeviceFuture`] poll and drop; safe to
/// call manually at any time.
///
/// [`DeviceFuture`]: crate::simt::device_future::DeviceFuture
pub fn sweep() -> usize {
    if PENDING.load(Ordering::Acquire) == 0 {
        return 0;
    }
    let completed: Vec<LimboEntry> = {
        let mut entries = limbo();
        let mut kept = Vec::with_capacity(entries.len());
        let mut done = Vec::new();
        for entry in entries.drain(..) {
            match entry.gate.passed() {
                Ok(true) => done.push(entry),
                // Still in flight, or the query failed. Dropping early
                // would release memory the device may still be using, so
                // the entry stays parked.
                Ok(false) | Err(_) => kept.push(entry),
            }
        }
        *entries = kept;
        PENDING.store(entries.len(), Ordering::Release);
        done
    };
    // Dropped outside the lock: a payload's own drop may park or sweep.
    let reclaimed = completed.len();
    drop(completed);
    reclaimed
}

/// Blocking drain: waits for every parked gate to pass, then drops the
/// payload. Returns the number of results reclaimed.
///
/// When even the blocking wait fails, the payload is deliberately leaked
/// with a message on stderr: if the driver cannot prove the GPU work
/// finished, leaking is safer than freeing memory the device may still
/// write to.
///
/// Call this before tearing down a process or test when deterministic
/// reclamation is required. It must not be called from a CUDA host
/// callback or while holding a thread-local device-context borrow.
pub fn drain() -> usize {
    let entries: Vec<LimboEntry> = {
        let mut entries = limbo();
        let drained = mem::take(&mut *entries);
        PENDING.store(0, Ordering::Release);
        drained
    };
    let mut reclaimed = 0;
    for entry in entries {
        let waited = match entry.gate.passed() {
            Ok(true) => Ok(()),
            _ => entry.gate.wait(),
        };
        match waited {
            Ok(()) => {
                drop(entry.payload);
                reclaimed += 1;
            }
            Err(error) => {
                crate::leak::report_leak(format_args!(
                    "cuda-async: leaking a cancelled in-flight result; the driver \
                     could not prove the GPU work finished: {error}"
                ));
                mem::forget(entry.payload);
            }
        }
    }
    reclaimed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// Serializes the tests in this module: they share the global limbo,
    /// and `drain` consumes (and waits on) every parked entry.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_lock() -> MutexGuard<'static, ()> {
        TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Host-side stand-in for a recorded CUDA event.
    struct MockGate {
        /// Mirrors "the device timeline has passed the recorded event".
        passed: Arc<AtomicBool>,
        /// When `true`, the blocking wait reports a driver failure.
        wait_fails: bool,
    }

    impl ReclaimGate for MockGate {
        fn passed(&self) -> Result<bool, DriverError> {
            Ok(self.passed.load(Ordering::Relaxed))
        }

        fn wait(&self) -> Result<(), DriverError> {
            if self.wait_fails {
                Err(DriverError(
                    cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
                ))
            } else {
                self.passed.store(true, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    struct CountDrop(Arc<AtomicUsize>);

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn sweep_keeps_payload_parked_until_gate_passes() {
        let _guard = test_lock();
        let drops = Arc::new(AtomicUsize::new(0));
        let gate_passed = Arc::new(AtomicBool::new(false));

        park(
            MockGate {
                passed: Arc::clone(&gate_passed),
                wait_fails: false,
            },
            Box::new(CountDrop(Arc::clone(&drops))),
        );

        sweep();
        assert_eq!(
            drops.load(Ordering::Relaxed),
            0,
            "payload must stay parked while the GPU work is in flight"
        );

        gate_passed.store(true, Ordering::Relaxed);
        sweep();
        assert_eq!(
            drops.load(Ordering::Relaxed),
            1,
            "payload must drop once the gate reports completion"
        );
    }

    #[test]
    fn drain_waits_for_gate_then_drops_payload() {
        let _guard = test_lock();
        let drops = Arc::new(AtomicUsize::new(0));
        let gate_passed = Arc::new(AtomicBool::new(false));

        park(
            MockGate {
                passed: Arc::clone(&gate_passed),
                wait_fails: false,
            },
            Box::new(CountDrop(Arc::clone(&drops))),
        );

        let reclaimed = drain();
        assert_eq!(reclaimed, 1);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert!(
            gate_passed.load(Ordering::Relaxed),
            "drain must have waited on the gate before dropping"
        );
    }

    #[test]
    fn drain_leaks_payload_when_wait_fails() {
        let _guard = test_lock();
        let drops = Arc::new(AtomicUsize::new(0));

        park(
            MockGate {
                passed: Arc::new(AtomicBool::new(false)),
                wait_fails: true,
            },
            Box::new(CountDrop(Arc::clone(&drops))),
        );

        let mut capture = crate::leak::capture::start();
        let reclaimed = drain();
        let reports = capture.take();
        assert_eq!(reclaimed, 0);
        assert_eq!(
            drops.load(Ordering::Relaxed),
            0,
            "an unprovable gate must leak the payload, never drop it early"
        );
        assert_eq!(reports.len(), 1, "the leak must be reported: {reports:?}");
    }
}
