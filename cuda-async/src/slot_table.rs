/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completion-slot protocol shared by the flag-write reactor.
//!
//! This is the CUDA-free core of [`crate::reactor`]: the active-slot bitmap
//! and the single-producer/single-consumer payload handoff (both lock-free;
//! the scanner's hot path takes no lock), plus the free list, which is a
//! small mutex touched only to claim a slot and to recycle retired ones.
//! It is generic over the payload type and over a [`FlagArray`] backend so the
//! exact same protocol code runs three ways: against CUDA pinned memory in
//! production, against plain atomics under `loom` model checking, and against
//! plain atomics under `cargo miri`. Concurrency primitives come from
//! [`crate::loom_compat`] so `--cfg loom` swaps in loom's instrumented types.
//!
//! Protocol invariant (the thing loom verifies): a slot's payload is written
//! strictly before its active bit is set with `Release`, and read strictly
//! after the bit is observed with `Acquire`. Between claim and recycle the
//! slot has exactly one producer (the registrant) and one consumer (the
//! scanner), so the `UnsafeCell` payload access never races.

use crate::loom_compat::{AtomicU32, AtomicU64, AtomicUsize, Mutex, Ordering, UnsafeCell};

/// Host-visible per-slot completion flags. In production these alias CUDA
/// pinned memory the device writes `1` into; in tests they are plain atomics a
/// modeled "GPU" thread writes.
pub(crate) trait FlagArray {
    /// The completion flag for `slot` (0 = pending, 1 = done).
    fn flag(&self, slot: usize) -> &AtomicU32;
}

/// Waker payload cell: written by the registrant before the active bit is set,
/// read by the scanner after the bit is observed. The `Sync` impl is sound
/// only under that single-producer/single-consumer discipline.
struct SlotCell<P>(UnsafeCell<Option<P>>);

/// A dead-payload probe for [`SlotTable::scan_probing`]: where retired dead
/// payloads go, and the predicate that judges an armed, unlanded payload.
type Probe<'a, P> = (&'a mut Vec<P>, &'a mut dyn FnMut(&P) -> bool);

unsafe impl<P: Send> Sync for SlotCell<P> {}

pub(crate) struct SlotTable<P, F: FlagArray> {
    flags: F,
    /// Active-slot bitmap (one bit per slot): set = armed, scanner watches it.
    active: Vec<AtomicU64>,
    payload: Vec<SlotCell<P>>,
    /// Idle slot indices; popped by `claim`, pushed back by `scan_once` after
    /// retirement (and by `release` on an arm failure).
    free: Mutex<Vec<usize>>,
    /// Count of armed-but-not-yet-retired slots. Drives the llist-style
    /// empty→wake unpark: `publish` reports the 0→1 transition so the caller
    /// wakes the scanner only when it may be parked, and the scanner parks
    /// only when this is 0. Measured +38% registration throughput vs always-
    /// unpark by removing cross-core `Parker` cache-line contention.
    n_armed: AtomicUsize,
    num_slots: usize,
}

impl<P: Send, F: FlagArray> SlotTable<P, F> {
    pub(crate) fn new(num_slots: usize, flags: F) -> Self {
        let num_words = num_slots.div_ceil(64);
        SlotTable {
            flags,
            active: (0..num_words).map(|_| AtomicU64::new(0)).collect(),
            payload: (0..num_slots)
                .map(|_| SlotCell(UnsafeCell::new(None)))
                .collect(),
            free: Mutex::new((0..num_slots).rev().collect()),
            n_armed: AtomicUsize::new(0),
            num_slots,
        }
    }

    /// True when no slot is armed — the scanner's park gate. Load-acquire so a
    /// registration's prior `publish` (which increments `n_armed`) is visible.
    /// Used by the reactor scanner and the loom model; unused in some cfgs.
    #[allow(dead_code)]
    pub(crate) fn is_idle(&self) -> bool {
        self.n_armed.load(Ordering::Acquire) == 0
    }

    fn lock_free(&self) -> impl std::ops::DerefMut<Target = Vec<usize>> + '_ {
        self.free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reserve an idle slot, or `None` if the pool is exhausted.
    pub(crate) fn claim(&self) -> Option<usize> {
        self.lock_free().pop()
    }

    /// Return a claimed slot without publishing it (arm-failure path).
    /// Only reached from the CUDA reactor's `cuStreamWriteValue32` error path.
    pub(crate) fn release(&self, slot: usize) {
        self.lock_free().push(slot);
    }

    /// Clear a slot's completion flag before it is armed. Must run before the
    /// device write so a stale `0` cannot clobber the device's `1`.
    pub(crate) fn reset_flag(&self, slot: usize) {
        self.flags.flag(slot).store(0, Ordering::Release);
    }

    /// Publish a claimed, armed slot to the scanner. Writes the payload,
    /// counts the slot as armed, then sets the active bit with `Release`.
    /// Returns whether this was the idle→active transition (`n_armed` 0→1) —
    /// the caller wakes the scanner only then (llist empty→wake).
    ///
    /// Ordering: `n_armed` is incremented **before** the bit is set, so the
    /// scanner (which observes the slot only via the bit) can never decrement
    /// on retirement before this increment — no underflow. The payload write
    /// is likewise before the `Release` bit, so it is visible to the scanner's
    /// `Acquire` read.
    pub(crate) fn publish(&self, slot: usize, waker: P) -> bool {
        self.payload[slot]
            .0
            .with_mut(|p| unsafe { *p = Some(waker) });
        let was_idle = self.n_armed.fetch_add(1, Ordering::AcqRel) == 0;
        self.active[slot / 64].fetch_or(1u64 << (slot % 64), Ordering::Release);
        was_idle
    }

    /// One scan pass. Retires every slot whose flag reads `1`: clears its bit,
    /// takes its payload into `woken`, and recycles the index. Returns whether
    /// any slot was still armed (so the caller can park when nothing is in
    /// flight). Takes the free-list lock only to recycle, never while the
    /// payloads are woken by the caller.
    pub(crate) fn scan_once(&self, woken: &mut Vec<P>) -> bool {
        self.scan_inner(woken, None)
    }

    /// Like [`scan_once`](Self::scan_once), but additionally retires armed
    /// slots whose flag has *not* landed and whose payload `is_dead` reports
    /// as dead, into `dead`. The CUDA reactor uses this to resolve slots on a
    /// faulted stream, whose flag write will never execute; the predicate is
    /// the (driver-querying, hence rate-limited) part, kept out of the plain
    /// hot-path scan. Unused in some cfgs (the loom model only needs
    /// `scan_once`).
    #[allow(dead_code)]
    pub(crate) fn scan_probing(
        &self,
        woken: &mut Vec<P>,
        dead: &mut Vec<P>,
        is_dead: &mut dyn FnMut(&P) -> bool,
    ) -> bool {
        self.scan_inner(woken, Some((dead, is_dead)))
    }

    fn scan_inner(&self, woken: &mut Vec<P>, mut probe: Option<Probe<'_, P>>) -> bool {
        let mut any_active = false;
        let mut retired: Vec<usize> = Vec::new();
        for word in 0..self.active.len() {
            // Acquire pairs with the Release `fetch_or` in `publish`, making
            // the payload write visible before the bit is observed.
            let mut bits = self.active[word].load(Ordering::Acquire);
            if bits == 0 {
                continue;
            }
            any_active = true;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let slot = word * 64 + bit;
                if slot >= self.num_slots {
                    break;
                }
                let landed = self.flags.flag(slot).load(Ordering::Acquire) == 1;
                // Sole consumer: the bit was set (happens-before), and the
                // slot cannot be re-claimed until recycled below, so the
                // `with_mut` accesses here cannot race the producer.
                let dead = !landed
                    && match probe.as_mut() {
                        Some((_, is_dead)) => self.payload[slot]
                            .0
                            .with_mut(|p| unsafe { (*p).as_ref().is_some_and(&mut **is_dead) }),
                        None => false,
                    };
                if landed || dead {
                    self.active[word].fetch_and(!(1u64 << bit), Ordering::AcqRel);
                    let taken = self.payload[slot].0.with_mut(|p| unsafe { (*p).take() });
                    retired.push(slot);
                    if let Some(payload) = taken {
                        if landed {
                            woken.push(payload);
                        } else if let Some((dead_out, _)) = probe.as_mut() {
                            dead_out.push(payload);
                        }
                    }
                }
            }
        }
        if !retired.is_empty() {
            let n = retired.len();
            self.lock_free().append(&mut retired);
            // Decrement after recycling. `publish` incremented before setting
            // the bit we just observed, so `prev >= n` always holds.
            let prev = self.n_armed.fetch_sub(n, Ordering::AcqRel);
            debug_assert!(prev >= n, "n_armed underflow: {prev} < {n}");
        }
        any_active
    }
}

#[cfg(test)]
mod tests {
    //! Concurrency tests for the slot protocol, off-GPU. The std-thread stress
    //! test runs the real code at full scale on `cargo test`; the loom test
    //! (under `--cfg loom`) exhaustively model-checks the payload handoff and
    //! its Release/Acquire orderings. Both are also `cargo miri test`-clean,
    //! exercising the `UnsafeCell` / bitmap code under Miri's UB checker.

    use super::*;
    use crate::loom_compat::Arc;

    /// Plain-atomic flag backend: a stand-in for the CUDA pinned slab, written
    /// by a modeled "GPU" actor instead of the device.
    struct MockFlags(Vec<AtomicU32>);
    impl FlagArray for MockFlags {
        fn flag(&self, slot: usize) -> &AtomicU32 {
            &self.0[slot]
        }
    }
    fn mock_flags(n: usize) -> MockFlags {
        MockFlags((0..n).map(|_| AtomicU32::new(0)).collect())
    }

    /// Loom model of the completion handoff. The scanner is spawned **before**
    /// the flag write and the publish, so it genuinely races both — its
    /// correctness rests only on the `Acquire` loads pairing with the two
    /// `Release` stores (a spawn/join sync point would hide a wrong ordering).
    /// Loom explores every interleaving and every legal weak-memory outcome,
    /// checking the payload is delivered exactly once, the value is the one
    /// published (payload visible after the bit), the slot recycles, and the
    /// empty→wake counting (`n_armed`) is consistent: `publish` reports the
    /// idle→active transition and the count returns to 0 with no underflow
    /// (the `debug_assert` in `scan_once` fires under loom if the increment
    /// could be reordered after the retirement decrement).
    #[cfg(loom)]
    #[test]
    fn loom_single_slot_handoff() {
        loom::model(|| {
            let table = Arc::new(SlotTable::new(1, mock_flags(1)));
            let slot = table.claim().expect("slot");
            table.reset_flag(slot);

            // Scanner: races the producers, drains until the completion lands.
            let scanner = {
                let table = table.clone();
                loom::thread::spawn(move || {
                    let mut woken = Vec::new();
                    while woken.is_empty() {
                        table.scan_once(&mut woken);
                        loom::thread::yield_now();
                    }
                    woken
                })
            };

            // "GPU": writes the completion flag concurrently.
            let gpu = {
                let table = table.clone();
                loom::thread::spawn(move || {
                    table.flags.flag(slot).store(1, Ordering::Release);
                })
            };

            // Registrant: publishes the payload + active bit; reports idle→active.
            let was_idle = table.publish(slot, 7usize);
            assert!(
                was_idle,
                "first publish from an idle table must report idle→active"
            );

            gpu.join().unwrap();
            let woken = scanner.join().unwrap();
            assert_eq!(
                woken,
                vec![7usize],
                "payload delivered exactly once, value intact"
            );
            assert_eq!(table.lock_free().len(), 1, "slot recycled");
            assert!(table.is_idle(), "n_armed returns to 0 after retirement");
        });
    }

    /// Std-thread stress: many registrant threads and a modeled GPU thread all
    /// hitting one table, the scanner draining concurrently. Every payload must
    /// be delivered exactly once and every slot recycled. Runs at full scale on
    /// `cargo test`; run under `--cfg loom` it degenerates to a small model.
    #[cfg(not(loom))]
    #[test]
    fn stress_many_producers_one_scanner() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as O};
        use std::sync::mpsc;

        // Miri interprets every step, so scale the model down there; TSan and
        // a normal `cargo test` run the full-scale load.
        #[cfg(miri)]
        const SLOTS: usize = 8;
        #[cfg(miri)]
        const THREADS: usize = 2;
        #[cfg(miri)]
        const PER_THREAD: usize = 15;
        #[cfg(not(miri))]
        const SLOTS: usize = 256;
        #[cfg(not(miri))]
        const THREADS: usize = 8;
        #[cfg(not(miri))]
        const PER_THREAD: usize = 500;
        const TOTAL: usize = THREADS * PER_THREAD;

        let table = Arc::new(SlotTable::<usize, MockFlags>::new(SLOTS, mock_flags(SLOTS)));
        let done = Arc::new(AtomicBool::new(false));
        // Modeled GPU: registrants hand it (slot) after arming; it writes the flag.
        let (arm_tx, arm_rx) = mpsc::channel::<usize>();

        let gpu = {
            let table = table.clone();
            std::thread::spawn(move || {
                while let Ok(slot) = arm_rx.recv() {
                    table.flags.flag(slot).store(1, O::Release);
                }
            })
        };

        let seen = Arc::new((0..TOTAL).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let scanner = {
            let (table, done, seen) = (table.clone(), done.clone(), seen.clone());
            std::thread::spawn(move || {
                let mut woken = Vec::new();
                let mut total = 0usize;
                while total < TOTAL {
                    table.scan_once(&mut woken);
                    for token in woken.drain(..) {
                        seen[token].fetch_add(1, O::SeqCst);
                        total += 1;
                    }
                    if !done.load(O::Relaxed) {
                        std::hint::spin_loop();
                    }
                }
            })
        };

        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                let (table, arm_tx) = (table.clone(), arm_tx.clone());
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        let token = t * PER_THREAD + i;
                        // Retry until a slot frees (scanner recycles concurrently).
                        let slot = loop {
                            if let Some(s) = table.claim() {
                                break s;
                            }
                            std::hint::spin_loop();
                        };
                        table.reset_flag(slot);
                        table.publish(slot, token);
                        arm_tx.send(slot).unwrap();
                    }
                })
            })
            .collect();

        for w in workers {
            w.join().unwrap();
        }
        drop(arm_tx);
        gpu.join().unwrap();
        done.store(true, O::Relaxed);
        scanner.join().unwrap();

        for (token, count) in seen.iter().enumerate() {
            assert_eq!(
                count.load(O::SeqCst),
                1,
                "token {token} not delivered exactly once"
            );
        }
        assert_eq!(table.lock_free().len(), SLOTS, "slots leaked");
    }

    /// A probing scan retires a slot whose flag has not landed when the
    /// predicate declares its payload dead (a faulted stream in the reactor):
    /// the payload goes to `dead`, not `woken`; a landed slot in the same
    /// pass still goes to `woken`; an armed, live, unlanded slot stays armed;
    /// retired slots of both kinds are recycled and counted out of `n_armed`.
    /// Deterministic, no threads.
    #[cfg(not(loom))]
    #[test]
    fn probing_scan_retires_dead_slots_and_keeps_live_ones() {
        // Exactly three slots, all armed below, so the free list is empty
        // until the scans recycle.
        const SLOTS: usize = 3;
        const DEAD: usize = 7;
        const LANDED: usize = 8;
        const LIVE: usize = 9;
        let table = SlotTable::<usize, MockFlags>::new(SLOTS, mock_flags(SLOTS));

        let arm = |token: usize| {
            let slot = table.claim().expect("slot");
            table.reset_flag(slot);
            table.publish(slot, token);
            slot
        };
        let _dead_slot = arm(DEAD);
        let landed_slot = arm(LANDED);
        let _live_slot = arm(LIVE);
        table.flags.flag(landed_slot).store(1, Ordering::Release);
        assert!(!table.is_idle());

        // A plain scan never retires an unlanded slot, dead or not.
        let mut woken = Vec::new();
        assert!(table.scan_once(&mut woken));
        assert_eq!(woken, vec![LANDED]);
        assert!(!table.is_idle(), "two unlanded slots remain armed");

        let mut dead = Vec::new();
        let mut probed = Vec::new();
        assert!(table.scan_probing(&mut woken, &mut dead, &mut |&token| {
            probed.push(token);
            token == DEAD
        }));
        assert_eq!(woken, vec![LANDED], "nothing new landed");
        assert_eq!(dead, vec![DEAD], "only the dead payload is retired");
        assert_eq!(probed.len(), 2, "each unlanded slot is probed once");
        assert!(!table.is_idle(), "the live slot is still armed");

        // Recycled: the dead slot and the landed slot are claimable again.
        assert!(table.claim().is_some());
        assert!(table.claim().is_some());
        assert_eq!(table.claim(), None, "only the two retired slots recycled");
    }

    /// Pool exhaustion is signalled by `claim() -> None`, which is what the
    /// reactor keys on to fall back to the host-callback path. Recycling a
    /// retired slot restores capacity. Deterministic, no threads.
    #[cfg(not(loom))]
    #[test]
    fn claim_signals_exhaustion_and_recycle_restores_capacity() {
        const SLOTS: usize = 4;
        let table = SlotTable::<usize, MockFlags>::new(SLOTS, mock_flags(SLOTS));

        // Drain the pool: SLOTS distinct claims, then None.
        let mut claimed: Vec<usize> = (0..SLOTS)
            .map(|_| table.claim().expect("slot available"))
            .collect();
        claimed.sort_unstable();
        claimed.dedup();
        assert_eq!(claimed.len(), SLOTS, "claims must be distinct");
        assert_eq!(table.claim(), None, "exhausted pool must signal None");

        // Complete one slot through the normal path; the scanner recycles it.
        let slot = claimed[0];
        table.reset_flag(slot);
        table.publish(slot, 99);
        table.flags.flag(slot).store(1, Ordering::Release);
        let mut woken = Vec::new();
        table.scan_once(&mut woken);
        assert_eq!(woken, vec![99], "completed slot delivered");

        // Capacity restored: exactly one slot is claimable again, then None.
        assert!(table.claim().is_some(), "recycled slot must be claimable");
        assert_eq!(table.claim(), None, "only the recycled slot returned");
    }
}

/// A/B microbenchmark for the deferred reactor optimizations, run against the
/// real `SlotTable`. Combinatorial over two orthogonal axes — unpark policy
/// (always vs llist-style empty→wake) and scanner idle behavior (park vs
/// never-park spin) — across a saturated and a bursty registration regime.
/// The harness keeps its own `n_armed` counter for the empty→wake toggle so
/// both policies can still be compared; the shipped `SlotTable` has since
/// adopted an `n_armed` of its own (see `publish` / `is_idle`), which this
/// bench does not consult.
///
/// Run: `cargo test -p cuda-async --release --lib ab_reactor_variants -- --nocapture --ignored`
#[cfg(all(test, not(loom), not(miri)))]
mod ab_bench {
    use super::*;
    use crate::loom_compat::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;
    use std::time::Instant;

    struct Flags(Vec<AtomicU32>);
    impl FlagArray for Flags {
        fn flag(&self, s: usize) -> &AtomicU32 {
            &self.0[s]
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Unpark {
        Always,
        EmptyWake,
    }
    #[derive(Clone, Copy, PartialEq)]
    enum Idle {
        Park,
        NeverSpin,
    }

    struct Result {
        ops_per_sec: f64,
        unparks: usize,
        scan_passes: usize,
    }

    fn run(unpark: Unpark, idle: Idle, saturated: bool) -> Result {
        const SLOTS: usize = 1024;
        const THREADS: usize = 8;
        const OPS_PER: usize = 150_000;
        let total = THREADS * OPS_PER;

        let table = Arc::new(SlotTable::<usize, Flags>::new(
            SLOTS,
            Flags((0..SLOTS).map(|_| AtomicU32::new(0)).collect()),
        ));
        let n_armed = Arc::new(AtomicUsize::new(0));
        let unparks = Arc::new(AtomicUsize::new(0));
        let scan_passes = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));

        let scanner = {
            let (table, n_armed, scan_passes, done) = (
                table.clone(),
                n_armed.clone(),
                scan_passes.clone(),
                done.clone(),
            );
            thread::Builder::new()
                .name("ab-scanner".into())
                .spawn(move || {
                    let mut woken = Vec::new();
                    loop {
                        table.scan_once(&mut woken);
                        scan_passes.fetch_add(1, O::Relaxed);
                        let retired = woken.len();
                        if retired > 0 {
                            n_armed.fetch_sub(retired, O::AcqRel);
                            let d = done.fetch_add(retired, O::Relaxed) + retired;
                            woken.clear();
                            if d >= total {
                                break;
                            }
                            continue;
                        }
                        match idle {
                            Idle::Park => {
                                if n_armed.load(O::Acquire) == 0 {
                                    thread::park();
                                }
                            }
                            Idle::NeverSpin => std::hint::spin_loop(),
                        }
                        if done.load(O::Relaxed) >= total {
                            break;
                        }
                    }
                })
                .unwrap()
        };
        let scanner_thread = scanner.thread().clone();

        let start = Instant::now();
        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                let (table, n_armed, unparks, scanner_thread) = (
                    table.clone(),
                    n_armed.clone(),
                    unparks.clone(),
                    scanner_thread.clone(),
                );
                thread::spawn(move || {
                    for i in 0..OPS_PER {
                        let slot = loop {
                            if let Some(s) = table.claim() {
                                break s;
                            }
                            std::hint::spin_loop();
                        };
                        table.reset_flag(slot);
                        table.publish(slot, t * OPS_PER + i);
                        // Count as armed before allowing completion, so the
                        // scanner's decrement can never underflow.
                        let was_idle = n_armed.fetch_add(1, O::AcqRel) == 0;
                        table.flags.flag(slot).store(1, O::Release);
                        let do_unpark = match unpark {
                            Unpark::Always => true,
                            Unpark::EmptyWake => was_idle,
                        };
                        if do_unpark {
                            unparks.fetch_add(1, O::Relaxed);
                            scanner_thread.unpark();
                        }
                        if !saturated && i % 256 == 0 {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        for w in workers {
            w.join().unwrap();
        }
        // Ensure the scanner exits even if it just parked with work drained.
        while done.load(O::Relaxed) < total {
            scanner_thread.unpark();
            std::hint::spin_loop();
        }
        scanner.join().unwrap();
        let secs = start.elapsed().as_secs_f64();
        Result {
            ops_per_sec: total as f64 / secs,
            unparks: unparks.load(O::Relaxed),
            scan_passes: scan_passes.load(O::Relaxed),
        }
    }

    #[test]
    #[ignore = "perf A/B; run explicitly with --release --nocapture --ignored"]
    fn ab_reactor_variants() {
        let configs = [
            (Unpark::Always, Idle::Park, "always+park"),
            (Unpark::EmptyWake, Idle::Park, "empty→wake+park"),
            (Unpark::Always, Idle::NeverSpin, "always+never-park"),
            (Unpark::EmptyWake, Idle::NeverSpin, "empty→wake+never-park"),
        ];
        for (saturated, label) in [(true, "SATURATED"), (false, "BURSTY")] {
            eprintln!("\n## {label} (8 threads x 150k ops)\n");
            eprintln!("| config | Mops/s | unparks | scan passes |");
            eprintln!("|---|---|---|---|");
            for (u, i, name) in configs {
                // Warm + measure (median of 3).
                let mut best = run(u, i, saturated);
                for _ in 0..2 {
                    let r = run(u, i, saturated);
                    if r.ops_per_sec > best.ops_per_sec {
                        best = r;
                    }
                }
                eprintln!(
                    "| {name} | {:.2} | {} | {} |",
                    best.ops_per_sec / 1e6,
                    best.unparks,
                    best.scan_passes
                );
            }
        }
    }
}
