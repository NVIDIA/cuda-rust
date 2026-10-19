/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Flag-write completion reactor (always compiled in; runtime-selectable via
//! `CUDA_ASYNC_HOST_SYNC`, see [`crate::device_future`]).
//!
//! Replaces per-completion `cuLaunchHostFunc` callbacks with a
//! `cuStreamWriteValue32` into a slot of pinned host memory at pipeline end,
//! plus one process-wide reactor thread that scans pending slots (plain
//! memory loads, no driver calls on the hot path) and fires wakers. The
//! wakeup cost amortizes across all in-flight pipelines instead of paying a
//! driver-thread hop per pipeline.
//!
//! The harvest protocol — the active-slot bitmap, the
//! single-producer/single-consumer payload handoff (both lock-free), and the
//! mutex-guarded free list — lives in [`crate::slot_table`], CUDA-free and
//! model-checked under `loom`/`miri`. This module is the thin CUDA binding:
//! it owns the pinned flag slab, the device write that arms a slot, and the
//! scanner thread.
//!
//! # Faulted streams
//!
//! A device fault (illegal address, `trap`, ...) kills the context; the
//! armed flag writes on that context never execute, so their slots would
//! stay armed forever, the awaiting futures would never resolve, and the
//! scanner — gated on "anything armed" — would never park. Once the scanner
//! has spun without progress for a while it therefore probes the stream of
//! every still-armed slot (rate-limited, one driver query per distinct
//! stream per probe): a stream the driver reports an error for is retired
//! and its future is woken *without* being marked complete, so its next poll
//! observes the driver error itself (see `DeviceFuture::poll`). A capturing
//! stream is never queried (that would invalidate the capture).

use crate::device_future::{probe_stream, StreamCallbackState, StreamHealth};
use crate::error::DeviceError;
use crate::slot_table::{FlagArray, SlotTable};
use cuda_core::Stream;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const NUM_SLOTS: usize = 1024;
/// Spin passes over the active set before yielding between scans.
const SPIN_PASSES: u32 = 10_000;
/// While armed slots make no progress, how often their streams are probed
/// for a fault. Long-running kernels pay one `cuStreamQuery` per distinct
/// stream per interval; a faulted stream resolves within about an interval.
const STALE_PROBE_INTERVAL: Duration = Duration::from_millis(2);

/// Completion flags backed by CUDA pinned memory. `host` is the CPU-visible
/// mapping the scanner loads; the device writes `1` into the same bytes
/// through the device-side alias (`Reactor::dptr`).
struct CudaFlags {
    host: *mut u32,
}

// SAFETY: `host` points at pinned, device-mapped memory that outlives the
// process; sharing the pointer across the registrant and scanner threads is
// sound because all access goes through atomic loads/stores.
unsafe impl Send for CudaFlags {}
unsafe impl Sync for CudaFlags {}

impl FlagArray for CudaFlags {
    fn flag(&self, slot: usize) -> &AtomicU32 {
        // Pinned memory is coherent between device writes and host loads; an
        // atomic view of the slot gives the compiler-level guarantees.
        unsafe { AtomicU32::from_ptr(self.host.add(slot)) }
    }
}

/// What a slot carries: the waker to fire, and the stream whose flag write
/// completes the slot (kept alive, and probed if the slot goes stale).
struct Registration {
    waker_state: Arc<StreamCallbackState>,
    stream: Arc<Stream>,
}

struct Reactor {
    table: SlotTable<Registration, CudaFlags>,
    /// Device-side alias of the flag slab (CU_MEMHOSTALLOC_DEVICEMAP).
    dptr: cuda_bindings::CUdeviceptr,
    scanner: thread::Thread,
}

fn internal(msg: String) -> DeviceError {
    DeviceError::Internal(msg)
}

/// Initializes the reactor on first use. Requires a current CUDA context on
/// the calling thread (true at registration time: the caller just launched
/// work on this thread).
fn reactor() -> Result<&'static Reactor, DeviceError> {
    static REACTOR: OnceLock<Result<Reactor, String>> = OnceLock::new();
    let result = REACTOR.get_or_init(|| unsafe {
        let mut host = MaybeUninit::uninit();
        let flags =
            cuda_bindings::CU_MEMHOSTALLOC_PORTABLE | cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP;
        let code = cuda_bindings::cuMemHostAlloc(
            host.as_mut_ptr(),
            NUM_SLOTS * std::mem::size_of::<u32>(),
            flags,
        );
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(format!("cuMemHostAlloc failed: {code}"));
        }
        let host = host.assume_init() as *mut u32;
        std::ptr::write_bytes(host, 0, NUM_SLOTS);
        let mut dptr = MaybeUninit::uninit();
        let code = cuda_bindings::cuMemHostGetDevicePointer_v2(dptr.as_mut_ptr(), host as _, 0);
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(format!("cuMemHostGetDevicePointer failed: {code}"));
        }
        let dptr = dptr.assume_init();
        let handle = thread::Builder::new()
            .name("cuda-async-reactor".into())
            .spawn(scan_loop)
            .map_err(|e| format!("failed to spawn reactor thread: {e}"))?;
        Ok(Reactor {
            table: SlotTable::new(NUM_SLOTS, CudaFlags { host }),
            dptr,
            scanner: handle.thread().clone(),
        })
    });
    result.as_ref().map_err(|e| internal(e.clone()))
}

fn scan_loop() {
    // The OnceLock is initialized by the spawner; spin briefly until visible.
    let reactor = loop {
        if let Ok(r) = reactor() {
            break r;
        }
        thread::yield_now();
    };
    let mut idle_passes: u32 = 0;
    let mut woken: Vec<Registration> = Vec::new();
    let mut faulted: Vec<Registration> = Vec::new();
    let mut last_probe = Instant::now();
    loop {
        reactor.table.scan_once(&mut woken);
        if !woken.is_empty() {
            // Wakers fire outside any lock the scan held, so a registration
            // is never blocked behind a waking phase.
            for reg in woken.drain(..) {
                reg.waker_state.signal();
            }
            idle_passes = 0;
            continue;
        }
        // Park gate is the armed count, not "did this pass see a bit": a slot
        // may be armed but not yet flag-complete, and must keep the scanner
        // awake. `register` unparks only on the idle→active transition.
        if reactor.table.is_idle() {
            // Nothing in flight: park until a registration unparks us. An
            // unpark that lands between the scan and the park is absorbed by
            // the park token, so no registration is missed. Re-measured on
            // the lock-free scan: never-parking is latency-neutral (10.0-10.5
            // vs 10.2 us medians at N=1, budget 0) — the old +4 us penalty
            // was the scan-lock contention — so parking wins on idle CPU.
            thread::park();
            idle_passes = 0;
            last_probe = Instant::now();
            continue;
        }
        idle_passes += 1;
        if idle_passes < SPIN_PASSES {
            std::hint::spin_loop();
            continue;
        }
        // Slow phase: something has been armed for a while without landing.
        // Either a long kernel, or a stream whose flag write will never
        // execute because the context faulted. Probe periodically so the
        // latter resolves instead of pinning this thread in yield forever.
        if last_probe.elapsed() >= STALE_PROBE_INTERVAL {
            last_probe = Instant::now();
            probe_stale_slots(&reactor.table, &mut woken, &mut faulted);
            for reg in woken.drain(..) {
                reg.waker_state.signal();
            }
            for reg in faulted.drain(..) {
                // Wake without completing: the poll observes the fault.
                reg.waker_state.wake();
            }
        }
        thread::yield_now();
    }
}

/// One probing pass: retires slots whose flag has landed into `woken`, and
/// slots whose stream the driver reports faulted into `faulted`. Each
/// distinct stream is queried once per pass.
fn probe_stale_slots(
    table: &SlotTable<Registration, CudaFlags>,
    woken: &mut Vec<Registration>,
    faulted: &mut Vec<Registration>,
) {
    let mut memo: Vec<(cuda_bindings::CUstream, bool)> = Vec::new();
    table.scan_probing(woken, faulted, &mut |reg: &Registration| {
        let handle = reg.stream.cu_stream();
        if let Some((_, dead)) = memo.iter().find(|(h, _)| *h == handle) {
            return *dead;
        }
        let dead = stream_is_faulted(&reg.stream);
        memo.push((handle, dead));
        dead
    });
}

/// Whether the driver reports an error for `stream`. Conservative: if the
/// device cannot be bound on this thread the slot is left armed for a later
/// probe, and a capturing or merely busy stream is never dead. A retired
/// slot's flag write can no longer land — the error is the context's sticky
/// fault — so recycling the slot cannot be clobbered by a late device write.
fn stream_is_faulted(stream: &Stream) -> bool {
    if stream.device().bind_to_thread().is_err() {
        return false;
    }
    matches!(probe_stream(stream), StreamHealth::Faulted(_))
}

/// Registers a completion slot for work already submitted on `stream`:
/// enqueues a device flag write after the submitted work and publishes the
/// waker payload to the scanner via the active bitmap. Lock-free except for
/// the free-list pop.
///
/// # Safety
/// `stream` must be valid and the owning context current on this thread.
pub(crate) unsafe fn register(
    stream: &Arc<Stream>,
    waker_state: Arc<StreamCallbackState>,
) -> Result<(), DeviceError> {
    let reactor = reactor()?;
    let slot = reactor
        .table
        .claim()
        .ok_or_else(|| internal("reactor slot pool exhausted".into()))?;
    reactor.table.reset_flag(slot);
    let addr = reactor.dptr + (slot * std::mem::size_of::<u32>()) as u64;
    let code = cuda_bindings::cuStreamWriteValue32_v2(stream.cu_stream(), addr, 1, 0);
    if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
        reactor.table.release(slot);
        return Err(internal(format!("cuStreamWriteValue32 failed: {code}")));
    }
    // empty→wake: unpark only when this registration transitioned the reactor
    // from idle to active (the scanner may be parked). At higher registration
    // rates the scanner is already awake and the skipped unparks avoid
    // cross-core `Parker` contention (+38% throughput in the A/B).
    let registration = Registration {
        waker_state,
        stream: Arc::clone(stream),
    };
    if reactor.table.publish(slot, registration) {
        reactor.scanner.unpark();
    }
    Ok(())
}
