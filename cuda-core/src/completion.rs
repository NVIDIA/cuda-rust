/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completion-bound retirement without waiting in destructors.

use crate::{DriverError, Event, Stream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

extern "C" {
    fn atexit(cb: extern "C" fn()) -> core::ffi::c_int;
}

/// Deferred resource cleanup closure executed once its completion gate clears.
type Cleanup = Box<dyn FnOnce() + Send>;

/// Completion predicate gating when a [`Pending`] cleanup may run.
enum Readiness {
    /// Immediate deferred cleanup (`Allocation::drop`) with no event wait.
    Immediate,
    /// GPU stream event that must report completion before `cleanup` runs.
    Event(Arc<Event>),
    /// Deterministic non-GPU probe for unit-testing exit and error transitions.
    #[cfg(test)]
    Probe(Box<dyn Fn() -> Result<bool, DriverError> + Send>),
}

impl Readiness {
    /// Polls whether the gated GPU work has completed without blocking.
    fn query(&self) -> Result<bool, DriverError> {
        match self {
            Self::Immediate => Ok(true),
            Self::Event(event) => event.query(),
            #[cfg(test)]
            Self::Probe(probe) => probe(),
        }
    }
}

/// Single retirement item coupling an optional completion event with its cleanup.
struct Pending {
    readiness: Readiness,
    cleanup: Cleanup,
}

/// Producer-side inbox and process-exit flag protected by [`RetirementQueue::queue`].
#[derive(Default)]
struct Inbox {
    items: Vec<Pending>,
    exiting: bool,
}

/// Worker-side active retirement set protected by [`RetirementQueue::driver_gate`].
///
/// Lock hierarchy: always acquire `driver_gate` before `queue` when both are
/// needed; `enqueue` acquires `queue` alone so callers never wait on driver
/// teardown calls.
#[derive(Default)]
struct DriverState {
    exiting: bool,
    in_flight: Vec<Pending>,
}

/// Bounded-poll retirement queue with process-exit quiescence.
///
/// Background retirement must stop touching `libcuda` and Compute Sanitizer
/// (`libsanitizer-public.so`) before their `atexit` / `.fini_array` destructors
/// tear down global driver and sanitizer state at process exit (NVBug 7090395).
struct RetirementQueue {
    shutting_down: AtomicBool,
    queue: Mutex<Inbox>,
    wake: Condvar,
    driver_gate: Mutex<DriverState>,
}

impl RetirementQueue {
    /// Creates an empty retirement queue before worker attachment.
    fn new() -> Self {
        Self {
            shutting_down: AtomicBool::new(false),
            queue: Mutex::new(Inbox::default()),
            wake: Condvar::new(),
            driver_gate: Mutex::new(DriverState::default()),
        }
    }

    /// Reports whether process-exit teardown or permanent disablement has started.
    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Permanently closes the queue and leaks any queued items when worker startup fails.
    fn disable_permanently(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let mut gate = self.driver_gate.lock().unwrap_or_else(|e| e.into_inner());
        let mut inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        inbox.exiting = true;
        gate.exiting = true;
        for item in inbox.items.drain(..) {
            std::mem::forget(item);
        }
        for item in gate.in_flight.drain(..) {
            std::mem::forget(item);
        }
        drop(inbox);
        drop(gate);
        self.wake.notify_all();
    }

    /// Pushes `pending` onto the inbox, or leaks it once process exit has begun.
    fn enqueue(&self, pending: Pending) {
        let mut inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        if inbox.exiting {
            drop(inbox);
            std::mem::forget(pending);
            return;
        }
        inbox.items.push(pending);
        drop(inbox);
        self.wake.notify_one();
    }

    /// Drains ready items until no cascading ready work remains in `inbox`.
    ///
    /// Returns `None` once process exit has claimed `driver_gate`, or
    /// `Some(has_in_flight)` indicating whether incomplete GPU events remain.
    fn step(&self) -> Option<bool> {
        let mut gate = self.driver_gate.lock().unwrap_or_else(|e| e.into_inner());
        if gate.exiting {
            return None;
        }
        loop {
            {
                let mut inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                if inbox.exiting {
                    gate.exiting = true;
                    return None;
                }
                if inbox.items.is_empty() && gate.in_flight.is_empty() {
                    return Some(false);
                }
                gate.in_flight.append(&mut inbox.items);
            }
            drain_ready(&mut gate.in_flight);
            let more_queued = !self
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .items
                .is_empty();
            if !more_queued {
                return Some(!gate.in_flight.is_empty());
            }
        }
    }

    /// Parks the worker until new items arrive, a 1 ms poll elapses for
    /// in-flight events, or process exit closes the queue.
    fn wait_for_work(&self, has_in_flight: bool) -> bool {
        let inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        if inbox.exiting {
            return false;
        }
        if !inbox.items.is_empty() {
            return true;
        }
        if has_in_flight {
            let (guard, _) = self
                .wake
                .wait_timeout(inbox, Duration::from_millis(1))
                .unwrap_or_else(|e| e.into_inner());
            !guard.exiting
        } else {
            let guard = self.wake.wait(inbox).unwrap_or_else(|e| e.into_inner());
            !guard.exiting
        }
    }

    /// Drains already-completed items (both newly queued and already held in
    /// `in_flight` by the worker) while the CUDA driver is still live, then
    /// fences out the background worker and leaks any unfinished GPU leases
    /// without joining the worker thread (avoiding Windows loader-lock deadlocks).
    fn shutdown_for_exit(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let mut gate = self.driver_gate.lock().unwrap_or_else(|e| e.into_inner());
        if gate.exiting {
            return;
        }
        {
            let mut inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            gate.in_flight.append(&mut inbox.items);
        }
        loop {
            drain_ready(&mut gate.in_flight);
            let mut inbox = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            if inbox.items.is_empty() {
                inbox.exiting = true;
                break;
            }
            gate.in_flight.append(&mut inbox.items);
        }
        gate.exiting = true;
        for item in gate.in_flight.drain(..) {
            std::mem::forget(item);
        }
        self.wake.notify_all();
    }
}

/// Polls each item in `in_flight` once, running ready cleanups before dropping
/// their completion event and leaking items whose completion status errored.
fn drain_ready(in_flight: &mut Vec<Pending>) {
    let mut i = 0;
    while i < in_flight.len() {
        match in_flight[i].readiness.query() {
            Ok(false) => i += 1,
            Ok(true) => {
                let Pending { readiness, cleanup } = in_flight.swap_remove(i);
                // Preserve the event's context until cleanup has finished.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup));
                drop(readiness);
            }
            Err(_) => {
                // Completion is unknown: leaking the owner and its access
                // lease is safer than freeing memory or allowing conflicting work.
                std::mem::forget(in_flight.swap_remove(i));
            }
        }
    }
}

/// Coordinates lazy worker startup against process-exit shutdown so `on_process_exit`
/// sees and records shutdown state even while first-use initialization is in flight.
struct RetirementRuntime {
    queue: OnceLock<Arc<RetirementQueue>>,
    started: AtomicBool,
    init_lock: Mutex<()>,
}

impl RetirementRuntime {
    /// Creates an unstarted retirement runtime before `libcuda` initialization.
    const fn new() -> Self {
        Self {
            queue: OnceLock::new(),
            started: AtomicBool::new(false),
            init_lock: Mutex::new(()),
        }
    }

    /// Returns the shared queue, allocating it ahead of `atexit` registration so
    /// exit callbacks always observe the same queue instance during initialization.
    fn queue(&self) -> &Arc<RetirementQueue> {
        self.queue.get_or_init(|| Arc::new(RetirementQueue::new()))
    }

    /// Quiesces the shared queue during process exit without blocking on `init_lock`.
    fn shutdown_for_exit(&self) {
        self.queue().shutdown_for_exit();
    }

    /// Enqueues `pending` onto the runtime worker, starting it on first use.
    fn enqueue(&self, pending: Pending) {
        self.enqueue_with(
            pending,
            // SAFETY: `on_process_exit` is a stateless `extern "C"` function safe
            // to invoke during normal CRT `exit` processing.
            || unsafe { atexit(on_process_exit) } == 0,
            || {},
        );
    }

    /// Enqueues `pending` with injectable `atexit` registration and pre-spawn hooks,
    /// returning `true` when this call spawned the background worker thread.
    fn enqueue_with(
        &self,
        pending: Pending,
        register_atexit: impl FnOnce() -> bool,
        before_spawn: impl FnOnce(),
    ) -> bool {
        let queue = self.queue();
        let spawned = if !self.started.load(Ordering::Acquire) && !queue.is_shutting_down() {
            self.ensure_worker(queue, register_atexit, before_spawn)
        } else {
            false
        };
        queue.enqueue(pending);
        spawned
    }

    /// Registers the exit hook and spawns the worker thread unless shutdown has begun.
    fn ensure_worker(
        &self,
        queue: &Arc<RetirementQueue>,
        register_atexit: impl FnOnce() -> bool,
        before_spawn: impl FnOnce(),
    ) -> bool {
        let _guard = self.init_lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.started.load(Ordering::Acquire) || queue.is_shutting_down() {
            return false;
        }
        if !register_atexit() {
            queue.disable_permanently();
            self.started.store(true, Ordering::Release);
            return false;
        }
        before_spawn();
        if queue.is_shutting_down() {
            self.started.store(true, Ordering::Release);
            return false;
        }
        let worker_shared = queue.clone();
        let spawned = std::thread::Builder::new()
            .name("cuda-resource-retirement".into())
            .spawn(move || {
                while let Some(has_in_flight) = worker_shared.step() {
                    if !worker_shared.wait_for_work(has_in_flight) {
                        break;
                    }
                }
            })
            .is_ok();
        if !spawned {
            queue.disable_permanently();
        }
        self.started.store(true, Ordering::Release);
        spawned
    }
}

/// Process-global retirement runtime initialized after `cuInit` has loaded `libcuda`.
static GLOBAL_RUNTIME: RetirementRuntime = RetirementRuntime::new();

/// C `atexit` callback registered after `libcuda` and Compute Sanitizer
/// initialization; LIFO ordering runs this before their exit destructors.
extern "C" fn on_process_exit() {
    GLOBAL_RUNTIME.shutdown_for_exit();
}

/// Enqueues `pending` onto the process-global retirement worker, spawning the
/// worker and registering `on_process_exit` on first use.
fn enqueue(pending: Pending) {
    GLOBAL_RUNTIME.enqueue(pending);
}

/// Schedules `cleanup` on the retirement worker without waiting on the caller thread.
pub(crate) fn defer(cleanup: impl FnOnce() + Send + 'static) {
    enqueue(Pending {
        readiness: Readiness::Immediate,
        cleanup: Box::new(cleanup),
    });
}

/// Keeps an owner or access lease alive until work already queued on `stream`
/// completes. Dropping the caller's future cannot shorten this lifetime.
///
/// Failure to record or observe completion intentionally leaks the owner;
/// releasing it could invalidate work that the driver has already accepted.
pub fn retain_until_complete(
    owner: impl Send + 'static,
    stream: &Stream,
) -> Result<(), DriverError> {
    let event = match stream.record_event(None) {
        Ok(event) => event,
        Err(error) => {
            std::mem::forget(owner);
            return Err(error);
        }
    };
    enqueue(Pending {
        readiness: Readiness::Event(Arc::new(event)),
        cleanup: Box::new(move || drop(owner)),
    });
    Ok(())
}

/// Retains `owner` on the retirement worker until `event` reports completion.
pub(crate) fn retain_until_event(owner: impl Send + 'static, event: Arc<Event>) {
    enqueue(Pending {
        readiness: Readiness::Event(event),
        cleanup: Box::new(move || drop(owner)),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Barrier;

    /// Counts `Drop` invocations to verify whether a retired owner ran or leaked.
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn shutdown_drains_ready_cascades_and_fences_out_incomplete_or_late_items() {
        let queue = Arc::new(RetirementQueue::new());
        let ready_runs = Arc::new(AtomicUsize::new(0));
        let cascaded_runs = Arc::new(AtomicUsize::new(0));
        let incomplete_drops = Arc::new(AtomicUsize::new(0));
        let late_drops = Arc::new(AtomicUsize::new(0));

        // 1. Ready item whose cleanup enqueues a cascading immediate item
        //    (models AllocationLease drop triggering Allocation::drop -> defer).
        {
            let queue_for_cascade = queue.clone();
            let ready_runs = ready_runs.clone();
            let cascaded_runs = cascaded_runs.clone();
            queue.enqueue(Pending {
                readiness: Readiness::Probe(Box::new(|| Ok(true))),
                cleanup: Box::new(move || {
                    ready_runs.fetch_add(1, Ordering::SeqCst);
                    queue_for_cascade.enqueue(Pending {
                        readiness: Readiness::Immediate,
                        cleanup: Box::new(move || {
                            cascaded_runs.fetch_add(1, Ordering::SeqCst);
                        }),
                    });
                }),
            });
        }

        // 2. Incomplete GPU item (`Ok(false)`): must leak at exit without running cleanup/Drop.
        {
            let guard = DropCounter(incomplete_drops.clone());
            queue.enqueue(Pending {
                readiness: Readiness::Probe(Box::new(|| Ok(false))),
                cleanup: Box::new(move || drop(guard)),
            });
        }

        queue.shutdown_for_exit();
        assert_eq!(ready_runs.load(Ordering::SeqCst), 1);
        assert_eq!(cascaded_runs.load(Ordering::SeqCst), 1);
        assert_eq!(incomplete_drops.load(Ordering::SeqCst), 0);
        assert_eq!(queue.step(), None);
        assert!(!queue.wait_for_work(false));

        // 3. Post-shutdown enqueue (e.g. TLS or later atexit destructor): must leak untouched.
        {
            let guard = DropCounter(late_drops.clone());
            queue.enqueue(Pending {
                readiness: Readiness::Immediate,
                cleanup: Box::new(move || drop(guard)),
            });
        }
        assert_eq!(late_drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn shutdown_drains_in_flight_work_completed_since_last_worker_poll() {
        let queue = Arc::new(RetirementQueue::new());
        let event_completed = Arc::new(AtomicBool::new(false));
        let in_flight_runs = Arc::new(AtomicUsize::new(0));
        let in_flight_owner_drops = Arc::new(AtomicUsize::new(0));
        let in_flight_event_drops = Arc::new(AtomicUsize::new(0));
        let cascaded_runs = Arc::new(AtomicUsize::new(0));
        let incomplete_drops = Arc::new(AtomicUsize::new(0));

        {
            let event_completed = event_completed.clone();
            let event_guard = DropCounter(in_flight_event_drops.clone());
            let owner_guard = DropCounter(in_flight_owner_drops.clone());
            let queue_for_cascade = queue.clone();
            let in_flight_runs = in_flight_runs.clone();
            let cascaded_runs = cascaded_runs.clone();
            queue.enqueue(Pending {
                readiness: Readiness::Probe(Box::new(move || {
                    let _keep_event = &event_guard;
                    Ok(event_completed.load(Ordering::SeqCst))
                })),
                cleanup: Box::new(move || {
                    drop(owner_guard);
                    in_flight_runs.fetch_add(1, Ordering::SeqCst);
                    queue_for_cascade.enqueue(Pending {
                        readiness: Readiness::Immediate,
                        cleanup: Box::new(move || {
                            cascaded_runs.fetch_add(1, Ordering::SeqCst);
                        }),
                    });
                }),
            });
        }

        {
            let guard = DropCounter(incomplete_drops.clone());
            queue.enqueue(Pending {
                readiness: Readiness::Probe(Box::new(|| Ok(false))),
                cleanup: Box::new(move || drop(guard)),
            });
        }

        // Worker polls once while both items are incomplete, moving them from
        // `inbox` into `in_flight` and leaving `inbox` empty.
        assert_eq!(queue.step(), Some(true));
        assert_eq!(in_flight_runs.load(Ordering::SeqCst), 0);

        // One event completes after the worker's last poll with nothing new enqueued.
        event_completed.store(true, Ordering::SeqCst);
        queue.shutdown_for_exit();

        assert_eq!(in_flight_runs.load(Ordering::SeqCst), 1);
        assert_eq!(in_flight_owner_drops.load(Ordering::SeqCst), 1);
        assert_eq!(in_flight_event_drops.load(Ordering::SeqCst), 1);
        assert_eq!(cascaded_runs.load(Ordering::SeqCst), 1);
        assert_eq!(incomplete_drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn exit_during_initialization_prevents_worker_spawn_and_leaks_pending_item() {
        let runtime = Arc::new(RetirementRuntime::new());
        let entered_init = Arc::new(Barrier::new(2));
        let exit_finished = Arc::new(Barrier::new(2));
        let cleanup_runs = Arc::new(AtomicUsize::new(0));
        let owner_drops = Arc::new(AtomicUsize::new(0));

        let init_thread = {
            let runtime = runtime.clone();
            let entered_init = entered_init.clone();
            let exit_finished = exit_finished.clone();
            let cleanup_runs = cleanup_runs.clone();
            let owner_guard = DropCounter(owner_drops.clone());
            std::thread::spawn(move || {
                runtime.enqueue_with(
                    Pending {
                        readiness: Readiness::Immediate,
                        cleanup: Box::new(move || {
                            drop(owner_guard);
                            cleanup_runs.fetch_add(1, Ordering::SeqCst);
                        }),
                    },
                    || true,
                    || {
                        entered_init.wait();
                        exit_finished.wait();
                    },
                )
            })
        };

        entered_init.wait();
        runtime.shutdown_for_exit();
        exit_finished.wait();

        let spawned = init_thread.join().unwrap();
        assert!(!spawned, "worker must not spawn after exit callback ran");
        assert_eq!(cleanup_runs.load(Ordering::SeqCst), 0);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 0);
        assert_eq!(runtime.queue().step(), None);
        assert!(!runtime.queue().wait_for_work(false));

        let late_drops = Arc::new(AtomicUsize::new(0));
        let late_guard = DropCounter(late_drops.clone());
        let late_spawned = runtime.enqueue_with(
            Pending {
                readiness: Readiness::Immediate,
                cleanup: Box::new(move || drop(late_guard)),
            },
            || unreachable!("atexit must not re-register after exit"),
            || unreachable!("pre-spawn hook must not run after exit"),
        );
        assert!(!late_spawned);
        assert_eq!(late_drops.load(Ordering::SeqCst), 0);
    }
}
