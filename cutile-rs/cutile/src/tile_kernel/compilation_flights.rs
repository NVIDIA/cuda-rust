/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Share CPU compilation between preparation workers and ordinary launches.
//! This registry owns no CUDA objects and never holds a map lock while compiling
//! or waiting. Entries live only while a caller retains their artifact/flight.

use crate::error::Error;
use cutile_compiler::compile_control::{CompileControl, CompileInterrupted};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::ThreadId;
use std::time::Duration;

enum State<V> {
    Running(ThreadId),
    Ready(Arc<V>),
    Interrupted,
    Failed(String),
}

struct Flight<V> {
    state: Mutex<State<V>>,
    changed: Condvar,
}

pub(super) struct Registry<K, V> {
    flights: Mutex<HashMap<K, Weak<Flight<V>>>>,
    #[cfg(test)]
    waiting: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

impl<K, V> Default for Registry<K, V> {
    fn default() -> Self {
        Self {
            flights: Mutex::new(HashMap::new()),
            #[cfg(test)]
            waiting: Mutex::new(None),
        }
    }
}

struct Lease<'a, K: Eq + Hash, V> {
    key: K,
    flight: Option<Arc<Flight<V>>>,
    registry: &'a Registry<K, V>,
}

impl<K: Eq + Hash, V> Lease<'_, K, V> {
    fn flight(&self) -> &Arc<Flight<V>> {
        self.flight.as_ref().expect("lease is live until drop")
    }
    fn unregister(&self) {
        let mut flights = self.registry.flights.lock().unwrap();
        if flights
            .get(&self.key)
            .is_some_and(|old| old.as_ptr() == Arc::as_ptr(self.flight()))
        {
            flights.remove(&self.key);
        }
    }
}

impl<K: Eq + Hash, V> Drop for Lease<'_, K, V> {
    fn drop(&mut self) {
        // Both the count check and strong-reference release must happen under
        // the map lock. Otherwise two releasers can both see count == 2 and
        // leave a dead weak entry when their fields drop after unlocking.
        let mut flights = self.registry.flights.lock().unwrap();
        let flight = self.flight.take().unwrap();
        if Arc::strong_count(&flight) == 1
            && flights
                .get(&self.key)
                .is_some_and(|old| old.as_ptr() == Arc::as_ptr(&flight))
        {
            flights.remove(&self.key);
        }
        drop(flight);
    }
}

/// Holding this keeps CPU code available to a launch until install or drop.
pub(super) struct Artifact<'a, K: Eq + Hash, V> {
    value: Arc<V>,
    _lease: Lease<'a, K, V>,
}

impl<K: Eq + Hash, V> std::ops::Deref for Artifact<'_, K, V> {
    type Target = V;
    fn deref(&self) -> &V {
        &self.value
    }
}

struct OwnerGuard<'a, 'r, K: Eq + Hash, V> {
    lease: &'a Lease<'r, K, V>,
    finished: bool,
}

impl<K: Eq + Hash, V> Drop for OwnerGuard<'_, '_, K, V> {
    fn drop(&mut self) {
        if !self.finished {
            // Unwinding must release waiters too. Each still-live caller can
            // retry with its own compiler closure and cancellation control.
            *self.lease.flight().state.lock().unwrap() = State::Interrupted;
            self.lease.unregister();
            self.lease.flight().changed.notify_all();
        }
    }
}

fn check(control: Option<&CompileControl>) -> Result<(), Error> {
    if let Some(control) = control {
        control.check().map_err(anyhow::Error::from)?;
    }
    Ok(())
}

fn interrupted(error: &Error) -> bool {
    let inner = match error {
        Error::Anyhow(error) => Some(error),
        Error::JIT(cutile_compiler::error::JITError::Anyhow(error)) => Some(error),
        _ => None,
    };
    inner.is_some_and(|e| e.downcast_ref::<CompileInterrupted>().is_some())
}

impl<K: Eq + Hash + Clone, V> Registry<K, V> {
    pub(super) fn compile(
        &self,
        key: K,
        control: Option<&CompileControl>,
        build: impl FnOnce() -> Result<V, Error>,
    ) -> Result<Artifact<'_, K, V>, Error> {
        let mut build = Some(build);
        'retry: loop {
            check(control)?;
            let (flight, owner) = {
                let mut flights = self.flights.lock().unwrap();
                match flights.get(&key).and_then(Weak::upgrade) {
                    Some(flight) => (flight, false),
                    None => {
                        let flight = Arc::new(Flight {
                            state: Mutex::new(State::Running(std::thread::current().id())),
                            changed: Condvar::new(),
                        });
                        flights.insert(key.clone(), Arc::downgrade(&flight));
                        (flight, true)
                    }
                }
            };
            let lease = Lease {
                key: key.clone(),
                flight: Some(flight),
                registry: self,
            };
            if owner {
                let mut guard = OwnerGuard {
                    lease: &lease,
                    finished: false,
                };
                let result = build.take().unwrap()().and_then(|value| {
                    check(control)?;
                    Ok(Arc::new(value))
                });
                *lease.flight().state.lock().unwrap() = match &result {
                    Ok(value) => State::Ready(Arc::clone(value)),
                    Err(error) if interrupted(error) => State::Interrupted,
                    Err(error) => State::Failed(error.to_string()),
                };
                if result.is_err() {
                    lease.unregister();
                }
                guard.finished = true;
                drop(guard);
                lease.flight().changed.notify_all();
                return result.map(|value| Artifact {
                    value,
                    _lease: lease,
                });
            }
            let mut state = lease.flight().state.lock().unwrap();
            #[cfg(test)]
            let mut announced = false;
            loop {
                check(control)?;
                match &*state {
                    State::Ready(value) => {
                        let value = Arc::clone(value);
                        drop(state);
                        return Ok(Artifact {
                            value,
                            _lease: lease,
                        });
                    }
                    State::Interrupted => {
                        drop(state);
                        lease.unregister();
                        continue 'retry;
                    }
                    State::Failed(message) => return Err(anyhow::anyhow!(message.clone()).into()),
                    State::Running(owner) if *owner == std::thread::current().id() => {
                        return Err(anyhow::anyhow!(
                            "recursive compilation of the same specialization"
                        )
                        .into());
                    }
                    State::Running(_) => {}
                }
                #[cfg(test)]
                if !announced {
                    if let Some(sender) = self.waiting.lock().unwrap().as_ref() {
                        let _ = sender.send(());
                    }
                    announced = true;
                }
                state = lease
                    .flight()
                    .changed
                    .wait_timeout(state, Duration::from_millis(10))
                    .unwrap()
                    .0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn recv<T>(rx: &mpsc::Receiver<T>) -> T {
        rx.recv_timeout(Duration::from_secs(30))
            .expect("deadlock watchdog")
    }

    #[test]
    fn concurrent_callers_share_one_result_and_drop_unregisters_it() {
        let registry = Registry::<u32, usize>::default();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        *registry.waiting.lock().unwrap() = Some(waiting_tx);
        let (started_tx, started_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let registry = &registry;
            let owner = scope.spawn(move || {
                registry
                    .compile(7, None, || {
                        started_tx.send(()).unwrap();
                        recv(&waiting_rx);
                        Ok(42)
                    })
                    .unwrap()
            });
            recv(&started_rx);
            let waiter = registry
                .compile(7, None, || panic!("duplicate compilation"))
                .unwrap();
            let owner = owner.join().unwrap();
            assert!(Arc::ptr_eq(&owner.value, &waiter.value));
            assert_eq!(*waiter, 42);
            assert_eq!(registry.flights.lock().unwrap().len(), 1);
        });
        assert!(registry.flights.lock().unwrap().is_empty());
    }

    #[test]
    fn simultaneous_last_holders_leave_no_stale_entries() {
        let registry = Registry::<u32, usize>::default();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let registry = &registry;
            let barrier = &barrier;
            let mut holders = Vec::new();
            for key in 0..256 {
                let a = registry.compile(key, None, || Ok(key as usize)).unwrap();
                let b = registry
                    .compile(key, None, || panic!("must share"))
                    .unwrap();
                holders.push((a, b));
            }
            let (a, b): (Vec<_>, Vec<_>) = holders.into_iter().unzip();
            let worker = scope.spawn(move || {
                for value in a {
                    barrier.wait();
                    drop(value);
                    barrier.wait();
                }
            });
            for value in b {
                barrier.wait();
                drop(value);
                barrier.wait();
            }
            worker.join().unwrap();
        });
        assert!(registry.flights.lock().unwrap().is_empty());
    }

    #[test]
    fn cancelled_waiter_does_not_cancel_owner() {
        let registry = Registry::<u32, usize>::default();
        let waiter_control = CompileControl::default();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        *registry.waiting.lock().unwrap() = Some(waiting_tx);
        let (started_tx, started_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let registry = &registry;
            let waiter_control = &waiter_control;
            let owner = scope.spawn(move || {
                registry
                    .compile(7, None, || {
                        started_tx.send(()).unwrap();
                        recv(&waiting_rx);
                        waiter_control.cancel();
                        Ok(42)
                    })
                    .unwrap()
            });
            recv(&started_rx);
            let error = registry
                .compile(7, Some(waiter_control), || panic!("duplicate compilation"))
                .err()
                .unwrap();
            assert!(interrupted(&error));
            assert_eq!(*owner.join().unwrap(), 42);
        });
    }

    #[test]
    fn interrupted_owner_is_retried_by_a_live_waiter() {
        let registry = Registry::<u32, usize>::default();
        let owner_control = CompileControl::default();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        *registry.waiting.lock().unwrap() = Some(waiting_tx);
        let (started_tx, started_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let registry = &registry;
            let owner = scope.spawn(move || {
                registry
                    .compile(7, Some(&owner_control), || {
                        started_tx.send(()).unwrap();
                        recv(&waiting_rx);
                        owner_control.cancel();
                        Ok(1)
                    })
                    .err()
                    .unwrap()
            });
            recv(&started_rx);
            let result = registry.compile(7, None, || Ok(42)).unwrap();
            assert!(interrupted(&owner.join().unwrap()));
            assert_eq!(*result, 42);
        });
        assert!(registry.flights.lock().unwrap().is_empty());
    }

    #[test]
    fn compiler_error_does_not_poison_future_attempts() {
        let registry = Registry::<u32, usize>::default();
        assert!(registry
            .compile(7, None, || Err(anyhow::anyhow!("failed").into()))
            .is_err());
        assert!(registry.flights.lock().unwrap().is_empty());
        assert_eq!(*registry.compile(7, None, || Ok(42)).unwrap(), 42);
        assert!(registry.flights.lock().unwrap().is_empty());
    }

    #[test]
    fn recursive_same_key_is_an_error_instead_of_a_deadlock() {
        let registry = Registry::<u32, usize>::default();
        let error = registry
            .compile(7, None, || {
                registry.compile(7, None, || Ok(1)).map(|value| *value)
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains("recursive compilation"));
        assert!(registry.flights.lock().unwrap().is_empty());
    }

    #[test]
    fn panic_releases_waiters_and_allows_retry() {
        let registry = Registry::<u32, usize>::default();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        *registry.waiting.lock().unwrap() = Some(waiting_tx);
        let (started_tx, started_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let registry = &registry;
            let owner = scope.spawn(move || {
                registry.compile(7, None, || {
                    started_tx.send(()).unwrap();
                    recv(&waiting_rx);
                    panic!("compiler panic");
                })
            });
            recv(&started_rx);
            assert_eq!(*registry.compile(7, None, || Ok(42)).unwrap(), 42);
            assert!(owner.join().is_err());
        });
    }
}
