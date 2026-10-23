/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owners held independently of an operation's (possibly borrowed) output.

use crate::device_future::{probe_stream, StreamHealth};
use crate::device_operation::ReplayResource;
use crate::error::DeviceError;
use cuda_core::Stream;
use std::sync::{Arc, Mutex};

/// Storage whose per-access bookkeeping is released when the submission
/// that holds the access completes. Implemented by tensor storage.
#[doc(hidden)]
pub trait AccessTracked: Send + Sync {
    /// Forget the in-flight access registered under `id`.
    fn release_access(&self, id: u64);
}

/// One in-flight access to an [`AccessTracked`] owner: keeps the owner alive
/// until the submission completes, then releases the access.
///
/// Stored inline in the submission (no per-lease box), so retaining a
/// tensor argument costs a refcount and a `Vec` push, not an allocation.
#[doc(hidden)]
pub struct AccessLease {
    owner: Arc<dyn AccessTracked>,
    id: u64,
}

impl AccessLease {
    pub fn new(owner: Arc<dyn AccessTracked>, id: u64) -> Self {
        Self { owner, id }
    }
}

impl Drop for AccessLease {
    fn drop(&mut self) {
        self.owner.release_access(self.id);
    }
}

/// Everything a submission keeps alive: access leases on the fast path, and
/// arbitrary owners (graph execs, host buffers) boxed on the slow path.
#[derive(Default)]
struct OwnerSet {
    leases: Vec<AccessLease>,
    others: Vec<Box<dyn Send>>,
}

impl OwnerSet {
    fn is_empty(&self) -> bool {
        self.leases.is_empty() && self.others.is_empty()
    }
}

#[derive(Default)]
struct Owners(Option<OwnerSet>);

impl Owners {
    fn set(&mut self) -> Result<&mut OwnerSet, DeviceError> {
        self.0
            .as_mut()
            .ok_or_else(|| DeviceError::Internal("submission already completed".into()))
    }

    fn retain(&mut self, owner: impl Send + 'static) -> Result<(), DeviceError> {
        self.set()?.others.push(Box::new(owner));
        Ok(())
    }

    fn retain_lease(&mut self, lease: AccessLease) -> Result<(), DeviceError> {
        self.set()?.leases.push(lease);
        Ok(())
    }

    fn complete(&mut self) {
        self.0.take();
    }
}

pub(crate) struct Submission {
    stream: Arc<Stream>,
    owners: Mutex<Owners>,
    recorded: Mutex<Vec<Arc<dyn ReplayResource>>>,
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

impl Submission {
    pub(crate) fn new(stream: Arc<Stream>) -> Self {
        Self {
            stream,
            owners: Mutex::new(Owners(Some(OwnerSet::default()))),
            recorded: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn retain(&self, owner: impl Send + 'static) -> Result<(), DeviceError> {
        self.owners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(owner)
    }

    pub(crate) fn retain_lease(&self, lease: AccessLease) -> Result<(), DeviceError> {
        self.owners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain_lease(lease)
    }

    pub(crate) fn record(&self, resource: Arc<dyn ReplayResource>) {
        self.recorded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(resource);
    }

    /// Hands the recorded resources to the graph that owns them; the graph
    /// reacquires each on every replay.
    pub(crate) fn take_recorded(&self) -> Vec<Arc<dyn ReplayResource>> {
        std::mem::take(&mut *self.recorded.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Releases tracking; no further work may use this submission.
    ///
    /// # Safety
    /// GPU work must be complete, or the caller must independently keep all
    /// resources alive and prevent conflicting accesses until completion.
    pub(crate) unsafe fn complete(&self) {
        self.owners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .complete();
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        let Some(owners) = self
            .owners
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .0
            .take()
        else {
            return;
        };
        if owners.is_empty() {
            return;
        }
        let stream = self.stream.clone();
        let health = match stream.context().bind_to_thread() {
            Ok(()) => probe_stream(&stream),
            Err(error) => StreamHealth::Faulted(error),
        };
        match health {
            StreamHealth::Idle => drop(owners),
            StreamHealth::Busy => {
                // CUDA events retain context; explicitly retain the stream too,
                // so access tracker stream identities remain valid until release.
                let _ =
                    cuda_core::completion::retain_until_complete((owners, stream.clone()), &stream);
            }
            StreamHealth::Faulted(_) | StreamHealth::Capturing => {
                // Completion cannot be established. Preserve both allocations
                // and exclusion instead of freeing storage or allowing a race.
                std::mem::forget((owners, stream));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owners_outlive_recovered_or_projected_results() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let mut submission = Owners(Some(OwnerSet::default()));
        submission.retain(owner).unwrap();
        assert!(weak.upgrade().is_some());
        submission.complete();
        assert!(weak.upgrade().is_none());
        assert!(submission.retain(()).is_err());
    }

    #[test]
    fn forgotten_submission_keeps_owners() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let mut submission = Owners(Some(OwnerSet::default()));
        submission.retain(owner).unwrap();
        std::mem::forget(submission);
        assert!(weak.upgrade().is_some());
    }
}
