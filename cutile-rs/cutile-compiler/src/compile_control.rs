/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative cancellation and monotonic deadlines for compilation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A shared cancellation flag and an optional absolute, monotonic deadline.
///
/// Checks are cooperative during frontend compilation. Controlled assembler
/// subprocesses are killed and reaped when interrupted. This cannot interrupt
/// arbitrary Rust callbacks, a CUDA API call, or an already-running GPU kernel.
#[derive(Clone, Debug, Default)]
pub struct CompileControl {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

/// Why work was interrupted. Neither outcome establishes kernel invalidity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileInterrupted {
    Cancelled,
    DeadlineExceeded,
}

impl std::fmt::Display for CompileInterrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Cancelled => "work cancelled",
            Self::DeadlineExceeded => "work deadline exceeded",
        })
    }
}

impl std::error::Error for CompileInterrupted {}

impl CompileControl {
    /// Sets a deadline without extending an existing earlier deadline.
    pub fn with_deadline(&self, deadline: Instant) -> Self {
        Self {
            cancelled: Arc::clone(&self.cancelled),
            deadline: Some(self.deadline.map_or(deadline, |old| old.min(deadline))),
        }
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Cancels this control and every clone/child sharing its flag.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn check(&self) -> Result<(), CompileInterrupted> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(CompileInterrupted::Cancelled)
        } else if self.deadline.is_some_and(|end| Instant::now() >= end) {
            Err(CompileInterrupted::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_share_cancellation_but_cannot_extend_deadline() {
        let root = CompileControl::default();
        let end = Instant::now();
        let child = root.with_deadline(end);
        assert_eq!(child.check(), Err(CompileInterrupted::DeadlineExceeded));
        assert_eq!(root.check(), Ok(()));
        assert_eq!(
            child
                .with_deadline(end + std::time::Duration::from_secs(1))
                .deadline(),
            Some(end)
        );
        root.cancel();
        assert_eq!(child.check(), Err(CompileInterrupted::Cancelled));
    }
}
