/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Loud-leak reporting.
//!
//! When a completion wait cannot prove the device is done with a result's
//! memory, the result is leaked (dropping buffers in-flight work may still
//! touch is worse) and the leak is reported here. Every report site in the
//! crate goes through [`report_leak`] so unit tests can capture and assert
//! the message instead of spraying stderr from passing tests.

use std::io::{self, Write};

/// Reports a leak to stderr — or to the active test capture, if one is
/// installed.
pub(crate) fn report_leak(message: std::fmt::Arguments<'_>) {
    #[cfg(test)]
    if capture::push(&message) {
        return;
    }
    let mut stderr = io::stderr().lock();
    let _ = writeln!(stderr, "{message}");
}

/// Test-only capture of leak reports, so tests that intentionally exercise
/// the loud-leak paths can assert on the message instead of printing it.
#[cfg(test)]
pub(crate) mod capture {
    use std::sync::{Mutex, MutexGuard};

    static ACTIVE: Mutex<()> = Mutex::new(());
    static CAPTURED: Mutex<Option<Vec<String>>> = Mutex::new(None);

    /// Serializes capturing tests and scopes the capture window; captured
    /// messages are retrieved (and the window closed) by [`Capture::take`].
    pub(crate) struct Capture {
        _serial: MutexGuard<'static, ()>,
    }

    pub(crate) fn start() -> Capture {
        let serial = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        *CAPTURED.lock().unwrap() = Some(Vec::new());
        Capture { _serial: serial }
    }

    impl Capture {
        pub(crate) fn take(&mut self) -> Vec<String> {
            CAPTURED.lock().unwrap().take().unwrap_or_default()
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            let _ = CAPTURED.lock().map(|mut c| c.take());
        }
    }

    pub(crate) fn push(message: &std::fmt::Arguments<'_>) -> bool {
        let mut captured = CAPTURED.lock().unwrap();
        match captured.as_mut() {
            Some(messages) => {
                messages.push(message.to_string());
                true
            }
            None => false,
        }
    }
}
