/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hold stream work until the test explicitly releases it, including on unwind.
//! The watchdog prevents a broken synchronization path from hanging the suite;
//! its expiry fails the test and is never evidence of correct ordering.

use cuda_core::Stream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Duration;

pub struct StreamGate {
    release: mpsc::Sender<()>,
    expired: Arc<AtomicBool>,
    watchdog: Option<JoinHandle<()>>,
}

impl StreamGate {
    pub fn arm(stream: &Stream) -> Self {
        let (open, wait) = mpsc::channel();
        let (release, stop) = mpsc::channel();
        let expired = Arc::new(AtomicBool::new(false));
        let flag = expired.clone();
        let watchdog = std::thread::spawn(move || {
            if stop.recv_timeout(Duration::from_secs(60)).is_err() {
                flag.store(true, Ordering::SeqCst);
            }
            let _ = open.send(());
        });
        let gate = Self {
            release,
            expired,
            watchdog: Some(watchdog),
        };
        stream
            .launch_host_function(move || {
                let _ = wait.recv();
            })
            .unwrap();
        gate
    }

    pub fn assert_held(&self) {
        assert!(
            !self.expired.load(Ordering::SeqCst),
            "stream gate watchdog expired before the operation returned"
        );
    }
}

impl Drop for StreamGate {
    fn drop(&mut self) {
        let _ = self.release.send(());
        let joined = self.watchdog.take().unwrap().join();
        if !std::thread::panicking() {
            joined.expect("gate watchdog panicked");
            self.assert_held();
        }
    }
}
