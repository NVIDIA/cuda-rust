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

/// Reports a leak to stderr.
pub(crate) fn report_leak(message: std::fmt::Arguments<'_>) {
    let mut stderr = io::stderr().lock();
    let _ = writeln!(stderr, "{message}");
}
