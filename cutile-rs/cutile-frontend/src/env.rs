/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Environment switches read by the compiler and the host launcher.
//!
//! Kept out of [`crate::hints`], which is option and hint TYPES and reads nothing.

use std::env;
use std::sync::OnceLock;

/// `CUTILE_JIT_LOG` (`1`/`true`/`yes`/`on`, like every other on/off switch
/// of the crate and of `cutile`) also reports every bounds check that stays
/// inside a loop body with the reason it could not hoist.
pub fn jit_hoist_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("CUTILE_JIT_LOG"))
}

/// Returns whether the environment variable `var` is set to a truthy value
/// (`1` / `true` / `yes` / `on`, case-insensitive, surrounding whitespace ignored).
///
/// Shared by the crate's on/off diagnostic env vars so they all parse the same way.
pub fn env_flag_enabled(var: &str) -> bool {
    env::var(var).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}
