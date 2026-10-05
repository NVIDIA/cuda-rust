/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Command implementations for cargo-oxide.
//!
//! These port the xtask commands with improvements:
//! - Backend path resolved via discovery chain instead of hardcoded relative path
//! - Workspace root resolved by walking up from CWD instead of assuming CWD

mod artifacts;
mod build_run;
mod clean;
mod codegen_env;
mod context;
mod doctor;
mod examples_list;
mod fingerprint;
mod fmt;
mod host_cargo;
mod interop;
mod ltoir;
mod materialize;
mod passthrough;
mod pipeline_debug;
mod scaffold;
mod setup_update;
#[cfg(test)]
mod tests;
mod toolkit;

use artifacts::*;
pub use build_run::*;
pub use clean::*;
pub use codegen_env::*;
pub use context::*;
pub use doctor::*;
pub use examples_list::*;
use fingerprint::*;
pub use fmt::*;
use host_cargo::*;
use interop::*;
pub use ltoir::*;
pub use materialize::*;
pub use passthrough::*;
pub use pipeline_debug::*;
pub use scaffold::*;
pub use setup_update::*;
use toolkit::*;
