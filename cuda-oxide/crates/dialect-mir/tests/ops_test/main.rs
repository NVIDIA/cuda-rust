/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests build kinded fixture types directly; production minting lives in
// mir-importer's facts.rs (see the workspace clippy.toml disallowed-methods).
#![allow(clippy::disallowed_methods)]

mod aggregates;
mod cast_authority;
mod control_flow;
mod enums;
mod memory;
mod pointer_kinds;
mod scalars_and_funcs;
