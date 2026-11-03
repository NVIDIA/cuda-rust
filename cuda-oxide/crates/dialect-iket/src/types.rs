/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Types belonging to the IKET dialect.

use pliron::context::Context;
use pliron::derive::pliron_type;
use pliron::r#type::Type;

/// Linear SSA token connecting `iket.range_start` to `iket.range_end`.
#[pliron_type(
    name = "iket.range_token",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct IketRangeTokenType;

pub fn register(ctx: &mut Context) {
    IketRangeTokenType::register(ctx);
}
