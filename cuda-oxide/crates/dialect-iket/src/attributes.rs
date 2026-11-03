/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes belonging to the IKET dialect.

use pliron::attribute::Attribute;
use pliron::context::Context;
use pliron::derive::pliron_attr;

/// Scalar payload representation preserved by an IKET event operation.
///
/// The attribute records source signedness and width even when lowering packs
/// the value into a 32-bit or 64-bit event record.
#[pliron_attr(name = "iket.payload_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum IketPayloadKindAttr {
    None,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Pointer,
}

impl IketPayloadKindAttr {
    pub const fn has_payload(self) -> bool {
        !matches!(self, Self::None)
    }
}

pub fn register(ctx: &mut Context) {
    IketPayloadKindAttr::register(ctx);
}
