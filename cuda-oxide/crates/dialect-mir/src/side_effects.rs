/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Marks the pure `dialect-mir` ops as having no side effects.
//!
//! pliron's `dce` only deletes an op whose result is unused if the op tells it
//! it has no side effects (via the [`SideEffects`] interface). Without that, dce
//! conservatively assumes every op might have side effects and keeps it. So a
//! value-only op left dead by a transform (for example the `x & mask` that the
//! unroller folds to a constant, or an induction-variable increment that becomes
//! unused after full unroll) would survive our own `dce` and only get cleaned up
//! later by LLVM's `opt`.
//!
//! Marking the pure arithmetic / comparison / constant / cast ops here lets our
//! middle-end `dce` remove them itself, so the IR we hand to lowering is already
//! clean (and future passes like sccp can rely on the same information).
//!
//! Only ops that compute a value with no memory, control-flow, or other
//! observable effect are listed. Memory ops (load/store/alloca/memcpy), calls,
//! asserts, debug, storage markers, and terminators are deliberately left out:
//! the safe default (no `SideEffects` impl) keeps them.

use pliron::context::Context;
use pliron::derive::op_interface_impl;
use pliron::opts::dce::SideEffects;

use crate::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCastOp, MirCheckedAddOp, MirCheckedMulOp,
    MirCheckedSubOp, MirCmpOp, MirConstantOp, MirDivOp, MirEqOp, MirFloatConstantOp, MirGeOp,
    MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirRemOp, MirShlOp, MirShrOp,
    MirSubOp, MirUndefOp,
};

/// Implement [`SideEffects`] returning `false` for each listed op.
macro_rules! pure_ops {
    ($($op:ty),+ $(,)?) => {$(
        #[op_interface_impl]
        impl SideEffects for $op {
            fn has_side_effects(&self, _ctx: &Context) -> bool {
                false
            }
        }
    )+};
}

pure_ops!(
    // Integer arithmetic (wrapping and overflow-checked), bitwise, and shifts.
    // Integer div/rem are included: on the GPU target a zero divisor does not
    // trap, and removing a *dead* div/rem is a sound refinement either way.
    MirAddOp,
    MirSubOp,
    MirMulOp,
    MirDivOp,
    MirRemOp,
    MirCheckedAddOp,
    MirCheckedSubOp,
    MirCheckedMulOp,
    MirNegOp,
    MirNotOp,
    MirShlOp,
    MirShrOp,
    MirBitAndOp,
    MirBitOrOp,
    MirBitXorOp,
    // Comparisons.
    MirLtOp,
    MirLeOp,
    MirGtOp,
    MirGeOp,
    MirEqOp,
    MirNeOp,
    MirCmpOp,
    // Constants and casts.
    MirConstantOp,
    MirFloatConstantOp,
    MirUndefOp,
    MirCastOp,
);
