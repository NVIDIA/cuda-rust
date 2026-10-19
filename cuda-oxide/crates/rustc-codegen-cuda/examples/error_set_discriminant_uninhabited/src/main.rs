/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: SetDiscriminant must not manufacture a value of an
//! uninhabited enum variant.

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use cuda_device::kernel;

enum Never {}

#[repr(u8)]
#[allow(dead_code)]
enum State {
    First = 1,
    Impossible(Never) = 3,
    Last = 7,
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn select_impossible(state: &mut State) {
    mir!({
        SetDiscriminant(*state, 1);
        Return()
    })
}

/// Attempts the intentionally invalid uninhabited discriminant write.
///
/// # Safety
///
/// If `value` is non-null, it must point to a valid, writable `u8` for the
/// duration of the call.
#[kernel]
pub unsafe fn set_discriminant_uninhabited(value: *mut u8) {
    if !value.is_null() {
        let mut state = State::First;
        select_impossible(&mut state);
        unsafe {
            value.write(matches!(state, State::Last) as u8);
        }
    }
}

fn main() {
    println!("This build must fail: SetDiscriminant cannot select an uninhabited variant.");
}
