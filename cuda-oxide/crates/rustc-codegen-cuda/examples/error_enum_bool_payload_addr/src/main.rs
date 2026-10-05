/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a MUTABLE borrow of a `bool` enum payload must fail closed.
//!
//! A bool payload is semantically `i1` but its enum storage byte is a
//! canonical `i8` (the value paths zero-extend and truncate exactly at the
//! construct/extract boundary). A raw payload address escapes that boundary:
//! an `i1` store through it would leave the byte's upper seven bits undefined
//! for every `i8` reader, including a niche tag sharing the byte. Shared
//! borrows of such payloads compile through a sound value copy (see the
//! `shared_borrow_bool_payload` kernel in `enum_payload_addr`), and a write
//! that stays inside its own function compiles by rebuilding the enum around
//! the new payload (see `mutate_bool_payload` there).
//!
//! Neither route reaches the borrow below. It crosses a `#[device]` call, so
//! the address genuinely escapes and no rewrite at the store site can stand in
//! for it. A copy would lose the write, so the compiler must reject it loudly
//! instead of miscompiling.

use cuda_device::{device, kernel};

pub enum Flag {
    On(bool),
    Off,
}

/// Flip a borrowed bool. Taking `&mut bool` across a call boundary keeps
/// the borrow from folding into a direct store.
#[device]
pub fn flip_in_place(value: &mut bool) {
    *value = !*value;
}

/// # Safety
///
/// `out` must point to writable device memory for one `u32`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn mutate_bool_payload(out: *mut u32, seed: u32) {
    let mut flag = if seed & 1 == 0 {
        Flag::On(seed & 2 != 0)
    } else {
        Flag::Off
    };
    if let Flag::On(value) = &mut flag {
        flip_in_place(value);
    }
    let result = match flag {
        Flag::On(true) => 2,
        Flag::On(false) => 1,
        Flag::Off => 0,
    };
    unsafe {
        *out = result;
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
