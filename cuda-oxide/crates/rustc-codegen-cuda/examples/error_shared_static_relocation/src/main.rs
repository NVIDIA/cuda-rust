/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a type-erased pointer to a `SharedArray` static stored
//! inside a constant.
//!
//! A `SharedArray` static is per-block shared memory, allocated by the device
//! with no initializer. A direct `&raw const TILE` operand reaches that shared
//! allocation. A pointer to `TILE` stored inside a constant is instead a
//! relocation in the constant's read-only device global, and a global
//! initializer can only point at storage that has its own initialized global.
//! Treating `TILE` as an ordinary static there would give it a second,
//! zero-byte `addrspace(1)` global and hand out an address that is not the
//! block's shared memory.
//!
//! The stored pointer is cast to `*const u32`, so its type no longer names
//! shared memory; the importer must still refuse the relocation at compile
//! time, from the static it points at.
//!
//! Usage:
//!   cargo oxide build error_shared_static_relocation
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! points at shared-memory static
//! ```

use cuda_device::{SharedArray, device, kernel};

/// The shared-memory tile the constant below points at.
type Tile = SharedArray<u32, 4>;

/// Per-block shared memory.
static mut TILE: Tile = Tile::UNINIT;

/// A constant that stores a type-erased pointer to `TILE` in its own
/// allocation.
const INDIRECT: &*const u32 = &(&raw const TILE as *const u32);

/// Reads the pointer stored in `INDIRECT`.
#[device]
#[inline(never)]
pub fn tile_address() -> *const u32 {
    // the stored pointer would be a relocation to shared memory, which has no
    // initialized global to point at, so the importer must refuse it
    *INDIRECT
}

/// # Safety
///
/// Never launched: the device compilation is expected to fail before a host
/// binary that could launch it exists.
#[kernel]
pub unsafe fn shared_relocation_kernel(output: *mut usize) {
    unsafe { output.write(tile_address() as usize) };
}

/// Host entry point; never reached because device compilation fails first.
fn main() {
    println!("This negative example should fail during device compilation.");
}
