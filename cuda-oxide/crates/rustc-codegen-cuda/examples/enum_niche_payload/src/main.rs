/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reproducer for enum payload leaves that round-trip through local memory.
//!
//! An enum whose niche carrier shares its bytes with an aggregate payload
//! used to leave the payload's non-pointer leaves in `[N x i8]` filler:
//! construction spilled the value byte by byte, payload reads loaded it
//! back as one wide value, and SROA cannot reassemble mixed-type slices —
//! so the spill slot survived `opt -O2` as a `.local` depot. The leaf-slot
//! rule gives every payload leaf its own typed slot, and construction and
//! payload reads stay in SSA.
//!
//! Three shapes, one kernel each:
//!
//! - `smooth_triplets`: `Option<(f32, f32, &mut f32)>` driven by `while let`
//!   (the rms_norm `next_cell` shape). The depot reproducer: 16 B in the
//!   entry plus 16 B in the outlined `Cells::next_cell` on the parent,
//!   none with the fix.
//! - `lease_borrow`: `Option<Lease>` where `Lease` holds `&mut` to a
//!   kernel-local struct (the rope `own` shape).
//! - `lease_raw`: the same lease copying the raw pointer it needs and
//!   holding only a `PhantomData` borrow.
//!
//! Both lease shapes are layout-neutral value checks: neither emits a
//! `.local` on the parent either, so they do not measure this fix.
//!
//! Usage:
//!   cargo oxide run enum_niche_payload
//!   cargo oxide build enum_niche_payload --verbose
//!   .../enum_niche_payload/target/release/enum_niche_payload --verify-ptx

use core::marker::PhantomData;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const TRIPLETS: usize = 4;

/// (a) A lending cursor over non-overlapping triples: each `next_cell`
/// returns the first two elements by value and the third by borrow, all in
/// one `Option<(f32, f32, &mut f32)>` whose niche is the borrow.
struct Cells<'a> {
    data: &'a mut [f32],
}

impl<'a> Cells<'a> {
    fn next_cell(&mut self) -> Option<(f32, f32, &mut f32)> {
        let rest = core::mem::take(&mut self.data);
        let (head, tail) = rest.split_at_mut_checked(3)?;
        let (prev, mid) = head.split_at_mut_checked(1)?;
        let (cur, cell) = mid.split_at_mut_checked(1)?;
        self.data = tail;
        Some((prev[0], cur[0], &mut cell[0]))
    }
}

/// The kernel-local state leased out by `own`/`own_raw`.
struct Stepper {
    step: usize,
    stride: usize,
}

/// (b) The pair borrows the stepper: the enum payload carries a pointer to
/// a kernel-local struct, so its address rides through every `None`/`Some`
/// merge and the struct's storage escapes into the depot.
struct Lease<'r> {
    src: &'r mut Stepper,
    base: usize,
}

/// (c) The pair copies the raw pointer it needs and holds only a
/// `PhantomData` borrow: same bytes for the enum (`{ptr, usize}`), but no
/// reference crosses the merge, so nothing forces the payload through
/// memory.
struct LeaseRaw<'r> {
    src: *mut Stepper,
    base: usize,
    _borrow: PhantomData<&'r mut Stepper>,
}

impl Stepper {
    fn own(&mut self, thread: u32) -> Option<Lease<'_>> {
        if thread % 2 != 0 {
            return None;
        }
        Some(Lease {
            src: self,
            base: thread as usize,
        })
    }

    fn own_raw(&mut self, thread: u32) -> Option<LeaseRaw<'_>> {
        if thread % 2 != 0 {
            return None;
        }
        Some(LeaseRaw {
            src: self as *mut Stepper,
            base: thread as usize,
            _borrow: PhantomData,
        })
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// (a) `Option<(f32, f32, &mut f32)>` driven by `while let`: the float
    /// pair below the pointer niche used to live in filler bytes.
    #[kernel]
    pub fn smooth_triplets(mut values: DisjointSlice<f32>, n: usize) {
        if thread::index_1d().get() != 0 {
            return;
        }
        // SAFETY: thread 0 alone owns the complete buffer for this launch;
        // `n` is the element count the host passed for the same buffer.
        let data = unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr(), n) };
        let mut cells = Cells { data };
        let mut visits = 0usize;
        while let Some((prev, cur, cell)) = cells.next_cell() {
            *cell += prev * cur;
            visits += 1;
        }
        // SAFETY: one writer (thread 0), slot `n` reserved for the count.
        unsafe { values.as_mut_ptr().add(n).write(visits as f32) };
    }

    /// (b) `Option<Lease>` with `Lease` holding `&mut` to a kernel-local
    /// struct. A layout-neutral value check: it emits no `.local` on the
    /// parent either, so it does not measure the depot fix.
    #[kernel]
    pub fn lease_borrow(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut stepper = Stepper { step: 1, stride: 7 };
        let mut acc = 0usize;
        for i in 0..4u32 {
            if let Some(lease) = stepper.own(i) {
                lease.src.step = lease
                    .src
                    .step
                    .wrapping_mul(lease.src.stride)
                    .wrapping_add(lease.base);
                acc = acc.wrapping_add(lease.src.step);
            }
        }
        // SAFETY: one writer (thread 0); two u32 slots were allocated.
        unsafe {
            out.as_mut_ptr().write(stepper.step as u32);
            out.as_mut_ptr().add(1).write(acc as u32);
        }
    }

    /// (c) The same lease with the raw pointer copied and only a
    /// `PhantomData` borrow: `{ptr, usize}` payload, all leaves typed. A
    /// layout-neutral value check, like (b).
    #[kernel]
    pub fn lease_raw(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut stepper = Stepper { step: 1, stride: 7 };
        let mut acc = 0usize;
        for i in 0..4u32 {
            if let Some(lease) = stepper.own_raw(i) {
                // SAFETY: `src` still points at this invocation's live
                // kernel-local `stepper`, and thread 0 is the only writer.
                unsafe {
                    (*lease.src).step = (*lease.src)
                        .step
                        .wrapping_mul((*lease.src).stride)
                        .wrapping_add(lease.base);
                    acc = acc.wrapping_add((*lease.src).step);
                }
            }
        }
        // SAFETY: one writer (thread 0); two u32 slots were allocated.
        unsafe {
            out.as_mut_ptr().write(stepper.step as u32);
            out.as_mut_ptr().add(1).write(acc as u32);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_ptx();
    }

    println!("=== enum_niche_payload ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let one_thread = LaunchConfig::for_num_elems(1);

    // (a) smooth_triplets: cell[i] += prev * cur over 4 disjoint triples,
    // then the visit count in slot n (buffer holds n + 1 elements).
    let n = TRIPLETS * 3;
    let mut upload: Vec<f32> = (0..n).map(|i| i as f32).collect();
    upload.push(0.0);
    let mut buf = DeviceBuffer::from_host(&stream, &upload)?;
    // SAFETY: launch shape/resources match the kernel; `n` triples plus the
    // count slot match the buffer length the host uploaded.
    unsafe { module.smooth_triplets(&stream, one_thread, &mut buf, n) }?;
    let got = buf.to_host_vec(&stream)?;
    let mut want = upload.clone();
    let mut visits = 0usize;
    for t in 0..TRIPLETS {
        let i = t * 3;
        want[i + 2] += want[i] * want[i + 1];
        visits += 1;
    }
    for (i, (g, w)) in got[..n].iter().zip(&want).enumerate() {
        assert_eq!(g, w, "smooth_triplets: triplet output {i}");
    }
    assert_eq!(
        got[n], visits as f32,
        "smooth_triplets: visit count must survive the while-let loop"
    );

    // (b) and (c): the borrowed lease and the raw-pointer lease must agree.
    let mut expected_step = 1usize;
    let mut expected_acc = 0usize;
    for i in 0..4u32 {
        if i % 2 == 0 {
            expected_step = expected_step.wrapping_mul(7).wrapping_add(i as usize);
            expected_acc = expected_acc.wrapping_add(expected_step);
        }
    }
    for name in ["lease_borrow", "lease_raw"] {
        let zeros = vec![0u32; 2];
        let mut out = DeviceBuffer::from_host(&stream, &zeros)?;
        // SAFETY: launch shape/resources match the kernel; the buffer owns
        // exactly the two u32 slots each kernel writes.
        if name == "lease_borrow" {
            unsafe { module.lease_borrow(&stream, one_thread, &mut out) }?;
        } else {
            unsafe { module.lease_raw(&stream, one_thread, &mut out) }?;
        }
        let got = out.to_host_vec(&stream)?;
        assert_eq!(
            got,
            vec![expected_step as u32, expected_acc as u32],
            "{name}: the lease must update the stepper exactly as the host model"
        );
    }

    println!("PASS: enum_niche_payload (all three shapes compute correctly)");
    Ok(())
}

/// Machine-checked depot criterion: every entry carries no `.local` traffic
/// with the fix. The parent fails on `smooth_triplets`, which keeps the
/// 16 B depot.
fn verify_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("enum_niche_payload.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    let document = ptx_parse::Document::parse(&ptx)?;

    for (name, expects_depot) in [
        ("smooth_triplets", false),
        ("lease_borrow", false),
        ("lease_raw", false),
    ] {
        let definition = document
            .definitions_named(name)
            .find(|definition| definition.callable().kind() == ptx_parse::CallableKind::Entry)
            .ok_or_else(|| format!("missing or incomplete PTX entry `{name}`"))?;
        let has_depot = definition.text().contains(".local");
        if has_depot != expects_depot {
            return Err(format!(
                "{name} local-depot expectation failed: expected depot={expects_depot}, found depot={has_depot}"
            )
            .into());
        }
    }
    println!("SUCCESS: depot expectations hold (no `.local` in any entry)");
    Ok(())
}
