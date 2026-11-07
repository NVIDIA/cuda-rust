/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]
// custom MIR's `Call(..)` terminator reads to clippy as a unit argument
#![allow(clippy::unit_arg)]

//! Shared-memory pointers stored through projected places.
//!
//! `DynamicSharedArray::get()` yields an `addrspace(3)` pointer, and a local
//! that only ever holds it gets a shared-memory slot. Writing that pointer
//! into a place declared as a generic `*mut u32` has to widen it to the
//! generic address space first. Every kernel here stores it through a
//! different kind of computed address:
//!
//! ```text
//! kernel                    store                       importer path
//! ────────────────────────  ──────────────────────────  ─────────────────────────────
//! call_destination_store    Call((*p) = get())          prepared call destination
//! deref_call_store          *p = get()                  Deref (via a call temporary)
//! deref_assign_store        *p = s                      Deref
//! field_store               h.ptr = s                   Field
//! deref_field_store         (*h).ptr = s                (Deref, Field)
//! nested_field_store        o.inner.ptr = s             (Field, Field)
//! array_element_store       a[i] = s                    array element
//! deref_index_store         (*a)[i] = s                 place-address walk
//! deref_nested_field_store  (*o).inner.ptr = s          place-address walk
//! volatile_store            write_volatile(p, s)        volatile_store intrinsic
//! ```
//!
//! Each kernel then reads the pointer back out of the place, writes a
//! lane-unique value through it, synchronizes, and reads its neighbor's
//! value. The host checks every lane, so a pointer that lost its shared
//! address on the way through the place shows up as a wrong value or a
//! fault rather than going unnoticed.
//!
//! Run: `cargo oxide run shared_ptr_projected_store`

use core::intrinsics::mir::*;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, DynamicSharedArray, device, kernel, thread};
use cuda_host::cuda_module;

/// Threads in the single block every kernel runs with.
///
/// Each lane owns one `u32` of the dynamic shared region and reads the next
/// lane's slot after the barrier, so the region holds exactly `LANES` values.
const LANES: usize = 32;

/// Value of [`Holder::tag`]; checked when the pointer is read back.
///
/// A store aimed at the wrong offset would overwrite it.
const HOLDER_TAG: u32 = 7;

/// Value of [`Outer::lead`]; checked when the pointer is read back.
///
/// A store aimed at the wrong offset would overwrite it.
const OUTER_LEAD: u32 = 9;

/// A struct with a generic raw-pointer field.
///
/// `repr(C)` and the leading `tag` put `ptr` at a nonzero offset, so a
/// field store that lands at the struct base instead of the field would be
/// visible.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Holder {
    /// Filler that keeps `ptr` off offset zero.
    ///
    /// Always [`HOLDER_TAG`]; [`holder_ptr`] checks it on every read back.
    tag: u32,
    /// The store target, declared as a generic `*mut u32`.
    ///
    /// It receives an `addrspace(3)` value, so the store has to widen it.
    ptr: *mut u32,
}

/// A struct that nests a [`Holder`] behind a leading field.
///
/// Gives the `(Field, Field)` and three-projection store paths a place two
/// field steps deep whose final field is a generic `*mut u32`. `repr(C)`
/// keeps `inner` behind `lead`, at a nonzero offset.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Outer {
    /// Filler that keeps `inner` off offset zero.
    ///
    /// Always [`OUTER_LEAD`]; [`outer_ptr`] checks it on every read back.
    lead: u32,
    /// Holds the generic pointer field the stores target.
    ///
    /// The nested stores write `inner.ptr`, two field steps below `Outer`.
    inner: Holder,
}

/// Returns a `Holder` whose pointer is null.
///
/// The kernels start every place null so that the projected store under test
/// is the only write that can make the pointer usable.
fn empty_holder() -> Holder {
    Holder {
        tag: HOLDER_TAG,
        ptr: core::ptr::null_mut(),
    }
}

/// `Call((*p) = DynamicSharedArray::get())`, kept as written by custom MIR.
///
/// Surface Rust would call into a temporary and then store it, so custom MIR
/// is the only way to make `(*p)` the call's own destination. The importer
/// then stores the shared pointer through the destination address it
/// prepared before the call. The pointer arrives as an argument so GVN cannot
/// see through it to a local.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn store_call_destination_through_deref(p: *mut *mut u32) {
    mir! {
        {
            Call((*p) = DynamicSharedArray::<u32>::get(), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// `*p = DynamicSharedArray::get()` in surface Rust.
///
/// The reproduction from the original report: rustc calls into a temporary
/// and then assigns it through `p`.
#[inline(never)]
fn store_call_result_through_deref(p: &mut *mut u32) {
    *p = DynamicSharedArray::<u32>::get();
}

/// Copies a shared-pointer local into `*p` and `*q`.
///
/// Storing `s` twice keeps it a separate local rather than letting rustc
/// turn the call into a projected destination, so `*p = s` is a `Deref`
/// assignment of an `addrspace(3)` value.
#[inline(never)]
fn store_local_through_deref(p: &mut *mut u32, q: &mut *mut u32) {
    let s = DynamicSharedArray::<u32>::get();
    *p = s;
    *q = s;
}

/// Stores a shared-pointer local into `(*h).ptr`.
///
/// The destination is a field of the referenced struct, the `(Deref, Field)`
/// assignment shape.
#[inline(never)]
fn store_local_through_deref_field(h: &mut Holder) {
    let s = DynamicSharedArray::<u32>::get();
    h.ptr = s;
}

/// Stores a shared-pointer local into `(*a)[i]`.
///
/// The destination is an element of a referenced array with a runtime index,
/// which the importer resolves with the place-address walker.
#[inline(never)]
fn store_local_through_deref_index(a: &mut [*mut u32; 2], i: usize) {
    let s = DynamicSharedArray::<u32>::get();
    a[i] = s;
}

/// Stores a shared-pointer local into `(*o).inner.ptr`.
///
/// Three projections deep, so this also goes through the place-address
/// walker rather than one of the fixed one- or two-level arms.
#[inline(never)]
fn store_local_through_deref_nested_field(o: &mut Outer) {
    let s = DynamicSharedArray::<u32>::get();
    o.inner.ptr = s;
}

/// Writes a shared-pointer local into `*p` with `write_volatile`.
///
/// `write_volatile` reaches the importer as the `volatile_store` intrinsic,
/// whose destination is an address rather than a projected place. The
/// pointer arrives as an argument so the store cannot be folded into a local.
#[inline(never)]
fn store_local_volatile(p: *mut *mut u32) {
    let s = DynamicSharedArray::<u32>::get();
    // SAFETY: the caller passes a valid, aligned pointer to its own local
    unsafe { core::ptr::write_volatile(p, s) };
}

/// Reads the pointer field out of a borrowed `Holder`.
///
/// Taking the struct by reference keeps the caller's local addressable, so
/// rustc cannot split it into scalars and the field store stays a field store.
/// A store that spilled into `tag` yields null, which faults the kernel.
#[inline(never)]
fn holder_ptr(h: &Holder) -> *mut u32 {
    if h.tag == HOLDER_TAG {
        h.ptr
    } else {
        core::ptr::null_mut()
    }
}

/// Reads the nested pointer field out of a borrowed `Outer`.
///
/// Like [`holder_ptr`], the borrow keeps the caller's aggregate in memory so
/// the nested field store is not rewritten into a scalar assignment, and a
/// clobbered `lead` or `tag` yields null.
#[inline(never)]
fn outer_ptr(o: &Outer) -> *mut u32 {
    if o.lead == OUTER_LEAD {
        holder_ptr(&o.inner)
    } else {
        core::ptr::null_mut()
    }
}

/// Reads one element out of a borrowed pointer array.
///
/// The borrow keeps the caller's array addressable, so its element store
/// is still an element store when it reaches the importer.
#[inline(never)]
fn array_ptr(a: &[*mut u32; 2], i: usize) -> *mut u32 {
    a[i]
}

/// The value lane `lane` writes for the kernel identified by `salt`.
///
/// Distinct per kernel and per lane, so the host can tell which kernel and
/// which neighbor produced every output.
fn lane_value(salt: u32, lane: usize) -> u32 {
    salt * 1000 + lane as u32
}

/// Exchanges lane values through `shared` and returns the neighbor's value.
///
/// Each lane writes its own slot, the block synchronizes, and each lane
/// then reads the next lane's slot. A wrong pointer makes the returned value
/// disagree with [`lane_value`].
///
/// # Safety
///
/// `shared` must point at the start of at least `LANES` `u32`s of shared
/// memory, and every thread of the block must call this together because it
/// contains a block-wide barrier.
unsafe fn exchange(shared: *mut u32, salt: u32) -> u32 {
    let lane = thread::threadIdx_x() as usize;
    let next = (lane + 1) % LANES;

    // SAFETY: each lane writes only its own slot and reads its neighbor's
    // slot after the barrier, inside the `LANES`-element shared region
    unsafe { shared.add(lane).write(lane_value(salt, lane)) };
    thread::sync_threads();
    unsafe { shared.add(next).read() }
}

/// The device kernels, one per importer store path.
///
/// `#[cuda_module]` also generates the host-side `load` function and the
/// typed launchers that [`main`] calls.
#[cuda_module]
mod kernels {
    use super::*;

    /// Writes `value` into this thread's slot of `out`.
    ///
    /// Every kernel launches one `LANES`-thread block over a `LANES`-element
    /// buffer, so each thread owns exactly one output.
    #[device]
    fn write_lane(out: &mut DisjointSlice<u32>, value: u32) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = value;
        }
    }

    /// `(*p)` as the destination of the `DynamicSharedArray::get()` call.
    ///
    /// A helper writes the call result straight into a generic pointer the
    /// kernel owns.
    #[kernel]
    pub fn call_destination_store(mut out: DisjointSlice<u32>) {
        let mut s: *mut u32 = core::ptr::null_mut();
        store_call_destination_through_deref(&raw mut s);
        // SAFETY: `s` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(s, 9) };
        write_lane(&mut out, neighbor);
    }

    /// `*p = DynamicSharedArray::get()` through a `&mut *mut u32`.
    ///
    /// The reproduction from the original report: a helper assigns the call
    /// result to a generic pointer the kernel owns.
    #[kernel]
    pub fn deref_call_store(mut out: DisjointSlice<u32>) {
        let mut s: *mut u32 = core::ptr::null_mut();
        store_call_result_through_deref(&mut s);
        // SAFETY: `s` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(s, 1) };
        write_lane(&mut out, neighbor);
    }

    /// `*p = s` where `s` is a shared-pointer local.
    ///
    /// Writes through `p` and reads the neighbor back through `q`, so both
    /// stored copies have to carry the same shared address.
    #[kernel]
    pub fn deref_assign_store(mut out: DisjointSlice<u32>) {
        let mut p: *mut u32 = core::ptr::null_mut();
        let mut q: *mut u32 = core::ptr::null_mut();
        store_local_through_deref(&mut p, &mut q);
        let lane = thread::threadIdx_x() as usize;

        // SAFETY: `p` and `q` are the dynamic shared base; each lane writes
        // its own slot and reads its neighbor's after the barrier
        unsafe { p.add(lane).write(lane_value(2, lane)) };
        thread::sync_threads();
        let neighbor = unsafe { q.add((lane + 1) % LANES).read() };
        write_lane(&mut out, neighbor);
    }

    /// `h.ptr = s` on a kernel-local struct.
    ///
    /// The one-level `Field` assignment into a local's own slot.
    #[kernel]
    pub fn field_store(mut out: DisjointSlice<u32>) {
        let s = DynamicSharedArray::<u32>::get();
        let mut h = empty_holder();
        h.ptr = s;
        let shared = holder_ptr(&h);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 3) };
        write_lane(&mut out, neighbor);
    }

    /// `(*h).ptr = s` through a `&mut Holder`.
    ///
    /// The `(Deref, Field)` assignment shape, applied by a helper.
    #[kernel]
    pub fn deref_field_store(mut out: DisjointSlice<u32>) {
        let mut h = empty_holder();
        store_local_through_deref_field(&mut h);
        let shared = holder_ptr(&h);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 4) };
        write_lane(&mut out, neighbor);
    }

    /// `o.inner.ptr = s` on a kernel-local nested struct.
    ///
    /// The `(Field, Field)` assignment into a local's own slot.
    #[kernel]
    pub fn nested_field_store(mut out: DisjointSlice<u32>) {
        let s = DynamicSharedArray::<u32>::get();
        let mut o = Outer {
            lead: OUTER_LEAD,
            inner: empty_holder(),
        };
        o.inner.ptr = s;
        let shared = outer_ptr(&o);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 5) };
        write_lane(&mut out, neighbor);
    }

    /// `a[i] = s` on a kernel-local array of generic pointers.
    ///
    /// The index depends on the lane, so it is a runtime-indexed element store
    /// into the array's own slot.
    #[kernel]
    pub fn array_element_store(mut out: DisjointSlice<u32>) {
        let s = DynamicSharedArray::<u32>::get();
        let i = thread::threadIdx_x() as usize & 1;
        let mut a: [*mut u32; 2] = [core::ptr::null_mut(); 2];
        a[i] = s;
        let shared = array_ptr(&a, i);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 6) };
        write_lane(&mut out, neighbor);
    }

    /// `(*a)[i] = s` through a `&mut [*mut u32; 2]`.
    ///
    /// The element store goes through the place-address walker.
    #[kernel]
    pub fn deref_index_store(mut out: DisjointSlice<u32>) {
        let i = thread::threadIdx_x() as usize & 1;
        let mut a: [*mut u32; 2] = [core::ptr::null_mut(); 2];
        store_local_through_deref_index(&mut a, i);
        let shared = array_ptr(&a, i);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 7) };
        write_lane(&mut out, neighbor);
    }

    /// `(*o).inner.ptr = s` through a `&mut Outer`.
    ///
    /// Three projections, so the store also goes through the place-address
    /// walker.
    #[kernel]
    pub fn deref_nested_field_store(mut out: DisjointSlice<u32>) {
        let mut o = Outer {
            lead: OUTER_LEAD,
            inner: empty_holder(),
        };
        store_local_through_deref_nested_field(&mut o);
        let shared = outer_ptr(&o);
        // SAFETY: `shared` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(shared, 8) };
        write_lane(&mut out, neighbor);
    }

    /// `write_volatile(p, s)` into a generic pointer the kernel owns.
    ///
    /// The volatile store intrinsic writes through its pointer argument, so
    /// it needs the same widening as a projected store.
    #[kernel]
    pub fn volatile_store(mut out: DisjointSlice<u32>) {
        let mut s: *mut u32 = core::ptr::null_mut();
        store_local_volatile(&raw mut s);
        // SAFETY: `s` is the dynamic shared base and the whole block calls in
        let neighbor = unsafe { exchange(s, 10) };
        write_lane(&mut out, neighbor);
    }
}

/// Signature shared by every launcher in [`main`].
///
/// Each launcher runs one kernel over a single `LANES`-thread block with
/// `LANES` `u32`s of dynamic shared memory.
type Launch = fn(
    &kernels::LoadedModule,
    &cuda_core::CudaStream,
    LaunchConfig,
    &mut DeviceBuffer<u32>,
) -> Result<(), cuda_core::DriverError>;

/// Runs every kernel once and checks all lanes against [`lane_value`].
///
/// Each kernel gets a fresh zeroed output buffer. Any mismatching lane is
/// reported and makes the process exit with status 1.
fn main() {
    println!("=== shared_ptr_projected_store (shared pointers stored through places) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (LANES as u32, 1, 1),
        shared_mem_bytes: (LANES * core::mem::size_of::<u32>()) as u32,
    };

    // SAFETY (every launcher): one `LANES`-thread block with `LANES` u32s of
    // dynamic shared memory, and `out` holds one element per thread
    let cases: [(&str, u32, Launch); 10] = [
        ("call_destination_store", 9, |m, s, c, o| unsafe {
            m.call_destination_store(s, c, o)
        }),
        ("deref_call_store", 1, |m, s, c, o| unsafe {
            m.deref_call_store(s, c, o)
        }),
        ("deref_assign_store", 2, |m, s, c, o| unsafe {
            m.deref_assign_store(s, c, o)
        }),
        ("field_store", 3, |m, s, c, o| unsafe {
            m.field_store(s, c, o)
        }),
        ("deref_field_store", 4, |m, s, c, o| unsafe {
            m.deref_field_store(s, c, o)
        }),
        ("nested_field_store", 5, |m, s, c, o| unsafe {
            m.nested_field_store(s, c, o)
        }),
        ("array_element_store", 6, |m, s, c, o| unsafe {
            m.array_element_store(s, c, o)
        }),
        ("deref_index_store", 7, |m, s, c, o| unsafe {
            m.deref_index_store(s, c, o)
        }),
        ("deref_nested_field_store", 8, |m, s, c, o| unsafe {
            m.deref_nested_field_store(s, c, o)
        }),
        ("volatile_store", 10, |m, s, c, o| unsafe {
            m.volatile_store(s, c, o)
        }),
    ];

    let mut failures = 0;
    for (name, salt, launch) in cases {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, LANES).expect("alloc");
        launch(&module, stream.as_ref(), cfg, &mut out).expect("Kernel launch failed");
        stream.synchronize().expect("sync");
        let result = out.to_host_vec(&stream).expect("copy back");

        let mut kernel_failures = 0;
        for (lane, &got) in result.iter().enumerate() {
            let expected = lane_value(salt, (lane + 1) % LANES);
            if got != expected {
                eprintln!("✗ {name} lane {lane}: got {got}, expected {expected}");
                kernel_failures += 1;
            }
        }
        if kernel_failures == 0 {
            println!("✓ {name}: all {LANES} lanes read their neighbor through the stored pointer");
        }
        failures += kernel_failures;
    }

    if failures > 0 {
        eprintln!("✗ {failures} lanes mismatched");
        std::process::exit(1);
    }
    println!("\n✓ SUCCESS: shared_ptr_projected_store passed!");
}
