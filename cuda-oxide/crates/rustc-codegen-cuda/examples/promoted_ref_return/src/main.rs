// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression for issue #1724: device helpers that return a promoted
//! `&'static` reference constant.
//!
//! A reference constant is `&'static`, so safe code can return it from the
//! function that names it. The importer used to build the pointee in that
//! function's frame (alloca + store + address), so an `#[inline(never)]`
//! helper handed its caller a pointer to a dead stack slot. Each helper below
//! is `#[inline(never)]` so the reference has to cross a real call boundary,
//! and the kernel reads every value back through it:
//!
//! - `pair`: a named `const PAIR: &(u32, u32)`
//! - `span`: a named reference to a struct constant
//! - `literal`: a tuple literal rustc promotes to `'static`
//! - `answer`: a promoted scalar
//! - `nested`: `&&u32`, whose outer allocation holds a pointer to an inner one
//! - `second`: `&PAIR.1`, a pointer 4 bytes into `PAIR`'s allocation
//! - `maybe`: a reference to a payload-carrying enum
//! - `wrapped`: a by-value tuple whose field points at a promoted `u32`
//! - `some_ref`: `Option<&u32>` whose payload points at a promoted `u32`
//! - `via_static`: `&&TABLE[2]`, a promoted pointer into a device static
//! - `static_ref`: a device static whose initializer is `&19`
//! - `static_slice`: a device static `&[u32]` whose data word points at a
//!   promoted array
//! - `both`: `&Option<u32>` and `&u32` into its payload, two views of one
//!   allocation; the kernel checks that the two payload addresses are equal
//! - `raw`, `raw_second`, `raw_field`: `*mut u32` constants formed with `as`
//!   casts from references, at offset 0, 4 bytes into `PAIR`, and as a tuple
//!   field; the kernel only reads through them, since the storage stays
//!   read-only
//!
//! `verify-code-shape.sh` checks the emitted LLVM IR and PTX: none of the
//! helpers has a stack slot, and the constants live in read-only
//! `addrspace(1)` globals. It also checks that splicing more kernels in at the
//! `ADDED KERNEL SLOT` comment leaves every helper's globals unchanged.
//!
//! Run with:
//!   cargo oxide run promoted_ref_return

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{device, kernel, thread};
use cuda_host::cuda_module;

/// A small struct constant reached only through a reference.
#[derive(Clone, Copy)]
pub struct Span {
    /// Inclusive lower bound.
    pub lo: u32,
    /// Exclusive upper bound.
    pub hi: u32,
}

/// Named reference to a tuple; its allocation is shared by `PAIR` and `SECOND`.
const PAIR: &(u32, u32) = &(3, 4);

/// Named reference to a struct constant.
const SPAN: &Span = &Span { lo: 8, hi: 16 };

/// Interior pointer: 4 bytes into `PAIR`'s allocation.
const SECOND: &u32 = &PAIR.1;

/// Reference to an enum whose variant carries a payload.
const MAYBE: &Option<u32> = &Some(11);

/// By-value tuple whose first field points into anonymous constant memory.
const WRAPPED: (&u32, u32) = (&13, 2);

/// Niche-encoded `Option<&u32>` whose payload points into constant memory.
const SOME_REF: Option<&u32> = Some(&17);

/// Device static reached through a promoted pointer.
static TABLE: [u32; 4] = [20, 21, 22, 23];

/// Promoted pointer to a pointer into `TABLE`.
const VIA_STATIC: &&u32 = &&TABLE[2];

/// Device static whose initializer points into anonymous constant memory.
static STATIC_REF: &u32 = &19;

/// Device static slice whose data word points into anonymous constant memory.
static STATIC_SLICE: &[u32] = &[31, 37, 41];

/// A reference to an enum and a reference into its payload. Both point into
/// one allocation, so the payload reached through either must be one address.
const BOTH: (&Option<u32>, &u32) = {
    let whole = &Some(53);
    let payload = match whole {
        Some(value) => value,
        None => panic!(),
    };
    (whole, payload)
};

/// A `*mut` raw pointer into read-only constant memory, formed by `as` casts.
const RAW: *mut u32 = &59_u32 as *const u32 as *mut u32;

/// A `*mut` raw pointer 4 bytes into `PAIR`'s allocation.
const RAW_SECOND: *mut u32 = SECOND as *const u32 as *mut u32;

/// A tuple whose `*mut` field points into read-only constant memory.
const RAW_FIELD: (*mut u32, u32) = (&61_u32 as *const u32 as *mut u32, 1);

/// Number of values the kernel writes.
const VALUE_COUNT: usize = 20;

/// Values the kernel must read back, in output order.
const EXPECTED: [u32; VALUE_COUNT] = [
    3, 4, 8, 16, 5, 6, 42, 7, 4, 11, 15, 17, 22, 19, 44, 53, 1, 59, 4, 62,
];

/// Device helpers that each return one form of reference constant, and the
/// kernel that reads them all back.
#[cuda_module]
mod kernels {
    use super::*;

    /// Returns the named tuple reference.
    #[device]
    #[inline(never)]
    pub fn pair() -> &'static (u32, u32) {
        PAIR
    }

    /// Returns the named struct reference.
    #[device]
    #[inline(never)]
    pub fn span() -> &'static Span {
        SPAN
    }

    /// Returns a tuple literal that rustc promotes to `'static`.
    #[device]
    #[inline(never)]
    pub fn literal() -> &'static (u32, u32) {
        &(5, 6)
    }

    /// Returns a promoted scalar.
    #[device]
    #[inline(never)]
    pub fn answer() -> &'static u32 {
        &42
    }

    /// Returns a reference to a promoted reference.
    #[device]
    #[inline(never)]
    pub fn nested() -> &'static &'static u32 {
        &&7
    }

    /// Returns a pointer into the middle of `PAIR`.
    #[device]
    #[inline(never)]
    pub fn second() -> &'static u32 {
        SECOND
    }

    /// Returns a reference to an enum constant with a payload.
    #[device]
    #[inline(never)]
    pub fn maybe() -> &'static Option<u32> {
        MAYBE
    }

    /// Returns a tuple whose pointer field refers to constant memory.
    #[device]
    #[inline(never)]
    pub fn wrapped() -> (&'static u32, u32) {
        WRAPPED
    }

    /// Returns an `Option<&u32>` whose payload refers to constant memory.
    #[device]
    #[inline(never)]
    pub fn some_ref() -> Option<&'static u32> {
        SOME_REF
    }

    /// Returns a promoted pointer whose target is a device static.
    #[device]
    #[inline(never)]
    pub fn via_static() -> &'static &'static u32 {
        VIA_STATIC
    }

    /// Returns the reference stored in a device static.
    #[device]
    #[inline(never)]
    pub fn static_ref() -> &'static u32 {
        STATIC_REF
    }

    /// Returns the slice stored in a device static.
    #[device]
    #[inline(never)]
    pub fn static_slice() -> &'static [u32] {
        STATIC_SLICE
    }

    /// Returns two references into one enum allocation.
    #[device]
    #[inline(never)]
    pub fn both() -> (&'static Option<u32>, &'static u32) {
        BOTH
    }

    /// Returns a `*mut` raw pointer into read-only constant memory.
    #[device]
    #[inline(never)]
    pub fn raw() -> *mut u32 {
        RAW
    }

    /// Returns a `*mut` raw pointer into the middle of `PAIR`.
    #[device]
    #[inline(never)]
    pub fn raw_second() -> *mut u32 {
        RAW_SECOND
    }

    /// Returns a tuple whose `*mut` field points into constant memory.
    #[device]
    #[inline(never)]
    pub fn raw_field() -> (*mut u32, u32) {
        RAW_FIELD
    }

    // ADDED KERNEL SLOT: verify-code-shape.sh splices kernels with their own
    // constants here and checks that no helper above gets a different global

    /// Thread 0 reads every helper's result and writes `VALUE_COUNT` values.
    ///
    /// # Safety
    ///
    /// `output` must address `VALUE_COUNT` writable `u32` values.
    #[kernel]
    pub unsafe fn read_promoted(output: *mut u32) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let pair = pair();
        let span = span();
        let literal = literal();
        let (wrapped_ref, wrapped_add) = wrapped();
        let maybe = match maybe() {
            Some(value) => *value,
            None => 0,
        };
        let some_ref = match some_ref() {
            Some(value) => *value,
            None => 0,
        };
        let slice = static_slice();
        let (whole, payload) = both();
        let (whole_value, same_payload) = match whole {
            Some(value) => (*value, core::ptr::eq(value, payload) as u32),
            None => (0, 0),
        };
        let (field_ptr, field_add) = raw_field();
        // SAFETY: each pointer addresses a live read-only `u32` constant, and
        // is only read
        let (raw, raw_second, field) = unsafe { (*raw(), *raw_second(), *field_ptr) };
        let values = [
            pair.0,
            pair.1,
            span.lo,
            span.hi,
            literal.0,
            literal.1,
            *answer(),
            **nested(),
            *second(),
            maybe,
            *wrapped_ref + wrapped_add,
            some_ref,
            **via_static(),
            *static_ref(),
            slice[2] + slice.len() as u32,
            whole_value,
            same_payload,
            raw,
            raw_second,
            field + field_add,
        ];

        let mut index = 0;
        while index < VALUE_COUNT {
            // SAFETY: the caller provides VALUE_COUNT writable elements
            unsafe { output.add(index).write(values[index]) };
            index += 1;
        }
    }
}

/// Launches the kernel and checks every value it read.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let output = DeviceBuffer::<u32>::zeroed(&stream, VALUE_COUNT)?;

    // SAFETY: the output allocation holds VALUE_COUNT u32 values
    unsafe {
        module.read_promoted(
            &stream,
            LaunchConfig::for_num_elems(1),
            output.cu_deviceptr() as *mut u32,
        )
    }?;

    let got = output.to_host_vec(&stream)?;
    assert_eq!(got, EXPECTED, "promoted reference values read on device");

    println!("promoted_ref_return: PASS ({VALUE_COUNT} values)");
    Ok(())
}
