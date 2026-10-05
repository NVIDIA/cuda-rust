/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! FNV-1a 64-bit trace state shared between the CPU oracle and the GPU runs.
//!
//! The trace is a single `u64` global. Every interesting intermediate value is
//! folded into it byte-by-byte. The hash is the program's fingerprint: if both
//! backends are correct, the CPU `u64` and the GPU `u64` are equal.
//!
//! The state starts at zero because cuda-oxide currently only supports
//! zero-initialized device statics. Each run must call [`trace_reset`] before
//! executing the program, and [`trace_finish`] after.
//!
//! All trace functions are marked `#[inline]` so their MIR is encoded in the
//! `fuzzer` rlib and reachable to cuda-oxide's MIR collector when the smoke
//! example is compiled for the device. The `static mut RL_TRACE` already
//! prevents the optimizer from constant-folding the trace state away, so we
//! don't need `#[inline]` to keep the byte mixers separate.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

static mut RL_TRACE: u64 = 0;

/// Reset the trace state to the FNV-1a 64-bit offset basis.
#[inline]
pub fn trace_reset() {
    unsafe {
        RL_TRACE = FNV_OFFSET;
    }
}

/// Read out the current trace state.
#[inline]
pub fn trace_finish() -> u64 {
    unsafe { RL_TRACE }
}

#[inline]
fn trace_write_byte(byte: u8) {
    unsafe {
        RL_TRACE = (RL_TRACE ^ byte as u64).wrapping_mul(FNV_PRIME);
    }
}

#[inline]
fn trace_write_u8(val: u8) {
    trace_write_byte(val);
}

#[inline]
fn trace_write_i8(val: i8) {
    trace_write_u8(val as u8);
}

#[inline]
fn trace_write_u16(val: u16) {
    trace_write_u8((val & 0xff) as u8);
    trace_write_u8(((val >> 8) & 0xff) as u8);
}

#[inline]
fn trace_write_i16(val: i16) {
    trace_write_u16(val as u16);
}

#[inline]
fn trace_write_u32(val: u32) {
    trace_write_u8((val & 0xff) as u8);
    trace_write_u8(((val >> 8) & 0xff) as u8);
    trace_write_u8(((val >> 16) & 0xff) as u8);
    trace_write_u8(((val >> 24) & 0xff) as u8);
}

#[inline]
fn trace_write_i32(val: i32) {
    trace_write_u32(val as u32);
}

#[inline]
fn trace_write_u64(val: u64) {
    trace_write_u32((val & 0xffff_ffff) as u32);
    trace_write_u32((val >> 32) as u32);
}

#[inline]
fn trace_write_i64(val: i64) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_u128(val: u128) {
    trace_write_u64((val & 0xffff_ffff_ffff_ffff) as u64);
    trace_write_u64((val >> 64) as u64);
}

#[inline]
fn trace_write_i128(val: i128) {
    trace_write_u128(val as u128);
}

#[inline]
fn trace_write_usize(val: usize) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_isize(val: isize) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_bool(val: bool) {
    trace_write_u8(val as u8);
}

#[inline]
fn trace_write_char(val: char) {
    trace_write_u32(val as u32);
}

/// Floats are folded as their bit patterns, so the trace stays an exact
/// comparison. A tolerance here would compare something other than the value
/// the backend produced, and the mismatches worth finding are the ones a
/// tolerance hides.
///
/// The bits agree for every non-NaN value the two backends produce. NaN
/// payload bits are not pinned down by Rust, so both writers canonicalize a
/// NaN to the quiet-NaN bit pattern before hashing, as Cranelift's fuzzgen
/// does. A payload divergence therefore cannot produce a false MISMATCH,
/// while a NaN against a non-NaN still hashes differently and remains a real
/// signal.
#[inline]
fn trace_write_f32(val: f32) {
    let bits = if val.is_nan() {
        f32::NAN.to_bits()
    } else {
        val.to_bits()
    };
    trace_write_u32(bits);
}

#[inline]
fn trace_write_f64(val: f64) {
    let bits = if val.is_nan() {
        f64::NAN.to_bits()
    } else {
        val.to_bits()
    };
    trace_write_u64(bits);
}

/// Values that can be folded into the trace.
///
/// A scalar folds as its bytes. An aggregate folds as its leaves, in
/// declaration order, through those same scalar writers, so the byte sequence
/// an aggregate produces is the one a scalar-by-scalar dump of the same leaves
/// would produce. Composing that way keeps one trace model rather than two.
///
/// The fold reads leaves and never the aggregate's bytes. Padding is
/// uninitialized, so folding raw bytes would be undefined behavior in the CPU
/// oracle, and any layout the two backends pad differently would report a
/// MISMATCH on every seed that touched it. Padding is also where the silent
/// miscompile behind #393 lived, so a byte fold would bury the signal this
/// fuzzer exists to find.
pub trait TraceValue {
    fn trace_write(self);
}

macro_rules! impl_trace_value {
    ($($ty:ty => $writer:ident),* $(,)?) => {
        $(
            impl TraceValue for $ty {
                #[inline]
                fn trace_write(self) {
                    $writer(self);
                }
            }
        )*
    };
}

impl_trace_value! {
    bool => trace_write_bool,
    i8 => trace_write_i8,
    i16 => trace_write_i16,
    i32 => trace_write_i32,
    i64 => trace_write_i64,
    i128 => trace_write_i128,
    isize => trace_write_isize,
    u8 => trace_write_u8,
    u16 => trace_write_u16,
    u32 => trace_write_u32,
    u64 => trace_write_u64,
    u128 => trace_write_u128,
    usize => trace_write_usize,
    char => trace_write_char,
    f32 => trace_write_f32,
    f64 => trace_write_f64,
}

/// An array folds each element in index order.
///
/// The walk is an indexed `while` rather than a `for` over the array's
/// `IntoIterator`, because this code is compiled for the device as well as the
/// host and #399 records that small local arrays consumed through iterator
/// adapters stay in local memory. An index loop states the same thing without
/// depending on how that resolves.
impl<T: TraceValue + Copy, const N: usize> TraceValue for [T; N] {
    #[inline]
    fn trace_write(self) {
        let mut index = 0;
        while index < N {
            self[index].trace_write();
            index += 1;
        }
    }
}

macro_rules! impl_trace_value_tuple {
    ($(($($field:tt: $ty:ident),+))+) => {
        $(
            impl<$($ty: TraceValue),+> TraceValue for ($($ty,)+) {
                #[inline]
                fn trace_write(self) {
                    $(self.$field.trace_write();)+
                }
            }
        )+
    };
}

impl_trace_value_tuple! {
    (0: A)
    (0: A, 1: B)
    (0: A, 1: B, 2: C)
    (0: A, 1: B, 2: C, 3: D)
    (0: A, 1: B, 2: C, 3: D, 4: E)
}

/// The unit type carries no leaves, so it folds to nothing.
///
/// The adapter prunes unit-typed values before it builds a dump tuple, so this
/// exists for a unit nested inside an aggregate, which the adapter cannot see.
impl TraceValue for () {
    #[inline]
    fn trace_write(self) {}
}

/// The argument bundle a single dump site folds into the trace.
///
/// `dump_var` accepts any `TraceDump`. We provide implementations for `()` and
/// tuples up to arity 5, which matches the largest argument bundle rustlantis
/// emits today after we prune unit values from its `dump_var` calls.
///
/// This is the outer shape of a dump site. What each element may be is
/// [`TraceValue`]'s question, and an element is free to be an aggregate.
pub trait TraceDump {
    fn trace_dump(self);
}

impl TraceDump for () {
    #[inline]
    fn trace_dump(self) {}
}

impl<A: TraceValue> TraceDump for (A,) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue> TraceDump for (A, B) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue> TraceDump for (A, B, C) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue, D: TraceValue> TraceDump for (A, B, C, D) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
        self.3.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue, D: TraceValue, E: TraceValue> TraceDump
    for (A, B, C, D, E)
{
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
        self.3.trace_write();
        self.4.trace_write();
    }
}

/// Fold a value into the trace.
///
/// Generic over any `TraceDump`, so a single call site handles every supported
/// argument shape. Generated programs typically materialize a tuple local with
/// the values to dump, then pass it here.
#[inline]
pub fn dump_var<T: TraceDump>(value: T) {
    value.trace_dump();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fold<T: TraceValue>(value: T) -> u64 {
        trace_reset();
        value.trace_write();
        trace_finish()
    }

    /// Every assertion lives in one test.
    ///
    /// The trace is a single `static mut`, and cargo runs test functions on
    /// parallel threads, so two tests folding at once would race on it and
    /// report whichever interleaving they happened to get.
    #[test]
    fn an_aggregate_folds_as_its_leaves() {
        // An array folds as its elements, in index order.
        assert_eq!(fold([1u8, 2, 3, 4]), {
            trace_reset();
            1u8.trace_write();
            2u8.trace_write();
            3u8.trace_write();
            4u8.trace_write();
            trace_finish()
        });

        // A tuple folds as its fields, in declaration order. `(u8, u32)` is
        // the discriminating case: rustc lays it out with padding, and this
        // holds only for an implementation that reads the two fields. Folding
        // the aggregate's bytes instead would mix three padding bytes in and
        // produce a different value here.
        assert_eq!(fold((7u8, 0x1122_3344u32)), {
            trace_reset();
            7u8.trace_write();
            0x1122_3344u32.trace_write();
            trace_finish()
        });

        // Nesting composes: an array of tuples reaches every leaf.
        assert_eq!(fold([(1u8, 2u16), (3u8, 4u16)]), {
            trace_reset();
            1u8.trace_write();
            2u16.trace_write();
            3u8.trace_write();
            4u16.trace_write();
            trace_finish()
        });

        // Order is part of the fold. An implementation walking an array
        // backwards passes every assertion above and fails this one.
        assert_ne!(fold([1u8, 2u8]), fold([2u8, 1u8]));

        // A shape with no leaves folds to nothing, leaving the offset basis.
        trace_reset();
        let empty = trace_finish();
        assert_eq!(fold([0u8; 0]), empty);
        assert_eq!(fold(()), empty);
    }
}
