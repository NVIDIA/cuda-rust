/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for MIR import of array constants.
//!
//! Covered shapes:
//! - bare `[T; N]` constants indexed by a runtime value,
//! - bare arrays of direct-tag enum constants,
//! - nested `[[T; M]; N]` constants,
//! - arrays of padded tuple constants containing no-payload enums,
//! - nested tuples with zero-sized fields,
//! - non-empty all-ZST tuples whose fields have equal offsets,
//! - tuple arrays whose fields rustc reorders in memory,
//! - tuple arrays containing an over-aligned zero-sized field,
//! - bare arrays of padded and nested struct constants,
//! - immutable promotion of packed struct arrays and references,
//! - bare arrays of over-aligned zero-sized struct constants,
//! - direct padded tuple constants,
//! - pointer-to-array constants (`&[T; N]`), including padded and nested structs,
//! - tuple arrays containing pointers to device statics,
//! - tuple-array pointer relocations with non-zero static addends,
//! - initialized union constants, including `[U; N]` runtime indexing,
//! - thin-pointer-only union constants that preserve device-static provenance,
//! - relocation-free pointer/integer union constants materialized as byte images,
//! - unions nested inside tuple and struct constants,
//! - `MaybeUninit<T>` array constants.
//!
//! Run with:
//!   cargo oxide run array_constants
//!   ./crates/rustc-codegen-cuda/examples/array_constants/verify-code-shape.sh

use core::mem::MaybeUninit;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const BARE_TABLE: [f32; 4] = [1.25, -2.5, 5.0, 10.5];
const NESTED_TABLE: [[u32; 3]; 2] = [[11, 13, 17], [19, 23, 29]];
const POINTER_TABLE: &[u32; 4] = &[31, 37, 41, 43];
const TUPLE_TABLE: [(bool, Side); 6] = [
    (false, Side::LowX),
    (true, Side::HighX),
    (false, Side::LowY),
    (true, Side::HighY),
    (false, Side::LowZ),
    (true, Side::HighZ),
];
const BARE_ENUM_TABLE: [Side; 6] = [
    Side::LowX,
    Side::HighZ,
    Side::HighY,
    Side::HighX,
    Side::LowZ,
    Side::LowY,
];
const EXPECTED_BARE_ENUM: [u32; 6] = [1, 6, 4, 2, 5, 3];
const NESTED_TUPLE_TABLE: [((u8, ()), u32); 2] = [((3, ()), 17), ((5, ()), 29)];
const ALL_ZST_TUPLE_TABLE: [(((), ()), u32); 2] = [(((), ()), 59), (((), ()), 61)];

// rustc lays these fields out at offsets 4, 0, and 8 respectively on the
// supported 64-bit target. Reading the allocation in declaration order would
// therefore corrupt both of the first two values.
const REORDERED_TUPLE_TABLE: [(u8, u32, u64); 2] = [
    (0xa5, 0x1122_3344, 0x0102_0304_0506_0708),
    (0x5a, 0x99aa_bbcc, 0x8877_6655_4433_2211),
];

#[derive(Clone, Copy)]
#[repr(C)]
struct PaddedStruct {
    tag: u8,
    value: u32,
}

const PADDED_STRUCT_TABLE: [PaddedStruct; 2] = [
    PaddedStruct {
        tag: 0xa5,
        value: 0x1122_3344,
    },
    PaddedStruct {
        tag: 0x5a,
        value: 0x99aa_bbcc,
    },
];
const PADDED_STRUCT_REF: &[PaddedStruct; 2] = &PADDED_STRUCT_TABLE;

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct PackedStruct {
    tag: u8,
    value: u32,
}

const PACKED_STRUCT_TABLE: [PackedStruct; 2] = [
    PackedStruct {
        tag: 0xa5,
        value: 0x1122_3344,
    },
    PackedStruct {
        tag: 0x5a,
        value: 0x99aa_bbcc,
    },
];
const PACKED_STRUCT_REF: &[PackedStruct; 2] = &PACKED_STRUCT_TABLE;

#[derive(Clone, Copy)]
#[repr(C)]
struct NestedStruct {
    inner: PaddedStruct,
    wide: u64,
}

const NESTED_STRUCT_TABLE: [NestedStruct; 2] = [
    NestedStruct {
        inner: PaddedStruct {
            tag: 0x33,
            value: 0x0102_0304,
        },
        wide: 0x1112_1314_1516_1718,
    },
    NestedStruct {
        inner: PaddedStruct {
            tag: 0xcc,
            value: 0xa1a2_a3a4,
        },
        wide: 0x8182_8384_8586_8788,
    },
];
const NESTED_STRUCT_REF: &[NestedStruct; 2] = &NESTED_STRUCT_TABLE;

#[derive(Clone, Copy)]
#[repr(align(32))]
struct ZstStruct;

const ZST_STRUCT_TABLE: [ZstStruct; 2] = [ZstStruct, ZstStruct];

#[derive(Clone, Copy)]
#[repr(align(32))]
struct Align32;

const OVERALIGNED_ZST_TUPLE_TABLE: [(Align32, u8); 2] = [(Align32, 0x12), (Align32, 0x34)];
const DIRECT_TUPLE: (u8, u32) = (7, 41);

static FIRST_POINTER_VALUE: u32 = 11;
static POINTER_VALUES: [u32; 3] = [11, 17, 23];

const POINTER_TUPLE_TABLE: [(&u32, bool); 2] =
    [(&FIRST_POINTER_VALUE, false), (&POINTER_VALUES[2], true)];

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
union PointerBits {
    word: *const u32,
    bytes: *const u8,
}

const DIRECT_POINTER_UNION: PointerBits = PointerBits {
    word: &FIRST_POINTER_VALUE as *const u32,
};
const POINTER_UNION_TABLE: [PointerBits; 2] = [
    PointerBits {
        word: &POINTER_VALUES[0] as *const u32,
    },
    PointerBits {
        // Initialize through the other union view and keep a non-zero addend.
        // Import must preserve the relocation rather than the placeholder bytes.
        bytes: &POINTER_VALUES[2] as *const u32 as *const u8,
    },
];

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
union PointerOrBits {
    pointer: *const u32,
    bits: u64,
}

const DIRECT_POINTER_INTEGER_UNION: PointerOrBits = PointerOrBits {
    bits: 0x1122_3344_5566_7788,
};
const POINTER_INTEGER_UNION_TABLE: [PointerOrBits; 2] = [
    PointerOrBits {
        bits: 0x0123_4567_89ab_cdef,
    },
    PointerOrBits {
        bits: 0xfedc_ba98_7654_3210,
    },
];

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
union Bits {
    word: u32,
    bytes: [u8; 4],
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
union PartialBits {
    byte: u8,
    word: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct UnionHolder {
    tag: u8,
    value: Bits,
}

const DIRECT_UNION: Bits = Bits { word: 0x1122_3344 };
const UNION_TABLE: [Bits; 4] = [
    Bits { word: 0x1122_3344 },
    Bits { word: 0x5566_7788 },
    Bits {
        bytes: [0xcc, 0xbb, 0xaa, 0x99],
    },
    Bits { word: 0xdead_beef },
];
const UNION_TUPLE: (u8, Bits) = (7, Bits { word: 0x1020_3040 });
const UNION_STRUCT: UnionHolder = UnionHolder {
    tag: 9,
    value: Bits {
        bytes: [0x04, 0x03, 0x02, 0x01],
    },
};
const PARTIAL_UNION: PartialBits = PartialBits { byte: 0x7f };
const MAYBE_UNINIT_TABLE: [MaybeUninit<u32>; 2] =
    [MaybeUninit::new(0x1357_9bdf), MaybeUninit::new(0x2468_ace0)];

#[derive(Clone, Copy)]
#[repr(u32)]
enum Side {
    LowX = 1,
    HighX = 2,
    LowY = 3,
    HighY = 4,
    LowZ = 5,
    HighZ = 6,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn bare_array_value(i: usize) -> f32 {
        BARE_TABLE[i & 3]
    }

    #[inline(never)]
    fn nested_array_value(i: usize) -> u32 {
        let row = i & 1;
        let col = (i / 2) % 3;
        NESTED_TABLE[row][col]
    }

    #[inline(never)]
    fn pointer_to_array_value(i: usize) -> u32 {
        POINTER_TABLE[i & 3]
    }

    #[inline(never)]
    fn tuple_array_value(i: usize) -> u32 {
        let (is_high, side) = TUPLE_TABLE[i % 6];
        (side as u32) * 10 + (is_high as u32)
    }

    #[inline(never)]
    fn bare_enum_array_value(i: usize) -> u32 {
        BARE_ENUM_TABLE[i % BARE_ENUM_TABLE.len()] as u32
    }

    #[inline(never)]
    fn nested_tuple_array_value(i: usize) -> u32 {
        let ((tag, ()), value) = NESTED_TUPLE_TABLE[i & 1];
        tag as u32 + value
    }

    #[inline(never)]
    fn all_zst_tuple_array_value(i: usize) -> u32 {
        let (((), ()), value) = ALL_ZST_TUPLE_TABLE[i & 1];
        value
    }

    #[inline(never)]
    fn reordered_tuple_array_value(i: usize) -> u32 {
        let (byte, word, wide) = REORDERED_TUPLE_TABLE[i & 1];
        (byte as u32)
            .wrapping_mul(257)
            .wrapping_add(word)
            .wrapping_mul(257)
            .wrapping_add(wide as u32)
            .wrapping_mul(257)
            .wrapping_add((wide >> 32) as u32)
    }

    #[inline(never)]
    fn overaligned_zst_tuple_array_value(i: usize) -> u32 {
        let pair = OVERALIGNED_ZST_TUPLE_TABLE[i & 1];
        let address_low_bits = (&pair as *const (Align32, u8) as usize) & 31;
        let (_, byte) = pair;
        byte as u32 + address_low_bits as u32
    }

    #[inline(never)]
    fn padded_struct_array_value(i: usize) -> u32 {
        let value = PADDED_STRUCT_TABLE[i & 1];
        (value.tag as u32)
            .wrapping_mul(257)
            .wrapping_add(value.value)
    }

    #[inline(never)]
    fn nested_struct_array_value(i: usize) -> u32 {
        let value = NESTED_STRUCT_TABLE[i & 1];
        (value.inner.tag as u32)
            .wrapping_mul(257)
            .wrapping_add(value.inner.value)
            .wrapping_mul(257)
            .wrapping_add(value.wide as u32)
            .wrapping_mul(257)
            .wrapping_add((value.wide >> 32) as u32)
    }

    #[inline(never)]
    fn padded_struct_ref_value(i: usize) -> u32 {
        let idx = i & 1;
        (PADDED_STRUCT_REF[idx].tag as u32)
            .wrapping_mul(257)
            .wrapping_add(PADDED_STRUCT_REF[idx].value)
    }

    #[inline(never)]
    fn packed_struct_array_value(i: usize) -> u32 {
        let value = PACKED_STRUCT_TABLE[i & 1];
        (value.tag as u32)
            .wrapping_mul(257)
            .wrapping_add(value.value)
    }

    #[inline(never)]
    fn packed_struct_ref_value(i: usize) -> u32 {
        let value = PACKED_STRUCT_REF[i & 1];
        (value.tag as u32)
            .wrapping_mul(257)
            .wrapping_add(value.value)
    }

    #[inline(never)]
    fn nested_struct_ref_value(i: usize) -> u32 {
        let idx = i & 1;
        (NESTED_STRUCT_REF[idx].inner.tag as u32)
            .wrapping_mul(257)
            .wrapping_add(NESTED_STRUCT_REF[idx].inner.value)
            .wrapping_mul(257)
            .wrapping_add(NESTED_STRUCT_REF[idx].wide as u32)
            .wrapping_mul(257)
            .wrapping_add((NESTED_STRUCT_REF[idx].wide >> 32) as u32)
    }

    #[inline(never)]
    fn zst_struct_array_value(i: usize) -> u32 {
        let value = ZST_STRUCT_TABLE[i & 1];
        ((&value as *const ZstStruct as usize) & 31) as u32
    }

    #[inline(never)]
    fn direct_tuple_value() -> (u8, u32) {
        DIRECT_TUPLE
    }

    #[inline(never)]
    fn pointer_tuple_array_value(i: usize) -> u32 {
        let (pointer, flag) = POINTER_TUPLE_TABLE[i & 1];
        *pointer + flag as u32
    }

    #[inline(never)]
    fn read_pointer_union(value: PointerBits) -> u32 {
        unsafe { *value.word }
    }

    #[inline(never)]
    fn direct_pointer_union_value() -> u32 {
        read_pointer_union(DIRECT_POINTER_UNION)
    }

    #[inline(never)]
    fn pointer_union_array_value(i: usize) -> u32 {
        read_pointer_union(POINTER_UNION_TABLE[i & 1])
    }

    #[inline(never)]
    fn fold_pointer_integer_union(value: PointerOrBits) -> u32 {
        let bits = unsafe { value.bits };
        (bits as u32)
            .wrapping_mul(257)
            .wrapping_add((bits >> 32) as u32)
    }

    #[inline(never)]
    fn direct_pointer_integer_union_value() -> u32 {
        fold_pointer_integer_union(DIRECT_POINTER_INTEGER_UNION)
    }

    #[inline(never)]
    fn pointer_integer_union_array_value(i: usize) -> u32 {
        fold_pointer_integer_union(POINTER_INTEGER_UNION_TABLE[i & 1])
    }

    #[inline(never)]
    fn read_bits_word(bits: Bits) -> u32 {
        unsafe { bits.word }
    }

    #[inline(never)]
    fn read_union_tuple(value: (u8, Bits)) -> u32 {
        let (tag, bits) = value;
        unsafe { (tag as u32).wrapping_add(bits.word) }
    }

    #[inline(never)]
    fn read_union_holder(value: UnionHolder) -> u32 {
        unsafe { (value.tag as u32).wrapping_add(value.value.word) }
    }

    #[inline(never)]
    fn read_partial_byte(value: PartialBits) -> u32 {
        unsafe { value.byte as u32 }
    }

    #[inline(never)]
    fn direct_union_value() -> u32 {
        read_bits_word(DIRECT_UNION)
    }

    #[inline(never)]
    fn union_array_value(i: usize) -> u32 {
        let bits = UNION_TABLE[i & 3];
        read_bits_word(bits)
    }

    #[inline(never)]
    fn union_tuple_value() -> u32 {
        read_union_tuple(UNION_TUPLE)
    }

    #[inline(never)]
    fn union_struct_value() -> u32 {
        read_union_holder(UNION_STRUCT)
    }

    #[inline(never)]
    fn partial_union_value() -> u32 {
        read_partial_byte(PARTIAL_UNION)
    }

    #[inline(never)]
    fn maybe_uninit_array_value(i: usize) -> u32 {
        unsafe { MAYBE_UNINIT_TABLE[i & 1].assume_init() }
    }

    #[inline(never)]
    fn initialized_union_constants_value(i: usize) -> u32 {
        direct_pointer_union_value()
            .wrapping_mul(257)
            .wrapping_add(pointer_union_array_value(i))
            .wrapping_mul(257)
            .wrapping_add(direct_pointer_integer_union_value())
            .wrapping_mul(257)
            .wrapping_add(pointer_integer_union_array_value(i))
            .wrapping_mul(257)
            .wrapping_add(direct_union_value())
            .wrapping_mul(257)
            .wrapping_add(union_array_value(i))
            .wrapping_mul(257)
            .wrapping_add(union_tuple_value())
            .wrapping_mul(257)
            .wrapping_add(union_struct_value())
            .wrapping_mul(257)
            .wrapping_add(partial_union_value())
            .wrapping_mul(257)
            .wrapping_add(maybe_uninit_array_value(i))
    }

    #[kernel]
    pub fn check_array_constants(
        mut out_f32: DisjointSlice<f32>,
        mut out_u32: DisjointSlice<u32>,
        mut out_enum: DisjointSlice<u32>,
        mut out_union: DisjointSlice<u32>,
    ) {
        let tid = thread::index_1d();
        let i = tid.get();

        if let Some(slot) = out_f32.get_mut(tid) {
            *slot = bare_array_value(i);
        }

        let tid_u32 = thread::index_1d();
        if let Some(slot) = out_u32.get_mut(tid_u32) {
            let nested = nested_array_value(i);
            let pointer = pointer_to_array_value(i);
            let pointer_tuple = pointer_tuple_array_value(i);
            let tuple = tuple_array_value(i);
            let nested_tuple = nested_tuple_array_value(i);
            let (direct_tag, direct_value) = direct_tuple_value();
            let direct = direct_tag as u32 + direct_value;
            let all_zst = all_zst_tuple_array_value(i);
            let reordered = reordered_tuple_array_value(i);
            let overaligned_zst = overaligned_zst_tuple_array_value(i);
            let padded_struct = padded_struct_array_value(i);
            let nested_struct = nested_struct_array_value(i);
            let padded_struct_ref = padded_struct_ref_value(i);
            let nested_struct_ref = nested_struct_ref_value(i);
            let packed_struct = packed_struct_array_value(i);
            let packed_struct_ref = packed_struct_ref_value(i);
            let zst_struct = zst_struct_array_value(i);

            *slot = nested
                .wrapping_mul(257)
                .wrapping_add(pointer)
                .wrapping_mul(257)
                .wrapping_add(pointer_tuple)
                .wrapping_mul(257)
                .wrapping_add(tuple)
                .wrapping_mul(257)
                .wrapping_add(nested_tuple)
                .wrapping_mul(257)
                .wrapping_add(direct)
                .wrapping_mul(257)
                .wrapping_add(all_zst)
                .wrapping_mul(257)
                .wrapping_add(reordered)
                .wrapping_mul(257)
                .wrapping_add(overaligned_zst)
                .wrapping_mul(257)
                .wrapping_add(padded_struct)
                .wrapping_mul(257)
                .wrapping_add(nested_struct)
                .wrapping_mul(257)
                .wrapping_add(padded_struct_ref)
                .wrapping_mul(257)
                .wrapping_add(nested_struct_ref)
                .wrapping_mul(257)
                .wrapping_add(packed_struct)
                .wrapping_mul(257)
                .wrapping_add(packed_struct_ref)
                .wrapping_mul(257)
                .wrapping_add(zst_struct);
        }

        let tid_enum = thread::index_1d();
        if let Some(slot) = out_enum.get_mut(tid_enum) {
            *slot = bare_enum_array_value(i);
        }

        let tid_union = thread::index_1d();
        if let Some(slot) = out_union.get_mut(tid_union) {
            *slot = initialized_union_constants_value(i);
        }
    }
}

fn fold_u64_for_union_test(bits: u64) -> u32 {
    (bits as u32)
        .wrapping_mul(257)
        .wrapping_add((bits >> 32) as u32)
}

fn expected_union(i: usize) -> u32 {
    let pointer = [11u32, 23][i & 1];
    let pointer_integer = [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210][i & 1];
    let table = [0x1122_3344u32, 0x5566_7788, 0x99aa_bbcc, 0xdead_beef][i & 3];
    let maybe = [0x1357_9bdfu32, 0x2468_ace0][i & 1];

    11u32
        .wrapping_mul(257)
        .wrapping_add(pointer)
        .wrapping_mul(257)
        .wrapping_add(fold_u64_for_union_test(0x1122_3344_5566_7788))
        .wrapping_mul(257)
        .wrapping_add(fold_u64_for_union_test(pointer_integer))
        .wrapping_mul(257)
        .wrapping_add(0x1122_3344)
        .wrapping_mul(257)
        .wrapping_add(table)
        .wrapping_mul(257)
        .wrapping_add(7u32.wrapping_add(0x1020_3040))
        .wrapping_mul(257)
        .wrapping_add(9u32.wrapping_add(0x0102_0304))
        .wrapping_mul(257)
        .wrapping_add(0x7f)
        .wrapping_mul(257)
        .wrapping_add(maybe)
}

fn expected_f32(i: usize) -> f32 {
    BARE_TABLE[i & 3]
}

fn expected_u32(i: usize) -> u32 {
    let row = i & 1;
    let col = (i / 2) % 3;
    let nested = NESTED_TABLE[row][col];
    let pointer = POINTER_TABLE[i & 3];

    let (pointer_ref, pointer_flag) = POINTER_TUPLE_TABLE[i & 1];
    let pointer_tuple = *pointer_ref + pointer_flag as u32;

    let (is_high, side) = TUPLE_TABLE[i % 6];
    let tuple = (side as u32) * 10 + (is_high as u32);

    let ((tag, ()), value) = NESTED_TUPLE_TABLE[i & 1];
    let nested_tuple = tag as u32 + value;

    let (direct_tag, direct_value) = DIRECT_TUPLE;
    let direct = direct_tag as u32 + direct_value;

    let (((), ()), all_zst) = ALL_ZST_TUPLE_TABLE[i & 1];

    let (byte, word, wide) = REORDERED_TUPLE_TABLE[i & 1];
    let reordered = (byte as u32)
        .wrapping_mul(257)
        .wrapping_add(word)
        .wrapping_mul(257)
        .wrapping_add(wide as u32)
        .wrapping_mul(257)
        .wrapping_add((wide >> 32) as u32);

    let (_, overaligned_zst) = OVERALIGNED_ZST_TUPLE_TABLE[i & 1];

    let padded_value = PADDED_STRUCT_TABLE[i & 1];
    let padded_struct = (padded_value.tag as u32)
        .wrapping_mul(257)
        .wrapping_add(padded_value.value);

    let nested_value = NESTED_STRUCT_TABLE[i & 1];
    let nested_struct = (nested_value.inner.tag as u32)
        .wrapping_mul(257)
        .wrapping_add(nested_value.inner.value)
        .wrapping_mul(257)
        .wrapping_add(nested_value.wide as u32)
        .wrapping_mul(257)
        .wrapping_add((nested_value.wide >> 32) as u32);

    let padded_struct_ref = padded_struct;
    let nested_struct_ref = nested_struct;

    let packed_value = PACKED_STRUCT_TABLE[i & 1];
    let packed_struct = (packed_value.tag as u32)
        .wrapping_mul(257)
        .wrapping_add(packed_value.value);
    let packed_struct_ref = packed_struct;

    let zst_value = ZST_STRUCT_TABLE[i & 1];
    let zst_struct = ((&zst_value as *const ZstStruct as usize) & 31) as u32;

    nested
        .wrapping_mul(257)
        .wrapping_add(pointer)
        .wrapping_mul(257)
        .wrapping_add(pointer_tuple)
        .wrapping_mul(257)
        .wrapping_add(tuple)
        .wrapping_mul(257)
        .wrapping_add(nested_tuple)
        .wrapping_mul(257)
        .wrapping_add(direct)
        .wrapping_mul(257)
        .wrapping_add(all_zst)
        .wrapping_mul(257)
        .wrapping_add(reordered)
        .wrapping_mul(257)
        .wrapping_add(overaligned_zst as u32)
        .wrapping_mul(257)
        .wrapping_add(padded_struct)
        .wrapping_mul(257)
        .wrapping_add(nested_struct)
        .wrapping_mul(257)
        .wrapping_add(padded_struct_ref)
        .wrapping_mul(257)
        .wrapping_add(nested_struct_ref)
        .wrapping_mul(257)
        .wrapping_add(packed_struct)
        .wrapping_mul(257)
        .wrapping_add(packed_struct_ref)
        .wrapping_mul(257)
        .wrapping_add(zst_struct)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== array_constants regression ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = 24;
    let mut out_f32 = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut out_u32 = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    let mut out_enum = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    let mut out_union = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: this is a 1D launch and the kernel bounds-checks each output
    // access against the corresponding slice length.
    unsafe {
        module.check_array_constants(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out_f32,
            &mut out_u32,
            &mut out_enum,
            &mut out_union,
        )
    }?;

    let got_f32 = out_f32.to_host_vec(&stream)?;
    let got_u32 = out_u32.to_host_vec(&stream)?;
    let got_enum = out_enum.to_host_vec(&stream)?;
    let got_union = out_union.to_host_vec(&stream)?;

    let mut failures = 0usize;
    for i in 0..N {
        let want_f32 = expected_f32(i);
        if got_f32[i] != want_f32 {
            println!(
                "FAIL bare array tid={i}: got={} expected={}",
                got_f32[i], want_f32
            );
            failures += 1;
        }

        let want_u32 = expected_u32(i);
        if got_u32[i] != want_u32 {
            println!(
                "FAIL nested/pointer array tid={i}: got={} expected={}",
                got_u32[i], want_u32
            );
            failures += 1;
        }

        let want_enum = EXPECTED_BARE_ENUM[i % EXPECTED_BARE_ENUM.len()];
        if got_enum[i] != want_enum {
            println!(
                "FAIL bare enum array tid={i}: got={} expected={want_enum}",
                got_enum[i]
            );
            failures += 1;
        }

        let want_union = expected_union(i);
        if got_union[i] != want_union {
            println!(
                "FAIL initialized union constants tid={i}: got={:#x} expected={want_union:#x}",
                got_union[i]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "array_constants: PASS ({N} threads; primitive, enum, initialized union/MaybeUninit, pointer-union provenance, relocation-free pointer/integer unions, padded/reordered/over-aligned tuple, nested/equal-offset ZST tuple, padded/nested/packed/ZST struct, pointer-to-array including packed struct references, and tuple-array static-pointer constants)"
        );
        Ok(())
    } else {
        println!("array_constants: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
