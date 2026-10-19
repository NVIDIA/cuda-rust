/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host/device ABI regression for basic and nested niche-encoded enums.
//!
//! The kernel reads enum bytes written by the host, then custom MIR
//! `SetDiscriminant` changes the input values to `None`. It also constructs an
//! `Option<bool>` on the device for the host to read back. `MaybeWrapper` is
//! the important later-field case: its niche is `Wrapper::nz`, at byte 4
//! rather than byte 0.

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use core::num::NonZeroU32;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Basic(Option<NonZeroU32>);

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Boolean(Option<bool>);

/// A standalone Rust `bool` is an SSA `i1`, but its payload storage is one
/// complete byte after this enum's four-byte direct tag. Keeping this in the
/// real example prevents the enum slot map from ever using `i1` as a physical
/// aggregate field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectBoolean {
    Value(bool),
    Empty,
}

/// rustc uses `B` as this enum's ordinary payload variant. `A` and `C` are
/// encoded in invalid `bool` values in the same byte; the niche-variant range
/// still spans source indices `A..=C`, so `B` lies inside that range even
/// though its `false`/`true` payload bytes remain the untagged representation.
/// Consequently `A` is byte 2, the range slot for `B` (byte 3) is skipped,
/// and `C` is byte 4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InteriorBoolean {
    A,
    B(bool),
    C,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Wrapper {
    pad: u32,
    nz: NonZeroU32,
}

#[derive(Clone, Copy)]
enum MaybeWrapper {
    None,
    Some(Wrapper),
}

/// A bool nested inside an aggregate payload. The niche carrier is the
/// bool's byte (at offset 4, after the u32): `None` stores the invalid bool
/// value 2 there. Device construction must canonicalize that physical byte
/// through the aggregate's byte-faithful twin, not store a bare `i1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Flagged {
    pad: u32,
    flag: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaybeFlagged {
    None,
    Some(Flagged),
}

// These values are copied as bytes to device memory. This example verifies
// that cuda-oxide now gives those bytes exactly the same meaning as rustc.
unsafe impl DeviceCopy for Basic {}
unsafe impl DeviceCopy for Boolean {}
unsafe impl DeviceCopy for DirectBoolean {}
unsafe impl DeviceCopy for InteriorBoolean {}
unsafe impl DeviceCopy for Wrapper {}
unsafe impl DeviceCopy for MaybeWrapper {}
unsafe impl DeviceCopy for Flagged {}
unsafe impl DeviceCopy for MaybeFlagged {}

// Import-only probes for rustc's Empty and Single layout forms. Keeping them
// in a kernel signature ensures the real importer sees zero- and one-variant
// enums; no impossible value is ever constructed or dereferenced.
pub enum Never {}
pub enum Only {
    Value = 1_000,
}
pub enum Impossible {
    Value(u32, Never),
}
pub enum ImpossibleMany {
    First(Never),
    Second(Never),
}
#[repr(i8)]
#[derive(Clone, Copy)]
pub enum Negative {
    Below = -1,
    Zero = 0,
}

unsafe impl DeviceCopy for Negative {}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_basic_none(value: &mut Option<NonZeroU32>) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_nested_none(value: &mut MaybeWrapper) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_boolean_none(value: &mut Option<bool>) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_pointer_none(value: &mut Option<&u32>) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_tuple_pointer_none(value: &mut Option<(u32, u32, &u32)>) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_flagged_none(value: &mut MaybeFlagged) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    // Each buffer is a separate host/device ABI case. Bundling them would
    // change the kernel parameter shapes this regression is meant to test.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn niche_roundtrip(
        mut basic: DisjointSlice<Basic>,
        mut boolean: DisjointSlice<Boolean>,
        mut constructed_boolean: DisjointSlice<Boolean>,
        mut direct_boolean: DisjointSlice<DirectBoolean>,
        mut constructed_direct_boolean: DisjointSlice<DirectBoolean>,
        mut observed_direct_boolean: DisjointSlice<u8>,
        mut interior_boolean: DisjointSlice<InteriorBoolean>,
        mut constructed_interior_boolean: DisjointSlice<InteriorBoolean>,
        mut observed_interior_boolean: DisjointSlice<u8>,
        mut pointer_niche_cleared: DisjointSlice<u8>,
        mut tuple_niche_checked: DisjointSlice<u8>,
        mut nested_boolean: DisjointSlice<MaybeFlagged>,
        mut constructed_nested_boolean: DisjointSlice<MaybeFlagged>,
        mut observed_nested_boolean: DisjointSlice<u32>,
        mut nested: DisjointSlice<MaybeWrapper>,
        mut negative: DisjointSlice<Negative>,
        mut observed: DisjointSlice<u32>,
    ) {
        let Some(basic) = basic.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(nested) = nested.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(boolean) = boolean.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(constructed_boolean) = constructed_boolean.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(direct_boolean) = direct_boolean.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(constructed_direct_boolean) =
            constructed_direct_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(observed_direct_boolean) = observed_direct_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(interior_boolean) = interior_boolean.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(constructed_interior_boolean) =
            constructed_interior_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(observed_interior_boolean) = observed_interior_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(pointer_niche_cleared) = pointer_niche_cleared.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(tuple_niche_checked) = tuple_niche_checked.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(nested_boolean) = nested_boolean.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(constructed_nested_boolean) =
            constructed_nested_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(observed_nested_boolean) = observed_nested_boolean.get_mut(thread::index_1d())
        else {
            return;
        };
        let Some(negative) = negative.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(out) = observed.get_mut(thread::index_1d()) else {
            return;
        };

        let basic_value = match basic.0 {
            Some(value) => value.get(),
            None => 0,
        };
        let nested_value = match *nested {
            MaybeWrapper::Some(value) => value.pad ^ value.nz.get().rotate_left(7),
            MaybeWrapper::None => u32::MAX,
        };
        let boolean_value = match boolean.0 {
            None => 0x10,
            Some(false) => 0x20,
            Some(true) => 0x40,
        };
        *observed_direct_boolean = match *direct_boolean {
            DirectBoolean::Value(false) => 0xD0,
            DirectBoolean::Value(true) => 0xD1,
            DirectBoolean::Empty => 0xDE,
        };
        *observed_interior_boolean = match *interior_boolean {
            InteriorBoolean::A => 0xA0,
            InteriorBoolean::B(false) => 0xB0,
            InteriorBoolean::B(true) => 0xB1,
            InteriorBoolean::C => 0xC0,
        };
        // This cast must sign-extend the physical i8 discriminant. In
        // particular, `Negative::Below` must become -1, not 255.
        let negative_value = *negative as i32;
        *out = basic_value ^ nested_value ^ boolean_value ^ (negative_value as u32);

        // Option<bool> has an i8 memory carrier (`0`, `1`, and the niche `2`)
        // even though an ordinary bool is i1 in LLVM. Returning a newly
        // constructed value exercises that physical i8 representation in the
        // opposite direction too.
        constructed_boolean.0 = Some(thread::index_1d().get() % 2 == 0);
        *constructed_direct_boolean = match thread::index_1d().get() % 3 {
            0 => DirectBoolean::Value(false),
            1 => DirectBoolean::Value(true),
            _ => DirectBoolean::Empty,
        };
        *constructed_interior_boolean = match thread::index_1d().get() % 4 {
            0 => InteriorBoolean::A,
            1 => InteriorBoolean::B(false),
            2 => InteriorBoolean::B(true),
            _ => InteriorBoolean::C,
        };

        // A reference is represented by a generic pointer. `None` must write
        // a null pointer through that physical carrier, not an integer-shaped
        // synthetic tag.
        let local = 0xCAFE_BABEu32;
        let mut pointer_niche = Some(&local);
        force_pointer_none(&mut pointer_niche);
        *pointer_niche_cleared = pointer_niche.is_none() as u8;

        // A multi-field tuple payload whose pointer field is the niche
        // carrier. rustc reorders the tuple (pointer first in memory) and
        // the recorded tuple field offsets carry that placement to the
        // device, so construction, payload reads, and a niche
        // SetDiscriminant all use rustc's real layout.
        let referent = 0x1111_2222u32;
        let mut tuple_niche: Option<(u32, u32, &u32)> = Some((5, 7, &referent));
        let tuple_sum = match tuple_niche {
            Some((a, b, r)) => a + b + *r,
            None => 0,
        };
        force_tuple_pointer_none(&mut tuple_niche);
        *tuple_niche_checked =
            u8::from(tuple_sum == 0x1111_2222 + 12) | (u8::from(tuple_niche.is_none()) << 1);

        // A bool nested inside an aggregate payload, with the niche in the
        // bool's byte. Decoding reads host-written bytes, construction must
        // write the canonical 0/1 flag byte through the aggregate's
        // byte-faithful twin, and SetDiscriminant(None) must write the
        // invalid bool value 2 into that same byte.
        *observed_nested_boolean = match *nested_boolean {
            MaybeFlagged::Some(value) => value.pad ^ u32::from(value.flag),
            MaybeFlagged::None => u32::MAX,
        };
        *constructed_nested_boolean = MaybeFlagged::Some(Flagged {
            pad: thread::index_1d().get() as u32,
            flag: thread::index_1d().get() % 2 == 0,
        });
        force_flagged_none(nested_boolean);

        force_basic_none(&mut basic.0);
        force_boolean_none(&mut boolean.0);
        force_nested_none(nested);
    }

    #[kernel]
    pub unsafe fn enum_layout_import_probe(
        _only: *const Only,
        _never: *const Never,
        _impossible: *const Impossible,
        _impossible_many: *const ImpossibleMany,
    ) {
    }
}

fn main() {
    assert_eq!(std::mem::size_of::<Basic>(), 4);
    assert_eq!(std::mem::size_of::<Boolean>(), 1);
    assert_eq!(std::mem::size_of::<DirectBoolean>(), 8);
    assert_eq!(std::mem::align_of::<DirectBoolean>(), 4);
    assert_eq!(std::mem::size_of::<InteriorBoolean>(), 1);
    assert_eq!(std::mem::size_of::<Wrapper>(), 8);
    assert_eq!(std::mem::size_of::<MaybeWrapper>(), 8);
    // Three-field tuple payload with a pointer niche: the pointer is the
    // carrier and rustc reorders it to byte 0.
    assert_eq!(std::mem::size_of::<Option<(u32, u32, &u32)>>(), 16);
    // Nested-bool niche: no separate tag; `None` is the invalid bool value 2
    // in the flag byte at offset 4.
    assert_eq!(std::mem::size_of::<MaybeFlagged>(), 8);
    let flagged_none_bytes =
        unsafe { std::mem::transmute::<MaybeFlagged, [u8; 8]>(MaybeFlagged::None) };
    assert_eq!(flagged_none_bytes[4], 2);

    // On this pinned rustc, `B(false)`/`B(true)` keep the valid bool carrier
    // bytes 0/1. `A` starts the niche range at 2; `B`'s position would be 3,
    // but is skipped because `B` is the untagged variant, so `C` becomes 4.
    // These assertions make the source-level regression's physical contract
    // explicit before the same bytes cross the host/device boundary.
    let interior_boolean_encodings = [
        unsafe { std::mem::transmute::<InteriorBoolean, u8>(InteriorBoolean::A) },
        unsafe { std::mem::transmute::<InteriorBoolean, u8>(InteriorBoolean::B(false)) },
        unsafe { std::mem::transmute::<InteriorBoolean, u8>(InteriorBoolean::B(true)) },
        unsafe { std::mem::transmute::<InteriorBoolean, u8>(InteriorBoolean::C) },
    ];
    assert_eq!(interior_boolean_encodings, [2, 0, 1, 4]);
    assert!(
        !interior_boolean_encodings.contains(&3),
        "the niche position corresponding to untagged B must stay unused"
    );

    // For the inhabited variant every byte is initialized, so this is also a
    // host-side proof that the nested NonZero carrier is the later u32 field.
    let layout_probe = MaybeWrapper::Some(Wrapper {
        pad: 0x1122_3344,
        nz: NonZeroU32::new(0x5566_7788).unwrap(),
    });
    let probe_bytes = unsafe { std::mem::transmute::<MaybeWrapper, [u8; 8]>(layout_probe) };
    assert_eq!(&probe_bytes[4..8], &0x5566_7788u32.to_ne_bytes());

    const N: usize = 64;
    let basic_host = (0..N)
        .map(|index| {
            Basic(if index % 3 == 0 {
                None
            } else {
                NonZeroU32::new(index as u32 + 1)
            })
        })
        .collect::<Vec<_>>();
    let nested_host = (0..N)
        .map(|index| {
            if index % 5 == 0 {
                MaybeWrapper::None
            } else {
                MaybeWrapper::Some(Wrapper {
                    pad: 0xA500_0000 | index as u32,
                    nz: NonZeroU32::new(index as u32 + 17).unwrap(),
                })
            }
        })
        .collect::<Vec<_>>();
    let boolean_host = (0..N)
        .map(|index| {
            Boolean(match index % 3 {
                0 => None,
                1 => Some(false),
                _ => Some(true),
            })
        })
        .collect::<Vec<_>>();
    let direct_boolean_host = (0..N)
        .map(|index| match index % 3 {
            0 => DirectBoolean::Value(false),
            1 => DirectBoolean::Value(true),
            _ => DirectBoolean::Empty,
        })
        .collect::<Vec<_>>();
    let nested_boolean_host = (0..N)
        .map(|index| match index % 3 {
            0 => MaybeFlagged::None,
            1 => MaybeFlagged::Some(Flagged {
                pad: 0x5A00_0000 | index as u32,
                flag: false,
            }),
            _ => MaybeFlagged::Some(Flagged {
                pad: 0x5A00_0000 | index as u32,
                flag: true,
            }),
        })
        .collect::<Vec<_>>();
    let expected_nested_boolean = nested_boolean_host
        .iter()
        .map(|value| match value {
            MaybeFlagged::Some(value) => value.pad ^ u32::from(value.flag),
            MaybeFlagged::None => u32::MAX,
        })
        .collect::<Vec<u32>>();
    let expected_direct_boolean = direct_boolean_host
        .iter()
        .map(|value| match value {
            DirectBoolean::Value(false) => 0xD0,
            DirectBoolean::Value(true) => 0xD1,
            DirectBoolean::Empty => 0xDE,
        })
        .collect::<Vec<u8>>();
    let negative_host = (0..N)
        .map(|index| {
            if index % 2 == 0 {
                Negative::Below
            } else {
                Negative::Zero
            }
        })
        .collect::<Vec<_>>();
    let interior_boolean_host = (0..N)
        .map(|index| match index % 4 {
            0 => InteriorBoolean::A,
            1 => InteriorBoolean::B(false),
            2 => InteriorBoolean::B(true),
            _ => InteriorBoolean::C,
        })
        .collect::<Vec<_>>();
    let expected_interior_boolean = interior_boolean_host
        .iter()
        .map(|value| match value {
            InteriorBoolean::A => 0xA0,
            InteriorBoolean::B(false) => 0xB0,
            InteriorBoolean::B(true) => 0xB1,
            InteriorBoolean::C => 0xC0,
        })
        .collect::<Vec<u8>>();
    let expected = basic_host
        .iter()
        .zip(&boolean_host)
        .zip(&nested_host)
        .zip(&negative_host)
        .map(|(((basic, boolean), nested), negative)| {
            let basic = basic.0.map_or(0, NonZeroU32::get);
            let boolean = match boolean.0 {
                None => 0x10,
                Some(false) => 0x20,
                Some(true) => 0x40,
            };
            let nested = match nested {
                MaybeWrapper::Some(value) => value.pad ^ value.nz.get().rotate_left(7),
                MaybeWrapper::None => u32::MAX,
            };
            let negative = *negative as i32;
            basic ^ nested ^ boolean ^ (negative as u32)
        })
        .collect::<Vec<_>>();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let mut basic_device = DeviceBuffer::from_host(&stream, &basic_host).unwrap();
    let mut boolean_device = DeviceBuffer::from_host(&stream, &boolean_host).unwrap();
    let mut constructed_boolean_device = DeviceBuffer::zeroed(&stream, N).unwrap();
    let mut direct_boolean_device = DeviceBuffer::from_host(&stream, &direct_boolean_host).unwrap();
    let mut constructed_direct_boolean_device = DeviceBuffer::zeroed(&stream, N).unwrap();
    let mut observed_direct_boolean_device = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    let mut interior_boolean_device =
        DeviceBuffer::from_host(&stream, &interior_boolean_host).unwrap();
    let mut constructed_interior_boolean_device = DeviceBuffer::zeroed(&stream, N).unwrap();
    let mut observed_interior_boolean_device = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    let mut pointer_niche_cleared_device = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    let mut tuple_niche_checked_device = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    let mut nested_boolean_device = DeviceBuffer::from_host(&stream, &nested_boolean_host).unwrap();
    let mut constructed_nested_boolean_device = DeviceBuffer::zeroed(&stream, N).unwrap();
    let mut observed_nested_boolean_device = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut nested_device = DeviceBuffer::from_host(&stream, &nested_host).unwrap();
    let mut negative_device = DeviceBuffer::from_host(&stream, &negative_host).unwrap();
    let mut observed_device = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    unsafe {
        module
            .niche_roundtrip(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &mut basic_device,
                &mut boolean_device,
                &mut constructed_boolean_device,
                &mut direct_boolean_device,
                &mut constructed_direct_boolean_device,
                &mut observed_direct_boolean_device,
                &mut interior_boolean_device,
                &mut constructed_interior_boolean_device,
                &mut observed_interior_boolean_device,
                &mut pointer_niche_cleared_device,
                &mut tuple_niche_checked_device,
                &mut nested_boolean_device,
                &mut constructed_nested_boolean_device,
                &mut observed_nested_boolean_device,
                &mut nested_device,
                &mut negative_device,
                &mut observed_device,
            )
            .expect("Kernel launch failed");
    }

    let observed = observed_device.to_host_vec(&stream).unwrap();
    let basic_after = basic_device.to_host_vec(&stream).unwrap();
    let boolean_after = boolean_device.to_host_vec(&stream).unwrap();
    let constructed_boolean_after = constructed_boolean_device.to_host_vec(&stream).unwrap();
    let constructed_direct_boolean_after = constructed_direct_boolean_device
        .to_host_vec(&stream)
        .unwrap();
    let observed_direct_boolean = observed_direct_boolean_device.to_host_vec(&stream).unwrap();
    let constructed_interior_boolean_after = constructed_interior_boolean_device
        .to_host_vec(&stream)
        .unwrap();
    let observed_interior_boolean = observed_interior_boolean_device
        .to_host_vec(&stream)
        .unwrap();
    let pointer_niche_cleared = pointer_niche_cleared_device.to_host_vec(&stream).unwrap();
    let tuple_niche_checked = tuple_niche_checked_device.to_host_vec(&stream).unwrap();
    let nested_boolean_after = nested_boolean_device.to_host_vec(&stream).unwrap();
    let constructed_nested_boolean = constructed_nested_boolean_device
        .to_host_vec(&stream)
        .unwrap();
    let observed_nested_boolean = observed_nested_boolean_device.to_host_vec(&stream).unwrap();
    let nested_after = nested_device.to_host_vec(&stream).unwrap();
    assert_eq!(
        observed, expected,
        "device must decode host-written enum bytes"
    );
    assert!(basic_after.iter().all(|value| value.0.is_none()));
    assert!(boolean_after.iter().all(|value| value.0.is_none()));
    assert!(
        constructed_boolean_after
            .iter()
            .enumerate()
            .all(|(index, value)| value.0 == Some(index % 2 == 0))
    );
    assert_eq!(observed_direct_boolean, expected_direct_boolean);
    assert_eq!(
        constructed_direct_boolean_after, direct_boolean_host,
        "device constructions must preserve a direct-tagged bool payload"
    );
    assert_eq!(observed_interior_boolean, expected_interior_boolean);
    assert_eq!(
        constructed_interior_boolean_after, interior_boolean_host,
        "device constructions must preserve A/B(false)/B(true)/C"
    );
    assert!(pointer_niche_cleared.iter().all(|value| *value == 1));
    assert!(
        tuple_niche_checked.iter().all(|value| *value == 3),
        "tuple pointer-niche payload reads and SetDiscriminant(None) must both succeed on device"
    );
    assert_eq!(
        observed_nested_boolean, expected_nested_boolean,
        "device must decode host-written nested-bool niche bytes"
    );
    assert!(
        nested_boolean_after
            .iter()
            .all(|value| matches!(value, MaybeFlagged::None)),
        "SetDiscriminant(None) must write the invalid bool value into the nested flag byte"
    );
    assert!(
        constructed_nested_boolean
            .iter()
            .enumerate()
            .all(|(index, value)| {
                *value
                    == MaybeFlagged::Some(Flagged {
                        pad: index as u32,
                        flag: index % 2 == 0,
                    })
            }),
        "device constructions must write canonical nested bool payload bytes"
    );
    assert!(
        nested_after
            .iter()
            .all(|value| matches!(value, MaybeWrapper::None))
    );

    println!(
        "set_discriminant_niche: PASS ({N} host->device decodes and device->host enum writes)"
    );
}
