/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end ABI regression coverage for packed aggregates.
//!
//! This example exercises nine paths that must agree on the same rustc byte
//! layout:
//!
//! - packed structs passed by value across the host -> kernel boundary;
//! - packed structs passed to and returned from an internal device helper;
//! - packed structs containing one shared pointer returned from an internal
//!   device helper through a target-stable generic-pointer carrier;
//! - direct field projections from that one-shared-pointer value after local
//!   materialization through the same target-stable carrier;
//! - packed structs containing multiple direct shared-pointer leaves crossing
//!   the same internal device ABI;
//! - packed structs containing recursively nested shared-pointer leaves crossing
//!   the same internal device ABI;
//! - packed structs containing bounded arrays of shared-pointer leaves crossing
//!   the same internal device ABI;
//! - whole-value stores of packed structs to device memory;
//! - whole-value loads of packed structs from device memory.
//!
//! `Packed1` has no interior padding and occupies 5 bytes. `Packed2` has one
//! explicit padding byte between `a` and `b`, so it occupies 6 bytes and places
//! `b` at byte offset 2. The padding byte itself is intentionally not checked:
//! Rust does not guarantee a stable value for padding bytes.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{SharedArray, cuda_module, device, kernel};

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Packed1 {
    pub a: u8,
    pub b: u32,
}

#[repr(C, packed(2))]
#[derive(Clone, Copy)]
pub struct Packed2 {
    pub a: u8,
    pub b: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PackedShared {
    pub tag: u8,
    pub ptr: *mut SharedArray<u32, 1>,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PackedSharedPair {
    pub tag: u8,
    pub left: *mut SharedArray<u32, 1>,
    pub right: *mut SharedArray<u32, 1>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SharedPair {
    pub left: *mut SharedArray<u32, 1>,
    pub right: *mut SharedArray<u32, 1>,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PackedNestedShared {
    pub tag: u8,
    pub pair: SharedPair,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PackedSharedArray {
    pub tag: u8,
    pub ptrs: [*mut SharedArray<u32, 1>; 2],
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn transform_packed1(value: Packed1) -> Packed1 {
        let a = value.a;
        let b = value.b;
        Packed1 {
            a: a.wrapping_add(1),
            b: b.wrapping_add(0x0102_0304),
        }
    }

    #[inline(never)]
    fn transform_packed2(value: Packed2) -> Packed2 {
        let a = value.a;
        let b = value.b;
        Packed2 {
            a: a.wrapping_add(2),
            b: b.wrapping_add(0x1112_1314),
        }
    }

    #[inline(never)]
    #[device]
    fn bounce_packed_shared(value: PackedShared) -> PackedShared {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_packed_shared_pair(value: PackedSharedPair) -> PackedSharedPair {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_packed_nested_shared(value: PackedNestedShared) -> PackedNestedShared {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_packed_shared_array(value: PackedSharedArray) -> PackedSharedArray {
        value
    }

    #[inline(never)]
    #[device]
    unsafe fn consume_packed_shared(
        value: PackedShared,
        shared: *mut SharedArray<u32, 1>,
        out: *mut u32,
    ) {
        // These two projections force the returned packed-AS3 value through a
        // compiler-owned local. The slot is physically <{ i8, p0 }>; loading
        // `ptr` reconstructs the semantic AS3 pointer explicitly at the memory
        // boundary before it is dereferenced.
        let tag = value.tag;
        let round_tripped = value.ptr;
        unsafe {
            (&mut *round_tripped)[0] = (&*round_tripped)[0].wrapping_add(0x0102_0304);
            out.write(u32::from(tag.wrapping_add(1)));
            out.add(1).write((&*shared)[0]);
        }
    }

    #[inline(never)]
    #[device]
    unsafe fn consume_packed_shared_pair(
        _value: PackedSharedPair,
        left: *mut SharedArray<u32, 1>,
        right: *mut SharedArray<u32, 1>,
        out: *mut u32,
    ) {
        unsafe {
            (&mut *left)[0] = (&*left)[0].wrapping_add(0x0102_0304);
            (&mut *right)[0] = (&*right)[0].wrapping_add(0x0203_0405);
            out.write(0x32);
            out.add(1).write((&*left)[0]);
            out.add(2).write((&*right)[0]);
        }
    }

    #[inline(never)]
    #[device]
    unsafe fn consume_packed_nested_shared(
        _value: PackedNestedShared,
        left: *mut SharedArray<u32, 1>,
        right: *mut SharedArray<u32, 1>,
        out: *mut u32,
    ) {
        unsafe {
            (&mut *left)[0] = (&*left)[0].wrapping_add(0x0304_0506);
            (&mut *right)[0] = (&*right)[0].wrapping_add(0x0405_0607);
            out.write(0x42);
            out.add(1).write((&*left)[0]);
            out.add(2).write((&*right)[0]);
        }
    }

    #[inline(never)]
    #[device]
    unsafe fn consume_packed_shared_array(
        _value: PackedSharedArray,
        left: *mut SharedArray<u32, 1>,
        right: *mut SharedArray<u32, 1>,
        out: *mut u32,
    ) {
        unsafe {
            (&mut *left)[0] = (&*left)[0].wrapping_add(0x0506_0708);
            (&mut *right)[0] = (&*right)[0].wrapping_add(0x0607_0809);
            out.write(0x52);
            out.add(1).write((&*left)[0]);
            out.add(2).write((&*right)[0]);
        }
    }

    #[kernel]
    pub unsafe fn packed1(value: Packed1, out: *mut u32) {
        let value = transform_packed1(value);
        let a = value.a;
        let b = value.b;

        // SAFETY: the host provides two writable u32 elements in `out`.
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn packed2(value: Packed2, out: *mut u32) {
        let value = transform_packed2(value);
        let a = value.a;
        let b = value.b;

        // SAFETY: the host provides two writable u32 elements in `out`.
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn packed_shared(out: *mut u32) {
        static mut SHARED: SharedArray<u32, 1> = SharedArray::UNINIT;

        let raw = &raw mut SHARED;
        unsafe { (&mut *raw)[0] = 0x1020_3040 };

        let value = bounce_packed_shared(PackedShared {
            tag: 0x21,
            ptr: raw,
        });

        // The callee now projects both fields from the returned packed value.
        // That forces the narrow #1036 local-carrier path while the independent
        // raw pointer keeps the reconstructed AS3 pointer's write observable.
        unsafe { consume_packed_shared(value, raw, out) };
    }

    #[kernel]
    pub unsafe fn packed_shared_pair(out: *mut u32) {
        static mut LEFT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RIGHT: SharedArray<u32, 1> = SharedArray::UNINIT;

        let left = &raw mut LEFT;
        let right = &raw mut RIGHT;
        unsafe {
            (&mut *left)[0] = 0x1020_3040;
            (&mut *right)[0] = 0x2030_4050;
        }

        let value = bounce_packed_shared_pair(PackedSharedPair {
            tag: 0x31,
            left,
            right,
        });

        // The returned packed value contains two direct AS3 leaves. Keep it in
        // SSA and pass it whole into another device helper; recursive/multi-leaf
        // carrier-local projection support remains deliberately out of scope.
        unsafe { consume_packed_shared_pair(value, left, right, out) };
    }

    #[kernel]
    pub unsafe fn packed_nested_shared(out: *mut u32) {
        static mut LEFT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RIGHT: SharedArray<u32, 1> = SharedArray::UNINIT;

        let left = &raw mut LEFT;
        let right = &raw mut RIGHT;
        unsafe {
            (&mut *left)[0] = 0x3040_5060;
            (&mut *right)[0] = 0x4050_6070;
        }

        let value = bounce_packed_nested_shared(PackedNestedShared {
            tag: 0x41,
            pair: SharedPair { left, right },
        });

        // The AS3 leaves live under a nested aggregate. Keep the packed outer
        // value in SSA so this still exercises only the internal ABI carrier
        // generalization, not recursive packed local storage.
        unsafe { consume_packed_nested_shared(value, left, right, out) };
    }

    #[kernel]
    pub unsafe fn packed_shared_array(out: *mut u32) {
        static mut LEFT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RIGHT: SharedArray<u32, 1> = SharedArray::UNINIT;

        let left = &raw mut LEFT;
        let right = &raw mut RIGHT;
        unsafe {
            (&mut *left)[0] = 0x5060_7080;
            (&mut *right)[0] = 0x6070_8090;
        }

        let value = bounce_packed_shared_array(PackedSharedArray {
            tag: 0x51,
            ptrs: [left, right],
        });

        // The two AS3 leaves live inside one fixed array. Keep the packed value
        // in SSA so the return boundary rebuilds the bounded target-stable
        // carrier without widening the narrow local-storage lane.
        unsafe { consume_packed_shared_array(value, left, right, out) };
    }

    #[kernel]
    pub unsafe fn store_packed1(value: Packed1, dst: *mut Packed1) {
        // SAFETY: `dst` points to device storage large enough for one Packed1.
        unsafe { dst.write(value) };
    }

    #[kernel]
    pub unsafe fn load_packed1(src: *const Packed1, out: *mut u32) {
        // SAFETY: `src` points to one initialized Packed1 and `out` has two u32s.
        let value = unsafe { src.read() };
        let a = value.a;
        let b = value.b;
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn store_packed2(value: Packed2, dst: *mut Packed2) {
        // SAFETY: `dst` points to device storage large enough and sufficiently
        // aligned for one Packed2. CUDA allocations satisfy this alignment.
        unsafe { dst.write(value) };
    }

    #[kernel]
    pub unsafe fn load_packed2(src: *const Packed2, out: *mut u32) {
        // SAFETY: `src` points to one initialized Packed2 and `out` has two u32s.
        let value = unsafe { src.read() };
        let a = value.a;
        let b = value.b;
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }
}

fn entry_header<'a>(ptx: &'a str, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let marker = format!(".visible .entry {name}(");
    let start = ptx
        .find(&marker)
        .ok_or_else(|| format!("missing PTX entry `{name}`"))?;
    let rest = &ptx[start..];
    let end = rest
        .find('{')
        .ok_or_else(|| format!("unterminated PTX entry header `{name}`"))?;
    Ok(&rest[..end])
}

fn require_aggregate_parameter(
    ptx: &str,
    name: &str,
    byte_size: usize,
    abi_align: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = entry_header(ptx, name)?;
    let size_token = format!("[{byte_size}]");
    let align_token = format!(".param .align {abi_align} .b8");

    if !header.contains(&size_token) || !header.contains(&align_token) {
        return Err(format!(
            "kernel `{name}` does not expose the expected {byte_size}-byte, align-{abi_align} aggregate parameter:\n{header}"
        )
        .into());
    }

    Ok(())
}

fn verify_generated_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("packed_aggregate_abi.ptx");
    let ptx = std::fs::read_to_string(&path)?;

    require_aggregate_parameter(
        &ptx,
        "packed1",
        core::mem::size_of::<Packed1>(),
        core::mem::align_of::<Packed1>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "packed2",
        core::mem::size_of::<Packed2>(),
        core::mem::align_of::<Packed2>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "store_packed1",
        core::mem::size_of::<Packed1>(),
        core::mem::align_of::<Packed1>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "store_packed2",
        core::mem::size_of::<Packed2>(),
        core::mem::align_of::<Packed2>(),
    )?;

    Ok(())
}

fn assert_host_layout() {
    assert_eq!(core::mem::size_of::<Packed1>(), 5);
    assert_eq!(core::mem::align_of::<Packed1>(), 1);
    assert_eq!(core::mem::offset_of!(Packed1, a), 0);
    assert_eq!(core::mem::offset_of!(Packed1, b), 1);

    assert_eq!(core::mem::size_of::<Packed2>(), 6);
    assert_eq!(core::mem::align_of::<Packed2>(), 2);
    assert_eq!(core::mem::offset_of!(Packed2, a), 0);
    assert_eq!(core::mem::offset_of!(Packed2, b), 2);

    assert_eq!(
        core::mem::size_of::<PackedShared>(),
        1 + core::mem::size_of::<usize>()
    );
    assert_eq!(core::mem::align_of::<PackedShared>(), 1);
    assert_eq!(core::mem::offset_of!(PackedShared, tag), 0);
    assert_eq!(core::mem::offset_of!(PackedShared, ptr), 1);

    assert_eq!(
        core::mem::size_of::<PackedSharedPair>(),
        1 + 2 * core::mem::size_of::<usize>()
    );
    assert_eq!(core::mem::align_of::<PackedSharedPair>(), 1);
    assert_eq!(core::mem::offset_of!(PackedSharedPair, tag), 0);
    assert_eq!(core::mem::offset_of!(PackedSharedPair, left), 1);
    assert_eq!(
        core::mem::offset_of!(PackedSharedPair, right),
        1 + core::mem::size_of::<usize>()
    );

    assert_eq!(
        core::mem::size_of::<SharedPair>(),
        2 * core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::align_of::<SharedPair>(),
        core::mem::align_of::<usize>()
    );
    assert_eq!(core::mem::offset_of!(SharedPair, left), 0);
    assert_eq!(
        core::mem::offset_of!(SharedPair, right),
        core::mem::size_of::<usize>()
    );

    assert_eq!(
        core::mem::size_of::<PackedNestedShared>(),
        1 + core::mem::size_of::<SharedPair>()
    );
    assert_eq!(core::mem::align_of::<PackedNestedShared>(), 1);
    assert_eq!(core::mem::offset_of!(PackedNestedShared, tag), 0);
    assert_eq!(core::mem::offset_of!(PackedNestedShared, pair), 1);

    assert_eq!(
        core::mem::size_of::<PackedSharedArray>(),
        1 + 2 * core::mem::size_of::<usize>()
    );
    assert_eq!(core::mem::align_of::<PackedSharedArray>(), 1);
    assert_eq!(core::mem::offset_of!(PackedSharedArray, tag), 0);
    assert_eq!(core::mem::offset_of!(PackedSharedArray, ptrs), 1);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_host_layout();

    if std::env::args().any(|arg| arg == "--verify-ptx") {
        verify_generated_ptx()?;
        println!("packed_aggregate_abi: PASS (host layout and PTX parameter shapes)");
        return Ok(());
    }

    verify_generated_ptx()?;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let by_value1_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let by_value2_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let packed_shared_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let packed_shared_pair_out = DeviceBuffer::<u32>::zeroed(&stream, 3)?;
    let packed_nested_shared_out = DeviceBuffer::<u32>::zeroed(&stream, 3)?;
    let packed_shared_array_out = DeviceBuffer::<u32>::zeroed(&stream, 3)?;
    let load1_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let load2_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let storage1 = DeviceBuffer::<u8>::zeroed(&stream, core::mem::size_of::<Packed1>())?;
    let storage2 = DeviceBuffer::<u8>::zeroed(&stream, core::mem::size_of::<Packed2>())?;

    let input1 = Packed1 {
        a: 0x21,
        b: 0x1020_3040,
    };
    let input2 = Packed2 {
        a: 0x31,
        b: 0x5060_7080,
    };

    let stored1 = Packed1 {
        a: 0x41,
        b: 0x90a0_b0c0,
    };
    let stored2 = Packed2 {
        a: 0x51,
        b: 0xd0e0_f001,
    };

    // SAFETY: every kernel launches one thread. The u32 output buffers contain
    // enough writable elements for their respective checks. The byte buffers
    // are CUDA allocations, so their base addresses satisfy Packed1/Packed2
    // alignment and have exact storage for one value of the corresponding
    // type. The packed-shared kernels create their AS3 pointers on device and
    // never expose them through the host ABI.
    unsafe {
        module.packed1(
            &stream,
            config,
            input1,
            by_value1_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed2(
            &stream,
            config,
            input2,
            by_value2_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed_shared(
            &stream,
            config,
            packed_shared_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed_shared_pair(
            &stream,
            config,
            packed_shared_pair_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed_nested_shared(
            &stream,
            config,
            packed_nested_shared_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed_shared_array(
            &stream,
            config,
            packed_shared_array_out.cu_deviceptr() as *mut u32,
        )?;

        module.store_packed1(
            &stream,
            config,
            stored1,
            storage1.cu_deviceptr() as *mut Packed1,
        )?;
        module.load_packed1(
            &stream,
            config,
            storage1.cu_deviceptr() as *const Packed1,
            load1_out.cu_deviceptr() as *mut u32,
        )?;

        module.store_packed2(
            &stream,
            config,
            stored2,
            storage2.cu_deviceptr() as *mut Packed2,
        )?;
        module.load_packed2(
            &stream,
            config,
            storage2.cu_deviceptr() as *const Packed2,
            load2_out.cu_deviceptr() as *mut u32,
        )?;
    }

    assert_eq!(by_value1_out.to_host_vec(&stream)?, [0x22, 0x1122_3344]);
    assert_eq!(by_value2_out.to_host_vec(&stream)?, [0x33, 0x6172_8394]);
    assert_eq!(packed_shared_out.to_host_vec(&stream)?, [0x22, 0x1122_3344]);
    assert_eq!(
        packed_shared_pair_out.to_host_vec(&stream)?,
        [0x32, 0x1122_3344, 0x2233_4455]
    );
    assert_eq!(
        packed_nested_shared_out.to_host_vec(&stream)?,
        [0x42, 0x3344_5566, 0x4455_6677]
    );
    assert_eq!(
        packed_shared_array_out.to_host_vec(&stream)?,
        [0x52, 0x5566_7788, 0x6677_8899]
    );
    assert_eq!(load1_out.to_host_vec(&stream)?, [0x41, 0x90a0_b0c0]);
    assert_eq!(load2_out.to_host_vec(&stream)?, [0x51, 0xd0e0_f001]);

    let bytes1 = storage1.to_host_vec(&stream)?;
    assert_eq!(bytes1[0], 0x41);
    assert_eq!(&bytes1[1..5], &0x90a0_b0c0u32.to_le_bytes());

    let bytes2 = storage2.to_host_vec(&stream)?;
    assert_eq!(bytes2[0], 0x51);
    // bytes2[1] is Rust padding and intentionally has no asserted value.
    assert_eq!(&bytes2[2..6], &0xd0e0_f001u32.to_le_bytes());

    println!(
        "packed_aggregate_abi: PASS (runtime values, direct packed-AS3 carrier-local projections, recursive/multi-leaf/bounded-array packed shared internal ABI, whole-value load/store, and PTX parameter shapes)"
    );
    Ok(())
}
