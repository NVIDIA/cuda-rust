/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ordinary device global static example.
//!
//! Build and run with:
//!   cargo oxide run device_global

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::kernel;
use cuda_host::cuda_module;

static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;

mod debug_left {
    pub(super) static mut SAME_LEAF: u32 = 1; // CUDA_OXIDE_DEBUG_GLOBAL_LEFT
}

mod debug_right {
    pub(super) static mut SAME_LEAF: u64 = 2; // CUDA_OXIDE_DEBUG_GLOBAL_RIGHT
}

#[unsafe(no_mangle)]
pub static mut DEBUG_REACHABLE: u32 = 3; // CUDA_OXIDE_DEBUG_GLOBAL_REACHABLE
static mut DEBUG_PRIVATE: u32 = 4; // CUDA_OXIDE_DEBUG_GLOBAL_PRIVATE
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];
static STATIC_NAN: f32 = f32::from_bits(0x7fc0_1234);

const STATIC_WEIGHT_PAIR: &[f32; 2] = &STATIC_WEIGHTS[2];

/// These targets are intentionally reached only through other static
/// initializers. Their materialization therefore exercises transitive device
/// global discovery rather than a direct reference from a kernel body.
static RELOCATION_TARGET_A: u32 = 0x1234_5678;
static RELOCATION_TARGET_B: u32 = 0xcafe_babe;
static RELOCATION_REFERENCE: &u32 = &RELOCATION_TARGET_A;
static RELOCATION_REFERENCES: [&u32; 2] = [&RELOCATION_TARGET_A, &RELOCATION_TARGET_B];
static INTERIOR_RELOCATION_REFERENCE: &f32 = &STATIC_WEIGHTS[2][1];

#[repr(C)]
struct SliceRelocationTarget {
    prefix: [u32; 2],
    view: [u32; 3],
    suffix: [u32; 3],
}

static SLICE_RELOCATION_TARGET: SliceRelocationTarget = SliceRelocationTarget {
    prefix: [11, 17],
    view: [23, 31, 41],
    suffix: [47, 59, 61],
};
static SLICE_RELOCATION_VIEW: &[u32] = &SLICE_RELOCATION_TARGET.view;

static UNION_RELOCATION_TARGETS: [u32; 3] = [7, 11, 23];

#[repr(C)]
union RelocatedPointerUnion {
    word: &'static u32,
    byte: &'static u8,
}

static UNION_RELOCATION: RelocatedPointerUnion = RelocatedPointerUnion {
    word: &UNION_RELOCATION_TARGETS[2],
};

#[repr(C, packed)]
struct PackedRelocation {
    tag: u8,
    ptr: &'static u32,
}

static PACKED_RELOCATION: PackedRelocation = PackedRelocation {
    tag: 0x7b,
    ptr: &RELOCATION_TARGET_A,
};

#[repr(C, packed)]
struct PackedInteriorRelocation {
    prefix: [u8; 3],
    ptr: &'static f32,
    suffix: u16,
}

static PACKED_INTERIOR_RELOCATION: PackedInteriorRelocation = PackedInteriorRelocation {
    prefix: [0x11, 0x22, 0x33],
    ptr: &STATIC_WEIGHTS[2][1],
    suffix: 0x4455,
};

#[repr(C, packed)]
struct NestedPackedRelocation {
    tag: u8,
    ptr: &'static u32,
}

#[repr(C)]
struct NestedPackedRelocationCarrier {
    head: u32,
    nested: NestedPackedRelocation,
}

static NESTED_PACKED_RELOCATION: NestedPackedRelocationCarrier = NestedPackedRelocationCarrier {
    head: 0x1122_3344,
    nested: NestedPackedRelocation {
        tag: 0x7d,
        ptr: &RELOCATION_TARGET_B,
    },
};

/// One-past-the-end interior pointer: const eval permits forming a pointer
/// whose addend equals the allocation size (32 bytes here). It is legal to
/// form and compare, only dereferencing it would be UB, so the translator
/// must materialize it instead of rejecting the offset.
const STATIC_WEIGHTS_END: *const [f32; 2] =
    unsafe { (&raw const STATIC_WEIGHTS as *const [f32; 2]).add(4) };

#[repr(C)]
struct PaddedStatic {
    tag: u8,
    value: u32,
}

static PADDED_STATIC: PaddedStatic = PaddedStatic {
    tag: 0xab,
    value: 0x1234_5678,
};

#[inline(never)]
fn get_static_weights() -> &'static [[f32; 2]; 4] {
    &STATIC_WEIGHTS
}

#[inline(never)]
fn get_static_weight_pair() -> &'static [f32; 2] {
    STATIC_WEIGHT_PAIR
}

#[inline(never)]
fn get_static_nan() -> &'static f32 {
    &STATIC_NAN
}

#[inline(never)]
fn get_padded_static() -> &'static PaddedStatic {
    &PADDED_STATIC
}

#[inline(never)]
fn get_padded_static_tag() -> &'static u8 {
    &PADDED_STATIC.tag
}

#[inline(never)]
fn get_padded_static_value() -> &'static u32 {
    &PADDED_STATIC.value
}

#[inline(never)]
fn get_static_weights_end() -> *const [f32; 2] {
    STATIC_WEIGHTS_END
}

#[inline(never)]
fn block_local_static_values(select_left: bool) -> (u64, u64) {
    if select_left {
        static VALUE: u64 = 11; // CUDA_OXIDE_DEBUG_BLOCK_LOCAL_LEFT_VALUE
        static VALUE_REF: &u64 = &VALUE; // CUDA_OXIDE_DEBUG_BLOCK_LOCAL_LEFT_REFERENCE
        (
            *VALUE_REF + *VALUE_REF,
            VALUE_REF as *const u64 as usize as u64,
        )
    } else {
        static VALUE: u64 = 29; // CUDA_OXIDE_DEBUG_BLOCK_LOCAL_RIGHT_VALUE
        static VALUE_REF: &u64 = &VALUE; // CUDA_OXIDE_DEBUG_BLOCK_LOCAL_RIGHT_REFERENCE
        (
            *VALUE_REF + *VALUE_REF,
            VALUE_REF as *const u64 as usize as u64,
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to a writable `u64` in device-accessible memory.
    /// The static globals `DEVICE_COUNTER` and `DEVICE_MARKER` are mutated
    /// without synchronisation; the test launches a single thread to dodge
    /// the race.
    #[kernel]
    pub unsafe fn device_global(out: *mut u64) {
        unsafe {
            DEVICE_COUNTER += 1;
            DEVICE_MARKER = 0x00C0_FFEE;
            // Both statics are initialized here. CUDA_OXIDE_DEBUG_GLOBAL_BREAKPOINT
            *out = DEVICE_COUNTER ^ (DEVICE_MARKER as u64);
        }
    }

    /// Materialize adversarial source identities for the global-debug shape
    /// verifier: duplicate leaves in distinct modules and reachable/private
    /// definitions. A second read of the left static also checks that repeated
    /// references still produce one physical global and one CU entry.
    #[kernel]
    pub unsafe fn debug_global_identity(out: *mut u64) {
        unsafe {
            let left = debug_left::SAME_LEAF;
            let left_again = debug_left::SAME_LEAF;
            let right = debug_right::SAME_LEAF;
            DEBUG_REACHABLE += 1;
            DEBUG_PRIVATE += 1;
            *out = left as u64
                + left_again as u64
                + right
                + DEBUG_REACHABLE as u64
                + DEBUG_PRIVATE as u64;
        }
    }

    /// Exercise two same-named block statics, and the relocations that point
    /// to them, which must remain physically distinct. Their source display
    /// paths are identical; only rustc's DefPath-disambiguated symbol identity
    /// separates them.
    #[kernel]
    pub unsafe fn block_local_static_identity(out: *mut u64) {
        let thread = cuda_device::thread::threadIdx_x();
        if thread >= 2 {
            return;
        }

        let (value, address) = block_local_static_values(thread == 0);
        let offset = thread as usize * 2;

        unsafe {
            *out.add(offset) = value;
            *out.add(offset + 1) = address;
        }
    }

    /// Read both the base address and an interior pointer into an immutable
    /// device static.
    ///
    /// `STATIC_WEIGHT_PAIR` carries the provenance of `STATIC_WEIGHTS` plus
    /// a 16-byte addend selecting element 2.
    #[kernel]
    pub unsafe fn nonzero_static_table(out: *mut f32) {
        let weights = get_static_weights();
        let pair = get_static_weight_pair();

        unsafe {
            *out = weights[0][0] + pair[0] + pair[1];
        }
    }

    /// Preserve exact initializer bits and Rust's evaluated field offsets.
    #[kernel]
    pub unsafe fn static_initializer_edges(nan_out: *mut f32, padded_out: *mut u64) {
        let padded = get_padded_static();
        unsafe {
            *nan_out = *get_static_nan();
            *padded_out = ((padded.value as u64) << 8) | padded.tag as u64;
        }
    }

    #[kernel]
    pub unsafe fn static_subobject_pointers(out: *mut u32) {
        unsafe {
            *out.add(0) = *get_padded_static_tag() as u32;
            *out.add(1) = *get_padded_static_value();
            *out.add(2) = get_static_weight_pair()[0].to_bits();
            *out.add(3) = get_static_weight_pair()[1].to_bits();
        }
    }

    /// Read through pointer relocations stored inside device-global
    /// initializers. The table covers a direct target, repeated/shared targets,
    /// a second target, and an interior pointer with a non-zero byte addend.
    #[kernel]
    pub unsafe fn static_initializer_relocations(out: *mut u32) {
        unsafe {
            *out.add(0) = *RELOCATION_REFERENCE;
            *out.add(1) = *RELOCATION_REFERENCES[0];
            *out.add(2) = *RELOCATION_REFERENCES[1];
            *out.add(3) = (*INTERIOR_RELOCATION_REFERENCE).to_bits();
        }
    }

    /// Read a slice stored directly in a device-global initializer.
    ///
    /// The first fat-pointer word carries a relocation to
    /// `SLICE_RELOCATION_TARGET + 8`; the second word is literal length metadata.
    #[kernel]
    pub unsafe fn slice_static_initializer_relocation(out: *mut u32) {
        unsafe {
            *out.add(0) = SLICE_RELOCATION_VIEW.len() as u32;
            *out.add(1) = SLICE_RELOCATION_VIEW[0];
            *out.add(2) = SLICE_RELOCATION_VIEW[2];
        }
    }

    /// Read a top-level union whose complete static initializer is one
    /// provenance-preserving pointer relocation with a non-zero target addend.
    #[kernel]
    pub unsafe fn union_static_initializer_relocation(out: *mut u32) {
        unsafe {
            *out = *UNION_RELOCATION.word;
        }
    }

    /// Read relocation slots whose pointer bytes start at unaligned offsets
    /// inside `repr(packed)` statics. The pointer fields are loaded with
    /// `read_unaligned` so no invalid aligned reference is ever formed.
    #[kernel]
    pub unsafe fn packed_static_initializer_relocations(out: *mut u32) {
        let tag = unsafe { core::ptr::addr_of!(PACKED_RELOCATION.tag).read_unaligned() };
        let direct_ptr = unsafe { core::ptr::addr_of!(PACKED_RELOCATION.ptr).read_unaligned() };
        let prefix0 = unsafe {
            core::ptr::addr_of!(PACKED_INTERIOR_RELOCATION.prefix)
                .cast::<u8>()
                .read_unaligned()
        };
        let interior_ptr =
            unsafe { core::ptr::addr_of!(PACKED_INTERIOR_RELOCATION.ptr).read_unaligned() };
        let suffix =
            unsafe { core::ptr::addr_of!(PACKED_INTERIOR_RELOCATION.suffix).read_unaligned() };

        unsafe {
            *out.add(0) = tag as u32;
            *out.add(1) = *direct_ptr;
            *out.add(2) = prefix0 as u32;
            *out.add(3) = (*interior_ptr).to_bits();
            *out.add(4) = suffix as u32;
        }
    }

    /// Read a relocation through one direct packed struct nested inside an
    /// ordinary `repr(C)` device static. Only the packed fields require
    /// unaligned loads; the outer `u32` remains naturally aligned.
    #[kernel]
    pub unsafe fn nested_packed_static_initializer_relocation(out: *mut u32) {
        let head = NESTED_PACKED_RELOCATION.head;
        let tag =
            unsafe { core::ptr::addr_of!(NESTED_PACKED_RELOCATION.nested.tag).read_unaligned() };
        let ptr =
            unsafe { core::ptr::addr_of!(NESTED_PACKED_RELOCATION.nested.ptr).read_unaligned() };

        unsafe {
            *out.add(0) = head;
            *out.add(1) = tag as u32;
            *out.add(2) = *ptr;
        }
    }

    /// A one-past-the-end constant pointer is formed and compared, never
    /// dereferenced. The distance from the static's base must equal the
    /// allocation size (32 bytes).
    #[kernel]
    pub unsafe fn static_one_past_end(out: *mut u32) {
        let base = get_static_weights() as *const [[f32; 2]; 4] as usize;
        let end = get_static_weights_end() as usize;
        unsafe {
            *out = (end - base) as u32;
        }
    }
}

fn main() {
    println!("=== Device Global Static Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate output");

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    for launch_idx in 1..=2 {
        unsafe {
            module.device_global(
                &stream,
                LaunchConfig::for_num_elems(1),
                out_dev.cu_deviceptr() as *mut u64,
            )
        }
        .expect("Kernel launch failed");

        let result = out_dev.to_host_vec(&stream).expect("Failed to copy result")[0];
        let expected = launch_idx ^ 0x00C0_FFEEu64;

        println!("Launch {launch_idx}: result = {result:#x}");
        if result != expected {
            eprintln!("FAILED: expected {expected:#x}, got {result:#x}");
            std::process::exit(1);
        }
    }

    let block_local_out_dev = DeviceBuffer::<u64>::zeroed(&stream, 4)
        .expect("Failed to allocate block-local static output");
    unsafe {
        module.block_local_static_identity(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (2, 1, 1),
                shared_mem_bytes: 0,
            },
            block_local_out_dev.cu_deviceptr() as *mut u64,
        )
    }
    .expect("Block-local static identity kernel launch failed");
    let block_local_result = block_local_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy block-local static output");

    println!(
        "Block-local statics: values = [{}, {}], addresses = [{:#x}, {:#x}]",
        block_local_result[0], block_local_result[2], block_local_result[1], block_local_result[3]
    );
    if block_local_result[0] != 22
        || block_local_result[2] != 58
        || block_local_result[1] == 0
        || block_local_result[3] == 0
        || block_local_result[1] == block_local_result[3]
    {
        eprintln!(
            "FAILED: expected distinct block-local values 22/58 and non-zero distinct addresses, got {block_local_result:?}"
        );
        std::process::exit(1);
    }

    let static_out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, 1).expect("Failed to allocate static output");
    unsafe {
        module.nonzero_static_table(
            &stream,
            LaunchConfig::for_num_elems(1),
            static_out_dev.cu_deviceptr() as *mut f32,
        )
    }
    .expect("Static table kernel launch failed");
    let static_result = static_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy static result")[0];
    let static_expected = 12.25f32;
    println!("Static table: result = {static_result}");
    if (static_result - static_expected).abs() > f32::EPSILON {
        eprintln!("FAILED: expected {static_expected}, got {static_result}");
        std::process::exit(1);
    }

    let nan_out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, 1).expect("Failed to allocate NaN output");
    let padded_out_dev =
        DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate padded output");
    unsafe {
        module.static_initializer_edges(
            &stream,
            LaunchConfig::for_num_elems(1),
            nan_out_dev.cu_deviceptr() as *mut f32,
            padded_out_dev.cu_deviceptr() as *mut u64,
        )
    }
    .expect("Static initializer edge-case kernel launch failed");

    let nan_bits = nan_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy NaN output")[0]
        .to_bits();
    let padded_result = padded_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy padded output")[0];
    let padded_expected = (0x1234_5678u64 << 8) | 0xabu64;
    println!("NaN payload: bits = {nan_bits:#010x}");
    println!("Padded static: result = {padded_result:#x}");
    if nan_bits != 0x7fc0_1234 || padded_result != padded_expected {
        eprintln!(
            "FAILED: expected NaN bits {:#010x} and padded value {padded_expected:#x}, got {nan_bits:#010x} and {padded_result:#x}",
            0x7fc0_1234u32
        );
        std::process::exit(1);
    }

    let subobject_out_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("Failed to allocate subobject output");

    unsafe {
        module.static_subobject_pointers(
            &stream,
            LaunchConfig::for_num_elems(1),
            subobject_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Static subobject kernel launch failed");

    let subobject_result = subobject_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy static subobject output");

    let subobject_expected = [0xabu32, 0x1234_5678, 4.0f32.to_bits(), 8.0f32.to_bits()];

    println!("Static subobjects: result = {subobject_result:?}");

    if subobject_result.as_slice() != subobject_expected.as_slice() {
        eprintln!(
            "FAILED: expected static subobjects {subobject_expected:?}, got {subobject_result:?}"
        );
        std::process::exit(1);
    }

    let relocation_out_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("Failed to allocate relocation output");

    unsafe {
        module.static_initializer_relocations(
            &stream,
            LaunchConfig::for_num_elems(1),
            relocation_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Static initializer relocation kernel launch failed");

    let relocation_result = relocation_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy relocation output");
    let relocation_expected = [0x1234_5678, 0x1234_5678, 0xcafe_babe, 8.0f32.to_bits()];

    println!("Static initializer relocations: result = {relocation_result:?}");

    if relocation_result.as_slice() != relocation_expected.as_slice() {
        eprintln!(
            "FAILED: expected static initializer relocations {relocation_expected:?}, got {relocation_result:?}"
        );
        std::process::exit(1);
    }

    let slice_relocation_out_dev = DeviceBuffer::<u32>::zeroed(&stream, 3)
        .expect("Failed to allocate slice relocation output");

    unsafe {
        module.slice_static_initializer_relocation(
            &stream,
            LaunchConfig::for_num_elems(1),
            slice_relocation_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Slice static initializer relocation kernel launch failed");

    let slice_relocation_result = slice_relocation_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy slice relocation output");
    let slice_relocation_expected = [3u32, 23, 41];

    println!("Slice static initializer relocation: result = {slice_relocation_result:?}");

    if slice_relocation_result.as_slice() != slice_relocation_expected.as_slice() {
        eprintln!(
            "FAILED: expected slice static initializer relocation {slice_relocation_expected:?}, got {slice_relocation_result:?}"
        );
        std::process::exit(1);
    }

    let union_relocation_out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)
        .expect("Failed to allocate union relocation output");

    unsafe {
        module.union_static_initializer_relocation(
            &stream,
            LaunchConfig::for_num_elems(1),
            union_relocation_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Union static initializer relocation kernel launch failed");

    let union_relocation_result = union_relocation_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy union relocation output")[0];

    println!("Union static initializer relocation: result = {union_relocation_result}");

    if union_relocation_result != 23 {
        eprintln!(
            "FAILED: expected union static initializer relocation 23, got {union_relocation_result}"
        );
        std::process::exit(1);
    }

    let packed_relocation_out_dev = DeviceBuffer::<u32>::zeroed(&stream, 5)
        .expect("Failed to allocate packed relocation output");

    unsafe {
        module.packed_static_initializer_relocations(
            &stream,
            LaunchConfig::for_num_elems(1),
            packed_relocation_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Packed static initializer relocation kernel launch failed");

    let packed_relocation_result = packed_relocation_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy packed relocation output");
    let packed_relocation_expected = [0x7bu32, 0x1234_5678, 0x11, 8.0f32.to_bits(), 0x4455];

    println!("Packed static initializer relocations: result = {packed_relocation_result:?}");

    if packed_relocation_result.as_slice() != packed_relocation_expected.as_slice() {
        eprintln!(
            "FAILED: expected packed static initializer relocations {packed_relocation_expected:?}, got {packed_relocation_result:?}"
        );
        std::process::exit(1);
    }

    let nested_packed_out_dev = DeviceBuffer::<u32>::zeroed(&stream, 3)
        .expect("Failed to allocate nested packed relocation output");

    unsafe {
        module.nested_packed_static_initializer_relocation(
            &stream,
            LaunchConfig::for_num_elems(1),
            nested_packed_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Nested packed static initializer relocation kernel launch failed");

    let nested_packed_result = nested_packed_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy nested packed relocation output");
    let nested_packed_expected = [0x1122_3344u32, 0x7d, 0xcafe_babe];

    println!("Nested packed static initializer relocation: result = {nested_packed_result:?}");

    if nested_packed_result.as_slice() != nested_packed_expected.as_slice() {
        eprintln!(
            "FAILED: expected nested packed static initializer relocation {nested_packed_expected:?}, got {nested_packed_result:?}"
        );
        std::process::exit(1);
    }

    let one_past_end_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 1).expect("Failed to allocate one-past-end output");

    unsafe {
        module.static_one_past_end(
            &stream,
            LaunchConfig::for_num_elems(1),
            one_past_end_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("One-past-end kernel launch failed");

    let one_past_end_result = one_past_end_dev
        .to_host_vec(&stream)
        .expect("Failed to copy one-past-end output")[0];

    println!("One-past-the-end offset: result = {one_past_end_result}");

    if one_past_end_result != 32 {
        eprintln!("FAILED: expected one-past-the-end offset 32, got {one_past_end_result}");
        std::process::exit(1);
    }

    println!(
        "\nSUCCESS: device globals preserved same-path static identities, storage, initializer bytes, aligned, packed, nested packed, and union pointer relocations, slice relocations, pointer addends, and subobject addresses."
    );
}
