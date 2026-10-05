/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `DeviceBuffer::cast_chunks`: choose the element type at the boundary.
//!
//! A buffer is allocated and filled as flat `f32`, then reinterpreted once on
//! the host into a 16-byte-aligned four-float element and launched into a
//! kernel that takes that element type directly. The kernel copies whole
//! quads; the output is cast back to `f32` and compared bit-for-bit against
//! the input. Finally, the checked failure path: a 3-float buffer does not
//! divide into quads, so `cast_chunks` must refuse it and hand the original
//! buffer back intact.
//!
//! Run: `cargo oxide run cast_chunks`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::cuda_module;

/// A 16-byte-aligned quad of `f32`: the CUDA `float4` layout. Whole-quad
/// loads and stores move as single 128-bit transactions.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quad(pub [f32; 4]);

// Plain POD aggregate, no pointers: safe to memcpy to/from the device.
unsafe impl cuda_core::DeviceCopy for Quad {}

#[cuda_module]
mod kernels {
    use super::Quad;
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Copy one quad per thread. The value is moved whole, never decomposed
    /// into lanes, so the copy is a wide load feeding a wide store.
    #[kernel]
    pub fn copy_quads(input: &[Quad], mut output: DisjointSlice<Quad>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }
}

fn main() {
    const QUADS: usize = 256;
    const FLOATS: usize = QUADS * 4;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load module");
    let cfg = LaunchConfig::for_num_elems(QUADS as u32);

    // Allocate and fill as flat f32: no over-aligned type in sight yet.
    let host: Vec<f32> = (0..FLOATS).map(|i| i as f32 * 0.5 - 100.0).collect();
    let flat_in = DeviceBuffer::from_host(&stream, &host).expect("input upload");
    let flat_out = DeviceBuffer::<f32>::zeroed(&stream, FLOATS).expect("output alloc");

    // Reinterpret both buffers at the boundary: 4N floats -> N quads.
    assert!(
        flat_in.can_cast_chunks::<Quad>(),
        "aligned + divisible input"
    );
    let quads_in: DeviceBuffer<Quad> = flat_in
        .cast_chunks()
        .unwrap_or_else(|_| panic!("cast_chunks refused an aligned, divisible input"));
    let mut quads_out: DeviceBuffer<Quad> = flat_out
        .cast_chunks()
        .unwrap_or_else(|_| panic!("cast_chunks refused an aligned, divisible output"));
    assert_eq!(quads_in.len(), QUADS, "length is recomputed from bytes");
    assert_eq!(quads_out.len(), QUADS);

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.copy_quads(&stream, cfg, &quads_in, &mut quads_out) }
        .expect("copy_quads launch");

    // Cast the output back down to f32 and verify the round trip bit-for-bit.
    let flat_back: DeviceBuffer<f32> = quads_out
        .cast_chunks()
        .unwrap_or_else(|_| panic!("casting quads back to f32 cannot fail"));
    assert_eq!(flat_back.len(), FLOATS);
    let round_trip = flat_back.to_host_vec(&stream).expect("readback");
    assert_eq!(round_trip, host, "quad copy must round-trip the f32 data");

    // The refusal path: 3 floats are not a whole number of quads. The buffer
    // must come back unchanged rather than truncated to zero quads.
    let ragged = DeviceBuffer::from_host(&stream, &[1.0f32, 2.0, 3.0]).expect("ragged upload");
    assert!(
        !ragged.can_cast_chunks::<Quad>(),
        "3 f32 is not a whole quad"
    );
    let ragged = match ragged.cast_chunks::<Quad>() {
        Ok(_) => panic!("cast_chunks accepted a 3-float buffer as quads"),
        Err(original) => original,
    };
    assert_eq!(ragged.len(), 3, "refused cast hands the buffer back intact");
    let ragged_host = ragged.to_host_vec(&stream).expect("ragged readback");
    assert_eq!(ragged_host, vec![1.0f32, 2.0, 3.0]);

    println!(
        "SUCCESS: cast_chunks round-tripped {FLOATS} f32 through {QUADS} quads and refused the ragged buffer"
    );
}
