// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end example for the SM100 packed f32x2 arithmetic family.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::f32x2::{
    add_f32x2, add_ftz_f32x2, fma_f32x2, fma_ftz_f32x2, mul_f32x2, mul_ftz_f32x2, pack_f32x2,
    sub_f32x2, sub_ftz_f32x2,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const NUM_OPS: usize = 8;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_f32x2_arith(mut out: DisjointSlice<[u64; NUM_OPS]>) {
        let idx = thread::index_1d();
        if let Some(row) = out.get_mut(idx) {
            let a = pack_f32x2(2.0, 4.0);
            let b = pack_f32x2(3.0, 5.0);
            let c = pack_f32x2(7.0, 11.0);

            row[0] = add_f32x2(a, b);
            row[1] = add_ftz_f32x2(a, b);
            row[2] = sub_f32x2(a, b);
            row[3] = sub_ftz_f32x2(a, b);
            row[4] = mul_f32x2(a, b);
            row[5] = mul_ftz_f32x2(a, b);
            row[6] = fma_f32x2(a, b, c);
            row[7] = fma_ftz_f32x2(a, b, c);
        }
    }
}

fn pack_host(lo: f32, hi: f32) -> u64 {
    u64::from(lo.to_bits()) | (u64::from(hi.to_bits()) << 32)
}

fn main() {
    println!("=== f32x2_arith ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 10 {
        println!(
            "skipping: packed f32x2 arithmetic requires sm_100+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");
    let mut out = DeviceBuffer::<[u64; NUM_OPS]>::zeroed(&stream, 1).unwrap();

    // SAFETY: one launched thread owns the single output row.
    unsafe { module.test_f32x2_arith(&stream, LaunchConfig::for_num_elems(1), &mut out) }
        .expect("launch test_f32x2_arith");

    let rows = out.to_host_vec(&stream).unwrap();
    let expected = [
        pack_host(5.0, 9.0),
        pack_host(5.0, 9.0),
        pack_host(-1.0, -1.0),
        pack_host(-1.0, -1.0),
        pack_host(6.0, 20.0),
        pack_host(6.0, 20.0),
        pack_host(13.0, 31.0),
        pack_host(13.0, 31.0),
    ];
    assert_eq!(rows.as_slice(), &[expected]);

    println!("PASS: all packed f32x2 arithmetic variants verified on sm_{major}{minor}");
}
