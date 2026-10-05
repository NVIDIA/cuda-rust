// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression for pointer provenance in a direct tuple constant.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

static DATA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Keep two pointer relocations in the same direct tuple. This exercises both
// a complete-static reference and an interior reference with a byte addend.
const POINTERS: (&[u8; 16], &u8, bool, u8) = (&DATA, &DATA[7], true, 3);

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `output` must address writable device storage for every launched thread.
    #[kernel]
    pub unsafe fn direct_tuple_pointer(output: *mut u32) {
        let index = thread::index_1d().get();
        let (base, interior, flag, interior_addend) = POINTERS;

        unsafe {
            output.add(index).write(
                base[index & 15] as u32 + flag as u32 + *interior as u32 + interior_addend as u32,
            );
        }
    }
}

fn expected(index: usize) -> u32 {
    DATA[index & 15] as u32 + 1 + DATA[7] as u32 + 3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 32;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let output = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: the output allocation contains N elements and the launch creates
    // exactly N logical thread indices.
    unsafe {
        module.direct_tuple_pointer(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            output.cu_deviceptr() as *mut u32,
        )
    }?;

    let got = output.to_host_vec(&stream)?;

    for (index, value) in got.iter().copied().enumerate() {
        assert_eq!(value, expected(index), "mismatch at index {index}");
    }

    println!("tuple_constant_provenance: PASS ({N} threads)");
    Ok(())
}
