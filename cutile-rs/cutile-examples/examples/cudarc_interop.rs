/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Run cutile kernels over device memory a foreign framework (cudarc, torch,
//! ...) already owns — no copy, no ownership transfer.
//!
//! `foreign::Buffer` stands in for the framework's allocation: it owns the
//! memory and frees it on drop. It implements [`DeviceAllocation`] once (the
//! pointer-validity assertion), after which `Tensor::from_foreign` verifies
//! everything positional at construction — the remaining `unsafe` at the
//! borrow asserts only exclusive access for the duration. The tensor holds an
//! `Arc` to the buffer as a liveness token, so the memory provably outlives
//! every cutile use — even after the framework drops its own handle.
//!
//! A real cudarc `CudaSlice` (or a torch storage) is wrapped the same way: a
//! few-line `unsafe impl DeviceAllocation` that returns its device pointer,
//! byte length, and ordinal.

use cuda_async::device_buffer::DeviceAllocation;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{free_async, malloc_async, memcpy_dtoh_async, memcpy_htod_async, Device, Stream};
use cutile::error::Error;
use cutile::prelude::*;
use std::sync::Arc;

/// A foreign framework's device allocation: owns the memory, frees it on drop.
mod foreign {
    use super::*;

    pub struct Buffer {
        dptr: CUdeviceptr,
        len_bytes: usize,
        device_id: usize,
        stream: Arc<Stream>,
    }

    impl Buffer {
        pub fn alloc(elems: usize, stream: &Arc<Stream>) -> Arc<Self> {
            let len_bytes = elems * std::mem::size_of::<f32>();
            let dptr = unsafe { malloc_async(len_bytes, stream) }.expect("device allocation");
            Arc::new(Self {
                dptr,
                len_bytes,
                device_id: stream.device().ordinal(),
                stream: stream.clone(),
            })
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // The framework frees its own memory when its last handle drops.
            // A destructor cannot report a refused free, so the error is dropped.
            let _ = unsafe { free_async(self.dptr, &self.stream) };
            println!("  [foreign] freed {:#x}", self.dptr);
        }
    }

    // SAFETY: `dptr` is a live `len_bytes` device allocation on `device_id` for
    // this buffer's whole lifetime (it is freed only in `Drop`).
    unsafe impl DeviceAllocation for Buffer {
        fn device_ptr(&self) -> CUdeviceptr {
            self.dptr
        }
        fn len_bytes(&self) -> usize {
            self.len_bytes
        }
        fn device_id(&self) -> usize {
            self.device_id
        }
    }
}

#[cutile::module]
mod tile_add {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("cudarc_interop", 0)? {
        return Ok(());
    }

    const N: usize = 1024;
    const TILE: usize = 128;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    // The framework allocates and owns three buffers, and fills the inputs.
    let x = foreign::Buffer::alloc(N, &stream);
    let y = foreign::Buffer::alloc(N, &stream);
    let z = foreign::Buffer::alloc(N, &stream);
    let ones = [1.0f32; N];
    unsafe {
        memcpy_htod_async(x.device_ptr(), ones.as_ptr(), N, &stream)?;
        memcpy_htod_async(y.device_ptr(), ones.as_ptr(), N, &stream)?;
    }

    // cutile borrows the foreign memory — no copy. Validity and liveness are
    // verified at construction (the Arc is the liveness token); the `unsafe`
    // asserts the one residual: exclusive access for the duration.
    // SAFETY: from here on, only cutile touches these buffers — x and y are
    // read-only inputs, z is written only by the kernel below, and the
    // framework's own last access (the h2d fills) completed above.
    let shape = vec![N as i32];
    let (tx, ty, tz) = unsafe {
        (
            Tensor::<f32>::from_foreign(x.clone(), shape.clone(), vec![1]),
            Tensor::<f32>::from_foreign(y.clone(), shape.clone(), vec![1]),
            Tensor::<f32>::from_foreign(z.clone(), shape.clone(), vec![1]),
        )
    };

    // The framework drops its OWN handle to z. The memory is not freed: cutile's
    // tensor still holds the buffer alive. (Keep z's address — a copyable u64 —
    // to read the result back afterward.)
    let z_dptr = z.device_ptr();
    drop(z);
    println!("foreign dropped its z handle; memory still alive (cutile holds it)");

    // cutile runs the kernel over the borrowed memory.
    let (tz, _tx, _ty) = tile_add::add(tz.partition([TILE]), tx, ty).sync_on(&stream)?;

    // Read the result back out of the (still-live) foreign z buffer.
    let mut host = [0.0f32; N];
    unsafe {
        memcpy_dtoh_async(host.as_mut_ptr(), z_dptr, N, &stream)?;
        stream.synchronize()?;
    }
    assert!(host.iter().all(|v| *v == 2.0));
    println!("kernel ran over foreign z; result correct");

    // Dropping cutile's tensors releases the last references — only now does the
    // framework's buffer free (see the "[foreign] freed" lines).
    drop((tz, _tx, _ty));
    drop((x, y));

    println!(
        "cudarc_interop: borrowed foreign memory, ran kernel, foreign owner freed exactly once."
    );
    Ok(())
}
