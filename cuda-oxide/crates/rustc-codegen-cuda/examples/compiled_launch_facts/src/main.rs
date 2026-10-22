/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{DisjointSlice, kernel, launch_contract, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;

#[repr(C, align(64))]
struct Scratch<const ITEMS: usize> {
    slots: [[u32; 32]; ITEMS],
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn exchange<const ITEMS: usize>() -> u32 {
        let lane = (thread::threadIdx_y() * 8 + thread::threadIdx_x()) as usize;
        // Both views alias the same base. Every lane writes its own slots;
        // the explicit barrier completes those writes before neighbor reads.
        unsafe {
            let scratch = DynamicSharedArray::<Scratch<ITEMS>, 16>::get_fixed().cast::<u32>();
            let alias = DynamicSharedArray::<[u32; 32], 128>::get_fixed().cast::<u32>();
            for item in 0..ITEMS {
                scratch
                    .add(item * 32 + lane)
                    .write_volatile((item * 100 + lane) as u32);
            }
            thread::sync_threads();
            let neighbor = (lane + 1) % 32;
            scratch.add((ITEMS - 1) * 32 + neighbor).read_volatile()
                + alias.add(neighbor).read_volatile()
        }
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 4, 1), dynamic_shared = inferred)]
    pub fn fixed_one(mut output: DisjointSlice<u32, thread::Index2D<8>>) {
        let value = exchange::<1>();
        if let Some(index) = thread::index_2d::<8>()
            && let Some(out) = output.get_mut(index)
        {
            *out = value;
        }
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 4, 1), dynamic_shared = inferred)]
    pub fn fixed_four(mut output: DisjointSlice<u32, thread::Index2D<8>>) {
        let value = exchange::<4>();
        if let Some(index) = thread::index_2d::<8>()
            && let Some(out) = output.get_mut(index)
        {
            *out = value;
        }
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 4, 1), dynamic_shared = inferred)]
    pub fn zero() {}

    #[cfg(feature = "generic")]
    #[launch_contract(domain = 2, block = (8, 4, 1), dynamic_shared = inferred)]
    #[kernel]
    pub fn generic<const ITEMS: usize>(mut output: DisjointSlice<u32, thread::Index2D<8>>) {
        let value = exchange::<ITEMS>();
        if let Some(index) = thread::index_2d::<8>()
            && let Some(out) = output.get_mut(index)
        {
            *out = value;
        }
    }

    #[cfg(feature = "invalid-raw")]
    #[kernel]
    #[launch_contract(domain = 2, block = (8, 4, 1), dynamic_shared = inferred)]
    pub fn incomplete() {
        unsafe {
            DynamicSharedArray::<[u32; 32], 16>::get_fixed()
                .cast::<u32>()
                .write_volatile(1);
            DynamicSharedArray::<u32, 16>::get()
                .add(64)
                .write_volatile(2);
        }
    }
}

fn verify(buffer: &cuda_core::DeviceBuffer<u32>, stream: &cuda_core::CudaStream, items: u32) {
    let mut values = [0; 32];
    buffer.copy_to_host(stream, &mut values).unwrap();
    stream.synchronize().unwrap();
    for (lane, value) in values.into_iter().enumerate() {
        assert_eq!(value, (items - 1) * 100 + 2 * ((lane as u32 + 1) % 32));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig2D};

    init_device_contexts(0, 1)?;
    let ctx = with_cuda_context(0, |ctx| ctx.clone())?;
    let stream = ctx.default_stream();
    // SAFETY: the embedded module is compiled from these kernel declarations.
    let module = unsafe { kernels::load_async(0)? };
    let config = LaunchConfig2D::auto_shared((1, 1), (8, 4));
    let one = module.prepare_fixed_one(config)?;
    let four = module.prepare_fixed_four(config)?;
    let zero = module.prepare_zero(config)?;
    assert_eq!(one.__raw_config().shared_mem_bytes, 128);
    assert_eq!(four.__raw_config().shared_mem_bytes, 512);
    assert_eq!(zero.__raw_config().shared_mem_bytes, 0);
    for prepared_facts in [
        one.function().launch_facts(),
        four.function().launch_facts(),
    ] {
        let facts = prepared_facts.unwrap();
        assert_eq!(facts.exact_block, Some((8, 4, 1)));
        assert_eq!(facts.dynamic_shared_alignment, 128);
    }
    assert_eq!(
        zero.function().launch_facts().unwrap().dynamic_shared_bytes,
        0
    );
    assert!(matches!(
        module.prepare_fixed_four(LaunchConfig2D::auto_shared((1, 1), (32, 1))),
        Err(cuda_core::LaunchContractError::BlockShapeMismatch { .. })
    ));
    assert!(matches!(
        module.prepare_fixed_four(LaunchConfig2D::new((1, 1), (8, 4), 511)),
        Err(cuda_core::LaunchContractError::DynamicSharedMemoryExactMismatch { .. })
    ));
    let explicit = module.prepare_fixed_four(LaunchConfig2D::new((1, 1), (8, 4), 512))?;

    let output = DeviceBuffer::from_host(&stream, &[0u32; 32])?;
    let output = module.fixed_one_async_owned(&one, output).sync()?;
    verify(&output, &stream, 1);
    let output = module.fixed_four_async_owned(&explicit, output).sync()?;
    verify(&output, &stream, 4);
    let output = module.fixed_four_async_owned(&four, output).sync()?;
    verify(&output, &stream, 4);

    #[cfg(feature = "generic")]
    {
        let one = module.prepare_generic::<1>(config)?;
        let four = module.prepare_generic::<4>(config)?;
        assert_eq!(one.__raw_config().shared_mem_bytes, 128);
        assert_eq!(four.__raw_config().shared_mem_bytes, 512);
        let output = module.generic_async_owned::<1, _>(&one, output).sync()?;
        verify(&output, &stream, 1);
        let output = module.generic_async_owned::<4, _>(&four, output).sync()?;
        verify(&output, &stream, 4);
    }
    println!(
        "PASS: compiled launch facts: helper layouts, exact shape, explicit/automatic bytes, zero, and owned async passed"
    );
    Ok(())
}
