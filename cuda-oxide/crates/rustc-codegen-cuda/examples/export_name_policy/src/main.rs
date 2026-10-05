/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for definition-side export names matching call-side
//! legalized names.
//!
//! This example pins two non-generic call shapes whose FQDNs contain characters
//! that are invalid in final PTX identifiers:
//! - a concrete trait impl method, `<Vec2 as DeviceMeasure>::trait_impl_value`
//! - a concrete inherent impl path, `Wrapper<Vec2>::inherent_concrete_value`
//!
//! Both calls should use their raw FQDN on the call and definition sides, then
//! rely on the shared pliron legalizer to produce the final symbol. They must
//! not be classified as generic merely because the MIR `FnDef` operand carries
//! substitutions or because the FQDN contains `<`, `>`, or `::`.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[derive(Clone, Copy)]
struct Vec2 {
    x: u32,
    y: u32,
}

trait DeviceMeasure {
    fn trait_impl_value(self) -> u32;
}

impl DeviceMeasure for Vec2 {
    #[inline(never)]
    fn trait_impl_value(self) -> u32 {
        self.x * 17 + self.y
    }
}

struct Wrapper<T>(T);

impl Wrapper<Vec2> {
    #[inline(never)]
    fn inherent_concrete_value(self) -> u32 {
        self.0.x + self.0.y * 31
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn export_name_policy(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let lane = idx.get() as u32;

        if let Some(out_elem) = out.get_mut(idx) {
            let value = Vec2 { x: lane + 3, y: 7 };
            let trait_value = <Vec2 as DeviceMeasure>::trait_impl_value(value);
            let inherent_value = Wrapper::<Vec2>(value).inherent_concrete_value();
            *out_elem = trait_value + inherent_value;
        }
    }
}

fn expected(lane: usize) -> u32 {
    let lane = lane as u32;
    let value = Vec2 { x: lane + 3, y: 7 };
    <Vec2 as DeviceMeasure>::trait_impl_value(value)
        + Wrapper::<Vec2>(value).inherent_concrete_value()
}

fn main() {
    println!("=== export_name_policy regression ===");

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load embedded CUDA module");

    const N: usize = 64;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).expect("failed to allocate output");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.export_name_policy(
            stream.as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            &mut out,
        )
    }
    .expect("export_name_policy launch failed");

    let got = out.to_host_vec(&stream).expect("failed to copy output");
    let failures = got
        .iter()
        .enumerate()
        .filter(|(lane, actual)| **actual != expected(*lane))
        .count();

    if failures == 0 {
        println!("export_name_policy: SUCCESS ({N} lanes)");
    } else {
        println!("export_name_policy: {failures} FAILURES");
        std::process::exit(1);
    }
}
