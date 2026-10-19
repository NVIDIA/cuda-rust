/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Comparisons on a bool produced by short-circuit `||` (PR #141).
 *
 * A bool assigned in two arms of a short-circuit `||`/`&&` becomes a
 * block argument (phi) after mem2reg. Bools are signless i1 in
 * dialect-mir, which `can_convert_type` rejects, so DialectConversion
 * records no pre-conversion type for the phi. `is_signed_int_op` used to
 * fail the whole lowering for such operands:
 *
 *     expected IntegerType or MirPtrType operand in arithmetic op
 *
 * It now falls back to the live operand type and lowers the comparison
 * as unsigned (`icmp eq i1` / `icmp ult i1`).
 *
 * The kernel feeds the phi into both `==` and `<`; the inputs cover all
 * four (p, q) truth combinations so a wrong predicate fails loudly.
 *
 * Build:
 *     cargo oxide run bool_phi_cmp
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `p` is the short-circuit bool phi; `q` is an independent bool.
    /// Encodes both comparisons in one lane: `(p == q) + 2 * (p < q)`.
    // The bool `<` is the regression under test: it must reach mir-lower as a
    // comparison on an untracked i1 phi and lower to `icmp ult i1`. Clippy's
    // simplification (`!p & q`) would change the MIR shape and skip that path.
    #[allow(clippy::bool_comparison)]
    #[kernel]
    pub fn bool_phi_cmp(a: &[u32], b: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            let p = a[i] > 1 || b[i] > 2;
            let q = a[i] == 0;
            *e = (p == q) as u32 + 2 * ((p < q) as u32);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Load embedded PTX");

    // Lanes cover all four (p, q) combinations:
    //   i=0: a=0, b=0 -> p=false, q=true  -> eq=0, lt=1 -> 2
    //   i=1: a=2, b=0 -> p=true,  q=false -> eq=0, lt=0 -> 0
    //   i=2: a=0, b=5 -> p=true,  q=true  -> eq=1, lt=0 -> 1
    //   i=3: a=1, b=1 -> p=false, q=false -> eq=1, lt=0 -> 1
    let a = vec![0u32, 2, 0, 1];
    let b = vec![0u32, 0, 5, 1];
    let expect = vec![2u32, 0, 1, 1];

    let dev_a = DeviceBuffer::from_host(&stream, &a).unwrap();
    let dev_b = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, a.len()).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.bool_phi_cmp(
            &stream,
            LaunchConfig::for_num_elems(a.len() as u32),
            &dev_a,
            &dev_b,
            &mut dev_out,
        )
    }
    .expect("launch");

    let out = dev_out.to_host_vec(&stream).unwrap();
    println!("got    : {out:?}");
    println!("expect : {expect:?}");
    if out == expect {
        println!("SUCCESS");
    } else {
        println!("FAILURE: bool-phi comparison produced wrong lanes");
        std::process::exit(1);
    }
}
