/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Functional NVFP4 GEMV through retained CuTe semantic operations.
//!
//! The device algorithm intentionally mirrors the CuTeDSL reference:
//! - A is M x K packed E2M1, b is K packed E2M1;
//! - one E8M0 scale is applied to each group of 16 logical values;
//! - a 128-thread CTA computes 128 output rows, one row per thread;
//! - K advances in tiles of 64 and accumulation is f32;
//! - C is stored as f16.
//!
//! cute-rs binds each packed value tensor to its scales, element formats, and
//! layout. Thread-row and K-tile views make ownership explicit while each
//! backend chooses how the semantic load and dot operations are implemented.
//! PASS requires every output f16 bit pattern to match the host reference.

#![feature(f16)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cute_rs::{BlockScaledTensor, E2M1, KMajor, Mkl, Nkl, Sm1xxBlockScaleKMajor, UE8M0};

const CTA_M: usize = 128;
const K_TILE: usize = 64;
const SF_VEC: usize = 16;
const DEFAULT_M: usize = 512;
const DEFAULT_K: usize = 256;
const DEFAULT_L: usize = 1;

type MatrixA<'a> = BlockScaledTensor<'a, E2M1, UE8M0, SF_VEC, KMajor<Mkl>>;

type VectorB<'a> = BlockScaledTensor<'a, E2M1, UE8M0, SF_VEC, KMajor<Nkl>>;

#[cuda_module]
mod kernels {
    use super::*;

    /// One thread computes one C row, matching the CuTeDSL reference kernel.
    ///
    /// # Safety
    ///
    /// Launch exactly `block=(128,1,1)` and `grid=(m/128,1,L)`, with positive
    /// `m % 128 == 0` and `k % 64 == 0`. A/B must contain at least
    /// `L*m*k/2` and `L*k/2` packed bytes, with 16-byte-aligned bases. SFA/SFB
    /// must use the canonical layout and contain at least
    /// `L*ceil(m/128)*ceil(k/64)*512` and `L*ceil(k/64)*512` bytes, with
    /// four-byte-aligned bases. C must contain `L*m` disjoint f16 elements.
    /// No other block/grid axes may contain work.
    #[kernel]
    pub fn nvfp4_gemv(
        a: &[u8],
        b: &[u8],
        sfa: &[u8],
        sfb: &[u8],
        mut c: DisjointSlice<f16>,
        m: usize,
        k: usize,
    ) {
        let row = thread::blockIdx_x() as usize * CTA_M + thread::threadIdx_x() as usize;
        let batch = thread::blockIdx_z() as usize;

        // Bind each value tensor to its scale tensor once. Element formats,
        // logical mode order, scale-vector size, and physical scale layout
        // are now part of these Rust types rather than comments on byte
        // slices.
        let a = MatrixA::from_slices(a, sfa, m, k);
        let b = VectorB::from_slices(b, sfb, 1, k);
        let a_row = a.thread_row(batch, row);
        let b_row = b.thread_row(batch, 0);

        let mut acc = 0.0f32;
        // Advance one logical K tile at a time.
        let mut k_base = 0usize;

        while k_base < k {
            let k_tile = k_base / K_TILE;
            // k_tile creates typed views only. load performs two 16-byte value
            // transactions plus one four-byte scale transaction per operand,
            // then hoists each scale conversion out of the dot product.
            let (a_tile, b_tile) =
                unsafe { (a_row.k_tile(k_tile).load(), b_row.k_tile(k_tile).load()) };
            acc = a_tile.dot_accumulate(b_tile, acc);
            k_base += K_TILE;
        }

        // SAFETY: (batch, row) is unique for every live thread and the host
        // allocated L*M output elements.
        unsafe {
            c.as_mut_ptr().add(batch * m + row).write(acc as f16);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Args {
    m: usize,
    k: usize,
    l: usize,
    device: usize,
}

fn usage() -> ! {
    println!("Usage: nvfp4_gemv_cute [--m M] [--k K] [--l L] [--device ORDINAL]");
    std::process::exit(0);
}

fn parse_args() -> Args {
    let mut parsed = Args {
        m: DEFAULT_M,
        k: DEFAULT_K,
        l: DEFAULT_L,
        device: 0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            usage();
        }
        let value = args
            .next()
            .unwrap_or_else(|| panic!("missing value after {flag}"));
        match flag.as_str() {
            "--m" => parsed.m = value.parse().expect("invalid --m"),
            "--k" => parsed.k = value.parse().expect("invalid --k"),
            "--l" => parsed.l = value.parse().expect("invalid --l"),
            "--device" => parsed.device = value.parse().expect("invalid --device"),
            _ => panic!("unknown argument {flag}; use --help"),
        }
    }
    assert!(
        parsed.m > 0 && parsed.m.is_multiple_of(CTA_M),
        "M must be a positive multiple of 128"
    );
    assert!(
        parsed.k > 0 && parsed.k.is_multiple_of(K_TILE),
        "K must be a positive multiple of 64"
    );
    assert!(parsed.l > 0, "L must be positive");
    validate_problem_sizes(parsed.m, parsed.k, parsed.l);
    parsed
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn checked_product(label: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{label} size overflows usize"))
}

fn validate_problem_sizes(m: usize, k: usize, l: usize) {
    let _ = checked_product("packed A", &[l, m, k / 2]);
    let _ = checked_product("packed B", &[l, k / 2]);
    let _ = checked_product("C", &[l, m]);
    let _ = scale_storage_len(l, m, k);
    let _ = scale_storage_len(l, 1, k);
    let _ = u32::try_from(ceil_div(m, CTA_M)).expect("grid.x exceeds u32");
    let _ = u32::try_from(l).expect("grid.z exceeds u32");
}

fn e2m1_to_f32(nibble: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = VALUES[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn ue8m0_to_f32(byte: u8) -> f32 {
    if byte == 0 {
        f32::from_bits(0x0040_0000)
    } else if byte == 0xff {
        f32::from_bits(0x7fff_ffff)
    } else {
        f32::from_bits((byte as u32) << 23)
    }
}

fn a_nibble(batch: usize, row: usize, k: usize) -> u8 {
    (((batch % 16) * 5 + (row % 16) * 13 + (k % 16) * 7 + 3) % 16) as u8
}

fn b_nibble(batch: usize, k: usize) -> u8 {
    (((batch % 16) * 3 + (k % 16) * 11 + 1) % 16) as u8
}

fn sfa_byte(batch: usize, row: usize, k_group: usize) -> u8 {
    126 + ((batch % 3 + row % 3 + k_group % 3) % 3) as u8
}

fn sfb_byte(batch: usize, k_group: usize) -> u8 {
    126 + ((batch % 3 + 2 * (k_group % 3)) % 3) as u8
}

fn pack_a(m: usize, k: usize, l: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(checked_product("packed A", &[l, m, k / 2]));
    for batch in 0..l {
        for row in 0..m {
            for byte_k in 0..k / 2 {
                let low = a_nibble(batch, row, 2 * byte_k);
                let high = a_nibble(batch, row, 2 * byte_k + 1);
                packed.push(low | (high << 4));
            }
        }
    }
    packed
}

fn pack_b(k: usize, l: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(checked_product("packed B", &[l, k / 2]));
    for batch in 0..l {
        for byte_k in 0..k / 2 {
            let low = b_nibble(batch, 2 * byte_k);
            let high = b_nibble(batch, 2 * byte_k + 1);
            packed.push(low | (high << 4));
        }
    }
    packed
}

fn scale_storage_len(l: usize, mn: usize, k: usize) -> usize {
    Sm1xxBlockScaleKMajor::<SF_VEC>::new(mn, k).storage_len(l)
}

fn scale_index(batch: usize, row: usize, k_group: usize, mn: usize, k: usize) -> usize {
    Sm1xxBlockScaleKMajor::<SF_VEC>::new(mn, k).offset(batch, row, k_group)
}

fn make_scales(m: usize, k: usize, l: usize) -> (Vec<u8>, Vec<u8>) {
    // CuTeDSL's f32-zero initialization converts unused canonical-layout
    // padding to raw UE8M0 byte zero. Valid coordinates overwrite it below.
    let mut sfa = vec![0u8; scale_storage_len(l, m, k)];
    let mut sfb = vec![0u8; scale_storage_len(l, 1, k)];
    for batch in 0..l {
        for row in 0..m {
            for k_group in 0..k / SF_VEC {
                sfa[scale_index(batch, row, k_group, m, k)] = sfa_byte(batch, row, k_group);
            }
        }
        for k_group in 0..k / SF_VEC {
            sfb[scale_index(batch, 0, k_group, 1, k)] = sfb_byte(batch, k_group);
        }
    }
    (sfa, sfb)
}

fn reference(m: usize, k: usize, l: usize) -> Vec<f16> {
    let mut expected = vec![0.0f16; checked_product("C", &[l, m])];
    for batch in 0..l {
        for row in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let k_group = kk / SF_VEC;
                acc += ((e2m1_to_f32(a_nibble(batch, row, kk))
                    * ue8m0_to_f32(sfa_byte(batch, row, k_group)))
                    * e2m1_to_f32(b_nibble(batch, kk)))
                    * ue8m0_to_f32(sfb_byte(batch, k_group));
            }
            expected[batch * m + row] = acc as f16;
        }
    }
    expected
}

fn validate(got: &[f16], expected: &[f16]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&actual, &want)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            want.to_bits(),
            "bit mismatch at output {index}: expected {} ({:#06x}), got {} ({:#06x})",
            want as f32,
            want.to_bits(),
            actual as f32,
            actual.to_bits(),
        );
    }
    println!(
        "correctness: all {} f16 values bit-exact, output_hash={:016x}",
        got.len(),
        output_hash(got)
    );
}

fn output_hash(values: &[f16]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn main() {
    let args = parse_args();
    println!(
        "NVFP4 GEMV cute-rs typed block-scaled kernel: M={} K={} L={}",
        args.m, args.k, args.l
    );

    let a = pack_a(args.m, args.k, args.l);
    let b = pack_b(args.k, args.l);
    let (sfa, sfb) = make_scales(args.m, args.k, args.l);
    let expected = reference(args.m, args.k, args.l);

    let ctx = CudaContext::new(args.device).expect("failed to create CUDA context");
    let (major, minor) = ctx
        .compute_capability()
        .expect("failed to query compute capability");
    let device_name = ctx.device_name().expect("failed to query device name");
    println!(
        "device={} name={device_name:?} compute_capability=sm_{}{}",
        args.device, major, minor,
    );
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load embedded CUDA module");

    let a_dev = DeviceBuffer::from_host(&stream, &a).expect("copy A");
    let b_dev = DeviceBuffer::from_host(&stream, &b).expect("copy b");
    let sfa_dev = DeviceBuffer::from_host(&stream, &sfa).expect("copy SFA");
    let sfb_dev = DeviceBuffer::from_host(&stream, &sfb).expect("copy SFB");
    let mut c_dev = DeviceBuffer::<f16>::zeroed(&stream, checked_product("C", &[args.l, args.m]))
        .expect("allocate C");
    let config = LaunchConfig {
        grid_dim: (
            u32::try_from(ceil_div(args.m, CTA_M)).expect("grid.x exceeds u32"),
            1,
            u32::try_from(args.l).expect("grid.z exceeds u32"),
        ),
        block_dim: (CTA_M as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let max_grid = ctx
        .launch_limits()
        .expect("failed to query launch limits")
        .max_grid_dim();
    assert!(
        config.grid_dim.0 <= max_grid.0 && config.grid_dim.2 <= max_grid.2,
        "grid {:?} exceeds device maximum {:?}",
        config.grid_dim,
        max_grid
    );

    // SAFETY: all buffers match the documented packed/canonical layouts;
    // the exact grid covers M rows, and each thread owns one C value.
    unsafe {
        module.nvfp4_gemv(
            &stream, config, &a_dev, &b_dev, &sfa_dev, &sfb_dev, &mut c_dev, args.m, args.k,
        )
    }
    .expect("GEMV launch failed");
    stream.synchronize().expect("GEMV synchronization failed");
    let got = c_dev.to_host_vec(&stream).expect("copy C to host");
    validate(&got, &expected);
    println!("NVFP4 GEMV passed its bit-exact correctness check");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn e2m1_decode_covers_all_nibbles() {
        let positives = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (nibble, expected) in positives.into_iter().enumerate() {
            assert_eq!(e2m1_to_f32(nibble as u8), expected);
            assert_eq!(e2m1_to_f32(nibble as u8 | 8), -expected);
        }
        assert_eq!(e2m1_to_f32(0).to_bits(), 0x0000_0000);
        assert_eq!(e2m1_to_f32(8).to_bits(), 0x8000_0000);
    }

    #[test]
    fn ue8m0_decode_covers_boundaries_and_nan() {
        assert_eq!(ue8m0_to_f32(0).to_bits(), 0x0040_0000);
        assert_eq!(ue8m0_to_f32(1).to_bits(), 0x0080_0000);
        assert_eq!(ue8m0_to_f32(126), 0.5);
        assert_eq!(ue8m0_to_f32(127), 1.0);
        assert_eq!(ue8m0_to_f32(128), 2.0);
        assert_eq!(ue8m0_to_f32(254).to_bits(), 0x7f00_0000);
        assert!(ue8m0_to_f32(255).is_nan());
    }

    #[test]
    fn canonical_scale_indices_are_unique_and_in_bounds() {
        let (m, k, l) = (256, 128, 2);
        let len = scale_storage_len(l, m, k);
        let mut indices = BTreeSet::new();
        for batch in 0..l {
            for row in 0..m {
                for group in 0..k / SF_VEC {
                    let index = scale_index(batch, row, group, m, k);
                    assert!(index < len);
                    assert!(indices.insert(index));
                }
            }
        }
        assert_eq!(indices.len(), l * m * k / SF_VEC);

        // Boundaries of the public ((32,4), (SFVec,4)) 512-byte atom:
        // row-32 rotates the second M mode, row-128 advances the M atom,
        // group-4 advances the K atom, and batch follows both rest modes.
        assert_eq!(scale_index(0, 0, 0, m, k), 0);
        assert_eq!(scale_index(0, 31, 0, m, k), 496);
        assert_eq!(scale_index(0, 32, 0, m, k), 4);
        assert_eq!(scale_index(0, 127, 3, m, k), 511);
        assert_eq!(scale_index(0, 0, 4, m, k), 512);
        assert_eq!(scale_index(0, 128, 0, m, k), 1_024);
        assert_eq!(scale_index(1, 0, 0, m, k), 2_048);
    }

    #[test]
    fn nvfp4_packing_uses_low_nibble_first() {
        let packed = pack_b(64, 1);
        assert_eq!(packed[0], b_nibble(0, 0) | (b_nibble(0, 1) << 4));
    }
}
