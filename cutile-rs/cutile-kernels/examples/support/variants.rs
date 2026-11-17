/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::cuda_core::Stream;
use cutile::half::f16;
use cutile::prelude::*;
use cutile_kernels::experimental::{attention_f16, gemm_variants, moe_f16};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn values(n: usize, offset: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i * 13 + offset) % 31) as f32 / 64.0 - 0.25))
        .collect()
}
fn upload(v: &[f16], shape: &[usize], stream: &Arc<Stream>) -> Result<Tensor<f16>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v.to_vec()))
        .reshape(shape)
        .sync_on(stream)?)
}
fn compare(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> Result {
    if actual.len() != reference.len() {
        return Err("output length mismatch".into());
    }
    for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        if !a.is_finite() || !r.is_finite() || (a - r).abs() > atol + rtol * r.abs() {
            return Err(format!("element {i}: got {a}, expected {r}").into());
        }
    }
    Ok(())
}

/// Exercise all memory variants, including non-divisible M, N and K.
pub fn gemm() -> Result {
    use gemm_variants::kernels as k;
    let stream = Context::new(0)?.new_stream()?;
    for (m, n, inner) in [(48usize, 80usize, 96usize), (35, 53, 71)] {
        let av = values(m * inner, 3);
        let bv = values(inner * n, 7);
        let mut reference = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                reference[row * n + col] = (0..inner)
                    .map(|j| av[row * inner + j].to_f32() * bv[j * n + col].to_f32())
                    .sum();
            }
        }
        let a = Arc::new(upload(&av, &[m, inner], &stream)?);
        let b = Arc::new(upload(&bv, &[inner, n], &stream)?);
        for mode in ["tma", "store", "non-tma", "transpose", "gather"] {
            // TMA tensor strides must meet the target's alignment requirements.
            if (inner % 8 != 0 || n % 8 != 0) && mode != "gather" {
                continue;
            }
            let out = api::full(f32::NAN, &[m, n]).sync_on(&stream)?;
            let partition = out.partition([32, 32]);
            let mut generics = vec!["32".into(), "32".into(), "32".into(), "1".into()];
            let result = if mode == "gather" {
                generics.extend(["1024".into(), "1024".into()]);
                // SAFETY: owned contiguous A/B match M,K,N; C is distinct. These
                // small fixtures and their padded coordinates fit i32.
                unsafe { k::gemm_gather(partition, a.clone(), b.clone()) }
                    .generics(generics)
                    .sync_on(&stream)?
            } else if mode == "transpose" {
                k::gemm_transposed(partition, a.clone(), b.clone())
                    .generics(generics)
                    .sync_on(&stream)?
            } else {
                generics.push(
                    match mode {
                        "tma" => "0",
                        "store" => "1",
                        _ => "2",
                    }
                    .into(),
                );
                k::gemm(partition, a.clone(), b.clone())
                    .generics(generics)
                    .sync_on(&stream)?
            };
            let actual = result.0.unpartition().to_host_vec().sync_on(&stream)?;
            compare(&actual, &reference, 0.002, 0.002)?;
            println!("GEMM {m}x{n}x{inner} {mode}: PASS");
        }
    }
    Ok(())
}

/// Test all causal algorithms, with multiple heads and incomplete final tiles.
pub fn attention() -> Result {
    use attention_f16::kernels as k;
    let stream = Context::new(0)?.new_stream()?;
    const D: usize = 32;
    const HEADS: usize = 2;
    for (s, kind, bm, bn, extra) in [
        (19usize, "single", 16usize, 32usize, 1usize),
        (73, "tail", 16, 64, 16),
        (279, "causal", 64, 128, 0),
    ] {
        let qv = values(s * HEADS * D, 1);
        let kv = values(qv.len(), 9);
        let vv = values(qv.len(), 21);
        let q = upload(&qv, &[s, HEADS, D], &stream)?;
        let keys = upload(&kv, &[s, HEADS, D], &stream)?;
        let v = upload(&vv, &[s, HEADS, D], &stream)?;
        let scale = (D as f64).sqrt().recip() as f32;
        let scale_log2 = ((D as f64).sqrt().recip() * std::f64::consts::LOG2_E) as f32;
        let mut reference = vec![0.0; qv.len()];
        for row in 0..s {
            for head in 0..HEADS {
                let scores = (0..=row)
                    .map(|col| {
                        (0..D)
                            .map(|d| {
                                qv[(row * HEADS + head) * D + d].to_f32()
                                    * kv[(col * HEADS + head) * D + d].to_f32()
                            })
                            .sum::<f32>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let p = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
                let den: f32 = p.iter().sum();
                for d in 0..D {
                    reference[(row * HEADS + head) * D + d] = p
                        .iter()
                        .enumerate()
                        .map(|(col, p)| p * vv[(col * HEADS + head) * D + d].to_f32())
                        .sum::<f32>()
                        / den;
                }
            }
        }
        let out = api::full(f32::NAN, &[s, HEADS, D])
            .sync_on(&stream)?
            .partition([bm, 1, D]);
        let mut g = vec![bm.to_string(), bn.to_string(), D.to_string()];
        let result = if kind == "causal" {
            g.push(s.to_string());
            k::causal(out, q, keys, v, value(scale_log2))
                .generics(g)
                .const_grid((s.div_ceil(bm) as u32, HEADS as u32, 1))
                .sync_on(&stream)?
        } else {
            g.extend([extra.to_string(), s.to_string()]);
            if kind == "single" {
                k::single(out, q, keys, v, value(scale_log2))
                    .generics(g)
                    .sync_on(&stream)?
            } else {
                k::tail(out, q, keys, v, value(scale_log2))
                    .generics(g)
                    .sync_on(&stream)?
            }
        };
        let actual = result.0.unpartition().to_host_vec().sync_on(&stream)?;
        // The kernel rounds softmax probabilities to f16 before the value MMA;
        // the CPU reference deliberately retains f32 probabilities.
        compare(&actual, &reference, 0.0005, 0.005)?;
        println!("causal attention S={s}, heads={HEADS}, {kind}: PASS");
    }
    Ok(())
}

fn checked_elements(dims: &[usize]) -> Result<usize> {
    let count = dims
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("element count overflow")?;
    if count == 0 || count > i32::MAX as usize {
        return Err("element count outside i32 range".into());
    }
    Ok(count)
}

/// Validate routed FP16 MoE against a host reference, including skewed routes.
pub fn moe() -> Result {
    use cutile_compiler::cuda_tile_runtime_utils::ToolkitCapabilities;
    use cutile_ir::{ir::Location, requirements::Feature};
    use moe_f16::kernels as k;
    // Only the MoE variants use explicit inbounds TMA loads. Discovery and
    // execution errors still propagate; only an unmet capability is a skip.
    let capabilities = ToolkitCapabilities::for_device(0)?;
    if let Err(reason) = Feature::Inbounds.check(&capabilities.target, &Location::Unknown) {
        eprintln!("SKIP kernel_variants/moe: {reason}");
        return Ok(());
    }
    let stream = Context::new(0)?.new_stream()?;
    const H: usize = 256;
    const I: usize = 256;
    const E: usize = 4;
    const R: usize = 2;
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 64;
    // Repeat large routing with and without materialization; every route goes
    // to expert 1 in the skewed cases, leaving the other experts empty.
    for (t, packed, skew) in [
        (5usize, false, false),
        (47, false, true),
        (69, false, true),
        (69, true, true),
    ] {
        let nr = checked_elements(&[t, R])?;
        let cap = nr
            .div_ceil(BM)
            .checked_add(E)
            .ok_or("task capacity overflow")?;
        let rows = if packed {
            checked_elements(&[cap, BM])?
        } else {
            nr
        };
        let pr = nr
            .checked_next_power_of_two()
            .ok_or("route padding overflow")?;
        let pg = nr
            .div_ceil(BM)
            .checked_next_power_of_two()
            .ok_or("task padding overflow")?;
        for dims in [
            &[E, I, H][..],
            &[E, H, I],
            &[E, nr],
            &[cap, 2],
            &[cap, BM, H],
            &[rows, I],
            &[t, H],
            &[pr, pr],
        ] {
            checked_elements(dims)?;
        }
        let xh = values(t * H, 5);
        let uph = values(E * I * H, 11);
        let downh = values(E * H * I, 17);
        let ids = (0..nr)
            .map(|j| if skew { 1i64 } else { (j % E) as i64 })
            .collect::<Vec<_>>();
        let rw = (0..nr)
            .map(|j| if j % R == 0 { 0.25f32 } else { 0.75f32 })
            .collect::<Vec<_>>();
        let x = upload(&xh, &[t, H], &stream)?;
        let up = upload(&uph, &[E, I, H], &stream)?;
        let down = upload(&downh, &[E, H, I], &stream)?;
        let experts = api::copy_host_vec_to_device(&Arc::new(ids.clone())).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(rw.clone())).sync_on(&stream)?;
        let map = api::zeros::<i32>(&[E, nr]).sync_on(&stream)?;
        let counts = api::zeros::<i32>(&[E]).sync_on(&stream)?;
        let tasks = api::zeros::<i32>(&[cap, 2]).sync_on(&stream)?;
        let total = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let input = api::zeros::<f16>(&[cap * BM, H]).sync_on(&stream)?;
        let activation = api::zeros::<f16>(&[rows, I]).sync_on(&stream)?;
        // Nonzero initial output verifies that preparation clears accumulation.
        let out = api::full(f32::NAN, &[t, H]).sync_on(&stream)?;
        let blocks = 4u32;
        // SAFETY: all buffers are distinct owned allocations with checked sizes;
        // IDs are valid, tile factors divide H/I, PR/PG cover NR, and CAP bounds
        // all tasks. Synchronizing each same-stream stage orders reads/writes
        // and retains all buffers. Small routing is used only for NR<=2*BM.
        unsafe {
            if nr <= 2 * BM {
                k::prepare_small(
                    experts.device_pointer(),
                    map.device_pointer(),
                    counts.device_pointer(),
                    tasks.device_pointer(),
                    total.device_pointer(),
                    out.device_pointer(),
                )
                .generics([nr, pr, BM, t * H].map(|x| x.to_string()).to_vec())
                .grid((blocks, 1, 1))
                .sync_on(&stream)?;
            } else {
                k::prepare_large(
                    experts.device_pointer(),
                    map.device_pointer(),
                    counts.device_pointer(),
                    out.device_pointer(),
                )
                .generics([nr, pr, t * H].map(|x| x.to_string()).to_vec())
                .grid((E as u32, 1, 1))
                .sync_on(&stream)?;
                k::schedule(
                    counts.device_pointer(),
                    tasks.device_pointer(),
                    total.device_pointer(),
                )
                .generics([E, pg, BM].map(|x| x.to_string()).to_vec())
                .grid((E as u32, 1, 1))
                .sync_on(&stream)?;
            }
            if packed {
                k::pack_input(
                    x.device_pointer(),
                    map.device_pointer(),
                    counts.device_pointer(),
                    tasks.device_pointer(),
                    total.device_pointer(),
                    input.device_pointer(),
                )
                .generics(
                    [H, R, BM, nr, cap, BM * 256]
                        .map(|x| x.to_string())
                        .to_vec(),
                )
                .grid((blocks, 1, 1))
                .sync_on(&stream)?;
            }
            let mut g = [H, I, E, R, BM, BN, BK, nr, cap]
                .map(|x| x.to_string())
                .to_vec();
            g.extend([
                packed.to_string(),
                (BM * BK).to_string(),
                (BM * BN).to_string(),
            ]);
            k::fc1(
                if packed {
                    input.device_pointer()
                } else {
                    x.device_pointer()
                },
                up.device_pointer(),
                map.device_pointer(),
                counts.device_pointer(),
                tasks.device_pointer(),
                total.device_pointer(),
                activation.device_pointer(),
            )
            .generics(g.clone())
            .grid((blocks, 1, 1))
            .sync_on(&stream)?;
            k::fc2(
                activation.device_pointer(),
                down.device_pointer(),
                map.device_pointer(),
                counts.device_pointer(),
                tasks.device_pointer(),
                total.device_pointer(),
                weights.device_pointer(),
                out.device_pointer(),
            )
            .generics(g)
            .grid((blocks, 1, 1))
            .sync_on(&stream)?;
        }
        let mut reference = vec![0.0f32; t * H];
        for route in 0..nr {
            let token = route / R;
            let expert = ids[route] as usize;
            let mut act = vec![f16::ZERO; I];
            for j in 0..I {
                let acc = (0..H)
                    .map(|d| xh[token * H + d].to_f32() * uph[(expert * I + j) * H + d].to_f32())
                    .sum::<f32>()
                    .max(0.0);
                act[j] = f16::from_f32(acc * acc);
            }
            for d in 0..H {
                reference[token * H + d] += (0..I)
                    .map(|j| act[j].to_f32() * downh[(expert * H + d) * I + j].to_f32())
                    .sum::<f32>()
                    * rw[route];
            }
        }
        let actual = out.to_host_vec().sync_on(&stream)?;
        compare(&actual, &reference, 0.003, 0.003)?;
        println!("FP16 MoE T={t}, packed={packed}, skew={skew}: PASS");
    }
    Ok(())
}
