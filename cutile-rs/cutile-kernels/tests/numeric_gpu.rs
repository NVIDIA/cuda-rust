/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU numeric coverage for the element-generic kernels — results checked
//! against host references, not just compilation. Device-only (run via the
//! Modal harness), so intentionally outside the CPU gate.

use std::sync::Arc;

use cutile::api;
use cutile::core::{bf16, f16, f8e4m3fn};
use cutile::prelude::*;
use cutile_kernels::argmax::{argmax_blocks, lm_head_argmax_blocks};
use cutile_kernels::attention::fmha_prefill_causal;
use cutile_kernels::attention::{
    fmha_causal, fmha_decode_gqa_split, fmha_prefill_gqa, flash_attn_causal_seq,
    flash_attn_causal_seq_dynpos, splitk_reduce_merge,
};
use cutile_kernels::embeddings::embedding_batch;
use cutile_kernels::kv_cache::{kv_cache_update_seq, kv_cache_update_seq_dynpos};
use cutile_kernels::norms::{add_rms_norm, qk_norm, rms_norm};
use cutile_kernels::pointwise::silu_mul_2d;
use cutile_kernels::pointwise::{add_2d, gather_row};
use cutile_kernels::positional::{qk_rope_dynpos, rope_seq, rope_seq_dynpos};

const STACK: usize = 64_000_000;

/// Element adapter so one test body can cover f16 and bf16.
trait Elem: DType {
    const TAG: &'static str;
    fn of(x: f32) -> Self;
    fn f32(self) -> f32;
}
impl Elem for f16 {
    const TAG: &'static str = "f16";
    fn of(x: f32) -> Self {
        f16::from_f32(x)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
}
impl Elem for bf16 {
    const TAG: &'static str = "bf16";
    fn of(x: f32) -> Self {
        bf16::from_f32(x)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
}

fn dev<E: Elem>(xs: &[f32]) -> Arc<Tensor<E>> {
    api::copy_host_vec_to_device(&Arc::new(xs.iter().map(|&x| E::of(x)).collect::<Vec<_>>()))
        .sync()
        .expect("upload")
        .into()
}

fn host<E: Elem>(t: Tensor<E>) -> Vec<f32> {
    t.to_host_vec()
        .sync()
        .expect("readback")
        .into_iter()
        .map(|v| v.f32())
        .collect()
}

fn on_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("thread panicked");
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: [{i}] got {g} want {w} (tol {tol})"
        );
    }
}

// Generic attention against a host softmax reference.
fn run_fmha<E: Elem>() {
    const D: usize = 4;
    const N: usize = 4; // kv_len == BN, EVEN_K, CAUSAL off -> full softmax
    let q = [0.5f32, -0.5, 1.0, 0.0];
    let k = [
        1.0f32, 0.0, 0.0, 0.0, // key 0
        0.0, 1.0, 0.0, 0.0, // key 1
        1.0, 1.0, 0.0, 0.0, // key 2
        0.0, 0.0, 1.0, 1.0, // key 3
    ];
    let v = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let scale = 1.0f32;

    let mut scores = [0.0f32; N];
    for (j, s) in scores.iter_mut().enumerate() {
        *s = scale * (0..D).map(|d| q[d] * k[j * D + d]).sum::<f32>();
    }
    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
    let denom: f32 = exps.iter().sum();
    let mut want = [0.0f32; D];
    for (j, e) in exps.iter().enumerate() {
        for d in 0..D {
            want[d] += (e / denom) * v[j * D + d];
        }
    }

    let qt = dev::<E>(&q).reshape(&[1, 1, D]).expect("q shape"); // [q_len, q_heads, D]
    let kt = dev::<E>(&k).reshape(&[1, N, D]).expect("k shape"); // [kv_heads, kv_len, D]
    let vt = dev::<E>(&v).reshape(&[1, N, D]).expect("v shape");
    let mut out = api::zeros::<E>(&[1, 1, D])
        .sync()
        .expect("out")
        .reshape(&[1, 1, D])
        .expect("o");

    let _ = unsafe {
        fmha_prefill_causal(
            &qt,
            &kt,
            &vt,
            (&mut out).partition([1, 1, D]),
            scale,
            1,        // query_group_size
            N as i32, // kv_len
            0,        // query_start
        )
    }
    .generics(
        [E::TAG, "1", "4", "4", "0", "1", "1"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
    .sync()
    .expect("launch fmha");

    assert_close(&host::<E>(out), &want, 0.15, &format!("fmha<{}>", E::TAG));
}

#[test]
fn fmha_prefill_causal_matches_reference() {
    on_stack(|| {
        run_fmha::<f16>();
        run_fmha::<bf16>();
    });
}

// Embedding gather over bf16.
#[test]
fn embedding_batch_gathers_exact() {
    on_stack(|| {
        const D: usize = 4;
        let table = [
            10.0f32, 11.0, 12.0, 13.0, // row 0
            20.0, 21.0, 22.0, 23.0, // row 1
            30.0, 31.0, 32.0, 33.0, // row 2
        ];
        let ids: Vec<u32> = vec![2, 0];
        let table_t = dev::<bf16>(&table).reshape(&[3, D]).expect("table shape");
        let ids_t: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(ids))
            .sync()
            .expect("ids")
            .into();
        let mut out = api::zeros::<bf16>(&[2, D])
            .sync()
            .expect("out")
            .reshape(&[2, D])
            .expect("o");

        let _ = embedding_batch(&ids_t, &table_t, (&mut out).partition([1, D]))
            .sync()
            .expect("launch embedding_batch");

        let want = [30.0f32, 31.0, 32.0, 33.0, 10.0, 11.0, 12.0, 13.0];
        assert_close(&host::<bf16>(out), &want, 0.0, "embedding_batch<bf16>");
    });
}

// SiLU*mul over bf16 against an f32 reference.
#[test]
fn silu_mul_matches_reference() {
    on_stack(|| {
        const B: usize = 8;
        let gate = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let up = [1.0f32, 2.0, -1.0, 0.5, 1.0, -2.0, 0.25, 1.5];
        let want: Vec<f32> = (0..B)
            .map(|i| {
                let s = 1.0 / (1.0 + (-gate[i]).exp());
                s * gate[i] * up[i]
            })
            .collect();

        let gate_t = dev::<bf16>(&gate).reshape(&[1, B]).expect("gate shape");
        let up_t = dev::<bf16>(&up).reshape(&[1, B]).expect("up shape");
        let mut out = api::zeros::<bf16>(&[1, B])
            .sync()
            .expect("out")
            .reshape(&[1, B])
            .expect("o");

        let _ = silu_mul_2d((&mut out).partition([1, B]), &gate_t, &up_t)
            .sync()
            .expect("launch silu_mul_2d");

        assert_close(&host::<bf16>(out), &want, 0.05, "silu_mul_2d<bf16>");
    });
}

// f8 storage with f32 computation; tolerance reflects f8 rounding.
#[test]
fn silu_mul_f8e4m3_matches_reference() {
    on_stack(|| {
        const B: usize = 8;
        let gate = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 3.0]; // all f8e4m3-exact
        let up = [1.0f32; 8];
        let want: Vec<f32> = (0..B)
            .map(|i| (1.0 / (1.0 + (-gate[i]).exp())) * gate[i] * up[i])
            .collect();

        let f32dev = |xs: &[f32]| -> Arc<Tensor<f32>> {
            api::copy_host_vec_to_device(&Arc::new(xs.to_vec()))
                .sync()
                .expect("upload")
                .into()
        };
        let gate_t = api::convert::<f32, f8e4m3fn>(f32dev(&gate))
            .sync()
            .expect("gate f8")
            .reshape(&[1, B])
            .expect("gate shape");
        let up_t = api::convert::<f32, f8e4m3fn>(f32dev(&up))
            .sync()
            .expect("up f8")
            .reshape(&[1, B])
            .expect("up shape");
        let mut out = api::zeros::<f8e4m3fn>(&[1, B])
            .sync()
            .expect("out")
            .reshape(&[1, B])
            .expect("o");

        let _ = silu_mul_2d((&mut out).partition([1, B]), &gate_t, &up_t)
            .sync()
            .expect("launch silu_mul_2d");

        let out = out.reshape(&[B]).expect("flatten out"); // api::convert takes rank-1
        let got: Vec<f32> = api::convert::<f8e4m3fn, f32>(Arc::new(out))
            .sync()
            .expect("out f32")
            .to_host_vec()
            .sync()
            .expect("read");
        assert_close(&got, &want, 0.3, "silu_mul_2d<f8e4m3fn>");
    });
}

fn gen(tags: &[&str]) -> Vec<String> {
    tags.iter().map(|s| s.to_string()).collect()
}
fn dev_f32(xs: &[f32]) -> Arc<Tensor<f32>> {
    api::copy_host_vec_to_device(&Arc::new(xs.to_vec()))
        .sync()
        .expect("upload f32")
        .into()
}
fn dev_u32(xs: &[u32]) -> Arc<Tensor<u32>> {
    api::copy_host_vec_to_device(&Arc::new(xs.to_vec()))
        .sync()
        .expect("upload u32")
        .into()
}
fn read_f32(t: Tensor<f32>) -> Vec<f32> {
    t.to_host_vec().sync().expect("read f32")
}
fn read_u32(t: Tensor<u32>) -> Vec<u32> {
    t.to_host_vec().sync().expect("read u32")
}

fn rms_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..x.len()).map(|i| x[i] * inv * w[i]).collect()
}

fn rope_ref(v: &[f32; 4], pos: f32, invf: &[f32; 2]) -> [f32; 4] {
    let (lo0, lo1, hi0, hi1) = (v[0], v[1], v[2], v[3]);
    let (t0, t1) = (pos * invf[0], pos * invf[1]);
    [
        lo0 * t0.cos() - hi0 * t0.sin(),
        lo1 * t1.cos() - hi1 * t1.sin(),
        hi0 * t0.cos() + lo0 * t0.sin(),
        hi1 * t1.cos() + lo1 * t1.sin(),
    ]
}

#[test]
fn rms_norm_matches_reference() {
    on_stack(|| {
        const N: usize = 8;
        let x = [0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 1.0];
        let w = [1.0f32, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0];
        let eps = 1.0e-5f32;
        let want = rms_ref(&x, &w, eps);

        let xt = dev::<bf16>(&x).reshape(&[1, N]).expect("x");
        let wt = dev::<bf16>(&w).reshape(&[N]).expect("w");
        let mut out = api::zeros::<bf16>(&[1, N])
            .sync()
            .expect("out")
            .reshape(&[1, N])
            .expect("o");

        let _ = unsafe { rms_norm(&xt, &wt, (&mut out).partition([1, N]), eps) }
            .generics(gen(&["bf16", "8", "4"]))
            .sync()
            .expect("launch rms_norm");

        assert_close(&host::<bf16>(out), &want, 0.05, "rms_norm<bf16>");
    });
}

#[test]
fn rope_seq_matches_reference() {
    on_stack(|| {
        const HALF: usize = 2;
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let invf = [1.0f32, 0.5];
        let pos = 2i32;
        let want = rope_ref(&[x[0], x[1], x[2], x[3]], pos as f32, &invf);

        let xt = dev::<bf16>(&x).reshape(&[1, 1, 4]).expect("x");
        let ft = dev_f32(&invf).reshape(&[HALF]).expect("invf");
        let mut out = api::zeros::<bf16>(&[1, 1, 4])
            .sync()
            .expect("out")
            .reshape(&[1, 1, 4])
            .expect("o");

        let _ = rope_seq(&xt, &ft, (&mut out).partition([1, 1, HALF]), pos)
            .generics(gen(&["bf16", "4", "2"]))
            .sync()
            .expect("launch rope_seq");

        assert_close(&host::<bf16>(out), &want, 0.1, "rope_seq<bf16>");
    });
}

#[test]
fn argmax_blocks_matches_reference() {
    on_stack(|| {
        let logits = [1.0f32, 3.0, -2.0, 5.0, 0.0, 4.0, 5.0, 2.0]; // max 5.0 first at idx 3
        let lt = dev::<bf16>(&logits).reshape(&[8]).expect("logits");
        let mut bmax = api::zeros::<f32>(&[1])
            .sync()
            .expect("bmax")
            .reshape(&[1])
            .expect("m");
        let mut bidx = api::zeros::<u32>(&[1])
            .sync()
            .expect("bidx")
            .reshape(&[1])
            .expect("i");

        let _ = argmax_blocks(
            &lt,
            (&mut bmax).partition([1]),
            (&mut bidx).partition([1]),
            8i32,
        )
        .generics(gen(&["bf16", "8"]))
        .sync()
        .expect("launch argmax_blocks");

        assert_close(&read_f32(bmax), &[5.0], 0.02, "argmax_blocks max");
        assert_eq!(read_u32(bidx), vec![3u32], "argmax_blocks idx");
    });
}

#[test]
fn add_2d_matches_reference() {
    on_stack(|| {
        const BLK: usize = 8;
        let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let want: Vec<f32> = (0..BLK).map(|i| lhs[i] + rhs[i]).collect();

        let lt = dev::<bf16>(&lhs).reshape(&[1, BLK]).expect("lhs");
        let rt = dev::<bf16>(&rhs).reshape(&[1, BLK]).expect("rhs");
        let mut out = api::zeros::<bf16>(&[1, BLK])
            .sync()
            .expect("out")
            .reshape(&[1, BLK])
            .expect("o");

        let _ = add_2d((&mut out).partition([1, BLK]), &lt, &rt)
            .generics(gen(&["bf16", "8"]))
            .sync()
            .expect("launch add_2d");

        assert_close(&host::<bf16>(out), &want, 0.0, "add_2d<bf16>");
    });
}

#[test]
fn gather_row_matches_reference() {
    on_stack(|| {
        const BLK: usize = 8;
        let src = [
            0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // row 0
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, // row 1
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, // row 2
        ];
        let st = dev::<bf16>(&src).reshape(&[3, BLK]).expect("src");
        let mut out = api::zeros::<bf16>(&[BLK])
            .sync()
            .expect("out")
            .reshape(&[BLK])
            .expect("o");

        let _ = gather_row(&st, (&mut out).partition([BLK]), 1i32)
            .generics(gen(&["bf16", "8"]))
            .sync()
            .expect("launch gather_row");

        let want = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        assert_close(&host::<bf16>(out), &want, 0.0, "gather_row<bf16>");
    });
}

#[test]
fn kv_cache_update_seq_matches_reference() {
    on_stack(|| {
        let new_k = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,1,4]
        let new_v = [9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let kt = dev::<bf16>(&new_k).reshape(&[2, 1, 4]).expect("k");
        let vt = dev::<bf16>(&new_v).reshape(&[2, 1, 4]).expect("v");
        let mut kc = api::zeros::<bf16>(&[1, 2, 4])
            .sync()
            .expect("kc")
            .reshape(&[1, 2, 4])
            .expect("kc");
        let mut vc = api::zeros::<bf16>(&[1, 2, 4])
            .sync()
            .expect("vc")
            .reshape(&[1, 2, 4])
            .expect("vc");

        let _ = unsafe {
            kv_cache_update_seq(
                &kt,
                &vt,
                (&mut kc).partition([1, 2, 4]),
                (&mut vc).partition([1, 2, 4]),
                0i32,
                2i32,
            )
        }
        .generics(gen(&["bf16", "4", "4", "2"]))
        .sync()
        .expect("launch kv_cache_update_seq");

        assert_close(&host::<bf16>(kc), &new_k, 0.0, "kv_cache k");
        assert_close(&host::<bf16>(vc), &new_v, 0.0, "kv_cache v");
    });
}

#[test]
fn add_rms_norm_matches_reference() {
    on_stack(|| {
        const N: usize = 8;
        let residual = [0.5f32, 1.0, -1.0, 0.25, 0.5, -0.5, 1.0, -1.0];
        let x = [0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 1.0];
        let w = [1.0f32, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0];
        let eps = 1.0e-5f32;
        let combined: Vec<f32> = (0..N).map(|i| residual[i] + x[i]).collect();
        let want = rms_ref(&combined, &w, eps);

        let rt = dev::<bf16>(&residual).reshape(&[1, N]).expect("res");
        let xt = dev::<bf16>(&x).reshape(&[1, N]).expect("x");
        let wt = dev::<bf16>(&w).reshape(&[N]).expect("w");
        let mut out = api::zeros::<bf16>(&[1, N])
            .sync()
            .expect("out")
            .reshape(&[1, N])
            .expect("o");
        let mut res_out = api::zeros::<bf16>(&[1, N])
            .sync()
            .expect("ro")
            .reshape(&[1, N])
            .expect("ro");

        let _ = unsafe {
            add_rms_norm(
                &rt,
                &xt,
                &wt,
                (&mut out).partition([1, N]),
                (&mut res_out).partition([1, N]),
                eps,
            )
        }
        .generics(gen(&["bf16", "8", "4"]))
        .sync()
        .expect("launch add_rms_norm");

        assert_close(&host::<bf16>(out), &want, 0.05, "add_rms_norm out");
        assert_close(
            &host::<bf16>(res_out),
            &combined,
            0.03,
            "add_rms_norm residual",
        );
    });
}

#[test]
fn qk_norm_matches_reference() {
    on_stack(|| {
        const N: usize = 8;
        let q = [0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 1.0];
        let k = [1.0f32, 1.0, -1.0, 0.5, 0.25, -0.25, 2.0, -1.0];
        let qw = [1.0f32, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0];
        let kw = [0.5f32, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0];
        let eps = 1.0e-5f32;
        let mut want = rms_ref(&q, &qw, eps);
        want.extend(rms_ref(&k, &kw, eps)); // row 0 = q, row 1 = k

        let qt = dev::<bf16>(&q).reshape(&[1, N]).expect("q");
        let kt = dev::<bf16>(&k).reshape(&[1, N]).expect("k");
        let qwt = dev::<bf16>(&qw).reshape(&[N]).expect("qw");
        let kwt = dev::<bf16>(&kw).reshape(&[N]).expect("kw");
        let mut out = api::zeros::<bf16>(&[2, N])
            .sync()
            .expect("out")
            .reshape(&[2, N])
            .expect("o");

        let _ = unsafe {
            qk_norm(
                &qt,
                &kt,
                &qwt,
                &kwt,
                (&mut out).partition([1, N]),
                eps,
                1i32,
            )
        }
        .generics(gen(&["bf16", "8", "4"]))
        .sync()
        .expect("launch qk_norm");

        assert_close(&host::<bf16>(out), &want, 0.05, "qk_norm<bf16>");
    });
}

#[test]
fn rope_seq_dynpos_matches_reference() {
    on_stack(|| {
        const HALF: usize = 2;
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let invf = [1.0f32, 0.5];
        let pos = 3u32;
        let want = rope_ref(&[x[0], x[1], x[2], x[3]], pos as f32, &invf);

        let xt = dev::<bf16>(&x).reshape(&[1, 1, 4]).expect("x");
        let ft = dev_f32(&invf).reshape(&[HALF]).expect("invf");
        let pt = dev_u32(&[pos]).reshape(&[1]).expect("pos");
        let mut out = api::zeros::<bf16>(&[1, 1, 4])
            .sync()
            .expect("out")
            .reshape(&[1, 1, 4])
            .expect("o");

        let _ = rope_seq_dynpos(&xt, &ft, &pt, (&mut out).partition([1, 1, HALF]))
            .generics(gen(&["bf16", "4", "2"]))
            .sync()
            .expect("launch rope_seq_dynpos");

        assert_close(&host::<bf16>(out), &want, 0.1, "rope_seq_dynpos<bf16>");
    });
}

#[test]
fn qk_rope_dynpos_matches_reference() {
    on_stack(|| {
        const HALF: usize = 2;
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let k = [0.5f32, -1.0, 2.0, 1.0];
        let invf = [1.0f32, 0.5];
        let pos = 3u32;
        let mut want = rope_ref(&q, pos as f32, &invf).to_vec(); // head 0 = q
        want.extend(rope_ref(&k, pos as f32, &invf)); // head 1 = k

        let qt = dev::<bf16>(&q).reshape(&[1, 1, 4]).expect("q");
        let kt = dev::<bf16>(&k).reshape(&[1, 1, 4]).expect("k");
        let ft = dev_f32(&invf).reshape(&[HALF]).expect("invf");
        let pt = dev_u32(&[pos]).reshape(&[1]).expect("pos");
        let mut out = api::zeros::<bf16>(&[1, 2, 4])
            .sync()
            .expect("out")
            .reshape(&[1, 2, 4])
            .expect("o");

        let _ =
            unsafe { qk_rope_dynpos(&qt, &kt, &ft, &pt, (&mut out).partition([1, 1, HALF]), 1i32) }
                .generics(gen(&["bf16", "4", "2", "1"]))
                .sync()
                .expect("launch qk_rope_dynpos");

        assert_close(&host::<bf16>(out), &want, 0.1, "qk_rope_dynpos<bf16>");
    });
}

#[test]
fn kv_cache_update_seq_dynpos_matches_reference() {
    on_stack(|| {
        let new_k = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,1,4]
        let new_v = [9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let kt = dev::<bf16>(&new_k).reshape(&[2, 1, 4]).expect("k");
        let vt = dev::<bf16>(&new_v).reshape(&[2, 1, 4]).expect("v");
        let pt = dev_u32(&[1]).reshape(&[1]).expect("pos");
        let mut kc = api::zeros::<bf16>(&[1, 4, 4])
            .sync()
            .expect("kc")
            .reshape(&[1, 4, 4])
            .expect("kc");
        let mut vc = api::zeros::<bf16>(&[1, 4, 4])
            .sync()
            .expect("vc")
            .reshape(&[1, 4, 4])
            .expect("vc");

        let _ = unsafe {
            kv_cache_update_seq_dynpos(
                &kt,
                &vt,
                (&mut kc).partition([1, 4, 4]),
                (&mut vc).partition([1, 4, 4]),
                &pt,
                2i32,
            )
        }
        .generics(gen(&["bf16", "4", "4", "4"]))
        .sync()
        .expect("launch kv_cache_update_seq_dynpos");

        let mut want_k = vec![0.0f32; 16];
        let mut want_v = vec![0.0f32; 16];
        for s in 0..2usize {
            for d in 0..4usize {
                want_k[(1 + s) * 4 + d] = new_k[s * 4 + d];
                want_v[(1 + s) * 4 + d] = new_v[s * 4 + d];
            }
        }
        assert_close(&host::<bf16>(kc), &want_k, 0.0, "kv_cache_dynpos k");
        assert_close(&host::<bf16>(vc), &want_v, 0.0, "kv_cache_dynpos v");
    });
}

#[test]
fn lm_head_argmax_blocks_matches_reference() {
    on_stack(|| {
        const VOCAB: usize = 64;
        const K: usize = 64;
        let mut weights = vec![0.0f32; VOCAB * K];
        for r in 0..VOCAB {
            weights[r * K] = r as f32;
        }
        let mut hidden = vec![0.0f32; K];
        hidden[0] = 1.0;

        let wt = dev::<bf16>(&weights).reshape(&[VOCAB, K]).expect("w");
        let ht = dev::<bf16>(&hidden).reshape(&[1, K]).expect("h");
        let mut bmax = api::zeros::<f32>(&[1])
            .sync()
            .expect("bmax")
            .reshape(&[1])
            .expect("m");
        let mut bidx = api::zeros::<u32>(&[1])
            .sync()
            .expect("bidx")
            .reshape(&[1])
            .expect("i");

        let _ = unsafe {
            lm_head_argmax_blocks(
                &wt,
                &ht,
                (&mut bmax).partition([1]),
                (&mut bidx).partition([1]),
                VOCAB as i32,
            )
        }
        .generics(gen(&["bf16", "64"]))
        .sync()
        .expect("launch lm_head_argmax_blocks");

        assert_close(&read_f32(bmax), &[63.0], 0.1, "lm_head max");
        assert_eq!(read_u32(bidx), vec![63u32], "lm_head idx");
    });
}

// Causal attention variants (fmha_prefill_causal above runs CAUSAL=0).

// out[i] = softmax_j( scale * q[i]·k[j] , over j with (!causal || qpos[i] >= kpos[j]) ) · v[j]
fn attn_ref(
    q: &[f32],
    qpos: &[i32],
    k: &[f32],
    kpos: &[i32],
    v: &[f32],
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let qn = qpos.len();
    let kn = kpos.len();
    let mut out = vec![0.0f32; qn * d];
    for i in 0..qn {
        let mut scores = vec![f32::NEG_INFINITY; kn];
        for j in 0..kn {
            if !causal || qpos[i] >= kpos[j] {
                let dot: f32 = (0..d).map(|t| q[i * d + t] * k[j * d + t]).sum();
                scores[j] = scale * dot;
            }
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for j in 0..kn {
            let w = exps[j] / denom;
            for t in 0..d {
                out[i * d + t] += w * v[j * d + t];
            }
        }
    }
    out
}

#[test]
fn flash_attn_causal_seq_matches_reference() {
    on_stack(|| {
        const D: usize = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 queries at pos 0,1
        let k = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 keys at pos 0,1
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let want = attn_ref(&q, &[0, 1], &k, &[0, 1], &v, D, 1.0, true);

        let qt = dev::<bf16>(&q).reshape(&[2, 1, D]).expect("q");
        let kt = dev::<bf16>(&k).reshape(&[1, 2, D]).expect("k");
        let vt = dev::<bf16>(&v).reshape(&[1, 2, D]).expect("v");
        let mut out = api::zeros::<bf16>(&[2, 1, D]).sync().expect("out").reshape(&[2, 1, D]).expect("o");

        let _ = unsafe {
            flash_attn_causal_seq(&qt, &kt, &vt, (&mut out).partition([2, 1, D]), 1.0f32, 1, 2, 0)
        }
        .generics(gen(&["bf16", "2", "2", "4"]))
        .sync()
        .expect("launch flash_attn_causal_seq");

        assert_close(&host::<bf16>(out), &want, 0.1, "flash_attn_causal_seq<bf16>");
    });
}

#[test]
fn fmha_causal_matches_reference() {
    on_stack(|| {
        const D: usize = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let want = attn_ref(&q, &[0, 1], &k, &[0, 1], &v, D, 1.0, true);

        let qt = dev::<bf16>(&q).reshape(&[2, 1, D]).expect("q");
        let kt = dev::<bf16>(&k).reshape(&[1, 2, D]).expect("k");
        let vt = dev::<bf16>(&v).reshape(&[1, 2, D]).expect("v");
        let pos = dev_u32(&[0]).reshape(&[1]).expect("pos");
        let mut out = api::zeros::<bf16>(&[2, 1, D]).sync().expect("out").reshape(&[2, 1, D]).expect("o");

        let _ = unsafe {
            fmha_causal(
                &qt,
                &kt,
                &vt,
                (&mut out).partition([2, 1, D]),
                bf16::from_f32(1.0),
                1,
                &pos,
            )
        }
        .generics(gen(&["bf16", "2", "2", "4", "1", "1"]))
        .sync()
        .expect("launch fmha_causal");

        assert_close(&host::<bf16>(out), &want, 0.1, "fmha_causal<bf16>");
    });
}

#[test]
fn flash_attn_causal_seq_dynpos_matches_reference() {
    on_stack(|| {
        // Decode: BM=1, one query at position P; kv_len = P + 1.
        const D: usize = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0]; // single query at pos 2
        let k = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // keys 0,1,2
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let want = attn_ref(&q, &[2], &k, &[0, 1, 2], &v, D, 1.0, true);

        let qt = dev::<bf16>(&q).reshape(&[1, 1, D]).expect("q");
        let kt = dev::<bf16>(&k).reshape(&[1, 3, D]).expect("k");
        let vt = dev::<bf16>(&v).reshape(&[1, 3, D]).expect("v");
        let pos = dev_u32(&[2]).reshape(&[1]).expect("pos");
        let mut out = api::zeros::<bf16>(&[1, 1, D]).sync().expect("out").reshape(&[1, 1, D]).expect("o");

        let _ = unsafe {
            flash_attn_causal_seq_dynpos(&qt, &kt, &vt, (&mut out).partition([1, 1, D]), 1.0f32, 1, &pos)
        }
        .generics(gen(&["bf16", "1", "2", "4"]))
        .sync()
        .expect("launch flash_attn_causal_seq_dynpos");

        assert_close(&host::<bf16>(out), &want, 0.1, "flash_attn_causal_seq_dynpos<bf16>");
    });
}

// GQA prefill: GROUP query heads share one kv-head; CAUSAL=0 -> full softmax.
#[test]
fn fmha_prefill_gqa_matches_reference() {
    on_stack(|| {
        const D: usize = 4;
        const GROUP: usize = 2;
        // 2 query heads (one group), 1 kv-head, 2 keys.
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // head0, head1
        let k = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let want = attn_ref(&q, &[0, 0], &k, &[0, 0], &v, D, 1.0, false);

        let qt = dev::<bf16>(&q).reshape(&[1, 2, D]).expect("q"); // [q_len, q_heads, D]
        let kt = dev::<bf16>(&k).reshape(&[1, 2, D]).expect("k"); // [kv_heads, kv_len, D]
        let vt = dev::<bf16>(&v).reshape(&[1, 2, D]).expect("v");
        let mut out = api::zeros::<bf16>(&[1, GROUP, D]).sync().expect("out").reshape(&[1, GROUP, D]).expect("o");

        let _ = unsafe {
            fmha_prefill_gqa(
                &qt,
                &kt,
                &vt,
                (&mut out).partition([1, GROUP, D]),
                1.0f32,
                2, // query_group_size
                2, // kv_len
                0, // query_start
            )
        }
        // [E,BM,BN,D,GROUP,M_EFF=BM*GROUP,CAUSAL,EVEN_K,LATENCY]
        .generics(gen(&["bf16", "1", "2", "4", "2", "2", "0", "1", "1"]))
        .sync()
        .expect("launch fmha_prefill_gqa");

        assert_close(&host::<bf16>(out), &want, 0.1, "fmha_prefill_gqa<bf16>");
    });
}

// Split-K decode: split emits per-split attention+LSE; merge recombines to full softmax.
#[test]
fn fmha_decode_gqa_split_merge_matches_reference() {
    on_stack(|| {
        const D: usize = 4;
        const GROUP: usize = 2;
        // 1 kv-head, GROUP=2 heads, s_kv=4 keys, NUM_KV_SPLITS=2 (2 keys each).
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let v = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let want = attn_ref(&q, &[0, 0], &k, &[0, 0, 0, 0], &v, D, 1.0, false);

        let qt = dev::<bf16>(&q).reshape(&[1, GROUP, D]).expect("q"); // [kv_heads, GROUP, D]
        let kt = dev::<bf16>(&k).reshape(&[1, 4, D]).expect("k");
        let vt = dev::<bf16>(&v).reshape(&[1, 4, D]).expect("v");
        let pos = dev_u32(&[3]).reshape(&[1]).expect("pos"); // s_kv = pos+1 = 4

        // Scratch: [kv_heads, NUM_KV_SPLITS*GROUP, D] and [kv_heads, NUM_KV_SPLITS*GROUP].
        let mut att = api::zeros::<bf16>(&[1, 4, D]).sync().expect("att").reshape(&[1, 4, D]).expect("a");
        let mut lse = api::zeros::<f32>(&[1, 4]).sync().expect("lse").reshape(&[1, 4]).expect("l");

        let _ = unsafe {
            fmha_decode_gqa_split(
                &qt,
                &kt,
                &vt,
                (&mut att).partition([1, GROUP, D]),
                (&mut lse).partition([1, GROUP]),
                bf16::from_f32(1.0),
                &pos,
            )
        }
        // [E,GROUP,BN,D,NUM_KV_SPLITS,LATENCY]
        .generics(gen(&["bf16", "2", "2", "4", "2", "1"]))
        .sync()
        .expect("launch fmha_decode_gqa_split");

        let mut out = api::zeros::<bf16>(&[1, GROUP, D]).sync().expect("out").reshape(&[1, GROUP, D]).expect("o");
        let _ = unsafe {
            splitk_reduce_merge(&att, &lse, (&mut out).partition([1, GROUP, D]))
        }
        // [E,GROUP,D,CHUNK_D,NUM_KV_SPLITS,NS_GROUP,LATENCY]
        .generics(gen(&["bf16", "2", "4", "4", "2", "4", "1"]))
        .sync()
        .expect("launch splitk_reduce_merge");

        assert_close(&host::<bf16>(out), &want, 0.1, "splitk decode<bf16>");
    });
}
