/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
//! GPU oracle for all eight standard-metadata sparse floating MMA forms.
//!
//! Why the inputs look the way they do:
//! - the first cut of the ordered oracle used val(x,y) = 1 + ((x+y) & 1); that
//!   parity pattern made the A and B fragments bitwise IDENTICAL in every
//!   lane, so an a<->b swap in the lowering still passed:
//!
//!   ```text
//!   old: A[r][q] = f(r+q)   B[k][c] = f(k+c)   -> same lane bytes
//!   new: A[r][j] = 1+(3r+5j)%7, B[k][c] = 1+(2k+3c)%8 -> distinct
//!   ```
//!
//! - 16-bit metadata cycles the seven supported codes
//!   (0x4 0x6 0x8 0x9 0xc 0xd 0xe) by (row+chunk). Decode stays
//!   `i0 = code & 3`, `i1 = (code >> 2) & 3`, so 0x6 keeps indices (2, 1).
//!   The decoy bump moves one slot forward in that same 7-code table;
//! - tf32 stays on 0x4 and 0xe only. 0x6 is not a tf32 code;
//! - every legal selector is a compile-time literal. The four 2-register
//!   forms each run 0, 1, 2, and 3. The four 4-register forms each run 0
//!   and 1. That is 24 calls. Non-selected lanes carry the decoy pattern,
//!   so a wrong selector reads a different metadata register and the sum
//!   changes. The selected lane always carries the real (bump 0) pattern,
//!   which is what the host GEMM checks;
//! - values stay small (A<=7, B<=8, C<=31, sums < 1024) so f16, bf16, and
//!   tf32 all accumulate exactly and the host reference is integer math.
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, wmma};

// Positionally distinct inputs: A over compressed (row, j), B over (k, col).
const fn a_val(r: u32, j: u32) -> u32 {
    1 + ((3 * r + 5 * j) % 7)
}
const fn b_val(k: u32, c: u32) -> u32 {
    1 + ((2 * k + 3 * c) % 8)
}
const fn c_val(r: u32, c: u32) -> u32 {
    (5 * r + 7 * c) % 32
}

// Standard metadata code for (row, chunk); bump=1 yields the decoy pattern
// (next code in the table) handed to non-selected lanes.
// tf=0: f16/bf16 7-code table; tf=1: tf32 2-code table (0x4 and 0xe only).
// The tf32 index mixes r/8 and g/3 so no wrong (lane, nibble) -> (row, chunk)
// reading can reproduce the true pattern by periodicity (rows 8 apart and
// chunks 4 apart would otherwise alias under a plain (r+g) parity).
const fn meta_code(r: u32, g: u32, bump: u32, tf: u32) -> u32 {
    if tf == 1 {
        if (r + r / 8 + g + g / 3 + bump) & 1 == 0 {
            0x4
        } else {
            0xe
        }
    } else {
        match (r + g + bump) % 7 {
            0 => 0x4,
            1 => 0x6,
            2 => 0x8,
            3 => 0x9,
            4 => 0xc,
            5 => 0xd,
            _ => 0xe,
        }
    }
}

// Exact bit patterns for small integers (all values here fit losslessly).
const fn f32_bits(x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    let mut e = 0;
    while (x >> e) > 1 {
        e += 1;
    }
    ((127 + e) << 23) | ((x << (23 - e)) & 0x007f_ffff)
}
const fn f16_bits(x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    let mut e = 0;
    while (x >> e) > 1 {
        e += 1;
    }
    ((15 + e) << 10) | ((x << (10 - e)) & 0x03ff)
}
const fn bf16_bits(x: u32) -> u32 {
    f32_bits(x) >> 16
}
const fn pack(lo: u32, hi: u32) -> u32 {
    lo | (hi << 16)
}

// 2-reg forms x selectors 0..=3, then 4-reg forms x selectors 0..=1.
const PACKED_RESULTS: usize = 6;
const F32_RESULTS: usize = 18;
const PACKED_STRIDE: usize = PACKED_RESULTS * 2;
const F32_STRIDE: usize = F32_RESULTS * 4;

#[cuda_module]
mod kernels {
    use super::*;

    // f16/bf16 A fragment register n: row l/4 (+8 for odd n),
    // compressed cols (l%4)*2 + 8*(n/2) and +1, packed lo/hi.
    fn af(l: u32, n: u32, bf: u32) -> u32 {
        let r = l / 4 + 8 * (n & 1);
        let j = (l & 3) * 2 + 8 * (n / 2);
        if bf == 1 {
            pack(bf16_bits(a_val(r, j)), bf16_bits(a_val(r, j + 1)))
        } else {
            pack(f16_bits(a_val(r, j)), f16_bits(a_val(r, j + 1)))
        }
    }
    // f16/bf16 B fragment register n: col l/4, rows (l%4)*2 + 8n and +1.
    fn bf(l: u32, n: u32, b16: u32) -> u32 {
        let c = l / 4;
        let k = (l & 3) * 2 + 8 * n;
        if b16 == 1 {
            pack(bf16_bits(b_val(k, c)), bf16_bits(b_val(k + 1, c)))
        } else {
            pack(f16_bits(b_val(k, c)), f16_bits(b_val(k + 1, c)))
        }
    }
    // tf32 A fragment register n: row l/4 (+8 odd n), compressed col (l%4)+4*(n/2).
    fn at(l: u32, n: u32) -> u32 {
        f32_bits(a_val(l / 4 + 8 * (n & 1), (l & 3) + 4 * (n / 2)))
    }
    // tf32 B fragment register n: row (l%4)+4n, col l/4.
    fn bt(l: u32, n: u32) -> u32 {
        f32_bits(b_val((l & 3) + 4 * n, l / 4))
    }
    // f32 accumulator quad: rows l/4, l/4+8; cols (l%4)*2, +1.
    fn cf(l: u32) -> [f32; 4] {
        let r = l / 4;
        let c = (l & 3) * 2;
        [
            c_val(r, c) as f32,
            c_val(r, c + 1) as f32,
            c_val(r + 8, c) as f32,
            c_val(r + 8, c + 1) as f32,
        ]
    }
    // packed f16 accumulator register n: row l/4+8n, cols (l%4)*2, +1.
    fn cp(l: u32, n: u32) -> u32 {
        let r = l / 4 + 8 * n;
        let c = (l & 3) * 2;
        pack(f16_bits(c_val(r, c)), f16_bits(c_val(r, c + 1)))
    }
    // One metadata thread per quad (k16 f16/bf16, k8 tf32; 4 chunks/row):
    //   bits [15:0] row g, bits [31:16] row g+8, nibble j = chunk j.
    // Only lane l%4 == sel carries the real pattern; the rest carry the decoy.
    fn meta_one(l: u32, sel: u32, tf: u32) -> u32 {
        let g = l / 4;
        let bump = if l % 4 == sel { 0 } else { 1 };
        let mut m = 0;
        let mut j = 0;
        while j < 4 {
            m |= meta_code(g, j, bump, tf) << (4 * j);
            m |= meta_code(g + 8, j, bump, tf) << (16 + 4 * j);
            j += 1;
        }
        m
    }
    // Metadata thread pair per quad (k32 f16/bf16, k16 tf32; 8 chunks/row):
    //   selector s picks lanes {2s, 2s+1} of the quad, and the pair splits
    //   the K dimension, keeping the one-thread [row g | row g+8] register
    //   split (verified against sm_120 hardware; a row-per-thread split
    //   fails the oracle):
    //
    //     lane 2s+t: bits [15:0] row g   chunks 4t..4t+3
    //                bits [31:16] row g+8 chunks 4t..4t+3
    fn meta_pair(l: u32, sel: u32, tf: u32) -> u32 {
        let g = l / 4;
        let t = (l & 3) % 2;
        let bump = if (l & 3) / 2 == sel { 0 } else { 1 };
        let mut m = 0;
        let mut j = 0;
        while j < 4 {
            m |= meta_code(g, 4 * t + j, bump, tf) << (4 * j);
            m |= meta_code(g + 8, 4 * t + j, bump, tf) << (16 + 4 * j);
            j += 1;
        }
        m
    }

    // The selector argument is an ImmArg. Each invocation passes a literal.
    macro_rules! store_packed {
        ($po:ident, $l:ident, $slot:literal, $value:expr) => {{
            let base = $l as usize * PACKED_STRIDE + $slot * 2;
            let value = $value;
            unsafe {
                *$po.get_unchecked_mut(base) = value[0];
                *$po.get_unchecked_mut(base + 1) = value[1];
            }
        }};
    }
    macro_rules! store_f32 {
        ($fo:ident, $l:ident, $slot:literal, $value:expr) => {{
            let base = $l as usize * F32_STRIDE + $slot * 4;
            let value = $value;
            unsafe {
                *$fo.get_unchecked_mut(base) = value[0];
                *$fo.get_unchecked_mut(base + 1) = value[1];
                *$fo.get_unchecked_mut(base + 2) = value[2];
                *$fo.get_unchecked_mut(base + 3) = value[3];
            }
        }};
    }
    macro_rules! call_f16_k16 {
        ($po:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_packed!($po, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k16_f16_f16(
                    [cp($l, 0), cp($l, 1)],
                    [af($l, 0, 0), af($l, 1, 0)],
                    [bf($l, 0, 0), bf($l, 1, 0)],
                    meta_one($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_f16_k32 {
        ($po:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_packed!($po, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k32_f16_f16(
                    [cp($l, 0), cp($l, 1)],
                    [af($l, 0, 0), af($l, 1, 0), af($l, 2, 0), af($l, 3, 0)],
                    [bf($l, 0, 0), bf($l, 1, 0), bf($l, 2, 0), bf($l, 3, 0)],
                    meta_pair($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_f32_f16_k16 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k16_f32_f16(
                    cf($l),
                    [af($l, 0, 0), af($l, 1, 0)],
                    [bf($l, 0, 0), bf($l, 1, 0)],
                    meta_one($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_f32_f16_k32 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k32_f32_f16(
                    cf($l),
                    [af($l, 0, 0), af($l, 1, 0), af($l, 2, 0), af($l, 3, 0)],
                    [bf($l, 0, 0), bf($l, 1, 0), bf($l, 2, 0), bf($l, 3, 0)],
                    meta_pair($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_bf16_k16 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k16_f32_bf16(
                    cf($l),
                    [af($l, 0, 1), af($l, 1, 1)],
                    [bf($l, 0, 1), bf($l, 1, 1)],
                    meta_one($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_bf16_k32 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k32_f32_bf16(
                    cf($l),
                    [af($l, 0, 1), af($l, 1, 1), af($l, 2, 1), af($l, 3, 1)],
                    [bf($l, 0, 1), bf($l, 1, 1), bf($l, 2, 1), bf($l, 3, 1)],
                    meta_pair($l, $sel, 0),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_tf32_k8 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k8_f32_tf32(
                    cf($l),
                    [at($l, 0), at($l, 1)],
                    [bt($l, 0), bt($l, 1)],
                    meta_one($l, $sel, 1),
                    $sel,
                )
            });
        };
    }
    macro_rules! call_tf32_k16 {
        ($fo:ident, $l:ident, $slot:literal, $sel:literal) => {
            store_f32!($fo, $l, $slot, unsafe {
                wmma::mma_sp_m16n8k16_f32_tf32(
                    cf($l),
                    [at($l, 0), at($l, 1), at($l, 2), at($l, 3)],
                    [bt($l, 0), bt($l, 1), bt($l, 2), bt($l, 3)],
                    meta_pair($l, $sel, 1),
                    $sel,
                )
            });
        };
    }

    #[kernel]
    pub fn oracle(mut po: DisjointSlice<u32>, mut fo: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        // 2-reg forms, selectors 0, 1, 2, 3.
        call_f16_k16!(po, l, 0, 0);
        call_f16_k16!(po, l, 1, 1);
        call_f16_k16!(po, l, 2, 2);
        call_f16_k16!(po, l, 3, 3);
        call_f32_f16_k16!(fo, l, 0, 0);
        call_f32_f16_k16!(fo, l, 1, 1);
        call_f32_f16_k16!(fo, l, 2, 2);
        call_f32_f16_k16!(fo, l, 3, 3);
        call_bf16_k16!(fo, l, 4, 0);
        call_bf16_k16!(fo, l, 5, 1);
        call_bf16_k16!(fo, l, 6, 2);
        call_bf16_k16!(fo, l, 7, 3);
        call_tf32_k8!(fo, l, 8, 0);
        call_tf32_k8!(fo, l, 9, 1);
        call_tf32_k8!(fo, l, 10, 2);
        call_tf32_k8!(fo, l, 11, 3);
        // 4-reg forms, selectors 0 and 1.
        call_f16_k32!(po, l, 4, 0);
        call_f16_k32!(po, l, 5, 1);
        call_f32_f16_k32!(fo, l, 12, 0);
        call_f32_f16_k32!(fo, l, 13, 1);
        call_bf16_k32!(fo, l, 14, 0);
        call_bf16_k32!(fo, l, 15, 1);
        call_tf32_k16!(fo, l, 16, 0);
        call_tf32_k16!(fo, l, 17, 1);
    }
}

/// Register slot -> (row, col) for one lane's four logical accumulators.
fn rc(l: usize, r: usize) -> (usize, usize) {
    (l / 4 + 8 * (r / 2), (l % 4) * 2 + r % 2)
}

/// Host GEMM for the f16/bf16 forms: per 4-wide chunk g the standard code
/// (r,g) keeps columns 4g+i0 and 4g+i1 of logical A. 0x6 selects (2, 1).
fn expect_h(r: usize, c: usize, kdim: usize) -> u32 {
    let (r, c) = (r as u32, c as u32);
    let mut s = c_val(r, c);
    for g in 0..(kdim as u32) / 4 {
        let m = meta_code(r, g, 0, 0);
        let (i0, i1) = (m & 3, (m >> 2) & 3);
        s += a_val(r, 2 * g) * b_val(4 * g + i0, c);
        s += a_val(r, 2 * g + 1) * b_val(4 * g + i1, c);
    }
    s
}

/// Host GEMM for the tf32 forms: per 2-wide chunk g, 0x4 keeps column 2g
/// and 0xe keeps column 2g+1.
fn expect_t(r: usize, c: usize, kdim: usize) -> u32 {
    let (r, c) = (r as u32, c as u32);
    let mut s = c_val(r, c);
    for g in 0..(kdim as u32) / 2 {
        let e = if meta_code(r, g, 0, 1) == 0x4 { 0 } else { 1 };
        s += a_val(r, g) * b_val(2 * g + e, c);
    }
    s
}

fn main() {
    // 0x6 is in the 16-bit table and selects indices (2, 1). tf32 never uses it.
    assert_eq!(meta_code(0, 1, 0, 0) & 3, 2);
    assert_eq!((meta_code(0, 1, 0, 0) >> 2) & 3, 1);
    assert_eq!(meta_code(0, 0, 0, 1), 0x4);
    assert_eq!(meta_code(0, 0, 1, 1), 0xe);

    let ctx = CudaContext::new(0).expect("CUDA context");
    let (major, minor) = ctx.compute_capability().unwrap();
    if major < 8 {
        println!("skipping: standard sparse MMA requires sm_80+, found sm_{major}{minor}");
        return;
    }
    let s = ctx.default_stream();
    let m = kernels::load(&ctx).expect("module");
    let mut pd = DeviceBuffer::<u32>::zeroed(&s, 32 * PACKED_STRIDE).unwrap();
    let mut fd = DeviceBuffer::<f32>::zeroed(&s, 32 * F32_STRIDE).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { m.oracle(&s, cfg, &mut pd, &mut fd) }.unwrap();
    let p = pd.to_host_vec(&s).unwrap();
    let f = fd.to_host_vec(&s).unwrap();
    let mut bad = 0;
    // packed slots: k16 f16 selectors 0..=3, then k32 f16 selectors 0..=1.
    let packed_k = [16u32, 16, 16, 16, 32, 32];
    // f32 slots: k16 f32.f16 x4, k16 bf16 x4, k8 tf32 x4, k32 f32.f16 x2,
    // k32 bf16 x2, k16 tf32 x2.
    let f32_kind = [0u32, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1];
    let f32_k = [
        16u32, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 32, 32, 32, 32, 16, 16,
    ];
    for l in 0..32 {
        for (slot, kdim) in packed_k.iter().copied().enumerate() {
            for r in 0..4 {
                let w = p[l * PACKED_STRIDE + slot * 2 + r / 2];
                let got = if r % 2 == 0 { w & 0xffff } else { w >> 16 };
                let (x, y) = rc(l, r);
                let want = f16_bits(expect_h(x, y, kdim as usize));
                if got != want {
                    eprintln!("f16 slot {slot} lane {l} reg {r}: {got:#x} != {want:#x}");
                    bad += 1;
                }
            }
        }
        for slot in 0..F32_RESULTS {
            for r in 0..4 {
                let got = f[l * F32_STRIDE + slot * 4 + r];
                let (x, y) = rc(l, r);
                let want = if f32_kind[slot] == 0 {
                    expect_h(x, y, f32_k[slot] as usize)
                } else {
                    expect_t(x, y, f32_k[slot] as usize)
                } as f32;
                if got != want {
                    eprintln!("f32 slot {slot} lane {l} reg {r}: {got} != {want}");
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "accumulator mismatches");
    println!(
        "SUCCESS: all 8 variants; 24 selector calls; all 32 lanes and 4 logical accumulators/lane match host GEMM"
    );
}
