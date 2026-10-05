/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */

//! Checks all four plain SM89 sparse FP8 MMA forms against an exact host GEMM.
//!
//! Each form covers all twelve standard-metadata codes and zero/nonzero C.
//! Values vary across K bands so swapped fragment registers change the result.
//! See the PTX ISA's "Matrix Fragments for sparse mma.m16n8k64" for the layout.
//! `--probe` additionally compares candidate layouts on the current GPU.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, wmma};

const M: usize = 16;
const N: usize = 8;
const K: usize = 64;
const METADATA_CODES: usize = 12;
const VARIANTS: usize = 2 * 4 * METADATA_CODES;
/// Compressed K: two kept elements per 4-wide group.
const KC: usize = K / 2;

/// Small exact integers with distinct K bands expose fragment-order mistakes.
fn a_val(r: u32, c: u32) -> u32 {
    1 + (3 * r + 5 * c + c / 16) % 7
}
fn b_val(k: u32, n: u32) -> u32 {
    1 + (2 * k + 3 * n + 5 * (k / 4) + k / 16) % 8
}
fn c_val(r: u32, n: u32) -> u32 {
    (5 * r + 7 * n) % 32
}

/// e4m3 / e5m2 encoding of the small non-negative integers used here.
///
/// The mantissa field is `(v - 2^e)` scaled by `2^(mantissa_bits - e)`: 5 is
/// `1.25 * 4`, so its e4m3 mantissa field is `2`, not `1`.
fn f8_bits(v: u32, e4m3: bool) -> u32 {
    if v == 0 {
        return 0;
    }
    let mut e = 0u32;
    if v >= 2 {
        e = 1;
    }
    if v >= 4 {
        e = 2;
    }
    if v >= 8 {
        e = 3;
    }
    let mant = v - (1u32 << e);
    if e4m3 {
        let sh = 3_u32.saturating_sub(e);
        ((e + 7) << 3) | (mant << sh)
    } else {
        let sh = 2_u32.saturating_sub(e);
        ((e + 15) << 2) | (mant << sh)
    }
}

/// Standard metadata accepts two distinct 2-bit indices, in either order.
fn nibble_of(i: u32) -> u32 {
    probe_nibble(i)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// A fragment register `n`, byte `i`, lane `l`.
    fn a_slot(l: u32, n: u32, i: u32) -> (u32, u32) {
        (l / 4 + 8 * (n & 1), 4 * (l & 3) + 16 * (n / 2) + i)
    }

    /// B fragment register `n`, byte `i`, lane `l`.
    fn b_slot(l: u32, n: u32, i: u32) -> (u32, u32) {
        (16 * n + 4 * (l & 3) + i, l / 4)
    }

    /// This lane's four packed A registers: the compressed 16x32 matrix.
    fn a_regs(l: u32, e4m3: bool) -> [u32; 4] {
        let mut a = [0u32; 4];
        let mut n = 0;
        while n < 4 {
            let mut i = 0;
            while i < 4 {
                let (r, c) = a_slot(l, n, i);
                a[n as usize] |= f8_bits(a_val(r, c), e4m3) << (8 * i);
                i += 1;
            }
            n += 1;
        }
        a
    }

    /// This lane's four packed B registers: the full 64x8 matrix.
    fn b_regs(l: u32, e4m3: bool) -> [u32; 4] {
        let mut b = [0u32; 4];
        let mut n = 0;
        while n < 4 {
            let mut i = 0;
            while i < 4 {
                let (k, col) = b_slot(l, n, i);
                b[n as usize] |= f8_bits(b_val(k, col), e4m3) << (8 * i);
                i += 1;
            }
            n += 1;
        }
        b
    }

    /// This lane's four f32 accumulators, one per (row, column) pair.
    fn c_regs(l: u32) -> [f32; 4] {
        let r = l / 4;
        let c = (l & 3) * 2;
        [
            c_val(r, c) as f32,
            c_val(r, c + 1) as f32,
            c_val(r + 8, c) as f32,
            c_val(r + 8, c + 1) as f32,
        ]
    }

    fn store(out: &mut DisjointSlice<f32>, variant: usize, l: u32, d: [f32; 4]) {
        let p = variant * 128 + (l as usize) * 4;
        let mut j = 0usize;
        while j < 4 {
            unsafe { *out.get_unchecked_mut(p + j) = d[j] };
            j += 1;
        }
    }

    /// Variant `v` occupies `out[v*128 + lane*4 .. +4]`, with
    /// `v = nonzero_accumulator*48 + form*12 + metadata_code`.
    #[kernel]
    pub fn oracle(mut out: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        let a4 = a_regs(l, true);
        let a5 = a_regs(l, false);
        let b4 = b_regs(l, true);
        let b5 = b_regs(l, false);
        let cz = [0.0f32; 4];
        let cn = c_regs(l);

        let mut base = 0usize;
        let mut acc = 0;
        while acc < 2 {
            let c = if acc == 0 { cz } else { cn };
            let mut ni = 0;
            while ni < METADATA_CODES as u32 {
                let meta = nibble_of(ni) * 0x1111_1111;
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e4m3_f32(c, a4, b4, meta, 0) };
                store(&mut out, base + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e5m2_f32(c, a4, b5, meta, 0) };
                store(&mut out, base + METADATA_CODES + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e5m2_e4m3_f32(c, a5, b4, meta, 0) };
                store(&mut out, base + 2 * METADATA_CODES + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e5m2_e5m2_f32(c, a5, b5, meta, 0) };
                store(&mut out, base + 3 * METADATA_CODES + ni as usize, l, d);
                ni += 1;
            }
            base += 4 * METADATA_CODES;
            acc += 1;
        }
    }

    /// Layout sweep: runs the e4m3/e4m3 form once per candidate fragment
    /// layout so the host can check every candidate against an exact GEMM.
    #[kernel]
    pub fn probe(mut out: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        let g = l / 4;
        let t = l % 4;
        let zero = [0.0f32; 4];

        let mut combo = 0usize;
        let mut ai = 0u32;
        while ai < A_LAYOUTS {
            let rp = ai / (A_BAND_OPTS * A_ORD_OPTS);
            let ab = (ai / A_ORD_OPTS) % A_BAND_OPTS;
            let ao = ai % A_ORD_OPTS;
            let mut a = [0u32; 4];
            let mut n = 0u32;
            while n < 4 {
                let r = g + 8 * a_row_off(rp, n);
                let mut i = 0u32;
                while i < 4 {
                    let c = 4 * t + a_band(ab, n) + byte_ord(ao, i);
                    a[n as usize] |= f8_bits(probe_a(r, c), true) << (8 * i);
                    i += 1;
                }
                n += 1;
            }

            let mut bi = 0u32;
            while bi < B_LAYOUTS {
                let bb = bi / B_ORD_OPTS;
                let bo = bi % B_ORD_OPTS;
                let mut b = [0u32; 4];
                let mut n = 0u32;
                while n < 4 {
                    let mut i = 0u32;
                    while i < 4 {
                        let k = b_band(bb, n) + 4 * t + byte_ord(bo, i);
                        b[n as usize] |= f8_bits(probe_b(k, g), true) << (8 * i);
                        i += 1;
                    }
                    n += 1;
                }

                let mut mi = 0u32;
                while mi < META_OPTS {
                    let nib = probe_nibble(mi % 12);
                    // `sw` exchanges the two 2-bit fields of the nibble.
                    let word = if mi / 12 == 0 {
                        nib
                    } else {
                        ((nib & 3) << 2) | (nib >> 2)
                    };
                    let meta = word * 0x1111_1111;
                    let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e4m3_f32(zero, a, b, meta, 0) };
                    let p = combo * 128 + (l as usize) * 4;
                    let mut j = 0usize;
                    while j < 4 {
                        unsafe { *out.get_unchecked_mut(p + j) = d[j] };
                        j += 1;
                    }
                    combo += 1;
                    mi += 1;
                }
                bi += 1;
            }
            ai += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host reference for the oracle.
// ---------------------------------------------------------------------------

/// D = A(compressed) @ B under the 2:4 pattern the metadata code selects, plus
/// the accumulator when `acc` is set. `nibble = i0 | i1 << 2` sends compressed
/// column `2g` to dense K `4g + i0` and column `2g + 1` to `4g + i1`.
fn expect(r: usize, col: usize, nibble: u32, acc: bool) -> f32 {
    let (r, col) = (r as u32, col as u32);
    let (i0, i1) = (nibble & 3, nibble >> 2);
    let mut s = if acc { c_val(r, col) } else { 0 };
    for g in 0..(K as u32) / 4 {
        s += a_val(r, 2 * g) * b_val(4 * g + i0, col);
        s += a_val(r, 2 * g + 1) * b_val(4 * g + i1, col);
    }
    s as f32
}

fn run_oracle(ctx: &std::sync::Arc<CudaContext>) {
    let s = ctx.default_stream();
    let module = kernels::load(ctx).expect("module");
    let mut out = DeviceBuffer::<f32>::zeroed(&s, VARIANTS * 128).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.oracle(&s, cfg, &mut out) }.unwrap();
    let buf = out.to_host_vec(&s).unwrap();

    let names = ["e4m3 x e4m3", "e4m3 x e5m2", "e5m2 x e4m3", "e5m2 x e5m2"];
    let mut bad = 0usize;
    for (form, name) in names.iter().enumerate() {
        for acc in 0..2 {
            let mut form_bad = 0usize;
            for ni in 0..METADATA_CODES as u32 {
                let variant = (acc * 4 + form) * METADATA_CODES + ni as usize;
                let nibble = nibble_of(ni);
                for l in 0..32usize {
                    for j in 0..4usize {
                        let r = l / 4 + 8 * (j / 2);
                        let col = (l % 4) * 2 + j % 2;
                        let want = expect(r, col, nibble, acc == 1);
                        let got = buf[variant * 128 + l * 4 + j];
                        if got.to_bits() != want.to_bits() {
                            eprintln!(
                                "{name} C{} nibble {nibble:#x} lane {l} reg {j} \
                                 (row {r}, col {col}): {got} != {want}",
                                if acc == 1 { "!=0" } else { "=0" }
                            );
                            form_bad += 1;
                        }
                    }
                }
            }
            println!(
                "  {name:12} {:>7} all 12 standard metadata codes: {} mismatches",
                if acc == 1 { "C!=0," } else { "C=0," },
                form_bad
            );
            bad += form_bad;
        }
    }
    assert_eq!(bad, 0, "sparse FP8 MMA accumulator mismatches");
    println!(
        "SUCCESS: all 4 sparse FP8 m16n8k64 forms x 12 metadata codes x C=0/C!=0; \
         all 32 lanes and 4 logical accumulators/lane match host GEMM exactly"
    );
}

// ---------------------------------------------------------------------------
// Layout sweep (`--probe`).
//
// The candidate space varies the axes independently:
//
//   A row   `a_row_off(rp, n)`  -- which A registers hold row group+8
//   A band  `a_band(ab, n)`     -- which compressed columns a register holds
//   A order `byte_ord(ao, i)`   -- byte position i -> column offset
//   B band  `b_band(bb, n)`     -- which dense K rows a register holds
//   B order `byte_ord(bo, i)`   -- byte position i -> K offset
//   code    the nibble and which of its two 2-bit fields names the even column
//
// Every candidate is filled into the registers and executed once; the host
// compares the result with an exact GEMM whose answer does not depend on the
// candidate at all. A wrong axis changes the sum, so only the true wiring
// reproduces it.
// ---------------------------------------------------------------------------

const A_ROW_OPTS: u32 = 4;
const A_BAND_OPTS: u32 = 6;
const A_ORD_OPTS: u32 = 2;
const B_BAND_OPTS: u32 = 6;
const B_ORD_OPTS: u32 = 2;
const META_SWAP_OPTS: u32 = 2;

const A_LAYOUTS: u32 = A_ROW_OPTS * A_BAND_OPTS * A_ORD_OPTS;
const B_LAYOUTS: u32 = B_BAND_OPTS * B_ORD_OPTS;
const META_OPTS: u32 = 12 * META_SWAP_OPTS;
const COMBOS: u32 = A_LAYOUTS * B_LAYOUTS * META_OPTS;

/// Separate K-band patterns for the optional layout sweep.
fn probe_a(r: u32, c: u32) -> u32 {
    1 + (3 * r + 5 * c) % 7 + c / 16
}
fn probe_b(k: u32, n: u32) -> u32 {
    1 + (2 * k + 3 * n + 4 * (k / 16)) % 8
}

fn a_row_off(rp: u32, n: u32) -> u32 {
    if rp == 0 {
        n & 1
    } else if rp == 1 {
        n / 2
    } else if rp == 2 {
        (n & 1) ^ 1
    } else {
        (n / 2) ^ 1
    }
}

fn a_band(ab: u32, n: u32) -> u32 {
    if ab == 0 {
        16 * (n / 2)
    } else if ab == 1 {
        4 * (n / 2)
    } else if ab == 2 {
        8 * n
    } else if ab == 3 {
        4 * n
    } else if ab == 4 {
        8 * (n / 2)
    } else {
        16 * n
    }
}

fn b_band(bb: u32, n: u32) -> u32 {
    if bb == 0 {
        16 * n
    } else if bb == 1 {
        8 * n
    } else if bb == 2 {
        4 * n
    } else if bb == 3 {
        16 * (2 * (n & 1) + n / 2)
    } else if bb == 4 {
        16 * (n / 2)
    } else {
        16 * (n & 1)
    }
}

fn byte_ord(o: u32, i: u32) -> u32 {
    if o == 0 { i } else { 3 - i }
}

/// The twelve legal nibbles: two 2-bit fields that differ.
fn probe_nibble(idx: u32) -> u32 {
    if idx == 0 {
        0x1
    } else if idx == 1 {
        0x2
    } else if idx == 2 {
        0x3
    } else if idx == 3 {
        0x4
    } else if idx == 4 {
        0x6
    } else if idx == 5 {
        0x7
    } else if idx == 6 {
        0x8
    } else if idx == 7 {
        0x9
    } else if idx == 8 {
        0xb
    } else if idx == 9 {
        0xc
    } else if idx == 10 {
        0xd
    } else {
        0xe
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    rp: u32,
    ab: u32,
    ao: u32,
    bb: u32,
    bo: u32,
    nib: u32,
    sw: u32,
}

impl Candidate {
    fn a_slot(&self, l: u32, n: u32, i: u32) -> (i64, i64) {
        (
            (l / 4 + 8 * a_row_off(self.rp, n)) as i64,
            (4 * (l % 4) + a_band(self.ab, n) + byte_ord(self.ao, i)) as i64,
        )
    }
    fn b_slot(&self, l: u32, n: u32, i: u32) -> (i64, i64) {
        (
            (b_band(self.bb, n) + 4 * (l % 4) + byte_ord(self.bo, i)) as i64,
            (l / 4) as i64,
        )
    }

    /// (A entries placed, B entries placed) out of 512 each.
    fn coverage(&self) -> (u32, u32) {
        let mut ap = 0;
        let mut bp = 0;
        for l in 0..32u32 {
            for n in 0..4u32 {
                for i in 0..4u32 {
                    let (r, c) = self.a_slot(l, n, i);
                    if (0..M as i64).contains(&r) && (0..KC as i64).contains(&c) {
                        ap += 1;
                    }
                    let (k, col) = self.b_slot(l, n, i);
                    if (0..K as i64).contains(&k) && (0..N as i64).contains(&col) {
                        bp += 1;
                    }
                }
            }
        }
        (ap, bp)
    }

    /// Dense K position that compressed column `c` feeds under this code.
    fn kmap(&self, c: u32) -> u32 {
        let low = self.nib & 3;
        let high = self.nib >> 2;
        let field = if c.is_multiple_of(2) == (self.sw == 0) {
            low
        } else {
            high
        };
        4 * (c / 2) + field
    }

    fn predict(&self) -> Vec<f32> {
        let mut a_seen = vec![vec![0.0f32; KC]; M];
        let mut b_seen = vec![vec![0.0f32; N]; K];
        for l in 0..32u32 {
            for n in 0..4u32 {
                for i in 0..4u32 {
                    let (r, c) = self.a_slot(l, n, i);
                    if (0..M as i64).contains(&r) && (0..KC as i64).contains(&c) {
                        a_seen[r as usize][c as usize] = probe_a(r as u32, c as u32) as f32;
                    }
                    let (k, col) = self.b_slot(l, n, i);
                    if (0..K as i64).contains(&k) && (0..N as i64).contains(&col) {
                        b_seen[k as usize][col as usize] = probe_b(k as u32, col as u32) as f32;
                    }
                }
            }
        }
        let mut d = vec![0.0f32; M * N];
        for r in 0..M {
            for n in 0..N {
                let mut s = 0.0f32;
                for c in 0..KC {
                    s += a_seen[r][c] * b_seen[self.kmap(c as u32) as usize][n];
                }
                d[r * N + n] = s;
            }
        }
        d
    }
}

/// Decode one lane's four accumulators into D[row][col].
fn observed(combo: usize, buf: &[f32]) -> Vec<f32> {
    let mut d = vec![0.0f32; M * N];
    for l in 0..32usize {
        for j in 0..4usize {
            let r = l / 4 + 8 * (j / 2);
            let c = (l % 4) * 2 + j % 2;
            d[r * N + c] = buf[combo * 128 + l * 4 + j];
        }
    }
    d
}

fn run_probe(ctx: &std::sync::Arc<CudaContext>) {
    let s = ctx.default_stream();
    let module = kernels::load(ctx).expect("module");
    let mut out = DeviceBuffer::<f32>::zeroed(&s, COMBOS as usize * 128).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.probe(&s, cfg, &mut out) }.unwrap();
    let buf = out.to_host_vec(&s).unwrap();

    let mut exact = Vec::new();
    let mut total = 0usize;
    for ai in 0..A_LAYOUTS {
        for bi in 0..B_LAYOUTS {
            for mi in 0..META_OPTS {
                let cand = Candidate {
                    rp: ai / (A_BAND_OPTS * A_ORD_OPTS),
                    ab: (ai / A_ORD_OPTS) % A_BAND_OPTS,
                    ao: ai % A_ORD_OPTS,
                    bb: bi / B_ORD_OPTS,
                    bo: bi % B_ORD_OPTS,
                    nib: probe_nibble(mi % 12),
                    sw: mi / 12,
                };
                let idx = total;
                total += 1;
                let want = cand.predict();
                let got = observed(idx, &buf);
                if want
                    .iter()
                    .zip(got.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    exact.push(cand);
                }
            }
        }
    }
    println!("probe: {total} candidates, {} exact", exact.len());
    for c in &exact {
        let (ap, bp) = c.coverage();
        println!(
            "  exact rp={} ab={} ao={} bb={} bo={} nib={:#x} sw={} (A {ap}/512, B {bp}/512 placed)",
            c.rp, c.ab, c.ao, c.bb, c.bo, c.nib, c.sw
        );
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let (major, minor) = ctx.compute_capability().unwrap();
    if major < 8 || (major == 8 && minor < 9) {
        println!("skipping: sparse FP8 MMA requires sm_89+, found sm_{major}{minor}");
        return;
    }
    if std::env::args().any(|a| a == "--probe") {
        run_probe(&ctx);
    } else {
        run_oracle(&ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operands_round_trip_through_both_fp8_formats() {
        for e4m3 in [false, true] {
            let (mantissa_bits, bias) = if e4m3 { (3, 7) } else { (2, 15) };
            for value in 1..=8 {
                let bits = f8_bits(value, e4m3);
                let exponent = (bits >> mantissa_bits) as i32 - bias;
                let mantissa = bits & ((1 << mantissa_bits) - 1);
                let decoded =
                    (1.0 + mantissa as f32 / (1 << mantissa_bits) as f32) * 2.0f32.powi(exponent);
                assert_eq!(decoded, value as f32);
            }
        }
    }

    #[test]
    fn metadata_covers_every_distinct_pair() {
        let mut seen = [false; 16];
        for index in 0..METADATA_CODES as u32 {
            let code = nibble_of(index) as usize;
            assert_ne!(code & 3, code >> 2);
            assert!(!seen[code]);
            seen[code] = true;
        }
        assert_eq!(seen.iter().filter(|&&present| present).count(), 12);
    }

    #[test]
    fn oracle_detects_every_b_register_swap() {
        for first in 0..4 {
            for second in first + 1..4 {
                let mut detected = false;
                for index in 0..METADATA_CODES as u32 {
                    let code = nibble_of(index);
                    for row in 0..M as u32 {
                        for col in 0..N as u32 {
                            let mut swapped = 0;
                            for compressed_k in 0..KC as u32 {
                                let field = if compressed_k.is_multiple_of(2) {
                                    code & 3
                                } else {
                                    code >> 2
                                };
                                let k = 4 * (compressed_k / 2) + field;
                                let band = k / 16;
                                let other = if band == first {
                                    second
                                } else if band == second {
                                    first
                                } else {
                                    band
                                };
                                swapped +=
                                    a_val(row, compressed_k) * b_val(16 * other + k % 16, col);
                            }
                            detected |=
                                swapped as f32 != expect(row as usize, col as usize, code, false);
                        }
                    }
                }
                assert!(
                    detected,
                    "B registers {first} and {second} are indistinguishable"
                );
            }
        }
    }
}
