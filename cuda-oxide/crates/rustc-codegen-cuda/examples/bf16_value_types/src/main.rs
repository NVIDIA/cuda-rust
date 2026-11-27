/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed bfloat16 values as kernel element types.
//!
//! Kernels take `&[Bf16]`, `DisjointSlice<Bf16>`, `&[Bf16x2]`, and
//! `DisjointSlice<Bf16x2>`. The host allocates plain `u16` / `u32` buffers and
//! retypes them with `DeviceBuffer::wrap_elem`, then reads results back
//! through `peel_elem`.
//!
//! Checks:
//!
//! 1. `scale`: widen, multiply, and round back to bf16, against bits the host
//!    computes.
//! 2. Host-capable methods give the same bits on the device as on the host.
//! 3. Each device-only method returns the same bits as the raw intrinsic it
//!    wraps, called in the same kernel on the same input bits. Inputs include
//!    NaN, both zeros, infinities, subnormals, and mixed signs.
//! 4. For ordinary values the results equal plain `f32` arithmetic.
//!
//! Data flow of one check:
//!
//! ```text
//!  host                              device                      host
//!  ┌──────────────────┐  wrap_elem  ┌──────────────────────┐  peel_elem  ┌───────────────┐
//!  │ Vec<u16>/<u32>   │ ──────────► │ &[Bf16] / &[Bf16x2]  │ ──────────► │ Vec<u16>/<u32>│
//!  │ raw input bits   │             │ kernel, one output   │             │ verify_* adds │
//!  │                  │             │ slot per thread      │             │ to `Report`   │
//!  └──────────────────┘             └──────────────────────┘             └───────────────┘
//! ```
//!
//! The five kernels run one after another on the default stream:
//!
//! ```text
//! [scale] ──► [host_capable] ──► [scalar_min_max] ──► [packed_arithmetic] ──► [packed_from_f32]
//! ```
//!
//! The program prints `PASS` when no check found a mismatch. Otherwise it
//! prints `FAIL` and exits with status 1.
//!
//! Run: `cargo oxide run bf16_value_types`

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer, Stream};
use cuda_device::bf16::Bf16;
use cuda_device::bf16x2::Bf16x2;
use cuda_device::convert::{
    bf16_to_f32, cvt_f32x2_bf16x2, f32_pair_to_packed_bf16, f32_to_bf16, f32_to_bf16_rne,
    pack_bf16_pair,
};
use cuda_device::cuda_module;

/// Scalar `Bf16` operations, in kernel slot order.
///
/// Entry `op` names the method that the `scalar_min_max` kernel runs for
/// operation index `op`. The host uses the names in mismatch messages and to
/// pick the matching `f32` arithmetic.
const SCALAR_OPS: [&str; 8] = [
    "min",
    "max",
    "min_nan",
    "max_nan",
    "min_xorsign_abs",
    "max_xorsign_abs",
    "min_nan_xorsign_abs",
    "max_nan_xorsign_abs",
];

/// Packed `Bf16x2` operations, in kernel slot order.
///
/// Entry `op` names the method that the `packed_arithmetic` kernel runs for
/// operation index `op`. Each operation acts on the low and the high half of
/// its operands independently.
const PACKED_OPS: [&str; 15] = [
    "abs",
    "neg",
    "add",
    "sub",
    "mul",
    "fma",
    "fma_relu",
    "min",
    "max",
    "min_nan",
    "max_nan",
    "min_xorsign_abs",
    "max_xorsign_abs",
    "min_nan_xorsign_abs",
    "max_nan_xorsign_abs",
];

/// Packing conversions from two `f32`, in kernel slot order.
///
/// Entry `op` names the constructor that the `packed_from_f32` kernel runs
/// for operation index `op`: round to nearest even, then round toward zero.
const CONVERT_OPS: [&str; 2] = ["from_f32_pair_rne", "from_f32_pair_rz"];

/// Results per input of the `host_capable` kernel.
///
/// The kernel writes this many `Bf16x2` values for each input element, one
/// per expression. `verify_host_capable` holds the same expressions in the
/// same order.
const HOST_CAPABLE_OPS: usize = 6;

/// Number of scalar operations, as a plain count for the kernels.
///
/// Kernel code must not reference the name table [`SCALAR_OPS`], so the
/// `scalar_min_max` kernel uses this count to split a slot into case,
/// operation, and route.
const SCALAR_OP_COUNT: usize = SCALAR_OPS.len();

/// Number of packed operations, as a plain count for the kernels.
///
/// Kernel code must not reference the name table [`PACKED_OPS`], so the
/// `packed_arithmetic` kernel uses this count to split a slot into case,
/// operation, and route.
const PACKED_OP_COUNT: usize = PACKED_OPS.len();

/// Number of packing conversions, as a plain count for the kernels.
///
/// Kernel code must not reference the name table [`CONVERT_OPS`], so the
/// `packed_from_f32` kernel uses this count to split a slot into case,
/// operation, and route.
const CONVERT_OP_COUNT: usize = CONVERT_OPS.len();

/// Device kernels that take and return typed bf16 values.
///
/// Every kernel writes one output element per thread and reads its inputs by
/// index, so no two threads write the same slot. The host sizes each launch
/// from the length of the output buffer.
#[cuda_module]
mod kernels {
    use super::{
        Bf16, Bf16x2, CONVERT_OP_COUNT, HOST_CAPABLE_OPS, PACKED_OP_COUNT, SCALAR_OP_COUNT,
    };
    use cuda_device::convert::{cvt_bf16x2_f32, cvt_rz_bf16x2_f32};
    use cuda_device::{DisjointSlice, bf16, bf16x2, kernel, thread};

    /// Scale every bf16 element by an `f32` factor, rounding to nearest even.
    ///
    /// Thread `i` widens `x[i]` to `f32`, multiplies by `s`, and writes the
    /// rounded bf16 result to `out[i]`. `out` and `x` have the same length.
    /// Threads past the end of `out` write nothing.
    #[kernel]
    pub fn scale(mut out: DisjointSlice<Bf16>, x: &[Bf16], s: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = Bf16::from_f32_rne(x[i].to_f32() * s);
        }
    }

    /// Methods that also run on the host, evaluated on the device.
    ///
    /// Output slot `HOST_CAPABLE_OPS * case + op`.
    ///
    /// `case` selects the input element `x[case]` and `op` selects one of
    /// the expressions built from constructors and accessors. `out` holds
    /// `HOST_CAPABLE_OPS` elements per input. `fill` is a bf16 value that
    /// arrives as a typed scalar kernel argument.
    #[kernel]
    pub fn host_capable(mut out: DisjointSlice<Bf16x2>, x: &[Bf16x2], fill: Bf16) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[slot / HOST_CAPABLE_OPS];
            *o = match slot % HOST_CAPABLE_OPS {
                0 => Bf16x2::new(v.hi(), v.lo()),
                1 => Bf16x2::from_bits(v.to_bits()),
                2 => {
                    let (lo, hi) = v.to_f32_pair();
                    Bf16x2::from_f32_pair(lo, hi)
                }
                3 => Bf16x2::new(Bf16::from_f32(v.lo().to_f32()), fill),
                4 => Bf16x2::new(
                    Bf16::from_f32_rne(v.hi().to_f32()),
                    Bf16::from_bits(v.lo().to_bits()),
                ),
                _ => Bf16x2::new(Bf16::default(), v.hi()),
            };
        }
    }

    /// Every scalar min/max form, through the method and through the raw
    /// intrinsic.
    ///
    /// Output slot `2 * (SCALAR_OP_COUNT * case + op) + route`; route 0 is
    /// the method and route 1 is the raw intrinsic.
    ///
    /// `case` selects the operand pair `(a[case], b[case])`; `a` and `b`
    /// have the same length. `out` holds `2 * SCALAR_OP_COUNT` elements per
    /// case, so both routes of one operation are neighbours.
    #[kernel]
    pub fn scalar_min_max(mut out: DisjointSlice<Bf16>, a: &[Bf16], b: &[Bf16]) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let case = slot / (2 * SCALAR_OP_COUNT);
            let op = (slot / 2) % SCALAR_OP_COUNT;
            let (x, y) = (a[case], b[case]);
            *o = if slot % 2 == 0 {
                match op {
                    0 => x.min(y),
                    1 => x.max(y),
                    2 => x.min_nan(y),
                    3 => x.max_nan(y),
                    4 => x.min_xorsign_abs(y),
                    5 => x.max_xorsign_abs(y),
                    6 => x.min_nan_xorsign_abs(y),
                    _ => x.max_nan_xorsign_abs(y),
                }
            } else {
                let (x, y) = (x.to_bits(), y.to_bits());
                Bf16::from_bits(match op {
                    0 => bf16::min_bf16(x, y),
                    1 => bf16::max_bf16(x, y),
                    2 => bf16::min_nan_bf16(x, y),
                    3 => bf16::max_nan_bf16(x, y),
                    4 => bf16::min_xorsign_abs_bf16(x, y),
                    5 => bf16::max_xorsign_abs_bf16(x, y),
                    6 => bf16::min_nan_xorsign_abs_bf16(x, y),
                    _ => bf16::max_nan_xorsign_abs_bf16(x, y),
                })
            };
        }
    }

    /// Every packed arithmetic and min/max form, through the method and
    /// through the raw intrinsic.
    ///
    /// Output slot `2 * (PACKED_OP_COUNT * case + op) + route`; route 0 is
    /// the method and route 1 is the raw intrinsic.
    ///
    /// `case` selects the operand triple `(a[case], b[case], c[case])`; the
    /// three inputs have the same length. Operations with fewer than three
    /// operands ignore the rest. `out` holds `2 * PACKED_OP_COUNT` elements
    /// per case.
    #[kernel]
    pub fn packed_arithmetic(
        mut out: DisjointSlice<Bf16x2>,
        a: &[Bf16x2],
        b: &[Bf16x2],
        c: &[Bf16x2],
    ) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let case = slot / (2 * PACKED_OP_COUNT);
            let op = (slot / 2) % PACKED_OP_COUNT;
            let (x, y, z) = (a[case], b[case], c[case]);
            *o = if slot % 2 == 0 {
                match op {
                    0 => x.abs(),
                    1 => x.neg(),
                    2 => x.add(y),
                    3 => x.sub(y),
                    4 => x.mul(y),
                    5 => x.fma(y, z),
                    6 => x.fma_relu(y, z),
                    7 => x.min(y),
                    8 => x.max(y),
                    9 => x.min_nan(y),
                    10 => x.max_nan(y),
                    11 => x.min_xorsign_abs(y),
                    12 => x.max_xorsign_abs(y),
                    13 => x.min_nan_xorsign_abs(y),
                    _ => x.max_nan_xorsign_abs(y),
                }
            } else {
                let (x, y, z) = (x.to_bits(), y.to_bits(), z.to_bits());
                Bf16x2::from_bits(match op {
                    0 => bf16x2::abs_bf16x2(x),
                    1 => bf16x2::neg_bf16x2(x),
                    2 => bf16x2::add_bf16x2(x, y),
                    3 => bf16x2::sub_bf16x2(x, y),
                    4 => bf16x2::mul_bf16x2(x, y),
                    5 => bf16x2::fma_bf16x2(x, y, z),
                    6 => bf16x2::fma_relu_bf16x2(x, y, z),
                    7 => bf16x2::min_bf16x2(x, y),
                    8 => bf16x2::max_bf16x2(x, y),
                    9 => bf16x2::min_nan_bf16x2(x, y),
                    10 => bf16x2::max_nan_bf16x2(x, y),
                    11 => bf16x2::min_xorsign_abs_bf16x2(x, y),
                    12 => bf16x2::max_xorsign_abs_bf16x2(x, y),
                    13 => bf16x2::min_nan_xorsign_abs_bf16x2(x, y),
                    _ => bf16x2::max_nan_xorsign_abs_bf16x2(x, y),
                })
            };
        }
    }

    /// Both rounding conversions from two `f32`, through the method and
    /// through the raw intrinsic.
    ///
    /// Output slot `2 * (CONVERT_OP_COUNT * case + op) + route`; route 0 is
    /// the method and route 1 is the raw intrinsic.
    ///
    /// `case` selects the pair `(lo[case], hi[case])`; `lo` becomes the low
    /// half of the packed result and `hi` the high half. `out` holds
    /// `2 * CONVERT_OP_COUNT` elements per case.
    #[kernel]
    pub fn packed_from_f32(mut out: DisjointSlice<Bf16x2>, lo: &[f32], hi: &[f32]) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let case = slot / (2 * CONVERT_OP_COUNT);
            let op = (slot / 2) % CONVERT_OP_COUNT;
            let (l, h) = (lo[case], hi[case]);
            *o = if slot % 2 == 0 {
                match op {
                    0 => Bf16x2::from_f32_pair_rne(l, h),
                    _ => Bf16x2::from_f32_pair_rz(l, h),
                }
            } else {
                Bf16x2::from_bits(match op {
                    0 => cvt_bf16x2_f32(l, h),
                    _ => cvt_rz_bf16x2_f32(l, h),
                })
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// bf16 values whose sums, differences, products, and fused products are
/// exact in bf16: 1, -1, 2, -2, 0.5, 3, -1.5.
///
/// Each entry is a raw bf16 bit pattern. Results on these values are
/// compared with plain `f32` arithmetic, because no rounding is involved.
const ORDINARY: [u16; 7] = [0x3F80, 0xBF80, 0x4000, 0xC000, 0x3F00, 0x4040, 0xBFC0];

/// Remaining bf16 inputs: both zeros, an inexact value, the largest finite
/// value, the smallest normal, subnormals, infinities, and NaNs with
/// different signs and payloads.
///
/// Each entry is a raw bf16 bit pattern. On these values the checks only
/// compare each method with the raw intrinsic it wraps.
const SPECIAL: [u16; 12] = [
    0x0000, 0x8000, 0x4049, 0x7F7F, 0x0080, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81,
];

/// `f32` bit patterns for the packing conversions: exact values, rounding
/// ties with an even and an odd retained bit, both neighbours of a tie, the
/// overflow boundary, zeros, subnormals, infinities, and NaNs.
///
/// `main` feeds every ordered pair of these patterns to the
/// `packed_from_f32` kernel as the low and the high input.
const F32_PATTERNS: [u32; 20] = [
    0x3F80_0000,
    0xC020_0000,
    0x3F80_8000,
    0x3F81_8000,
    0x3F80_7FFF,
    0x3F80_8001,
    0xBF81_8000,
    0x4049_0FDB,
    0x0000_0000,
    0x8000_0000,
    0x7F7F_FFFF,
    0x7F7F_7FFF,
    0x7F80_0000,
    0xFF80_0000,
    0x0080_0000,
    0x0000_8001,
    0x8001_8000,
    0x7FC0_0000,
    0x7F80_0001,
    0xFFC1_2345,
];

/// All bf16 input bit patterns: [`ORDINARY`] followed by [`SPECIAL`].
///
/// `main` builds the scalar and packed operand tables from every ordered
/// pair of these values.
fn bf16_inputs() -> Vec<u16> {
    ORDINARY.iter().chain(SPECIAL.iter()).copied().collect()
}

/// Whether a bf16 bit pattern is one of the [`ORDINARY`] values.
///
/// The verifiers run the `f32` arithmetic comparison only when every operand
/// that an operation reads passes this test.
fn is_ordinary(bits: u16) -> bool {
    ORDINARY.contains(&bits)
}

/// Finite bf16 inputs for `scale` whose scaled values stay normal and finite,
/// followed by zeros, infinities, and the largest finite value.
///
/// The first 4096 entries come from a fixed-seed linear congruential
/// generator, so every run uses the same bit patterns. The last entries are
/// fixed special patterns.
fn scale_inputs() -> Vec<u16> {
    let mut state = 0x2545_F491_u32;
    let mut inputs: Vec<u16> = (0..4096)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sign = (state >> 31) << 15;
            let exponent = (0x60 + ((state >> 20) & 0x3F)) << 7;
            let mantissa = (state >> 8) & 0x7F;
            (sign | exponent | mantissa) as u16
        })
        .collect();
    inputs.extend([
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F, 0x3F80, 0x3F81,
    ]);
    inputs
}

// ---------------------------------------------------------------------------
// Typed buffers
// ---------------------------------------------------------------------------

/// Copies raw bf16 bits to the device and retypes the buffer as `Bf16`.
///
/// The buffer has one element per entry of `bits`. Panics when the upload
/// fails.
fn upload_bf16(stream: &Stream, bits: &[u16]) -> DeviceBuffer<Bf16> {
    DeviceBuffer::from_host(stream, bits)
        .expect("upload bf16 bits")
        .wrap_elem()
}

/// Copies raw packed bits to the device and retypes the buffer as `Bf16x2`.
///
/// Each `u32` holds two bf16 values, the low half in bits 0..16. Panics when
/// the upload fails.
fn upload_bf16x2(stream: &Stream, bits: &[u32]) -> DeviceBuffer<Bf16x2> {
    DeviceBuffer::from_host(stream, bits)
        .expect("upload bf16x2 bits")
        .wrap_elem()
}

/// Allocates a zero-filled device buffer of `len` `Bf16` elements.
///
/// Kernels write their results into this buffer. Panics when the allocation
/// fails.
fn zeroed_bf16(stream: &Stream, len: usize) -> DeviceBuffer<Bf16> {
    DeviceBuffer::<u16>::zeroed(stream, len)
        .expect("allocate bf16 output")
        .wrap_elem()
}

/// Allocates a zero-filled device buffer of `len` `Bf16x2` elements.
///
/// Kernels write their results into this buffer. Panics when the allocation
/// fails.
fn zeroed_bf16x2(stream: &Stream, len: usize) -> DeviceBuffer<Bf16x2> {
    DeviceBuffer::<u32>::zeroed(stream, len)
        .expect("allocate bf16x2 output")
        .wrap_elem()
}

/// Retypes a `Bf16` device buffer as `u16` and copies it to the host.
///
/// The result has one raw bit pattern per buffer element, in buffer order.
/// Panics when the read back fails.
fn download_bf16(stream: &Stream, buffer: DeviceBuffer<Bf16>) -> Vec<u16> {
    buffer
        .peel_elem::<u16>()
        .to_host_vec(stream)
        .expect("read back bf16 bits")
}

/// Retypes a `Bf16x2` device buffer as `u32` and copies it to the host.
///
/// The result has one raw packed bit pattern per buffer element, in buffer
/// order. Panics when the read back fails.
fn download_bf16x2(stream: &Stream, buffer: DeviceBuffer<Bf16x2>) -> Vec<u32> {
    buffer
        .peel_elem::<u32>()
        .to_host_vec(stream)
        .expect("read back bf16x2 bits")
}

/// Launch shape for a kernel that writes `threads` output elements.
///
/// `threads` is the length of the output buffer of the kernel. Panics when
/// the count does not fit in `u32`.
fn launch_config(threads: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(u32::try_from(threads).expect("thread count fits in u32"))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Counts mismatches and prints the first few of each check.
///
/// Every `verify_*` function adds to the same report. `main` reads the total
/// at the end to choose between `PASS` and `FAIL`.
struct Report {
    /// Mismatches over the whole run.
    ///
    /// This count never resets. A value other than zero makes the program
    /// exit with status 1.
    failures: usize,
    /// Mismatches since the last call to [`Report::begin_check`].
    ///
    /// This count decides whether a mismatch is printed or only counted, so
    /// one noisy check cannot hide the output of a later check.
    failures_in_check: usize,
}

/// Counting and printing of mismatches.
///
/// A verifier calls `begin_check` once, then `fail` once per mismatch.
impl Report {
    /// Largest number of mismatch messages printed for one check.
    ///
    /// The mismatch after this many prints one notice that the rest are
    /// counted but not printed. Later mismatches of the check print nothing.
    const PRINT_LIMIT: usize = 8;

    /// Starts a new check, which gets its own print allowance.
    ///
    /// Resets the per-check count to zero. The total count stays as it is.
    fn begin_check(&mut self) {
        self.failures_in_check = 0;
    }

    /// Records one mismatch described by `message`.
    ///
    /// Prints the message while the check is under [`Report::PRINT_LIMIT`],
    /// then increments both the per-check count and the total count.
    fn fail(&mut self, message: String) {
        if self.failures_in_check < Self::PRINT_LIMIT {
            println!("  FAIL: {message}");
        } else if self.failures_in_check == Self::PRINT_LIMIT {
            println!("  FAIL: further mismatches of this check are counted but not printed");
        }
        self.failures_in_check += 1;
        self.failures += 1;
    }
}

/// Sign product applied to a magnitude: the `xorsign.abs` result for
/// ordinary values.
///
/// Returns `-magnitude` when exactly one of `x` and `y` has its sign bit
/// set, and `magnitude` otherwise.
fn xorsign(magnitude: f32, x: f32, y: f32) -> f32 {
    if x.is_sign_negative() != y.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// `f32` arithmetic for one scalar `Bf16` operation on ordinary values.
///
/// `op` is an index into [`SCALAR_OPS`]. The NaN-propagating forms share the
/// arithmetic of the plain forms, because ordinary values are never NaN.
/// Panics when a name in the table has no arithmetic here.
fn scalar_expected(op: usize, x: f32, y: f32) -> f32 {
    match SCALAR_OPS[op] {
        "min" | "min_nan" => x.min(y),
        "max" | "max_nan" => x.max(y),
        "min_xorsign_abs" | "min_nan_xorsign_abs" => xorsign(x.abs().min(y.abs()), x, y),
        "max_xorsign_abs" | "max_nan_xorsign_abs" => xorsign(x.abs().max(y.abs()), x, y),
        other => panic!("no host arithmetic for scalar operation {other}"),
    }
}

/// Number of operands a packed operation reads.
///
/// `op` is an index into [`PACKED_OPS`]. The result is 1, 2, or 3.
/// `verify_packed` requires only that many operands to be ordinary before it
/// runs the arithmetic comparison.
fn packed_arity(op: usize) -> usize {
    match PACKED_OPS[op] {
        "abs" | "neg" => 1,
        "fma" | "fma_relu" => 3,
        _ => 2,
    }
}

/// `f32` arithmetic for one half of a packed `Bf16x2` operation on ordinary
/// values.
///
/// `op` is an index into [`PACKED_OPS`]; `x`, `y`, and `z` are the same half
/// of the three operands. Operations with fewer operands ignore the rest.
/// Panics when a name in the table has no arithmetic here.
fn packed_expected(op: usize, x: f32, y: f32, z: f32) -> f32 {
    match PACKED_OPS[op] {
        "abs" => x.abs(),
        "neg" => -x,
        "add" => x + y,
        "sub" => x - y,
        "mul" => x * y,
        "fma" => x * y + z,
        "fma_relu" => (x * y + z).max(0.0),
        "min" | "min_nan" => x.min(y),
        "max" | "max_nan" => x.max(y),
        "min_xorsign_abs" | "min_nan_xorsign_abs" => xorsign(x.abs().min(y.abs()), x, y),
        "max_xorsign_abs" | "max_nan_xorsign_abs" => xorsign(x.abs().max(y.abs()), x, y),
        other => panic!("no host arithmetic for packed operation {other}"),
    }
}

/// Requires every operation to have had at least one arithmetic check.
///
/// `names` and `checked` are parallel: `checked[op]` is the number of
/// arithmetic comparisons that ran for `names[op]`. A zero count is recorded
/// as a mismatch, so a filter that skips every input cannot pass silently.
fn require_coverage(report: &mut Report, group: &str, names: &[&str], checked: &[usize]) {
    for (name, count) in names.iter().zip(checked) {
        if *count == 0 {
            report.fail(format!("{group} {name}: no arithmetic check ran"));
        }
    }
}

/// Compares the output of the `scale` kernel with bits the host computes.
///
/// `results[i]` must equal `inputs[i]` widened to `f32`, multiplied by
/// `factor`, and rounded to nearest even. Panics when the two slices differ
/// in length.
fn verify_scale(report: &mut Report, inputs: &[u16], factor: f32, results: &[u16]) {
    report.begin_check();
    assert_eq!(results.len(), inputs.len(), "scale result count");
    for (&input, &got) in inputs.iter().zip(results) {
        let want = f32_to_bf16_rne(bf16_to_f32(input) * factor);
        if got != want {
            report.fail(format!(
                "scale x={input:#06x} s={factor}: got {got:#06x}, want {want:#06x}"
            ));
        }
    }
}

/// Compares the output of the `host_capable` kernel with host bits.
///
/// `results` holds [`HOST_CAPABLE_OPS`] packed values per entry of `inputs`,
/// in the expression order of the kernel. The host builds each expected
/// value from the raw-bit conversion functions. `fill` is the raw bf16 value
/// that the kernel received as its scalar argument.
fn verify_host_capable(report: &mut Report, inputs: &[u32], fill: u16, results: &[u32]) {
    report.begin_check();
    assert_eq!(
        results.len(),
        inputs.len() * HOST_CAPABLE_OPS,
        "host-capable result count"
    );
    for (case, &input) in inputs.iter().enumerate() {
        let lo = input as u16;
        let hi = (input >> 16) as u16;
        let (lo_f32, hi_f32) = cvt_f32x2_bf16x2(input);
        let want = [
            ("new(hi, lo)", pack_bf16_pair(hi, lo)),
            ("from_bits(to_bits)", input),
            (
                "from_f32_pair(to_f32_pair)",
                f32_pair_to_packed_bf16(lo_f32, hi_f32),
            ),
            (
                "new(from_f32(lo.to_f32), fill)",
                pack_bf16_pair(f32_to_bf16(bf16_to_f32(lo)), fill),
            ),
            (
                "new(from_f32_rne(hi.to_f32), lo)",
                pack_bf16_pair(f32_to_bf16_rne(bf16_to_f32(hi)), lo),
            ),
            ("new(default, hi)", pack_bf16_pair(0, hi)),
        ];
        for (op, (name, want)) in want.into_iter().enumerate() {
            let got = results[case * HOST_CAPABLE_OPS + op];
            if got != want {
                report.fail(format!(
                    "host-capable {name} x={input:#010x}: got {got:#010x}, want {want:#010x}"
                ));
            }
        }
    }
}

/// Checks the output of the `scalar_min_max` kernel.
///
/// For every case and operation, the method result must equal the raw
/// intrinsic result bit for bit. When both operands are ordinary, the method
/// result must also equal the `f32` arithmetic of [`scalar_expected`].
///
/// Returns the number of arithmetic checks on ordinary values.
fn verify_scalar(report: &mut Report, a: &[u16], b: &[u16], results: &[u16]) -> usize {
    report.begin_check();
    let ops = SCALAR_OPS.len();
    assert_eq!(results.len(), a.len() * ops * 2, "scalar result count");
    let mut checked = vec![0_usize; ops];
    for (case, (&x, &y)) in a.iter().zip(b).enumerate() {
        for (op, name) in SCALAR_OPS.iter().enumerate() {
            let method = results[2 * (case * ops + op)];
            let raw = results[2 * (case * ops + op) + 1];
            if method != raw {
                report.fail(format!(
                    "Bf16::{name} a={x:#06x} b={y:#06x}: method {method:#06x}, raw intrinsic {raw:#06x}"
                ));
            }
            if is_ordinary(x) && is_ordinary(y) {
                let want = scalar_expected(op, bf16_to_f32(x), bf16_to_f32(y));
                let got = bf16_to_f32(method);
                if got != want {
                    report.fail(format!(
                        "Bf16::{name} a={x:#06x} b={y:#06x}: got {got} ({method:#06x}), want {want}"
                    ));
                }
                checked[op] += 1;
            }
        }
    }
    require_coverage(report, "Bf16", &SCALAR_OPS, &checked);
    checked.iter().sum()
}

/// Checks the output of the `packed_arithmetic` kernel.
///
/// For every case and operation, the method result must equal the raw
/// intrinsic result bit for bit. Each half is then compared separately with
/// the `f32` arithmetic of [`packed_expected`], when every operand the
/// operation reads is ordinary in that half.
///
/// Returns the number of arithmetic checks on ordinary values.
fn verify_packed(report: &mut Report, a: &[u32], b: &[u32], c: &[u32], results: &[u32]) -> usize {
    report.begin_check();
    let ops = PACKED_OPS.len();
    assert_eq!(results.len(), a.len() * ops * 2, "packed result count");
    let mut checked = vec![0_usize; ops];
    for (case, ((&x, &y), &z)) in a.iter().zip(b).zip(c).enumerate() {
        for (op, name) in PACKED_OPS.iter().enumerate() {
            let method = results[2 * (case * ops + op)];
            let raw = results[2 * (case * ops + op) + 1];
            if method != raw {
                report.fail(format!(
                    "Bf16x2::{name} a={x:#010x} b={y:#010x} c={z:#010x}: method {method:#010x}, raw intrinsic {raw:#010x}"
                ));
            }
            for (half, shift) in [("low", 0), ("high", 16)] {
                let operands = [x, y, z].map(|packed| (packed >> shift) as u16);
                if !operands[..packed_arity(op)]
                    .iter()
                    .all(|&bits| is_ordinary(bits))
                {
                    continue;
                }
                let [xf, yf, zf] = operands.map(bf16_to_f32);
                let want = packed_expected(op, xf, yf, zf);
                let got_bits = (method >> shift) as u16;
                let got = bf16_to_f32(got_bits);
                if got != want {
                    report.fail(format!(
                        "Bf16x2::{name} {half} half a={x:#010x} b={y:#010x} c={z:#010x}: got {got} ({got_bits:#06x}), want {want}"
                    ));
                }
                checked[op] += 1;
            }
        }
    }
    require_coverage(report, "Bf16x2", &PACKED_OPS, &checked);
    checked.iter().sum()
}

/// Checks the output of the `packed_from_f32` kernel.
///
/// For every case and conversion, the method result must equal the raw
/// intrinsic result bit for bit. Each half that came from a value that is
/// not NaN must also equal the host conversion with the same rounding.
///
/// Returns the number of arithmetic checks on values that are not NaN.
fn verify_convert(report: &mut Report, lo: &[f32], hi: &[f32], results: &[u32]) -> usize {
    report.begin_check();
    let ops = CONVERT_OPS.len();
    assert_eq!(results.len(), lo.len() * ops * 2, "conversion result count");
    let mut checked = vec![0_usize; ops];
    for (case, (&l, &h)) in lo.iter().zip(hi).enumerate() {
        let (l_bits, h_bits) = (l.to_bits(), h.to_bits());
        for (op, name) in CONVERT_OPS.iter().enumerate() {
            let method = results[2 * (case * ops + op)];
            let raw = results[2 * (case * ops + op) + 1];
            if method != raw {
                report.fail(format!(
                    "Bf16x2::{name} lo={l_bits:#010x} hi={h_bits:#010x}: method {method:#010x}, raw intrinsic {raw:#010x}"
                ));
            }
            for (half, shift, value) in [("low", 0, l), ("high", 16, h)] {
                if value.is_nan() {
                    continue;
                }
                // Toward-zero rounding of a value that is not NaN keeps the
                // high 16 bits, which is what the truncating helper does.
                let want = match *name {
                    "from_f32_pair_rne" => f32_to_bf16_rne(value),
                    _ => f32_to_bf16(value),
                };
                let got = (method >> shift) as u16;
                if got != want {
                    report.fail(format!(
                        "Bf16x2::{name} {half} half of f32 bits {:#010x}: got {got:#06x}, want {want:#06x}",
                        value.to_bits()
                    ));
                }
                checked[op] += 1;
            }
        }
    }
    require_coverage(report, "Bf16x2", &CONVERT_OPS, &checked);
    checked.iter().sum()
}

/// Runs every kernel once per input set and verifies the results on the host.
///
/// Steps: create the context on device 0, skip when the compute capability
/// major version is below 9, load the embedded PTX, then for each kernel
/// upload inputs, launch, read back, and verify. Prints one summary line per
/// check and exits with status 1 when any mismatch was counted.
fn main() {
    println!("=== bf16_value_types ===");

    let ctx = CudaContext::new(0).expect("CUDA init");

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!(
            "skipping: packed bf16x2 add/sub/mul require sm_90+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut report = Report {
        failures: 0,
        failures_in_check: 0,
    };

    // --- scale -------------------------------------------------------------
    // 1.5 turns every odd bf16 mantissa into an exact rounding tie; -1.1
    // gives inexact products and flips the sign.
    let scale_bits = scale_inputs();
    let scale_in = upload_bf16(&stream, &scale_bits);
    for factor in [1.5_f32, -1.1] {
        let mut scale_out = zeroed_bf16(&stream, scale_bits.len());
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale(
                &stream,
                launch_config(scale_bits.len()),
                &mut scale_out,
                &scale_in,
                factor,
            )
        }
        .expect("launch scale");
        let results = download_bf16(&stream, scale_out);
        verify_scale(&mut report, &scale_bits, factor, &results);
    }
    println!(
        "scale: {} elements, 2 factors compared with host bits",
        scale_bits.len()
    );

    // --- shared operand tables ---------------------------------------------
    // Both halves of `a` and `b` together visit every ordered pair of inputs.
    let values = bf16_inputs();
    let count = values.len();
    let mut scalar_a = Vec::new();
    let mut scalar_b = Vec::new();
    let mut packed_a = Vec::new();
    let mut packed_b = Vec::new();
    let mut packed_c = Vec::new();
    for i in 0..count {
        for j in 0..count {
            scalar_a.push(values[i]);
            scalar_b.push(values[j]);
            packed_a.push(pack_bf16_pair(values[i], values[j]));
            packed_b.push(pack_bf16_pair(values[j], values[(i + 5) % count]));
            packed_c.push(pack_bf16_pair(
                values[(i + j) % count],
                values[(i + 2 * j + 1) % count],
            ));
        }
    }

    // --- host-capable methods on the device --------------------------------
    let fill = 0x4049_u16;
    let host_capable_in = upload_bf16x2(&stream, &packed_a);
    let host_capable_len = packed_a.len() * HOST_CAPABLE_OPS;
    let mut host_capable_out = zeroed_bf16x2(&stream, host_capable_len);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.host_capable(
            &stream,
            launch_config(host_capable_len),
            &mut host_capable_out,
            &host_capable_in,
            Bf16::from_bits(fill),
        )
    }
    .expect("launch host_capable");
    let results = download_bf16x2(&stream, host_capable_out);
    verify_host_capable(&mut report, &packed_a, fill, &results);
    println!(
        "host-capable methods: {} inputs, {HOST_CAPABLE_OPS} expressions compared with host bits",
        packed_a.len()
    );

    // --- scalar min/max ----------------------------------------------------
    let a = upload_bf16(&stream, &scalar_a);
    let b = upload_bf16(&stream, &scalar_b);
    let scalar_len = scalar_a.len() * SCALAR_OPS.len() * 2;
    let mut scalar_out = zeroed_bf16(&stream, scalar_len);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.scalar_min_max(&stream, launch_config(scalar_len), &mut scalar_out, &a, &b) }
        .expect("launch scalar_min_max");
    let results = download_bf16(&stream, scalar_out);
    let scalar_checked = verify_scalar(&mut report, &scalar_a, &scalar_b, &results);
    println!(
        "Bf16: {} operations on {} input pairs match the raw intrinsics, {scalar_checked} arithmetic checks",
        SCALAR_OPS.len(),
        scalar_a.len()
    );

    // --- packed arithmetic and min/max -------------------------------------
    let a = upload_bf16x2(&stream, &packed_a);
    let b = upload_bf16x2(&stream, &packed_b);
    let c = upload_bf16x2(&stream, &packed_c);
    let packed_len = packed_a.len() * PACKED_OPS.len() * 2;
    let mut packed_out = zeroed_bf16x2(&stream, packed_len);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.packed_arithmetic(
            &stream,
            launch_config(packed_len),
            &mut packed_out,
            &a,
            &b,
            &c,
        )
    }
    .expect("launch packed_arithmetic");
    let results = download_bf16x2(&stream, packed_out);
    let packed_checked = verify_packed(&mut report, &packed_a, &packed_b, &packed_c, &results);
    println!(
        "Bf16x2: {} operations on {} input triples match the raw intrinsics, {packed_checked} arithmetic checks",
        PACKED_OPS.len(),
        packed_a.len()
    );

    // --- packing conversions from f32 --------------------------------------
    let mut convert_lo = Vec::new();
    let mut convert_hi = Vec::new();
    for lo in F32_PATTERNS {
        for hi in F32_PATTERNS {
            convert_lo.push(f32::from_bits(lo));
            convert_hi.push(f32::from_bits(hi));
        }
    }
    let lo = DeviceBuffer::from_host(&stream, &convert_lo).expect("upload low f32 values");
    let hi = DeviceBuffer::from_host(&stream, &convert_hi).expect("upload high f32 values");
    let convert_len = convert_lo.len() * CONVERT_OPS.len() * 2;
    let mut convert_out = zeroed_bf16x2(&stream, convert_len);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.packed_from_f32(
            &stream,
            launch_config(convert_len),
            &mut convert_out,
            &lo,
            &hi,
        )
    }
    .expect("launch packed_from_f32");
    let results = download_bf16x2(&stream, convert_out);
    let convert_checked = verify_convert(&mut report, &convert_lo, &convert_hi, &results);
    println!(
        "Bf16x2: {} conversions on {} f32 pairs match the raw intrinsics, {convert_checked} arithmetic checks",
        CONVERT_OPS.len(),
        convert_lo.len()
    );

    if report.failures != 0 {
        println!(
            "FAIL: bf16_value_types, {} mismatches on sm_{major}{minor}",
            report.failures
        );
        std::process::exit(1);
    }
    println!(
        "PASS: bf16_value_types, typed Bf16 and Bf16x2 kernels match host bits, raw intrinsics, and f32 arithmetic on sm_{major}{minor}"
    );
}
