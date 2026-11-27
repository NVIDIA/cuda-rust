/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Every F32/F64 `cuda_device::float` scalar arithmetic form, built for sm_75
//! and checked against IEEE-754 directed rounding and PTX `.sat` bounds

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_host::cuda_module;

// Positive normal triples (a, b, c) covering both unsaturated results inside
// (0, 1) and results above 1 where `.sat` clamps to 1.0.
const INPUTS: [[f32; 3]; 5] = [
    [0.3, 0.35, 0.2],
    [0.45, 0.55, 0.15],
    [0.75, 0.6, 0.7],
    [1.1, 1.3, 0.5],
    [1.7, 1.9, 1.1],
];
const F32_FORMS: usize = 48;
const F64_FORMS: usize = 16;

#[cuda_module]
mod kernels {
    use super::{F32_FORMS, F64_FORMS};
    use cuda_device::{DisjointSlice, float, kernel, thread};

    #[kernel]
    pub fn evaluate(
        inputs: &[[f32; 3]],
        mut output_f32: DisjointSlice<f32>,
        mut output_f64: DisjointSlice<f64>,
    ) {
        let index = thread::index_1d().get();
        if index >= inputs.len() {
            return;
        }
        let [a, b, c] = inputs[index];
        let [da, db, dc] = [a as f64, b as f64, c as f64];
        let values_f32: [f32; F32_FORMS] = [
            float::mul_rn_f32(a, b),
            float::mul_rz_f32(a, b),
            float::mul_rm_f32(a, b),
            float::mul_rp_f32(a, b),
            float::mul_rn_ftz_f32(a, b),
            float::mul_rz_ftz_f32(a, b),
            float::mul_rm_ftz_f32(a, b),
            float::mul_rp_ftz_f32(a, b),
            float::div_rn_f32(a, b),
            float::div_rz_f32(a, b),
            float::div_rm_f32(a, b),
            float::div_rp_f32(a, b),
            float::div_rn_ftz_f32(a, b),
            float::div_rz_ftz_f32(a, b),
            float::div_rm_ftz_f32(a, b),
            float::div_rp_ftz_f32(a, b),
            float::fma_rn_f32(a, b, c),
            float::fma_rz_f32(a, b, c),
            float::fma_rm_f32(a, b, c),
            float::fma_rp_f32(a, b, c),
            float::fma_rn_ftz_f32(a, b, c),
            float::fma_rz_ftz_f32(a, b, c),
            float::fma_rm_ftz_f32(a, b, c),
            float::fma_rp_ftz_f32(a, b, c),
            float::fma_rn_sat_f32(a, b, c),
            float::fma_rz_sat_f32(a, b, c),
            float::fma_rm_sat_f32(a, b, c),
            float::fma_rp_sat_f32(a, b, c),
            float::fma_rn_ftz_sat_f32(a, b, c),
            float::fma_rz_ftz_sat_f32(a, b, c),
            float::fma_rm_ftz_sat_f32(a, b, c),
            float::fma_rp_ftz_sat_f32(a, b, c),
            float::add_rn_f32(a, b),
            float::add_rz_f32(a, b),
            float::add_rm_f32(a, b),
            float::add_rp_f32(a, b),
            float::add_rn_ftz_f32(a, b),
            float::add_rz_ftz_f32(a, b),
            float::add_rm_ftz_f32(a, b),
            float::add_rp_ftz_f32(a, b),
            float::add_rn_sat_f32(a, b),
            float::add_rz_sat_f32(a, b),
            float::add_rm_sat_f32(a, b),
            float::add_rp_sat_f32(a, b),
            float::add_rn_ftz_sat_f32(a, b),
            float::add_rz_ftz_sat_f32(a, b),
            float::add_rm_ftz_sat_f32(a, b),
            float::add_rp_ftz_sat_f32(a, b),
        ];
        let values_f64: [f64; F64_FORMS] = [
            float::mul_rn_f64(da, db),
            float::mul_rz_f64(da, db),
            float::mul_rm_f64(da, db),
            float::mul_rp_f64(da, db),
            float::div_rn_f64(da, db),
            float::div_rz_f64(da, db),
            float::div_rm_f64(da, db),
            float::div_rp_f64(da, db),
            float::fma_rn_f64(da, db, dc),
            float::fma_rz_f64(da, db, dc),
            float::fma_rm_f64(da, db, dc),
            float::fma_rp_f64(da, db, dc),
            float::add_rn_f64(da, db),
            float::add_rz_f64(da, db),
            float::add_rm_f64(da, db),
            float::add_rp_f64(da, db),
        ];
        for (form, value) in values_f32.into_iter().enumerate() {
            let slot = index * F32_FORMS + form;
            if slot < output_f32.len() {
                // SAFETY: each thread owns the slots of its own input row.
                unsafe { *output_f32.get_unchecked_mut(slot) = value };
            }
        }
        for (form, value) in values_f64.into_iter().enumerate() {
            let slot = index * F64_FORMS + form;
            if slot < output_f64.len() {
                // SAFETY: each thread owns the slots of its own input row.
                unsafe { *output_f64.get_unchecked_mut(slot) = value };
            }
        }
    }
}

/// Checks the rn/rz/rm/rp quartet of a positive result. `residual` returns the
/// exact sign of `value - exact`, so rm and rp must be the representable
/// neighbours that bracket the exact result.
fn check_rounded(
    failures: &mut Vec<String>,
    label: &str,
    [rn, rz, rm, rp]: [f64; 4],
    host_rn: f64,
    rm_next_up: f64,
    residual: impl Fn(f64) -> f64,
) {
    let exact = residual(rm) == 0.0;
    let ok = rn == host_rn
        && rz == rm
        && residual(rm) <= 0.0
        && residual(rp) >= 0.0
        && rp == if exact { rm } else { rm_next_up };
    if !ok {
        failures.push(format!(
            "{label}: rn={rn:e} rz={rz:e} rm={rm:e} rp={rp:e}, host rn={host_rn:e}"
        ));
    }
}

fn check_saturated(
    failures: &mut Vec<String>,
    label: &str,
    actual: [f32; 4],
    unsaturated: [f32; 4],
) {
    let expected = unsaturated.map(|v| v.clamp(0.0, 1.0));
    if actual != expected {
        failures.push(format!(
            "{label}: actual={actual:?} vs clamped={expected:?}"
        ));
    }
}

fn check_input(failures: &mut Vec<String>, [a, b, c]: [f32; 3], f32s: &[f32], f64s: &[f64]) {
    let [da, db, dc] = [a as f64, b as f64, c as f64];
    let quartet_f32 =
        |start: usize| -> [f64; 4] { std::array::from_fn(|i| f32s[start + i] as f64) };
    let raw_quartet_f32 = |start: usize| -> [f32; 4] { std::array::from_fn(|i| f32s[start + i]) };
    let next_up_f32 = |value: f64| (value as f32).next_up() as f64;

    // f32 sums, products, quotients, and FMAs of f32 inputs have exact f64
    // residuals because da * db is exact in 53-bit precision (24 + 24 <= 53).
    for (start, name) in [(0, "mul.f32"), (4, "mul.ftz.f32")] {
        let rounded = quartet_f32(start);
        check_rounded(
            failures,
            &format!("{name}({a},{b})"),
            rounded,
            (a * b) as f64,
            next_up_f32(rounded[2]),
            |v| v - da * db,
        );
    }
    for (start, name) in [(8, "div.f32"), (12, "div.ftz.f32")] {
        let rounded = quartet_f32(start);
        check_rounded(
            failures,
            &format!("{name}({a},{b})"),
            rounded,
            (a / b) as f64,
            next_up_f32(rounded[2]),
            |v| v * db - da,
        );
    }
    for (start, name) in [(16, "fma.f32"), (20, "fma.ftz.f32")] {
        let rounded = quartet_f32(start);
        check_rounded(
            failures,
            &format!("{name}({a},{b},{c})"),
            rounded,
            a.mul_add(b, c) as f64,
            next_up_f32(rounded[2]),
            |v| (v - da * db) - dc,
        );
    }
    check_saturated(
        failures,
        &format!("fma.sat.f32({a},{b},{c})"),
        raw_quartet_f32(24),
        raw_quartet_f32(16),
    );
    check_saturated(
        failures,
        &format!("fma.ftz.sat.f32({a},{b},{c})"),
        raw_quartet_f32(28),
        raw_quartet_f32(20),
    );
    for (start, name) in [(32, "add.f32"), (36, "add.ftz.f32")] {
        let rounded = quartet_f32(start);
        check_rounded(
            failures,
            &format!("{name}({a},{b})"),
            rounded,
            (a + b) as f64,
            next_up_f32(rounded[2]),
            |v| (v - da) - db,
        );
    }
    check_saturated(
        failures,
        &format!("add.sat.f32({a},{b})"),
        raw_quartet_f32(40),
        raw_quartet_f32(32),
    );
    check_saturated(
        failures,
        &format!("add.ftz.sat.f32({a},{b})"),
        raw_quartet_f32(44),
        raw_quartet_f32(36),
    );

    // Since da, db, dc come from f32 values (<= 24 significand bits), da * db
    // and da + db are exact in f64! We also perturb the f64 quartet check
    // through div_f64 where da / db is inexact in f64.
    let mul64: [f64; 4] = f64s[0..4].try_into().unwrap();
    let div64: [f64; 4] = f64s[4..8].try_into().unwrap();
    let fma64: [f64; 4] = f64s[8..12].try_into().unwrap();
    let add64: [f64; 4] = f64s[12..16].try_into().unwrap();
    check_rounded(
        failures,
        &format!("mul.f64({a},{b})"),
        mul64,
        da * db,
        mul64[2].next_up(),
        |v| v - da * db,
    );
    check_rounded(
        failures,
        &format!("div.f64({a},{b})"),
        div64,
        da / db,
        div64[2].next_up(),
        |v| v.mul_add(db, -da),
    );
    check_rounded(
        failures,
        &format!("fma.f64({a},{b},{c})"),
        fma64,
        da.mul_add(db, dc),
        fma64[2].next_up(),
        |v| (v - da * db) - dc,
    );
    check_rounded(
        failures,
        &format!("add.f64({a},{b})"),
        add64,
        da + db,
        add64[2].next_up(),
        |v| (v - da) - db,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scalar arithmetic sm_75 GPU validation ===");
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    if major * 10 + minor < 75 {
        println!("skipping: the scalar arithmetic floor is sm_75 (device is sm_{major}{minor})");
        return Ok(());
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let inputs = DeviceBuffer::from_host(&stream, &INPUTS)?;
    let mut output_f32 = DeviceBuffer::<f32>::zeroed(&stream, INPUTS.len() * F32_FORMS)?;
    let mut output_f64 = DeviceBuffer::<f64>::zeroed(&stream, INPUTS.len() * F64_FORMS)?;

    // SAFETY: the launch covers exactly the inputs, and each output holds one
    // row per input.
    unsafe {
        module.evaluate(
            &stream,
            LaunchConfig::for_num_elems(INPUTS.len() as u32),
            &inputs,
            &mut output_f32,
            &mut output_f64,
        )
    }?;

    let measured_f32 = output_f32.to_host_vec(&stream)?;
    let measured_f64 = output_f64.to_host_vec(&stream)?;
    println!("device: sm_{major}{minor}");
    let mut failures = Vec::new();
    for (row, &triple) in INPUTS.iter().enumerate() {
        check_input(
            &mut failures,
            triple,
            &measured_f32[row * F32_FORMS..][..F32_FORMS],
            &measured_f64[row * F64_FORMS..][..F64_FORMS],
        );
    }
    for failure in &failures {
        println!("FAIL {failure}");
    }
    assert!(
        failures.is_empty(),
        "{} scalar arithmetic checks failed",
        failures.len()
    );
    println!(
        "PASS: {} scalar arithmetic forms on {} input triples meet IEEE-754 directed rounding and PTX saturation bounds",
        F32_FORMS + F64_FORMS,
        INPUTS.len()
    );
    Ok(())
}
