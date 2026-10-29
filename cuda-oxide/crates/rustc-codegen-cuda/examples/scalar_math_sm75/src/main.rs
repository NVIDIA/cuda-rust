/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Every F32/F64 `cuda_device::float` scalar math form, built for sm_75 and
//! checked against the PTX ISA error bounds

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_host::cuda_module;

// positive normal inputs inside (0.5, 2), where every bound below applies and
// the ftz forms behave like the plain ones
const INPUTS: [f32; 5] = [0.75, 1.0, 1.1, 1.5, 1.9];
const F32_FORMS: usize = 30;
const F64_FORMS: usize = 10;

#[cuda_module]
mod kernels {
    use super::{F32_FORMS, F64_FORMS};
    use cuda_device::{DisjointSlice, float, kernel, thread};

    #[kernel]
    pub fn evaluate(
        inputs: &[f32],
        mut output_f32: DisjointSlice<f32>,
        mut output_f64: DisjointSlice<f64>,
    ) {
        let index = thread::index_1d().get();
        if index >= inputs.len() {
            return;
        }
        let x = inputs[index];
        let y = x as f64;
        let values_f32: [f32; F32_FORMS] = [
            float::sin_approx_f32(x),
            float::sin_approx_ftz_f32(x),
            float::cos_approx_f32(x),
            float::cos_approx_ftz_f32(x),
            float::lg2_approx_f32(x),
            float::lg2_approx_ftz_f32(x),
            float::ex2_approx_f32(x),
            float::ex2_approx_ftz_f32(x),
            float::tanh_approx_f32(x),
            float::rcp_approx_ftz_f32(x),
            float::rsqrt_approx_f32(x),
            float::rsqrt_approx_ftz_f32(x),
            float::sqrt_approx_f32(x),
            float::sqrt_approx_ftz_f32(x),
            float::rcp_rn_f32(x),
            float::rcp_rz_f32(x),
            float::rcp_rm_f32(x),
            float::rcp_rp_f32(x),
            float::rcp_rn_ftz_f32(x),
            float::rcp_rz_ftz_f32(x),
            float::rcp_rm_ftz_f32(x),
            float::rcp_rp_ftz_f32(x),
            float::sqrt_rn_f32(x),
            float::sqrt_rz_f32(x),
            float::sqrt_rm_f32(x),
            float::sqrt_rp_f32(x),
            float::sqrt_rn_ftz_f32(x),
            float::sqrt_rz_ftz_f32(x),
            float::sqrt_rm_ftz_f32(x),
            float::sqrt_rp_ftz_f32(x),
        ];
        let values_f64: [f64; F64_FORMS] = [
            float::rcp_approx_ftz_f64(y),
            float::rsqrt_approx_f64(y),
            float::rcp_rn_f64(y),
            float::rcp_rz_f64(y),
            float::rcp_rm_f64(y),
            float::rcp_rp_f64(y),
            float::sqrt_rn_f64(y),
            float::sqrt_rz_f64(y),
            float::sqrt_rm_f64(y),
            float::sqrt_rp_f64(y),
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

enum Bound {
    Absolute(f64),
    Relative(f64),
    Ulps(u32),
}

/// Compares an approximate result with the host reference, rounded to the
/// result's precision for ulp bounds
fn check_approx(
    failures: &mut Vec<String>,
    label: &str,
    actual: f64,
    reference: f64,
    bound: Bound,
) {
    let within = match bound {
        Bound::Absolute(limit) => (actual - reference).abs() <= limit,
        Bound::Relative(limit) => ((actual - reference) / reference).abs() <= limit,
        Bound::Ulps(limit) => {
            (actual as f32)
                .to_bits()
                .abs_diff((reference as f32).to_bits())
                <= limit
        }
    };
    if !within {
        failures.push(format!("{label}: {actual:e} vs reference {reference:e}"));
    }
}

/// Checks the rn/rz/rm/rp quartet of a positive result. `residual` returns the
/// exact sign of `value - exact`, so rm and rp must be the representable
/// neighbours that bracket the exact result
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

fn check_input(failures: &mut Vec<String>, x: f32, f32s: &[f32], f64s: &[f64]) {
    let y = x as f64;
    let at = |values: &[f32], form: usize| values[form] as f64;
    let quartet = |start: usize| -> [f64; 4] { std::array::from_fn(|i| at(f32s, start + i)) };

    for (form, name, reference, bound) in [
        (0, "sin", y.sin(), Bound::Absolute(2f64.powf(-20.5))),
        (1, "sin.ftz", y.sin(), Bound::Absolute(2f64.powf(-20.5))),
        (2, "cos", y.cos(), Bound::Absolute(2f64.powf(-20.4))),
        (3, "cos.ftz", y.cos(), Bound::Absolute(2f64.powf(-20.4))),
        (4, "lg2", y.log2(), Bound::Absolute(2f64.powi(-22))),
        (5, "lg2.ftz", y.log2(), Bound::Absolute(2f64.powi(-22))),
        (6, "ex2", y.exp2(), Bound::Ulps(2)),
        (7, "ex2.ftz", y.exp2(), Bound::Ulps(2)),
        (8, "tanh", y.tanh(), Bound::Relative(2f64.powi(-11))),
        (9, "rcp.approx.ftz", 1.0 / y, Bound::Ulps(1)),
        (
            10,
            "rsqrt",
            1.0 / y.sqrt(),
            Bound::Relative(2f64.powf(-22.9)),
        ),
        (
            11,
            "rsqrt.ftz",
            1.0 / y.sqrt(),
            Bound::Relative(2f64.powf(-22.9)),
        ),
        (12, "sqrt.approx", y.sqrt(), Bound::Relative(2f64.powi(-23))),
        (
            13,
            "sqrt.approx.ftz",
            y.sqrt(),
            Bound::Relative(2f64.powi(-23)),
        ),
    ] {
        check_approx(
            failures,
            &format!("{name}.f32({x})"),
            at(f32s, form),
            reference,
            bound,
        );
    }

    // f32 products and squares are exact in f64, so the residuals are exact
    let next_up_f32 = |value: f64| (value as f32).next_up() as f64;
    for (start, name) in [(14, "rcp.f32"), (18, "rcp.ftz.f32")] {
        let rounded = quartet(start);
        let label = format!("{name}({x})");
        let next = next_up_f32(rounded[2]);
        check_rounded(failures, &label, rounded, (1.0 / x) as f64, next, |v| {
            v * y - 1.0
        });
    }
    for (start, name) in [(22, "sqrt.f32"), (26, "sqrt.ftz.f32")] {
        let rounded = quartet(start);
        let label = format!("{name}({x})");
        let next = next_up_f32(rounded[2]);
        check_rounded(failures, &label, rounded, x.sqrt() as f64, next, |v| {
            v * v - y
        });
    }

    // the ISA documents no error bound for these two, so 2^-16 is only a
    // sanity bound; rcp.approx.ftz.f64 does document a zero low word
    let rcp_approx = f64s[0];
    check_approx(
        failures,
        &format!("rcp.approx.ftz.f64({x})"),
        rcp_approx,
        1.0 / y,
        Bound::Relative(2f64.powi(-16)),
    );
    if rcp_approx.to_bits() as u32 != 0 {
        failures.push(format!(
            "rcp.approx.ftz.f64({x}): low word of {rcp_approx:e} is not zero"
        ));
    }
    check_approx(
        failures,
        &format!("rsqrt.approx.f64({x})"),
        f64s[1],
        1.0 / y.sqrt(),
        Bound::Relative(2f64.powi(-16)),
    );

    // fused multiply-add rounds once, so its sign is the exact residual's sign
    let rcp: [f64; 4] = f64s[2..6].try_into().unwrap();
    let sqrt: [f64; 4] = f64s[6..10].try_into().unwrap();
    check_rounded(
        failures,
        &format!("rcp.f64({x})"),
        rcp,
        1.0 / y,
        rcp[2].next_up(),
        |v| v.mul_add(y, -1.0),
    );
    check_rounded(
        failures,
        &format!("sqrt.f64({x})"),
        sqrt,
        y.sqrt(),
        sqrt[2].next_up(),
        |v| v.mul_add(v, -y),
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scalar math sm_75 GPU validation ===");
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    if major * 10 + minor < 75 {
        println!("skipping: the scalar math floor is sm_75 (device is sm_{major}{minor})");
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
    for (row, &x) in INPUTS.iter().enumerate() {
        check_input(
            &mut failures,
            x,
            &measured_f32[row * F32_FORMS..][..F32_FORMS],
            &measured_f64[row * F64_FORMS..][..F64_FORMS],
        );
    }
    for failure in &failures {
        println!("FAIL {failure}");
    }
    assert!(
        failures.is_empty(),
        "{} scalar math checks failed",
        failures.len()
    );
    println!(
        "PASS: {} scalar math forms on {} inputs meet the PTX ISA bounds and the f64 approximation sanity checks",
        F32_FORMS + F64_FORMS,
        INPUTS.len()
    );
    Ok(())
}
