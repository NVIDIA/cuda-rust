/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small dense GP: Matérn 5/2 covariance and Cholesky posterior solves.
//! Standard GP regression equations (Rasmussen & Williams, GPML, chapter 2),
//! independently implemented and checked against scikit-learn fixtures.
//! No inverse, optimizer, external math runtime or copied library source.

use std::time::Instant;

pub(super) fn covariance(a: &[f64], b: &[f64], length: f64, variance: f64) -> f64 {
    let r = a
        .iter()
        .zip(b)
        .map(|(a, b)| ((a - b) / length).powi(2))
        .sum::<f64>()
        .sqrt();
    let s = 5.0_f64.sqrt() * r;
    variance * (1.0 + s + s * s / 3.0) * (-s).exp()
}

pub(super) struct Posterior<'a> {
    x: Vec<&'a [f64]>,
    lower: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    length: f64,
    variance: f64,
}

impl<'a> Posterior<'a> {
    pub(super) fn fit(
        x: Vec<&'a [f64]>,
        y: &[f64],
        length: f64,
        variance: f64,
        noise: f64,
        deadline: Option<Instant>,
    ) -> Option<Self> {
        if x.len() != y.len() {
            return None;
        }
        let n = x.len();
        let mut lower = vec![vec![0.0; n]; n];
        for i in 0..n {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            for j in 0..=i {
                let value = covariance(x[i], x[j], length, variance)
                    + if i == j {
                        noise + variance * 1e-10
                    } else {
                        0.0
                    }
                    - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return None;
                    }
                    value.sqrt()
                } else {
                    value / lower[j][j]
                };
            }
        }
        let mut alpha = forward(&lower, y);
        for i in (0..n).rev() {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            alpha[i] = (alpha[i] - ((i + 1)..n).map(|j| lower[j][i] * alpha[j]).sum::<f64>())
                / lower[i][i];
        }
        if alpha.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Self {
            x,
            lower,
            alpha,
            length,
            variance,
        })
    }

    /// Residual posterior mean and latent variance (excludes measurement noise).
    pub(super) fn predict(&self, x: &[f64]) -> (f64, f64) {
        let k: Vec<_> = self
            .x
            .iter()
            .map(|row| covariance(row, x, self.length, self.variance))
            .collect();
        let mean = k.iter().zip(&self.alpha).map(|(a, b)| a * b).sum();
        let v = forward(&self.lower, &k);
        (
            mean,
            (self.variance - v.iter().map(|v| v * v).sum::<f64>()).max(0.0),
        )
    }
}

fn forward(lower: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rhs.len()];
    for i in 0..out.len() {
        out[i] = (rhs[i] - (0..i).map(|j| lower[i][j] * out[j]).sum::<f64>()) / lower[i][i];
    }
    out
}

/// Expected improvement for minimization. A&S 26.2.17 normal CDF approximation
/// (absolute CDF error < 7.5e-8); tested independently against scipy.stats.norm.
pub(super) fn expected_improvement(best: f64, mean: f64, variance: f64) -> f64 {
    let delta = best - mean;
    // A fixed positive cutoff breaks EI's scale equivariance: a small but
    // representable standard deviation still carries expected improvement.
    if variance <= 0.0 {
        return delta.max(0.0);
    }
    let sigma = variance.sqrt();
    let z = delta / sigma;
    if z >= 8.0 {
        return delta.max(0.0);
    }
    if z <= -8.0 {
        return 0.0;
    }
    let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let tail = density
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let cdf = if z >= 0.0 { 1.0 - tail } else { tail };
    (delta * cdf + sigma * density).max(0.0)
}
