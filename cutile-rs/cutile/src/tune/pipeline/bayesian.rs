/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Finite-space Bayesian optimization with a neutral GP prior by default.

use super::*;
use crate::tune::ParamValue;
use std::collections::BTreeMap;
mod gp;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum ParameterScale {
    /// Normalize integer values to [0, 1]. Strings are always categorical.
    #[default]
    Linear,
    /// Positive integer knobs such as powers-of-two tile sizes.
    Log2,
    /// No ordering assumption, including for integer-valued categories.
    Categorical,
}

/// Fixed, explicit GP settings. No hyperparameter optimizer or offline training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianOptions {
    pub seed: u64,
    /// Correlation length in normalized parameter coordinates.
    pub length_scale: f64,
    /// Standard deviation in natural-log latency units.
    pub signal_std: f64,
    /// Observation noise standard deviation in natural-log latency units.
    pub noise_std: f64,
    pub scales: BTreeMap<String, ParameterScale>,
    /// Optional relative log-latency preferences by configuration ID. Negative
    /// favors a config. Missing entries are neutral. Magnitudes are limited to
    /// ln(2); these hints never add observations or reduce initial uncertainty.
    pub prior_log_offsets: BTreeMap<String, f64>,
}

impl Default for BayesianOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            length_scale: 0.5,
            signal_std: 1.0,
            noise_std: 0.05,
            scales: BTreeMap::new(),
            prior_log_offsets: BTreeMap::new(),
        }
    }
}

/// A small-data GP over log latency, scored by expected improvement. Every
/// remaining candidate is scored; no acquisition optimizer is required.
///
/// Pending points are temporarily conditioned at their predicted means to
/// reduce redundant proposals. They never become measured observations or
/// improve the incumbent. Invalid and timed-out points use the same uncertainty
/// penalty so failed regions do not repeatedly attract exploration. Their
/// latencies never train the GP or improve the incumbent. This is a search
/// heuristic, not a learned probability of validity.
/// A numerical failure falls back to seeded random selection and increments
/// `numerical_fallbacks`. Instantiate a fresh selector for each run.
pub struct BayesianSearch {
    configs: Vec<Config>,
    features: Vec<Vec<f64>>,
    prior: Vec<f64>,
    options: BayesianOptions,
    random: RandomOrder,
    initial: std::collections::VecDeque<usize>,
    numerical_fallbacks: usize,
}

impl BayesianSearch {
    pub fn new(configs: &[Config], options: BayesianOptions) -> Result<Self, Failure> {
        for (name, v) in [
            ("length_scale", options.length_scale),
            ("signal_std", options.signal_std),
            ("noise_std", options.noise_std),
        ] {
            if !v.is_finite() || !(1e-6..=100.0).contains(&v) {
                return Err(Failure::infrastructure(format!(
                    "{name} must be finite and between 1e-6 and 100"
                )));
            }
        }
        if configs.is_empty() {
            return Err(Failure::infrastructure(
                "Bayesian search needs a nonempty space",
            ));
        }
        let ids: HashSet<_> = configs.iter().map(|c| &c.id).collect();
        if ids.len() != configs.len()
            || configs
                .iter()
                .any(|c| Config::new(c.params.clone()).id != c.id)
        {
            return Err(Failure::infrastructure(
                "duplicate or inconsistent configuration IDs",
            ));
        }
        if options
            .prior_log_offsets
            .iter()
            .any(|(id, v)| !ids.contains(id) || !v.is_finite() || v.abs() > std::f64::consts::LN_2)
        {
            return Err(Failure::infrastructure(
                "prior offsets need known IDs and finite magnitudes <= ln(2)",
            ));
        }
        let features = encode(configs, &options.scales)?;
        let prior = configs
            .iter()
            .map(|c| options.prior_log_offsets.get(&c.id).copied().unwrap_or(0.0))
            .collect();
        Ok(Self {
            configs: configs.to_vec(),
            features,
            prior,
            random: RandomOrder::new(options.seed),
            options,
            initial: Default::default(),
            numerical_fallbacks: 0,
        })
    }

    /// Explicit starting candidates, e.g. the caller's incumbent. Empty by
    /// default: equally scored neutral-prior candidates use seeded tie-breaking.
    pub fn with_initial(
        mut self,
        indices: impl IntoIterator<Item = usize>,
    ) -> Result<Self, Failure> {
        for i in indices {
            if i >= self.configs.len() {
                return Err(Failure::infrastructure("initial candidate out of range"));
            }
            self.initial.push_back(i);
        }
        Ok(self)
    }

    pub fn numerical_fallbacks(&self) -> usize {
        self.numerical_fallbacks
    }

    fn scores(
        &self,
        view: &SearchView<'_>,
        available: &[usize],
        deadline: Option<Instant>,
    ) -> Option<Vec<f64>> {
        let valid: Vec<_> = view
            .completed
            .iter()
            .filter_map(|o| {
                o.score_ms()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| (o.trial_id, f64::from(v).ln()))
            })
            .collect();
        // A common offset changes no preference. Estimate it from this run;
        // historical/architecture hints remain relative, never old timings.
        let center = if valid.is_empty() {
            0.0
        } else {
            valid.iter().map(|&(i, y)| y - self.prior[i]).sum::<f64>() / valid.len() as f64
        };
        let variance = self.options.signal_std.powi(2);
        let noise = self.options.noise_std.powi(2);
        let length = self.options.length_scale;
        let mut rows: Vec<_> = valid
            .iter()
            .map(|&(i, _)| self.features[i].as_slice())
            .collect();
        let mut residuals: Vec<_> = valid
            .iter()
            .map(|&(i, y)| y - center - self.prior[i])
            .collect();
        let observed =
            gp::Posterior::fit(rows.clone(), &residuals, length, variance, noise, deadline)?;
        // Plug-in incumbent from posterior means, not the luckiest noisy sample.
        // This is not a full noisy-EI Monte Carlo integral.
        let best = valid
            .iter()
            .map(|&(i, _)| center + self.prior[i] + observed.predict(&self.features[i]).0)
            .min_by(f64::total_cmp)
            .unwrap_or(center);
        let failed = view.completed.iter().filter_map(|o| {
            o.failure
                .as_ref()
                .filter(|f| matches!(f.kind, FailureKind::Invalid | FailureKind::TimedOut))
                .map(|_| o.trial_id)
        });
        for i in view.pending.iter().copied().chain(failed) {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            rows.push(self.features[i].as_slice());
            residuals.push(observed.predict(&self.features[i]).0);
        }
        let pending = gp::Posterior::fit(rows, &residuals, length, variance, noise, deadline)?;
        available
            .iter()
            .map(|&i| {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    return None;
                }
                let (mean, var) = pending.predict(&self.features[i]);
                let score = gp::expected_improvement(best, center + self.prior[i] + mean, var);
                score.is_finite().then_some(score)
            })
            .collect()
    }
}

impl Strategy for BayesianSearch {
    fn validate(&self, configs: &[Config]) -> Result<(), Failure> {
        if configs != self.configs {
            return Err(Failure::infrastructure(
                "Bayesian selector constructed for a different space/order",
            ));
        }
        Ok(())
    }

    fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize> {
        // Pipeline validates once before dispatch. Also fail closed for direct
        // callers with a mismatched space or malformed observation indices.
        if self.validate(view.configs).is_err()
            || view.completed.iter().any(|o| {
                o.trial_id >= self.configs.len() || o.config_id != self.configs[o.trial_id].id
            })
            || view.pending.iter().any(|&i| i >= self.configs.len())
        {
            return None;
        }
        let deadline = view.remaining.and_then(|t| Instant::now().checked_add(t));
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let available: Vec<_> = (0..self.configs.len())
            .filter(|i| {
                !view.pending.contains(i) && !view.completed.iter().any(|o| o.trial_id == *i)
            })
            .collect();
        if available.is_empty() {
            return None;
        }
        while let Some(i) = self.initial.pop_front() {
            if available.contains(&i) {
                return Some(i);
            }
        }
        let Some(scores) = self.scores(view, &available, deadline) else {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            self.numerical_fallbacks += 1;
            return self.random.suggest(view);
        };
        let best = scores.iter().copied().max_by(f64::total_cmp)?;
        // Resolve exact/negligible ties without a bias toward declaration order.
        let tied: Vec<_> = available
            .iter()
            .zip(&scores)
            .filter(|(_, s)| best - **s <= 1e-12 * best.abs().max(1.0))
            .map(|(&i, _)| i)
            .collect();
        let n = tied.len() as u64;
        let threshold = n.wrapping_neg() % n;
        let value = loop {
            let v = self.random.random();
            if v >= threshold {
                break v;
            }
        };
        Some(tied[(value % n) as usize])
    }
}

fn encode(
    configs: &[Config],
    scales: &BTreeMap<String, ParameterScale>,
) -> Result<Vec<Vec<f64>>, Failure> {
    let first = &configs[0].params;
    if configs.iter().any(|c| !c.params.keys().eq(first.keys()))
        || scales.keys().any(|k| !first.contains_key(k))
    {
        return Err(Failure::infrastructure(
            "configs must have the same parameter keys; scale keys must exist",
        ));
    }
    let mut rows = vec![Vec::new(); configs.len()];
    for (key, value) in first {
        let scale = scales.get(key).copied().unwrap_or_default();
        let values: Vec<_> = configs.iter().map(|c| &c.params[key]).collect();
        if values
            .iter()
            .any(|v| std::mem::discriminant(*v) != std::mem::discriminant(value))
        {
            return Err(Failure::infrastructure(
                "a parameter must have the same type in every config",
            ));
        }
        if matches!(scale, ParameterScale::Categorical) || matches!(value, ParamValue::Str(_)) {
            if matches!(scale, ParameterScale::Log2) {
                return Err(Failure::infrastructure("Log2 requires positive integers"));
            }
            let mut categories = Vec::new();
            for v in &values {
                if !categories.contains(v) {
                    categories.push(*v);
                }
            }
            // One-hot / sqrt(2): any distinct pair of categories is distance 1.
            if categories.len() > 1 {
                for (row, v) in rows.iter_mut().zip(values) {
                    row.extend(categories.iter().map(|c| {
                        if *c == v {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            0.0
                        }
                    }));
                }
            }
        } else {
            let numbers: Vec<_> = values
                .iter()
                .map(|v| match v {
                    ParamValue::Int(v) => *v,
                    _ => unreachable!(),
                })
                .collect();
            if matches!(scale, ParameterScale::Log2) && numbers.iter().any(|&n| n <= 0) {
                return Err(Failure::infrastructure("Log2 requires positive integers"));
            }
            let low = *numbers.iter().min().unwrap();
            let high = *numbers.iter().max().unwrap();
            if low == high {
                continue;
            }
            // i128 differences preserve adjacent large i64 values before conversion.
            for (row, n) in rows.iter_mut().zip(numbers) {
                let value = if matches!(scale, ParameterScale::Log2) {
                    (((i128::from(n) - i128::from(low)) as f64 / low as f64).ln_1p())
                        / (((i128::from(high) - i128::from(low)) as f64 / low as f64).ln_1p())
                } else {
                    (i128::from(n) - i128::from(low)) as f64
                        / (i128::from(high) - i128::from(low)) as f64
                };
                if !value.is_finite() {
                    return Err(Failure::infrastructure(
                        "parameter scale loses numeric resolution",
                    ));
                }
                row.push(value);
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests;
