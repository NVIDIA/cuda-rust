/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independently written property/regression tests selected from the survey in
//! ../TESTING.md. No upstream test implementation or fixture is copied.

use super::*;

#[test]
fn posterior_interpolates_and_returns_to_prior_far_from_observations() {
    // scikit-learn: interpolation and prior tests. These properties complement
    // golden numbers: catch wrong variance/noise conventions or solve direction.
    let rows = [[0.0], [0.4], [0.9]];
    let values = [-0.7, 0.2, 0.8];
    let model = gp::Posterior::fit(
        rows.iter().map(|x| x.as_slice()).collect(),
        &values,
        0.2,
        1.7,
        1e-12,
        None,
    )
    .unwrap();
    for (x, expected) in rows.iter().zip(values) {
        let (mean, variance) = model.predict(x);
        close(mean, expected, 1e-8);
        assert!((0.0..1e-8).contains(&variance));
    }
    let (mean, variance) = model.predict(&[100.0]);
    close(mean, 0.0, 1e-12);
    close(variance, 1.7, 1e-12);
}

#[test]
fn near_duplicate_inputs_and_reordered_observations_give_same_posterior() {
    // scikit-learn: equal vs near-equal inputs with conflicting measurements.
    let rows = [[0.2], [0.2], [0.9]];
    let near = [[0.9], [0.2 + 1e-13], [0.2]];
    let model = gp::Posterior::fit(
        rows.iter().map(|x| x.as_slice()).collect(),
        &[0.2, 0.4, -0.5],
        0.4,
        1.0,
        0.02,
        None,
    )
    .unwrap();
    let reordered = gp::Posterior::fit(
        near.iter().map(|x| x.as_slice()).collect(),
        &[-0.5, 0.4, 0.2],
        0.4,
        1.0,
        0.02,
        None,
    )
    .unwrap();
    for query in [0.0, 0.2, 0.6, 1.0] {
        let (a, av) = model.predict(&[query]);
        let (b, bv) = reordered.predict(&[query]);
        close(a, b, 1e-10);
        close(av, bv, 1e-10);
    }
}

#[test]
fn clustered_observations_remain_finite_after_convergence() {
    // Optuna: many repeated / nearly optimal points can destabilize inversion.
    // Use constant residuals so the mean is known without another GP solver.
    let rows: Vec<_> = (0..32)
        .map(|i| [if i < 16 { 0.0 } else { (i - 16) as f64 * 1e-8 }])
        .collect();
    let model = gp::Posterior::fit(
        rows.iter().map(|x| x.as_slice()).collect(),
        &[0.0; 32],
        0.5,
        1.0,
        1e-12,
        None,
    )
    .unwrap();
    for query in [0.0, 1e-8, 1e-5, 0.1, 1.0, 100.0] {
        let (mean, variance) = model.predict(&[query]);
        close(mean, 0.0, 1e-12);
        assert!(variance.is_finite() && (0.0..=1.0).contains(&variance));
    }
    assert!(model.predict(&[0.0]).1 < 1e-8);
    close(model.predict(&[100.0]).1, 1.0, 1e-12);
}

#[test]
fn acquisition_minimizes_and_preserves_scale_and_uncertainty_ordering() {
    // BoTorch analytic acquisition tests: direction, scaling, numeric ranges.
    // Our implementation has no autograd or log-EI interface to test.
    for variance in [0.0, 1e-6, 0.01, 1.0, 16.0] {
        let mut previous = f64::INFINITY;
        for i in -100..=100 {
            let mean = f64::from(i) / 10.0;
            let score = gp::expected_improvement(0.0, mean, variance);
            assert!(score.is_finite() && score >= 0.0);
            assert!(score <= previous + 1e-12, "minimization reversed at {mean}");
            previous = score;
        }
    }
    for mean in [-2.0, 0.0, 2.0] {
        let mut previous = 0.0;
        for variance in [0.0, 1e-6, 0.01, 1.0, 16.0] {
            let score = gp::expected_improvement(0.0, mean, variance);
            assert!(score + 1e-12 >= previous, "uncertainty must not reduce EI");
            previous = score;
        }
    }
    let reference = gp::expected_improvement(0.0, 0.25, 1.0);
    for scale in [1e-12, 1e-6, 1.0, 1e6] {
        close(
            gp::expected_improvement(0.0, scale * 0.25, scale * scale) / scale,
            reference,
            1e-12,
        );
    }
}

#[test]
fn latency_units_and_observation_order_do_not_change_acquisition() {
    // scikit-learn normalization tests, adapted to our log-latency centering:
    // changing ms to another positive unit must not change recommendations.
    let configs = space(&[0, 1, 2, 3, 4, 5, 6]);
    let search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let completed = [
        observation(&configs, 0, 16.0),
        observation(&configs, 3, 4.0),
        observation(&configs, 6, 8.0),
    ];
    let reference = search
        .scores(&view(&configs, &completed, &[1]), &[2, 4, 5], None)
        .unwrap();
    for scale in [2.0_f32.powi(-20), 1.0, 2.0_f32.powi(20)] {
        let mut scaled: Vec<_> = completed
            .iter()
            .map(|o| observation(&configs, o.trial_id, o.score_ms().unwrap() * scale))
            .collect();
        scaled.reverse();
        let scores = search
            .scores(&view(&configs, &scaled, &[1]), &[2, 4, 5], None)
            .unwrap();
        for (&a, &b) in scores.iter().zip(&reference) {
            close(a, b, 1e-12);
        }
    }
}

#[test]
fn categorical_names_and_candidate_order_do_not_change_model_scores() {
    // Kernel symmetry / sampler parameter identity: category spelling and the
    // order of the finite list carry no performance information. Ties may still
    // choose different indices; compare model scores by corresponding config.
    let configs: Vec<_> = [(1, "a"), (2, "b"), (4, "a"), (8, "c"), (16, "b")]
        .into_iter()
        .map(|(x, k)| {
            Config::new([
                ("x", ParamValue::Int(x)),
                ("kind", ParamValue::Str(k.into())),
            ])
        })
        .collect();
    let renamed: Vec<_> = configs
        .iter()
        .rev()
        .map(|c| {
            let mut params = c.params.clone();
            let name = match &params["kind"] {
                ParamValue::Str(s) => match s.as_str() {
                    "a" => "zebra",
                    "b" => "aardvark",
                    _ => "mango",
                },
                _ => unreachable!(),
            };
            params.insert("kind".into(), ParamValue::Str(name.into()));
            Config::new(params)
        })
        .collect();
    let a = BayesianSearch::new(&configs, Default::default()).unwrap();
    let b = BayesianSearch::new(&renamed, Default::default()).unwrap();
    let a_done = [observation(&configs, 0, 8.0), observation(&configs, 3, 2.0)];
    let b_done = [observation(&renamed, 4, 8.0), observation(&renamed, 1, 2.0)];
    let a_scores = a
        .scores(&view(&configs, &a_done, &[1]), &[2, 4], None)
        .unwrap();
    let b_scores = b
        .scores(&view(&renamed, &b_done, &[3]), &[2, 0], None)
        .unwrap();
    for (&x, &y) in a_scores.iter().zip(&b_scores) {
        close(x, y, 1e-12);
    }
}

#[test]
fn constant_objective_is_reproducible_through_pending_completions() {
    // Optuna full-sequence reproducibility and scikit-learn constant targets.
    // The same completion order is required; concurrency changes available data.
    let configs = space(&(0..12).collect::<Vec<_>>());
    let run = |seed| {
        let mut search = BayesianSearch::new(
            &configs,
            BayesianOptions {
                seed,
                ..Default::default()
            },
        )
        .unwrap();
        let mut completed = Vec::new();
        let mut pending = Vec::new();
        for _ in 0..configs.len() {
            while pending.len() < 2 {
                let Some(i) = search.suggest(&view(&configs, &completed, &pending)) else {
                    break;
                };
                assert!(!pending.contains(&i));
                assert!(!completed.iter().any(|o| o.trial_id == i));
                pending.push(i);
            }
            let i = pending.remove(0);
            let o = observation(&configs, i, 3.0);
            search.observe(&o);
            completed.push(o);
        }
        assert!(pending.is_empty());
        assert!(search.suggest(&view(&configs, &completed, &[])).is_none());
        assert_eq!(search.numerical_fallbacks(), 0);
        completed
            .into_iter()
            .map(|o| o.trial_id)
            .collect::<Vec<_>>()
    };
    assert_eq!(run(17), run(17));
    assert_ne!(run(17), run(18));
}

#[test]
fn all_invalid_candidates_finish_without_a_measured_winner() {
    // Helion finite-search failure coverage, adapted to our explicit invalid
    // observations and no-winner result (compiler infrastructure errors abort).
    let configs = space(&[1, 2, 3, 4]);
    let mut search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let output = Pipeline::default()
        .run(
            &configs,
            &mut search,
            |_| Ok(()),
            |(), _| Ok(()),
            |_, (), _| Err(Failure::invalid("correctness gate rejected config")),
        )
        .unwrap();
    assert_eq!(output.status, RunStatus::Completed);
    assert!(output.best_observed.is_none());
    assert_eq!(output.observations.len(), configs.len());
    assert!(output.observations.iter().all(
        |o| o.score_ms().is_none() && o.failure.as_ref().unwrap().kind == FailureKind::Invalid
    ));
    assert_eq!(search.numerical_fallbacks(), 0);
}
