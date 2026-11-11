/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

fn space(values: &[i64]) -> Vec<Config> {
    values
        .iter()
        .map(|&v| Config::new([("tile", ParamValue::Int(v))]))
        .collect()
}

fn observation(configs: &[Config], index: usize, time: f32) -> Observation {
    Observation {
        trial_id: index,
        config_id: configs[index].id.clone(),
        stages: Default::default(),
        trial: Some(Trial::measured(&configs[index].id, time, time, 20)),
        failure: None,
    }
}

fn view<'a>(
    configs: &'a [Config],
    completed: &'a [Observation],
    pending: &'a [usize],
) -> SearchView<'a> {
    SearchView {
        configs,
        completed,
        pending,
        remaining: None,
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn posterior_matches_sklearn_reference() {
    // Independently generated with sklearn 1.6.1, numpy 2.1.3:
    // GaussianProcessRegressor(kernel=ConstantKernel(1.3)*Matern(.6, nu=2.5),
    //     alpha=.04+1.3e-10, optimizer=None, normalize_y=False).fit(x, y)
    // predict(q, return_std=True); compare mean and std**2 (latent variance).
    let x = [[0.0, 0.0], [0.2, 1.0], [0.75, 0.0], [1.0, 1.0]];
    let y = [0.3, -0.8, 0.5, -0.1];
    let q = [[0.0, 0.0], [0.4, 0.5], [0.9, 1.0], [1.4, -0.2]];
    let means = [
        0.29104903025570256,
        -0.11831890455448082,
        -0.17241436498090834,
        0.22597481346396037,
    ];
    let vars = [
        0.0385624036746266,
        0.5494209726380859,
        0.08078549948782744,
        1.0304379258449008,
    ];
    let gp = gp::Posterior::fit(
        x.iter().map(|x| x.as_slice()).collect(),
        &y,
        0.6,
        1.3,
        0.04,
        None,
    )
    .unwrap();
    for i in 0..q.len() {
        let (m, v) = gp.predict(&q[i]);
        close(m, means[i], 1e-12);
        close(v, vars[i], 1e-12);
    }
}

#[test]
fn expected_improvement_matches_scipy_including_tails() {
    // scipy 1.15.3: z*scipy.stats.norm.cdf(z)+scipy.stats.norm.pdf(z).
    for (z, expected) in [
        (-10.0, 7.474560254595003e-25),
        (-5.0, 5.346165533833156e-8),
        (-2.0, 0.008490702616829673),
        (-1.0, 0.08331547058768629),
        (0.0, 0.3989422804014327),
        (1.0, 1.0833154705876864),
        (2.0, 2.0084907026168297),
        (5.0, 5.000000053461655),
        (10.0, 10.0),
    ] {
        close(gp::expected_improvement(z, 0.0, 1.0), expected, 2e-7);
    }
    assert_eq!(gp::expected_improvement(2.0, 1.0, 0.0), 1.0);
    assert_eq!(gp::expected_improvement(1.0, 2.0, 0.0), 0.0);
}

#[test]
fn duplicate_rows_are_stable_and_noise_prevents_overconfidence() {
    let rows = vec![&[0.5][..], &[0.5][..]];
    let quiet = gp::Posterior::fit(rows.clone(), &[1.0, 1.0], 0.5, 1.0, 1e-12, None).unwrap();
    let noisy = gp::Posterior::fit(rows, &[0.5, 1.5], 0.5, 1.0, 0.2, None).unwrap();
    assert!(quiet.predict(&[0.5]).1.is_finite());
    assert!(noisy.predict(&[0.5]).1 > quiet.predict(&[0.5]).1);
    close(noisy.predict(&[0.5]).0, 1.0 / 1.1, 1e-9);
    assert!(gp::Posterior::fit(vec![&[0.0]], &[f64::NAN], 0.5, 1.0, 0.2, None).is_none());
}

#[test]
fn prior_is_neutral_and_starting_ties_are_seeded() {
    let configs = space(&[1, 2, 4, 8, 16]);
    let search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let scores = search
        .scores(&view(&configs, &[], &[]), &[0, 1, 2, 3, 4], None)
        .unwrap();
    assert!(scores.iter().all(|&s| s == scores[0]));
    let mut selected = HashSet::new();
    for seed in 0..32 {
        let options = BayesianOptions {
            seed,
            ..Default::default()
        };
        let mut a = BayesianSearch::new(&configs, options.clone()).unwrap();
        let mut b = BayesianSearch::new(&configs, options).unwrap();
        let i = a.suggest(&view(&configs, &[], &[]));
        assert_eq!(i, b.suggest(&view(&configs, &[], &[])));
        selected.insert(i.unwrap());
    }
    assert_eq!(selected.len(), configs.len());
}

#[test]
fn explicit_hint_biases_start_but_current_measurements_override_it() {
    let configs = space(&[0, 1, 2, 3, 4]);
    let mut options = BayesianOptions::default();
    options
        .prior_log_offsets
        .insert(configs[0].id.clone(), -0.4);
    let mut search = BayesianSearch::new(&configs, options).unwrap();
    assert_eq!(search.suggest(&view(&configs, &[], &[])), Some(0));
    let observations = [
        observation(&configs, 0, 100.0),
        observation(&configs, 4, 1.0),
    ];
    let scores = search
        .scores(&view(&configs, &observations, &[]), &[1, 3], None)
        .unwrap();
    assert!(scores[1] > scores[0]);
}

#[test]
fn pending_mean_fantasy_keeps_mean_and_reduces_uncertainty() {
    let observed = gp::Posterior::fit(vec![&[0.0]], &[0.7], 0.5, 1.0, 0.01, None).unwrap();
    let fantasy = observed.predict(&[0.6]).0;
    let pending =
        gp::Posterior::fit(vec![&[0.0], &[0.6]], &[0.7, fantasy], 0.5, 1.0, 0.01, None).unwrap();
    for x in [0.0, 0.4, 0.6, 1.0] {
        let before = observed.predict(&[x]);
        let after = pending.predict(&[x]);
        close(before.0, after.0, 1e-12);
        assert!(after.1 <= before.1 + 1e-12);
    }
    let configs = space(&[0, 1, 2, 3, 4]);
    let mut search = BayesianSearch::new(&configs, Default::default()).unwrap();
    assert_eq!(search.suggest(&view(&configs, &[], &[0, 1, 2, 3])), Some(4));
}

#[test]
fn failures_and_late_measurements_do_not_train_latency_model() {
    let configs = space(&[0, 1, 2, 3, 4, 5]);
    let search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let mut observations = Vec::new();
    for (i, kind) in [
        FailureKind::Invalid,
        FailureKind::TimedOut,
        FailureKind::Cancelled,
        FailureKind::Infrastructure,
    ]
    .into_iter()
    .enumerate()
    {
        let mut o = observation(&configs, i, 0.00001);
        o.failure = Some(Failure {
            kind,
            reason: "not a valid timing".into(),
        });
        observations.push(o);
    }
    assert_eq!(
        search.scores(&view(&configs, &observations, &[]), &[4, 5], None),
        search.scores(&view(&configs, &[], &[]), &[4, 5], None)
    );
}

#[test]
fn parameter_geometry_preserves_logs_categories_and_large_integers() {
    let configs = space(&[16, 32, 64]);
    let scales = BTreeMap::from([("tile".into(), ParameterScale::Log2)]);
    let logarithmic = encode(&configs, &scales).unwrap();
    for (row, expected) in logarithmic.iter().zip([0.0, 0.5, 1.0]) {
        close(row[0], expected, 1e-12);
    }
    let huge = space(&[i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    assert_eq!(
        encode(&huge, &BTreeMap::new()).unwrap(),
        vec![vec![0.0], vec![0.5], vec![1.0]]
    );
    let logarithmic = encode(&huge, &scales).unwrap();
    close(logarithmic[1][0], 0.5, 1e-12);
    let categories =
        ["foo", "bar", "baz"].map(|s| Config::new([("kind", ParamValue::Str(s.into()))]));
    let x = encode(&categories, &BTreeMap::new()).unwrap();
    close(
        gp::covariance(&x[0], &x[1], 0.5, 1.0),
        gp::covariance(&x[1], &x[2], 0.5, 1.0),
        1e-12,
    );
    close(
        gp::covariance(&x[0], &x[1], 0.5, 1.0),
        gp::covariance(&[0.0], &[1.0], 0.5, 1.0),
        1e-12,
    );
}

#[test]
fn numerical_failure_falls_back_without_revisiting_work() {
    let configs = space(&[0, 1, 2, 3]);
    let mut search = BayesianSearch::new(&configs, Default::default()).unwrap();
    // Fault injection at the numerical boundary, after public input validation.
    search.features[0][0] = f64::NAN;
    let observations = [observation(&configs, 0, 1.0)];
    assert_eq!(
        search.suggest(&view(&configs, &observations, &[1, 2])),
        Some(3)
    );
    assert_eq!(search.numerical_fallbacks(), 1);
}

#[test]
fn rejects_malformed_spaces_and_priors() {
    let configs = space(&[0, 1]);
    for value in [f64::NAN, 0.0, -1.0] {
        assert!(BayesianSearch::new(
            &configs,
            BayesianOptions {
                length_scale: value,
                ..Default::default()
            }
        )
        .is_err());
    }
    for (id, offset) in [
        (configs[0].id.clone(), f64::NAN),
        (configs[0].id.clone(), 0.7),
        ("unknown".into(), 0.1),
    ] {
        assert!(BayesianSearch::new(
            &configs,
            BayesianOptions {
                prior_log_offsets: BTreeMap::from([(id, offset)]),
                ..Default::default()
            }
        )
        .is_err());
    }
    assert!(BayesianSearch::new(&[], Default::default()).is_err());
    assert!(BayesianSearch::new(
        &[configs[0].clone(), configs[0].clone()],
        Default::default()
    )
    .is_err());
    let mismatched = [
        configs[0].clone(),
        Config::new([("other", ParamValue::Int(1))]),
    ];
    assert!(BayesianSearch::new(&mismatched, Default::default()).is_err());
    assert!(BayesianSearch::new(
        &configs,
        BayesianOptions {
            scales: BTreeMap::from([("tile".into(), ParameterScale::Log2)]),
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn pipeline_rejects_wrong_space_before_invoking_user_callbacks() {
    let configs = space(&[0, 1]);
    let mut search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let other = space(&[2, 3]);
    let result = Pipeline::default().run(
        &other,
        &mut search,
        |_| -> Result<(), Failure> { panic!("must validate first") },
        |(), _| Ok(()),
        |_, (), _| -> Result<Trial, Failure> { panic!("must validate first") },
    );
    assert!(result.unwrap_err().reason.contains("different space"));
}

#[test]
fn expired_budget_and_malformed_views_do_not_propose_work() {
    let configs = space(&[0, 1]);
    let mut search = BayesianSearch::new(&configs, Default::default()).unwrap();
    let mut v = view(&configs, &[], &[]);
    v.remaining = Some(Duration::ZERO);
    assert_eq!(search.suggest(&v), None);
    assert_eq!(search.numerical_fallbacks(), 0);
    assert_eq!(search.suggest(&view(&configs, &[], &[2])), None);
    let mut o = observation(&configs, 0, 1.0);
    o.trial_id = 9;
    assert_eq!(search.suggest(&view(&configs, &[o], &[])), None);
}

#[test]
fn exhausts_space_once_including_failed_candidates_and_explicit_starts() {
    let configs = space(&[1, 2, 3, 4, 5]);
    let mut search = BayesianSearch::new(&configs, Default::default())
        .unwrap()
        .with_initial([2, 2])
        .unwrap();
    let mut observations = Vec::new();
    for step in 0..configs.len() {
        let i = search.suggest(&view(&configs, &observations, &[])).unwrap();
        if step == 0 {
            assert_eq!(i, 2);
        }
        assert!(!observations.iter().any(|o| o.trial_id == i));
        let mut o = observation(&configs, i, (i + 1) as f32);
        if step == 1 {
            o.failure = Some(Failure::invalid("bad configuration"));
        }
        search.observe(&o);
        observations.push(o);
    }
    assert_eq!(search.suggest(&view(&configs, &observations, &[])), None);
    assert_eq!(search.numerical_fallbacks(), 0);
}
