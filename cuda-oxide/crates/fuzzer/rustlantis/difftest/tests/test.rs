use difftest::backends;
use difftest::{Source, run_diff_test};

#[test]
fn correct_mir() {
    let config = config::load("tests/config.toml");
    let backends = backends::from_config(config);

    let results = run_diff_test(&Source::File("tests/inputs/simple.rs".into()), backends);
    println!("{}", results);
    assert!(results.all_same());
    assert!(
        results["llvm"]
            .as_ref()
            .is_ok_and(|output| output.status.success() && output.stdout == "5\n")
    )
}

#[test]
fn invalid_mir() {
    let config = config::load("tests/config.toml");
    let backends = backends::from_config(config);

    let results = run_diff_test(
        &Source::File("tests/inputs/invalid_mir.rs".into()),
        backends,
    );
    println!("{}", results);
    assert!(results.all_same());
    assert!(results["miri"].is_err());
    assert_eq!(results.has_ub(), Some(false));
}

#[test]
fn ub() {
    let config = config::load("tests/config.toml");
    let backends = backends::from_config(config);

    let results = run_diff_test(&Source::File("tests/inputs/ub.rs".into()), backends);
    println!("{}", results);
    assert_eq!(results.has_ub(), Some(true));
}

#[test]
fn nonzero_exit_is_not_a_backend_error() {
    let config = config::load("tests/config.toml");
    let backends = backends::from_config(config);

    let results = run_diff_test(
        &Source::File("tests/inputs/nonzero_exit.rs".into()),
        backends,
    );

    for backend_name in ["miri", "llvm"] {
        let output = results[backend_name].as_ref().unwrap_or_else(|error| {
            panic!("{backend_name} treated a normal nonzero exit as an error: {error:?}")
        });

        assert_eq!(
            output.status.code(),
            Some(1),
            "{backend_name} did not preserve the program exit code"
        );
        assert_eq!(
            output.stderr.to_string_lossy(),
            "expected stderr\n",
            "{backend_name} did not preserve program stderr"
        );
    }
}
