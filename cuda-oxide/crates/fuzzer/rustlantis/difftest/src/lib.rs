#![feature(iter_intersperse)]

pub mod backends;

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Display},
    ops::Index,
    path::PathBuf,
    time::Instant,
};

use backends::{Backend, CompExecError, ExecResult};
use colored::Colorize;
use log::{debug, log_enabled};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

pub enum Source {
    File(PathBuf),
    Stdin(String),
}

impl Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::File(path) => f.write_str(&path.to_string_lossy()),
            Source::Stdin(_) => f.write_str("[stdin]"),
        }
    }
}

pub struct ExecResults {
    // Equivalence classes of exec results and backends
    results: HashMap<ExecResult, HashSet<String>>,
}

impl ExecResults {
    fn exec_results_eq(lhs: &ExecResult, rhs: &ExecResult) -> bool {
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => lhs.stdout == rhs.stdout,
            _ => lhs == rhs,
        }
    }

    fn insert_exec_result(
        eq_classes: &mut HashMap<ExecResult, HashSet<String>>,
        name: String,
        result: ExecResult,
    ) {
        for (class_result, names) in eq_classes.iter_mut() {
            if Self::exec_results_eq(class_result, &result) {
                names.insert(name);
                return;
            }
        }

        eq_classes.insert(result, HashSet::from([name]));
    }

    fn from_exec_results(map: impl Iterator<Item = (String, ExecResult)>) -> Self {
        let mut map = map;

        let Some((first_name, first_result)) = map.next() else {
            return Self {
                results: HashMap::new(),
            };
        };

        let mut first_names = HashSet::from([first_name]);

        while let Some((name, result)) = map.next() {
            if Self::exec_results_eq(&first_result, &result) {
                first_names.insert(name);
                continue;
            }

            // Slow path: at least one backend disagrees, so split into equivalence classes.
            let mut eq_classes = HashMap::new();
            eq_classes.insert(first_result, first_names);

            Self::insert_exec_result(&mut eq_classes, name, result);

            for (name, result) in map {
                Self::insert_exec_result(&mut eq_classes, name, result);
            }

            return Self {
                results: eq_classes,
            };
        }

        let mut eq_classes = HashMap::new();
        eq_classes.insert(first_result, first_names);

        Self {
            results: eq_classes,
        }
    }

    pub fn all_same(&self) -> bool {
        self.results.len() == 1
    }

    pub fn all_success(&self) -> bool {
        self.results.keys().all(|r| r.is_ok())
    }

    pub fn has_ub(&self) -> Option<bool> {
        self.results
            .iter()
            .find_map(|(result, backends)| {
                if backends.contains("miri") {
                    Some(result)
                } else {
                    None
                }
            })
            .map(|result| {
                result.clone().is_err_and(|err| {
                    err.0
                        .stderr
                        .to_string_lossy()
                        .contains("Undefined Behavior")
                })
            })
    }

    pub fn miri_result(&self) -> Option<&ExecResult> {
        self.results.iter().find_map(|(result, backends)| {
            if backends.contains("miri") {
                Some(result)
            } else {
                None
            }
        })
    }
}

impl Index<&str> for ExecResults {
    type Output = ExecResult;

    fn index(&self, index: &str) -> &Self::Output {
        for (result, names) in &self.results {
            if names.contains(index) {
                return result;
            }
        }
        panic!("no result for {index}")
    }
}

impl fmt::Display for ExecResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (result, names) in &self.results {
            f.write_fmt(format_args!(
                "{} produced the following output:\n",
                names
                    .iter()
                    .map(String::as_str)
                    .intersperse(", ")
                    .collect::<String>()
                    .blue()
            ))?;
            match result {
                Ok(out) => {
                    f.write_fmt(format_args!("stdout:\n{}", out.stdout.to_string_lossy()))?;
                }
                Err(CompExecError(out)) => {
                    f.write_fmt(format_args!("status: {}\n", out.status))?;
                    f.write_fmt(format_args!(
                        "stdout:\n{}================\n",
                        out.stdout.to_string_lossy()
                    ))?;
                    f.write_fmt(format_args!(
                        "{}:\n{}================\n",
                        "stderr".red(),
                        out.stderr.to_string_lossy()
                    ))?;
                }
            }
        }
        Ok(())
    }
}

pub fn run_diff_test<'a>(
    source: &Source,
    backends: HashMap<String, Box<dyn Backend + 'a>>,
) -> ExecResults {
    let target_dir = tempfile::tempdir().unwrap();
    let exec_results: HashMap<String, ExecResult> = backends
        .into_par_iter()
        .map(|(name, b)| {
            let target_path = target_dir.path().join(&name);
            let result = if log_enabled!(log::Level::Debug) {
                let time = Instant::now();
                let result = b.execute(source, &target_path);
                let dur = time.elapsed();
                debug!("{name} took {}s", dur.as_secs_f32());
                result
            } else {
                b.execute(source, &target_path)
            };
            (name.clone(), result)
        })
        .collect();

    ExecResults::from_exec_results(exec_results.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ffi::OsString, process::ExitStatus};

    fn exit_status(code: u32) -> ExitStatus {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            ExitStatus::from_raw((code as i32) << 8)
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            ExitStatus::from_raw(code)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = code;
            unimplemented!("test exit statuses are only defined for Unix and Windows")
        }
    }

    fn process_output(status: u32, stdout: &str, stderr: &str) -> backends::ProcessOutput {
        backends::ProcessOutput {
            status: exit_status(status),
            stdout: OsString::from(stdout),
            stderr: OsString::from(stderr),
        }
    }

    fn ok(status: u32, stdout: &str, stderr: &str) -> ExecResult {
        Ok(process_output(status, stdout, stderr))
    }

    fn err(status: u32, stdout: &str, stderr: &str) -> ExecResult {
        Err(CompExecError(process_output(status, stdout, stderr)))
    }

    fn class_for<'a>(results: &'a ExecResults, backend: &str) -> &'a HashSet<String> {
        results
            .results
            .values()
            .find(|names| names.contains(backend))
            .expect("backend has an equivalence class")
    }

    #[test]
    fn empty_input_stays_empty() {
        let results = ExecResults::from_exec_results(std::iter::empty());

        assert!(results.results.is_empty());
        assert!(!results.all_same());
        assert!(results.all_success());
    }

    #[test]
    fn successful_results_are_equivalent_by_stdout_only() {
        let results = ExecResults::from_exec_results(
            [
                ("llvm".to_owned(), ok(0, "same output", "first stderr")),
                (
                    "cranelift".to_owned(),
                    ok(7, "same output", "different stderr"),
                ),
            ]
            .into_iter(),
        );

        assert_eq!(results.results.len(), 1);
        assert_eq!(
            class_for(&results, "llvm"),
            &HashSet::from(["llvm".to_owned(), "cranelift".to_owned()])
        );
        assert!(results.all_same());
        assert!(results.all_success());
    }

    #[test]
    fn errors_use_structural_equivalence() {
        let matching = err(1, "compiler stdout", "same error");
        let results = ExecResults::from_exec_results(
            [
                ("miri".to_owned(), matching.clone()),
                ("miri-opt".to_owned(), matching),
                (
                    "miri-unchecked".to_owned(),
                    err(1, "compiler stdout", "different error"),
                ),
            ]
            .into_iter(),
        );

        assert_eq!(results.results.len(), 2);
        assert_eq!(
            class_for(&results, "miri"),
            &HashSet::from(["miri".to_owned(), "miri-opt".to_owned()])
        );
        assert_eq!(
            class_for(&results, "miri-unchecked"),
            &HashSet::from(["miri-unchecked".to_owned()])
        );
        assert!(!results.all_same());
        assert!(!results.all_success());
    }

    #[test]
    fn mismatch_falls_back_and_builds_all_equivalence_classes() {
        let repeated_error = err(1, "", "compile failed");
        let results = ExecResults::from_exec_results(
            [
                ("llvm".to_owned(), ok(0, "a", "")),
                ("cranelift".to_owned(), ok(0, "b", "")),
                ("gcc".to_owned(), ok(9, "a", "ignored")),
                ("miri".to_owned(), repeated_error.clone()),
                ("miri-opt".to_owned(), repeated_error),
            ]
            .into_iter(),
        );

        assert_eq!(results.results.len(), 3);
        assert_eq!(
            class_for(&results, "llvm"),
            &HashSet::from(["llvm".to_owned(), "gcc".to_owned()])
        );
        assert_eq!(
            class_for(&results, "cranelift"),
            &HashSet::from(["cranelift".to_owned()])
        );
        assert_eq!(
            class_for(&results, "miri"),
            &HashSet::from(["miri".to_owned(), "miri-opt".to_owned()])
        );
    }
}
