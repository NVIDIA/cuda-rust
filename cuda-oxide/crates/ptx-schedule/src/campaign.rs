/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end schedule campaigns for existing cuda-oxide examples.
//!
//! A campaign builds an example once, mutates the generated PTX in memory,
//! patches a copy of the executable's embedded `.oxart` section for each
//! mutation, and runs that copy without changing the production loader.

use crate::{InjectionOptions, RewriteReport};
use oxide_artifacts::{
    ArtifactBundleSpec, ArtifactEntrySpec, ArtifactPayloadKind, ArtifactPayloadSpec,
    build_artifact_blob, read_artifact_bundles_from_object_bytes,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CampaignError {
    #[error("invalid seed range '{0}', expected START..END with END > START")]
    InvalidSeedRange(String),
    #[error("confirmation runs must be at least 1, got {0}")]
    InvalidConfirmRuns(u32),
    #[error("example '{0}' was not found at {1}")]
    ExampleNotFound(String, PathBuf),
    #[error("campaign I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("could not parse Cargo metadata: {0}")]
    Metadata(String),
    #[error("cuda-oxide build failed with {0}")]
    BuildFailed(ExitStatus),
    #[error("generated PTX was not found at {0}")]
    MissingPtx(PathBuf),
    #[error("example executable was not found at {0}")]
    MissingExecutable(PathBuf),
    #[error("PTX schedule rewrite failed: {0}")]
    Schedule(#[from] crate::ScheduleError),
    #[error("could not patch embedded PTX: {0}")]
    ArtifactPatch(String),
    #[error("could not serialize campaign report: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
pub struct CampaignOptions {
    pub workspace_root: PathBuf,
    pub simt_root: PathBuf,
    pub oxide_binary: PathBuf,
    pub example: String,
    /// Half-open seed interval: `0..100` runs seeds 0 through 99.
    pub seed_start: u64,
    pub seed_end: u64,
    pub intensity: f64,
    pub max_sleep_ns: u32,
    pub timeout: Duration,
    pub arch: Option<String>,
    pub focus: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub keep_going: bool,
    /// Total executions for a finding, including the initial execution.
    pub confirm_runs: u32,
    /// Treat a changed stdout stream as a finding when the example has no
    /// explicit failure marker.
    pub compare_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum RunKind {
    Pass,
    Skipped,
    Hang,
    Crash,
    Mismatch,
    OutputChanged,
    GpuWedged,
    /// The harness failed to perturb or patch this seed, so the variant
    /// never ran. Not a schedule finding.
    HarnessError,
}

impl RunKind {
    fn finding_label(&self) -> Option<&'static str> {
        match self {
            Self::Mismatch => Some("SCHEDULE-SENSITIVE CORRECTNESS FAILURE"),
            Self::OutputChanged => Some("SCHEDULE-SENSITIVE OUTPUT CHANGE"),
            Self::Hang => Some("TIMEOUT CANDIDATE"),
            Self::Crash => Some("CRASH CANDIDATE"),
            Self::GpuWedged => Some("GPU WEDGE CANDIDATE"),
            Self::Pass | Self::Skipped | Self::HarnessError => None,
        }
    }

    fn is_finding(&self) -> bool {
        self.finding_label().is_some()
    }

    /// What this outcome means when it is the *baseline* run.
    ///
    /// The campaign and its callers have to agree on this, and the
    /// distinction that matters is not pass/not-pass. An example that declares
    /// it cannot run on this device has not failed: `scripts/smoketest.sh`
    /// reports that decline as `PASS (skipped)`, and #665 exists because
    /// reporting it as anything else made an arch-gated example
    /// indistinguishable from a broken one.
    ///
    /// The match is exhaustive on purpose, so a new [`RunKind`] cannot join a
    /// class by default.
    pub fn baseline_verdict(&self) -> BaselineVerdict {
        match self {
            Self::Pass => BaselineVerdict::Usable,
            Self::Skipped => BaselineVerdict::Declined,
            Self::Hang
            | Self::Crash
            | Self::Mismatch
            | Self::OutputChanged
            | Self::GpuWedged
            | Self::HarnessError => BaselineVerdict::Broken,
        }
    }
}

/// Whether a baseline outcome lets a campaign proceed, and if not, why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaselineVerdict {
    /// The baseline passed; schedule variants are meaningful.
    Usable,
    /// The example declared it cannot run here. Not a failure.
    Declined,
    /// The baseline did not complete, so nothing could be concluded from a
    /// variant that behaved the same way.
    Broken,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunResult {
    pub kind: RunKind,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SeedResult {
    pub seed: u64,
    pub artifact_dir: PathBuf,
    pub report: RewriteReport,
    pub run: RunResult,
    pub confirmation: Option<ConfirmationSummary>,
    pub replay: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConfirmationSummary {
    pub attempts: u32,
    pub findings: u32,
    pub confirmed: bool,
    pub outcomes: Vec<RunKind>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StaticSiteReport {
    pub sites_total: usize,
    pub sites_by_kind: BTreeMap<String, usize>,
    pub sites: Vec<crate::ScheduleSite>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CampaignSettings {
    pub seed_start: u64,
    pub seed_end: u64,
    pub intensity: f64,
    pub max_sleep_ns: u32,
    pub timeout_secs: u64,
    pub arch: Option<String>,
    pub focus: Option<String>,
    pub confirm_runs: u32,
    pub compare_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CampaignSummary {
    pub example: String,
    pub ptx: PathBuf,
    pub executable: PathBuf,
    pub settings: CampaignSettings,
    pub static_sites: StaticSiteReport,
    pub baseline: RunResult,
    pub seeds: Vec<SeedResult>,
}

impl CampaignSummary {
    pub fn finding_count(&self) -> usize {
        self.seeds
            .iter()
            .filter(|seed| seed.run.kind.is_finding())
            .count()
    }
}

pub fn parse_seed_range(value: &str) -> Result<(u64, u64), CampaignError> {
    let Some((start, end)) = value.split_once("..") else {
        return Err(CampaignError::InvalidSeedRange(value.to_string()));
    };
    let start = start
        .parse::<u64>()
        .map_err(|_| CampaignError::InvalidSeedRange(value.to_string()))?;
    let end = end
        .parse::<u64>()
        .map_err(|_| CampaignError::InvalidSeedRange(value.to_string()))?;
    if start >= end {
        return Err(CampaignError::InvalidSeedRange(value.to_string()));
    }
    Ok((start, end))
}

pub fn run_campaign(options: &CampaignOptions) -> Result<CampaignSummary, CampaignError> {
    if options.seed_start >= options.seed_end {
        return Err(CampaignError::InvalidSeedRange(format!(
            "{}..{}",
            options.seed_start, options.seed_end
        )));
    }
    if options.confirm_runs == 0 {
        return Err(CampaignError::InvalidConfirmRuns(options.confirm_runs));
    }

    let example_dir = options
        .simt_root
        .join("crates/rustc-codegen-cuda/examples")
        .join(&options.example);
    if !example_dir.join("Cargo.toml").is_file() {
        return Err(CampaignError::ExampleNotFound(
            options.example.clone(),
            example_dir,
        ));
    }

    let build_status = build_example(options)?;
    if !build_status.success() {
        return Err(CampaignError::BuildFailed(build_status));
    }

    let stem = options.example.replace('-', "_");
    let ptx_path = example_dir.join(format!("{stem}.ptx"));
    if !ptx_path.is_file() {
        return Err(CampaignError::MissingPtx(ptx_path));
    }
    let executable = find_executable(&example_dir, &options.example)?;
    let pristine = fs::read_to_string(&ptx_path)?;
    let analysis = crate::analyze_ptx(&pristine)?;
    let static_sites = static_site_report(&analysis);
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        options
            .simt_root
            .join("crates/fuzzer/artifacts/schedule")
            .join(&options.example)
    });
    fs::create_dir_all(&output_dir)?;
    fs::write(
        output_dir.join("sites.json"),
        serde_json::to_vec_pretty(&static_sites)?,
    )?;
    println!(
        "schedule-fuzz: static sites={} kinds={}",
        static_sites.sites_total,
        format_site_kinds(&static_sites.sites_by_kind)
    );

    println!("schedule-fuzz: baseline {}", executable.display());
    let baseline = run_binary(&executable, &example_dir, options.timeout);
    println!("schedule-fuzz: baseline {:?}", baseline.kind);
    if baseline.kind.baseline_verdict() != BaselineVerdict::Usable {
        let summary = CampaignSummary {
            example: options.example.clone(),
            ptx: ptx_path.clone(),
            executable: executable.clone(),
            settings: campaign_settings(options),
            static_sites,
            baseline,
            seeds: Vec::new(),
        };
        match summary.baseline.kind.baseline_verdict() {
            BaselineVerdict::Declined => println!(
                "schedule-fuzz: BASELINE DECLINED: the example opted out on this \
                 device; no schedule variants were run"
            ),
            _ => println!(
                "schedule-fuzz: BASELINE FAILURE: {:?}; no schedule variants were run",
                summary.baseline.kind
            ),
        }
        fs::write(
            output_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
        return Ok(summary);
    }

    let mut seeds = Vec::new();
    for seed in options.seed_start..options.seed_end {
        let artifact_dir = output_dir.join(format!("seed-{seed}"));
        // A perturbation or patching failure is scoped to this seed: record
        // it and keep going so the campaign still covers the remaining seeds
        // and still writes summary.json.
        let result = run_seed(
            options,
            seed,
            &artifact_dir,
            &pristine,
            &executable,
            &example_dir,
            &baseline,
        )
        .unwrap_or_else(|error| {
            harness_error_result(seed, options.intensity, artifact_dir, &error)
        });
        print_seed_result(&result);
        let stop = matches!(result.run.kind, RunKind::GpuWedged) && !options.keep_going;
        seeds.push(result);
        if stop {
            break;
        }
    }

    let summary = CampaignSummary {
        example: options.example.clone(),
        ptx: ptx_path,
        executable,
        settings: campaign_settings(options),
        static_sites,
        baseline,
        seeds,
    };
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    print_campaign_result(&summary, &output_dir);
    Ok(summary)
}

fn run_seed(
    options: &CampaignOptions,
    seed: u64,
    artifact_dir: &Path,
    pristine: &str,
    executable: &Path,
    example_dir: &Path,
    baseline: &RunResult,
) -> Result<SeedResult, CampaignError> {
    let rewrite = crate::perturb_ptx(
        pristine,
        &InjectionOptions {
            seed,
            intensity: options.intensity,
            max_sleep_ns: options.max_sleep_ns,
            focus: options.focus.clone(),
        },
    )?;
    fs::create_dir_all(artifact_dir)?;
    let mutated_ptx = artifact_dir.join("module.ptx");
    let report_path = artifact_dir.join("report.json");
    let stdout_path = artifact_dir.join("stdout.log");
    let stderr_path = artifact_dir.join("stderr.log");
    let replay_path = artifact_dir.join("replay.sh");
    fs::write(&mutated_ptx, &rewrite.ptx)?;
    fs::write(&report_path, serde_json::to_vec_pretty(&rewrite.report)?)?;

    let variant_executable = artifact_dir.join(
        executable
            .file_name()
            .ok_or_else(|| CampaignError::ArtifactPatch("executable has no file name".into()))?,
    );
    patch_executable(
        executable,
        &variant_executable,
        pristine.as_bytes(),
        rewrite.ptx.as_bytes(),
    )?;
    write_replay_script(&replay_path, example_dir, &variant_executable)?;

    let mut run = run_variant(
        &variant_executable,
        executable,
        example_dir,
        options.timeout,
    );
    classify_output_change(baseline, &mut run, options.compare_output);

    fs::write(&stdout_path, &run.stdout)?;
    fs::write(&stderr_path, &run.stderr)?;
    let confirmation = if run.kind.is_finding() && options.confirm_runs > 1 {
        let mut outcomes = vec![run.kind.clone()];
        let mut findings = 1;
        for attempt in 1..options.confirm_runs {
            let mut confirmed_run = run_variant(
                &variant_executable,
                executable,
                example_dir,
                options.timeout,
            );
            classify_output_change(baseline, &mut confirmed_run, options.compare_output);
            if confirmed_run.kind.is_finding() {
                findings += 1;
            }
            fs::write(
                artifact_dir.join(format!("confirm-{attempt}-stdout.log")),
                &confirmed_run.stdout,
            )?;
            fs::write(
                artifact_dir.join(format!("confirm-{attempt}-stderr.log")),
                &confirmed_run.stderr,
            )?;
            outcomes.push(confirmed_run.kind);
        }
        Some(ConfirmationSummary {
            attempts: options.confirm_runs,
            findings,
            confirmed: findings == options.confirm_runs,
            outcomes,
        })
    } else {
        None
    };
    Ok(SeedResult {
        seed,
        artifact_dir: artifact_dir.to_path_buf(),
        report: rewrite.report,
        run,
        confirmation,
        replay: replay_path,
    })
}

/// The seed's variant never ran. The error text lands in the run's stderr,
/// a zeroed rewrite report keeps summary.json uniform, and the replay path
/// names where the script would have been written.
fn harness_error_result(
    seed: u64,
    intensity: f64,
    artifact_dir: PathBuf,
    error: &CampaignError,
) -> SeedResult {
    let replay = artifact_dir.join("replay.sh");
    SeedResult {
        seed,
        artifact_dir,
        report: RewriteReport {
            seed,
            intensity,
            sites_total: 0,
            sites_injected: 0,
            injected_ns_per_visit: 0,
            decisions: Vec::new(),
        },
        run: RunResult {
            kind: RunKind::HarnessError,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: error.to_string(),
        },
        confirmation: None,
        replay,
    }
}

fn print_seed_result(result: &SeedResult) {
    if matches!(result.run.kind, RunKind::HarnessError) {
        println!(
            "schedule-fuzz: seed={} HARNESS ERROR: {}",
            result.seed,
            result.run.stderr.trim()
        );
    } else if let Some(label) = result.run.kind.finding_label() {
        println!("schedule-fuzz: FINDING seed={}: {}", result.seed, label);
        if let Some(confirmation) = &result.confirmation {
            println!(
                "  confirmation: {}/{} reproductions{}",
                confirmation.findings,
                confirmation.attempts,
                if confirmation.confirmed {
                    " (CONFIRMED)"
                } else {
                    ""
                }
            );
        } else {
            println!("  confirmation: not requested");
        }
        println!("  replay: {}", result.artifact_dir.display());
        println!(
            "  logs:   {}/stdout.log and stderr.log",
            result.artifact_dir.display()
        );
    } else {
        println!(
            "schedule-fuzz: seed={} PASS sites={}/{}",
            result.seed, result.report.sites_injected, result.report.sites_total
        );
    }
}

fn print_campaign_result(summary: &CampaignSummary, output_dir: &Path) {
    let findings: Vec<&SeedResult> = summary
        .seeds
        .iter()
        .filter(|result| result.run.kind.finding_label().is_some())
        .collect();

    let harness_errors = summary
        .seeds
        .iter()
        .filter(|result| matches!(result.run.kind, RunKind::HarnessError))
        .count();

    println!();
    println!("=== schedule-fuzz result ===");
    println!("example: {}", summary.example);
    println!("baseline: PASS");
    println!("variants: {}", summary.seeds.len());
    if harness_errors > 0 {
        println!("harness errors: {harness_errors} (seeds not run, not schedule findings)");
    }
    if findings.is_empty() {
        println!("RESULT: no schedule-sensitive failures found");
    } else {
        println!("RESULT: FOUND {} CANDIDATE(S)", findings.len());
        for result in findings {
            let confirmation = result
                .confirmation
                .as_ref()
                .map(|confirmation| {
                    format!(
                        ", reproduced {}/{}{}",
                        confirmation.findings,
                        confirmation.attempts,
                        if confirmation.confirmed {
                            " CONFIRMED"
                        } else {
                            ""
                        }
                    )
                })
                .unwrap_or_default();
            println!(
                "  seed {}: {}{} [{}]",
                result.seed,
                result.run.kind.finding_label().unwrap_or("failure"),
                confirmation,
                result.artifact_dir.display()
            );
        }
    }
    println!("report: {}/summary.json", output_dir.display());
    println!("sites:  {}/sites.json", output_dir.display());
}

fn campaign_settings(options: &CampaignOptions) -> CampaignSettings {
    CampaignSettings {
        seed_start: options.seed_start,
        seed_end: options.seed_end,
        intensity: options.intensity,
        max_sleep_ns: options.max_sleep_ns,
        timeout_secs: options.timeout.as_secs(),
        arch: options.arch.clone(),
        focus: options.focus.clone(),
        confirm_runs: options.confirm_runs,
        compare_output: options.compare_output,
    }
}

fn static_site_report(analysis: &crate::ScheduleAnalysis) -> StaticSiteReport {
    let mut sites_by_kind = BTreeMap::new();
    for site in analysis.sites() {
        *sites_by_kind.entry(format!("{:?}", site.kind)).or_insert(0) += 1;
    }
    StaticSiteReport {
        sites_total: analysis.sites().len(),
        sites_by_kind,
        sites: analysis.sites().to_vec(),
    }
}

fn format_site_kinds(sites_by_kind: &BTreeMap<String, usize>) -> String {
    sites_by_kind
        .iter()
        .map(|(kind, count)| format!("{kind}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn write_replay_script(path: &Path, cwd: &Path, executable: &Path) -> Result<(), CampaignError> {
    let mut script = "#!/bin/sh\nset -eu\n".to_string();
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "GEMM_SOL_PHASE",
    ] {
        if let Ok(value) = std::env::var(key) {
            script.push_str(&format!("export {key}={}\n", shell_quote_value(&value)));
        }
    }
    script.push_str(&format!(
        "cd {}\nexec {}\n",
        shell_quote(cwd),
        shell_quote(executable)
    ));
    fs::write(path, script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn shell_quote(path: &Path) -> String {
    shell_quote_value(&path.display().to_string())
}

fn shell_quote_value(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn build_example(options: &CampaignOptions) -> Result<ExitStatus, CampaignError> {
    let mut command = Command::new(&options.oxide_binary);
    command.args(["build", &options.example]);
    if let Some(arch) = &options.arch {
        command.args(["--arch", arch]);
    }
    command.current_dir(&options.workspace_root);
    Ok(command.status()?)
}

fn find_executable(example_dir: &Path, example: &str) -> Result<PathBuf, CampaignError> {
    let metadata = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(example_dir)
        .output()?;
    if !metadata.status.success() {
        return Err(CampaignError::Metadata(
            String::from_utf8_lossy(&metadata.stderr).trim().to_string(),
        ));
    }
    let document: Value = serde_json::from_slice(&metadata.stdout)
        .map_err(|error| CampaignError::Metadata(error.to_string()))?;
    let target_dir = document
        .get("target_directory")
        .and_then(Value::as_str)
        .ok_or_else(|| CampaignError::Metadata("target_directory is missing".to_string()))?;
    let packages = document
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| CampaignError::Metadata("package is missing".to_string()))?;
    // Examples with a nested kernel-lib crate are multi-package workspaces,
    // and cargo does not put the binary package first, so every package's bin
    // targets are candidates. A bin named after the example or picked by its
    // package's default_run wins; otherwise the first bin found is used.
    let normalized_example = example.replace('-', "_");
    let mut bins: Vec<(&str, bool)> = Vec::new();
    for package in packages {
        let default_run = package.get("default_run").and_then(Value::as_str);
        let Some(targets) = package.get("targets").and_then(Value::as_array) else {
            continue;
        };
        for target in targets {
            let is_bin = target
                .get("kind")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
            let Some(name) = target.get("name").and_then(Value::as_str) else {
                continue;
            };
            if is_bin {
                bins.push((name, Some(name) == default_run));
            }
        }
    }
    let target_name = bins
        .iter()
        .find(|(name, is_default_run)| *is_default_run || *name == normalized_example)
        .or_else(|| bins.first())
        .map(|(name, _)| *name)
        .ok_or_else(|| CampaignError::Metadata("no binary target found".to_string()))?;
    let executable = PathBuf::from(target_dir).join("release").join(target_name);
    if executable.is_file() {
        Ok(executable)
    } else {
        Err(CampaignError::MissingExecutable(executable))
    }
}

fn run_variant(variant: &Path, pristine: &Path, cwd: &Path, timeout: Duration) -> RunResult {
    let mut run = run_binary(variant, cwd, timeout);

    // A CUDA watchdog timeout can leave the device unusable. Re-run the
    // pristine binary before continuing so a real device wedge is not
    // misreported as a collection of independent schedule failures.
    if matches!(run.kind, RunKind::Hang) {
        let health = run_binary(pristine, cwd, timeout);
        if !matches!(health.kind, RunKind::Pass) {
            run.kind = RunKind::GpuWedged;
        }
    }
    run
}

fn classify_output_change(baseline: &RunResult, run: &mut RunResult, compare_output: bool) {
    if compare_output
        && matches!(run.kind, RunKind::Pass)
        && run.stdout.trim() != baseline.stdout.trim()
    {
        run.kind = RunKind::OutputChanged;
    }
}

fn run_binary(executable: &Path, cwd: &Path, timeout: Duration) -> RunResult {
    let mut command = Command::new(executable);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let Ok(mut child) = command.spawn() else {
        return RunResult {
            kind: RunKind::Crash,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: "could not start example executable".to_string(),
        };
    };
    // Drain both pipes on their own threads while the watchdog polls. A child
    // that writes more than a pipe buffer would otherwise block on the full
    // pipe until the watchdog kills it, turning a large failure dump into a
    // spurious hang.
    let stdout_reader = drain_pipe(child.stdout.take());
    let stderr_reader = drain_pipe(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                let _ = child.kill();
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(_) => break,
        }
    }

    let status = child.wait();
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    let (exit_code, success, stderr) = match status {
        Ok(status) => (status.code(), status.success(), stderr),
        Err(error) => (None, false, error.to_string()),
    };
    // Both marker predicates below fold case themselves, so this is the raw
    // combined stream rather than a lowercased copy.
    let combined = format!("{stdout}\n{stderr}");
    let kind = classify_process_result(timed_out, success, &combined);
    RunResult {
        kind,
        exit_code,
        timed_out,
        stdout,
        stderr,
    }
}

/// Classify stronger failures before a graceful skip. This is the same order
/// as `scripts/smoketest.sh`:
///
/// ```text
/// timeout -> nonzero exit -> mismatch -> skip -> pass
/// ```
fn classify_process_result(timed_out: bool, success: bool, output: &str) -> RunKind {
    if timed_out {
        RunKind::Hang
    } else if !success {
        RunKind::Crash
    } else if has_mismatch_marker(output) {
        RunKind::Mismatch
    } else if has_skip_marker(output) {
        RunKind::Skipped
    } else {
        RunKind::Pass
    }
}

fn drain_pipe<R: io::Read + Send + 'static>(pipe: Option<R>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut buffer);
        }
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

fn patch_executable(
    original: &Path,
    variant: &Path,
    pristine_ptx: &[u8],
    mutated_ptx: &[u8],
) -> Result<(), CampaignError> {
    fs::copy(original, variant)?;
    let original_bytes = fs::read(original)?;
    let bundles = read_artifact_bundles_from_object_bytes(&original_bytes)
        .map_err(|error| CampaignError::ArtifactPatch(error.to_string()))?;
    let section = rebuild_artifact_section(bundles, pristine_ptx, mutated_ptx)?;
    let section_path = variant.with_extension("oxart");
    fs::write(&section_path, section)?;

    let objcopy = std::env::var_os("OBJCOPY").unwrap_or_else(|| "objcopy".into());
    let section_arg = format!(".oxart={}", section_path.display());
    let status = Command::new(objcopy)
        .arg("--update-section")
        .arg(&section_arg)
        .arg(variant)
        .status()
        .map_err(|error| CampaignError::ArtifactPatch(format!("could not run objcopy: {error}")))?;
    if !status.success() {
        return Err(CampaignError::ArtifactPatch(format!(
            "objcopy failed with {status}"
        )));
    }
    Ok(())
}

fn rebuild_artifact_section(
    mut bundles: Vec<oxide_artifacts::OwnedArtifactBundle>,
    pristine_ptx: &[u8],
    mutated_ptx: &[u8],
) -> Result<Vec<u8>, CampaignError> {
    let ptx_count = bundles
        .iter()
        .flat_map(|bundle| bundle.payloads.iter())
        .filter(|payload| payload.kind == ArtifactPayloadKind::Ptx)
        .count();
    let mut replaced = false;
    let mut section = Vec::new();

    for bundle in &mut bundles {
        for payload in &mut bundle.payloads {
            let exact_match = payload.kind == ArtifactPayloadKind::Ptx
                && payload.bytes.as_slice() == pristine_ptx;
            let only_ptx_fallback =
                payload.kind == ArtifactPayloadKind::Ptx && ptx_count == 1 && !replaced;
            if exact_match || only_ptx_fallback {
                payload.bytes = mutated_ptx.to_vec();
                replaced = true;
            }
        }

        let payloads = bundle
            .payloads
            .iter()
            .map(|payload| ArtifactPayloadSpec::new(payload.kind, &payload.name, &payload.bytes))
            .collect();
        let entries = bundle
            .entries
            .iter()
            .map(|entry| {
                let spec = ArtifactEntrySpec::new(&entry.symbol, entry.kind);
                match entry.metadata {
                    Some(metadata) => spec.with_metadata(metadata),
                    None => spec,
                }
            })
            .collect();
        let spec = ArtifactBundleSpec {
            name: &bundle.name,
            target: &bundle.target,
            compile_options: bundle.compile_options,
            payloads,
            entries,
        };
        let blob = build_artifact_blob(&spec)
            .map_err(|error| CampaignError::ArtifactPatch(error.to_string()))?;
        section.extend(blob);
    }

    if !replaced {
        return Err(CampaignError::ArtifactPatch(
            "the generated PTX was not found in the executable's .oxart section".into(),
        ));
    }
    Ok(section)
}

/// Both spellings an example uses to decline a run.
///
/// The convention belongs to `scripts/smoketest.sh`, whose `verdict_standard`
/// greps `^[[:space:]]*(skipping:|pass \(skipped\))` case-insensitively and
/// whose own comment names the second form: "`PASS (skipped): ...` form below
/// sm_75". `generated_ldmatrix` prints exactly that when the device is under
/// sm_75.
///
/// Only the first form used to be accepted here. An example that declined with
/// the second one therefore exited 0 with no mismatch marker, so `run_binary`
/// called it [`RunKind::Pass`], the baseline gate let the campaign through, and
/// every seed "passed" a kernel that never ran -- a clean report from a
/// campaign that measured nothing.
const SKIP_MARKERS: [&str; 2] = ["skipping:", "pass (skipped)"];

/// Case is folded here rather than by the caller.
///
/// This matches the smoketest's `grep -i` rule and lets callers pass the raw
/// process output. For example, `Skipping:` and `PASS (skipped):` both match.
fn has_skip_marker(output: &str) -> bool {
    output.lines().any(|line| {
        let line = line.trim_start().as_bytes();
        SKIP_MARKERS.iter().any(|marker| {
            let marker = marker.as_bytes();
            line.len() >= marker.len() && line[..marker.len()].eq_ignore_ascii_case(marker)
        })
    })
}

/// Case is folded here too, so `MISMATCH` and `Mismatch` mean the same thing.
fn has_mismatch_marker(output: &str) -> bool {
    let output = output.to_ascii_lowercase();
    [
        "mismatch",
        "max error too large",
        "barrier sync failed",
        "fail:",
        "failed:",
        "failed!",
        "not unique",
        "incorrect",
        "wrong",
        "wrong answer",
        "does not match",
        "did not match",
        "validation failed",
        "deadlock",
    ]
    .iter()
    .any(|marker| output.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The markers must match the way the smoketest's `grep -i` does, without
    /// the caller having lowercased anything first. Every spelling here is one
    /// an example or a harness actually prints.
    #[test]
    fn the_marker_predicates_fold_case_themselves() {
        for declined in [
            "Skipping: cluster launch requires sm_90",
            "SKIPPING: needs two devices",
            "PASS (skipped): ldmatrix.m8n8.x4.b16 requires sm_75+",
            "  Pass (Skipped): no peer access",
        ] {
            assert!(has_skip_marker(declined), "{declined}");
        }
        for failed in [
            "MISMATCH at index 3",
            "Mismatch: host and device disagree",
            "Max Error Too Large",
            "DEADLOCK detected",
            "Validation Failed",
        ] {
            assert!(has_mismatch_marker(failed), "{failed}");
        }
        // Folding case must not widen what counts as a marker.
        for ran in [
            "pass",
            "PASS",
            "pass: 1024 elements verified",
            "SUCCESS",
            "no skipping: here",
        ] {
            assert!(!has_skip_marker(ran), "{ran}");
        }
        for ran in ["all checks passed", "PASS", "1024 elements verified"] {
            assert!(!has_mismatch_marker(ran), "{ran}");
        }
    }

    #[test]
    fn failures_take_priority_over_skip_markers() {
        assert!(matches!(
            classify_process_result(true, true, "Skipping: slow device"),
            RunKind::Hang
        ));
        assert!(matches!(
            classify_process_result(false, false, "Skipping: after an error"),
            RunKind::Crash
        ));
        assert!(matches!(
            classify_process_result(false, true, "Skipping: maybe\nMISMATCH at index 3"),
            RunKind::Mismatch
        ));
        assert!(matches!(
            classify_process_result(false, true, "Skipping: needs sm_90"),
            RunKind::Skipped
        ));
        assert!(matches!(
            classify_process_result(false, true, "PASS"),
            RunKind::Pass
        ));
    }

    #[test]
    fn seed_ranges_are_half_open() {
        assert_eq!(parse_seed_range("3..8").unwrap(), (3, 8));
        assert!(parse_seed_range("8..8").is_err());
        assert!(parse_seed_range("8").is_err());
    }

    /// The baseline classes, pinned exhaustively. A decline must not be
    /// lumped in with a failure: an arch-gated example on a device below its
    /// floor is not a broken example, and #665 is the precedent for keeping
    /// those apart.
    #[test]
    fn only_a_declined_baseline_sits_between_usable_and_broken() {
        assert_eq!(RunKind::Pass.baseline_verdict(), BaselineVerdict::Usable);
        assert_eq!(
            RunKind::Skipped.baseline_verdict(),
            BaselineVerdict::Declined
        );
        for broken in [
            RunKind::Hang,
            RunKind::Crash,
            RunKind::Mismatch,
            RunKind::OutputChanged,
            RunKind::GpuWedged,
            RunKind::HarnessError,
        ] {
            assert_eq!(
                broken.baseline_verdict(),
                BaselineVerdict::Broken,
                "{broken:?}"
            );
        }
    }

    /// A finding is about a *variant*; as a baseline the same outcome means the
    /// campaign has no ground truth to compare against, so it is broken, not a
    /// finding to report.
    #[test]
    fn a_finding_as_the_baseline_is_a_broken_baseline() {
        for kind in [
            RunKind::Hang,
            RunKind::Crash,
            RunKind::Mismatch,
            RunKind::OutputChanged,
            RunKind::GpuWedged,
        ] {
            assert!(kind.is_finding(), "{kind:?}");
            assert_eq!(kind.baseline_verdict(), BaselineVerdict::Broken, "{kind:?}");
        }
    }

    #[test]
    fn findings_use_schedule_sensitive_labels() {
        assert_eq!(
            RunKind::Mismatch.finding_label(),
            Some("SCHEDULE-SENSITIVE CORRECTNESS FAILURE")
        );
        assert_eq!(RunKind::Hang.finding_label(), Some("TIMEOUT CANDIDATE"));
        assert!(!RunKind::Pass.is_finding());
    }

    #[test]
    fn output_comparison_is_opt_in() {
        let baseline = RunResult {
            kind: RunKind::Pass,
            exit_code: Some(0),
            timed_out: false,
            stdout: "ok\n".to_string(),
            stderr: String::new(),
        };
        let mut unchanged = baseline.clone();
        classify_output_change(&baseline, &mut unchanged, false);
        assert!(matches!(unchanged.kind, RunKind::Pass));

        let mut changed = RunResult {
            stdout: "different\n".to_string(),
            ..baseline.clone()
        };
        classify_output_change(&baseline, &mut changed, true);
        assert!(matches!(changed.kind, RunKind::OutputChanged));
    }

    #[test]
    fn shell_quotes_replay_values() {
        assert_eq!(shell_quote_value("a'b"), "'a'\\''b'");
    }

    #[cfg(unix)]
    #[test]
    fn run_binary_drains_output_larger_than_a_pipe_buffer() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("ptx-schedule-drain-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("spam.sh");
        // 1 MiB of stdout, far beyond the pipe buffer the watchdog loop used
        // to deadlock against before the reader threads were added.
        fs::write(
            &script,
            "#!/bin/sh\ndd if=/dev/zero bs=65536 count=16 2>/dev/null | tr '\\0' 'a'\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let result = run_binary(&script, &dir, Duration::from_secs(30));
        fs::remove_dir_all(&dir).ok();
        assert!(matches!(result.kind, RunKind::Pass), "{:?}", result.kind);
        assert!(!result.timed_out);
        assert_eq!(result.stdout.len(), 16 * 65536);
    }

    #[test]
    fn harness_errors_are_recorded_without_becoming_findings() {
        let error = CampaignError::ArtifactPatch("objcopy failed".into());
        let result = harness_error_result(7, 1.5, PathBuf::from("seed-7"), &error);
        assert!(matches!(result.run.kind, RunKind::HarnessError));
        assert!(!result.run.kind.is_finding());
        assert!(result.run.stderr.contains("objcopy failed"));
        assert_eq!(result.seed, 7);
        assert_eq!(result.report.seed, 7);
        assert_eq!(result.report.sites_injected, 0);
    }
}
