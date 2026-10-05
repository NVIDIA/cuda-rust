/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use clap::{ArgGroup, Parser};
use ptx_schedule::{InjectionOptions, analyze_ptx, perturb_ptx};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "ptx-schedule",
    about = "Inspect or perturb one PTX file",
    override_usage = "ptx-schedule <INPUT.ptx> --list-sites\n       ptx-schedule <INPUT.ptx> -o <OUTPUT> [OPTIONS]",
    group(ArgGroup::new("mode").required(true).multiple(false).args(["list_sites", "output"]))
)]
struct Cli {
    /// PTX module to inspect or rewrite.
    #[arg(value_name = "INPUT.ptx")]
    input: PathBuf,

    /// Print schedule-sensitive sites as JSON.
    #[arg(long, group = "mode")]
    list_sites: bool,

    /// Write a perturbed PTX module.
    #[arg(short = 'o', long, value_name = "OUTPUT", group = "mode")]
    output: Option<PathBuf>,

    /// Deterministic perturbation seed.
    #[arg(
        long,
        value_name = "N",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    seed: Option<u64>,

    /// Fraction of eligible sites to perturb.
    #[arg(
        long,
        value_name = "F",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    intensity: Option<f64>,

    /// Maximum sleep inserted at one site.
    #[arg(
        long,
        value_name = "N",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    max_sleep_ns: Option<u32>,

    /// Prefer sites whose instruction contains this text.
    #[arg(
        long,
        value_name = "TEXT",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    focus: Option<String>,

    /// Save the injection decisions as JSON.
    #[arg(
        long,
        value_name = "FILE",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    decisions_json: Option<PathBuf>,
}

/// Canonical path of a destination that may not exist yet. The existing
/// parent is canonicalized so `./` and `../` aliases compare equal.
fn canonical_destination(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if let Ok(canonical) = fs::canonicalize(&absolute) {
        return Ok(canonical);
    }
    let file_name = absolute
        .file_name()
        .ok_or_else(|| format!("--decisions-json path has no file name: {}", path.display()))?;
    let parent = absolute
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(fs::canonicalize(parent)?.join(file_name))
}

fn reject_decisions_clobber(
    input: &Path,
    output: &Path,
    decisions: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let decisions_path = canonical_destination(decisions)?;
    let same_as_input = decisions_path == canonical_destination(input)?;
    let same_as_output = decisions_path == canonical_destination(output)?;
    if same_as_input || same_as_output {
        let which = if same_as_input && same_as_output {
            "the PTX input and --output"
        } else if same_as_input {
            "the PTX input"
        } else {
            "--output"
        };
        return Err(format!(
            "--decisions-json resolves to {}, which is {which}; refusing to overwrite PTX",
            decisions_path.display()
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let source = fs::read_to_string(&cli.input)?;
    if cli.list_sites {
        let analysis = analyze_ptx(&source)?;
        println!("{}", serde_json::to_string_pretty(analysis.sites())?);
        return Ok(());
    }

    let mut options = InjectionOptions::default();
    if let Some(seed) = cli.seed {
        options.seed = seed;
    }
    if let Some(intensity) = cli.intensity {
        options.intensity = intensity;
    }
    if let Some(max_sleep_ns) = cli.max_sleep_ns {
        options.max_sleep_ns = max_sleep_ns;
    }
    if let Some(focus) = cli.focus {
        options.focus = Some(focus);
    }

    let output = cli
        .output
        .expect("clap requires either --list-sites or --output");
    if let Some(decisions) = &cli.decisions_json {
        reject_decisions_clobber(&cli.input, &output, decisions)?;
    }
    let rewrite = perturb_ptx(&source, &options)?;
    fs::write(&output, &rewrite.ptx)?;
    if let Some(path) = &cli.decisions_json {
        fs::write(path, serde_json::to_string_pretty(&rewrite.report)?)?;
    }
    println!(
        "ptx-schedule: seed={} intensity={} sites={} injected={} ns_per_visit={}",
        rewrite.report.seed,
        rewrite.report.intensity,
        rewrite.report.sites_total,
        rewrite.report.sites_injected,
        rewrite.report.injected_ns_per_visit
    );
    Ok(())
}
