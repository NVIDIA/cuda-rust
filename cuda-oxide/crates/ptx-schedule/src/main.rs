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

    /// Save the injection decisions as JSON. Must differ from the input
    /// and --output; a collision is refused before anything is written.
    #[arg(
        long,
        value_name = "FILE",
        requires = "output",
        conflicts_with = "list_sites"
    )]
    decisions_json: Option<PathBuf>,
}

/// Longest chain of symbolic links followed for one destination. Linux
/// follows at most 40 links in one path lookup, so a chain the kernel accepts
/// is never refused here; the loop below runs one extra time to resolve the
/// final non-link component.
const MAX_SYMLINK_FOLLOWS: usize = 40;

/// Canonical form of a path that may not exist yet, so that two names for one
/// destination compare equal. An existing path is canonicalized. A dangling
/// symbolic link (a link whose target does not exist yet) is followed to its
/// target, because a write through the link lands there. Otherwise the
/// existing parent is canonicalized and the file name is appended, so `./`
/// and `../` forms compare equal. `label` names the flag in error messages.
/// Hard links keep distinct canonical paths; `same_file` covers them.
fn canonical_destination(path: &Path, label: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    for _ in 0..=MAX_SYMLINK_FOLLOWS {
        if let Ok(canonical) = fs::canonicalize(&absolute) {
            return Ok(canonical);
        }
        let (Some(file_name), Some(parent)) = (absolute.file_name(), absolute.parent()) else {
            return Err(format!("{label} path has no file name: {}", absolute.display()).into());
        };
        if !absolute.is_symlink() {
            let parent = fs::canonicalize(parent).map_err(|err| {
                format!("cannot resolve {label} path {}: {err}", absolute.display())
            })?;
            return Ok(parent.join(file_name));
        }
        let target = fs::read_link(&absolute)
            .map_err(|err| format!("cannot resolve {label} path {}: {err}", absolute.display()))?;
        absolute = if target.is_absolute() {
            target
        } else {
            parent.join(target)
        };
    }
    Err(format!(
        "{label} path has too many levels of symbolic links: {}",
        path.display()
    )
    .into())
}

/// Whether two existing paths name one file on disk. A hard link has its own
/// canonical path but shares the device and inode of the file it aliases, so
/// the path comparison alone misses it. A path that does not exist yet
/// answers false, and the path comparison still applies to it.
#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (fs::metadata(a), fs::metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

/// Only Unix exposes a file identity through `std`; other platforms keep the
/// canonical path comparison.
#[cfg(not(unix))]
fn same_file(_a: &Path, _b: &Path) -> bool {
    false
}

/// Refuses a `--decisions-json` destination that would replace the PTX input
/// or `--output`, or that is a directory, before anything is written.
fn reject_decisions_clobber(
    input: &Path,
    output: &Path,
    decisions: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let decisions_path = canonical_destination(decisions, "--decisions-json")?;
    if decisions_path.is_dir() {
        return Err(format!(
            "--decisions-json names a directory: {}",
            decisions_path.display()
        )
        .into());
    }
    let input_path = canonical_destination(input, "the PTX input")?;
    let output_path = canonical_destination(output, "--output")?;
    let same_as_input = decisions_path == input_path || same_file(&decisions_path, &input_path);
    let same_as_output = decisions_path == output_path || same_file(&decisions_path, &output_path);
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
