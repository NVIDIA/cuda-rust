/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-target `extra-rustflags` from `[target.<triple>]` tables.
//!
//! `.cargo/cuda-oxide.toml` can hold one `extra-rustflags` list for every
//! build and one more per target triple:
//!
//! ```toml
//! extra-rustflags = ["--cfg", "my_device_cfg"]   # every triple
//!
//! [target.x86_64-unknown-linux-gnu]
//! extra-rustflags = ["-C", "target-cpu=znver3"]
//! ```
//!
//! A build gets the top-level list, then the list of the triple it builds
//! for. The triple comes from the Cargo command cargo-oxide is about to run:
//!
//! ```text
//! --target <triple> in the Cargo arguments   (before any `--`)
//!         │ none
//!         ▼
//! CARGO_BUILD_TARGET   (the command's own value, else the inherited one)
//!         │ unset
//!         ▼
//! `host:` line of `rustc -vV`   (RUSTC, else CARGO_BUILD_RUSTC, else rustc,
//!                                in the command's directory and environment)
//! ```
//!
//! `host-tuple` in the first two steps means the host triple of the third.
//! A config without target tables never runs the `rustc -vV` probe.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Command;

use super::*;

/// The `--target` / `CARGO_BUILD_TARGET` spelling Cargo accepts for "the
/// host's own triple". It is resolved through the same `rustc -vV` probe as
/// the host fallback, so `--target host-tuple` and the host's name agree.
pub(super) const HOST_TUPLE: &str = "host-tuple";

/// The only field a `[target.<triple>]` table may hold. Cargo's own target
/// fields (`linker`, `runner`, `rustflags`, ...) belong in
/// `.cargo/config.toml`, and silently accepting them here would hide typos.
const TARGET_TABLE_FIELD: &str = "extra-rustflags";

/// Parse the `target` value of `.cargo/cuda-oxide.toml` into a map from
/// triple to its `extra-rustflags` list.
///
/// Every problem is collected, not only the first, so `doctor` can list them
/// all at once. Errors name the config file and the offending table.
pub(super) fn parse_config_targets(
    value: &toml::Value,
    config_path: &Path,
) -> Result<BTreeMap<String, Vec<String>>, Vec<String>> {
    let path = config_path.display();
    let Some(table) = value.as_table() else {
        return Err(vec![format!(
            "cuda-oxide config {path} field `target` must be a table of \
             `[target.<triple>]` tables"
        )]);
    };

    let mut targets = BTreeMap::new();
    let mut errors = Vec::new();
    for (key, entry) in table {
        if let Err(error) = check_target_key(key) {
            errors.push(format!(
                "cuda-oxide config {path} table `[target.{}]`: {error}",
                quoted_toml_key(key)
            ));
            continue;
        }
        let Some(entry) = entry.as_table() else {
            errors.push(format!(
                "cuda-oxide config {path} `[target.{key}]` must be a table"
            ));
            continue;
        };
        let mut entry_ok = true;
        for field in entry.keys().filter(|field| *field != TARGET_TABLE_FIELD) {
            let hint = if field == "rustflags" {
                "; did you mean `extra-rustflags`?"
            } else {
                ""
            };
            errors.push(format!(
                "cuda-oxide config {path} `[target.{key}]` field `{field}` is not \
                 supported; only `{TARGET_TABLE_FIELD}` is{hint}"
            ));
            entry_ok = false;
        }
        let flags = match entry.get(TARGET_TABLE_FIELD) {
            None => Vec::new(),
            Some(flags) => match string_array(flags) {
                Some(flags) => flags,
                None => {
                    errors.push(format!(
                        "cuda-oxide config {path} field `target.{key}.{TARGET_TABLE_FIELD}` \
                         must be an array of strings"
                    ));
                    entry_ok = false;
                    Vec::new()
                }
            },
        };
        if entry_ok {
            targets.insert(key.clone(), flags);
        }
    }

    if errors.is_empty() {
        Ok(targets)
    } else {
        Err(errors)
    }
}

/// Reject `[target.<key>]` names that can never match a build triple.
///
/// `cfg(...)` predicates are a Cargo feature this table does not implement,
/// `host-tuple` is a command-line spelling rather than a triple, and a triple
/// is one word of ASCII letters, digits, `-`, `_` and `.`.
fn check_target_key(key: &str) -> Result<(), String> {
    if key.starts_with("cfg(") {
        return Err(
            "`cfg(...)` keys are not supported; name a target triple such as \
             `x86_64-unknown-linux-gnu`"
                .to_string(),
        );
    }
    if key == HOST_TUPLE {
        return Err(format!(
            "`{HOST_TUPLE}` is not a triple; name the host's triple (the `host:` line \
             of `rustc -vV`)"
        ));
    }
    let well_formed = !key.is_empty()
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    if !well_formed {
        return Err(format!(
            "`{key}` is not a target triple; use letters, digits, `-`, `_` and `.` only"
        ));
    }
    Ok(())
}

/// A table key as the user would have to write it in TOML: bare when it is
/// a plain word, quoted otherwise (`'cfg(unix)'`). Used only so error
/// messages point at the exact header in the file.
fn quoted_toml_key(key: &str) -> String {
    let bare = !key.is_empty()
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    if bare {
        key.to_string()
    } else {
        format!("'{key}'")
    }
}

/// The value as a list of strings, or `None` when it is not an array or one
/// element is not a string. Each element stays one rustc argument, spaces
/// included, exactly like the top-level `extra-rustflags`.
fn string_array(value: &toml::Value) -> Option<Vec<String>> {
    value
        .as_array()?
        .iter()
        .map(|item| item.as_str().map(str::to_string))
        .collect()
}

/// The project rustflags a Cargo command gets: the top-level
/// `extra-rustflags`, then the `[target.<triple>]` list for the triple the
/// command builds for.
///
/// Reads the process environment and may run `rustc -vV`; the decision lives
/// in [`select_extra_rustflags`] so tests can inject both.
pub(super) fn configured_rustflags_for_command(
    config: &OxideConfig,
    cmd: &Command,
) -> Result<Vec<String>, String> {
    select_extra_rustflags(config, cmd, |key| std::env::var_os(key), run_host_probe)
}

/// [`configured_rustflags_for_command`] with the inherited environment and the
/// host probe injected.
///
/// `inherited` reads a variable cargo-oxide itself was started with.
/// `probe_host` receives the ready-to-run `rustc -vV` command from
/// [`host_probe_command`] and returns the host triple. It is called at most
/// once, and never when the config has no target tables.
pub(super) fn select_extra_rustflags(
    config: &OxideConfig,
    cmd: &Command,
    inherited: impl Fn(&str) -> Option<OsString>,
    probe_host: impl FnOnce(Command) -> Result<String, String>,
) -> Result<Vec<String>, String> {
    let mut flags = config.extra_rustflags.clone();
    if config.target_rustflags.is_empty() {
        return Ok(flags);
    }
    let triple = resolve_build_triple(cmd, &inherited, probe_host)?;
    if let Some(target_flags) = config.target_rustflags.get(&triple) {
        flags.extend(target_flags.iter().cloned());
    }
    Ok(flags)
}

/// The one triple `cmd` builds for, as a `[target.<triple>]` key.
///
/// `--target` arguments win over `CARGO_BUILD_TARGET`, which wins over the
/// host. Several `--target` values that name the same triple are one target;
/// different ones are an error, because `CARGO_ENCODED_RUSTFLAGS` reaches
/// every target of the command and cannot carry two tables at once.
pub(super) fn resolve_build_triple(
    cmd: &Command,
    inherited: &impl Fn(&str) -> Option<OsString>,
    probe_host: impl FnOnce(Command) -> Result<String, String>,
) -> Result<String, String> {
    let mut requested = cargo_target_args(cmd)?;
    if requested.is_empty()
        && let Some(value) = command_env(cmd, "CARGO_BUILD_TARGET", inherited)
    {
        let value = value
            .into_string()
            .map_err(|value| format!("CARGO_BUILD_TARGET is not valid UTF-8: {value:?}"))?;
        if !value.trim().is_empty() {
            requested.push(value.trim().to_string());
        }
    }

    // The host probe runs at most once, and only when it is needed: for the
    // fallback, or to name `host-tuple`.
    let mut probe_host = Some(probe_host);
    let mut host: Option<String> = None;
    let mut host_triple = || -> Result<String, String> {
        if let Some(host) = &host {
            return Ok(host.clone());
        }
        let probe = probe_host.take().expect("the host probe runs once");
        let triple = probe(host_probe_command(cmd, inherited)).map_err(|error| {
            format!(
                "`.cargo/cuda-oxide.toml` has `[target.<triple>]` tables, but the \
                 build's target triple is unknown: {error}. Pass `--target <triple>` \
                 to Cargo or set CARGO_BUILD_TARGET."
            )
        })?;
        host = Some(triple.clone());
        Ok(triple)
    };

    if requested.is_empty() {
        return host_triple();
    }

    let mut triples: Vec<String> = Vec::new();
    for target in requested {
        let triple = if target == HOST_TUPLE {
            host_triple()?
        } else {
            target_table_key(&target)
        };
        if !triples.contains(&triple) {
            triples.push(triple);
        }
    }
    if let [triple] = triples.as_slice() {
        return Ok(triple.clone());
    }
    Err(format!(
        "this command builds for more than one target ({}), but \
         `.cargo/cuda-oxide.toml` has `[target.<triple>]` tables and \
         CARGO_ENCODED_RUSTFLAGS reaches every target of one command. Run one \
         command per target.",
        triples
            .iter()
            .map(|triple| format!("`{triple}`"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The `--target` values among `cmd`'s arguments, in order.
///
/// Both `--target <value>` and `--target=<value>` count; `--target-dir` does
/// not. A `--` ends the search: what follows belongs to the test harness or
/// the program being run.
fn cargo_target_args(cmd: &Command) -> Result<Vec<String>, String> {
    let mut args = cmd.get_args().take_while(|arg| *arg != OsStr::new("--"));
    let mut targets = Vec::new();
    while let Some(arg) = args.next() {
        let value = if arg == OsStr::new("--target") {
            match args.next() {
                Some(value) => value,
                // Cargo itself reports the missing value.
                None => break,
            }
        } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--target=")) {
            OsStr::new(value)
        } else {
            continue;
        };
        let value = value
            .to_str()
            .ok_or_else(|| format!("a `--target` value is not valid UTF-8: {value:?}"))?;
        targets.push(value.to_string());
    }
    Ok(targets)
}

/// The `[target.<key>]` name of a `--target` value.
///
/// A path to a custom target specification (`specs/my-gpu-host.json`) is
/// looked up by its file stem, as Cargo does for its own `[target]` tables;
/// any other value is already the triple.
fn target_table_key(target: &str) -> String {
    let path = Path::new(target);
    if path.extension() == Some(OsStr::new("json"))
        && let Some(stem) = path.file_stem().and_then(OsStr::to_str)
    {
        return stem.to_string();
    }
    target.to_string()
}

/// `key` as `cmd` will see it: the command's own setting when it has one
/// (`None` when the command removes the variable), else the inherited value.
fn command_env(
    cmd: &Command,
    key: &str,
    inherited: &impl Fn(&str) -> Option<OsString>,
) -> Option<OsString> {
    match cmd.get_envs().find(|(name, _)| *name == OsStr::new(key)) {
        Some((_, value)) => value.map(OsStr::to_os_string),
        None => inherited(key),
    }
}

/// The `rustc -vV` command that finds the host triple for `cmd`.
///
/// It runs the compiler Cargo would run (`RUSTC`, else `CARGO_BUILD_RUSTC`,
/// else `rustc`; empty values count as unset), in `cmd`'s directory and with
/// `cmd`'s environment changes, so rustup picks the same toolchain. Cargo's
/// `[build] rustc` config key and `RUSTC_WRAPPER` are not consulted.
pub(super) fn host_probe_command(
    cmd: &Command,
    inherited: &impl Fn(&str) -> Option<OsString>,
) -> Command {
    let compiler = ["RUSTC", "CARGO_BUILD_RUSTC"]
        .into_iter()
        .filter_map(|key| command_env(cmd, key, inherited))
        .find(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from("rustc"));
    let mut probe = Command::new(compiler);
    probe.arg("-vV");
    if let Some(dir) = cmd.get_current_dir() {
        probe.current_dir(dir);
    }
    for (key, value) in cmd.get_envs() {
        match value {
            Some(value) => probe.env(key, value),
            None => probe.env_remove(key),
        };
    }
    probe
}

/// Run a [`host_probe_command`] and read its `host:` line.
///
/// Fails with the reason when the compiler cannot be started, exits
/// unsuccessfully, or prints no `host:` line.
fn run_host_probe(mut probe: Command) -> Result<String, String> {
    let program = probe.get_program().to_string_lossy().into_owned();
    let output = probe
        .output()
        .map_err(|error| format!("could not run `{program} -vV`: {error}"))?;
    if !output.status.success() {
        return Err(format!("`{program} -vV` failed with {}", output.status));
    }
    parse_rustc_host(&String::from_utf8_lossy(&output.stdout))
        .ok_or_else(|| format!("`{program} -vV` printed no `host:` line"))
}

/// The triple on the `host: <triple>` line of `rustc -vV` output, if any.
/// Kept separate from [`run_host_probe`] so the parsing is testable without
/// a compiler.
pub(super) fn parse_rustc_host(output: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix("host:"))
        .map(str::trim)
        .filter(|triple| !triple.is_empty())
        .map(str::to_string)
}
