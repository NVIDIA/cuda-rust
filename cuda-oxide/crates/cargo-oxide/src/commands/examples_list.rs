/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::Path;

use super::*;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct ExampleInfo {
    pub(super) name: String,
    pub(super) title: String,
    pub(super) description: String,
    pub(super) requirements: Vec<String>,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct ParsedReadme {
    pub(super) title: Option<String>,
    pub(super) description: Option<String>,
    pub(super) requirements: Vec<String>,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct ManifestInfo {
    description: Option<String>,
}

pub fn list_examples(ctx: &Context, json: bool) {
    if !ctx.is_workspace {
        eprintln!("Error: `cargo oxide list` must be run from inside a cuda-oxide checkout.");
        eprintln!();
        eprintln!("The command lists examples under crates/rustc-codegen-cuda/examples/.");
        std::process::exit(1);
    }

    let examples = discover_examples(&ctx.examples_dir).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let output = if json {
        format_examples_json(&examples).unwrap_or_else(|error| {
            eprintln!("Error: could not serialize example list: {error}");
            std::process::exit(1);
        })
    } else {
        format_examples_human(&examples)
    };

    print!("{output}");
}

pub(super) fn discover_examples(examples_dir: &Path) -> Result<Vec<ExampleInfo>, String> {
    let entries = fs::read_dir(examples_dir).map_err(|error| {
        format!(
            "could not read examples directory {}: {error}",
            examples_dir.display()
        )
    })?;

    let mut examples = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "could not read an entry under {}: {error}",
                examples_dir.display()
            )
        })?;

        let file_type = entry
            .file_type()
            .map_err(|error| format!("could not inspect {}: {error}", entry.path().display()))?;

        if !file_type.is_dir() {
            continue;
        }

        let example_dir = entry.path();
        let name = entry.file_name().into_string().map_err(|name| {
            format!(
                "example directory name is not valid UTF-8: {}",
                name.to_string_lossy()
            )
        })?;

        // A directory without a manifest is not an example (scratch dirs,
        // checked-out tooling, ...). Skip it instead of failing the listing.
        let manifest_path = example_dir.join("Cargo.toml");
        if !manifest_path.is_file() {
            eprintln!(
                "Warning: skipping {}: no top-level Cargo.toml",
                example_dir.display()
            );
            continue;
        }

        let manifest = parse_example_manifest(&manifest_path)?;

        let readme_path = example_dir.join("README.md");
        let parsed_readme = if readme_path.is_file() {
            let contents = fs::read_to_string(&readme_path)
                .map_err(|error| format!("could not read {}: {error}", readme_path.display()))?;
            parse_example_readme(&name, &contents)
        } else {
            ParsedReadme::default()
        };

        let ParsedReadme {
            title,
            description,
            requirements,
        } = parsed_readme;

        let title = title.unwrap_or_else(|| name.clone());

        let description = description.or(manifest.description).unwrap_or_else(|| {
            if title != name {
                title.clone()
            } else {
                "No description documented.".to_string()
            }
        });

        examples.push(ExampleInfo {
            name,
            title,
            description,
            requirements,
        });
    }

    examples.sort_by(|left, right| left.name.cmp(&right.name));

    if examples.is_empty() {
        return Err(format!(
            "no examples found under {}",
            examples_dir.display()
        ));
    }

    Ok(examples)
}

fn parse_example_manifest(path: &Path) -> Result<ManifestInfo, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;

    let manifest: toml::Value = toml::from_str(&contents)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;

    let package = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("{} has no [package] table", path.display()))?;

    let description = package
        .get("description")
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
        .map(str::to_owned);

    Ok(ManifestInfo { description })
}

pub(super) fn parse_example_readme(crate_name: &str, contents: &str) -> ParsedReadme {
    let lines: Vec<&str> = contents.lines().collect();
    let mut headings = Vec::new();
    let mut in_code_fence = false;

    for (index, line) in lines.iter().enumerate() {
        if is_code_fence(line) {
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        if let Some((level, title)) = parse_markdown_heading(line) {
            headings.push((index, level, title));
        }
    }

    let crate_heading = normalize_heading(crate_name);

    let selected_heading = match headings.first() {
        Some(first) if normalize_heading(&first.2) == crate_heading => headings
            .get(1)
            .filter(|heading| {
                heading.1 == 2
                    && first_prose_paragraph_in_range(&lines, first.0 + 1, heading.0).is_none()
                    && !is_generic_section_heading(&normalize_heading(&heading.2))
            })
            .or(Some(first)),
        Some(first) => Some(first),
        None => None,
    };

    let title = selected_heading
        .map(|(_, _, title)| strip_inline_markdown(title))
        .filter(|title| !title.is_empty());

    let description_start = selected_heading.map(|(index, _, _)| index + 1).unwrap_or(0);

    let description_end = headings
        .iter()
        .find(|(index, _, _)| *index >= description_start)
        .map(|(index, _, _)| *index)
        .unwrap_or(lines.len());

    let description = first_prose_paragraph_in_range(&lines, description_start, description_end);
    let requirements = extract_requirements(&lines);

    ParsedReadme {
        title,
        description,
        requirements,
    }
}

fn parse_markdown_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.bytes().take_while(|byte| *byte == b'#').count();

    if !(1..=6).contains(&level) {
        return None;
    }

    let remainder = &trimmed[level..];
    if !remainder.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }

    let title = remainder.trim().trim_end_matches('#').trim();

    if title.is_empty() {
        None
    } else {
        Some((level, title.to_string()))
    }
}

fn is_code_fence(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn strip_inline_markdown(value: &str) -> String {
    value
        .replace("**", "")
        .replace("__", "")
        .replace('`', "")
        .trim()
        .to_string()
}

fn normalize_heading(value: &str) -> String {
    strip_inline_markdown(value)
        .trim_matches(|character: char| {
            character == ':' || character == '-' || character.is_whitespace()
        })
        .to_ascii_lowercase()
}

fn is_generic_section_heading(heading: &str) -> bool {
    matches!(
        heading,
        "overview"
            | "what this example does"
            | "key concepts"
            | "key concepts demonstrated"
            | "build"
            | "build and run"
            | "usage"
            | "expected output"
            | "requirements"
            | "hardware requirements"
            | "prerequisites"
            | "potential errors"
            | "how it works"
            | "how it works under the hood"
            | "generated ptx"
            | "run"
            | "test"
            | "tests"
            | "correctness"
            | "trigger"
            | "kernels"
            | "features tested"
            | "what this tests"
            | "what it tests"
            | "what this demonstrates"
            | "why this exists"
            | "the bug"
            | "final design"
    )
}

fn first_prose_paragraph_in_range(lines: &[&str], start: usize, end: usize) -> Option<String> {
    let end = end.min(lines.len());
    let start = start.min(end);
    let mut paragraph = Vec::new();
    let mut in_code_fence = false;

    for line in &lines[start..end] {
        if is_code_fence(line) {
            if !paragraph.is_empty() {
                break;
            }
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        let trimmed = line.trim();

        if parse_markdown_heading(trimmed).is_some() {
            break;
        }

        if trimmed.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }

        if is_non_prose_markdown(trimmed) {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }

        paragraph.push(trimmed);
    }

    if paragraph.is_empty() {
        None
    } else {
        Some(paragraph.join(" "))
    }
}

fn is_non_prose_markdown(line: &str) -> bool {
    line.starts_with("- ")
        || line.starts_with("* ")
        || line.starts_with("+ ")
        || line.starts_with('>')
        || line.starts_with('|')
        || line.starts_with("![")
        || line.starts_with("<!--")
        || is_ordered_list_item(line)
}

fn is_ordered_list_item(line: &str) -> bool {
    strip_ordered_list_marker(line).is_some()
}

/// Strip a `1. ` / `42. ` ordered-list marker, returning the item text.
fn strip_ordered_list_marker(line: &str) -> Option<&str> {
    let (marker, item) = line.split_once(". ")?;
    if !marker.is_empty() && marker.bytes().all(|byte| byte.is_ascii_digit()) {
        Some(item.trim_start())
    } else {
        None
    }
}

/// Collect the requirement entries documented under a requirements-style
/// heading ([`is_requirements_heading`]).
///
/// Recognized forms:
/// - unordered list items (`- ` / `* ` / `+ `), with indented
///   wrap-continuation lines joined onto the item;
/// - ordered list items (`1. `), same continuation rule;
/// - two-column markdown tables, emitted as `name: value` per data row.
///
/// Tables with any other column count are skipped whole: without knowing
/// which columns carry the requirement, half-parsing them would produce
/// garbage entries.
fn extract_requirements(lines: &[&str]) -> Vec<String> {
    let mut requirements = Vec::new();
    let mut current_requirement: Option<String> = None;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut requirement_level = None;
    let mut in_code_fence = false;

    for line in lines {
        if is_code_fence(line) {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        if let Some((level, heading)) = parse_markdown_heading(line) {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);

            let normalized = normalize_heading(&heading);

            if is_requirements_heading(&normalized) {
                requirement_level = Some(level);
            } else if requirement_level.is_some_and(|active| level <= active) {
                requirement_level = None;
            }

            continue;
        }

        if requirement_level.is_none() {
            continue;
        }

        let trimmed = line.trim();

        // A blank line terminates the current list item or table. Whatever
        // follows is a new paragraph (prose, a code fence, ...), not a
        // wrapped continuation of the bullet above it.
        if trimmed.is_empty() {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);
            continue;
        }

        if trimmed.starts_with('|') {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            table_rows.push(split_table_row(trimmed));
            continue;
        }
        flush_requirement_table(&mut table_rows, &mut requirements);

        let item = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("+ "))
            .or_else(|| strip_ordered_list_marker(trimmed));

        if let Some(item) = item {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }

            let item = strip_inline_markdown(item);
            if !item.is_empty() {
                current_requirement = Some(item);
            }
        } else if let Some(requirement) = &mut current_requirement {
            requirement.push(' ');
            requirement.push_str(&strip_inline_markdown(trimmed));
        }
    }

    if let Some(requirement) = current_requirement {
        requirements.push(requirement);
    }
    flush_requirement_table(&mut table_rows, &mut requirements);

    requirements.dedup();
    requirements
}

/// Split a markdown table row into trimmed cells, honoring `\|` escapes and
/// dropping the empty leading/trailing cells produced by the outer pipes.
fn split_table_row(row: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut characters = row.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            '\\' if characters.peek() == Some(&'|') => {
                cell.push('|');
                characters.next();
            }
            '|' => {
                cells.push(cell.trim().to_string());
                cell.clear();
            }
            _ => cell.push(character),
        }
    }
    cells.push(cell.trim().to_string());

    if cells.first().is_some_and(|first| first.is_empty()) {
        cells.remove(0);
    }
    if cells.last().is_some_and(|last| last.is_empty()) {
        cells.pop();
    }

    cells
}

/// The `|---|:---:|` row separating a table header from its data rows.
fn is_table_separator_row(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells.iter().all(|cell| {
            !cell.is_empty()
                && cell
                    .chars()
                    .all(|character| character == '-' || character == ':')
        })
}

/// Convert a buffered `| name | value |` requirements table into one
/// `name: value` entry per data row. Tables whose header or data rows are
/// not exactly two columns are dropped whole rather than half-parsed.
fn flush_requirement_table(table_rows: &mut Vec<Vec<String>>, requirements: &mut Vec<String>) {
    let rows = std::mem::take(table_rows);

    // Header, separator, and at least one data row.
    if rows.len() < 3 || !is_table_separator_row(&rows[1]) {
        return;
    }

    if !rows.iter().all(|row| row.len() == 2) {
        return;
    }

    for row in &rows[2..] {
        let name = strip_inline_markdown(&row[0]);
        let value = strip_inline_markdown(&row[1]);
        if !name.is_empty() && !value.is_empty() {
            requirements.push(format!("{name}: {value}"));
        }
    }
}

fn is_requirements_heading(heading: &str) -> bool {
    matches!(
        heading,
        "requirements"
            | "hardware requirements"
            | "software requirements"
            | "system requirements"
            | "toolkit requirements"
            | "build requirements"
            | "prerequisites"
    )
}

fn format_examples_human(examples: &[ExampleInfo]) -> String {
    let mut output = String::new();

    for (index, example) in examples.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }

        output.push_str(&example.name);
        output.push('\n');

        if example.title != example.name {
            output.push_str("  ");
            output.push_str(&example.title);
            output.push('\n');
        }

        output.push_str("  ");
        output.push_str(&example.description);
        output.push('\n');

        if !example.requirements.is_empty() {
            output.push_str("  Requirements:\n");
            for requirement in &example.requirements {
                output.push_str("    - ");
                output.push_str(requirement);
                output.push('\n');
            }
        }
    }

    output
}

pub(super) fn format_examples_json(examples: &[ExampleInfo]) -> Result<String, serde_json::Error> {
    let examples = examples
        .iter()
        .map(|example| {
            serde_json::json!({
                "name": example.name,
                "title": example.title,
                "description": example.description,
                "requirements": example.requirements,
            })
        })
        .collect::<Vec<_>>();

    let document = serde_json::json!({
        "schema_version": 1,
        "examples": examples,
    });

    let mut output = serde_json::to_string_pretty(&document)?;
    output.push('\n');
    Ok(output)
}
