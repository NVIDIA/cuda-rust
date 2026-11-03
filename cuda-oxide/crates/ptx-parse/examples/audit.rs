/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use ptx_parse::Document;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut paths = Vec::new();
    for argument in std::env::args_os().skip(1) {
        collect(Path::new(&argument), &mut paths)?;
    }
    paths.sort();

    let mut failures = 0usize;
    let mut bytes = 0usize;
    let mut tokens = 0usize;
    let mut statements = 0usize;
    let mut scopes = 0usize;
    let mut instructions = 0usize;
    for path in &paths {
        let source = fs::read_to_string(path)?;
        bytes += source.len();
        match Document::parse(&source) {
            Ok(document) => {
                tokens += document.tokens().len();
                statements += document.statements().len();
                scopes += document.scopes().len();
                instructions += document.instructions().len();
                if !document.coverage().is_complete() {
                    failures += 1;
                    eprintln!("{}: coverage={:?}", path.display(), document.coverage());
                    for diagnostic in document.diagnostics().iter().take(20) {
                        let span = diagnostic.span();
                        let line = source[..span.start]
                            .bytes()
                            .filter(|byte| *byte == b'\n')
                            .count()
                            + 1;
                        let text = source[span]
                            .lines()
                            .next()
                            .unwrap_or_default()
                            .chars()
                            .take(120)
                            .collect::<String>();
                        eprintln!("  line {line}: {:?}: {text:?}", diagnostic.kind());
                    }
                }
            }
            Err(error) => {
                failures += 1;
                eprintln!("{}: {error}", path.display());
            }
        }
    }
    println!(
        "files={} bytes={} tokens={} statements={} scopes={} instructions={} incomplete={failures}",
        paths.len(),
        bytes,
        tokens,
        statements,
        scopes,
        instructions
    );
    if failures == 0 {
        Ok(())
    } else {
        Err("PTX structural coverage is incomplete".into())
    }
}

fn collect(path: &Path, paths: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            collect(&entry?.path(), paths)?;
        }
    } else if path.extension().is_some_and(|extension| extension == "ptx") {
        paths.push(path.to_owned());
    }
    Ok(())
}
