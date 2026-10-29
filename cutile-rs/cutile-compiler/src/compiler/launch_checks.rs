/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retain check origins while deduplicating canonical launch predicates.
//! Export them through the existing host diagnostic string, keeping source
//! metadata and Tile IR types out of the shared host API.

use cuda_async::predicate::LaunchCheck;
use cutile_ir::ir::{DebugScope, Location};
use std::fmt::{self, Write};

#[derive(Default)]
pub(super) struct LaunchChecks {
    checks: Vec<CheckWithOrigins>,
}

struct CheckWithOrigins {
    check: LaunchCheck,
    origins: Vec<CheckOrigin>,
}

#[derive(PartialEq, Eq)]
struct CheckOrigin {
    // Innermost frame first. Retain only the source identity, not complete
    // subprogram/compilation-unit metadata for every protected access.
    frames: Vec<CheckFrame>,
}

#[derive(PartialEq, Eq)]
struct CheckFrame {
    file: String,
    line: u32,
    column: u32,
    function: Option<String>,
}

impl LaunchChecks {
    pub(super) fn insert(&mut self, check: LaunchCheck, location: &Location) {
        // Preserve the original cause and one comparison per canonical
        // predicate, while retaining every distinct access/inline chain.
        let existing = match self
            .checks
            .iter_mut()
            .find(|entry| entry.check.predicate == check.predicate)
        {
            Some(existing) => existing,
            None => {
                self.checks.push(CheckWithOrigins {
                    check,
                    origins: Vec::new(),
                });
                self.checks.last_mut().unwrap()
            }
        };
        let mut origin = CheckOrigin { frames: Vec::new() };
        origin.collect(location);
        if !origin.frames.is_empty() && !existing.origins.contains(&origin) {
            existing.origins.push(origin);
        }
    }

    pub(super) fn to_host_checks(&self) -> Vec<LaunchCheck> {
        self.checks
            .iter()
            .map(|entry| {
                let mut check = entry.check.clone();
                for origin in &entry.origins {
                    write!(&mut check.cause, "{origin}").expect("writing to a String cannot fail");
                }
                check
            })
            .collect()
    }
}

impl CheckOrigin {
    fn collect(&mut self, location: &Location) {
        match location {
            Location::CallSite { callee, caller } => {
                self.collect(callee);
                self.collect(caller);
            }
            Location::FileLineCol {
                filename,
                line,
                column,
            } => self.frames.push(CheckFrame {
                file: filename.clone(),
                line: *line,
                column: *column,
                function: None,
            }),
            Location::DebugInfo(info) => self.frames.push(CheckFrame {
                file: info.filename.clone(),
                line: info.line,
                column: info.column,
                function: match &info.scope {
                    DebugScope::Subprogram(scope) => Some(scope.name.clone()),
                    _ => None,
                },
            }),
            Location::Unknown => {}
        }
    }
}

impl fmt::Display for CheckOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, frame) in self.frames.iter().enumerate() {
            let relation = if index == 0 { "at" } else { "inlined at" };
            write!(
                f,
                "\n  {relation} {}:{}:{}",
                frame.file,
                frame.line,
                u64::from(frame.column) + 1
            )?;
            if let Some(function) = &frame.function {
                write!(f, " ({function})")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::predicate::{Atom, Predicate, Term};

    fn check(param: usize, cause: &str) -> LaunchCheck {
        // The existing two-field public struct literal must remain usable
        // from a downstream crate.
        LaunchCheck {
            predicate: Predicate::Nonzero(Term::atom(Atom::Dim { param, axis: 0 })),
            cause: cause.into(),
        }
    }

    fn location(file: &str, line: u32, column: u32) -> Location {
        Location::FileLineCol {
            filename: file.into(),
            line,
            column,
        }
    }

    #[test]
    fn deduplication_retains_distinct_callers_and_export_is_repeatable() {
        let mut checks = LaunchChecks::default();
        for caller_line in [20, 20, 30] {
            checks.insert(
                check(0, "non-empty extent"),
                &Location::CallSite {
                    callee: Box::new(location("helper.rs", 10, 2)),
                    caller: Box::new(location("kernel.rs", caller_line, 4)),
                },
            );
        }
        let exported = checks.to_host_checks();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].predicate, check(0, "").predicate);
        assert_eq!(
            exported[0].cause,
            "non-empty extent\n  at helper.rs:10:3\n  inlined at kernel.rs:20:5\n  at helper.rs:10:3\n  inlined at kernel.rs:30:5"
        );
        assert_eq!(checks.to_host_checks(), exported);
    }

    #[test]
    fn unknown_origins_preserve_causes_and_distinct_predicates() {
        let mut checks = LaunchChecks::default();
        checks.insert(check(0, "first extent"), &Location::Unknown);
        checks.insert(check(1, "second extent"), &Location::Unknown);
        assert_eq!(
            checks.to_host_checks(),
            vec![check(0, "first extent"), check(1, "second extent")]
        );
        checks.insert(check(0, "duplicate cause"), &location("kernel.rs", 9, 0));
        let exported = checks.to_host_checks();
        assert_eq!(exported[0].cause, "first extent\n  at kernel.rs:9:1");
        assert_eq!(exported[1], check(1, "second extent"));
    }
}
