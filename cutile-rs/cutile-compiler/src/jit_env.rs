/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Process-wide `CUTILE_JIT_*` overrides of the code's [`CompileOptions`].
//!
//! They let a wrapper switch an already-built application into debug or
//! sanitizer compilation. Both compile stages read the overridden options, so
//! the frontend's check placement and the `tileiras` flags always agree.

use crate::error::JITError;
use crate::hints::{CompileOptions, DebugInfoLevel};
use std::env::{self, VarError};
use std::sync::OnceLock;

pub const JIT_DEBUG_ENV: &str = "CUTILE_JIT_DEBUG";
pub const JIT_OPT_LEVEL_ENV: &str = "CUTILE_JIT_OPT_LEVEL";
pub const JIT_SANITIZE_ENV: &str = "CUTILE_JIT_SANITIZE";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct JitEnvOverrides {
    debug_info: Option<DebugInfoLevel>,
    opt_level: Option<u8>,
    sanitize_memcheck: Option<bool>,
}

impl JitEnvOverrides {
    fn from_env() -> Result<Self, String> {
        Self::parse(
            env::var(JIT_DEBUG_ENV),
            env::var(JIT_OPT_LEVEL_ENV),
            env::var(JIT_SANITIZE_ENV),
        )
    }

    fn parse(
        debug_info: Result<String, VarError>,
        opt_level: Result<String, VarError>,
        sanitize: Result<String, VarError>,
    ) -> Result<Self, String> {
        let overrides = Self {
            debug_info: choice(
                JIT_DEBUG_ENV,
                debug_info,
                &[
                    ("none", DebugInfoLevel::None),
                    ("line", DebugInfoLevel::Line),
                    ("full", DebugInfoLevel::Full),
                ],
            )?,
            opt_level: choice(
                JIT_OPT_LEVEL_ENV,
                opt_level,
                &[("0", 0), ("1", 1), ("2", 2), ("3", 3)],
            )?,
            sanitize_memcheck: choice(
                JIT_SANITIZE_ENV,
                sanitize,
                &[("memcheck", true), ("none", false)],
            )?,
        };
        if overrides.debug_info == Some(DebugInfoLevel::Full) && overrides.opt_level.is_some() {
            return Err(format!(
                "{JIT_DEBUG_ENV}=full compiles at level 0; unset {JIT_OPT_LEVEL_ENV}"
            ));
        }
        Ok(overrides)
    }

    fn apply(self, options: &CompileOptions) -> CompileOptions {
        let mut options = options.clone();
        if let Some(level) = self.debug_info {
            options = options.debug_info(level);
        }
        if let Some(level) = self.opt_level {
            options = options.opt_level(level);
        }
        if let Some(memcheck) = self.sanitize_memcheck {
            options = options.sanitize_memcheck(memcheck);
        }
        options
    }
}

fn choice<T: Copy>(
    name: &str,
    value: Result<String, VarError>,
    choices: &[(&str, T)],
) -> Result<Option<T>, String> {
    let expected = || {
        let texts: Vec<_> = choices.iter().map(|&(text, _)| text).collect();
        texts.join(", ")
    };
    match value {
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(_)) => Err(format!(
            "{name} is not valid Unicode; expected one of {}",
            expected()
        )),
        Ok(value) => choices
            .iter()
            .find(|&&(text, _)| text == value)
            .map(|&(_, choice)| Some(choice))
            .ok_or_else(|| format!("invalid {name}={value:?}; expected one of {}", expected())),
    }
}

impl CompileOptions {
    /// These options with the process's `CUTILE_JIT_*` overrides applied,
    /// which is what both compile stages see. The environment is read once
    /// per process.
    pub fn with_jit_env_overrides(&self) -> Result<CompileOptions, JITError> {
        static OVERRIDES: OnceLock<Result<JitEnvOverrides, String>> = OnceLock::new();
        let overrides = OVERRIDES
            .get_or_init(JitEnvOverrides::from_env)
            .as_ref()
            .map_err(|message| JITError::Generic(message.clone()))?;
        Ok(overrides.apply(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hints::CodegenMode;
    use std::ffi::OsString;

    const UNSET: Result<String, VarError> = Err(VarError::NotPresent);

    fn set(value: &str) -> Result<String, VarError> {
        Ok(value.to_string())
    }

    fn debug_only(level: DebugInfoLevel) -> JitEnvOverrides {
        JitEnvOverrides {
            debug_info: Some(level),
            ..Default::default()
        }
    }

    #[test]
    fn unset_variables_keep_code_options() {
        let overrides = JitEnvOverrides::parse(UNSET, UNSET, UNSET).unwrap();
        let code = CompileOptions::new()
            .opt_level(2)
            .debug_info(DebugInfoLevel::Line)
            .sanitize_memcheck(true);
        assert_eq!(overrides.apply(&code), code);
    }

    #[test]
    fn parses_every_documented_value() {
        for (text, level) in [
            ("none", DebugInfoLevel::None),
            ("line", DebugInfoLevel::Line),
            ("full", DebugInfoLevel::Full),
        ] {
            let overrides = JitEnvOverrides::parse(set(text), UNSET, UNSET).unwrap();
            assert_eq!(overrides.debug_info, Some(level));
        }
        for level in 0..=3u8 {
            let overrides = JitEnvOverrides::parse(UNSET, set(&level.to_string()), UNSET).unwrap();
            assert_eq!(overrides.opt_level, Some(level));
        }
        for (text, memcheck) in [("memcheck", true), ("none", false)] {
            let overrides = JitEnvOverrides::parse(UNSET, UNSET, set(text)).unwrap();
            assert_eq!(overrides.sanitize_memcheck, Some(memcheck));
        }
    }

    #[test]
    fn invalid_values_name_the_variable() {
        let not_unicode = Err(VarError::NotUnicode(OsString::from("full")));
        for (parsed, name) in [
            (
                JitEnvOverrides::parse(set("FULL"), UNSET, UNSET),
                JIT_DEBUG_ENV,
            ),
            (
                JitEnvOverrides::parse(not_unicode, UNSET, UNSET),
                JIT_DEBUG_ENV,
            ),
            (
                JitEnvOverrides::parse(UNSET, set("4"), UNSET),
                JIT_OPT_LEVEL_ENV,
            ),
            (
                JitEnvOverrides::parse(UNSET, UNSET, set("address")),
                JIT_SANITIZE_ENV,
            ),
        ] {
            let message = parsed.unwrap_err();
            assert!(message.contains(name), "{message}");
        }
    }

    #[test]
    fn env_full_debug_outranks_an_optimized_level_in_code() {
        let code = CompileOptions::new()
            .debug_info(DebugInfoLevel::None)
            .opt_level(3);
        let effective = debug_only(DebugInfoLevel::Full).apply(&code);
        assert_eq!(effective.codegen, CodegenMode::Debug);
    }

    #[test]
    fn full_debug_with_an_opt_level_is_rejected() {
        let message = JitEnvOverrides::parse(set("full"), set("2"), UNSET).unwrap_err();
        assert!(
            message.contains(JIT_DEBUG_ENV) && message.contains(JIT_OPT_LEVEL_ENV),
            "{message}"
        );
    }

    #[test]
    fn env_line_info_and_opt_level_combine() {
        let overrides = JitEnvOverrides::parse(set("line"), set("1"), UNSET).unwrap();
        let effective = overrides.apply(&CompileOptions::new().debug_info(DebugInfoLevel::Full));
        assert_eq!(
            effective.codegen,
            CodegenMode::Optimized {
                level: 1,
                lineinfo: true
            }
        );
    }

    #[test]
    fn env_debug_matches_the_code_setter() {
        let debug = CompileOptions::new()
            .debug_info(DebugInfoLevel::Full)
            .occupancy(2);
        let optimized = CompileOptions::new().opt_level(2);
        for code in [debug, optimized] {
            for level in [
                DebugInfoLevel::None,
                DebugInfoLevel::Line,
                DebugInfoLevel::Full,
            ] {
                assert_eq!(
                    debug_only(level).apply(&code),
                    code.clone().debug_info(level)
                );
            }
        }
    }

    #[test]
    fn env_sanitize_overrides_in_both_directions() {
        for (code, env, expected) in [(false, true, true), (true, false, false)] {
            let overrides = JitEnvOverrides {
                sanitize_memcheck: Some(env),
                ..Default::default()
            };
            let effective = overrides.apply(&CompileOptions::new().sanitize_memcheck(code));
            assert_eq!(effective.sanitize_memcheck, expected);
        }
    }
}
