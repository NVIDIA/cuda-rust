/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

/// Compiler control for materializing semantic IKET annotations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IketInstrumentation {
    /// Erase annotations before ordinary code generation.
    Disabled,
    /// Prefer NativeDump and switch to ExtendedNativeDump above 30 names.
    Auto,
    /// Require NativeDump.
    NativeDump,
    /// Require ExtendedNativeDump.
    ExtendedNativeDump,
    /// Preserve an invalid environment value for a pipeline diagnostic.
    Invalid(String),
}

/// A device hint parsed at the frontend boundary, retaining invalid input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceArchHint {
    /// A GPU spelled with the `sm_` prefix.
    Valid(cuda_target_spec::DeviceArch),
    /// Raw input retained for a target-selection diagnostic.
    Invalid(String),
}

impl DeviceArchHint {
    /// Parse once while preserving the original input on failure.
    pub fn parse(value: String) -> Self {
        match value.parse() {
            Ok(arch) => Self::Valid(arch),
            Err(_) => Self::Invalid(value),
        }
    }

    /// Parse an environment read without silently dropping non-Unicode input.
    pub fn from_env_value(value: Result<String, std::env::VarError>) -> Option<Self> {
        match value {
            Ok(value) => Some(Self::parse(value)),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(value)) => {
                Some(Self::Invalid(value.to_string_lossy().into_owned()))
            }
        }
    }

    pub(crate) fn as_device_arch(
        &self,
    ) -> Result<&cuda_target_spec::DeviceArch, crate::error::PipelineError> {
        match self {
            Self::Valid(arch) => Ok(arch),
            Self::Invalid(value) => Err(crate::error::PipelineError::TargetSelection {
                target: value.clone(),
                reason: format!(
                    "invalid CUDA_OXIDE_DEVICE_ARCH `{value}`: expected sm_<capability> with an optional `a` suffix"
                ),
            }),
        }
    }
}

/// Explicit backend knobs; replaces every `CUDA_OXIDE_*` env read inside the
/// backend. `run_pipeline` (mir-importer) builds one from the environment at
/// its own boundary. The standalone API builds one from typed compile
/// options without reading the environment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackendOptions {
    /// IKET physical instrumentation policy.
    pub iket: IketInstrumentation,
    /// Hard target override (`llc -mcpu=`), e.g. `"sm_120"`.
    pub target_arch: Option<String>,
    /// Human-readable name for whatever set `target_arch`, used only to
    /// describe target provenance in diagnostics and errors (e.g.
    /// `"CUDA_OXIDE_TARGET"` for the env-driven rustc pipeline, or a
    /// caller-facing description for the standalone API).
    ///
    /// Keep this in step with `target_arch`: whoever writes one writes the
    /// other, or a target error names a source the caller never used.
    pub target_arch_source: &'static str,
    /// Advisory local-GPU arch; used only when it satisfies detected features.
    pub device_arch_hint: Option<DeviceArchHint>,
    /// Skip the `opt -O2` middle-end.
    pub no_opt: bool,
    /// Suppress `llc -fp-contract=fast` (fmul+fadd fusion to fma).
    pub no_fma: bool,
    /// Print progress and tool-selection notes to stderr.
    pub verbose: bool,
    /// Explicit `llc` binary (was `CUDA_OXIDE_LLC`).
    pub llc_override: Option<PathBuf>,
    /// Explicit `opt` binary (was `CUDA_OXIDE_OPT`).
    pub opt_override: Option<PathBuf>,
    /// Optional staged dialect-mir pass pipeline (`CUDA_OXIDE_MIR_PASSES`).
    ///
    /// Empty or `None` preserves the default pipeline. The available names
    /// are defined by the cuda-oxide-owned optimization registry. Each entry
    /// declares whether it runs before or after standard MIR preparation.
    pub mir_pass_pipeline: Option<String>,
    /// Stable per-compilation identity woven into counter-named module-scope
    /// symbols (`__shared_mem_*`, `__device_global_*`) and the dynamic
    /// shared-memory pool externs (`__dynamic_smem_*`) during MIR lowering.
    ///
    /// Required when the emitted PTX can be textually merged with other
    /// crates' bundles (`load_all_ptx_bundles_merged`): without it every
    /// module names its first shared allocation `__shared_mem_0` and the
    /// merged module fails driver JIT compilation on the duplicate
    /// definition (#1277). The rustc pipeline passes the crate's
    /// `StableCrateId` hash; `None` keeps the undecorated historical names.
    pub module_disambiguator: Option<u64>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            iket: IketInstrumentation::Auto,
            target_arch: None,
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: None,
            no_opt: false,
            no_fma: false,
            verbose: false,
            llc_override: None,
            opt_override: None,
            mir_pass_pipeline: None,
            module_disambiguator: None,
        }
    }
}

impl BackendOptions {
    /// Reads the historical `CUDA_OXIDE_*` variables; called by rustc-pipeline
    /// hosts, never by the backend itself. The only other env access in this
    /// crate is `CUDA_OXIDE_LLVM_LINK` in `llvm_tools::resolve_sibling_tool`
    /// (a per-toolchain tool override, not a compile option).
    pub fn from_env() -> Self {
        Self::from_env_with_device_hint(None)
    }

    /// Read backend controls, reusing a hint already parsed by the frontend.
    pub fn from_env_with_device_hint(device_arch_hint: Option<DeviceArchHint>) -> Self {
        let iket = match std::env::var("CUDA_OXIDE_IKET") {
            Err(std::env::VarError::NotPresent) => IketInstrumentation::Auto,
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "" | "1" | "on" | "true" | "auto" => IketInstrumentation::Auto,
                "native" | "native_dump" | "nativedump" => IketInstrumentation::NativeDump,
                "extended" | "extended_native_dump" | "extendednativedump" => {
                    IketInstrumentation::ExtendedNativeDump
                }
                "0" | "off" | "false" => IketInstrumentation::Disabled,
                _ => IketInstrumentation::Invalid(value),
            },
            Err(std::env::VarError::NotUnicode(value)) => {
                IketInstrumentation::Invalid(value.to_string_lossy().into_owned())
            }
        };
        Self {
            iket,
            target_arch: std::env::var("CUDA_OXIDE_TARGET").ok(),
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: device_arch_hint.or_else(|| {
                DeviceArchHint::from_env_value(std::env::var("CUDA_OXIDE_DEVICE_ARCH"))
            }),
            no_opt: std::env::var("CUDA_OXIDE_NO_OPT").is_ok(),
            no_fma: std::env::var("CUDA_OXIDE_NO_FMA").is_ok(),
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            llc_override: std::env::var("CUDA_OXIDE_LLC").ok().map(PathBuf::from),
            opt_override: std::env::var("CUDA_OXIDE_OPT").ok().map(PathBuf::from),
            mir_pass_pipeline: std::env::var("CUDA_OXIDE_MIR_PASSES").ok(),
            // Compilation identity, not an environment compatibility knob:
            // the pipeline host sets it from its own crate identity.
            module_disambiguator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceArchHint;

    #[test]
    fn supplied_device_hint_takes_precedence_over_the_environment() {
        const CHILD: &str = "CUDA_OXIDE_TEST_DEVICE_HINT_PRECEDENCE";
        if std::env::var_os(CHILD).is_some() {
            let hint = DeviceArchHint::parse("sm_120".to_string());
            let explicit = super::BackendOptions::from_env_with_device_hint(Some(hint.clone()));
            assert_eq!(explicit.device_arch_hint, Some(hint));
            assert_eq!(
                super::BackendOptions::from_env().device_arch_hint,
                Some(DeviceArchHint::Invalid("compute_120".to_string()))
            );
            return;
        }
        // Keep process-global environment mutation out of parallel tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "options::tests::supplied_device_hint_takes_precedence_over_the_environment",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("CUDA_OXIDE_DEVICE_ARCH", "compute_120")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn device_hint_boundary_preserves_invalid_values_and_absence() {
        assert_eq!(
            DeviceArchHint::from_env_value(Err(std::env::VarError::NotPresent)),
            None
        );
        for raw in ["compute_120", "sm_120f", "sm_", "foo"] {
            assert_eq!(
                DeviceArchHint::from_env_value(Ok(raw.to_string())),
                Some(DeviceArchHint::Invalid(raw.to_string()))
            );
        }
        assert!(matches!(
            DeviceArchHint::from_env_value(Ok("sm_120a".to_string())),
            Some(DeviceArchHint::Valid(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_device_hint_is_invalid_instead_of_absent() {
        use std::os::unix::ffi::OsStringExt;
        let raw = std::ffi::OsString::from_vec(vec![0xff]);
        let hint =
            DeviceArchHint::from_env_value(Err(std::env::VarError::NotUnicode(raw))).unwrap();
        assert!(matches!(hint, DeviceArchHint::Invalid(_)));
        assert!(hint.as_device_arch().is_err());
    }
}
