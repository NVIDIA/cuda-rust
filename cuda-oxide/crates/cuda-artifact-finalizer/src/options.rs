/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use libnvvm_sys::CudaArch;

/// Amount of device debug information preserved during finalization.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DebugPolicy {
    /// Do not request debug information from the CUDA compiler tools.
    #[default]
    None,
    /// Preserve source line mappings without disabling optimization.
    LineTables,
    /// Emit full debug information and disable libNVVM optimization.
    Full,
}

impl DebugPolicy {
    /// Parse a `CUDA_OXIDE_DEBUG` value into a debug policy.
    ///
    /// This is the single alias table for the environment variable. The
    /// rustc codegen backend uses it to select the DWARF emission level,
    /// and cargo-oxide uses it to decide build policy (a full-debug build
    /// applies the selective MIR/local-preservation and targeted outlining
    /// controls required before DWARF emission).
    /// Keeping both behind one parser means every accepted spelling, such
    /// as `2` for `full`, drives the whole pipeline consistently.
    ///
    /// Returns `None` for unrecognized values so callers fall back to
    /// their own defaults.
    #[must_use]
    pub fn parse_env_override(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "none" => Some(Self::None),
            "1" | "line" | "lines" | "line-tables" | "line-tables-only" => Some(Self::LineTables),
            "2" | "full" => Some(Self::Full),
            _ => None,
        }
    }
}

/// Typed options shared by the libNVVM and nvJitLink stages.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FinalizationOptions {
    target: CudaArch,
    allow_fma_contraction: bool,
    debug: DebugPolicy,
}

impl FinalizationOptions {
    /// Start with cuda-oxide's ordinary optimized compilation policy.
    pub fn new(target: CudaArch) -> Self {
        Self {
            target,
            allow_fma_contraction: true,
            debug: DebugPolicy::None,
        }
    }

    /// Select whether multiply-add contraction is permitted.
    #[must_use]
    pub fn with_fma_contraction(mut self, allow: bool) -> Self {
        self.allow_fma_contraction = allow;
        self
    }

    /// Select the device debug-information policy.
    #[must_use]
    pub fn with_debug_policy(mut self, debug: DebugPolicy) -> Self {
        self.debug = debug;
        self
    }

    /// Concrete CUDA architecture used by both compiler stages.
    pub fn target(&self) -> &CudaArch {
        &self.target
    }

    /// Whether multiply-add contraction is permitted.
    pub fn allow_fma_contraction(&self) -> bool {
        self.allow_fma_contraction
    }

    /// Device debug-information policy.
    pub fn debug_policy(&self) -> DebugPolicy {
        self.debug
    }

    pub(crate) fn nvvm_compile_options(&self) -> Vec<String> {
        let mut options = vec![
            format!("-arch={}", self.target.compute()),
            "-gen-lto".to_string(),
            self.fma_option().to_string(),
        ];
        if self.debug == DebugPolicy::Full {
            options.push("-g".to_string());
            options.push("-opt=0".to_string());
        }
        options
    }

    pub(crate) fn nvjitlink_ltoir_options(&self, output: FinalizerOutput) -> Vec<String> {
        let mut options = vec![format!("-arch={}", self.target.sm()), "-lto".to_string()];
        if output == FinalizerOutput::Ptx {
            options.push("-ptx".to_string());
        }
        self.append_nvjitlink_codegen_options(&mut options);
        options
    }

    /// Options for compiling one PTX module to cubin (no `-lto`).
    ///
    /// `-lineinfo` and `-g` are honored on this route (verified on CUDA 13.3:
    /// debug sections appear in the cubin). `-fma=<n>` is accepted without
    /// `-lto` but observed to be inert for PTX input on CUDA 13.3 nvJitLink:
    /// `-fma=0` and `-fma=1` produce byte-identical cubins even for
    /// contractable modeless `mul.f32`+`add.f32` PTX, which standalone
    /// `ptxas --fmad=false` does split. FMA policy for PTX inputs is
    /// therefore decided by the PTX producer (cuda-oxide's llc pass emits
    /// pre-fused `fma.rn` and modeless mul/add pairs that nvJitLink's
    /// internal SASS stage contracts by default). The option is still passed
    /// and digested so a future toolkit that honors it stays cache-correct.
    pub(crate) fn nvjitlink_ptx_options(&self) -> Vec<String> {
        let mut options = vec![format!("-arch={}", self.target.sm())];
        self.append_nvjitlink_codegen_options(&mut options);
        options
    }

    /// Semantic options for standalone PTX assembly with `ptxas`.
    pub(crate) fn ptxas_options(&self) -> Vec<String> {
        let mut options = vec![
            format!("--gpu-name={}", self.target.sm()),
            format!("--fmad={}", self.allow_fma_contraction),
        ];
        match self.debug {
            DebugPolicy::None => {}
            DebugPolicy::LineTables => options.push("--generate-line-info".to_string()),
            DebugPolicy::Full => options.push("--device-debug".to_string()),
        }
        options
    }

    fn append_nvjitlink_codegen_options(&self, options: &mut Vec<String>) {
        options.push(self.fma_option().to_string());
        match self.debug {
            DebugPolicy::None => {}
            DebugPolicy::LineTables => options.push("-lineinfo".to_string()),
            DebugPolicy::Full => options.push("-g".to_string()),
        }
    }

    /// Non-semantic nvJitLink options used only to collect resource diagnostics.
    ///
    /// These options deliberately stay out of artifact provenance and cache keys:
    /// they request compiler reporting without changing the generated program.
    ///
    /// `-no-cache` bypasses nvJitLink's own JIT cache for cubin output. A
    /// cache hit skips ptxas entirely and replays no info log, so a warm link
    /// would silently return an empty resource report; the report path exists
    /// to observe a real compile.
    pub(crate) fn nvjitlink_diagnostic_options(&self, output: FinalizerOutput) -> Vec<String> {
        match output {
            FinalizerOutput::Cubin => {
                vec![
                    "-verbose".to_string(),
                    "-Xptxas=-v".to_string(),
                    "-no-cache".to_string(),
                ]
            }
            FinalizerOutput::Ptx => vec!["-verbose".to_string()],
        }
    }

    fn fma_option(&self) -> &'static str {
        if self.allow_fma_contraction {
            "-fma=1"
        } else {
            "-fma=0"
        }
    }
}

/// Final artifact requested from nvJitLink.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FinalizerOutput {
    /// Native, target-specific CUDA ELF image.
    Cubin,
    /// Forward-compatible PTX assembly.
    Ptx,
}

/// One named linker input. Slice order is preserved exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NamedInput<'a> {
    /// Name shown by CUDA-tool diagnostics and included in provenance.
    pub name: &'a str,
    /// Complete input bytes in the format selected by the linker operation.
    pub bytes: &'a [u8],
}

impl<'a> NamedInput<'a> {
    /// Construct a named input without copying its bytes.
    pub const fn new(name: &'a str, bytes: &'a [u8]) -> Self {
        Self { name, bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_policy_env_override_accepts_every_alias() {
        for (value, expected) in [
            ("0", DebugPolicy::None),
            ("off", DebugPolicy::None),
            ("none", DebugPolicy::None),
            ("1", DebugPolicy::LineTables),
            ("line", DebugPolicy::LineTables),
            ("lines", DebugPolicy::LineTables),
            ("line-tables", DebugPolicy::LineTables),
            ("line-tables-only", DebugPolicy::LineTables),
            ("2", DebugPolicy::Full),
            ("full", DebugPolicy::Full),
        ] {
            assert_eq!(
                DebugPolicy::parse_env_override(value),
                Some(expected),
                "alias `{value}` must parse"
            );
        }
        // Whitespace and case are ignored, unknown values are rejected.
        assert_eq!(
            DebugPolicy::parse_env_override(" Full "),
            Some(DebugPolicy::Full)
        );
        assert_eq!(DebugPolicy::parse_env_override("verbose"), None);
        assert_eq!(DebugPolicy::parse_env_override(""), None);
    }

    #[test]
    fn option_order_preserves_target_lto_output_fma_and_debug_policy() {
        let target: CudaArch = "sm_90a".parse().unwrap();
        let base = FinalizationOptions::new(target).with_fma_contraction(false);

        assert_eq!(
            base.nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0"]
        );
        assert_eq!(
            base.nvjitlink_ltoir_options(FinalizerOutput::Cubin),
            ["-arch=sm_90a", "-lto", "-fma=0"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .nvjitlink_ltoir_options(FinalizerOutput::Ptx),
            ["-arch=sm_90a", "-lto", "-ptx", "-fma=0", "-lineinfo"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .nvjitlink_ptx_options(),
            ["-arch=sm_90a", "-fma=0", "-lineinfo"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::Full)
                .nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0", "-g", "-opt=0"]
        );
        assert_eq!(
            base.with_debug_policy(DebugPolicy::Full)
                .nvjitlink_ltoir_options(FinalizerOutput::Cubin),
            ["-arch=sm_90a", "-lto", "-fma=0", "-g"]
        );

        let base = FinalizationOptions::new("sm_90a".parse().unwrap()).with_fma_contraction(false);
        assert_eq!(base.ptxas_options(), ["--gpu-name=sm_90a", "--fmad=false"]);
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .ptxas_options(),
            ["--gpu-name=sm_90a", "--fmad=false", "--generate-line-info"]
        );
        assert_eq!(
            base.with_debug_policy(DebugPolicy::Full).ptxas_options(),
            ["--gpu-name=sm_90a", "--fmad=false", "--device-debug"]
        );
    }

    #[test]
    fn resource_diagnostics_are_separate_from_semantic_link_options() {
        let options = FinalizationOptions::new("sm_90a".parse().unwrap());

        assert_eq!(
            options.nvjitlink_diagnostic_options(FinalizerOutput::Cubin),
            ["-verbose", "-Xptxas=-v", "-no-cache"]
        );
        assert_eq!(
            options.nvjitlink_diagnostic_options(FinalizerOutput::Ptx),
            ["-verbose"]
        );
        let diagnostic = options.nvjitlink_diagnostic_options(FinalizerOutput::Cubin);
        assert!(
            options
                .nvjitlink_ltoir_options(FinalizerOutput::Cubin)
                .iter()
                .all(|option| !diagnostic.contains(option))
        );
        assert!(
            options
                .nvjitlink_ptx_options()
                .iter()
                .all(|option| !diagnostic.contains(option))
        );
    }

    #[test]
    fn fma_policy_is_explicit_in_both_stages() {
        let target: CudaArch = "sm_120".parse().unwrap();
        for allow in [false, true] {
            let options = FinalizationOptions::new(target.clone()).with_fma_contraction(allow);
            let expected = if allow { "-fma=1" } else { "-fma=0" };
            assert_eq!(
                options
                    .nvvm_compile_options()
                    .iter()
                    .filter(|option| option.starts_with("-fma="))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [expected]
            );
            assert_eq!(
                options
                    .nvjitlink_ltoir_options(FinalizerOutput::Cubin)
                    .iter()
                    .filter(|option| option.starts_with("-fma="))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [expected]
            );
            assert_eq!(
                options
                    .nvjitlink_ptx_options()
                    .iter()
                    .filter(|option| option.starts_with("-fma="))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [expected]
            );
        }
    }
}
