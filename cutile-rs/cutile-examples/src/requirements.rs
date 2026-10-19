/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime prerequisites for examples, not Cargo features. Check before any
//! allocations or launches; errors after the check remain errors.

use cuda_async::error::DeviceError;
use cutile_compiler::cuda_tile_runtime_utils::ToolkitCapabilities;
use cutile_ir::capabilities::{CapabilityError, TargetCapabilities};
use cutile_ir::ir::{Location, Module};
use cutile_ir::requirements::Feature;

#[derive(Clone, Copy)]
pub struct Requirements {
    feature: Feature,
}

/// Ordinary arithmetic, pointer/tensor access, reductions and unscaled MMA.
/// The compiler's target rules also apply (e.g. Hopper needs Tile IR 13.3).
pub const BASELINE: Requirements = Requirements {
    feature: Feature::Baseline,
};

/// The NVFP4 and MXFP8 examples both use `mmaf_scaled`.
pub const BLOCK_SCALED_MMA: Requirements = Requirements {
    feature: Feature::ScaledMma,
};

/// Device GDC operations and the host-side PDL launch opt-in.
pub const PDL: Requirements = Requirements {
    feature: Feature::ProgrammaticDependentLaunch,
};

impl Requirements {
    /// Returns false with a `SKIP <example>:` diagnostic for unmet requirements.
    /// Uses the selected assembler, including `CUTILE_TILEIRAS_PATH`, and the
    /// requested device. Driver/tool discovery errors propagate, not skip.
    /// Driver entry-point availability is still checked by the launch itself.
    pub fn check(self, example: &str, device_id: usize) -> Result<bool, DeviceError> {
        let target = ToolkitCapabilities::for_device(device_id)
            .map(|caps| caps.target.clone())
            .map_err(|error| DeviceError::Launch(error.to_string()));
        self.report(example, target)
    }

    fn report(
        self,
        example: &str,
        target: Result<TargetCapabilities, DeviceError>,
    ) -> Result<bool, DeviceError> {
        let target = target?;
        match self.validate(&target) {
            Ok(()) => Ok(true),
            Err(reason) => {
                eprintln!("SKIP {example}: {reason}");
                Ok(false)
            }
        }
    }

    fn validate(self, target: &TargetCapabilities) -> Result<(), CapabilityError> {
        self.feature.check(target, &Location::Unknown)?;
        // Reuse the compiler's architecture support and per-target version
        // floors. No kernel is compiled here, so this cannot swallow JIT errors.
        target.validate_module(&Module::new("example_prerequisites"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile_ir::bytecode::BytecodeVersion;

    #[test]
    fn baseline_examples_stay_enabled_on_older_assemblers() {
        for version in BytecodeVersion::SUPPORTED {
            for sm in ["sm_80", "sm_100", "sm_120", "sm_121"] {
                BASELINE
                    .validate(&TargetCapabilities::new(version, sm))
                    .unwrap();
            }
        }
    }

    #[test]
    fn version_and_architecture_are_both_required() {
        for (requirements, version, sm, supported) in [
            (BLOCK_SCALED_MMA, BytecodeVersion::V13_2, "sm_120", false),
            (BLOCK_SCALED_MMA, BytecodeVersion::V13_3, "sm_90", false),
            (BLOCK_SCALED_MMA, BytecodeVersion::V13_3, "sm_120", true),
            (BLOCK_SCALED_MMA, BytecodeVersion::V13_3, "sm_121", true),
            (PDL, BytecodeVersion::V13_3, "sm_120", false),
            (PDL, BytecodeVersion::V13_4, "sm_89", false),
            (PDL, BytecodeVersion::V13_4, "sm_90", true),
            (PDL, BytecodeVersion::V13_4, "sm_120", true),
        ] {
            assert_eq!(
                requirements
                    .validate(&TargetCapabilities::new(version, sm))
                    .is_ok(),
                supported,
                "{version}, {sm}",
            );
        }
    }

    #[test]
    fn target_floors_match_the_compiler() {
        assert!(BASELINE
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_2, "sm_90"))
            .is_err());
        assert!(BASELINE
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_3, "sm_90"))
            .is_ok());
        assert!(BASELINE
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_3, "sm_107"))
            .is_err());
        assert!(BASELINE
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_4, "sm_107"))
            .is_ok());
        assert!(BASELINE
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_4, "sm_75"))
            .is_err());
    }

    #[test]
    fn skip_reason_names_the_requirement_and_selected_target() {
        let error = PDL
            .validate(&TargetCapabilities::new(BytecodeVersion::V13_3, "sm_120"))
            .unwrap_err();
        assert_eq!(error.to_string(), "programmatic dependent launch requires Tile IR 13.4 or newer; selected Tile IR 13.3, target sm_120");
    }

    #[test]
    fn discovery_errors_are_not_skips() {
        let error = BASELINE
            .report(
                "hello_world",
                Err(DeviceError::Launch("broken tileiras".into())),
            )
            .unwrap_err();
        assert!(error.to_string().contains("broken tileiras"));
        assert!(!PDL
            .report(
                "pdl",
                Ok(TargetCapabilities::new(BytecodeVersion::V13_3, "sm_120"))
            )
            .unwrap());
        assert!(BASELINE
            .report(
                "hello_world",
                Ok(TargetCapabilities::new(BytecodeVersion::V13_2, "sm_120"))
            )
            .unwrap());
    }

    #[test]
    fn every_example_declares_its_requirements() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let name = path.file_stem().unwrap().to_str().unwrap();
            let requirement = match name {
                "nvfp4" | "mxfp8" => "BLOCK_SCALED_MMA",
                "pdl" => "PDL",
                _ => "BASELINE",
            };
            let source: String = std::fs::read_to_string(&path)
                .unwrap()
                .split_whitespace()
                .collect();
            assert!(
                source.contains(&format!(
                    "cutile_examples::requirements::{requirement}.check(\"{name}\",0)"
                )),
                "missing requirement check in {}",
                path.display()
            );
        }
    }
}
