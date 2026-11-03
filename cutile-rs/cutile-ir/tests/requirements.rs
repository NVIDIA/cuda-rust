/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[path = "support/feature_matrix.rs"]
mod feature_matrix;

use cutile_ir::bytecode::{BytecodeVersion, Opcode};
use cutile_ir::capabilities::TargetCapabilities;
use cutile_ir::ir::{Attribute, Location, ScalarType};
use cutile_ir::requirements::{
    attribute_feature, opcode_requirement, scalar_requirement, Architecture, Feature, OPCODES,
    TARGETS,
};

#[test]
fn all_registry_entries_require_both_version_and_architecture() {
    let loc = Location::FileLineCol {
        filename: "requirements.rs".into(),
        line: 10,
        column: 3,
    };
    for &feature in Feature::ALL {
        let requirement = feature.requirement();
        for version in BytecodeVersion::SUPPORTED {
            for &(sm, _) in TARGETS {
                let arch_ok = match requirement.architecture {
                    Architecture::Any => true,
                    Architecture::AtLeast(min) => sm >= min,
                    Architecture::Only(exact) => sm == exact,
                };
                let target = TargetCapabilities::new(version, format!("sm_{sm}"));
                let result = feature.check(&target, &loc);
                assert_eq!(
                    result.is_ok(),
                    version >= requirement.since && arch_ok,
                    "{feature:?}, {version}, sm_{sm}"
                );
                if let Err(error) = result {
                    assert_eq!(*error.location, loc);
                }
            }
        }
    }
}

#[test]
fn architecture_families_are_not_a_numeric_feature_ladder() {
    for sm in ["sm_107", "sm_107a", "sm_107f"] {
        let target = TargetCapabilities::new(BytecodeVersion::V13_4, sm);
        Feature::NewScaleMma
            .check(&target, &Location::Unknown)
            .unwrap();
    }
    for sm in ["sm_100", "sm_110", "sm_120", "sm_121"] {
        let target = TargetCapabilities::new(BytecodeVersion::V13_4, sm);
        Feature::ScaledMma
            .check(&target, &Location::Unknown)
            .unwrap();
        assert!(Feature::NewScaleMma
            .check(&target, &Location::Unknown)
            .is_err());
    }
}

#[test]
fn writer_versions_come_from_the_registry() {
    let mut names = std::collections::HashSet::new();
    for &op in OPCODES {
        assert!(names.insert(op.name()), "duplicate opcode {op:?}");
        assert_eq!(op.minimum_version(), opcode_requirement(op).since);
    }
    for scalar in [ScalarType::I4, ScalarType::F4E2M1FN, ScalarType::F8E5M3FNU] {
        assert_eq!(scalar.minimum_version(), scalar_requirement(scalar).since);
    }
    assert_eq!(
        opcode_requirement(Opcode::Insert).since,
        BytecodeVersion::V13_4
    );
    assert_eq!(
        opcode_requirement(Opcode::MmaFScaled).since,
        BytecodeVersion::V13_3
    );
}

#[test]
fn default_attributes_do_not_raise_the_version_floor() {
    assert_eq!(
        attribute_feature(Opcode::FToI, "saturating", &Attribute::Bool(false)),
        None
    );
    assert_eq!(
        attribute_feature(Opcode::MmaF, "fast_acc", &Attribute::Bool(false)),
        None
    );
    assert_eq!(
        attribute_feature(Opcode::Exp, "rounding_mode", &Attribute::i32(5)),
        None
    );
    assert_eq!(
        attribute_feature(
            Opcode::LoadViewTko,
            "inbounds",
            &Attribute::Array(vec![Attribute::Bool(false)])
        ),
        None
    );
    assert_eq!(
        attribute_feature(Opcode::FToI, "saturating", &Attribute::Bool(true)),
        Some(Feature::SaturatingFToI)
    );
    assert_eq!(
        attribute_feature(Opcode::Exp, "rounding_mode", &Attribute::i32(0)),
        Some(Feature::ExpRounding)
    );
}

#[test]
fn documented_matrices_match_the_registry() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // Workspace documentation is not part of the published IR crate.
    if !root.join("../cutile/Cargo.toml").exists() {
        return;
    }
    for (file, marker, expected) in [
        (
            "../README.md",
            "TILE IR TARGETS",
            feature_matrix::readme_targets(),
        ),
        (
            "../README.md",
            "TILE IR REQUIREMENTS",
            feature_matrix::readme_matrix(),
        ),
        (
            "../cutile-book/reference/compatibility.md",
            "TILE IR REQUIREMENTS",
            feature_matrix::full_matrix(),
        ),
    ] {
        let path = root.join(file);
        let text = std::fs::read_to_string(path).unwrap();
        let actual = text
            .split(&format!("<!-- BEGIN {marker} -->\n"))
            .nth(1)
            .unwrap()
            .split(&format!("<!-- END {marker} -->"))
            .next()
            .unwrap();
        assert_eq!(
            actual, expected,
            "{file}: regenerate with cargo run -p cutile-ir --example feature_matrix"
        );
    }
}
