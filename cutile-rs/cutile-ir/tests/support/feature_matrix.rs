/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile_ir::bytecode::BytecodeVersion;
use cutile_ir::requirements::{target_requirement, Architecture, Feature, Requirement, TARGETS};
use std::fmt::Write;

fn describe(requirement: Requirement) -> String {
    let arch = match requirement.architecture {
        Architecture::Any => String::new(),
        Architecture::AtLeast(sm) => format!(" and `sm_{sm}+`"),
        Architecture::Only(sm) => format!(" and `sm_{sm}` only"),
    };
    format!("Tile IR {}{arch}", requirement.since)
}

pub fn readme_targets() -> String {
    let mut out = String::from("| GPU compute capability | Minimum Tile IR version |\n|---|---|\n");
    let mut documented = Vec::new();
    for (label, targets) in [
        (
            "`sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89` (Ampere / Ada)",
            &[80, 86, 87, 88, 89][..],
        ),
        ("`sm_90` (Hopper)", &[90][..]),
        (
            "Blackwell `sm_100`, `sm_103`, `sm_110`, `sm_120`, `sm_121`",
            &[100, 103, 110, 120, 121][..],
        ),
        ("`sm_107`", &[107][..]),
    ] {
        let since = target_requirement(targets[0]).unwrap().since;
        for &sm in targets {
            assert_eq!(
                target_requirement(sm).unwrap().since,
                since,
                "split the README target row"
            );
            documented.push(sm);
        }
        writeln!(out, "| {label} | {since} |").unwrap();
    }
    documented.sort_unstable();
    assert_eq!(
        documented,
        TARGETS.iter().map(|(sm, _)| *sm).collect::<Vec<_>>(),
        "README must cover every supported target exactly once"
    );
    out
}

pub fn readme_matrix() -> String {
    // These groups share a README row. If a future version splits their
    // requirements, split the row instead of publishing a misleading summary.
    for group in [
        &[
            Feature::Insert,
            Feature::IntegerPower,
            Feature::GdcTokens,
            Feature::AliasFence,
        ][..],
        &[
            Feature::SaturatingFToI,
            Feature::PointerAttribute,
            Feature::Inbounds,
        ][..],
    ] {
        assert!(group
            .iter()
            .all(|feature| feature.requirement() == group[0].requirement()));
    }
    let mut out = String::from("| Raw feature | Requires |\n|---|---|\n");
    for (name, feature, extra) in [
        ("Allocation", Feature::Allocation, ""),
        ("Gather/scatter views", Feature::GatherScatterView, ""),
        ("Strided views", Feature::StridedView, ""),
        ("View atomic reduction", Feature::AtomicRedView, ""),
        ("FP4 packing", Feature::Fp4, ""),
        (
            "Block-scaled MMA",
            Feature::ScaledMma,
            "; valid operand/scale configuration",
        ),
        (
            "`insert`, `fpowi`, GDC tokens, alias fence",
            Feature::Insert,
            "",
        ),
        (
            "Saturating float-to-int, explicit pointer classification, view `inbounds`",
            Feature::SaturatingFToI,
            "",
        ),
        ("`f8e5m3fnu`", Feature::F8E5M3FNU, ""),
        (
            "Scaled MMA with `f8e5m3fnu` scales",
            Feature::NewScaleMma,
            "",
        ),
        (
            "Programmatic dependent launch (unsafe, per launch)",
            Feature::ProgrammaticDependentLaunch,
            "; driver `cuLaunchKernelEx` support",
        ),
    ] {
        writeln!(
            out,
            "| {name} | {}{extra} |",
            describe(feature.requirement())
        )
        .unwrap();
    }
    out
}

pub fn full_matrix() -> String {
    let mut out = String::from("| Feature |");
    for version in BytecodeVersion::SUPPORTED {
        write!(out, " {version} |").unwrap();
    }
    out.push_str(" Target requirement |\n|---|");
    for _ in BytecodeVersion::SUPPORTED {
        out.push_str("---|");
    }
    out.push_str("---|\n");
    for &feature in Feature::ALL {
        let requirement = feature.requirement();
        write!(out, "| {} |", feature.name()).unwrap();
        for version in BytecodeVersion::SUPPORTED {
            write!(
                out,
                " {} |",
                if version >= requirement.since {
                    "Yes"
                } else {
                    "—"
                }
            )
            .unwrap();
        }
        let arch = match requirement.architecture {
            Architecture::Any => "Any supported target".to_string(),
            Architecture::AtLeast(sm) => format!("`sm_{sm}+`"),
            Architecture::Only(sm) => format!("`sm_{sm}` only"),
        };
        writeln!(out, " {arch} |").unwrap();
    }
    out.push_str("\n| Target | Minimum emitted Tile IR version |\n|---|---|\n");
    for &(sm, feature) in TARGETS {
        writeln!(out, "| `sm_{sm}` | {} |", feature.since()).unwrap();
    }
    out
}
