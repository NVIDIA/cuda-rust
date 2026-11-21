/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, EvidenceFixture, EvidenceStage, EvidenceStageKind, PtxVersion,
};
use crate::util::{pretty_json, sha256_file};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;
use std::process::{Command, Stdio};

const PREFIX: &str = "cuda-oxide:sparse_mma_fp8:v1 ";
const COMPARISONS_PER_FORM: usize = 2 * 6 * 32 * 4;
const IDS: [&str; 4] = [
    "mma_sp_m16n8k64_f32_e4m3_e4m3_f32",
    "mma_sp_m16n8k64_f32_e4m3_e5m2_f32",
    "mma_sp_m16n8k64_f32_e5m2_e4m3_f32",
    "mma_sp_m16n8k64_f32_e5m2_e5m2_f32",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    device: String,
    capability: (u16, u16),
    ptx_version: PtxVersion,
    ptx_target: String,
    ptx_sha256: String,
    coverage: Vec<(String, usize)>,
}

pub fn run(root: &Path, oracle: &str, llc: &Path) -> Result<()> {
    ensure!(
        oracle == "sparse_mma_fp8",
        "unsupported runtime oracle {oracle:?}"
    );
    let llc = llc.canonicalize().context("resolve llc path")?;
    let compiler = crate::probe::llc_identity(&llc)?;
    let dir = tempfile::tempdir().context("create oracle build directory")?;
    let cargo = |action: &str| {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(root)
            .env("CUDA_OXIDE_DEVICE_BACKEND", "native")
            .env_remove("CUDA_OXIDE_EMIT_NVVM_IR")
            .env("CUDA_OXIDE_LLC", &llc)
            .env("CUDA_OXIDE_PTX_DIR", dir.path())
            .args(["oxide", action, oracle, "--arch", "sm_89"])
            .stderr(Stdio::inherit());
        cmd
    };
    let build = cargo("build").output().context("build sparse FP8 oracle")?;
    eprint!("{}", String::from_utf8_lossy(&build.stdout));
    ensure!(
        build.status.success(),
        "oracle build failed ({})",
        build.status
    );
    // LLVM's default for sm_89 can be newer than the catalog's exact PTX floor.
    let ptx = dir.path().join("runtime.ptx");
    let compilation = Command::new(&llc)
        .args([
            "-march=nvptx64",
            "-mcpu=sm_89",
            "-mattr=+ptx84",
            "--nvptx-short-ptr",
            "-disable-branch-fold",
            "-disable-block-placement",
            "-fp-contract=fast",
        ])
        .arg(dir.path().join("sparse_mma_fp8.opt.ll"))
        .arg("-o")
        .arg(&ptx)
        .status()
        .context("compile runtime PTX")?;
    ensure!(
        compilation.success(),
        "runtime PTX compilation failed ({compilation})"
    );
    let digest = sha256_file(&ptx)?;
    let output = cargo("run")
        .args(["--", "--record-runtime"])
        .arg(&ptx)
        .output()
        .context("launch sparse FP8 oracle")?;
    let stdout = std::str::from_utf8(&output.stdout).context("read oracle stdout")?;
    eprint!("{stdout}");
    ensure!(
        output.status.success(),
        "GPU oracle failed ({})",
        output.status
    );
    let mut evidence = fixture(stdout, &digest)?;
    let stage = &mut evidence.stages[0];
    stage.tool_path = Some(llc.display().to_string());
    stage.tool_version = Some(compiler.version);
    stage.tool_sha256 = Some(compiler.sha256);
    print!("{}", pretty_json(&evidence)?);
    Ok(())
}

fn fixture(stdout: &str, expected_sha256: &str) -> Result<EvidenceFixture> {
    let mut reports = stdout.lines().filter_map(|line| line.strip_prefix(PREFIX));
    let line = reports
        .next()
        .context("GPU oracle did not report completion (skipped or incomplete)")?;
    ensure!(reports.next().is_none(), "ambiguous GPU oracle output");
    let mut report: Report = serde_json::from_str(line).context("parse GPU oracle report")?;
    let (major, minor) = report.capability;
    ensure!(
        !report.device.trim().is_empty() && minor <= 9 && (major, minor) >= (8, 9),
        "invalid device for sparse FP8 MMA"
    );
    ensure!(
        report.ptx_target == "sm_89" && report.ptx_version.encoded() == 84,
        "expected sm_89 PTX 8.4 for the LLVM fixture, got {} PTX {}",
        report.ptx_target,
        report.ptx_version
    );
    ensure!(
        report.ptx_sha256 == expected_sha256,
        "the oracle loaded a different PTX image"
    );
    report.coverage.sort_unstable();
    ensure!(
        report
            .coverage
            .iter()
            .map(|(id, count)| (id.as_str(), *count))
            .eq(IDS.map(|id| (id, COMPARISONS_PER_FORM))),
        "GPU oracle did not verify each of the four FP8 forms completely"
    );
    Ok(EvidenceFixture {
        id: "sparse_fp8_f32_runtime".into(),
        coverage_count: report.coverage.len(),
        stages: vec![EvidenceStage {
            stage: EvidenceStageKind::Runtime,
            mechanism: Some(BackendLoweringMechanism::InlinePtx),
            outcome: "succeeded".into(),
            targets: vec![
                format!("sm_{major}{minor}"),
                format!("ptx{}", report.ptx_version.encoded()),
            ],
            representation: "LLVM NVPTX module loaded by the CUDA driver".into(),
            detail: format!(
                "{} (CC {major}.{minor}), {} PTX SHA-256 {}. Exact host GEMM: {COMPARISONS_PER_FORM} comparisons per intrinsic (six metadata codes, zero and nonzero C, 32 lanes, four accumulators). IDs: {}.",
                report.device,
                report.ptx_target,
                report.ptx_sha256,
                IDS.join(", ")
            ),
            artifact_kind: None,
            tool_path: None,
            tool_version: None,
            tool_sha256: None,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn fixture(stdout: &str) -> Result<EvidenceFixture> {
        super::fixture(stdout, &"ab".repeat(32))
    }

    fn report() -> Value {
        json!({
            "device": "test device", "capability": [12, 0],
            "ptx_target": "sm_89", "ptx_version": "8.4", "ptx_sha256": "ab".repeat(32),
            "coverage": IDS.map(|id| (id, COMPARISONS_PER_FORM)),
        })
    }

    #[test]
    fn emits_a_schema_6_fixture_for_observed_hardware() {
        let fixture = fixture(&format!("build log\n{PREFIX}{}", report())).unwrap();
        let decoded: EvidenceFixture =
            serde_json::from_str(&pretty_json(&fixture).unwrap()).unwrap();
        assert_eq!(decoded.coverage_count, 4);
        let stage = &decoded.stages[0];
        assert_eq!(stage.stage, EvidenceStageKind::Runtime);
        assert_eq!(stage.mechanism, Some(BackendLoweringMechanism::InlinePtx));
        assert_eq!(stage.targets, ["sm_120", "ptx84"]);
        assert_eq!(stage.outcome, "succeeded");
    }

    #[test]
    fn rejects_skips_probe_output_and_ambiguous_completion() {
        for log in [
            "skipping: requires sm_89+",
            "probe: 13824 candidates",
            "SUCCESS",
        ] {
            assert!(fixture(log).is_err());
        }
        let line = format!("{PREFIX}{}", report());
        assert!(fixture(&format!("{line}\n{line}")).is_err());
        assert!(fixture(&format!("{PREFIX}{{}}")).is_err());
    }

    #[test]
    fn rejects_partial_and_duplicate_coverage() {
        let mut missing = report();
        missing["coverage"].as_array_mut().unwrap().pop();
        let mut duplicate = report();
        duplicate["coverage"][1] = duplicate["coverage"][0].clone();
        let mut partial = report();
        partial["coverage"][2][1] = json!(COMPARISONS_PER_FORM - 1);
        for invalid in [missing, duplicate, partial] {
            assert!(fixture(&format!("{PREFIX}{invalid}")).is_err());
        }
    }

    #[test]
    fn rejects_wrong_hardware_ptx_and_artifact_identity() {
        for (key, value) in [
            ("capability", json!([8, 6])),
            ("ptx_target", json!("sm_90")),
            ("ptx_version", json!("9.0")),
            ("ptx_sha256", json!("missing")),
        ] {
            let mut invalid = report();
            invalid[key] = value;
            assert!(fixture(&format!("{PREFIX}{invalid}")).is_err());
        }
    }
}
