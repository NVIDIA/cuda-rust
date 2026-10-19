/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::extract::read_upstream_lock;
use crate::model::{
    BackendLoweringMechanism, CatalogFile, CatalogInputs, CatalogIntrinsic, CatalogLlvm,
    CatalogTargetRequirement, CpAsyncSourceSize, EvidenceStageKind, IntrinsicBackend,
    IntrinsicSource, RustcCommit, SparseMmaSelector, WarpShuffleAdapter,
};
use crate::ptx::{
    InstructionPattern, OperandPattern, instructions_with_matching_head, matching_instructions,
};
use crate::render::render_probe;
use crate::resolve::{resolve, resolve_candidate};
use crate::util::{pretty_json, sha256_bytes, sha256_file};
use anyhow::{Context, Result, ensure};
use cuda_target_spec::{CudaArch, PtxSpelling, recorded_ptx_floor};
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolIdentity {
    version: String,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedBackendIdentity {
    tool: ToolIdentity,
    rustc_commit: RustcCommit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeBackendIdentity {
    SelectedEvidence(SelectedBackendIdentity),
    Comparison(ToolIdentity),
}

impl ProbeBackendIdentity {
    fn tool(&self) -> &ToolIdentity {
        match self {
            Self::SelectedEvidence(identity) => &identity.tool,
            Self::Comparison(identity) => identity,
        }
    }

    fn is_selected_evidence(&self) -> bool {
        matches!(self, Self::SelectedEvidence(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateProbeOptions {
    pub intrinsic_id: String,
    pub llc: PathBuf,
    pub gpu_target: String,
    pub ptx_feature: String,
    pub ptxas: Option<PathBuf>,
    pub skip_terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CandidateProbeDraft {
    schema: u32,
    kind: String,
    admitted: bool,
    comparison_only: bool,
    intrinsic_id: String,
    operation_key: String,
    source: IntrinsicSource,
    mechanism: BackendLoweringMechanism,
    expected_ptx: InstructionPattern,
    target: CandidateTargetDraft,
    catalog_inputs: CatalogInputs,
    tools: Vec<CandidateToolDraft>,
    artifacts: Vec<CandidateArtifactDraft>,
    stages: Vec<CandidateStageDraft>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CandidateTargetDraft {
    requirement: CatalogTargetRequirement,
    target_triple: String,
    gpu_target: String,
    ptx_feature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CandidateToolDraft {
    role: String,
    path: String,
    version: String,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CandidateArtifactDraft {
    kind: String,
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CandidateStageDraft {
    stage: String,
    outcome: String,
    detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CandidateArtifactPaths {
    llvm_ir: PathBuf,
    llvm_bitcode: PathBuf,
    canonical_llvm_ir: PathBuf,
    ptx: PathBuf,
    cubin: PathBuf,
    manifest: PathBuf,
    manifest_temp: PathBuf,
}

pub fn run(
    repo_root: &Path,
    intrinsic_id: &str,
    llc: Option<PathBuf>,
    skip_terminal: bool,
    per_target: bool,
) -> Result<()> {
    let catalog = resolve(repo_root)?;
    let record = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == intrinsic_id)
        .with_context(|| format!("unknown catalog intrinsic {intrinsic_id}"))?;
    let runner = ProbeRunner::new(repo_root, &catalog, llc, skip_terminal)?;
    runner.validate_backend(record)?;
    runner.run(record, per_target).map(|_| ())
}

pub fn run_all(
    repo_root: &Path,
    llc: Option<PathBuf>,
    skip_terminal: bool,
    per_target: bool,
) -> Result<()> {
    let catalog = resolve(repo_root)?;
    ensure!(
        !catalog.intrinsics.is_empty(),
        "resolved catalog has no intrinsics to probe"
    );
    let runner = ProbeRunner::new(repo_root, &catalog, llc, skip_terminal)?;

    validate_backend_identities(
        &runner.identity,
        &runner.expected_commit,
        catalog.intrinsics.iter().map(|record| BackendExpectation {
            intrinsic_id: &record.id,
            version: &record.backend.version,
        }),
    )?;

    let total = catalog.intrinsics.len();
    let mut target_probes = 0;
    for (index, record) in catalog.intrinsics.iter().enumerate() {
        eprintln!("[{}/{}] probing {}", index + 1, total, record.id);
        target_probes += runner
            .run(record, per_target)
            .with_context(|| format!("probe {}", record.id))?;
    }
    println!(
        "probed {target_probes} target configurations across all {total} generated intrinsic routes"
    );
    Ok(())
}

struct ProbeRunner<'a> {
    catalog: &'a CatalogFile,
    llc: PathBuf,
    identity: ProbeBackendIdentity,
    expected_commit: RustcCommit,
    output_dir: PathBuf,
    catalog_hash: String,
    skip_terminal: bool,
}

impl<'a> ProbeRunner<'a> {
    fn new(
        repo_root: &Path,
        catalog: &'a CatalogFile,
        llc: Option<PathBuf>,
        skip_terminal: bool,
    ) -> Result<Self> {
        let expected_commit = read_upstream_lock(repo_root)?.rust_toolchain.commit_hash;
        let (llc, identity) = match llc {
            Some(path) => {
                let identity = ProbeBackendIdentity::Comparison(llc_identity(&path)?);
                (path, identity)
            }
            None => {
                let (path, rustc_commit) = rust_toolchain_llc()?;
                let tool = llc_identity(&path)?;
                let identity = ProbeBackendIdentity::SelectedEvidence(SelectedBackendIdentity {
                    tool,
                    rustc_commit,
                });
                (path, identity)
            }
        };
        let output_dir = repo_root.join("target/intrinsics/probes");
        fs::create_dir_all(&output_dir)?;
        let catalog_json = pretty_json(catalog)?;
        let catalog_hash = sha256_bytes(catalog_json.as_bytes());
        Ok(Self {
            catalog,
            llc,
            identity,
            expected_commit,
            output_dir,
            catalog_hash,
            skip_terminal,
        })
    }

    fn validate_backend(&self, record: &CatalogIntrinsic) -> Result<()> {
        validate_backend_identity(
            &record.backend.version,
            &self.expected_commit,
            &self.identity,
        )
    }

    fn run(&self, record: &CatalogIntrinsic, per_target: bool) -> Result<usize> {
        let intrinsic_id = &record.id;
        let input = self.output_dir.join(format!("{intrinsic_id}.ll"));
        fs::write(
            &input,
            render_probe(self.catalog, record, &self.catalog_hash),
        )
        .with_context(|| format!("write in-memory probe {}", input.display()))?;

        if record.llvm.is_some()
            && self.identity.is_selected_evidence()
            && uses_typed_llvm_nvptx_lowering(record)
        {
            assert_intrinsic_declaration_canonicalizes(
                &self.llc,
                &input,
                &self.output_dir,
                intrinsic_id,
                record,
            )?;
        }
        let targets = if per_target && record.target.targets.contains('|') {
            record.target.targets.split('|').collect::<Vec<_>>()
        } else {
            vec![record.backend.gpu_target.as_str()]
        };
        let target_count = targets.len();
        for gpu_target in targets {
            self.run_target(record, &input, gpu_target, per_target)?;
        }
        Ok(target_count)
    }

    fn run_target(
        &self,
        record: &CatalogIntrinsic,
        input: &Path,
        gpu_target: &str,
        per_target: bool,
    ) -> Result<()> {
        let intrinsic_id = &record.id;
        let selected_evidence_target = gpu_target == record.backend.gpu_target;
        let (ptx_feature, effective_floor) = if !per_target || selected_evidence_target {
            (
                Some(record.backend.ptx_feature.clone()),
                record.target.minimum_ptx.encoded(),
            )
        } else {
            derived_ptx_feature(gpu_target, record.target.minimum_ptx.encoded()).with_context(
                || format!("derive PTX configuration for {intrinsic_id} target {gpu_target}"),
            )?
        };
        let output = self.output_dir.join(if per_target {
            format!("{intrinsic_id}.{gpu_target}.ptx")
        } else {
            format!("{intrinsic_id}.ptx")
        });
        let mut command = Command::new(&self.llc);
        command
            .arg(input)
            .arg("-march=nvptx64")
            .arg(format!("-mcpu={gpu_target}"));
        if let Some(ptx_feature) = &ptx_feature {
            command.arg(format!("-mattr={ptx_feature}"));
        }
        let status = command
            .arg("-o")
            .arg(&output)
            .status()
            .with_context(|| format!("run {}", self.llc.display()))?;
        ensure!(status.success(), "LLVM probe failed with {status}");
        let ptx = fs::read_to_string(&output)
            .with_context(|| format!("read generated PTX {}", output.display()))?;
        validate_probe_instructions(record, &ptx)?;
        let has_terminal_stage = record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LlvmNvptx
                && lowering
                    .stages
                    .iter()
                    .any(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
        });
        if self.identity.is_selected_evidence() && has_terminal_stage {
            if !selected_evidence_target {
                println!(
                    "derived target {gpu_target} is LLVM-selection-only; terminal assembly evidence exists only for selected target {}",
                    record.backend.gpu_target
                );
            } else if self.skip_terminal {
                println!(
                    "backend-only probe: `--skip-terminal` was explicit, so recorded ptxas evidence was not revalidated"
                );
            } else {
                assemble_probe_ptx(record, &output, &self.output_dir, intrinsic_id)?;
            }
        }
        let ptx_configuration = ptx_feature.unwrap_or_else(|| {
            format!(
                "target default (effective floor PTX {}.{})",
                effective_floor / 10,
                effective_floor % 10
            )
        });
        let tool = self.identity.tool();
        match &self.identity {
            ProbeBackendIdentity::SelectedEvidence(identity) if selected_evidence_target => {
                println!(
                    "selected evidence backend {} (rustc commit {}; llc SHA-256 {}) lowered {} to `{}` for selected evidence target {} {}",
                    tool.version,
                    identity.rustc_commit,
                    tool.sha256,
                    intrinsic_id,
                    record.expected_ptx,
                    gpu_target,
                    ptx_configuration,
                )
            }
            ProbeBackendIdentity::SelectedEvidence(identity) => println!(
                "selected evidence backend {} (rustc commit {}; llc SHA-256 {}) lowered {} to `{}` for derived target {} {}; record evidence selects {} {}",
                tool.version,
                identity.rustc_commit,
                tool.sha256,
                intrinsic_id,
                record.expected_ptx,
                gpu_target,
                ptx_configuration,
                record.backend.gpu_target,
                record.backend.ptx_feature,
            ),
            ProbeBackendIdentity::Comparison(_) => println!(
                "comparison backend {} (SHA-256 {}) lowered {} to `{}` for {} target {} {}; record evidence selects {} {}; this does not validate selected evidence {} (SHA-256 {})",
                tool.version,
                tool.sha256,
                intrinsic_id,
                record.expected_ptx,
                if selected_evidence_target {
                    "evidence-selected"
                } else {
                    "derived"
                },
                gpu_target,
                ptx_configuration,
                record.backend.gpu_target,
                record.backend.ptx_feature,
                record.backend.version,
                record.backend.sha256,
            ),
        }
        println!("PTX: {}", output.display());
        Ok(())
    }
}

fn derived_ptx_feature(gpu_target: &str, instruction_floor: u16) -> Result<(Option<String>, u16)> {
    ensure!(
        gpu_target.starts_with("sm_"),
        "unsupported catalog GPU target {gpu_target:?}"
    );
    let arch = CudaArch::from_str(gpu_target).context("parse derived catalog GPU target")?;
    let target_floor = recorded_ptx_floor(&arch).with_context(|| {
        format!(
            "derived target {gpu_target} has no recorded PTX ISA floor; cannot derive the PTX configuration production uses"
        )
    })?;
    if instruction_floor <= 60 {
        return Ok((None, target_floor));
    }
    let spelling = PtxSpelling::round_up(instruction_floor).with_context(|| {
        format!("instruction PTX floor {instruction_floor} has no supported llc feature spelling")
    })?;
    match spelling.feature_beyond_floor(target_floor) {
        Some(feature) => Ok((Some(feature.to_string()), spelling.get())),
        None => Ok((None, target_floor)),
    }
}

struct BackendExpectation<'a> {
    intrinsic_id: &'a str,
    version: &'a str,
}

fn validate_backend_identities<'a>(
    actual: &ProbeBackendIdentity,
    expected_commit: &RustcCommit,
    identities: impl IntoIterator<Item = BackendExpectation<'a>>,
) -> Result<()> {
    for expectation in identities {
        validate_backend_identity(expectation.version, expected_commit, actual)
            .with_context(|| format!("validate probe backend for {}", expectation.intrinsic_id))?;
    }
    Ok(())
}

pub(crate) fn run_candidate(repo_root: &Path, options: CandidateProbeOptions) -> Result<()> {
    ensure!(
        options.skip_terminal ^ options.ptxas.is_some(),
        "candidate probe requires either --ptxas FILE or explicit --skip-terminal"
    );
    let output_dir = repo_root.join("target/intrinsics/probes");
    fs::create_dir_all(&output_dir)?;
    let stem = candidate_artifact_stem(&options.intrinsic_id)?;
    let paths = candidate_artifact_paths(&output_dir, stem);
    clear_candidate_artifacts(&paths)?;

    let backend_identity = llc_identity(&options.llc)?;
    let candidate = resolve_candidate(
        repo_root,
        &options.intrinsic_id,
        &backend_identity.version,
        &backend_identity.sha256,
        &options.gpu_target,
        &options.ptx_feature,
    )?;
    let record = candidate
        .catalog
        .intrinsics
        .first()
        .context("candidate resolver returned no intrinsic")?;
    ensure!(
        record.id == options.intrinsic_id || record.rust.abi_id == options.intrinsic_id,
        "candidate resolver returned the wrong intrinsic"
    );

    let catalog_json = pretty_json(&candidate.catalog)?;
    let catalog_hash = sha256_bytes(catalog_json.as_bytes());
    write_new_file(
        &paths.llvm_ir,
        render_probe(&candidate.catalog, record, &catalog_hash),
    )
    .with_context(|| format!("write candidate probe {}", paths.llvm_ir.display()))?;

    let mut tools = vec![candidate_tool("llc", &options.llc, &backend_identity)];
    let canonicalizes =
        record.llvm.is_some() && candidate.mechanism == BackendLoweringMechanism::TypedNvvm;
    if canonicalizes {
        let (llvm_as, llvm_dis) = sibling_llvm_tools(&options.llc)?;
        tools.push(candidate_tool(
            "llvm-as",
            &llvm_as,
            &llc_identity(&llvm_as)?,
        ));
        tools.push(candidate_tool(
            "llvm-dis",
            &llvm_dis,
            &llc_identity(&llvm_dis)?,
        ));
    }
    let ptxas_identity = options.ptxas.as_deref().map(ptxas_identity).transpose()?;
    if let (Some(ptxas), Some(identity)) = (options.ptxas.as_deref(), ptxas_identity.as_ref()) {
        tools.push(candidate_tool("ptxas", ptxas, identity));
    }
    let mut draft = CandidateProbeDraft {
        schema: 1,
        kind: "candidate_probe".into(),
        admitted: false,
        comparison_only: true,
        intrinsic_id: record.id.clone(),
        operation_key: record.operation_key.clone(),
        source: record.source.clone(),
        mechanism: candidate.mechanism,
        expected_ptx: record.expected_ptx.clone(),
        target: CandidateTargetDraft {
            requirement: candidate.requirement,
            target_triple: record.backend.target_triple.clone(),
            gpu_target: options.gpu_target.clone(),
            ptx_feature: options.ptx_feature.clone(),
        },
        catalog_inputs: candidate.catalog.inputs.clone(),
        tools,
        artifacts: Vec::new(),
        stages: Vec::new(),
    };
    write_candidate_draft(repo_root, &output_dir, &paths, &mut draft)?;

    if canonicalizes {
        let result = assert_intrinsic_declaration_canonicalizes(
            &options.llc,
            &paths.llvm_ir,
            &output_dir,
            &format!("{stem}.candidate"),
            record,
        );
        match result {
            Ok(()) => draft.stages.push(candidate_stage(
                "declaration_canonicalization",
                "succeeded",
                "LLVM declaration facts matched the imported record",
            )),
            Err(error) => {
                draft.stages.push(candidate_stage(
                    "declaration_canonicalization",
                    "failed",
                    &format!("{error:#}"),
                ));
                write_candidate_draft(repo_root, &output_dir, &paths, &mut draft)?;
                return Err(error.context("candidate declaration canonicalization failed"));
            }
        }
    } else {
        draft.stages.push(candidate_stage(
            "declaration_canonicalization",
            "not_applicable",
            "candidate route does not use a typed LLVM intrinsic",
        ));
    }
    write_candidate_draft(repo_root, &output_dir, &paths, &mut draft)?;

    let ptx = match lower_candidate_ptx(
        &options.llc,
        &paths.llvm_ir,
        &options.gpu_target,
        &options.ptx_feature,
        &paths.ptx,
    ) {
        Ok(ptx) => ptx,
        Err(error) => {
            record_candidate_failure(
                repo_root,
                &output_dir,
                &paths,
                &mut draft,
                "backend_codegen",
                &error,
            )?;
            return Err(error.context("candidate LLVM probe failed"));
        }
    };
    if let Err(error) = validate_probe_instructions(record, &ptx) {
        record_candidate_failure(
            repo_root,
            &output_dir,
            &paths,
            &mut draft,
            "backend_codegen",
            &error,
        )?;
        return Err(error.context("candidate PTX instruction validation failed"));
    }
    draft.stages.push(candidate_stage(
        "backend_codegen",
        "succeeded",
        "llc output matched the reviewed PTX instruction shape",
    ));
    write_candidate_draft(repo_root, &output_dir, &paths, &mut draft)?;

    if options.skip_terminal {
        draft.stages.push(candidate_stage(
            "ptx_assembly",
            "skipped",
            "--skip-terminal was explicit; this is a backend-only draft",
        ));
    } else {
        let ptxas = options
            .ptxas
            .as_deref()
            .context("candidate terminal validation has no explicit ptxas")?;
        if let Err(error) =
            assemble_candidate_ptx(ptxas, &options.gpu_target, &paths.ptx, &paths.cubin)
        {
            record_candidate_failure(
                repo_root,
                &output_dir,
                &paths,
                &mut draft,
                "ptx_assembly",
                &error,
            )?;
            return Err(error.context("candidate PTX assembly failed"));
        }
        draft.stages.push(candidate_stage(
            "ptx_assembly",
            "succeeded",
            "the explicit ptxas accepted the generated PTX",
        ));
    }
    write_candidate_draft(repo_root, &output_dir, &paths, &mut draft)?;
    println!(
        "candidate comparison lowered {} to `{}` for {} {}; admitted=false",
        record.id, record.expected_ptx, options.gpu_target, options.ptx_feature
    );
    println!("draft: {}", paths.manifest.display());
    Ok(())
}

fn candidate_artifact_stem(intrinsic_id: &str) -> Result<&str> {
    ensure!(
        !intrinsic_id.is_empty()
            && intrinsic_id
                .bytes()
                .all(|byte| { byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' }),
        "candidate intrinsic ID {intrinsic_id:?} is not safe for an artifact name"
    );
    Ok(intrinsic_id)
}

fn candidate_artifact_paths(output_dir: &Path, stem: &str) -> CandidateArtifactPaths {
    CandidateArtifactPaths {
        llvm_ir: output_dir.join(format!("{stem}.candidate.ll")),
        llvm_bitcode: output_dir.join(format!("{stem}.candidate.bc")),
        canonical_llvm_ir: output_dir.join(format!("{stem}.candidate.canonical.ll")),
        ptx: output_dir.join(format!("{stem}.candidate.ptx")),
        cubin: output_dir.join(format!("{stem}.candidate.cubin")),
        manifest: output_dir.join(format!("{stem}.candidate.json")),
        manifest_temp: output_dir.join(format!("{stem}.candidate.json.tmp")),
    }
}

fn candidate_artifact_entries(paths: &CandidateArtifactPaths) -> [(&'static str, &Path); 5] {
    [
        ("llvm_ir", paths.llvm_ir.as_path()),
        ("llvm_bitcode", paths.llvm_bitcode.as_path()),
        ("canonical_llvm_ir", paths.canonical_llvm_ir.as_path()),
        ("ptx", paths.ptx.as_path()),
        ("cubin", paths.cubin.as_path()),
    ]
}

fn clear_candidate_artifacts(paths: &CandidateArtifactPaths) -> Result<()> {
    for path in [
        &paths.llvm_ir,
        &paths.llvm_bitcode,
        &paths.canonical_llvm_ir,
        &paths.ptx,
        &paths.cubin,
        &paths.manifest,
        &paths.manifest_temp,
    ] {
        remove_if_present(path)?;
    }
    Ok(())
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove stale {}", path.display())),
    }
}

fn write_new_file(path: &Path, contents: impl AsRef<str>) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(contents.as_ref().as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

fn lower_candidate_ptx(
    llc: &Path,
    input: &Path,
    gpu_target: &str,
    ptx_feature: &str,
    output: &Path,
) -> Result<String> {
    let status = Command::new(llc)
        .arg(input)
        .arg("-march=nvptx64")
        .arg(format!("-mcpu={gpu_target}"))
        .arg(format!("-mattr={ptx_feature}"))
        .arg("-o")
        .arg(output)
        .status()
        .with_context(|| format!("run {}", llc.display()))?;
    ensure!(status.success(), "llc exited with {status}");
    ensure_regular_artifact(output, "candidate PTX")?;
    fs::read_to_string(output).with_context(|| format!("read candidate PTX {}", output.display()))
}

fn assemble_candidate_ptx(ptxas: &Path, gpu_target: &str, ptx: &Path, cubin: &Path) -> Result<()> {
    remove_if_present(cubin)?;
    let status = Command::new(ptxas)
        .arg(format!("-arch={gpu_target}"))
        .arg(ptx)
        .arg("-o")
        .arg(cubin)
        .status()
        .with_context(|| format!("run {}", ptxas.display()))?;
    ensure!(status.success(), "ptxas exited with {status}");
    validate_candidate_cubin(cubin)
}

fn ensure_regular_artifact(path: &Path, kind: &str) -> Result<fs::Metadata> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("{kind} was not produced at {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "{kind} at {} is not a regular file",
        path.display()
    );
    Ok(metadata)
}

fn validate_candidate_cubin(cubin: &Path) -> Result<()> {
    const ELF64_HEADER_LEN: usize = 64;
    let metadata = ensure_regular_artifact(cubin, "candidate cubin")?;
    ensure!(
        metadata.len() >= ELF64_HEADER_LEN as u64,
        "candidate cubin at {} is too small to be an ELF file",
        cubin.display()
    );
    let mut header = [0_u8; ELF64_HEADER_LEN];
    fs::File::open(cubin)
        .with_context(|| format!("open candidate cubin {}", cubin.display()))?
        .read_exact(&mut header)
        .with_context(|| format!("read candidate cubin {}", cubin.display()))?;
    ensure!(
        header[..7] == [0x7f, b'E', b'L', b'F', 2, 1, 1]
            && header[16..18] == [2, 0]
            && header[18..20] == [0xbe, 0],
        "candidate cubin at {} is not an ELF64 NVIDIA CUDA executable",
        cubin.display()
    );
    ensure!(
        u32::from_le_bytes(header[20..24].try_into().unwrap()) == 1
            && u16::from_le_bytes(header[52..54].try_into().unwrap()) == ELF64_HEADER_LEN as u16,
        "candidate cubin at {} has a malformed ELF64 header",
        cubin.display()
    );
    let program_offset = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let section_offset = u64::from_le_bytes(header[40..48].try_into().unwrap());
    let program_entry_size = u16::from_le_bytes(header[54..56].try_into().unwrap()) as u64;
    let program_count = u16::from_le_bytes(header[56..58].try_into().unwrap()) as u64;
    let section_entry_size = u16::from_le_bytes(header[58..60].try_into().unwrap()) as u64;
    let section_count = u16::from_le_bytes(header[60..62].try_into().unwrap()) as u64;
    ensure!(
        program_offset >= ELF64_HEADER_LEN as u64
            && program_entry_size == 56
            && program_count > 0
            && section_offset >= ELF64_HEADER_LEN as u64
            && section_entry_size == 64
            && section_count > 0,
        "candidate cubin at {} has malformed ELF64 tables",
        cubin.display()
    );
    let program_end = program_entry_size
        .checked_mul(program_count)
        .and_then(|size| program_offset.checked_add(size));
    let section_end = section_entry_size
        .checked_mul(section_count)
        .and_then(|size| section_offset.checked_add(size));
    ensure!(
        program_end.is_some_and(|end| end <= metadata.len())
            && section_end.is_some_and(|end| end <= metadata.len()),
        "candidate cubin at {} has ELF64 tables outside the file",
        cubin.display()
    );
    Ok(())
}

fn candidate_tool(role: &str, path: &Path, identity: &ToolIdentity) -> CandidateToolDraft {
    CandidateToolDraft {
        role: role.into(),
        path: path.display().to_string(),
        version: identity.version.clone(),
        sha256: identity.sha256.clone(),
    }
}

fn candidate_stage(stage: &str, outcome: &str, detail: &str) -> CandidateStageDraft {
    CandidateStageDraft {
        stage: stage.into(),
        outcome: outcome.into(),
        detail: detail.into(),
    }
}

fn record_candidate_failure(
    repo_root: &Path,
    output_dir: &Path,
    paths: &CandidateArtifactPaths,
    draft: &mut CandidateProbeDraft,
    stage: &str,
    error: &anyhow::Error,
) -> Result<()> {
    draft
        .stages
        .push(candidate_stage(stage, "failed", &format!("{error:#}")));
    write_candidate_draft(repo_root, output_dir, paths, draft)
}

fn write_candidate_draft(
    repo_root: &Path,
    output_dir: &Path,
    paths: &CandidateArtifactPaths,
    draft: &mut CandidateProbeDraft,
) -> Result<()> {
    let mut artifacts = Vec::new();
    for (kind, path) in candidate_artifact_entries(paths) {
        match fs::symlink_metadata(path) {
            Ok(metadata) => {
                ensure!(
                    metadata.file_type().is_file(),
                    "candidate artifact {} is not a regular file",
                    path.display()
                );
                artifacts.push(candidate_artifact(repo_root, output_dir, kind, path)?);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("inspect artifact {}", path.display()));
            }
        }
    }
    draft.artifacts = artifacts;
    remove_if_present(&paths.manifest_temp)?;
    write_new_file(&paths.manifest_temp, pretty_json(draft)?)?;
    fs::rename(&paths.manifest_temp, &paths.manifest)
        .with_context(|| format!("write candidate draft {}", paths.manifest.display()))
}

fn candidate_artifact(
    repo_root: &Path,
    output_dir: &Path,
    kind: &str,
    path: &Path,
) -> Result<CandidateArtifactDraft> {
    ensure!(
        path.parent() == Some(output_dir),
        "candidate artifact escaped {}",
        output_dir.display()
    );
    let relative = path.strip_prefix(repo_root).with_context(|| {
        format!(
            "candidate artifact {} is outside the repository",
            path.display()
        )
    })?;
    Ok(CandidateArtifactDraft {
        kind: kind.into(),
        path: relative.display().to_string(),
        sha256: sha256_file(path)?,
        bytes: fs::metadata(path)?.len(),
    })
}

fn validate_probe_instructions(record: &CatalogIntrinsic, ptx: &str) -> Result<()> {
    ensure!(
        record.expected_ptx.matches(ptx)?,
        "probe PTX has no instruction matching `{}`",
        record.expected_ptx
    );
    if record.vote.is_some() {
        validate_register_and_immediate_forms(&record.expected_ptx, 2, "-1", ptx)?;
    }
    if record.warp_match.is_some() {
        validate_two_register_and_immediate_forms(&record.expected_ptx, 1, "7", 2, "-1", ptx)?;
    }
    if record.family == "elect" {
        validate_register_and_immediate_forms(&record.expected_ptx, 1, "-1", ptx)?;
    }
    if record.family == "counted_barrier" {
        validate_two_register_and_immediate_forms(&record.expected_ptx, 0, "1", 1, "32", ptx)?;
    }
    if record.warp_barrier.is_some() {
        validate_register_and_immediate_forms(&record.expected_ptx, 0, "-1", ptx)?;
    }
    if record
        .cp_async_copy
        .as_ref()
        .is_some_and(|copy| copy.source_size == CpAsyncSourceSize::Runtime)
    {
        validate_register_and_immediate_forms(&record.expected_ptx, 3, "3", ptx)?;
    }
    if let Some(shuffle) = &record.warp_shuffle {
        match shuffle.adapter {
            WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp => {
                validate_warp_shuffle_forms(&record.expected_ptx, shuffle.clamp, ptx)?;
            }
            WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble => {
                validate_wide_warp_shuffle_recipe(&record.expected_ptx, shuffle.clamp, ptx)?;
            }
        }
    }
    if record.packed_alu.is_some() || record.packed_conversion.is_some() {
        validate_exact_pure_instruction(&record.expected_ptx, ptx)?;
    }
    if let Some(mma) = &record.sparse_mma {
        let selectors: &[u32] = match mma.selector {
            SparseMmaSelector::ImmediateZeroThroughThree => &[0, 1, 2, 3],
            SparseMmaSelector::ImmediateZeroOrOne => &[0, 1],
            SparseMmaSelector::ImmediateZero => &[0],
        };
        validate_sparse_mma_selectors(&record.expected_ptx, selectors, ptx)?;
    }
    Ok(())
}

fn uses_typed_llvm_nvptx_lowering(record: &CatalogIntrinsic) -> bool {
    record.backend_lowerings.is_empty()
        || record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LlvmNvptx
                && lowering.mechanism == BackendLoweringMechanism::TypedNvvm
        })
}

fn validate_exact_pure_instruction(expected: &InstructionPattern, ptx: &str) -> Result<()> {
    let instructions = instructions_with_matching_head(ptx, expected)?;
    ensure!(
        instructions.len() == 1,
        "packed pure probe must contain exactly one `{}` instruction; found {}",
        expected
            .modifiers
            .iter()
            .fold(expected.mnemonic.clone(), |head, modifier| format!(
                "{head}.{modifier}"
            )),
        instructions.len()
    );
    ensure!(
        matching_instructions(ptx, expected)?.len() == 1,
        "packed pure probe instruction does not match `{expected}`"
    );
    ensure!(
        instructions[0].prefix.is_empty(),
        "packed pure probe instruction must be unguarded"
    );
    Ok(())
}

fn validate_sparse_mma_selectors(
    expected: &InstructionPattern,
    selectors: &[u32],
    ptx: &str,
) -> Result<()> {
    let selector = expected
        .operands
        .get(5)
        .context("sparse MMA selector operand is missing")?;
    ensure!(
        *selector == OperandPattern::Immediate,
        "sparse MMA selector must be an immediate operand"
    );
    let instructions = instructions_with_matching_head(ptx, expected)?;
    ensure!(
        instructions.len() == selectors.len(),
        "sparse MMA probe must contain exactly {} matching instructions; found {}",
        selectors.len(),
        instructions.len()
    );
    ensure!(
        instructions
            .iter()
            .all(|instruction| instruction.prefix.is_empty()),
        "sparse MMA probe instructions must be unguarded"
    );
    for selector in selectors {
        let mut pattern = expected.clone();
        pattern.operands[5] = OperandPattern::Exact {
            value: selector.to_string(),
        };
        ensure!(
            matching_instructions(ptx, &pattern)?.len() == 1,
            "sparse MMA probe PTX must contain exactly one selector {selector} form matching `{pattern}`"
        );
    }
    Ok(())
}

fn validate_warp_shuffle_forms(expected: &InstructionPattern, clamp: u32, ptx: &str) -> Result<()> {
    let clamp_operand = expected
        .operands
        .get(3)
        .context("shuffle probe clamp operand index is out of range")?;
    ensure!(
        matches!(clamp_operand, OperandPattern::Exact { value } if value == &clamp.to_string()),
        "shuffle probe clamp operand is not the exact catalog clamp {clamp}"
    );
    validate_two_register_and_immediate_forms(expected, 2, "1", 4, "-1", ptx)
}

fn validate_wide_warp_shuffle_recipe(
    expected: &InstructionPattern,
    clamp: u32,
    ptx: &str,
) -> Result<()> {
    let clamp = clamp.to_string();
    let low_operands = vec![
        OperandPattern::Exact { value: "lo".into() },
        OperandPattern::Exact { value: "lo".into() },
        OperandPattern::Register,
        OperandPattern::Exact {
            value: clamp.clone(),
        },
        OperandPattern::Register,
    ];
    ensure!(
        expected.operands == low_operands,
        "wide shuffle expected PTX must be `lo, lo, <register>, {clamp}, <register>`"
    );

    let all_shuffles = instructions_with_matching_head(ptx, expected)?;
    ensure!(
        all_shuffles.len() == 2,
        "wide shuffle probe must contain exactly two `{}` instructions; found {}",
        expected
            .modifiers
            .iter()
            .fold(expected.mnemonic.clone(), |head, modifier| format!(
                "{head}.{modifier}"
            )),
        all_shuffles.len()
    );

    let low = matching_instructions(ptx, expected)?;
    ensure!(
        low.len() == 1,
        "wide shuffle probe must contain exactly one low-half instruction matching `{expected}`"
    );
    let mut high_pattern = expected.clone();
    high_pattern.operands[0] = OperandPattern::Exact { value: "hi".into() };
    high_pattern.operands[1] = OperandPattern::Exact { value: "hi".into() };
    let high = matching_instructions(ptx, &high_pattern)?;
    ensure!(
        high.len() == 1,
        "wide shuffle probe must contain exactly one high-half instruction matching `{high_pattern}`"
    );
    ensure!(
        low[0].operands[2..=4] == high[0].operands[2..=4],
        "wide shuffle halves must use the same lane, clamp, and member mask"
    );

    let split_pattern = InstructionPattern {
        mnemonic: "mov".into(),
        modifiers: vec!["b64".into()],
        operands: vec![
            OperandPattern::Exact {
                value: "{lo, hi}".into(),
            },
            OperandPattern::Register,
        ],
    };
    let reassemble_pattern = InstructionPattern {
        mnemonic: "mov".into(),
        modifiers: vec!["b64".into()],
        operands: vec![
            OperandPattern::Register,
            OperandPattern::Exact {
                value: "{lo, hi}".into(),
            },
        ],
    };
    let split = matching_instructions(ptx, &split_pattern)?;
    let reassemble = matching_instructions(ptx, &reassemble_pattern)?;
    ensure!(
        split.len() == 1,
        "wide shuffle probe must contain exactly one split matching `{split_pattern}`"
    );
    ensure!(
        reassemble.len() == 1,
        "wide shuffle probe must contain exactly one reassembly matching `{reassemble_pattern}`"
    );
    ensure!(
        split[0].offset < low[0].offset
            && low[0].offset < high[0].offset
            && high[0].offset < reassemble[0].offset,
        "wide shuffle probe must split, shuffle low, shuffle high, then reassemble"
    );
    ensure!(
        [&split[0], &low[0], &high[0], &reassemble[0]]
            .iter()
            .all(|instruction| instruction.prefix.is_empty()),
        "wide shuffle recipe instructions must be unguarded"
    );
    ensure!(
        [
            (&split[0], &low[0]),
            (&low[0], &high[0]),
            (&high[0], &reassemble[0])
        ]
        .iter()
        .all(|(before, after)| ptx[before.end..after.offset].trim().is_empty()),
        "wide shuffle recipe instructions must be consecutive"
    );
    Ok(())
}

fn validate_register_and_immediate_forms(
    expected: &InstructionPattern,
    operand_index: usize,
    immediate: &str,
    ptx: &str,
) -> Result<()> {
    let mut register = expected.clone();
    let register_operand = register
        .operands
        .get_mut(operand_index)
        .context("probe register-or-immediate operand index is out of range")?;
    ensure!(
        *register_operand == OperandPattern::RegisterOrImmediate,
        "probe operand {operand_index} is not register-or-immediate"
    );
    *register_operand = OperandPattern::Register;

    let mut immediate_pattern = expected.clone();
    immediate_pattern.operands[operand_index] = OperandPattern::Exact {
        value: immediate.into(),
    };

    ensure!(
        register.matches(ptx)?,
        "probe PTX has no register form matching `{register}`"
    );
    ensure!(
        immediate_pattern.matches(ptx)?,
        "probe PTX has no immediate form matching `{immediate_pattern}`"
    );
    Ok(())
}

fn validate_two_register_and_immediate_forms(
    expected: &InstructionPattern,
    first_operand_index: usize,
    first_immediate: &str,
    second_operand_index: usize,
    second_immediate: &str,
    ptx: &str,
) -> Result<()> {
    ensure!(
        first_operand_index != second_operand_index,
        "probe register-or-immediate operand indices must be distinct"
    );
    for operand_index in [first_operand_index, second_operand_index] {
        let operand = expected
            .operands
            .get(operand_index)
            .context("probe register-or-immediate operand index is out of range")?;
        ensure!(
            *operand == OperandPattern::RegisterOrImmediate,
            "probe operand {operand_index} is not register-or-immediate"
        );
    }

    let combinations = [
        ("rr", OperandPattern::Register, OperandPattern::Register),
        (
            "ri",
            OperandPattern::Register,
            OperandPattern::Exact {
                value: second_immediate.into(),
            },
        ),
        (
            "ir",
            OperandPattern::Exact {
                value: first_immediate.into(),
            },
            OperandPattern::Register,
        ),
        (
            "ii",
            OperandPattern::Exact {
                value: first_immediate.into(),
            },
            OperandPattern::Exact {
                value: second_immediate.into(),
            },
        ),
    ];

    for (name, first, second) in combinations {
        let mut pattern = expected.clone();
        pattern.operands[first_operand_index] = first;
        pattern.operands[second_operand_index] = second;
        ensure!(
            pattern.matches(ptx)?,
            "probe PTX has no {name} form matching `{pattern}`"
        );
    }
    Ok(())
}

fn assert_intrinsic_declaration_canonicalizes(
    llc: &Path,
    input: &Path,
    output_dir: &Path,
    intrinsic_id: &str,
    record: &CatalogIntrinsic,
) -> Result<()> {
    let (llvm_as, llvm_dis) = sibling_llvm_tools(llc)?;
    let bitcode = output_dir.join(format!("{intrinsic_id}.bc"));
    let canonical = output_dir.join(format!("{intrinsic_id}.canonical.ll"));
    let status = Command::new(&llvm_as)
        .arg(input)
        .arg("-o")
        .arg(&bitcode)
        .status()
        .with_context(|| format!("run {}", llvm_as.display()))?;
    ensure!(status.success(), "llvm-as probe failed with {status}");
    let status = Command::new(&llvm_dis)
        .arg(&bitcode)
        .arg("-o")
        .arg(&canonical)
        .status()
        .with_context(|| format!("run {}", llvm_dis.display()))?;
    ensure!(status.success(), "llvm-dis probe failed with {status}");
    let canonical = fs::read_to_string(&canonical)?;
    assert_canonical_intrinsic_declaration(
        &canonical,
        record
            .llvm
            .as_ref()
            .context("LLVM-backed probe has no LLVM facts")?,
    )
}

fn sibling_llvm_tools(llc: &Path) -> Result<(PathBuf, PathBuf)> {
    let tool_dir = llc
        .parent()
        .context("selected llc has no containing tool directory")?;
    let llvm_as = tool_dir.join("llvm-as");
    let llvm_dis = tool_dir.join("llvm-dis");
    ensure!(
        llvm_as.is_file() && llvm_dis.is_file(),
        "selected LLVM toolchain omits llvm-as or llvm-dis"
    );
    Ok((llvm_as, llvm_dis))
}

fn assert_canonical_intrinsic_declaration(canonical: &str, llvm: &CatalogLlvm) -> Result<()> {
    let symbol = llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol);
    let symbol_marker = format!("@{symbol}(");
    let declaration = canonical
        .lines()
        .find(|line| {
            let line = line.trim_start();
            line.starts_with("declare ") && line.contains(&symbol_marker)
        })
        .with_context(|| format!("canonical module has no declaration for @{symbol}"))?;
    let declaration_prefix = declaration
        .split_once(&symbol_marker)
        .map(|(prefix, _)| prefix)
        .context("canonical intrinsic declaration has a malformed symbol")?;
    let arguments = declaration_arguments(declaration, &symbol_marker)?;
    let function_attributes = declaration_attribute_group(canonical, declaration)?;

    let mut no_memory = false;
    let mut argument_memory_only = false;
    let mut inaccessible_memory_only = false;
    let mut reads_memory = false;
    let mut writes_memory = false;
    let mut has_side_effects = false;

    for property in &llvm.properties {
        match property.as_str() {
            "IntrConvergent" => {
                require_attribute_token(function_attributes, "convergent", symbol, "function")?
            }
            "IntrNoCallback" => {
                require_attribute_token(function_attributes, "nocallback", symbol, "function")?
            }
            "IntrNoFree" => {
                require_attribute_token(function_attributes, "nofree", symbol, "function")?
            }
            "IntrSpeculatable" => {
                require_attribute_token(function_attributes, "speculatable", symbol, "function")?
            }
            "IntrWillReturn" => {
                require_attribute_token(function_attributes, "willreturn", symbol, "function")?
            }
            "IntrNoMem" => no_memory = true,
            "IntrArgMemOnly" => argument_memory_only = true,
            "IntrInaccessibleMemOnly" => inaccessible_memory_only = true,
            "IntrReadMem" => reads_memory = true,
            "IntrWriteMem" => writes_memory = true,
            "IntrHasSideEffects" => has_side_effects = true,
            "Commutative" | "IntrNoCreateUndefOrPoison" => {
                // These are TableGen selection semantics, not LLVM IR attributes.
            }
            "NoUndef<ret>" => {
                // Return attributes are asserted from the normalized result facts below.
                ensure!(
                    llvm.result_facts.no_undef,
                    "@{symbol} imported NoUndef return property disagrees with its normalized result facts"
                );
            }
            property if property.starts_with("Range<") => {
                let fields = property
                    .strip_prefix("Range<")
                    .and_then(|value| value.strip_suffix('>'))
                    .with_context(|| format!("malformed imported LLVM property {property:?}"))?
                    .split(',')
                    .collect::<Vec<_>>();
                ensure!(
                    fields.len() == 3,
                    "malformed imported LLVM property {property:?}"
                );
                if fields[0] == "ret" {
                    let range = llvm.result_facts.range.as_ref().with_context(|| {
                        format!(
                            "@{symbol} imported range property disagrees with its normalized result facts"
                        )
                    })?;
                    ensure!(
                        fields[1] == range.lower && fields[2] == range.upper_exclusive,
                        "malformed imported LLVM range property {property:?} on @{symbol}"
                    );
                } else {
                    let index = fields[0]
                        .strip_prefix("arg")
                        .with_context(|| {
                            format!("unsupported imported LLVM range property {property:?}")
                        })?
                        .parse::<usize>()
                        .with_context(|| {
                            format!("malformed imported LLVM property {property:?}")
                        })?;
                    let imported_type = llvm.arguments.get(index).with_context(|| {
                        format!("@{symbol} has no argument {index} required by {property}")
                    })?;
                    let width = imported_type
                        .strip_prefix('i')
                        .with_context(|| {
                            format!(
                                "@{symbol} has an argument range on unsupported type {imported_type}"
                            )
                        })?
                        .parse::<u32>()
                        .with_context(|| {
                            format!(
                                "@{symbol} has an argument range on malformed type {imported_type}"
                            )
                        })?;
                    let lower = canonical_integer_literal(fields[1], width)?;
                    let upper = canonical_integer_literal(fields[2], width)?;
                    let argument = arguments.get(index).with_context(|| {
                        format!("@{symbol} has no argument {index} required by {property}")
                    })?;
                    require_attribute_fragment(
                        argument,
                        &format!("range(i{width} {lower}, {upper})"),
                        symbol,
                        "argument",
                    )?;
                }
            }
            property if property.starts_with("NoCapture<") => {
                let index = property_argument_index(property, "NoCapture")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_fragment(argument, "captures(none)", symbol, "argument")?;
            }
            property if property.starts_with("NoAlias<") => {
                let index = property_argument_index(property, "NoAlias")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "noalias", symbol, "argument")?;
            }
            property if property.starts_with("ReadOnly<") => {
                let index = property_argument_index(property, "ReadOnly")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "readonly", symbol, "argument")?;
            }
            property if property.starts_with("WriteOnly<") => {
                let index = property_argument_index(property, "WriteOnly")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "writeonly", symbol, "argument")?;
            }
            property if property.starts_with("ImmArg<") => {
                let index = property_argument_index(property, "ImmArg")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "immarg", symbol, "argument")?;
            }
            property if property.starts_with("NoUndef<") => {
                let index = property_argument_index(property, "NoUndef")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "noundef", symbol, "argument")?;
            }
            unsupported => anyhow::bail!(
                "cannot verify unsupported imported LLVM property {unsupported:?} on @{symbol}"
            ),
        }
    }

    // TableGen's `IntrHasSideEffects` deliberately suppresses the otherwise
    // inferred `memory(none)` attribute: the instruction has an observable
    // non-memory effect and must not become removable as a readnone call.
    // Other imported memory effects still have to canonicalize exactly.
    let sideeffect_only = has_side_effects
        && no_memory
        && !argument_memory_only
        && !inaccessible_memory_only
        && !reads_memory
        && !writes_memory;
    let memory = if sideeffect_only {
        None
    } else {
        canonical_memory_attribute(
            no_memory,
            argument_memory_only,
            inaccessible_memory_only,
            reads_memory,
            writes_memory,
        )?
    };
    if has_side_effects && !sideeffect_only {
        let memory = memory.as_deref().with_context(|| {
            format!(
                "@{symbol} IntrHasSideEffects requires a concrete non-`memory(none)` canonical memory effect"
            )
        })?;
        ensure!(
            memory != "memory(none)",
            "@{symbol} IntrHasSideEffects requires a concrete non-`memory(none)` canonical memory effect"
        );
    }
    if let Some(memory) = memory {
        require_attribute_fragment(function_attributes, &memory, symbol, "function")?;
    }

    if llvm.result_facts.no_undef {
        require_attribute_token(declaration_prefix, "noundef", symbol, "return")?;
    }
    if let Some(range) = &llvm.result_facts.range {
        ensure!(
            llvm.results.len() == 1,
            "@{symbol} has a return range but not exactly one imported result"
        );
        let width = llvm.results[0]
            .strip_prefix('i')
            .with_context(|| {
                format!(
                    "@{symbol} has a return range on unsupported result type {}",
                    llvm.results[0]
                )
            })?
            .parse::<u32>()
            .with_context(|| {
                format!(
                    "@{symbol} has a return range on malformed result type {}",
                    llvm.results[0]
                )
            })?;
        let lower = canonical_integer_literal(&range.lower, width)?;
        let upper = canonical_integer_literal(&range.upper_exclusive, width)?;
        require_attribute_fragment(
            declaration_prefix,
            &format!("range(i{width} {lower}, {upper})"),
            symbol,
            "return",
        )?;
    }
    Ok(())
}

fn declaration_attribute_group<'a>(canonical: &'a str, declaration: &str) -> Result<&'a str> {
    let Some(group) = declaration
        .split_ascii_whitespace()
        .rev()
        .find(|token| token.starts_with('#'))
    else {
        return Ok("");
    };
    let prefix = format!("attributes {group} = ");
    canonical
        .lines()
        .find_map(|line| line.trim_start().strip_prefix(&prefix))
        .with_context(|| format!("canonical intrinsic declaration references missing {group}"))
}

fn declaration_arguments<'a>(declaration: &'a str, symbol_marker: &str) -> Result<Vec<&'a str>> {
    let start = declaration
        .find(symbol_marker)
        .map(|offset| offset + symbol_marker.len())
        .context("canonical intrinsic declaration has no argument list")?;
    let arguments = &declaration[start..];
    let mut parentheses = 0_u32;
    let mut braces = 0_u32;
    let mut brackets = 0_u32;
    let mut angles = 0_u32;
    let mut argument_start = 0;
    let mut split = Vec::new();

    for (offset, character) in arguments.char_indices() {
        match character {
            '(' => parentheses += 1,
            ')' if parentheses == 0 => {
                let argument = arguments[argument_start..offset].trim();
                if !argument.is_empty() {
                    split.push(argument);
                }
                return Ok(split);
            }
            ')' => parentheses -= 1,
            '{' => braces += 1,
            '}' => braces = braces.saturating_sub(1),
            '[' => brackets += 1,
            ']' => brackets = brackets.saturating_sub(1),
            '<' => angles += 1,
            '>' => angles = angles.saturating_sub(1),
            ',' if parentheses == 0 && braces == 0 && brackets == 0 && angles == 0 => {
                split.push(arguments[argument_start..offset].trim());
                argument_start = offset + character.len_utf8();
            }
            _ => {}
        }
    }
    anyhow::bail!("canonical intrinsic declaration has an unterminated argument list")
}

fn property_argument_index(property: &str, property_name: &str) -> Result<usize> {
    let prefix = format!("{property_name}<arg");
    property
        .strip_prefix(&prefix)
        .and_then(|index| index.strip_suffix('>'))
        .with_context(|| format!("malformed imported LLVM property {property:?}"))?
        .parse::<usize>()
        .with_context(|| format!("malformed imported LLVM property {property:?}"))
}

fn canonical_memory_attribute(
    no_memory: bool,
    argument_memory_only: bool,
    inaccessible_memory_only: bool,
    reads_memory: bool,
    writes_memory: bool,
) -> Result<Option<String>> {
    ensure!(
        !(argument_memory_only && inaccessible_memory_only),
        "imported LLVM properties specify two incompatible memory locations"
    );
    if no_memory {
        ensure!(
            !argument_memory_only && !inaccessible_memory_only && !reads_memory && !writes_memory,
            "IntrNoMem is combined with another imported LLVM memory property"
        );
        return Ok(Some("memory(none)".into()));
    }

    let access = match (reads_memory, writes_memory) {
        (true, false) => "read",
        (false, true) => "write",
        _ => "readwrite",
    };
    let location = if argument_memory_only {
        Some("argmem")
    } else if inaccessible_memory_only {
        Some("inaccessiblemem")
    } else {
        None
    };
    match location {
        Some(location) => Ok(Some(format!("memory({location}: {access})"))),
        None if reads_memory && !writes_memory => Ok(Some("memory(read)".into())),
        None if writes_memory && !reads_memory => Ok(Some("memory(write)".into())),
        // Read-write access to unrestricted memory is LLVM's default and has no
        // canonical attribute to assert.
        None => Ok(None),
    }
}

fn canonical_integer_literal(value: &str, width: u32) -> Result<String> {
    ensure!(
        (1..=64).contains(&width),
        "cannot verify a canonical range for unsupported i{width}"
    );
    if value.starts_with('-') {
        let signed = value
            .parse::<i128>()
            .with_context(|| format!("invalid signed LLVM range bound {value:?}"))?;
        let minimum = -(1_i128 << (width - 1));
        ensure!(
            signed >= minimum,
            "LLVM range bound {value} does not fit in i{width}"
        );
        return Ok(signed.to_string());
    }

    let unsigned = value
        .parse::<u128>()
        .with_context(|| format!("invalid unsigned LLVM range bound {value:?}"))?;
    let modulus = 1_u128 << width;
    ensure!(
        unsigned < modulus,
        "LLVM range bound {value} does not fit in i{width}"
    );
    let sign_bit = 1_u128 << (width - 1);
    if unsigned < sign_bit {
        Ok(unsigned.to_string())
    } else {
        Ok((unsigned as i128 - modulus as i128).to_string())
    }
}

fn require_attribute_token(text: &str, required: &str, symbol: &str, position: &str) -> Result<()> {
    let present = text.split_ascii_whitespace().any(|token| {
        token.trim_matches(|character| matches!(character, '{' | '}' | ',')) == required
    });
    ensure!(
        present,
        "canonicalized @{symbol} {position} attributes are missing {required:?}"
    );
    Ok(())
}

fn require_attribute_fragment(
    text: &str,
    required: &str,
    symbol: &str,
    position: &str,
) -> Result<()> {
    ensure!(
        text.contains(required),
        "canonicalized @{symbol} {position} attributes are missing {required:?}"
    );
    Ok(())
}

fn assemble_probe_ptx(
    record: &crate::model::CatalogIntrinsic,
    ptx: &Path,
    output_dir: &Path,
    intrinsic_id: &str,
) -> Result<()> {
    let stage = record
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .and_then(|lowering| {
            lowering
                .stages
                .iter()
                .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
        })
        .context("selected LLVM evidence has no PTX-assembly stage")?;
    let recorded_tool = stage
        .tool_path
        .as_deref()
        .context("PTX-assembly stage has no tool path")?;
    let tool = resolve_evidence_tool(recorded_tool)?;
    let expected_sha256 = stage
        .tool_sha256
        .as_deref()
        .context("PTX-assembly stage has no tool SHA-256")?;
    ensure!(
        sha256_file(&tool)? == expected_sha256,
        "ptxas binary does not match selected evidence"
    );
    let cubin = output_dir.join(format!("{intrinsic_id}.cubin"));
    let architecture = stage
        .targets
        .iter()
        .find(|target| target.starts_with("sm_"))
        .context("PTX-assembly evidence has no sm_NN target")?;
    let status = Command::new(&tool)
        .arg(format!("-arch={architecture}"))
        .arg(ptx)
        .arg("-o")
        .arg(&cubin)
        .status()
        .with_context(|| format!("run {}", tool.display()))?;
    ensure!(status.success(), "ptxas probe failed with {status}");
    println!(
        "terminal PTX assembly revalidated with {} for {}",
        tool.display(),
        architecture
    );
    Ok(())
}

fn resolve_evidence_tool(recorded: &str) -> Result<PathBuf> {
    resolve_evidence_tool_with_path(recorded, std::env::var_os("PATH").as_deref())
}

fn resolve_evidence_tool_with_path(
    recorded: &str,
    search_path: Option<&std::ffi::OsStr>,
) -> Result<PathBuf> {
    let recorded = PathBuf::from(recorded);
    let has_directory = recorded
        .parent()
        .is_some_and(|parent| !parent.as_os_str().is_empty());
    if recorded.is_absolute() || has_directory {
        ensure!(
            recorded.is_file(),
            "recorded evidence tool does not exist: {}",
            recorded.display()
        );
        return Ok(recorded);
    }

    let search_path = search_path.context("PATH is unset; cannot resolve evidence tool")?;
    std::env::split_paths(search_path)
        .map(|directory| directory.join(&recorded))
        .find(|candidate| candidate.is_file())
        .with_context(|| {
            format!(
                "evidence tool {:?} was not found on PATH",
                recorded.as_os_str()
            )
        })
}

fn llc_identity(llc: &Path) -> Result<ToolIdentity> {
    let version = Command::new(llc)
        .arg("--version")
        .output()
        .with_context(|| format!("query {} --version", llc.display()))?;
    ensure!(
        version.status.success(),
        "{} --version failed",
        llc.display()
    );
    let version = String::from_utf8_lossy(&version.stdout)
        .lines()
        .find(|line| line.contains("LLVM version"))
        .context("llc --version did not report an LLVM version")?
        .trim()
        .to_owned();
    Ok(ToolIdentity {
        version,
        sha256: sha256_file(llc)?,
    })
}

fn ptxas_identity(ptxas: &Path) -> Result<ToolIdentity> {
    let output = Command::new(ptxas)
        .arg("--version")
        .output()
        .with_context(|| format!("query {} --version", ptxas.display()))?;
    ensure!(
        output.status.success(),
        "{} --version failed",
        ptxas.display()
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = parse_ptxas_version(&stdout, &stderr)?;
    Ok(ToolIdentity {
        version,
        sha256: sha256_file(ptxas)?,
    })
}

fn parse_ptxas_version(stdout: &str, stderr: &str) -> Result<String> {
    let lines = stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    ensure!(
        lines.contains(&"ptxas: NVIDIA (R) Ptx optimizing assembler"),
        "ptxas --version did not report the NVIDIA PTX optimizing assembler banner"
    );
    let version = lines
        .iter()
        .copied()
        .find(|line| line.starts_with("Cuda compilation tools, release ") && line.contains(", V"))
        .context("ptxas --version did not report a CUDA release and version")?;
    Ok(version.to_owned())
}

fn validate_backend_identity(
    expected_version: &str,
    expected_commit: &RustcCommit,
    actual: &ProbeBackendIdentity,
) -> Result<()> {
    let ProbeBackendIdentity::SelectedEvidence(actual) = actual else {
        return Ok(());
    };
    ensure!(
        actual.tool.version == expected_version,
        "rust-toolchain llc version mismatch: selected evidence records {expected_version:?}, found {:?}; use an explicit `--llc` only for a comparison probe",
        actual.tool.version
    );
    ensure!(
        actual.rustc_commit == *expected_commit,
        "rust-toolchain commit mismatch: upstream.lock records {expected_commit}, found {}; use an explicit `--llc` only for a comparison probe",
        actual.rustc_commit
    );
    Ok(())
}

fn rust_toolchain_llc() -> Result<(PathBuf, RustcCommit)> {
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .context("query rustc sysroot")?;
    ensure!(sysroot.status.success(), "rustc --print sysroot failed");
    let verbose = Command::new("rustc")
        .arg("-vV")
        .output()
        .context("query rustc host")?;
    ensure!(verbose.status.success(), "rustc -vV failed");
    let verbose = String::from_utf8_lossy(&verbose.stdout);
    let (host, commit) = parse_rustc_verbose(&verbose)?;
    let path = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/llc");
    ensure!(
        path.is_file(),
        "rust toolchain has no llc at {}",
        path.display()
    );
    Ok((path, commit))
}

fn parse_rustc_verbose(stdout: &str) -> Result<(String, RustcCommit)> {
    let host = stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .context("rustc -vV did not report a host")?
        .to_owned();
    let commit = stdout
        .lines()
        .find_map(|line| line.strip_prefix("commit-hash: "))
        .context("rustc -vV did not report a commit-hash")?
        .parse()
        .map_err(anyhow::Error::msg)
        .context("parse rustc -vV commit-hash")?;
    Ok((host, commit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_target_uses_default_when_it_meets_the_floor() {
        assert_eq!(derived_ptx_feature("sm_90a", 78).unwrap(), (None, 80));
        assert_eq!(derived_ptx_feature("sm_100a", 86).unwrap(), (None, 86));
        assert_eq!(derived_ptx_feature("sm_100f", 86).unwrap(), (None, 88));
        assert_eq!(derived_ptx_feature("sm_103f", 86).unwrap(), (None, 88));
        assert_eq!(derived_ptx_feature("sm_110a", 86).unwrap(), (None, 90));
        assert_eq!(derived_ptx_feature("sm_120a", 86).unwrap(), (None, 87));
        assert_eq!(derived_ptx_feature("sm_121f", 87).unwrap(), (None, 88));
    }

    #[test]
    fn derived_target_rejects_canonical_unknown_architecture() {
        let error = derived_ptx_feature("sm_999a", 90).unwrap_err();
        assert!(error.to_string().contains("sm_999a"), "{error:#}");
        assert!(
            error.to_string().contains("no recorded PTX ISA floor"),
            "{error:#}"
        );
    }

    #[test]
    fn derived_target_raises_a_default_below_the_instruction_floor() {
        assert_eq!(
            derived_ptx_feature("sm_80", 88).unwrap(),
            (Some("+ptx88".into()), 88)
        );
    }

    #[test]
    fn derived_target_rounds_like_production() {
        assert_eq!(
            derived_ptx_feature("sm_80", 74).unwrap(),
            (Some("+ptx78".into()), 78)
        );
        assert_eq!(
            derived_ptx_feature("sm_87", 74).unwrap(),
            (Some("+ptx78".into()), 78)
        );
        assert_eq!(
            derived_ptx_feature("sm_75", 63).unwrap(),
            (Some("+ptx65".into()), 65)
        );
    }
    use crate::model::{
        CatalogHalfOpenRange, CatalogHardwareAlternative, CatalogHardwareTarget,
        CatalogLlvmResultFacts,
    };

    #[cfg(unix)]
    struct CandidateToolTestDir(PathBuf);

    #[cfg(unix)]
    impl Drop for CandidateToolTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn candidate_tool_test_dir() -> CandidateToolTestDir {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cuda-intrinsics-probe-tool-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        CandidateToolTestDir(path)
    }

    #[cfg(unix)]
    fn write_fake_tool(root: &Path, name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = root.join(name);
        fs::write(&path, script).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[test]
    fn vote_probe_requires_register_and_negative_one_mask_forms() {
        let expected = InstructionPattern::new(
            "vote",
            &["sync", "all", "pred"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let register = "vote.sync.all.pred %p1, %p2, %r3;";
        let immediate = "vote.sync.all.pred %p4, %p5, -1;";

        validate_register_and_immediate_forms(
            &expected,
            2,
            "-1",
            &format!("{register}\n{immediate}"),
        )
        .unwrap();

        let error =
            validate_register_and_immediate_forms(&expected, 2, "-1", register).unwrap_err();
        assert!(error.to_string().contains("no immediate form"));

        let error =
            validate_register_and_immediate_forms(&expected, 2, "-1", immediate).unwrap_err();
        assert!(error.to_string().contains("no register form"));
    }

    #[test]
    fn warp_barrier_probe_requires_register_and_negative_one_mask_forms() {
        let expected = InstructionPattern::new(
            "bar",
            &["warp", "sync"],
            vec![OperandPattern::RegisterOrImmediate],
        );
        let register = "bar.warp.sync %r1;";
        let immediate = "bar.warp.sync -1;";

        validate_register_and_immediate_forms(
            &expected,
            0,
            "-1",
            &format!("{register}\n{immediate}"),
        )
        .unwrap();

        let error =
            validate_register_and_immediate_forms(&expected, 0, "-1", register).unwrap_err();
        assert!(error.to_string().contains("no immediate form"));

        let error =
            validate_register_and_immediate_forms(&expected, 0, "-1", immediate).unwrap_err();
        assert!(error.to_string().contains("no register form"));
    }

    #[test]
    fn warp_match_probe_requires_every_register_and_immediate_combination() {
        let expected = InstructionPattern::new(
            "match",
            &["any", "sync", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let forms = [
            ("rr", "match.any.sync.b32 %r1, %r2, %r3;"),
            ("ri", "match.any.sync.b32 %r4, %r5, -1;"),
            ("ir", "match.any.sync.b32 %r6, 7, %r7;"),
            ("ii", "match.any.sync.b32 %r8, 7, -1;"),
        ];
        let complete = forms
            .iter()
            .map(|(_, instruction)| *instruction)
            .collect::<Vec<_>>()
            .join("\n");

        validate_two_register_and_immediate_forms(&expected, 1, "7", 2, "-1", &complete).unwrap();

        for (missing_index, (name, _)) in forms.iter().enumerate() {
            let incomplete = forms
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != missing_index)
                .map(|(_, (_, instruction))| *instruction)
                .collect::<Vec<_>>()
                .join("\n");
            let error =
                validate_two_register_and_immediate_forms(&expected, 1, "7", 2, "-1", &incomplete)
                    .unwrap_err();
            assert!(
                error.to_string().contains(&format!("no {name} form")),
                "{error:#}"
            );
        }
    }

    #[test]
    fn warp_shuffle_probe_requires_lane_mask_forms_and_exact_clamp() {
        let expected = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let forms = [
            ("rr", "shfl.sync.idx.b32 %r1, %r2, %r3, 31, %r4;"),
            ("ri", "shfl.sync.idx.b32 %r5, %r6, %r7, 31, -1;"),
            ("ir", "shfl.sync.idx.b32 %r8, %r9, 1, 31, %r10;"),
            ("ii", "shfl.sync.idx.b32 %r11, %r12, 1, 31, -1;"),
        ];
        let complete = forms
            .iter()
            .map(|(_, instruction)| *instruction)
            .collect::<Vec<_>>()
            .join("\n");

        validate_warp_shuffle_forms(&expected, 31, &complete).unwrap();

        for (missing_index, (name, _)) in forms.iter().enumerate() {
            let incomplete = forms
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != missing_index)
                .map(|(_, (_, instruction))| *instruction)
                .collect::<Vec<_>>()
                .join("\n");
            let error = validate_warp_shuffle_forms(&expected, 31, &incomplete).unwrap_err();
            assert!(
                error.to_string().contains(&format!("no {name} form")),
                "{error:#}"
            );
        }

        let wrong_clamp = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::Exact { value: "0".into() },
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let error = validate_warp_shuffle_forms(&wrong_clamp, 31, &complete).unwrap_err();
        assert!(error.to_string().contains("exact catalog clamp 31"));
    }

    fn wide_shuffle_pattern() -> InstructionPattern {
        InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Register,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::Register,
            ],
        )
    }

    const WIDE_SHUFFLE_PTX: &str = r#"
        mov.b64 {lo, hi}, %rd1;
        shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;
        shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;
        mov.b64 %rd2, {lo, hi};
    "#;

    const WIDE_SHUFFLE_INLINE_BLOCK: &str = "{ .reg .b32 lo; .reg .b32 hi; mov.b64 {lo, hi}, %rd1; shfl.sync.idx.b32 lo, lo, %r1, 31, %r2; shfl.sync.idx.b32 hi, hi, %r1, 31, %r2; mov.b64 %rd2, {lo, hi}; }";

    #[test]
    fn wide_warp_shuffle_probe_requires_the_exact_two_half_recipe() {
        let expected = wide_shuffle_pattern();
        validate_wide_warp_shuffle_recipe(&expected, 31, WIDE_SHUFFLE_PTX).unwrap();
        validate_wide_warp_shuffle_recipe(&expected, 31, WIDE_SHUFFLE_INLINE_BLOCK).unwrap();

        let cases = [
            (
                "missing high half",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "",
                ),
                "exactly two",
            ),
            (
                "reversed halves",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;\n        shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;\n        shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;",
                ),
                "split, shuffle low, shuffle high",
            ),
            (
                "different lane",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "shfl.sync.idx.b32 hi, hi, %r3, 31, %r2;",
                ),
                "same lane, clamp, and member mask",
            ),
            (
                "different mask",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r3;",
                ),
                "same lane, clamp, and member mask",
            ),
            (
                "wrong clamp",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "shfl.sync.idx.b32 hi, hi, %r1, 30, %r2;",
                ),
                "high-half instruction",
            ),
            (
                "missing split",
                WIDE_SHUFFLE_PTX.replace("mov.b64 {lo, hi}, %rd1;", ""),
                "exactly one split",
            ),
            (
                "missing reassembly",
                WIDE_SHUFFLE_PTX.replace("mov.b64 %rd2, {lo, hi};", ""),
                "exactly one reassembly",
            ),
            (
                "duplicate pair",
                WIDE_SHUFFLE_PTX.replace(
                    "mov.b64 %rd2, {lo, hi};",
                    "shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;\n        shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;\n        mov.b64 %rd2, {lo, hi};",
                ),
                "exactly two",
            ),
            (
                "predicated low half",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;",
                    "@%p1 shfl.sync.idx.b32 lo, lo, %r1, 31, %r2;",
                ),
                "must be unguarded",
            ),
            (
                "intervening instruction",
                WIDE_SHUFFLE_PTX.replace(
                    "shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                    "mov.u32 %r9, %r9;\n        shfl.sync.idx.b32 hi, hi, %r1, 31, %r2;",
                ),
                "must be consecutive",
            ),
        ];

        for (name, ptx, message) in cases {
            let error = validate_wide_warp_shuffle_recipe(&expected, 31, &ptx).unwrap_err();
            assert!(error.to_string().contains(message), "{name}: {error:#}");
        }

        let wrong_policy = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::Register,
            ],
        );
        let error =
            validate_wide_warp_shuffle_recipe(&wrong_policy, 31, WIDE_SHUFFLE_PTX).unwrap_err();
        assert!(error.to_string().contains("expected PTX must be"));
    }

    fn commit(value: &str) -> RustcCommit {
        value.parse().unwrap()
    }

    fn tool_identity() -> ToolIdentity {
        ToolIdentity {
            version: "LLVM version 22.1.2-test".into(),
            sha256: "abc123".into(),
        }
    }

    fn selected_identity(commit_hash: &str) -> ProbeBackendIdentity {
        ProbeBackendIdentity::SelectedEvidence(SelectedBackendIdentity {
            tool: tool_identity(),
            rustc_commit: commit(commit_hash),
        })
    }

    #[test]
    fn selected_probe_accepts_same_version_and_commit_with_different_llc_sha256() {
        validate_backend_identity(
            "LLVM version 22.1.2-test",
            &commit("1111111111111111111111111111111111111111"),
            &ProbeBackendIdentity::SelectedEvidence(SelectedBackendIdentity {
                tool: ToolIdentity {
                    sha256: "different".into(),
                    ..tool_identity()
                },
                rustc_commit: commit("1111111111111111111111111111111111111111"),
            }),
        )
        .unwrap();
    }

    #[test]
    fn selected_probe_rejects_different_commit() {
        let commit_error = validate_backend_identity(
            "LLVM version 22.1.2-test",
            &commit("2222222222222222222222222222222222222222"),
            &selected_identity("1111111111111111111111111111111111111111"),
        )
        .unwrap_err();
        assert!(commit_error.to_string().contains("commit"));
    }

    #[test]
    fn selected_probe_rejects_different_version() {
        let version_error = validate_backend_identity(
            "LLVM version 21",
            &commit("1111111111111111111111111111111111111111"),
            &selected_identity("1111111111111111111111111111111111111111"),
        )
        .unwrap_err();
        assert!(version_error.to_string().contains("version mismatch"));
    }

    #[test]
    fn all_probe_preflight_checks_every_backend_identity() {
        let error = validate_backend_identities(
            &selected_identity("1111111111111111111111111111111111111111"),
            &commit("1111111111111111111111111111111111111111"),
            [
                BackendExpectation {
                    intrinsic_id: "first",
                    version: "LLVM version 22.1.2-test",
                },
                BackendExpectation {
                    intrinsic_id: "last",
                    version: "LLVM version 21",
                },
            ],
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("validate probe backend for last"));
        assert!(message.contains("version mismatch"));
    }

    #[test]
    fn explicit_probe_is_always_comparison_only() {
        validate_backend_identity(
            "different",
            &commit("2222222222222222222222222222222222222222"),
            &ProbeBackendIdentity::Comparison(tool_identity()),
        )
        .unwrap();
    }

    fn llvm_facts(
        symbol: &str,
        resolved_symbol: Option<&str>,
        arguments: &[&str],
        results: &[&str],
        properties: &[&str],
        no_undef: bool,
        range: Option<(&str, &str)>,
    ) -> CatalogLlvm {
        CatalogLlvm {
            symbol: symbol.into(),
            resolved_symbol: resolved_symbol.map(str::to_owned),
            arguments: arguments.iter().map(|value| (*value).into()).collect(),
            results: results.iter().map(|value| (*value).into()).collect(),
            properties: properties.iter().map(|value| (*value).into()).collect(),
            result_facts: CatalogLlvmResultFacts {
                no_undef,
                range: range.map(|(lower, upper_exclusive)| CatalogHalfOpenRange {
                    lower: lower.into(),
                    upper_exclusive: upper_exclusive.into(),
                }),
            },
        }
    }

    #[test]
    fn verifies_lane_id_result_and_function_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.read.ptx.sreg.laneid",
            None,
            &[],
            &["i32"],
            &[
                "IntrNoMem",
                "IntrSpeculatable",
                "NoUndef<ret>",
                "Range<ret,0,32>",
            ],
            true,
            Some(("0", "32")),
        );
        let canonical = r#"
declare noundef range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.laneid() #0
attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_redux_convergence_callback_and_inaccessible_memory() {
        let llvm = llvm_facts(
            "llvm.nvvm.redux.sync.add",
            None,
            &["i32", "i32"],
            &["i32"],
            &[
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #0
attributes #0 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_timer_lifetime_and_return_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.read.ptx.sreg.clock",
            None,
            &[],
            &["i32"],
            &[
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
                "IntrNoFree",
                "IntrWillReturn",
                "NoUndef<ret>",
            ],
            true,
            None,
        );
        let canonical = r#"
declare noundef i32 @llvm.nvvm.read.ptx.sreg.clock() #0
attributes #0 = { nocallback nofree willreturn memory(inaccessiblemem: readwrite) }
"#;
        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();

        let missing_nofree = r#"
declare noundef i32 @llvm.nvvm.read.ptx.sreg.clock() #0
attributes #0 = { nocallback willreturn memory(inaccessiblemem: readwrite) }
"#;
        assert!(assert_canonical_intrinsic_declaration(missing_nofree, &llvm).is_err());
    }

    #[test]
    fn verifies_side_effects_have_a_concrete_memory_effect() {
        let llvm = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &[
                "IntrConvergent",
                "IntrHasSideEffects",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.activemask() #0
attributes #0 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn side_effects_require_memory_unless_tablegen_marks_them_no_memory() {
        let llvm = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &["IntrHasSideEffects"],
            false,
            None,
        );
        let error =
            assert_canonical_intrinsic_declaration("declare i32 @llvm.nvvm.activemask()\n", &llvm)
                .unwrap_err();
        assert!(error.to_string().contains("concrete non-`memory(none)`"));

        let no_memory = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &["IntrHasSideEffects", "IntrNoMem"],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.activemask() #0
attributes #0 = { nounwind }
"#;
        assert_canonical_intrinsic_declaration(canonical, &no_memory).unwrap();
    }

    #[test]
    fn verifies_sync_threads_convergence_and_callback_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.barrier.cta.sync.aligned.all",
            None,
            &["i32"],
            &[],
            &["IntrConvergent", "IntrNoCallback"],
            false,
            None,
        );
        let canonical = r#"
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #0
attributes #0 = { convergent nocallback nounwind }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_dp2a_immediate_argument_attribute() {
        let llvm = llvm_facts(
            "llvm.nvvm.idp2a.s.s",
            None,
            &["i32", "i32", "i1", "i32"],
            &["i32"],
            &["ImmArg<arg2>", "IntrNoMem", "IntrSpeculatable"],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.idp2a.s.s(i32, i32, i1 immarg, i32) #0
attributes #0 = { speculatable memory(none) }
"#;
        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();

        let missing_immarg = r#"
declare i32 @llvm.nvvm.idp2a.s.s(i32, i32, i1, i32) #0
attributes #0 = { speculatable memory(none) }
"#;
        let error = assert_canonical_intrinsic_declaration(missing_immarg, &llvm).unwrap_err();
        assert!(error.to_string().contains("missing \"immarg\""));
    }

    #[test]
    fn verifies_argument_range_attribute() {
        let llvm = llvm_facts(
            "llvm.nvvm.test.range",
            None,
            &["i32", "i32"],
            &[],
            &["Range<arg1,0,3>"],
            false,
            None,
        );
        let canonical = "declare void @llvm.nvvm.test.range(i32, i32 range(i32 0, 3))\n";
        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();

        let missing = "declare void @llvm.nvvm.test.range(i32, i32)\n";
        let error = assert_canonical_intrinsic_declaration(missing, &llvm).unwrap_err();
        assert!(error.to_string().contains("range(i32 0, 3)"));
    }

    #[test]
    fn retains_ldmatrix_function_and_argument_requirements() {
        let llvm = llvm_facts(
            "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
            Some("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3"),
            &["anyptr"],
            &["i32", "i32", "i32", "i32"],
            &[
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrReadMem",
                "NoCapture<arg0>",
                "ReadOnly<arg0>",
            ],
            false,
            None,
        );
        let canonical = r#"
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) readonly captures(none)) #0
attributes #0 = { convergent nocallback nounwind memory(argmem: read) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn canonicalizes_unsigned_range_bounds_as_llvm_signed_literals() {
        let llvm = llvm_facts(
            "llvm.nvvm.read.ptx.sreg.nctaid.x",
            None,
            &[],
            &["i32"],
            &[
                "IntrNoMem",
                "IntrSpeculatable",
                "NoUndef<ret>",
                "Range<ret,1,2147483648>",
            ],
            true,
            Some(("1", "2147483648")),
        );
        let canonical = r#"
declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
attributes #0 = { speculatable memory(none) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn fails_when_a_required_attribute_is_only_mentioned_outside_the_declaration() {
        let llvm = llvm_facts(
            "llvm.nvvm.redux.sync.add",
            None,
            &["i32", "i32"],
            &["i32"],
            &["IntrConvergent", "IntrInaccessibleMemOnly"],
            false,
            None,
        );
        let canonical = r#"
; convergent memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #0
attributes #0 = { nounwind }
"#;

        let error = assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap_err();
        assert!(error.to_string().contains("missing \"convergent\""));
    }

    #[test]
    fn unsupported_imported_properties_fail_closed() {
        let llvm = llvm_facts(
            "llvm.nvvm.test",
            None,
            &[],
            &["i32"],
            &["UnmodeledProperty"],
            false,
            None,
        );
        let canonical = "declare i32 @llvm.nvvm.test()\n";

        let error = assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported imported LLVM property")
        );
    }

    #[test]
    fn accepts_known_tablegen_semantics_without_ir_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.test",
            None,
            &["f64", "f64"],
            &["f64"],
            &[
                "Commutative",
                "IntrNoCreateUndefOrPoison",
                "IntrNoMem",
                "IntrSpeculatable",
            ],
            false,
            None,
        );
        let canonical = r#"
declare double @llvm.nvvm.test(double, double) #0
attributes #0 = { speculatable memory(none) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn packed_pure_probe_requires_one_exact_unguarded_instruction() {
        let expected =
            InstructionPattern::new("fma", &["rn", "bf16x2"], vec![OperandPattern::Register; 4]);
        let exact = "fma.rn.bf16x2 %r1, %r2, %r3, %r4;";
        validate_exact_pure_instruction(&expected, exact).unwrap();

        for invalid in [
            format!("{exact}\n{exact}"),
            format!("@%p0 {exact}"),
            "fma.rn.bf16x2 %r1, %r2, %r3;".into(),
            "fma.rn.relu.bf16x2 %r1, %r2, %r3, %r4;".into(),
        ] {
            assert!(
                validate_exact_pure_instruction(&expected, &invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn sparse_mma_probe_requires_each_variant_selector_set() {
        let k32 = InstructionPattern::new(
            "mma",
            &[
                "sp", "sync", "aligned", "m16n8k32", "row", "col", "s32", "s8", "u8", "s32",
            ],
            vec![
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 2 },
                OperandPattern::RegisterList { length: 2 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::Register,
                OperandPattern::Immediate,
            ],
        );
        let selector_zero = "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8}, {%r9, %r10, %r11, %r12}, %r13, 0;";
        let selector_one = "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8}, {%r9, %r10, %r11, %r12}, %r13, 1;";

        validate_sparse_mma_selectors(&k32, &[0, 1], &format!("{selector_zero}\n{selector_one}"))
            .unwrap();
        assert!(validate_sparse_mma_selectors(&k32, &[0, 1], selector_zero).is_err());
        assert!(validate_sparse_mma_selectors(&k32, &[0, 1], selector_one).is_err());

        let mut register_selector = k32.clone();
        register_selector.operands[5] = OperandPattern::Register;
        assert!(validate_sparse_mma_selectors(&register_selector, &[0, 1], selector_zero).is_err());

        let selector_two = selector_one.replace(", 1;", ", 2;");
        assert!(
            validate_sparse_mma_selectors(
                &k32,
                &[0, 1],
                &format!("{selector_zero}\n{selector_one}\n{selector_two}")
            )
            .is_err()
        );
        assert!(
            validate_sparse_mma_selectors(
                &k32,
                &[0, 1],
                &format!("@%p0 {selector_zero}\n{selector_one}")
            )
            .is_err()
        );

        let k64 = InstructionPattern::new(
            "mma",
            &[
                "sp::ordered_metadata",
                "sync",
                "aligned",
                "m16n8k64",
                "row",
                "col",
                "s32",
                "s8",
                "u8",
                "s32",
            ],
            vec![
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::RegisterList { length: 4 },
                OperandPattern::Register,
                OperandPattern::Immediate,
            ],
        );
        let k64_zero = "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32 {%r1, %r2, %r3, %r4}, {%r5, %r6, %r7, %r8}, {%r9, %r10, %r11, %r12}, {%r13, %r14, %r15, %r16}, %r17, 0;";
        let k64_one = k64_zero.replace(", 0;", ", 1;");
        validate_sparse_mma_selectors(&k64, &[0], k64_zero).unwrap();
        assert!(validate_sparse_mma_selectors(&k64, &[0], &k64_one).is_err());
        assert!(
            validate_sparse_mma_selectors(&k64, &[0], &format!("{k64_zero}\n{k64_one}")).is_err()
        );

        let standard_k64 = InstructionPattern::new(
            "mma",
            &[
                "sp", "sync", "aligned", "m16n8k64", "row", "col", "s32", "s8", "u8", "s32",
            ],
            k64.operands.clone(),
        );
        let standard_k64_zero = k64_zero.replace("sp::ordered_metadata", "sp");
        validate_sparse_mma_selectors(&standard_k64, &[0], &standard_k64_zero).unwrap();
        assert!(validate_sparse_mma_selectors(&standard_k64, &[0], k64_zero).is_err());
        assert!(validate_sparse_mma_selectors(&k64, &[0], &standard_k64_zero).is_err());
    }

    #[test]
    fn candidate_artifact_names_are_confined_to_one_safe_stem() {
        assert_eq!(
            candidate_artifact_stem("thread_idx_x").unwrap(),
            "thread_idx_x"
        );
        for invalid in ["", "../evidence", "thread.idx.x", "ThreadIdxX"] {
            assert!(candidate_artifact_stem(invalid).is_err(), "{invalid}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn portable_evidence_tool_names_resolve_through_path() {
        let temp = candidate_tool_test_dir();
        let expected = write_fake_tool(&temp.0, "ptxas", "#!/bin/sh\nexit 0\n");
        let search_path = std::env::join_paths([&temp.0]).unwrap();

        assert_eq!(
            resolve_evidence_tool_with_path("ptxas", Some(&search_path)).unwrap(),
            expected
        );
        assert!(resolve_evidence_tool_with_path("missing-tool", Some(&search_path)).is_err());
    }

    #[test]
    fn ptxas_version_requires_the_nvidia_banner_and_cuda_release() {
        let valid = "ptxas: NVIDIA (R) Ptx optimizing assembler\n\
                     Copyright (c) 2005-2026 NVIDIA Corporation\n\
                     Cuda compilation tools, release 13.3, V13.3.33\n";
        assert_eq!(
            parse_ptxas_version(valid, "").unwrap(),
            "Cuda compilation tools, release 13.3, V13.3.33"
        );
        for invalid in [
            "printf (GNU coreutils) 9.4\n",
            "Cuda compilation tools, release 13.3, V13.3.33\n",
            "ptxas: NVIDIA (R) Ptx optimizing assembler\n",
        ] {
            assert!(parse_ptxas_version(invalid, "").is_err(), "{invalid}");
        }
    }

    #[test]
    fn rustc_verbose_requires_a_concrete_commit_hash() {
        let valid = "rustc 1.100.0-nightly (e457a7b0d 2026-08-27)\n\
                     binary: rustc\n\
                     commit-hash: e457a7b0d326d67b4322ef0d11bd715cfaeda48f\n\
                     commit-date: 2026-08-27\n\
                     host: x86_64-unknown-linux-gnu\n\
                     release: 1.100.0-nightly\n\
                     LLVM version: 23.1.0\n";
        let (host, commit) = parse_rustc_verbose(valid).unwrap();
        assert_eq!(host, "x86_64-unknown-linux-gnu");
        assert_eq!(
            commit,
            "e457a7b0d326d67b4322ef0d11bd715cfaeda48f".parse().unwrap()
        );

        let missing = parse_rustc_verbose("host: x86_64-unknown-linux-gnu\n").unwrap_err();
        assert!(missing.to_string().contains("commit-hash"));

        let unknown = parse_rustc_verbose("commit-hash: unknown\nhost: x86_64-unknown-linux-gnu\n")
            .unwrap_err();
        assert!(format!("{unknown:#}").contains("commit-hash"));
    }

    #[cfg(unix)]
    #[test]
    fn successful_fake_ptxas_without_a_real_cubin_is_rejected() {
        const BANNER: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\n' 'ptxas: NVIDIA (R) Ptx optimizing assembler' 'Cuda compilation tools, release 13.3, V13.3.33'
fi
exit 0
"#;
        const NON_ELF: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\n' 'ptxas: NVIDIA (R) Ptx optimizing assembler' 'Cuda compilation tools, release 13.3, V13.3.33'
else
  printf '%s' 'this is definitely not an ELF cubin; padding keeps it longer than one ELF64 header' > "$4"
fi
exit 0
"#;
        let temp = candidate_tool_test_dir();
        let ptx = temp.0.join("probe.ptx");
        fs::write(&ptx, ".version 7.0\n.target sm_80\n").unwrap();

        let no_output = write_fake_tool(&temp.0, "no-output-ptxas", BANNER);
        assert_eq!(
            ptxas_identity(&no_output).unwrap().version,
            "Cuda compilation tools, release 13.3, V13.3.33"
        );
        let cubin = temp.0.join("missing.cubin");
        let error = assemble_candidate_ptx(&no_output, "sm_80", &ptx, &cubin).unwrap_err();
        assert!(error.to_string().contains("was not produced"), "{error:#}");

        let non_elf = write_fake_tool(&temp.0, "non-elf-ptxas", NON_ELF);
        let cubin = temp.0.join("non-elf.cubin");
        let error = assemble_candidate_ptx(&non_elf, "sm_80", &ptx, &cubin).unwrap_err();
        assert!(error.to_string().contains("not an ELF64"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn fake_non_nvidia_ptxas_banner_is_rejected() {
        let temp = candidate_tool_test_dir();
        let tool = write_fake_tool(
            &temp.0,
            "not-ptxas",
            "#!/bin/sh\nprintf '%s\\n' 'printf (GNU coreutils) 9.4'\n",
        );
        let error = ptxas_identity(&tool).unwrap_err();
        assert!(error.to_string().contains("NVIDIA PTX"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn candidate_artifact_cleanup_removes_stale_drafts() {
        let temp = candidate_tool_test_dir();
        let paths = candidate_artifact_paths(&temp.0, "test");
        for path in [
            &paths.llvm_ir,
            &paths.llvm_bitcode,
            &paths.canonical_llvm_ir,
            &paths.ptx,
            &paths.cubin,
            &paths.manifest,
            &paths.manifest_temp,
        ] {
            fs::write(path, "stale").unwrap();
        }
        clear_candidate_artifacts(&paths).unwrap();
        for path in [
            &paths.llvm_ir,
            &paths.llvm_bitcode,
            &paths.canonical_llvm_ir,
            &paths.ptx,
            &paths.cubin,
            &paths.manifest,
            &paths.manifest_temp,
        ] {
            assert!(!path.exists(), "{}", path.display());
        }
    }

    #[test]
    fn candidate_manifest_is_deterministic_and_never_claims_admission() {
        let draft = CandidateProbeDraft {
            schema: 1,
            kind: "candidate_probe".into(),
            admitted: false,
            comparison_only: true,
            intrinsic_id: "test".into(),
            operation_key: "test.operation".into(),
            source: IntrinsicSource::PtxNative {
                instruction: "test".into(),
            },
            mechanism: BackendLoweringMechanism::InlinePtx,
            expected_ptx: InstructionPattern::new("test", &[], vec![]),
            target: CandidateTargetDraft {
                requirement: CatalogTargetRequirement {
                    minimum_ptx: "7.0".parse().unwrap(),
                    hardware: CatalogHardwareTarget::AnyOf {
                        alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 80 }],
                    },
                },
                target_triple: "nvptx64-nvidia-cuda".into(),
                gpu_target: "sm_80".into(),
                ptx_feature: "+ptx70".into(),
            },
            catalog_inputs: CatalogInputs {
                imported_sha256: "a".repeat(64),
                overlay_sha256: "b".repeat(64),
                abi_ledger_sha256: "c".repeat(64),
                evidence_sha256: Vec::new(),
            },
            tools: Vec::new(),
            artifacts: Vec::new(),
            stages: Vec::new(),
        };
        let first = pretty_json(&draft).unwrap();
        let second = pretty_json(&draft).unwrap();
        assert_eq!(first.as_bytes(), second.as_bytes());
        let json: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert_eq!(json["admitted"], false);
        assert_eq!(json["comparison_only"], true);
        assert_eq!(
            json["catalog_inputs"]["evidence_sha256"],
            serde_json::json!([])
        );
    }
}
