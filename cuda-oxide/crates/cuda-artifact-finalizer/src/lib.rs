/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-independent CUDA artifact finalization.
//!
//! This crate is the single owner of cuda-oxide's libNVVM and nvJitLink
//! compilation policy. It deliberately does not link the CUDA Driver. Both
//! build-time materialization and runtime fallback use the same typed target,
//! FMA, debug, input-order, validation, and provenance rules.

mod diagnostics;
mod link;
mod nvvm;
mod options;
mod provenance;
mod ptx;
mod validation;

pub use diagnostics::KernelResourceUsage;
pub use libnvvm_sys::{CudaArch, CudaArchParseError, LibdeviceNotFound, NvvmError, find_libdevice};
pub use link::{LinkReport, LtoLinker};
pub use nvjitlink_sys::NvJitLinkError;
pub use nvvm::NvvmCompiler;
pub use options::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
pub use provenance::{
    MaterializerHandshakeV1, PinnedToolProvenance, ToolFileIdentity, ToolProvenance, recipe_digest,
};
pub use ptx::PtxAssembler;
pub use validation::is_valid_cubin;

use provenance::common_provenance_digest;
use std::path::PathBuf;
use thiserror::Error;

/// Failures while compiling or linking CUDA artifacts.
#[derive(Debug, Error)]
pub enum FinalizerError {
    /// libNVVM failed to load, validate, or compile.
    #[error("libnvvm: {0}")]
    Nvvm(#[from] libnvvm_sys::NvvmError),

    /// nvJitLink failed to load or link.
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] nvjitlink_sys::NvJitLinkError),

    /// No standalone PTX assembler could be discovered.
    #[error(
        "Could not locate ptxas. Set CUDA_OXIDE_PTXAS, CUDA_TOOLKIT_PATH, CUDA_HOME, or CUDA_PATH, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    PtxasNotFound {
        /// Newline-separated discovery paths.
        tried: String,
    },

    /// A discovered executable was not NVIDIA's PTX assembler.
    #[error("the discovered ptxas executable is invalid ({path}): {details}")]
    InvalidPtxas {
        /// Candidate executable path.
        path: PathBuf,
        /// Version-probe failure details.
        details: String,
    },

    /// `ptxas` rejected the supplied PTX or options.
    #[error("ptxas failed with {status}: {diagnostics}")]
    PtxasFailed {
        /// Process exit status.
        status: String,
        /// Combined standard output and error diagnostics.
        diagnostics: String,
    },

    /// `libdevice.10.bc` could not be found.
    #[error(
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibdeviceNotFound {
        /// Newline-separated discovery paths.
        tried: String,
    },

    /// A finalizer input could not be read.
    #[error("Failed reading {path}: {source}")]
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// The installed toolkit does not accept cuda-oxide's NVVM IR version.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Runtime toolkit dialect discovery disagreed with the target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// A diagnostic input name cannot be represented by the CUDA C APIs.
    #[error("CUDA artifact input name contains an interior NUL byte: {name:?}")]
    InvalidInputName { name: String },

    /// A supplied compiler or linker input contained no bytes.
    #[error("CUDA artifact input is empty: {name}")]
    EmptyInput { name: String },

    /// PTX is a C-string input to nvJitLink and may only contain its optional
    /// terminating NUL at the end.
    #[error("PTX input contains an interior NUL byte: {name}")]
    InteriorNulPtx { name: String },

    /// nvJitLink was invoked without an input module.
    #[error("at least one link input is required (ordered LTOIR modules or a single PTX module)")]
    NoLinkInputs,

    /// A CUDA finalization tool returned bytes that are not a complete CUDA ELF image.
    #[error("CUDA artifact finalization returned an invalid or truncated cubin")]
    InvalidCubin,

    /// nvJitLink returned no PTX bytes.
    #[error("nvJitLink returned an empty PTX artifact")]
    EmptyPtx,

    /// A pinned CUDA compilation tool changed around an operation. Its output can
    /// no longer be attributed to the provenance used by Cargo or a cache key.
    #[error(
        "the pinned {tool} file changed before or during CUDA artifact finalization; refusing the unverified output"
    )]
    ToolIdentityChanged { tool: &'static str },

    /// A serialized digest hint was internally inconsistent or from another
    /// protocol version.
    #[error("invalid materializer provenance handshake")]
    InvalidMaterializerHandshake,
}

/// Complete NVVM IR to cubin/PTX finalizer.
#[derive(Clone)]
pub struct Finalizer {
    compiler: NvvmCompiler,
    linker: LtoLinker,
}

impl Finalizer {
    /// Discover libNVVM, libdevice, and nvJitLink without loading the Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            compiler: NvvmCompiler::discover()?,
            linker: LtoLinker::discover()?,
        })
    }

    /// Discover tools using a validated parent-process handoff as a digest
    /// acceleration hint.
    ///
    /// Each DSO is opened and its retained-file identity is checked in this
    /// process. A mismatch falls back to hashing the newly opened file. The
    /// caller must still compare [`Self::provenance_digest`] with the semantic
    /// provenance it expected.
    pub fn discover_with_handshake(
        handshake: &MaterializerHandshakeV1,
    ) -> Result<Self, FinalizerError> {
        if !handshake.has_consistent_provenance() {
            return Err(FinalizerError::InvalidMaterializerHandshake);
        }
        Ok(Self {
            compiler: NvvmCompiler::discover_with_expected(Some(&handshake.libnvvm))?,
            linker: LtoLinker::discover_with_expected(Some(&handshake.nvjitlink))?,
        })
    }

    /// Export content digests together with the retained descriptors that
    /// produced them for a child materializer process.
    pub fn materializer_handshake(&self) -> Option<MaterializerHandshakeV1> {
        Some(MaterializerHandshakeV1::new(
            self.compiler.pinned_tool_provenance()?,
            self.linker.pinned_tool_provenance()?,
            self.compiler.libdevice_digest(),
        ))
    }

    /// Compile one NVVM IR module and return a validated target-specific cubin.
    pub fn materialize_nvvm_ir(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        let ltoir = self
            .compiler
            .compile_nvvm_ir_to_ltoir(module_name, nvvm_ir, options)?;
        let ltoir_name = format!("{module_name}.ltoir");
        self.linker.link_ltoir(
            &[NamedInput::new(&ltoir_name, &ltoir)],
            options,
            FinalizerOutput::Cubin,
        )
    }

    /// Compile NVVM IR to cubin and collect ptxas resource diagnostics.
    ///
    /// The diagnostics are best-effort: on an nvJitLink too old to accept the
    /// reporting options, the link is retried without them and the report's
    /// info log and resource usage are empty. See
    /// [`LtoLinker::link_ltoir_with_report`].
    pub fn materialize_nvvm_ir_with_report(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<LinkReport, FinalizerError> {
        let ltoir = self
            .compiler
            .compile_nvvm_ir_to_ltoir(module_name, nvvm_ir, options)?;
        let ltoir_name = format!("{module_name}.ltoir");
        self.linker.link_ltoir_with_report(
            &[NamedInput::new(&ltoir_name, &ltoir)],
            options,
            FinalizerOutput::Cubin,
        )
    }

    /// Link ordered LTOIR modules to cubin or PTX.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.linker.link_ltoir(inputs, options, output)
    }

    /// Link ordered LTOIR modules while collecting ptxas resource diagnostics.
    ///
    /// The diagnostics are best-effort: on an nvJitLink too old to accept the
    /// reporting options, the link is retried without them and the report's
    /// info log and resource usage are empty. See
    /// [`LtoLinker::link_ltoir_with_report`].
    pub fn link_ltoir_with_report(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<LinkReport, FinalizerError> {
        self.linker.link_ltoir_with_report(inputs, options, output)
    }

    /// Compiler component, including exact libdevice bytes and provenance.
    pub fn compiler(&self) -> &NvvmCompiler {
        &self.compiler
    }

    /// CUDA artifact linker component.
    pub fn linker(&self) -> &LtoLinker {
        &self.linker
    }

    /// Exact discovered tool and libdevice digests.
    pub fn provenance(&self) -> ToolProvenance {
        ToolProvenance {
            libnvvm_sha256: self.compiler.libnvvm_digest(),
            nvjitlink_sha256: self.linker.nvjitlink_digest(),
            libdevice_sha256: self.compiler.libdevice_digest(),
        }
    }

    /// Exact full-pipeline provenance, or `None` if a loaded DSO is unknown.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        let provenance = self.provenance();
        Some(common_provenance_digest(
            &provenance.libnvvm_sha256?,
            &provenance.nvjitlink_sha256?,
            &provenance.libdevice_sha256,
        ))
    }

    /// Digest the full NVVM IR to output recipe, including ordered options.
    pub fn nvvm_ir_artifact_digest(
        &self,
        module_name: &str,
        ltoir_module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Option<[u8; 32]> {
        nvvm_ir_artifact_digest_with_provenance(
            module_name,
            ltoir_module_name,
            nvvm_ir,
            options,
            output,
            self.provenance(),
        )
    }
}

/// Digest a complete finalization plan from already-established provenance.
///
/// This is useful to fingerprint Cargo work before executing the plan. It
/// returns `None` unless both loaded tool identities are exact.
pub fn nvvm_ir_artifact_digest_with_provenance(
    module_name: &str,
    ltoir_module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    provenance: ToolProvenance,
) -> Option<[u8; 32]> {
    let compiler_digest = nvvm::nvvm_ir_artifact_digest_parts(
        module_name,
        nvvm_ir,
        options,
        &provenance.libdevice_sha256,
        &provenance.libnvvm_sha256?,
    );
    let linker_digest = link::ltoir_artifact_digest_parts(
        &[NamedInput::new(ltoir_module_name, &compiler_digest)],
        options,
        output,
        &provenance.nvjitlink_sha256?,
    );
    Some(
        provenance::StableDigest::new()
            .field("recipe", recipe_digest())
            .field("route", b"nvvm-ir-to-final-output")
            .field("compiler-plan", compiler_digest)
            .field("linker-plan", linker_digest)
            .finish(),
    )
}

/// Digest an ordered LTOIR link from an established exact linker identity.
pub fn ltoir_artifact_digest_with_provenance(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_sha256: &[u8; 32],
) -> [u8; 32] {
    link::ltoir_artifact_digest_parts(inputs, options, output, nvjitlink_sha256)
}

fn validate_name(name: &str) -> Result<(), FinalizerError> {
    if name.as_bytes().contains(&0) {
        Err(FinalizerError::InvalidInputName {
            name: name.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// `@llvm.used` keeps the kernel alive through LTO, exactly as cuda-oxide's
    /// NVVM exporter emits it for real kernels. Without it nvJitLink's link-time
    /// optimizer dead-strips the annotation-marked kernel as unreachable and
    /// links an empty module: the pipeline still succeeds, so every assertion
    /// below passed while nothing was being compiled. Measured on CUDA 13.3
    /// before this line existed, the linked PTX was 202 bytes with zero
    /// functions and the cubin had no entry points.
    const LEGACY_NVVM_IR: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ()* @kernel to i8*)], section "llvm.metadata"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void ()* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, nvJitLink, and libdevice"]
    fn live_pipeline_accepts_toolkit_cubins_and_emits_ptx_for_both_fma_policies() {
        let finalizer = Finalizer::discover().unwrap();
        assert!(finalizer.provenance_digest().is_some());
        let target: CudaArch = "sm_86".parse().unwrap();

        for (allow_fma, debug) in [
            (false, DebugPolicy::None),
            (false, DebugPolicy::LineTables),
            (true, DebugPolicy::Full),
        ] {
            let options = FinalizationOptions::new(target.clone())
                .with_fma_contraction(allow_fma)
                .with_debug_policy(debug);
            let ltoir = finalizer
                .compiler()
                .compile_nvvm_ir_to_ltoir("kernel.ll", LEGACY_NVVM_IR, &options)
                .unwrap();
            assert!(!ltoir.is_empty());
            let input = [NamedInput::new("kernel.ltoir", &ltoir)];
            let cubin = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Cubin)
                .unwrap();
            assert!(is_valid_cubin(&cubin));
            // A cubin whose kernel was stripped is still a well-formed ELF, so
            // `is_valid_cubin` alone cannot tell the two apart. Require the
            // kernel's name to survive into the image.
            assert!(
                cubin.windows(b"kernel".len()).any(|part| part == b"kernel"),
                "cubin has no `kernel` symbol ({} bytes): the kernel was \
                 dead-stripped and this test is validating an empty module",
                cubin.len()
            );
            let ptx = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Ptx)
                .unwrap();
            assert!(
                ptx.windows(b".version".len())
                    .any(|part| part == b".version")
            );
            // `.version` is in the header of even an empty module, so it proves
            // only that something was emitted. The entry point is what proves
            // the kernel made it through libNVVM and nvJitLink.
            let ptx_text = String::from_utf8_lossy(&ptx);
            assert!(
                ptx_text.contains(".entry kernel"),
                "linked PTX has no `.entry kernel` ({} bytes):\n{ptx_text}",
                ptx.len()
            );
        }
    }

    /// Legacy-dialect NVVM IR for a kernel that ptxas must spill: `values`
    /// floats stay live across `rounds` of all-to-all mixing while a
    /// `maxnreg` annotation caps the kernel far below that live set.
    fn spilling_kernel_nvvm_ir(values: usize, rounds: usize, maxnreg: u32) -> String {
        use std::fmt::Write;

        let mut body = String::new();
        for i in 0..values {
            let _ = writeln!(
                body,
                "  %ptr{i} = getelementptr inbounds float, float* %data, i64 {i}"
            );
            let _ = writeln!(body, "  %v0_{i} = load float, float* %ptr{i}, align 4");
        }
        for round in 0..rounds {
            let next = round + 1;
            for i in 0..values {
                let mul_with = (i + 1) % values;
                let add_with = (i + 3) % values;
                let _ = writeln!(
                    body,
                    "  %m{next}_{i} = fmul float %v{round}_{i}, %v{round}_{mul_with}"
                );
                let _ = writeln!(
                    body,
                    "  %v{next}_{i} = fadd float %m{next}_{i}, %v{round}_{add_with}"
                );
            }
        }
        for i in 0..values {
            let _ = writeln!(
                body,
                "  store float %v{rounds}_{i}, float* %ptr{i}, align 4"
            );
        }

        // `@llvm.used` keeps the kernel alive through LTO, exactly as
        // cuda-oxide's NVVM exporter emits it for real kernels; without it
        // nvJitLink dead-strips the annotation-marked kernel and links an
        // empty module.
        format!(
            r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*)* @spill_kernel to i8*)], section "llvm.metadata"

define void @spill_kernel(float* %data) {{
entry:
{body}  ret void
}}

!nvvm.annotations = !{{!0, !1}}
!nvvmir.version = !{{!2}}
!0 = !{{void (float*)* @spill_kernel, !"kernel", i32 1}}
!1 = !{{void (float*)* @spill_kernel, !"maxnreg", i32 {maxnreg}}}
!2 = !{{i32 2, i32 0, i32 3, i32 1}}
"#
        )
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, nvJitLink, and libdevice"]
    fn live_link_report_parses_ptxas_resource_usage_for_a_spilling_kernel() {
        const MAXNREG: u32 = 32;

        let finalizer = Finalizer::discover().unwrap();
        let options = FinalizationOptions::new("sm_86".parse().unwrap());
        let nvvm_ir = spilling_kernel_nvvm_ir(64, 4, MAXNREG);
        let ltoir = finalizer
            .compiler()
            .compile_nvvm_ir_to_ltoir("spill.ll", nvvm_ir.as_bytes(), &options)
            .unwrap();
        let input = [NamedInput::new("spill.ltoir", &ltoir)];

        let report = finalizer
            .link_ltoir_with_report(&input, &options, FinalizerOutput::Cubin)
            .unwrap();
        assert!(is_valid_cubin(&report.image));

        // A missing info log means the toolkit silently degraded the report;
        // this test exists exactly to catch the diagnostic options or the log
        // format drifting away from what the parser expects.
        let raw_log = report
            .info_log
            .as_deref()
            .expect("nvJitLink accepted the diagnostic options and produced an info log");
        println!("--- raw nvJitLink info log ---\n{raw_log}");
        println!(
            "--- parsed resource usage ---\n{:#?}",
            report.resource_usage
        );

        let usage = report
            .resource_usage
            .iter()
            .find(|usage| usage.kernel == "spill_kernel")
            .expect("ptxas resource lines for spill_kernel were parsed from the info log");
        let registers = usage
            .registers
            .expect("ptxas reported a register count for spill_kernel");
        assert!(
            (1..=MAXNREG).contains(&registers),
            "maxnreg({MAXNREG}) kernel reported an implausible register count: {registers}"
        );
        assert!(
            usage.has_spills(),
            "64 live values under maxnreg({MAXNREG}) must spill: {usage:?}"
        );
        for (figure, bytes) in [
            ("stack frame", usage.stack_frame_bytes),
            ("spill stores", usage.spill_store_bytes),
            ("spill loads", usage.spill_load_bytes),
        ] {
            assert!(
                (1..=65_536).contains(&bytes),
                "implausible {figure} byte count for spill_kernel: {bytes}"
            );
        }
    }
}
