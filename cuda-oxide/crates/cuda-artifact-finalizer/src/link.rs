/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::diagnostics::{KernelResourceUsage, parse_ptxas_resource_usage};
use crate::nvvm::{loaded_tool_digest_with_expected, report_changed_tool};
use crate::options::{FinalizationOptions, FinalizerOutput, NamedInput};
use crate::provenance::{
    PinnedToolProvenance, StableDigest, linker_provenance_digest, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::validation::is_valid_cubin;
use crate::{FinalizerError, validate_name};
use nvjitlink_sys::{InputType, LibNvJitLink, LinkOutput, Linker, NvJitLinkError};
use std::sync::{Arc, Mutex, OnceLock};

struct LoadedLinkerTool {
    library: Arc<LibNvJitLink>,
    digest: Option<[u8; 32]>,
}

static LINKER_TOOL: OnceLock<Arc<LoadedLinkerTool>> = OnceLock::new();
static LINKER_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Final linker output plus resource diagnostics collected from ptxas.
#[derive(Debug)]
pub struct LinkReport {
    /// Complete cubin or PTX image.
    pub image: Vec<u8>,
    /// Compiler or linker informational output, when requested and available.
    pub info_log: Option<String>,
    /// Per-kernel ptxas resource usage parsed from the informational output.
    pub resource_usage: Vec<KernelResourceUsage>,
}

/// Driver-independent linker for LTOIR and PTX inputs.
#[derive(Clone)]
pub struct LtoLinker {
    tool: Arc<LoadedLinkerTool>,
}

impl LtoLinker {
    /// Discover and pin nvJitLink without loading libNVVM or the CUDA Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Self::discover_with_expected(None)
    }

    pub(crate) fn discover_with_expected(
        expected: Option<&PinnedToolProvenance>,
    ) -> Result<Self, FinalizerError> {
        Ok(Self {
            tool: load_linker_tool(expected)?,
        })
    }

    pub(crate) fn pinned_tool_provenance(&self) -> Option<PinnedToolProvenance> {
        let sha256 = self.tool.digest?;
        let file = self.tool.library.loaded_file_if_unchanged()?;
        Some(PinnedToolProvenance {
            sha256,
            file: crate::provenance::ToolFileIdentity::capture(file)?,
        })
    }

    /// Digest of the exact loaded nvJitLink file, when its identity is known.
    pub fn nvjitlink_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if self.tool.library.loaded_file_if_unchanged().is_some() {
            Some(digest)
        } else {
            report_changed_tool("nvJitLink");
            None
        }
    }

    /// Exact route provenance, or `None` when the loaded DSO is unidentifiable.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        self.nvjitlink_digest()
            .map(|digest| linker_provenance_digest(&digest))
    }

    /// Link one or more LTOIR modules in the exact supplied order.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        Ok(self.link_ltoir_impl(inputs, options, output, false)?.image)
    }

    /// Link LTOIR while collecting non-semantic ptxas resource diagnostics.
    ///
    /// For cubin output this requests nvJitLink verbose output and `ptxas -v`,
    /// and bypasses nvJitLink's own JIT cache: a cache hit skips ptxas and
    /// would return an empty report. The reporting flags are intentionally
    /// excluded from artifact digests.
    ///
    /// The report is best-effort: if the loaded nvJitLink rejects the
    /// reporting options with `NVJITLINK_ERROR_UNRECOGNIZED_OPTION`, the link
    /// is retried once without them and the returned [`LinkReport`] carries
    /// the linked image with no info log and no resource usage.
    ///
    /// The reported image is code-identical to a [`Self::link_ltoir`] image
    /// but not byte-identical: ptxas records its own option line inside the
    /// cubin's `.note.nv.tkinfo` section, so `-v` shows up there. Every
    /// other section, including all generated code, is unchanged (verified
    /// section-by-section on CUDA 13.3, sm_120).
    pub fn link_ltoir_with_report(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<LinkReport, FinalizerError> {
        self.link_ltoir_impl(inputs, options, output, true)
    }

    fn link_ltoir_impl(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
        collect_resource_usage: bool,
    ) -> Result<LinkReport, FinalizerError> {
        validate_inputs(inputs)?;
        self.link_inputs(
            inputs,
            InputType::Ltoir,
            options,
            output,
            collect_resource_usage,
        )
    }

    /// Compile and link one PTX module to a validated target-specific cubin.
    pub fn link_ptx_to_cubin(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        validate_inputs(std::slice::from_ref(&input))?;
        // Enforce the PTX C-string rule up front so this route rejects
        // exactly the inputs `ptx_artifact_digest` rejects, with the
        // finalizer's error vocabulary. `Linker::add` NUL-terminates the FFI
        // backing itself through the same shared nvjitlink-sys rule.
        logical_ptx(input)?;
        Ok(self
            .link_inputs(
                std::slice::from_ref(&input),
                InputType::Ptx,
                options,
                FinalizerOutput::Cubin,
                false,
            )?
            .image)
    }

    fn link_inputs(
        &self,
        inputs: &[NamedInput<'_>],
        input_type: InputType,
        options: &FinalizationOptions,
        output: FinalizerOutput,
        collect_resource_usage: bool,
    ) -> Result<LinkReport, FinalizerError> {
        with_revalidated_tool_identity(
            "nvJitLink",
            self.tool.digest,
            || current_linker_tool_digest(&self.tool),
            || {
                match self.run_link(inputs, input_type, options, output, collect_resource_usage) {
                    // Older nvJitLink versions reject the diagnostic-only
                    // reporting options with NVJITLINK_ERROR_UNRECOGNIZED_OPTION.
                    // The caller asked for the same program plus a best-effort
                    // report, so degrade to a plain link with an empty report
                    // rather than failing the whole link.
                    Err(FinalizerError::NvJitLink(error))
                        if collect_resource_usage && error.is_unrecognized_option() =>
                    {
                        self.run_link(inputs, input_type, options, output, false)
                    }
                    result => result,
                }
            },
        )
    }

    fn run_link(
        &self,
        inputs: &[NamedInput<'_>],
        input_type: InputType,
        options: &FinalizationOptions,
        output: FinalizerOutput,
        collect_resource_usage: bool,
    ) -> Result<LinkReport, FinalizerError> {
        let mut option_storage = if input_type == InputType::Ptx {
            options.nvjitlink_ptx_options()
        } else {
            options.nvjitlink_ltoir_options(output)
        };
        if collect_resource_usage {
            option_storage.extend(options.nvjitlink_diagnostic_options(output));
        }
        let option_refs = option_storage
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut linker = Linker::new(&self.tool.library, &option_refs)?;
        for input in inputs {
            linker.add(input_type, input.bytes, input.name)?;
        }

        let LinkOutput { image, info_log } = match output {
            FinalizerOutput::Cubin => linker.finish_with_info_log()?,
            FinalizerOutput::Ptx => linker.finish_ptx_with_info_log()?,
        };
        if output == FinalizerOutput::Cubin && !is_valid_cubin(&image) {
            return Err(FinalizerError::InvalidCubin);
        }
        if output == FinalizerOutput::Ptx && image.is_empty() {
            return Err(FinalizerError::EmptyPtx);
        }

        let resource_usage = if collect_resource_usage {
            info_log
                .as_deref()
                .map(parse_ptxas_resource_usage)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(LinkReport {
            image,
            info_log,
            resource_usage,
        })
    }

    /// Digest every semantic input to an ordered LTOIR link.
    pub fn artifact_digest(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Option<[u8; 32]> {
        let nvjitlink = self.nvjitlink_digest()?;
        Some(ltoir_artifact_digest_parts(
            inputs, options, output, &nvjitlink,
        ))
    }

    /// Digest every semantic input to a PTX-to-cubin link.
    pub fn ptx_artifact_digest(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Option<[u8; 32]>, FinalizerError> {
        validate_inputs(std::slice::from_ref(&input))?;
        let ptx = logical_ptx(input)?;
        let Some(nvjitlink) = self.nvjitlink_digest() else {
            return Ok(None);
        };
        Ok(Some(ptx_artifact_digest_parts(
            NamedInput::new(input.name, ptx),
            options,
            &nvjitlink,
        )))
    }
}

fn current_linker_tool_digest(tool: &LoadedLinkerTool) -> Option<[u8; 32]> {
    tool.library.loaded_file_if_unchanged()?;
    tool.digest
}

fn validate_inputs(inputs: &[NamedInput<'_>]) -> Result<(), FinalizerError> {
    if inputs.is_empty() {
        return Err(FinalizerError::NoLinkInputs);
    }
    for input in inputs {
        validate_name(input.name)?;
        if input.bytes.is_empty() {
            return Err(FinalizerError::EmptyInput {
                name: input.name.to_string(),
            });
        }
    }
    Ok(())
}

/// Logical PTX bytes shared by the digest and link routes.
///
/// nvjitlink-sys owns the PTX C-string rule (strip the single optional
/// trailing NUL, reject any other NUL); this wrapper only adds the
/// finalizer's non-empty-input policy and error vocabulary on top.
pub(crate) fn logical_ptx(input: NamedInput<'_>) -> Result<&[u8], FinalizerError> {
    let logical =
        nvjitlink_sys::logical_ptx(input.bytes, input.name).map_err(|error| match error {
            NvJitLinkError::InteriorNulPtx { name, .. } => FinalizerError::InteriorNulPtx { name },
            other => FinalizerError::NvJitLink(other),
        })?;
    if logical.is_empty() {
        return Err(FinalizerError::EmptyInput {
            name: input.name.to_string(),
        });
    }
    Ok(logical)
}

fn load_linker_tool(
    expected: Option<&PinnedToolProvenance>,
) -> Result<Arc<LoadedLinkerTool>, FinalizerError> {
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = LINKER_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvJitLink::load_for_cache()?;
    let digest =
        loaded_tool_digest_with_expected("nvJitLink", library.loaded_file_if_unchanged(), expected);
    let digest = if digest.is_some() && library.loaded_file_if_unchanged().is_none() {
        report_changed_tool("nvJitLink");
        None
    } else {
        digest
    };
    let loaded = Arc::new(LoadedLinkerTool {
        library: Arc::new(library),
        digest,
    });
    let _ = LINKER_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

pub(crate) fn ltoir_artifact_digest_parts(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    let output_name = match output {
        FinalizerOutput::Cubin => b"elf-cubin".as_slice(),
        FinalizerOutput::Ptx => b"ptx".as_slice(),
    };
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ltoir-to-output")
        .field("output", output_name);
    for input in inputs {
        digest = digest
            .field("ltoir-name", input.name.as_bytes())
            .field("ltoir", input.bytes);
    }
    for option in options.nvjitlink_ltoir_options(output) {
        digest = digest.field("nvjitlink-option", option.as_bytes());
    }
    digest
        .field("libnvjitlink-sha256", nvjitlink_digest)
        .finish()
}

pub(crate) fn ptx_artifact_digest_parts(
    input: NamedInput<'_>,
    options: &FinalizationOptions,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ptx-to-cubin")
        .field("ptx-name", input.name.as_bytes())
        .field("ptx", input.bytes);
    for option in options.nvjitlink_ptx_options() {
        digest = digest.field("nvjitlink-option", option.as_bytes());
    }
    digest
        .field("libnvjitlink-sha256", nvjitlink_digest)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_digest_preserves_input_order_names_output_and_policy() {
        let options = FinalizationOptions::new("sm_120".parse().unwrap());
        let a = NamedInput::new("a.ltoir", b"a");
        let b = NamedInput::new("b.ltoir", b"b");
        let baseline =
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[7; 32]);
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[b, a], &options, FinalizerOutput::Cubin, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[NamedInput::new("renamed.ltoir", b"a"), b],
                &options,
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &FinalizationOptions::new("sm_90".parse().unwrap()),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options
                    .clone()
                    .with_debug_policy(crate::DebugPolicy::LineTables),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[8; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Ptx, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options.clone().with_fma_contraction(false),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
    }

    #[test]
    fn input_validation_rejects_zero_inputs_empty_data_and_nul_names() {
        assert!(matches!(
            validate_inputs(&[]),
            Err(FinalizerError::NoLinkInputs)
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("empty", b"")]),
            Err(FinalizerError::EmptyInput { .. })
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("bad\0name", b"x")]),
            Err(FinalizerError::InvalidInputName { .. })
        ));
    }

    #[test]
    fn ptx_digest_covers_name_bytes_policy_and_linker_identity() {
        let options = FinalizationOptions::new("sm_80".parse().unwrap());
        let baseline =
            ptx_artifact_digest_parts(NamedInput::new("kernel.ptx", b"ptx"), &options, &[7; 32]);
        assert_ne!(
            baseline,
            ptx_artifact_digest_parts(NamedInput::new("other.ptx", b"ptx"), &options, &[7; 32])
        );
        assert_ne!(
            baseline,
            ptx_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"changed"),
                &options,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ptx_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx"),
                &options.clone().with_fma_contraction(false),
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ptx_artifact_digest_parts(NamedInput::new("kernel.ptx", b"ptx"), &options, &[8; 32])
        );
    }

    #[test]
    fn ptx_normalization_ignores_one_terminator_and_rejects_interior_nuls() {
        let plain = NamedInput::new("kernel.ptx", b"ptx");
        let terminated = NamedInput::new("kernel.ptx", b"ptx\0");
        assert_eq!(logical_ptx(plain).unwrap(), b"ptx");
        assert_eq!(logical_ptx(terminated).unwrap(), b"ptx");
        assert!(matches!(
            logical_ptx(NamedInput::new("bad.ptx", b"p\0tx")),
            Err(FinalizerError::InteriorNulPtx { ref name }) if name == "bad.ptx"
        ));
        assert!(matches!(
            logical_ptx(NamedInput::new("bad.ptx", b"ptx\0\0")),
            Err(FinalizerError::InteriorNulPtx { ref name }) if name == "bad.ptx"
        ));
        assert!(matches!(
            logical_ptx(NamedInput::new("empty.ptx", b"\0")),
            Err(FinalizerError::EmptyInput { .. })
        ));
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    const ACQUIRE_LOAD_PTX: &[u8] = br#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry acquire_load(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .b32 value;
    .reg .b64 input_ptr;
    .reg .b64 output_ptr;

    ld.param.u64 input_ptr, [input];
    ld.param.u64 output_ptr, [output];
    ld.acquire.gpu.global.u32 value, [input_ptr];
    st.global.u32 [output_ptr], value;
    ret;
}
"#;

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit nvJitLink"]
    fn live_ptx_pipeline_compiles_acquire_load_to_cubin() {
        let linker = LtoLinker::discover().unwrap();
        let options = FinalizationOptions::new("sm_80".parse().unwrap());
        let cubin = linker
            .link_ptx_to_cubin(
                NamedInput::new("acquire-load.ptx", ACQUIRE_LOAD_PTX),
                &options,
            )
            .unwrap();
        assert!(is_valid_cubin(&cubin));
    }
}
