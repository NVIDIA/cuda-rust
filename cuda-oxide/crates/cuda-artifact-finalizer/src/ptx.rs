/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Standalone assembly of an already-linked PTX module with toolkit `ptxas`.
//!
//! This route is deliberately separate from [`crate::Finalizer::discover`].
//! Consumers that only finalize NVVM IR therefore do not need an external
//! executable, while PTX producers can request the same typed target, FMA,
//! debug, validation, diagnostics, and provenance policy without loading the
//! CUDA Driver.

use crate::diagnostics::parse_ptxas_resource_usage;
use crate::link::logical_ptx;
use crate::nvvm::report_changed_tool;
use crate::provenance::{
    StableDigest, ToolFileIdentity, digest_file_handle, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::{FinalizationOptions, FinalizerError, LinkReport, NamedInput, is_valid_cubin};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;

static TEMP_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new(prefix: &str) -> Result<Self, FinalizerError> {
        let root = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for _ in 0..128 {
            let sequence = TEMP_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = root.join(format!(
                "{prefix}-{}-{timestamp:x}-{sequence:x}",
                std::process::id()
            ));
            let builder = fs::DirBuilder::new();
            #[cfg(unix)]
            let mut builder = builder;
            #[cfg(unix)]
            builder.mode(0o700);
            match builder.create(&candidate) {
                Ok(()) => return Ok(Self { path: candidate }),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(source) => {
                    return Err(FinalizerError::Io {
                        path: candidate,
                        source,
                    });
                }
            }
        }
        Err(FinalizerError::Io {
            path: root,
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique PTX assembly directory",
            ),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct PtxasTool {
    path: PathBuf,
    file: File,
    identity: ToolFileIdentity,
    digest: [u8; 32],
    #[cfg(target_os = "linux")]
    execute_from_fd: bool,
}

impl PtxasTool {
    fn open(path: PathBuf) -> Result<Self, FinalizerError> {
        let file = File::open(&path).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;
        if !file
            .metadata()
            .map_err(|source| FinalizerError::Io {
                path: path.clone(),
                source,
            })?
            .is_file()
        {
            return Err(FinalizerError::InvalidPtxas {
                path,
                details: "candidate is not a regular file".to_string(),
            });
        }
        let identity =
            ToolFileIdentity::capture(&file).ok_or_else(|| FinalizerError::InvalidPtxas {
                path: path.clone(),
                details: "could not capture a stable file identity".to_string(),
            })?;
        let digest = digest_file_handle(&file).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;

        #[cfg(target_os = "linux")]
        let execute_from_fd = {
            let mut magic = [0_u8; 4];
            file.read_exact_at(&mut magic, 0).is_ok() && magic == *b"\x7fELF"
        };

        let tool = Self {
            path,
            file,
            identity,
            digest,
            #[cfg(target_os = "linux")]
            execute_from_fd,
        };
        tool.validate_version()?;
        Ok(tool)
    }

    fn validate_version(&self) -> Result<(), FinalizerError> {
        let output = self.invoke([OsStr::new("--version")])?;
        let details = combined_diagnostics(&output);
        let recognized = output.status.success()
            && details
                .to_ascii_lowercase()
                .contains("ptx optimizing assembler");
        if recognized {
            Ok(())
        } else {
            Err(FinalizerError::InvalidPtxas {
                path: self.path.clone(),
                details: if details.is_empty() {
                    format!("version probe exited with {}", output.status)
                } else {
                    details
                },
            })
        }
    }

    fn invoke<I, S>(&self, args: I) -> Result<Output, FinalizerError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let args = args
            .into_iter()
            .map(|arg| arg.as_ref().to_owned())
            .collect::<Vec<_>>();
        with_revalidated_tool_identity(
            "ptxas",
            Some(self.digest),
            || self.current_digest(),
            || {
                let mut command = self.command();
                command.args(&args);
                run_tolerating_busy_text_file(&mut command).map_err(|source| FinalizerError::Io {
                    path: self.path.clone(),
                    source,
                })
            },
        )
    }

    fn current_digest(&self) -> Option<[u8; 32]> {
        if !self.identity.matches_file(&self.file) {
            return None;
        }

        #[cfg(target_os = "linux")]
        if self.execute_from_fd {
            return Some(self.digest);
        }

        let current = File::open(&self.path).ok()?;
        self.identity.matches_file(&current).then_some(self.digest)
    }

    fn command(&self) -> Command {
        #[cfg(target_os = "linux")]
        if self.execute_from_fd {
            return Command::new(format!("/proc/self/fd/{}", self.file.as_raw_fd()));
        }

        Command::new(&self.path)
    }
}

/// Runs `command`, retrying briefly when the kernel reports ETXTBSY.
///
/// Executing a just-written tool by path can collide with an unrelated
/// `Command` spawn on another thread: the concurrent fork inherits the
/// writer's still-open descriptor for the moment before its own exec, and
/// exec of the tool during that moment fails with "Text file busy". The
/// descriptor vanishes as soon as that child execs, so a short bounded
/// retry rides out the collision while a persistent error still surfaces.
fn run_tolerating_busy_text_file(command: &mut Command) -> io::Result<Output> {
    let mut delay = Duration::from_millis(2);
    for _ in 0..8 {
        match command.output() {
            Err(error) if error.kind() == io::ErrorKind::ExecutableFileBusy => {
                thread::sleep(delay);
                delay = delay.saturating_mul(2);
            }
            result => return result,
        }
    }
    command.output()
}

/// Driver-independent assembler for one already-linked PTX module.
///
/// Discovery opens and hashes the selected `ptxas` executable. On Linux, a
/// native ELF executable is subsequently invoked through that retained file
/// descriptor, so replacing its pathname cannot change the compiler used by
/// an in-flight or cached operation. Each assembly receives its own temporary
/// directory and can run concurrently with other calls.
#[derive(Clone)]
pub struct PtxAssembler {
    tool: Arc<PtxasTool>,
}

impl PtxAssembler {
    /// Discover `ptxas` without loading libNVVM, nvJitLink, or the Driver.
    ///
    /// Search order is `CUDA_OXIDE_PTXAS`, toolkit roots selected by
    /// `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, or `CUDA_PATH`, conventional toolkit
    /// roots, then `PATH`.
    pub fn discover() -> Result<Self, FinalizerError> {
        let (candidates, explicit) = ptxas_candidates(|name| std::env::var_os(name));
        let mut tried = Vec::new();
        let mut first_error = None;
        for (index, path) in candidates.into_iter().enumerate() {
            tried.push(path.display().to_string());
            if !path.is_file() {
                continue;
            }
            match Self::from_path(path) {
                Ok(assembler) => return Ok(assembler),
                Err(error) if explicit && index == 0 => return Err(error),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Err(FinalizerError::PtxasNotFound {
            tried: tried.join("\n  "),
        })
    }

    fn from_path(path: PathBuf) -> Result<Self, FinalizerError> {
        Ok(Self {
            tool: Arc::new(PtxasTool::open(path)?),
        })
    }

    /// Path from which the pinned assembler was discovered.
    pub fn ptxas_path(&self) -> &Path {
        &self.tool.path
    }

    /// Digest of the exact assembler executable, if its identity still holds.
    pub fn ptxas_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.current_digest();
        if digest.is_none() {
            report_changed_tool("ptxas");
        }
        digest
    }

    /// Digest every semantic input to standalone PTX assembly.
    pub fn artifact_digest(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Option<[u8; 32]>, FinalizerError> {
        crate::validate_name(input.name)?;
        let logical = logical_ptx(input)?;
        let Some(ptxas) = self.ptxas_digest() else {
            return Ok(None);
        };
        Ok(Some(ptx_assembly_artifact_digest_parts(
            NamedInput::new(input.name, logical),
            options,
            &ptxas,
        )))
    }

    /// Assemble one PTX module into a validated target-specific cubin.
    pub fn assemble_ptx(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        Ok(self.assemble_ptx_impl(input, options, false)?.image)
    }

    /// Assemble one PTX module and collect per-kernel resource diagnostics.
    pub fn assemble_ptx_with_report(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<LinkReport, FinalizerError> {
        self.assemble_ptx_impl(input, options, true)
    }

    fn assemble_ptx_impl(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
        collect_resource_usage: bool,
    ) -> Result<LinkReport, FinalizerError> {
        crate::validate_name(input.name)?;
        let logical = logical_ptx(input)?;
        let directory = TemporaryDirectory::new("cuda-oxide-ptxas")?;
        let input_path = directory.path().join("module.ptx");
        let output_path = directory.path().join("module.cubin");
        fs::write(&input_path, logical).map_err(|source| FinalizerError::Io {
            path: input_path.clone(),
            source,
        })?;

        let mut arguments = options
            .ptxas_options()
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        if collect_resource_usage {
            arguments.push(OsString::from("--verbose"));
        }
        arguments.push(OsString::from("--output-file"));
        arguments.push(output_path.as_os_str().to_owned());
        arguments.push(input_path.as_os_str().to_owned());

        let output = self.tool.invoke(&arguments)?;
        let diagnostics = combined_diagnostics(&output);
        if !output.status.success() {
            return Err(FinalizerError::PtxasFailed {
                status: output.status.to_string(),
                diagnostics,
            });
        }

        let image = fs::read(&output_path).map_err(|source| FinalizerError::Io {
            path: output_path,
            source,
        })?;
        if !is_valid_cubin(&image) {
            return Err(FinalizerError::InvalidCubin);
        }

        let info_log = collect_resource_usage.then_some(diagnostics);
        let resource_usage = info_log
            .as_deref()
            .map(parse_ptxas_resource_usage)
            .unwrap_or_default();
        Ok(LinkReport {
            image,
            info_log,
            resource_usage,
        })
    }
}

fn ptxas_candidates(mut get_env: impl FnMut(&str) -> Option<OsString>) -> (Vec<PathBuf>, bool) {
    let mut candidates = Vec::new();
    let explicit = get_env("CUDA_OXIDE_PTXAS");
    if let Some(path) = explicit.as_ref() {
        push_unique(&mut candidates, PathBuf::from(path));
    }

    let executable = if cfg!(windows) { "ptxas.exe" } else { "ptxas" };
    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = get_env(variable) {
            push_unique(
                &mut candidates,
                PathBuf::from(root).join("bin").join(executable),
            );
        }
    }
    #[cfg(unix)]
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        push_unique(
            &mut candidates,
            PathBuf::from(root).join("bin").join(executable),
        );
    }
    if let Some(path) = get_env("PATH") {
        for directory in std::env::split_paths(&path) {
            push_unique(&mut candidates, directory.join(executable));
        }
    }
    (candidates, explicit.is_some())
}

fn push_unique(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.contains(&candidate) {
        paths.push(candidate);
    }
}

fn combined_diagnostics(output: &Output) -> String {
    let mut diagnostics = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !diagnostics.is_empty() && !diagnostics.ends_with('\n') {
            diagnostics.push('\n');
        }
        diagnostics.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    diagnostics
}

pub(crate) fn ptx_assembly_artifact_digest_parts(
    input: NamedInput<'_>,
    options: &FinalizationOptions,
    ptxas_digest: &[u8; 32],
) -> [u8; 32] {
    let input = NamedInput::new(
        input.name,
        input.bytes.strip_suffix(&[0]).unwrap_or(input.bytes),
    );
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ptx-to-cubin-standalone")
        .field("input-name", input.name.as_bytes())
        .field("input", input.bytes);
    for option in options.ptxas_options() {
        digest = digest.field("ptxas-option", option.as_bytes());
    }
    digest.field("ptxas-sha256", ptxas_digest).finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[cfg(unix)]
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    fn options() -> FinalizationOptions {
        FinalizationOptions::new("sm_103a".parse().unwrap())
    }

    #[cfg(unix)]
    fn fake_ptxas(body: &str, fixture: Option<&[u8]>) -> (TemporaryDirectory, PathBuf) {
        let directory = TemporaryDirectory::new("cuda-oxide-fake-ptxas").unwrap();
        let path = directory.path().join("ptxas");
        let script = format!(
            "#!/bin/sh\n\
             if [ \"$1\" = \"--version\" ]; then\n\
               echo 'ptxas: NVIDIA (R) Ptx optimizing assembler'\n\
               exit 0\n\
             fi\n\
             {body}\n"
        );
        let mut executable = File::create(&path).unwrap();
        executable.write_all(script.as_bytes()).unwrap();
        executable.sync_all().unwrap();
        drop(executable);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        if let Some(bytes) = fixture {
            let mut fixture = File::create(directory.path().join("fixture.cubin")).unwrap();
            fixture.write_all(bytes).unwrap();
            fixture.sync_all().unwrap();
        }
        (directory, path)
    }

    #[cfg(unix)]
    fn successful_fake_ptxas() -> (TemporaryDirectory, PathBuf) {
        fake_ptxas(
            r#"
output=
input=
target=no
fmad=no
verbose=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-name=sm_103a) target=yes ;;
    --fmad=true) fmad=yes ;;
    --verbose) verbose=yes ;;
    --output-file) shift; output="$1" ;;
    *.ptx) input="$1" ;;
  esac
  shift
done
if [ "$target" != yes ] || [ "$fmad" != yes ] || [ -z "$output" ] || [ ! -s "$input" ]; then
  echo 'unexpected ptxas invocation' >&2
  exit 9
fi
cp "$(dirname "$0")/fixture.cubin" "$output"
if [ "$verbose" = yes ]; then
  echo "ptxas info    : Compiling entry function 'kernel'" >&2
  echo 'ptxas info    : Used 7 registers, 0 bytes smem' >&2
fi
"#,
            Some(&minimal_cubin()),
        )
    }

    #[cfg(unix)]
    fn minimal_cubin() -> Vec<u8> {
        const ELF64_HEADER_LENGTH: usize = 64;
        const ELF64_SECTION_HEADER_LENGTH: usize = 64;
        const PAYLOAD_LENGTH: usize = 4;
        let section_table_length = 2 * ELF64_SECTION_HEADER_LENGTH;
        let payload_offset = ELF64_HEADER_LENGTH + section_table_length;
        let mut bytes = vec![0; payload_offset + PAYLOAD_LENGTH];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[40..48].copy_from_slice(&(ELF64_HEADER_LENGTH as u64).to_le_bytes());
        bytes[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        bytes[58..60].copy_from_slice(&(ELF64_SECTION_HEADER_LENGTH as u16).to_le_bytes());
        bytes[60..62].copy_from_slice(&2_u16.to_le_bytes());
        let section = ELF64_HEADER_LENGTH + ELF64_SECTION_HEADER_LENGTH;
        bytes[section + 4..section + 8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[section + 24..section + 32].copy_from_slice(&(payload_offset as u64).to_le_bytes());
        bytes[section + 32..section + 40].copy_from_slice(&(PAYLOAD_LENGTH as u64).to_le_bytes());
        bytes[payload_offset..].copy_from_slice(b"CUDA");
        bytes
    }

    #[test]
    fn discovery_order_is_explicit_tool_then_roots_then_path() {
        let executable = if cfg!(windows) { "ptxas.exe" } else { "ptxas" };
        let explicit_path = PathBuf::from("explicit").join(executable);
        let search_path = std::env::join_paths(["first", "second"]).unwrap();
        let environment = HashMap::from([
            ("CUDA_OXIDE_PTXAS", explicit_path.clone().into_os_string()),
            ("CUDA_TOOLKIT_PATH", OsString::from("toolkit")),
            ("CUDA_HOME", OsString::from("home")),
            ("CUDA_PATH", OsString::from("cuda-path")),
            ("PATH", search_path),
        ]);
        let (candidates, explicit) = ptxas_candidates(|name| environment.get(name).cloned());
        assert!(explicit);
        assert_eq!(candidates[0], explicit_path);
        assert_eq!(
            candidates[1],
            PathBuf::from("toolkit").join("bin").join(executable)
        );
        assert_eq!(
            candidates[2],
            PathBuf::from("home").join("bin").join(executable)
        );
        assert_eq!(
            candidates[3],
            PathBuf::from("cuda-path").join("bin").join(executable)
        );
        assert!(candidates.ends_with(&[
            PathBuf::from("first").join(executable),
            PathBuf::from("second").join(executable)
        ]));
    }

    #[test]
    fn assembly_digest_covers_name_bytes_options_and_tool() {
        let options = options();
        let base = ptx_assembly_artifact_digest_parts(
            NamedInput::new("kernel.ptx", b"ptx"),
            &options,
            &[1; 32],
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("other.ptx", b"ptx"),
                &options,
                &[1; 32]
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"changed"),
                &options,
                &[1; 32]
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx"),
                &options.clone().with_fma_contraction(false),
                &[1; 32]
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx"),
                &options,
                &[2; 32]
            )
        );
        assert_eq!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx\0"),
                &options,
                &[1; 32]
            )
        );
    }

    #[test]
    fn standalone_ptx_validation_normalizes_one_terminator_and_rejects_invalid_inputs() {
        assert_eq!(
            logical_ptx(NamedInput::new("kernel.ptx", b"ptx\0")).unwrap(),
            b"ptx"
        );
        assert!(matches!(
            logical_ptx(NamedInput::new("bad.ptx", b"abc\0def")),
            Err(FinalizerError::InteriorNulPtx { ref name }) if name == "bad.ptx"
        ));
        assert!(matches!(
            logical_ptx(NamedInput::new("empty.ptx", b"\0")),
            Err(FinalizerError::EmptyInput { ref name }) if name == "empty.ptx"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_route_translates_options_validates_cubin_and_parses_resources() {
        let (_directory, path) = successful_fake_ptxas();
        let assembler = PtxAssembler::from_path(path).unwrap();
        let report = assembler
            .assemble_ptx_with_report(NamedInput::new("kernel.ptx", b"ptx"), &options())
            .unwrap();
        assert!(is_valid_cubin(&report.image));
        assert_eq!(
            report.resource_usage,
            [crate::KernelResourceUsage {
                kernel: "kernel".to_string(),
                registers: Some(7),
                stack_frame_bytes: 0,
                spill_store_bytes: 0,
                spill_load_bytes: 0,
            }]
        );
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_route_preserves_exit_status_and_diagnostics() {
        let (_directory, path) = fake_ptxas("echo 'synthetic failure' >&2; exit 42", None);
        let assembler = PtxAssembler::from_path(path).unwrap();
        let error = assembler
            .assemble_ptx(NamedInput::new("kernel.ptx", b"ptx"), &options())
            .unwrap_err();
        assert!(matches!(
            error,
            FinalizerError::PtxasFailed { status, diagnostics }
                if status.contains("42") && diagnostics.contains("synthetic failure")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_route_rejects_an_invalid_cubin() {
        let body = r#"
output=
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-file" ]; then shift; output="$1"; fi
  shift
done
echo 'not a cubin' > "$output"
"#;
        let (_directory, path) = fake_ptxas(body, None);
        let assembler = PtxAssembler::from_path(path).unwrap();
        assert!(matches!(
            assembler.assemble_ptx(NamedInput::new("kernel.ptx", b"ptx"), &options()),
            Err(FinalizerError::InvalidCubin)
        ));
    }

    /// Exec of the fake tool must survive a write descriptor that another
    /// process (here: this test) still holds when the version probe runs.
    /// Without the bounded ETXTBSY retry this fails immediately with
    /// "Text file busy", which is the race CI hits when a concurrent
    /// test's spawn inherits a freshly written script's descriptor.
    #[cfg(unix)]
    #[test]
    fn version_probe_survives_a_briefly_held_write_descriptor() {
        let (_directory, path) = fake_ptxas("exit 0", None);
        let held = fs::OpenOptions::new().append(true).open(&path).unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            drop(held);
        });
        let assembler = PtxAssembler::from_path(path);
        release.join().unwrap();
        assert!(assembler.is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_route_uses_isolated_files_for_concurrent_calls() {
        let (_directory, path) = successful_fake_ptxas();
        let assembler = PtxAssembler::from_path(path).unwrap();
        let threads = (0..8)
            .map(|_| {
                let assembler = assembler.clone();
                std::thread::spawn(move || {
                    assembler.assemble_ptx(NamedInput::new("kernel.ptx", b"ptx"), &options())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert!(is_valid_cubin(&thread.join().unwrap().unwrap()));
        }
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    const PTX: &[u8] = br#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry kernel() {
    ret;
}
"#;

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit ptxas"]
    fn live_assembly_emits_a_kernel_cubin_and_resource_report() {
        let assembler = PtxAssembler::discover().unwrap();
        let options = FinalizationOptions::new("sm_80".parse().unwrap());
        assert!(assembler.ptxas_digest().is_some());
        assert!(
            assembler
                .artifact_digest(NamedInput::new("kernel.ptx", PTX), &options)
                .unwrap()
                .is_some()
        );
        let report = assembler
            .assemble_ptx_with_report(NamedInput::new("kernel.ptx", PTX), &options)
            .unwrap();
        assert!(is_valid_cubin(&report.image));
        assert!(
            report
                .image
                .windows(b"kernel".len())
                .any(|bytes| bytes == b"kernel")
        );
        assert!(report.info_log.is_some());
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit ptxas"]
    fn live_assembly_supports_concurrent_invocations() {
        let assembler = PtxAssembler::discover().unwrap();
        let options = FinalizationOptions::new("sm_80".parse().unwrap());
        let threads = (0..4)
            .map(|_| {
                let assembler = assembler.clone();
                let options = options.clone();
                std::thread::spawn(move || {
                    assembler.assemble_ptx(NamedInput::new("kernel.ptx", PTX), &options)
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert!(is_valid_cubin(&thread.join().unwrap().unwrap()));
        }
    }
}
