/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assembler discovery and bytecode-version negotiation, shared by the
//! workspace tools and by in-tree consumers of this crate. It must not
//! depend on a CUDA driver: standalone bytecode tests and examples use the
//! same path as JIT.

use crate::bytecode::{write_bytecode_version, BytecodeVersion};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};
use uuid::Uuid;

/// `CUTILE_BYTECODE_VERSION`: an explicit bytecode version
/// (`major.minor[.tag]`), checked against the writer and the assembler and
/// never downgraded.
pub const BYTECODE_VERSION_ENV: &str = "CUTILE_BYTECODE_VERSION";
/// `CUTILE_TILEIRAS_PATH`: the assembler executable, taking precedence over
/// the toolkit roots below.
pub const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";
/// `CUDA_TOOLKIT_PATH`: a toolkit root whose `bin/tileiras` is used when no
/// executable is named.
pub const CUDA_TOOLKIT_PATH_ENV: &str = "CUDA_TOOLKIT_PATH";
/// `CUDA_HOME`: the toolkit root tried after `CUDA_TOOLKIT_PATH`.
pub const CUDA_HOME_ENV: &str = "CUDA_HOME";

/// Selection for the standalone IR tools. The runtime supplies its richer
/// toolkit discovery result directly to `negotiate_bytecode_version`.
pub fn tileiras_from_env() -> PathBuf {
    env::var_os(TILEIRAS_PATH_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            [CUDA_TOOLKIT_PATH_ENV, CUDA_HOME_ENV]
                .iter()
                .find_map(|var| {
                    env::var_os(var)
                        .filter(|value| !value.is_empty())
                        .map(|root| {
                            PathBuf::from(root).join("bin").join(if cfg!(windows) {
                                "tileiras.exe"
                            } else {
                                "tileiras"
                            })
                        })
                })
        })
        .unwrap_or_else(|| PathBuf::from("tileiras"))
}

/// Select only formats both the writer and the exact executable accept.
/// An explicit request is checked, never silently downgraded.
///
/// Automatic selection stops at [`BytecodeVersion::CURRENT`]: versions the
/// writer can emit beyond it (13.5, whose only purpose here is public-preview
/// library content) are used only when requested explicitly.
pub fn negotiate_bytecode_version(
    tileiras: &Path,
    requested: Option<&std::ffi::OsStr>,
) -> std::io::Result<BytecodeVersion> {
    let requested = requested
        .map(|text| {
            let text = text.to_string_lossy();
            let version = parse_bytecode_version(&text).ok_or_else(|| {
                std::io::Error::other(format!(
                    "invalid {BYTECODE_VERSION_ENV}={text}: expected major.minor[.tag]"
                ))
            })?;
            if !BytecodeVersion::SUPPORTED.contains(&version) {
                return Err(std::io::Error::other(format!(
                    "{BYTECODE_VERSION_ENV}={version} is not supported by the cutile-ir writer"
                )));
            }
            Ok(version)
        })
        .transpose()?;

    // Recent assemblers enumerate their supported input formats. Do not
    // infer this list from --version: tool and bytecode versions are distinct.
    let versions = list_bytecode_versions(tileiras).unwrap_or_else(|e| {
        emit_setup_diagnostic(format_args!("{e}; falling back to compile probes"));
        Vec::new()
    });
    if !versions.is_empty() {
        if let Some(version) = requested {
            if versions.contains(&version) {
                return Ok(version);
            }
            return Err(std::io::Error::other(format!(
                "{BYTECODE_VERSION_ENV}={version} is not accepted by {} (supports {})",
                tileiras.display(),
                versions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return default_candidates()
            .find(|v| versions.contains(v))
            .ok_or_else(|| std::io::Error::other(format!(
                "{} accepts none of the bytecode versions this compiler can emit; cuTile requires CUDA 13.2 or newer (assembler supports {})",
                tileiras.display(), versions.iter().map(ToString::to_string)
                    .collect::<Vec<_>>().join(", ")
            )));
    }
    // Older/custom assemblers may not implement --list-versions. Compile a
    // representative image, not an empty module, against the same executable.
    let candidates =
        requested.map_or_else(|| default_candidates().collect(), |version| vec![version]);
    probe_bytecode_versions(tileiras, &candidates)
}

/// The bytecode versions `tileiras --list-versions` names, one per line.
///
/// A launch failure (no such file, not executable) is returned as the error
/// it is, with the path in the message and the kind kept. An assembler that
/// ran but predates the flag exits non-zero or prints a usage message
/// instead of versions; both give an empty list, which
/// [`negotiate_bytecode_version`] answers with compile probes. Lines that
/// are not versions are skipped: a development assembler names `internal`
/// after its numbered versions (the CUDA 13.6 build lists `13.1` through
/// `13.6` and then `internal`), and a usage message has no version line at
/// all, so it still gives an empty list.
pub fn list_bytecode_versions(tileiras: &Path) -> std::io::Result<Vec<BytecodeVersion>> {
    let output = Command::new(tileiras)
        .arg("--list-versions")
        .output()
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("could not run {} --list-versions: {e}", tileiras.display()),
            )
        })?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.lines().filter_map(parse_bytecode_version).collect())
}

/// Versions tried without an explicit request, newest first, capped at the
/// writer's default.
fn default_candidates() -> impl Iterator<Item = BytecodeVersion> {
    BytecodeVersion::SUPPORTED
        .iter()
        .rev()
        .copied()
        .filter(|v| *v <= BytecodeVersion::CURRENT)
}

/// Parses an exact version; an explicit unsupported request is never clamped.
fn parse_bytecode_version(text: &str) -> Option<BytecodeVersion> {
    let mut parts = text.trim().split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next()?.trim().parse().ok()?;
    let tag = match parts.next() {
        Some(part) => part.trim().parse().ok()?,
        None => 0,
    };
    if parts.next().is_some() {
        return None;
    }
    Some(BytecodeVersion { major, minor, tag })
}

/// Builds the version-probe module: one entry with a pointer parameter, a
/// token, a tensor/partition view, and a `cuda_tile.for` region whose body
/// loads through the view. An EMPTY module is not a valid probe — an older
/// `tileiras` accepted a newer version's empty bytecode while rejecting the
/// same version's region encoding, so the probe selected a version real
/// kernels could not compile at (grout B200 evaluation, 2026-08). The probe
/// must contain the independently versioned encodings a real kernel has:
/// regions were the construct that caught it, and view/token types are the
/// other independently versioned family (2026-08-18 review, R1). If a
/// version-gated encoding is ever added outside these families, extend this
/// module alongside it.
fn build_probe_module() -> crate::Module {
    use crate::builder::{append_op, build_single_block_region, OpBuilder};
    use crate::bytecode::Opcode;
    use crate::ir::{
        Attribute, DenseElements, FuncType, Location, Module, PartitionViewType, PointerType,
        ScalarType, TensorViewType, TileElementType, TileType, Type,
    };

    let tile_i32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I32),
        shape: vec![],
    });
    let tile_ptr_f32 = Type::Tile(TileType {
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
        shape: vec![],
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![128],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![16],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![128],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let tile_16_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![16],
    });
    let mut module = Module::new("__cutile_probe");
    let (region_id, block_id, entry_args) =
        build_single_block_region(&mut module, std::slice::from_ref(&tile_ptr_f32));
    let const_i32 = |module: &mut Module, val: i32| {
        let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
            .attr(
                "value",
                Attribute::DenseElements(DenseElements {
                    element_type: tile_i32.clone(),
                    shape: vec![],
                    data: val.to_le_bytes().to_vec(),
                }),
            )
            .result(tile_i32.clone())
            .build(module);
        append_op(module, block_id, op);
        res[0]
    };
    let (tok_op, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
        .result(Type::Token)
        .build(&mut module);
    append_op(&mut module, block_id, tok_op);
    let seg_i32 = |n: i64| Attribute::Integer(n, tile_i32.clone());
    let (mtv, mtv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
        .operand(entry_args[0])
        .result(tv_ty)
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![seg_i32(1), seg_i32(0), seg_i32(0)]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, mtv);
    let (mpv, mpv_res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
        .operand(mtv_res[0])
        .result(pv_ty)
        .build(&mut module);
    append_op(&mut module, block_id, mpv);
    let lb = const_i32(&mut module, 0);
    let ub = const_i32(&mut module, 4);
    let step = const_i32(&mut module, 1);
    let (body_region, body_blk, body_args) =
        build_single_block_region(&mut module, std::slice::from_ref(&tile_i32));
    // The load sits INSIDE the region and references parent-scope values
    // (view, token) plus the block argument — the cross-region encoding a
    // real kernel exercises.
    let (load, _) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
        .operand(mpv_res[0])
        .operand(body_args[0])
        .operand(tok_res[0])
        .attr("memory_ordering_semantics", seg_i32(0))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![seg_i32(1), seg_i32(1), seg_i32(1)]),
        )
        .result(tile_16_f32)
        .result(Type::Token)
        .build(&mut module);
    append_op(&mut module, body_blk, load);
    let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(&mut module);
    append_op(&mut module, body_blk, cont);
    let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
        .operand(lb)
        .operand(ub)
        .operand(step)
        .region(body_region)
        .build(&mut module);
    append_op(&mut module, block_id, for_op);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("__cutile_probe_entry".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![tile_ptr_f32],
                results: vec![],
            })),
        )
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

/// Probes `tileiras` for the newest bytecode version it accepts by compiling a
/// tiny but REPRESENTATIVE module (an entry with a `for` region) at each
/// candidate version, newest first.
///
/// Only a `tileiras` that ran to completion and exited non-zero counts as a
/// rejection of that version. Anything environmental — the probe image
/// cannot be written, the binary cannot be launched, or it dies from a
/// signal — is an error: the former fallback to `MIN_SUPPORTED` in those
/// cases handed the real kernel to a toolchain nothing had been verified
/// against, and produced the confusing failure there instead of here.
/// Rejection of every supported version is an error too, carrying the
/// assembler's own diagnostic.
#[cfg(test)]
fn probe_max_supported_bytecode_version(tileiras: &Path) -> std::io::Result<BytecodeVersion> {
    probe_bytecode_versions(
        tileiras,
        &BytecodeVersion::SUPPORTED
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>(),
    )
}

fn probe_bytecode_versions(
    tileiras: &Path,
    versions: &[BytecodeVersion],
) -> std::io::Result<BytecodeVersion> {
    let tmp_dir = env::temp_dir();
    let mut last_rejection = String::new();
    for &version in versions {
        let module = build_probe_module();
        let bytes = write_bytecode_version(&module, version).map_err(|e| {
            std::io::Error::other(format!(
                "internal: the bytecode-version probe module does not encode at {version}: {e}"
            ))
        })?;
        let base = tmp_dir.join(Uuid::new_v4().to_string());
        let bc_file = ScopedTempFile::new(base.with_extension("bc"));
        let cubin_file = ScopedTempFile::new(base.with_extension("cubin"));
        let bc_filename = bc_file.path().to_string_lossy().into_owned();
        let cubin_filename = cubin_file.path().to_string_lossy().into_owned();
        std::fs::write(bc_file.path(), &bytes).map_err(|e| {
            std::io::Error::other(format!(
                "cannot write the bytecode-version probe to {bc_filename}: {e} \
                 (the temporary directory {} must be writable to run tileiras)",
                tmp_dir.display()
            ))
        })?;
        let args = ["--gpu-name", "sm_120", "-o", &cubin_filename, &bc_filename];
        let output = Command::new(tileiras).args(args).output().map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("failed to launch tileiras {}: {e}", tileiras.display()),
            )
        })?;
        if output.status.success() {
            return Ok(version);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.code().is_none() {
            // Killed by a signal: the binary crashed, it did not judge the
            // image.
            return Err(std::io::Error::other(format!(
                "{} crashed ({}) while probing bytecode version {version}.\n\
                 command: {tileiras:?} {args:?}\n\
                 stderr:\n{stderr}",
                tileiras.display(),
                output.status,
            )));
        }
        emit_setup_diagnostic(format_args!(
            "{} rejected bytecode version {version} ({})",
            tileiras.display(),
            output.status
        ));
        last_rejection = format!("{version}: {}", stderr.trim());
    }
    Err(std::io::Error::other(format!(
        "{} accepts none of the bytecode versions this compiler can emit ({}).\n\
         last rejection ({last_rejection})\n\
         hint: cuTile requires CUDA 13.2+; set {CUDA_TOOLKIT_PATH_ENV}/{CUDA_HOME_ENV} or \
         {TILEIRAS_PATH_ENV} to a matching toolkit, or force a version with {BYTECODE_VERSION_ENV}.",
        tileiras.display(),
        BytecodeVersion::SUPPORTED
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    )))
}

struct ScopedTempFile(PathBuf);
impl ScopedTempFile {
    fn new(path: PathBuf) -> Self {
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for ScopedTempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn emit_setup_diagnostic(args: std::fmt::Arguments<'_>) {
    if env::var("CUTILE_SETUP_DIAGNOSTICS").is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }) {
        eprintln!("cutile setup: {args}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    #[cfg(unix)]
    fn advertised_versions_select_the_newest_common_format() {
        let path = env::temp_dir().join(format!("cutile_advertised_{}", Uuid::new_v4()));
        let _cleanup = ScopedTempFile::new(path.clone());
        for (versions, expected) in [
            ("13.1\\n13.2\\n", BytecodeVersion::V13_2),
            ("13.1\\n13.2\\n13.3\\n", BytecodeVersion::V13_3),
            (
                "13.1\\n13.2\\n13.3\\n13.4\\n13.5\\n",
                BytecodeVersion::V13_4,
            ),
        ] {
            // Answer only `--list-versions`; reject compile probes, so a
            // fallthrough to probing fails loudly instead of accepting the
            // newest version (seen once on CI when the list launch failed).
            write_fake_tileiras_script(
                &path,
                &format!(
                    "case \"$1\" in --list-versions) printf '{versions}';; *) exit 1;; esac\n"
                ),
            );
            assert_eq!(negotiate_bytecode_version(&path, None).unwrap(), expected);
            assert_eq!(
                negotiate_bytecode_version(&path, Some("13.2".as_ref())).unwrap(),
                BytecodeVersion::V13_2
            );
        }
        write_fake_tileiras_script(&path, "printf '13.1\\n'\n");
        assert!(negotiate_bytecode_version(&path, None).is_err());
    }

    /// `list_bytecode_versions` separates "cannot run" from "lists nothing":
    /// a missing executable is an error, an assembler that rejects or does
    /// not understand the flag is an empty list.
    #[test]
    #[cfg(unix)]
    fn listing_versions_separates_launch_failures_from_empty_listings() {
        let missing = env::temp_dir().join(format!("cutile_missing_lister_{}", Uuid::new_v4()));
        let error = list_bytecode_versions(&missing).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert!(error.to_string().contains("--list-versions"), "{error}");

        let path = env::temp_dir().join(format!("cutile_lister_{}", Uuid::new_v4()));
        let _cleanup = ScopedTempFile::new(path.clone());
        write_fake_tileiras_script(&path, "printf '13.4\\n 13.5 \\n\\n'\n");
        assert_eq!(
            list_bytecode_versions(&path).unwrap(),
            [BytecodeVersion::V13_4, BytecodeVersion::V13_5]
        );
        // A development assembler names `internal` after its numbered
        // versions, and a version newer than this crate knows is still a
        // version; neither empties the list.
        write_fake_tileiras_script(&path, "printf '13.4\\n13.5\\n13.6\\ninternal\\n'\n");
        assert_eq!(
            list_bytecode_versions(&path)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["13.4", "13.5", "13.6"]
        );
        for body in ["exit 1\n", "printf 'usage: tileiras [options]\\n'\n", ":\n"] {
            write_fake_tileiras_script(&path, body);
            assert!(
                list_bytecode_versions(&path).unwrap().is_empty(),
                "{body:?}"
            );
        }
    }

    /// 13.5 is emittable (public-preview libraries need it) but beyond the
    /// default: an assembler that lists it still negotiates 13.4 on its own,
    /// and only an explicit request selects it.
    #[test]
    #[cfg(unix)]
    fn versions_past_the_default_need_an_explicit_request() {
        let path = env::temp_dir().join(format!("cutile_preview_{}", Uuid::new_v4()));
        let _cleanup = ScopedTempFile::new(path.clone());
        write_fake_tileiras_script(
            &path,
            "case \"$1\" in --list-versions) printf '13.4\\n13.5\\n';; *) exit 1;; esac\n",
        );
        assert_eq!(
            negotiate_bytecode_version(&path, None).unwrap(),
            BytecodeVersion::CURRENT
        );
        assert_eq!(
            negotiate_bytecode_version(&path, Some("13.5".as_ref())).unwrap(),
            BytecodeVersion::V13_5
        );
        write_fake_tileiras_script(
            &path,
            "case \"$1\" in --list-versions) printf '13.4\\n';; *) exit 1;; esac\n",
        );
        assert!(negotiate_bytecode_version(&path, Some("13.5".as_ref())).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn legacy_assembler_is_probed_and_explicit_versions_are_not_downgraded() {
        let path = env::temp_dir().join(format!("cutile_legacy_{}", Uuid::new_v4()));
        let _cleanup = ScopedTempFile::new(path.clone());
        // Like 13.2: --list-versions fails without a diagnostic. For compile
        // probes inspect the emitted header rather than accepting an empty
        // file or whichever candidate happened to be tried first.
        write_fake_tileiras_script(
            &path,
            r#"
if [ "$1" = --list-versions ]; then exit 1; fi
minor=$(od -An -tu1 -j9 -N1 "$5" | tr -d ' ')
[ "$minor" = 2 ]
"#,
        );
        assert_eq!(
            negotiate_bytecode_version(&path, None).unwrap(),
            BytecodeVersion::V13_2
        );
        assert_eq!(
            negotiate_bytecode_version(&path, Some("13.2".as_ref())).unwrap(),
            BytecodeVersion::V13_2
        );
        for request in ["13.3", "13.4", "13.1", "13.9", "13.2.3", "nonsense"] {
            assert!(
                negotiate_bytecode_version(&path, Some(request.as_ref())).is_err(),
                "{request}"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn failed_discovery_is_not_reported_as_baseline_support() {
        let path = env::temp_dir().join(format!("cutile_broken_{}", Uuid::new_v4()));
        let _cleanup = ScopedTempFile::new(path.clone());
        assert_eq!(
            negotiate_bytecode_version(&path, None).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
        for body in [
            "exit 1\n",
            "kill -SEGV $$\n",
            "printf 'not a version\\n'; exit 1\n",
        ] {
            write_fake_tileiras_script(&path, body);
            assert!(negotiate_bytecode_version(&path, None).is_err());
        }
    }
    /// The probe module must be a valid, encodable kernel at every supported
    /// version, and it must contain a `for` region — an EMPTY probe passed a
    /// version the installed tileiras then rejected on real kernels (grout
    /// B200 evaluation, 2026-08).
    #[test]
    fn probe_module_is_representative_and_valid() {
        // Reads the tileiras env resolution: serialize with the tests that
        // mutate it.
        let module = build_probe_module();
        module.verify_dominance().expect("probe module dominance");
        module
            .verify_bytecode_indices()
            .expect("probe module bytecode indices");
        assert!(
            !module.functions.is_empty() && module.num_values() >= 4,
            "the probe must carry an entry with real ops (a `for` region), not \
             be the empty module that once passed a version real kernels failed at"
        );
        for &version in BytecodeVersion::SUPPORTED.iter() {
            write_bytecode_version(&module, version)
                .unwrap_or_else(|e| panic!("probe must encode at {version}: {e}"));
        }
        // When a real tileiras is reachable, the probe must find SOME
        // accepted version (i.e. real-construct bytecode compiles, not just
        // an empty module).
        let tileiras = tileiras_from_env();
        if Command::new(&tileiras).arg("--version").output().is_ok() {
            let version = probe_max_supported_bytecode_version(&tileiras)
                .unwrap_or_else(|e| panic!("probe against {}: {e}", tileiras.display()));
            assert!(
                version >= BytecodeVersion::MIN_SUPPORTED,
                "probe found no accepted version against {}",
                tileiras.display()
            );
        }
    }

    /// A `tileiras` that cannot be launched is an error, not a silent fall
    /// back to `MIN_SUPPORTED`.
    #[test]
    fn probe_reports_an_unlaunchable_tileiras_as_an_error() {
        let missing = env::temp_dir().join(format!("cutile_missing_tileiras_{}", Uuid::new_v4()));
        let err = probe_max_supported_bytecode_version(&missing)
            .expect_err("a missing tileiras must fail the probe");
        assert!(
            err.to_string().contains("failed to launch tileiras"),
            "unexpected error: {err}"
        );
    }

    /// A `tileiras` that dies from a signal did not judge the image: error.
    #[test]
    #[cfg(unix)]
    fn probe_reports_a_crashing_tileiras_as_an_error() {
        let temp_dir = env::temp_dir().join(format!("cutile_crashing_tileiras_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let fake = temp_dir.join("tileiras");
        write_fake_tileiras_script(&fake, "kill -SEGV $$\n");
        let err = probe_max_supported_bytecode_version(&fake)
            .expect_err("a crashing tileiras must fail the probe");
        assert!(
            err.to_string().contains("crashed"),
            "unexpected error: {err}"
        );
        let _ = fs::remove_dir_all(temp_dir);
    }

    /// A `tileiras` that rejects every supported version is an error carrying
    /// its own diagnostic, not a fall back to a version it just refused.
    #[test]
    #[cfg(unix)]
    fn probe_reports_total_rejection_as_an_error() {
        let temp_dir =
            env::temp_dir().join(format!("cutile_rejecting_tileiras_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let fake = temp_dir.join("tileiras");
        write_fake_tileiras_script(&fake, "echo 'unsupported bytecode' >&2\nexit 1\n");
        let err = probe_max_supported_bytecode_version(&fake)
            .expect_err("a tileiras rejecting every version must fail the probe");
        let text = err.to_string();
        assert!(
            text.contains("accepts none of the bytecode versions")
                && text.contains("unsupported bytecode"),
            "unexpected error: {text}"
        );
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn parses_bytecode_version_override() {
        assert_eq!(parse_bytecode_version("13.2"), Some(BytecodeVersion::V13_2));
        assert_eq!(
            parse_bytecode_version(" 13.3 "),
            Some(BytecodeVersion::V13_3)
        );
        assert_eq!(
            parse_bytecode_version("13.3.0"),
            Some(BytecodeVersion::V13_3)
        );
        // Parse exactly; negotiation rejects unsupported requests.
        assert_eq!(
            parse_bytecode_version("13.9"),
            Some(BytecodeVersion {
                major: 13,
                minor: 9,
                tag: 0
            })
        );
        assert_eq!(parse_bytecode_version("13"), None);
        assert_eq!(parse_bytecode_version("nonsense"), None);
        assert_eq!(parse_bytecode_version("13.2.3.4"), None);
    }

    #[cfg(unix)]
    fn write_fake_tileiras_script(path: &Path, body: &str) {
        use std::io::Write;
        use std::process::Stdio;

        // Keep writable executable fds out of this multithreaded process.
        // A concurrent child can inherit one until exec, causing ETXTBSY in
        // another launch even after fs::write has returned. Wait for a separate
        // writer process to exit before executing the fixture.
        let mut writer = Command::new("sh")
            .args([
                "-c",
                "cat > \"$1\" && chmod 755 \"$1\"",
                "write-fake-tileiras",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        writer
            .stdin
            .take()
            .unwrap()
            .write_all(format!("#!/bin/sh\n{body}").as_bytes())
            .unwrap();
        assert!(writer.wait().unwrap().success());
    }
}
