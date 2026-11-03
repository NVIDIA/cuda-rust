/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `cargo oxide tilelib`: package a crate's `#[tile_export]` functions as a
//! cuTile tile library.
//!
//! ```text
//! emit-ltoir ──> <crate>.ltoir   <crate>.ll   <crate>.tileexports.json
//!                                   │                  │
//!             define @<symbol>(...) checked against abi_params()
//!                                                      │
//!                           TileExportDesc ──> oxide-tilebc Declaration + Implementation
//!                                                      │
//!                <crate>.iface.tilebc   <crate>.impl.tilebc   <crate>.tilelib.mlir
//!                                                      │
//!          tileiras --preview-create-tile-lib (lists 13.5 or newer) ──> <crate>.tilelib
//! ```
//!
//! The descriptions are Tile IR public-preview content, which exists from
//! bytecode 13.5 on and which an assembler reads only at the version it
//! calls current, the newest numbered version its `--list-versions` prints.
//! The command therefore asks the installed tileiras first and writes the
//! descriptions at that version (13.5 for the CUDA 13.5 assembler, and the
//! newer version a newer one names), or at 13.5 when no assembler can read
//! them yet. An older
//! assembler stops the command after the descriptions with an explanation;
//! `--library-optional` makes that outcome a success. Either way the summary
//! ends with one of two lines scripts can read: [`LIBRARY_WRITTEN_MARKER`]
//! or [`LIBRARY_SKIPPED_MARKER`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use oxide_tile_export::{
    AbiParamKind, DESCRIPTIONS_SUFFIX, TileAccess, TileElem, TileExportDesc, TileSync,
};
use oxide_tilebc::{
    Access, BytecodeVersion, Declaration, FragmentArg, FragmentType, Implementation, Layout,
    Library, MemorySpace, ScalarType, Synchronization, TileArg, TileType,
};

use super::*;

/// cuTile's own override for the assembler, honoured here so one variable
/// selects the same `tileiras` for both tools.
pub(super) const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";

/// Suffix of the interface bytecode (`preview$func extern` declarations).
pub(super) const TILELIB_INTERFACE_SUFFIX: &str = "iface.tilebc";
/// Suffix of the implementation catalog bytecode (`preview$extern_impl`).
pub(super) const TILELIB_IMPLEMENTATION_SUFFIX: &str = "impl.tilebc";
/// Suffix of the readable dump of both modules.
pub(super) const TILELIB_DUMP_SUFFIX: &str = "tilelib.mlir";
/// Suffix of the library `tileiras` assembles from the three inputs.
pub(super) const TILELIB_LIBRARY_SUFFIX: &str = "tilelib";

/// Fallback locations probed for `tileiras`, in the idiom of the other
/// toolkit tools. It ships in the toolkit's `bin/` beside `nvcc`.
pub(super) const TILEIRAS_FALLBACK_PATHS: &[&str] = &[
    "/usr/local/cuda/bin/tileiras",
    "/opt/cuda/bin/tileiras",
    "/usr/bin/tileiras",
];

/// The line the summary prints once both bytecode halves exist. Scripts
/// grep for it (the smoketest's tilelib lane), so it is a constant here.
pub(super) const DESCRIPTIONS_WRITTEN_MARKER: &str = "Tile library descriptions written";

/// The line the summary ends with when tileiras wrote the `.tilelib`; the
/// path and size follow it. The smoketest requires the library when it
/// sees this line, so the script and the command share the text.
pub(super) const LIBRARY_WRITTEN_MARKER: &str = "Tile library written to";

/// The line the summary ends with when no library was written, followed by
/// the tileiras that was found and the bytecode versions it lists (or that
/// none was found). The smoketest accepts the lane without a `.tilelib` only
/// when it sees this line.
pub(super) const LIBRARY_SKIPPED_MARKER: &str = "Tile library skipped";

// =============================================================================
// tilelib command
// =============================================================================

/// Package a crate's `#[tile_export]` functions as a cuTile tile library.
///
/// Runs [`emit_ltoir`] for the device half, checks that the emitted NVVM IR
/// defines every implementation symbol with the parameter list the ABI table
/// predicts, maps the descriptions to oxide-tilebc's model, writes the
/// interface and implementation bytecode plus a readable dump, and asks
/// `tileiras --preview-create-tile-lib` for the library when the installed
/// assembler lists bytecode 13.5 or newer, writing the descriptions at its
/// current version. Exits 1 on any error, including a tileiras that cannot
/// be run; exits 2 when no library could be created and `library_optional`
/// was not passed.
#[allow(clippy::too_many_arguments)]
pub fn tilelib(
    ctx: &Context,
    example: &str,
    arch: &str,
    features: Option<&str>,
    output: Option<&Path>,
    library_optional: bool,
    print: bool,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    // Step 1: the device half. Exits on a build or libNVVM failure, and
    // clears stale descriptions and libraries before building.
    emit_ltoir(
        ctx,
        example,
        arch,
        features,
        None,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );

    let paths = TileLibraryPaths::new(&example_dir, example, output);
    let descriptions = read_descriptions(&paths.descriptions, example).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    // Step 2: the LTOIR must define what the descriptions promise. The
    // producer side of tileiras never looks inside the code payload, so a
    // missing or misdeclared symbol would otherwise surface only when a
    // cuTile program is compiled against the library.
    let ll = std::fs::read_to_string(&paths.ll).unwrap_or_else(|error| {
        eprintln!(
            "Error: could not read the emitted NVVM IR at {}: {error}",
            paths.ll.display()
        );
        std::process::exit(1);
    });
    if let Err(problems) = check_implementation_symbols(&ll, &descriptions) {
        eprintln!(
            "Error: {} does not define every implementation the descriptions name:",
            paths.ll.display()
        );
        for problem in problems {
            eprintln!("  {problem}");
        }
        std::process::exit(1);
    }

    // Step 3: descriptions to Tile IR, verified the way the consumer will.
    let stem = artifact_stem(example);
    let library = TileLibrary::from_descriptions(&descriptions)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        })
        .library(&stem);
    library.verify().unwrap_or_else(|error| {
        eprintln!("Error: the tile library does not verify: {error}");
        std::process::exit(1);
    });

    // The assembler decides the bytecode version. Preview content is read
    // only at the version an assembler calls current (the newest numbered
    // version it lists), so the descriptions are written at that version
    // when the installed tileiras can build a library, and at 13.5, the
    // first version with preview content, when no assembler can read them
    // yet. A tileiras that was named (CUTILE_TILEIRAS_PATH, the toolkit
    // root) or found but cannot be started is a configuration error, not an
    // old assembler: only a run that lists nothing means "predates
    // --list-versions". The error names the path and the OS error.
    let tileiras = find_tileiras(ctx);
    let versions = match &tileiras {
        Some(tileiras) => list_bytecode_versions(tileiras).unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }),
        None => Vec::new(),
    };
    let library_version = description_version(&versions);
    let version = library_version.unwrap_or(BytecodeVersion::PREVIEW_FIRST);
    let interface_bytes = library.interface_bytecode(version).unwrap_or_else(|error| {
        eprintln!("Error: could not serialize the interface module: {error}");
        std::process::exit(1);
    });
    let implementation_bytes = library
        .implementation_bytecode(version)
        .unwrap_or_else(|error| {
            eprintln!("Error: could not serialize the implementation module: {error}");
            std::process::exit(1);
        });
    let dump = library_dump(&library, &stem);
    for (path, bytes) in [
        (&paths.interface, interface_bytes.as_slice()),
        (&paths.implementation, implementation_bytes.as_slice()),
        (&paths.dump, dump.as_bytes()),
    ] {
        std::fs::write(path, bytes).unwrap_or_else(|error| {
            eprintln!("Error: could not write {}: {error}", path.display());
            std::process::exit(1);
        });
    }
    if print {
        println!();
        print!("{dump}");
    }

    println!();
    println!(
        "✓ {DESCRIPTIONS_WRITTEN_MARKER}: {} export(s), Tile IR {version} with the preview tag",
        descriptions.len(),
    );
    println!(
        "    {} ({} bytes)",
        paths.interface.display(),
        interface_bytes.len()
    );
    println!(
        "    {} ({} bytes)",
        paths.implementation.display(),
        implementation_bytes.len()
    );
    println!("    {} (readable form of both)", paths.dump.display());

    // Step 4: the library itself, when the installed assembler can build one.
    let (Some(tileiras), Some(version)) = (tileiras.as_deref(), library_version) else {
        explain_no_library(&paths, tileiras.as_deref(), &versions);
        std::process::exit(no_library_exit_code(library_optional));
    };

    let command = create_tile_lib_command(tileiras, &paths);
    println!();
    println!(
        "tileiras {} lists bytecode {version} as its current version; creating the tile library:",
        tileiras.display(),
    );
    println!("  {}", render_command(&command));
    let output = Command::new(&command[0])
        .args(&command[1..])
        .output()
        .unwrap_or_else(|error| {
            eprintln!("Error: could not run {}: {error}", tileiras.display());
            std::process::exit(1);
        });
    if !output.status.success() {
        eprintln!(
            "Error: tileiras --preview-create-tile-lib exited with {}",
            output.status
        );
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
    let size = std::fs::metadata(&paths.library)
        .map(|meta| meta.len())
        .unwrap_or(0);
    println!();
    println!(
        "✓ {LIBRARY_WRITTEN_MARKER} {} ({size} bytes)",
        paths.library.display()
    );
}

/// Every file the command reads or writes for one crate.
pub(super) struct TileLibraryPaths {
    /// The NVVM IR the backend emitted, checked for the implementation symbols.
    pub(super) ll: PathBuf,
    /// The `#[tile_export]` descriptions the backend wrote next to the `.ll`.
    pub(super) descriptions: PathBuf,
    /// The LTOIR `emit_ltoir` wrote; the code payload of the library.
    pub(super) ltoir: PathBuf,
    /// The interface bytecode.
    pub(super) interface: PathBuf,
    /// The implementation catalog bytecode.
    pub(super) implementation: PathBuf,
    /// The readable dump of both modules.
    pub(super) dump: PathBuf,
    /// The library `tileiras` writes; `-o` overrides the default.
    pub(super) library: PathBuf,
}

impl TileLibraryPaths {
    pub(super) fn new(example_dir: &Path, example: &str, library: Option<&Path>) -> Self {
        let stem = artifact_stem(example);
        Self {
            ll: emitted_ll_path(example_dir, example),
            descriptions: example_dir.join(format!("{stem}.{DESCRIPTIONS_SUFFIX}")),
            ltoir: default_ltoir_path(example_dir, example),
            interface: example_dir.join(format!("{stem}.{TILELIB_INTERFACE_SUFFIX}")),
            implementation: example_dir.join(format!("{stem}.{TILELIB_IMPLEMENTATION_SUFFIX}")),
            dump: example_dir.join(format!("{stem}.{TILELIB_DUMP_SUFFIX}")),
            library: library
                .map(Path::to_path_buf)
                .unwrap_or_else(|| example_dir.join(format!("{stem}.{TILELIB_LIBRARY_SUFFIX}"))),
        }
    }
}

/// The descriptions the backend wrote, validated. A crate without
/// `#[tile_export]` functions has no descriptions file at all, which is the
/// common mistake and gets the direct message.
fn read_descriptions(path: &Path, example: &str) -> Result<Vec<TileExportDesc>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "no #[tile_export] functions in {example}: the build wrote no {} (a tile \
                 library needs at least one exported function)",
                path.display()
            ));
        }
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    };
    let descriptions = oxide_tile_export::descriptions_from_json(&text)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
    if descriptions.is_empty() {
        return Err(format!(
            "no #[tile_export] functions in {example}: {} describes no exports",
            path.display()
        ));
    }
    for description in &descriptions {
        description.validate().map_err(|error| {
            format!(
                "description of `{}` in {}: {error}",
                description.decl,
                path.display()
            )
        })?;
    }
    Ok(descriptions)
}

// =============================================================================
// Descriptions to Tile IR
// =============================================================================

/// A crate's exports as the two halves of a tile library.
#[derive(Debug)]
pub(super) struct TileLibrary {
    /// One `preview$func extern` declaration per export.
    pub(super) funcs: Vec<Declaration>,
    /// One `preview$extern_impl` per export.
    pub(super) impls: Vec<Implementation>,
}

impl TileLibrary {
    /// Map every description; the first one that cannot be described stops
    /// the command with its reason.
    pub(super) fn from_descriptions(descriptions: &[TileExportDesc]) -> Result<Self, String> {
        let mut funcs = Vec::with_capacity(descriptions.len());
        let mut impls = Vec::with_capacity(descriptions.len());
        for description in descriptions {
            let (func, implementation) = describe_export(description)
                .map_err(|error| format!("export `{}`: {error}", description.decl))?;
            funcs.push(func);
            impls.push(implementation);
        }
        Ok(Self { funcs, impls })
    }

    /// Both halves under one module name, ready to verify and write.
    pub(super) fn library(&self, name: &str) -> Library {
        Library::new(name, self.funcs.clone(), self.impls.clone())
    }
}

/// One export as its declaration and its implementation.
///
/// Every parameter becomes a tile of the export's shape on the declaration
/// and a fragment on the implementation: a by-value scalar is the rank-0
/// `():()` buffer, everything else a contiguous footprint `(F):(1)`. The
/// fragmap is shared by all parameters, see [`fragmap`]. A `bf16` scalar
/// passed by value needs the `bf16 = i16` frontend ABI entry because
/// cuda-device's bf16 is a `u16` wrapper, so the LLVM parameter is `i16`
/// while the tile element is `bf16`.
pub(super) fn describe_export(
    desc: &TileExportDesc,
) -> Result<(Declaration, Implementation), String> {
    desc.validate().map_err(|error| error.to_string())?;
    let tile_size = desc.shape[0];
    // Every parameter is a fragment of a tile of the same shape and the
    // fragmaps of one implementation share their invocation count, so the
    // footprints have to agree: a 4-element window beside a scalar would
    // need 32 invocations for one argument and 128 for the other.
    let footprint = desc.params[0].footprint;
    if let Some(other) = desc
        .params
        .iter()
        .find(|param| param.footprint != footprint)
    {
        return Err(format!(
            "parameters `{}` (footprint {}) and `{}` (footprint {}) differ; every parameter \
             of one export shares the invocation count, so all footprints must be equal",
            desc.params[0].name, footprint, other.name, other.footprint
        ));
    }
    let mut args = Vec::with_capacity(desc.params.len());
    let mut fragments = Vec::with_capacity(desc.params.len());
    for param in &desc.params {
        let element = scalar_type(param.elem);
        let access = access(param.access);
        args.push(TileArg::new(
            param.name.clone(),
            TileType::new(desc.shape.clone(), element),
            access,
        ));
        let footprint = i64::from(param.footprint);
        let fragmap = fragmap(desc.sync, tile_size, footprint)
            .map_err(|error| format!("parameter `{}`: {error}", param.name))?;
        let (shape, mem_layout) = if param.by_value {
            (vec![1], Layout::scalar_mem())
        } else {
            (vec![footprint], Layout::contiguous_mem(footprint))
        };
        fragments.push(FragmentArg::new(
            param.name.clone(),
            FragmentType::new(shape, element, MemorySpace::Local, mem_layout, fragmap),
            access,
        ));
    }
    let mut implementation = Implementation::new(desc.symbol.clone(), desc.decl.clone(), fragments);
    implementation.synchronization = synchronization(desc.sync);
    implementation.arch = desc.arch.clone();
    if desc
        .params
        .iter()
        .any(|param| param.by_value && param.elem == TileElem::BF16)
    {
        implementation.frontend_abi = vec![("bf16".to_string(), "i16".to_string())];
    }
    Ok((Declaration::new(desc.decl.clone(), args), implementation))
}

/// How one invocation's footprint maps onto the tile.
///
/// Without synchronization every invocation owns `footprint` contiguous
/// elements: `(N, 1, 1):(1, 0, 0)` for a footprint of one, `(N/F, 1, F):(F,
/// 0, 1)` otherwise. With `sync = warp` the 32 lanes of a warp are the
/// participants of one invocation: `(N/32, 32, 1):(32, 1, 0)` for a footprint
/// of one, and `(N/(32F), 32, F):(32F, F, 1)` for a footprint of `F`, so each
/// lane keeps a contiguous footprint and adjacent lanes are adjacent in the
/// tile. The footprint must divide the tile size in either case.
pub(super) fn fragmap(sync: TileSync, tile_size: i64, footprint: i64) -> Result<Layout, String> {
    if footprint <= 0 {
        return Err(format!("footprint {footprint} must be positive"));
    }
    if tile_size % footprint != 0 {
        return Err(format!(
            "footprint {footprint} does not divide the tile size {tile_size}"
        ));
    }
    match sync {
        TileSync::None if footprint == 1 => Ok(Layout::elementwise(tile_size)),
        TileSync::None => Ok(Layout::footprint_blocks(tile_size, footprint)),
        TileSync::Warp => {
            let per_warp = 32 * footprint;
            if tile_size % per_warp != 0 {
                return Err(format!(
                    "sync = warp with footprint {footprint} needs a tile size that is a \
                     multiple of {per_warp}, found {tile_size}"
                ));
            }
            if footprint == 1 {
                Ok(Layout::warp_lanes(tile_size))
            } else {
                Ok(Layout::flat(
                    &[tile_size / per_warp, 32, footprint],
                    &[per_warp, footprint, 1],
                ))
            }
        }
    }
}

/// Tile IR has no unsigned integers; the unsigned Rust types are the signless
/// `iN` of the same width.
fn scalar_type(elem: TileElem) -> ScalarType {
    match elem {
        TileElem::F16 => ScalarType::F16,
        TileElem::BF16 => ScalarType::BF16,
        TileElem::F32 => ScalarType::F32,
        TileElem::F64 => ScalarType::F64,
        TileElem::I8 | TileElem::U8 => ScalarType::I8,
        TileElem::I16 | TileElem::U16 => ScalarType::I16,
        TileElem::I32 | TileElem::U32 => ScalarType::I32,
        TileElem::I64 | TileElem::U64 => ScalarType::I64,
    }
}

fn access(access: TileAccess) -> Access {
    match access {
        TileAccess::In => Access::In,
        TileAccess::Out => Access::Out,
        TileAccess::InOut => Access::InOut,
    }
}

/// The implementation's `synchronization` clause. An export without `sync`
/// gets no clause at all rather than an explicit `synchronization<none>`:
/// the two mean the same thing, and leaving it out keeps the printed and
/// written form identical to an upstream module parsed without the clause.
fn synchronization(sync: TileSync) -> Option<Synchronization> {
    match sync {
        TileSync::None => None,
        TileSync::Warp => Some(Synchronization::Warp),
    }
}

/// The readable form of both modules, one dump so a reader sees the
/// declaration and its implementation together. Lines starting with `//`
/// are comments in the Tile IR textual syntax.
pub(super) fn library_dump(library: &Library, stem: &str) -> String {
    format!(
        "// {stem}.{TILELIB_INTERFACE_SUFFIX}: the interface a cuTile program calls\n{}\
         // {stem}.{TILELIB_IMPLEMENTATION_SUFFIX}: the implementations the LTOIR defines\n{}",
        library.interface_text(),
        library.implementation_text()
    )
}

/// The bytecode versions `tileiras --list-versions` names, one per line.
///
/// A launch failure is returned as the error it is, with the path in the
/// message. An assembler that ran but predates the flag exits non-zero or
/// prints a usage message instead of versions; both give an empty list.
/// Lines that are not versions (an assembler may list a named build next
/// to the numbers) are skipped.
pub(super) fn list_bytecode_versions(tileiras: &Path) -> std::io::Result<Vec<BytecodeVersion>> {
    let output = Command::new(tileiras)
        .arg("--list-versions")
        .output()
        .map_err(|error| {
            std::io::Error::new(
                error.kind(),
                format!(
                    "could not run {} --list-versions: {error}",
                    tileiras.display()
                ),
            )
        })?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(BytecodeVersion::parse)
        .collect())
}

// =============================================================================
// Implementation symbols in the emitted NVVM IR
// =============================================================================

/// One `define` of an LLVM IR module, reduced to what the ABI check reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LlDefine {
    /// The return type, with linkage and calling-convention keywords removed.
    pub(super) ret: String,
    /// One type per parameter, names and parameter attributes removed.
    pub(super) params: Vec<String>,
    /// A linkage keyword from [`LL_LOCAL_LINKAGE`] on the define, if any.
    pub(super) local_linkage: Option<String>,
}

/// Linkages that keep a definition out of what the module exports. The Tile
/// compiler reaches an implementation as `.extern .func`, so a definition
/// with one of these is a reported problem, not a stripped keyword.
const LL_LOCAL_LINKAGE: &[&str] = &["private", "internal", "linkonce", "linkonce_odr"];

/// Keywords that can sit between `define` and the return type without
/// changing whether the symbol can be linked.
const LL_DEFINE_KEYWORDS: &[&str] = &[
    "available_externally",
    "weak",
    "common",
    "appending",
    "extern_weak",
    "weak_odr",
    "external",
    "dso_local",
    "dso_preemptable",
    "default",
    "hidden",
    "protected",
    "ptx_kernel",
    "ptx_device",
    "fastcc",
    "coldcc",
    "ccc",
];

/// Every `define ... @name(params)` of `ll`, keyed by symbol. Textual IR is
/// the robust place to check symbols: the LTOIR is an NVVM bitcode container
/// with no stable public reader, and the `.ll` beside it is what libNVVM
/// compiled.
pub(super) fn ll_defines(ll: &str) -> BTreeMap<String, LlDefine> {
    let mut defines = BTreeMap::new();
    for line in ll.lines() {
        let Some(rest) = line.trim_start().strip_prefix("define ") else {
            continue;
        };
        let Some(at) = rest.find('@') else {
            continue;
        };
        let (prefix, after_at) = rest.split_at(at);
        let after_at = &after_at[1..];
        let Some(open) = after_at.find('(') else {
            continue;
        };
        let name = after_at[..open].trim().trim_matches('"').to_string();
        let Some(params) = balanced_parameter_list(&after_at[open..]) else {
            continue;
        };
        let local_linkage = prefix
            .split_whitespace()
            .find(|token| LL_LOCAL_LINKAGE.contains(token))
            .map(str::to_string);
        let ret = prefix
            .split_whitespace()
            .filter(|token| {
                !LL_DEFINE_KEYWORDS.contains(token) && !LL_LOCAL_LINKAGE.contains(token)
            })
            .collect::<Vec<_>>()
            .join(" ");
        defines.insert(
            name,
            LlDefine {
                ret,
                params: split_top_level(params)
                    .into_iter()
                    .map(parameter_type)
                    .collect(),
                local_linkage,
            },
        );
    }
    defines
}

/// The text inside the parentheses that open `text`, honoring nesting.
fn balanced_parameter_list(text: &str) -> Option<&str> {
    let mut depth = 0i32;
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[1..index]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split on commas outside any bracket pair.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    let last = &text[start..];
    if !last.trim().is_empty() {
        parts.push(last);
    }
    parts
}

/// The type of one parameter: its tokens without the `%name` and without
/// parameter attributes.
fn parameter_type(param: &str) -> String {
    const ATTRIBUTES: &[&str] = &[
        "noundef",
        "nonnull",
        "noalias",
        "nocapture",
        "readonly",
        "writeonly",
        "readnone",
        "signext",
        "zeroext",
        "inreg",
        "returned",
        "nofree",
        "nest",
        "immarg",
    ];
    param
        .split_whitespace()
        .filter(|token| !token.starts_with('%'))
        .filter(|token| !ATTRIBUTES.contains(token))
        .filter(|token| !token.starts_with("align") && !token.starts_with("dereferenceable"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether an LLVM parameter type is the one the ABI predicts. A pointer is
/// the opaque `ptr` on the modern NVVM path and a typed pointer (`i8*`) in
/// the legacy LLVM 7 dialect of pre-Blackwell targets; the pointee type
/// carries no ABI information, so any typed pointer passes.
fn llvm_type_matches(actual: &str, kind: AbiParamKind) -> bool {
    match kind {
        AbiParamKind::Scalar(elem) => actual == elem.llvm_name(),
        AbiParamKind::Pointer(_) => actual == "ptr" || actual.ends_with('*'),
    }
}

/// The type text the message quotes for an expected parameter.
fn expected_type_text(kind: AbiParamKind) -> String {
    match kind {
        AbiParamKind::Scalar(elem) => elem.llvm_name().to_string(),
        AbiParamKind::Pointer(_) => "ptr (i8* on the legacy path)".to_string(),
    }
}

/// Check that `ll` defines every description's implementation symbol as a
/// `void` function whose parameters are the ones [`TileExportDesc::abi_params`]
/// predicts. Every mismatch is reported, symbol by symbol.
pub(super) fn check_implementation_symbols(
    ll: &str,
    descriptions: &[TileExportDesc],
) -> Result<(), Vec<String>> {
    let defines = ll_defines(ll);
    let mut problems = Vec::new();
    for desc in descriptions {
        let Some(define) = defines.get(&desc.symbol) else {
            let defined: Vec<_> = defines.keys().map(|name| format!("@{name}")).collect();
            problems.push(format!(
                "`{}` (implementation of `{}`): no `define ... @{}(` in the module; it defines {}",
                desc.symbol,
                desc.decl,
                desc.symbol,
                if defined.is_empty() {
                    "no functions".to_string()
                } else {
                    defined.join(", ")
                }
            ));
            continue;
        };
        if let Some(linkage) = &define.local_linkage {
            problems.push(format!(
                "`{}`: defined with `{linkage}` linkage, so it is not externally visible; \
                 the Tile compiler links it as .extern .func",
                desc.symbol
            ));
        }
        if define.ret != "void" {
            problems.push(format!(
                "`{}`: returns `{}`, but a tile library implementation returns void and \
                 writes its result through the trailing out pointer",
                desc.symbol, define.ret
            ));
        }
        let expected = desc.abi_params();
        if define.params.len() != expected.len() {
            problems.push(format!(
                "`{}`: {} parameter(s) in the module ({}), {} in the description ({})",
                desc.symbol,
                define.params.len(),
                define.params.join(", "),
                expected.len(),
                expected
                    .iter()
                    .map(|param| expected_type_text(param.kind))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            continue;
        }
        for (index, (actual, param)) in define.params.iter().zip(&expected).enumerate() {
            if !llvm_type_matches(actual, param.kind) {
                problems.push(format!(
                    "`{}`: parameter {index} (`{}`) is `{actual}` in the module, but the \
                     description predicts `{}`",
                    desc.symbol,
                    param.name,
                    expected_type_text(param.kind)
                ));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}

// =============================================================================
// tileiras
// =============================================================================

/// `tileiras` from cuTile's override variable (also honoured as
/// [`TILEIRAS_PATH_ENV`], so one variable selects the same assembler for
/// both products), then the toolkit lookup every other CUDA tool uses
/// (PATH, the configured toolkit root, the standard install roots). An
/// override is returned as given: whether it runs is for
/// [`list_bytecode_versions`] to report.
pub(super) fn find_tileiras(ctx: &Context) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(TILEIRAS_PATH_ENV).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(path));
    }
    find_cuda_toolkit_executable(ctx, "tileiras", TILEIRAS_FALLBACK_PATHS)
}

/// The bytecode version to write the descriptions at for an assembler that
/// lists `versions`: its current version, the newest numbered one it lists,
/// when that is 13.5 or newer. An assembler reads preview content only at
/// its current version, so an older listing (CUDA 13.4 and before) means no
/// library can be built. The header tag bits are not part of a release, so
/// `13.5` and `13.5.1` both name 13.5.
pub(super) fn description_version(versions: &[BytecodeVersion]) -> Option<BytecodeVersion> {
    versions
        .iter()
        .copied()
        .filter(|version| *version >= BytecodeVersion::PREVIEW_FIRST)
        .max()
}

/// Exit status when no library was created because the installed assembler
/// cannot build one: a success when the caller said the library is optional,
/// and a shortfall otherwise.
pub(super) fn no_library_exit_code(library_optional: bool) -> i32 {
    if library_optional { 0 } else { 2 }
}

/// The `tileiras --preview-create-tile-lib` invocation, argv form.
pub(super) fn create_tile_lib_command(tileiras: &Path, paths: &TileLibraryPaths) -> Vec<String> {
    vec![
        tileiras.display().to_string(),
        "--preview-create-tile-lib".to_string(),
        format!("--preview-interface={}", paths.interface.display()),
        format!(
            "--preview-implementation={}",
            paths.implementation.display()
        ),
        format!("--preview-implementation={}", paths.ltoir.display()),
        "-o".to_string(),
        paths.library.display().to_string(),
    ]
}

fn render_command(command: &[String]) -> String {
    command.join(" ")
}

/// The [`LIBRARY_SKIPPED_MARKER`] line naming the assembler that was found
/// and what it lists, then why that is not enough and everything a reader
/// needs to finish the job on a newer toolkit.
fn explain_no_library(
    paths: &TileLibraryPaths,
    tileiras: Option<&Path>,
    versions: &[BytecodeVersion],
) {
    let needed = BytecodeVersion::PREVIEW_FIRST;
    println!();
    match tileiras {
        Some(tileiras) => {
            let lists = if versions.is_empty() {
                "no bytecode versions (it predates --list-versions)".to_string()
            } else {
                format!(
                    "bytecode versions {}",
                    versions
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            println!(
                "{LIBRARY_SKIPPED_MARKER}: {} lists {lists}; the descriptions need {needed} or newer",
                tileiras.display()
            );
            println!(
                "The installed tileiras cannot create tile libraries: the descriptions are \
                 Tile IR public-preview content, which exists from bytecode {needed} on and \
                 needs the CUDA 13.5 or newer assembler."
            );
        }
        None => {
            println!("{LIBRARY_SKIPPED_MARKER}: no tileiras found");
            println!(
                "No tileiras was found on PATH, under {TILEIRAS_PATH_ENV}, the configured CUDA \
                 toolkit root or the standard install roots; creating a tile library needs the \
                 CUDA 13.5 or newer assembler, the first one that lists bytecode {needed}."
            );
        }
    }
    println!(
        "The LTOIR and the descriptions were written to {}; a capable tileiras turns them \
         into {} with:",
        paths
            .interface
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .display(),
        paths.library.display()
    );
    let placeholder = Path::new("tileiras");
    println!(
        "  {}",
        render_command(&create_tile_lib_command(placeholder, paths))
    );
}
