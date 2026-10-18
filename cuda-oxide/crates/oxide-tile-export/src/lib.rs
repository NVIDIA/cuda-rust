/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Descriptions of Rust device functions exported to cuTile tile libraries.
//!
//! `#[tile_export]` (in `cuda-macros`) serializes one [`TileExportDesc`] per
//! exported function at macro-expansion time and embeds it as the body of a
//! host-only metadata function. The codegen backend reads that JSON back out
//! of MIR, validates it, and writes every description of a crate to
//! `<crate>.tileexports.json` next to the device artifact. `cargo oxide`
//! turns the file into the Tile IR interface and implementation modules a
//! tile library needs. All three sides share this crate so the vocabulary and
//! the rules are defined once.
//!
//! # The native SIMT ABI
//!
//! A tile library implementation is a plain `void` function. Each declaration
//! argument becomes one parameter, in order, and the result of an exported
//! Rust function becomes a trailing `out` parameter (destination-passing
//! style). The implementation symbol is what the Tile IR call resolves to, so
//! the Rust wrapper must have this signature and no other:
//!
//! | Rust parameter        | access  | footprint | ABI parameter            |
//! |-----------------------|---------|-----------|--------------------------|
//! | `x: T`                | `in`    | 1         | `T` by value             |
//! | `x: &T`               | `in`    | 1         | `*const T`               |
//! | `x: &[T; F]`          | `in`    | F         | `*const T`               |
//! | `x: &mut T`           | `inout` | 1         | `*mut T`                 |
//! | `x: &mut [T; F]`      | `inout` | F         | `*mut T`                 |
//! | `-> T` (return value) | `out`   | 1         | `*mut T`, appended last  |
//!
//! Only `x: T` travels by value; every other parameter is a pointer to the
//! invocation's elements, and the function returns nothing.
//! [`TileExportDesc::abi_params`] applies these rules and nothing else.
//!
//! Every parameter of one export has the same footprint: the fragmaps of one
//! implementation share their invocation count, so a 4-element window beside
//! a scalar would need 32 invocations for one argument and 128 for the other.
//! With `sync = warp` the shape must be a multiple of `32 * footprint`, so
//! every invocation group fills whole warps.
//!
//! Thread identity inside the function is the hardware `threadIdx.x`. The
//! callee may not allocate shared memory; version 1 of the description has no
//! scratch buffer.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// The description format version this crate reads and writes.
///
/// Version 1 describes one-dimensional tiles, by-value or pointer parameters
/// with a per-invocation footprint, and the `none` and `warp` synchronization
/// levels. [`TileExportDesc::validate`] rejects any other version.
pub const CURRENT_VERSION: u32 = 1;

/// File suffix of the per-crate description file the backend writes next to
/// the device artifact: `<crate>.tileexports.json` holds a JSON array of
/// [`TileExportDesc`].
pub const DESCRIPTIONS_SUFFIX: &str = "tileexports.json";

/// Element type of a tile, named by its Rust type.
///
/// Tile IR has no unsigned integer types, so the unsigned variants map to the
/// signless `i`-types of the same width in [`TileElem::tile_ir_name`] and to
/// the same LLVM integer types in [`TileElem::llvm_name`]. The Rust name is
/// kept in the description so the wrapper signature and the host-side
/// element interpretation stay recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TileElem {
    /// IEEE half precision, Rust `f16`, LLVM `half`.
    F16,
    /// Brain floating point. `cuda-device` carries bf16 values as `u16` bit
    /// patterns, so a by-value bf16 is LLVM `i16` in the native ABI (see
    /// [`TileElem::llvm_name`]). `#[tile_export]` does not accept `bf16` as
    /// an element type; the variant is part of the vocabulary for
    /// descriptions written by other tools.
    BF16,
    /// Rust `f32`, LLVM `float`.
    F32,
    /// Rust `f64`, LLVM `double`.
    F64,
    /// Rust `i8`.
    I8,
    /// Rust `i16`.
    I16,
    /// Rust `i32`.
    I32,
    /// Rust `i64`.
    I64,
    /// Rust `u8`; signless `i8` in Tile IR and LLVM.
    U8,
    /// Rust `u16`; signless `i16` in Tile IR and LLVM.
    U16,
    /// Rust `u32`; signless `i32` in Tile IR and LLVM.
    U32,
    /// Rust `u64`; signless `i64` in Tile IR and LLVM.
    U64,
}

impl TileElem {
    /// Every element type, in declaration order.
    pub const ALL: [TileElem; 12] = [
        TileElem::F16,
        TileElem::BF16,
        TileElem::F32,
        TileElem::F64,
        TileElem::I8,
        TileElem::I16,
        TileElem::I32,
        TileElem::I64,
        TileElem::U8,
        TileElem::U16,
        TileElem::U32,
        TileElem::U64,
    ];

    /// The element type whose Rust name is `name`, if any.
    ///
    /// ```
    /// use oxide_tile_export::TileElem;
    /// assert_eq!(TileElem::from_rust_name("f32"), Some(TileElem::F32));
    /// assert_eq!(TileElem::from_rust_name("usize"), None);
    /// ```
    pub fn from_rust_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|elem| elem.rust_name() == name)
    }

    /// The Rust name of the element type (`"f16"`, `"u32"`, ...).
    pub fn rust_name(self) -> &'static str {
        match self {
            TileElem::F16 => "f16",
            TileElem::BF16 => "bf16",
            TileElem::F32 => "f32",
            TileElem::F64 => "f64",
            TileElem::I8 => "i8",
            TileElem::I16 => "i16",
            TileElem::I32 => "i32",
            TileElem::I64 => "i64",
            TileElem::U8 => "u8",
            TileElem::U16 => "u16",
            TileElem::U32 => "u32",
            TileElem::U64 => "u64",
        }
    }

    /// The Tile IR name: floats keep their name, integers are signless.
    pub fn tile_ir_name(self) -> &'static str {
        match self {
            TileElem::F16 => "f16",
            TileElem::BF16 => "bf16",
            TileElem::F32 => "f32",
            TileElem::F64 => "f64",
            TileElem::I8 | TileElem::U8 => "i8",
            TileElem::I16 | TileElem::U16 => "i16",
            TileElem::I32 | TileElem::U32 => "i32",
            TileElem::I64 | TileElem::U64 => "i64",
        }
    }

    /// The LLVM IR type the element has in the native ABI.
    ///
    /// `bf16` is `i16`, not `bfloat`: `cuda-device` has no bf16 scalar type
    /// and carries bf16 values as `u16` bit patterns, so that is what the
    /// wrapper's parameter lowers to. The Tile IR element stays `bf16`, and
    /// `cargo oxide tilelib` records the difference as the implementation's
    /// `frontendABI` entry `{bf16 = i16}`.
    pub fn llvm_name(self) -> &'static str {
        match self {
            TileElem::F16 => "half",
            TileElem::BF16 => "i16",
            TileElem::F32 => "float",
            TileElem::F64 => "double",
            TileElem::I8 | TileElem::U8 => "i8",
            TileElem::I16 | TileElem::U16 => "i16",
            TileElem::I32 | TileElem::U32 => "i32",
            TileElem::I64 | TileElem::U64 => "i64",
        }
    }

    /// Width of one element in bits.
    pub fn bit_width(self) -> u32 {
        match self {
            TileElem::I8 | TileElem::U8 => 8,
            TileElem::F16 | TileElem::BF16 | TileElem::I16 | TileElem::U16 => 16,
            TileElem::F32 | TileElem::I32 | TileElem::U32 => 32,
            TileElem::F64 | TileElem::I64 | TileElem::U64 => 64,
        }
    }

    /// True for the floating-point element types.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            TileElem::F16 | TileElem::BF16 | TileElem::F32 | TileElem::F64
        )
    }
}

/// How the implementation touches one declaration argument.
///
/// The declaration's results are the tile types of its `out` and `inout`
/// arguments, in order, so every export needs at least one of those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TileAccess {
    /// Read only.
    In,
    /// Written, never read; the return value of a Rust function.
    Out,
    /// Read and written in place.
    InOut,
}

impl TileAccess {
    /// The Tile IR name of the access mode (`in`, `out`, `inout`).
    pub fn tile_ir_name(self) -> &'static str {
        match self {
            TileAccess::In => "in",
            TileAccess::Out => "out",
            TileAccess::InOut => "inout",
        }
    }

    /// True when the parameter is a result of the declaration.
    pub fn is_output(self) -> bool {
        matches!(self, TileAccess::Out | TileAccess::InOut)
    }
}

/// How many threads must call the implementation together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TileSync {
    /// Every invocation is independent.
    #[default]
    None,
    /// Whole warps participate; the shape must be a multiple of 32 times the
    /// footprint.
    Warp,
}

impl TileSync {
    /// The Tile IR name of the synchronization level.
    pub fn tile_ir_name(self) -> &'static str {
        match self {
            TileSync::None => "none",
            TileSync::Warp => "warp",
        }
    }
}

/// One parameter of an exported function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileParamDesc {
    /// Parameter name; the Rust parameter name, or `<fn>_out` for the return
    /// value.
    pub name: String,
    /// Element type of the tile this parameter is a fragment of.
    pub elem: TileElem,
    /// Read, write, or both.
    pub access: TileAccess,
    /// Number of contiguous elements one invocation owns; at least 1, and the
    /// same for every parameter of the export.
    pub footprint: u32,
    /// True when the element travels by value instead of behind a pointer.
    /// Only legal for `in` parameters with footprint 1.
    pub by_value: bool,
}

/// A `#[tile_export]` function as the tile library sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileExportDesc {
    /// Format version; see [`CURRENT_VERSION`].
    pub version: u32,
    /// Declaration name: what the cuTile caller names in `foreign_call`.
    pub decl: String,
    /// Implementation symbol: the `extern "C"` device function the call
    /// resolves to.
    pub symbol: String,
    /// Static tile extents; one positive entry in version 1.
    pub shape: Vec<i64>,
    /// Declaration arguments, in ABI order.
    pub params: Vec<TileParamDesc>,
    /// Synchronization the implementation requires.
    #[serde(default)]
    pub sync: TileSync,
    /// Compute capabilities the implementation supports (`sm_120`, ...).
    /// Empty means every architecture.
    #[serde(default)]
    pub arch: Vec<String>,
}

/// What a description can get wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TileExportError {
    /// The description was written by a newer or older format.
    #[error(
        "tile export description version {found} is not supported; this build reads version {CURRENT_VERSION}"
    )]
    UnsupportedVersion {
        /// The version found in the description.
        found: u32,
    },
    /// A name that must be an identifier is not one.
    #[error(
        "{field} `{value}` is not a valid identifier (ASCII letters, digits and `_`, not starting with a digit)"
    )]
    InvalidIdentifier {
        /// Which name: `decl`, `symbol`, or `parameter name`.
        field: &'static str,
        /// The offending text.
        value: String,
    },
    /// The shape has the wrong number of dimensions.
    #[error("shape must have one dimension in version {CURRENT_VERSION}, found {0}")]
    ShapeRank(usize),
    /// A shape extent is zero or negative.
    #[error("shape extents must be positive, found {0}")]
    ShapeNotPositive(i64),
    /// The export has no parameters at all.
    #[error("a tile export needs at least one parameter")]
    NoParams,
    /// Nothing is written: no `out` and no `inout` parameter.
    #[error("a tile export needs at least one output: return a value or take an `inout` parameter")]
    NoOutputs,
    /// An `in` parameter sits between two written parameters. The Tile
    /// compiler requires the `out` and `inout` arguments of a declaration to
    /// be one contiguous run (PreviewOps.cpp, "out/inout fragment args must
    /// form a single ...").
    #[error(
        "`out` and `inout` parameters must be one contiguous run, but `{between}` is `in` and sits between `{first}` and `{last}`; move `in` parameters before the `&mut` ones"
    )]
    OutputsNotContiguous {
        /// The first written parameter.
        first: String,
        /// The `in` parameter inside the run.
        between: String,
        /// The last written parameter (the appended return slot, when present).
        last: String,
    },
    /// Two parameters share a name.
    #[error("parameter `{name}` is declared twice")]
    DuplicateParam {
        /// The repeated name.
        name: String,
    },
    /// A footprint of zero elements.
    #[error(
        "parameter `{name}` has footprint 0; a footprint counts elements and must be at least 1"
    )]
    ZeroFootprint {
        /// The parameter.
        name: String,
    },
    /// By-value passing on something that is written.
    #[error(
        "parameter `{name}` is `{access}` and cannot travel by value; only `in` parameters are passed by value"
    )]
    ByValueNotIn {
        /// The parameter.
        name: String,
        /// Its access mode.
        access: &'static str,
    },
    /// By-value passing of more than one element.
    #[error(
        "parameter `{name}` has footprint {footprint} and cannot travel by value; a footprint above 1 needs pointer passing"
    )]
    ByValueFootprint {
        /// The parameter.
        name: String,
        /// Its footprint.
        footprint: u32,
    },
    /// Two parameters own a different number of elements per invocation. The
    /// fragmaps of one implementation share their invocation count, so this
    /// has no Tile IR form.
    #[error(
        "parameters `{first}` (footprint {first_footprint}) and `{other}` (footprint {other_footprint}) differ; every parameter of one export shares the invocation count, so all footprints must be equal"
    )]
    MixedFootprints {
        /// The first parameter, whose footprint the others were compared to.
        first: String,
        /// Its footprint.
        first_footprint: u32,
        /// The first parameter with another footprint.
        other: String,
        /// That footprint.
        other_footprint: u32,
    },
    /// Warp synchronization over a shape that does not fill whole warps; the
    /// footprint-1 case of [`TileExportError::WarpFootprintShape`].
    #[error("sync = warp needs a shape that is a multiple of 32, found {shape}")]
    WarpShape {
        /// The offending extent.
        shape: i64,
    },
    /// Warp synchronization over a shape that does not fill whole warps of
    /// footprints: each lane owns `footprint` elements, so a warp owns
    /// `32 * footprint` of them.
    #[error(
        "sync = warp with footprint {footprint} needs a shape that is a multiple of {} (32 lanes times the footprint), found {shape}",
        32 * i64::from(*.footprint)
    )]
    WarpFootprintShape {
        /// The shared footprint.
        footprint: u32,
        /// The offending extent.
        shape: i64,
    },
    /// An empty architecture string.
    #[error("arch entries must be non-empty names such as `sm_120`")]
    EmptyArch,
}

/// How one ABI parameter is passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiParamKind {
    /// The element itself, by value.
    Scalar(TileElem),
    /// A pointer to the invocation's buffer of this element type.
    Pointer(TileElem),
}

impl AbiParamKind {
    /// The LLVM IR type of the parameter: the element type, or `ptr`.
    pub fn llvm_type(self) -> &'static str {
        match self {
            AbiParamKind::Scalar(elem) => elem.llvm_name(),
            AbiParamKind::Pointer(_) => "ptr",
        }
    }

    /// The element type behind the parameter.
    pub fn elem(self) -> TileElem {
        match self {
            AbiParamKind::Scalar(elem) | AbiParamKind::Pointer(elem) => elem,
        }
    }
}

/// One parameter of the implementation symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiParam {
    /// The description parameter this ABI parameter carries.
    pub name: String,
    /// By value or by pointer.
    pub kind: AbiParamKind,
}

/// True for an ASCII identifier: `[A-Za-z_][A-Za-z0-9_]*`.
///
/// Both Tile IR symbol names and the LLVM symbols the wrapper gets have this
/// form; a Rust raw identifier or a non-ASCII name would not survive the
/// trip.
pub fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl TileExportDesc {
    /// Check every rule of version 1.
    ///
    /// The rules, in the order they are checked:
    ///
    /// - `version` is [`CURRENT_VERSION`];
    /// - `decl`, `symbol`, and every parameter name are identifiers, and no
    ///   parameter name repeats;
    /// - `shape` has one positive extent;
    /// - there is at least one parameter and at least one `out` or `inout`
    ///   parameter, and the `out`/`inout` parameters are one contiguous run;
    /// - every footprint is at least 1, and every parameter has the same
    ///   footprint;
    /// - `by_value` appears only on `in` parameters with footprint 1;
    /// - `sync = warp` needs a shape that is a multiple of `32 * footprint`;
    /// - every `arch` entry is non-empty.
    pub fn validate(&self) -> Result<(), TileExportError> {
        if self.version != CURRENT_VERSION {
            return Err(TileExportError::UnsupportedVersion {
                found: self.version,
            });
        }
        for (field, value) in [("decl", &self.decl), ("symbol", &self.symbol)] {
            if !is_identifier(value) {
                return Err(TileExportError::InvalidIdentifier {
                    field,
                    value: value.clone(),
                });
            }
        }
        if self.shape.len() != 1 {
            return Err(TileExportError::ShapeRank(self.shape.len()));
        }
        let extent = self.shape[0];
        if extent <= 0 {
            return Err(TileExportError::ShapeNotPositive(extent));
        }
        if self.params.is_empty() {
            return Err(TileExportError::NoParams);
        }
        let mut seen = std::collections::HashSet::new();
        for param in &self.params {
            if !is_identifier(&param.name) {
                return Err(TileExportError::InvalidIdentifier {
                    field: "parameter name",
                    value: param.name.clone(),
                });
            }
            if !seen.insert(param.name.as_str()) {
                return Err(TileExportError::DuplicateParam {
                    name: param.name.clone(),
                });
            }
            if param.footprint == 0 {
                return Err(TileExportError::ZeroFootprint {
                    name: param.name.clone(),
                });
            }
            if param.by_value {
                if param.access != TileAccess::In {
                    return Err(TileExportError::ByValueNotIn {
                        name: param.name.clone(),
                        access: param.access.tile_ir_name(),
                    });
                }
                if param.footprint != 1 {
                    return Err(TileExportError::ByValueFootprint {
                        name: param.name.clone(),
                        footprint: param.footprint,
                    });
                }
            }
        }
        // One invocation count per implementation: every fragmap of it maps
        // `shape / footprint` invocations, so the footprints cannot differ.
        let first = &self.params[0];
        if let Some(other) = self
            .params
            .iter()
            .find(|param| param.footprint != first.footprint)
        {
            return Err(TileExportError::MixedFootprints {
                first: first.name.clone(),
                first_footprint: first.footprint,
                other: other.name.clone(),
                other_footprint: other.footprint,
            });
        }
        let footprint = first.footprint;
        if !self.params.iter().any(|param| param.access.is_output()) {
            return Err(TileExportError::NoOutputs);
        }
        // The written arguments must be adjacent: the Tile compiler ties a
        // declaration's results to one contiguous run of out/inout arguments.
        let written: Vec<usize> = self
            .params
            .iter()
            .enumerate()
            .filter(|(_, param)| param.access.is_output())
            .map(|(index, _)| index)
            .collect();
        if let (Some(&first), Some(&last)) = (written.first(), written.last())
            && last - first + 1 != written.len()
        {
            let between = self.params[first + 1..last]
                .iter()
                .find(|param| !param.access.is_output())
                .expect("a gap in the written run holds an `in` parameter");
            return Err(TileExportError::OutputsNotContiguous {
                first: self.params[first].name.clone(),
                between: between.name.clone(),
                last: self.params[last].name.clone(),
            });
        }
        // A warp is one invocation of 32 lanes, each owning `footprint`
        // elements, so the tile must split into whole `32 * footprint` runs.
        if self.sync == TileSync::Warp && extent % (32 * i64::from(footprint)) != 0 {
            return Err(if footprint == 1 {
                TileExportError::WarpShape { shape: extent }
            } else {
                TileExportError::WarpFootprintShape {
                    footprint,
                    shape: extent,
                }
            });
        }
        if self.arch.iter().any(|arch| arch.trim().is_empty()) {
            return Err(TileExportError::EmptyArch);
        }
        Ok(())
    }

    /// The implementation symbol's parameters, in order.
    ///
    /// Follows the ABI table in the crate documentation: an `in` parameter
    /// marked `by_value` is passed as its element type, and everything else
    /// (pointer inputs, `inout` buffers, and the trailing `out` result) is a
    /// pointer to the element type. The description is assumed to have passed
    /// [`TileExportDesc::validate`].
    pub fn abi_params(&self) -> Vec<AbiParam> {
        self.params
            .iter()
            .map(|param| {
                let kind = if param.access == TileAccess::In && param.by_value {
                    AbiParamKind::Scalar(param.elem)
                } else {
                    AbiParamKind::Pointer(param.elem)
                };
                AbiParam {
                    name: param.name.clone(),
                    kind,
                }
            })
            .collect()
    }

    /// The description as compact JSON, the form the macro embeds.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a tile export description always serializes")
    }

    /// The description as indented JSON.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("a tile export description always serializes")
    }

    /// Parse one description. Syntax only; call [`TileExportDesc::validate`]
    /// for the semantic rules.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// Serialize a crate's descriptions as the indented JSON array the
/// description file holds.
pub fn descriptions_to_json_pretty(descs: &[TileExportDesc]) -> String {
    serde_json::to_string_pretty(descs).expect("tile export descriptions always serialize")
}

/// Parse a description file's JSON array.
pub fn descriptions_from_json(text: &str) -> Result<Vec<TileExportDesc>, serde_json::Error> {
    serde_json::from_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(name: &str, access: TileAccess, footprint: u32, by_value: bool) -> TileParamDesc {
        TileParamDesc {
            name: name.to_string(),
            elem: TileElem::F32,
            access,
            footprint,
            by_value,
        }
    }

    /// `fn relu(x: f32) -> f32` with `shape = 128`.
    fn relu() -> TileExportDesc {
        TileExportDesc {
            version: CURRENT_VERSION,
            decl: "relu".into(),
            symbol: "relu_impl".into(),
            shape: vec![128],
            params: vec![
                param("x", TileAccess::In, 1, true),
                param("relu_out", TileAccess::Out, 1, false),
            ],
            sync: TileSync::None,
            arch: Vec::new(),
        }
    }

    /// `fn scale(x: &mut f32, factor: f32)` exported as `scale_in_place`.
    fn scale_in_place() -> TileExportDesc {
        TileExportDesc {
            version: CURRENT_VERSION,
            decl: "scale_in_place".into(),
            symbol: "scale_in_place_impl".into(),
            shape: vec![128],
            params: vec![
                param("x", TileAccess::InOut, 1, false),
                param("factor", TileAccess::In, 1, true),
            ],
            sync: TileSync::None,
            arch: vec!["sm_120".into()],
        }
    }

    #[test]
    fn the_examples_validate() {
        relu().validate().unwrap();
        scale_in_place().validate().unwrap();
        let mut warp = relu();
        warp.sync = TileSync::Warp;
        warp.validate().unwrap();
    }

    #[test]
    fn version_is_pinned() {
        assert_eq!(CURRENT_VERSION, 1);
        let mut desc = relu();
        desc.version = 2;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::UnsupportedVersion { found: 2 })
        );
    }

    #[test]
    fn names_must_be_identifiers() {
        for (field, mutate) in [
            (
                "decl",
                (|d: &mut TileExportDesc| d.decl = "re-lu".into()) as fn(&mut TileExportDesc),
            ),
            ("symbol", |d: &mut TileExportDesc| d.symbol = "1impl".into()),
            ("parameter name", |d: &mut TileExportDesc| {
                d.params[0].name = "x y".into()
            }),
        ] {
            let mut desc = relu();
            mutate(&mut desc);
            match desc.validate() {
                Err(TileExportError::InvalidIdentifier { field: found, .. }) => {
                    assert_eq!(found, field)
                }
                other => panic!("{field}: expected an identifier error, got {other:?}"),
            }
        }
        assert!(is_identifier("_x1"));
        assert!(!is_identifier(""));
        assert!(!is_identifier("9x"));
        assert!(!is_identifier("r#type"));
    }

    #[test]
    fn duplicate_parameter_names_are_rejected() {
        let mut desc = relu();
        desc.params[1].name = "x".into();
        assert_eq!(
            desc.validate(),
            Err(TileExportError::DuplicateParam { name: "x".into() })
        );
    }

    #[test]
    fn shape_is_one_positive_extent() {
        let mut desc = relu();
        desc.shape = vec![128, 1];
        assert_eq!(desc.validate(), Err(TileExportError::ShapeRank(2)));
        desc.shape = Vec::new();
        assert_eq!(desc.validate(), Err(TileExportError::ShapeRank(0)));
        desc.shape = vec![0];
        assert_eq!(desc.validate(), Err(TileExportError::ShapeNotPositive(0)));
        desc.shape = vec![-4];
        assert_eq!(desc.validate(), Err(TileExportError::ShapeNotPositive(-4)));
    }

    #[test]
    fn every_export_has_a_parameter_and_an_output() {
        let mut desc = relu();
        desc.params.clear();
        assert_eq!(desc.validate(), Err(TileExportError::NoParams));
        let mut desc = relu();
        desc.params.pop();
        assert_eq!(desc.validate(), Err(TileExportError::NoOutputs));
    }

    #[test]
    fn written_parameters_form_one_contiguous_run() {
        // `fn f(a: &mut f32, b: f32) -> f32` is [inout, in, out]: the Tile
        // compiler ties results to one adjacent run of written arguments.
        let mut desc = relu();
        desc.params = vec![
            param("a", TileAccess::InOut, 1, false),
            param("b", TileAccess::In, 1, true),
            param("f_out", TileAccess::Out, 1, false),
        ];
        assert_eq!(
            desc.validate(),
            Err(TileExportError::OutputsNotContiguous {
                first: "a".into(),
                between: "b".into(),
                last: "f_out".into(),
            })
        );
        // Grouping the written parameters last is accepted.
        desc.params = vec![
            param("b", TileAccess::In, 1, true),
            param("a", TileAccess::InOut, 1, false),
            param("f_out", TileAccess::Out, 1, false),
        ];
        assert_eq!(desc.validate(), Ok(()));
        // Written parameters first and `in` parameters after them are also
        // one run.
        desc.params = vec![
            param("a", TileAccess::InOut, 1, false),
            param("c", TileAccess::InOut, 1, false),
            param("b", TileAccess::In, 1, true),
        ];
        assert_eq!(desc.validate(), Ok(()));
    }

    #[test]
    fn footprints_start_at_one() {
        let mut desc = scale_in_place();
        desc.params[0].footprint = 0;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::ZeroFootprint { name: "x".into() })
        );
        // `fn scale4(x: &mut [f32; 4], factor: &[f32; 4])`.
        for param in &mut desc.params {
            param.footprint = 4;
            param.by_value = false;
        }
        desc.validate().unwrap();
    }

    #[test]
    fn footprints_are_the_same_for_every_parameter() {
        // `fn window(acc: &mut [i32; 4], bias: &i32)`: the window needs 16
        // invocations of a 64-element tile, the scalar 64.
        let mut desc = scale_in_place();
        desc.params[0].footprint = 4;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::MixedFootprints {
                first: "x".into(),
                first_footprint: 4,
                other: "factor".into(),
                other_footprint: 1,
            })
        );
        let message = desc.validate().unwrap_err().to_string();
        assert!(
            message.contains("`x` (footprint 4) and `factor` (footprint 1) differ"),
            "{message}"
        );
        // The comparison is against the first parameter, whichever it is.
        desc.params[0].footprint = 1;
        desc.params[1].footprint = 2;
        desc.params[1].by_value = false;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::MixedFootprints {
                first: "x".into(),
                first_footprint: 1,
                other: "factor".into(),
                other_footprint: 2,
            })
        );
    }

    #[test]
    fn by_value_needs_in_and_footprint_one() {
        let mut desc = scale_in_place();
        desc.params[0].by_value = true;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::ByValueNotIn {
                name: "x".into(),
                access: "inout",
            })
        );
        let mut desc = relu();
        desc.params[1].by_value = true;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::ByValueNotIn {
                name: "relu_out".into(),
                access: "out",
            })
        );
        let mut desc = relu();
        desc.params[0].footprint = 2;
        assert_eq!(
            desc.validate(),
            Err(TileExportError::ByValueFootprint {
                name: "x".into(),
                footprint: 2,
            })
        );
    }

    #[test]
    fn warp_sync_needs_whole_warps() {
        let mut desc = relu();
        desc.sync = TileSync::Warp;
        desc.shape = vec![100];
        assert_eq!(
            desc.validate(),
            Err(TileExportError::WarpShape { shape: 100 })
        );
        desc.shape = vec![32];
        desc.validate().unwrap();
        // `none` has no shape rule.
        desc.sync = TileSync::None;
        desc.shape = vec![100];
        desc.validate().unwrap();
    }

    #[test]
    fn warp_sync_with_a_footprint_needs_whole_warps_of_footprints() {
        // `fn scale4(x: &mut [f32; 4], factor: &[f32; 4])` with `sync =
        // warp`: a warp owns 128 elements, so a 64-element tile holds half
        // a warp even though 64 is a multiple of 32.
        let mut desc = scale_in_place();
        desc.sync = TileSync::Warp;
        for param in &mut desc.params {
            param.footprint = 4;
            param.by_value = false;
        }
        desc.shape = vec![64];
        assert_eq!(
            desc.validate(),
            Err(TileExportError::WarpFootprintShape {
                footprint: 4,
                shape: 64,
            })
        );
        let message = desc.validate().unwrap_err().to_string();
        assert!(message.contains("multiple of 128"), "{message}");
        desc.shape = vec![256];
        desc.validate().unwrap();
        // Footprint 1 keeps the plain multiple-of-32 rule and its message.
        for param in &mut desc.params {
            param.footprint = 1;
        }
        desc.shape = vec![100];
        assert_eq!(
            desc.validate(),
            Err(TileExportError::WarpShape { shape: 100 })
        );
        // Without synchronization only the footprint rules apply.
        desc.sync = TileSync::None;
        for param in &mut desc.params {
            param.footprint = 4;
        }
        desc.shape = vec![64];
        desc.validate().unwrap();
    }

    #[test]
    fn arch_entries_are_non_empty() {
        let mut desc = relu();
        desc.arch = vec!["sm_120".into(), " ".into()];
        assert_eq!(desc.validate(), Err(TileExportError::EmptyArch));
    }

    #[test]
    fn abi_params_follow_the_table() {
        let abi = relu().abi_params();
        assert_eq!(
            abi,
            vec![
                AbiParam {
                    name: "x".into(),
                    kind: AbiParamKind::Scalar(TileElem::F32),
                },
                AbiParam {
                    name: "relu_out".into(),
                    kind: AbiParamKind::Pointer(TileElem::F32),
                },
            ]
        );
        assert_eq!(abi[0].kind.llvm_type(), "float");
        assert_eq!(abi[1].kind.llvm_type(), "ptr");

        let abi = scale_in_place().abi_params();
        assert_eq!(abi[0].kind, AbiParamKind::Pointer(TileElem::F32));
        assert_eq!(abi[1].kind, AbiParamKind::Scalar(TileElem::F32));

        // `x: &[T; F]` is `in` behind a pointer, `x: &mut [T; F]` is `inout`
        // behind a pointer; neither is by value.
        let mut desc = relu();
        desc.params[0].by_value = false;
        desc.params[0].footprint = 8;
        desc.params[1].footprint = 8;
        desc.params
            .insert(1, param("window", TileAccess::InOut, 8, false));
        desc.validate().unwrap();
        let abi = desc.abi_params();
        assert_eq!(abi[0].kind, AbiParamKind::Pointer(TileElem::F32));
        assert_eq!(abi[1].kind, AbiParamKind::Pointer(TileElem::F32));
        assert_eq!(abi[1].name, "window");
        assert_eq!(abi[2].kind, AbiParamKind::Pointer(TileElem::F32));

        // A by-value bf16 is the `u16` bit pattern cuda-device carries, so
        // the ABI parameter is `i16`; the Tile IR element stays `bf16`.
        let mut desc = relu();
        desc.params[0].elem = TileElem::BF16;
        assert_eq!(desc.abi_params()[0].kind.llvm_type(), "i16");
        assert_eq!(TileElem::BF16.tile_ir_name(), "bf16");
    }

    #[test]
    fn element_names_agree_across_rust_tile_ir_and_llvm() {
        let expected = [
            (TileElem::F16, "f16", "f16", "half", 16),
            (TileElem::BF16, "bf16", "bf16", "i16", 16),
            (TileElem::F32, "f32", "f32", "float", 32),
            (TileElem::F64, "f64", "f64", "double", 64),
            (TileElem::I8, "i8", "i8", "i8", 8),
            (TileElem::I16, "i16", "i16", "i16", 16),
            (TileElem::I32, "i32", "i32", "i32", 32),
            (TileElem::I64, "i64", "i64", "i64", 64),
            (TileElem::U8, "u8", "i8", "i8", 8),
            (TileElem::U16, "u16", "i16", "i16", 16),
            (TileElem::U32, "u32", "i32", "i32", 32),
            (TileElem::U64, "u64", "i64", "i64", 64),
        ];
        assert_eq!(expected.len(), TileElem::ALL.len());
        for (elem, rust, tile_ir, llvm, bits) in expected {
            assert_eq!(elem.rust_name(), rust);
            assert_eq!(elem.tile_ir_name(), tile_ir);
            assert_eq!(elem.llvm_name(), llvm);
            assert_eq!(elem.bit_width(), bits);
            assert_eq!(TileElem::from_rust_name(rust), Some(elem));
            assert_eq!(
                elem.is_float(),
                matches!(tile_ir, "f16" | "bf16" | "f32" | "f64")
            );
        }
        assert_eq!(TileElem::from_rust_name("usize"), None);
        assert_eq!(TileElem::from_rust_name("bool"), None);
    }

    #[test]
    fn json_round_trips_and_uses_lowercase_vocabulary() {
        let desc = scale_in_place();
        let json = desc.to_json();
        assert!(json.contains("\"access\":\"inout\""), "{json}");
        assert!(json.contains("\"elem\":\"f32\""), "{json}");
        assert!(json.contains("\"sync\":\"none\""), "{json}");
        assert!(json.contains("\"arch\":[\"sm_120\"]"), "{json}");
        assert_eq!(TileExportDesc::from_json(&json).unwrap(), desc);

        let mut warp = relu();
        warp.sync = TileSync::Warp;
        warp.params[0].elem = TileElem::BF16;
        let json = warp.to_json();
        assert!(json.contains("\"sync\":\"warp\""), "{json}");
        assert!(json.contains("\"elem\":\"bf16\""), "{json}");
        assert_eq!(TileExportDesc::from_json(&json).unwrap(), warp);

        // `sync` and `arch` default when absent.
        let minimal = r#"{"version":1,"decl":"relu","symbol":"relu_impl","shape":[128],
            "params":[{"name":"x","elem":"f32","access":"in","footprint":1,"by_value":true},
                      {"name":"relu_out","elem":"f32","access":"out","footprint":1,"by_value":false}]}"#;
        assert_eq!(TileExportDesc::from_json(minimal).unwrap(), relu());

        let all = vec![relu(), scale_in_place()];
        let text = descriptions_to_json_pretty(&all);
        assert!(text.starts_with('['), "{text}");
        assert_eq!(descriptions_from_json(&text).unwrap(), all);
    }
}
