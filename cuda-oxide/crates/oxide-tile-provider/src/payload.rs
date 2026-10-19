/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! What a provider shared object carries: the exports it can serve and the
//! device code that implements them.

use oxide_tile_export::TileExportDesc;
use oxide_tilebc::{BytecodeVersion, describe};

/// The device code a provider hands to the assembler.
#[derive(Debug, Clone)]
pub enum DeviceCode {
    /// LTOIR compiled ahead of time, defining every export's implementation
    /// symbol under exactly that name. It serves static declarations only:
    /// the compiler asks for a mangled symbol per instantiation of a
    /// parametric one, and finished LTOIR cannot grow a new name. `arch` is
    /// the compute capability it was compiled for, when known.
    Ltoir {
        /// The LTOIR bytes.
        bytes: Vec<u8>,
        /// `sm_120` and the like, or `None` when unknown.
        arch: Option<String>,
    },
    /// The crate's NVVM IR, compiled by the provider when the assembler asks
    /// for code: the symbols the compiler names are added as thin wrappers
    /// around the bodies and the whole module is compiled for the kernel's
    /// architecture with the toolkit's libNVVM.
    NvvmIr {
        /// The textual NVVM IR `cargo oxide` wrote next to the crate.
        text: String,
        /// Whether FMA contraction was requested when the IR was built.
        fma_contraction: bool,
    },
}

/// The exports and the device code a provider serves.
#[derive(Debug, Clone)]
pub struct Payload {
    /// The library's name, used in diagnostics and as the module name of the
    /// implementation catalogs.
    pub name: String,
    /// The exports, validated and unique by declaration and by symbol.
    pub exports: Vec<TileExportDesc>,
    /// The bytecode version the catalogs are written at. Preview content is
    /// read only at the assembler's current version, so this is the version
    /// the library was built against.
    pub bytecode_version: BytecodeVersion,
    /// The device code.
    pub code: DeviceCode,
}

/// Why a payload cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum PayloadError {
    /// The descriptions are not the JSON array `#[tile_export]` writes.
    #[error("tile export descriptions: {0}")]
    Descriptions(String),
    /// There is nothing to export.
    #[error("the payload has no exports")]
    Empty,
    /// Two exports share a declaration name or an implementation symbol.
    #[error("two exports share the {what} `{name}`")]
    Duplicate {
        /// `declaration` or `symbol`.
        what: &'static str,
        /// The repeated name.
        name: String,
    },
    /// A parametric export came with finished LTOIR, which cannot serve it.
    #[error(
        "export `{0}` names no tile_shape, so the compiler will ask for one symbol per tile \
         extent; embed the crate's NVVM IR instead of finished LTOIR to serve it"
    )]
    ParametricNeedsIr(String),
    /// The descriptions do not make a valid library.
    #[error(transparent)]
    Library(#[from] oxide_tilebc::Error),
}

impl Payload {
    /// Parse the descriptions `cargo oxide` wrote next to the device artifact
    /// and check that together they make a valid library.
    pub fn from_parts(
        name: impl Into<String>,
        descriptions_json: &str,
        bytecode_version: BytecodeVersion,
        code: DeviceCode,
    ) -> Result<Self, PayloadError> {
        let name = name.into();
        let exports: Vec<TileExportDesc> = serde_json::from_str(descriptions_json)
            .map_err(|error| PayloadError::Descriptions(error.to_string()))?;
        if exports.is_empty() {
            return Err(PayloadError::Empty);
        }
        for (index, export) in exports.iter().enumerate() {
            for earlier in &exports[..index] {
                if earlier.decl == export.decl {
                    return Err(PayloadError::Duplicate {
                        what: "declaration",
                        name: export.decl.clone(),
                    });
                }
                if earlier.symbol == export.symbol {
                    return Err(PayloadError::Duplicate {
                        what: "symbol",
                        name: export.symbol.clone(),
                    });
                }
            }
            if export.is_parametric() && matches!(code, DeviceCode::Ltoir { .. }) {
                return Err(PayloadError::ParametricNeedsIr(export.decl.clone()));
            }
        }
        // Every description is checked once here, so the callbacks can treat
        // the exports as valid.
        describe::library(name.clone(), &exports)?;
        Ok(Self {
            name,
            exports,
            bytecode_version,
            code,
        })
    }

    /// The export a Tile kernel calls by this declaration name.
    pub fn export_for_decl(&self, decl: &str) -> Option<&TileExportDesc> {
        self.exports.iter().find(|export| export.decl == decl)
    }

    /// The export whose implementation symbol this is.
    pub fn export_for_symbol(&self, symbol: &str) -> Option<&TileExportDesc> {
        self.exports.iter().find(|export| export.symbol == symbol)
    }

    /// The export behind a symbol the compiler asks for at `instantiate`.
    ///
    /// A static export is asked for under its own symbol. A parametric one
    /// is asked for under the name the compiler gave the instantiation: the
    /// symbol, `__` and the bound parameters (`relu_impl__N256`), with a
    /// `_k` suffix when the same instantiation recurs in another execution
    /// context. The longest export symbol that is the requested name or a
    /// prefix of it followed by `_` wins.
    pub fn export_for_requested(&self, requested: &str) -> Option<&TileExportDesc> {
        self.exports
            .iter()
            .filter(|export| {
                requested
                    .strip_prefix(export.symbol.as_str())
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .max_by_key(|export| export.symbol.len())
    }

    /// The implementation catalog for these declarations: a Tile IR module
    /// holding only their `preview$extern_impl` descriptions, which is what
    /// `provide_impls` returns. A parametric export is described in its
    /// parametric form, a static one at its extent.
    pub fn catalog(&self, decls: &[&str]) -> Result<Vec<u8>, PayloadError> {
        let selected: Vec<TileExportDesc> = self
            .exports
            .iter()
            .filter(|export| decls.contains(&export.decl.as_str()))
            .cloned()
            .collect();
        let library = describe::library(self.name.clone(), &selected)?;
        Ok(library.implementation_bytecode(self.bytecode_version)?)
    }

    /// The implementation symbols, in export order.
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.exports.iter().map(|export| export.symbol.as_str())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const RELU_JSON: &str = r#"[{"version":1,"decl":"relu","symbol":"relu_impl","shape":[128],
        "params":[{"name":"x","elem":"f32","access":"in","footprint":1,"by_value":true},
                  {"name":"relu_out","elem":"f32","access":"out","footprint":1,"by_value":false}],
        "sync":"none","arch":[]},
        {"version":1,"decl":"warp_sum","symbol":"warp_sum_impl","shape":[128],
        "params":[{"name":"x","elem":"f32","access":"in","footprint":1,"by_value":true},
                  {"name":"warp_sum_out","elem":"f32","access":"out","footprint":1,"by_value":false}],
        "sync":"warp","arch":[]}]"#;

    /// The same two exports without extents: parametric declarations.
    pub(crate) fn parametric_json() -> String {
        RELU_JSON.replace("\"shape\":[128],", "")
    }

    pub(crate) fn payload() -> Payload {
        Payload::from_parts(
            "relu_lib",
            RELU_JSON,
            BytecodeVersion::new(13, 6),
            DeviceCode::Ltoir {
                bytes: b"not really ltoir".to_vec(),
                arch: Some("sm_120".into()),
            },
        )
        .unwrap()
    }

    pub(crate) fn parametric_payload() -> Payload {
        Payload::from_parts(
            "relu_lib",
            &parametric_json(),
            BytecodeVersion::new(13, 6),
            DeviceCode::NvvmIr {
                text: "; not really IR\n".into(),
                fma_contraction: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn parses_and_indexes_the_descriptions() {
        let payload = payload();
        assert_eq!(payload.exports.len(), 2);
        assert_eq!(payload.export_for_decl("relu").unwrap().symbol, "relu_impl");
        assert_eq!(
            payload.export_for_symbol("warp_sum_impl").unwrap().decl,
            "warp_sum"
        );
        assert!(payload.export_for_decl("gelu").is_none());
    }

    #[test]
    fn requested_symbols_map_back_to_exports() {
        let payload = parametric_payload();
        for (requested, decl) in [
            ("relu_impl", "relu"),
            ("relu_impl__N256", "relu"),
            ("relu_impl__N128_1", "relu"),
            ("warp_sum_impl__N64", "warp_sum"),
        ] {
            assert_eq!(
                payload.export_for_requested(requested).unwrap().decl,
                decl,
                "{requested}"
            );
        }
        assert!(
            payload
                .export_for_requested("relu_implementation")
                .is_none()
        );
        assert!(payload.export_for_requested("gelu_impl__N128").is_none());
    }

    #[test]
    fn a_catalog_holds_only_the_asked_declarations() {
        let payload = payload();
        let relu_only = payload.catalog(&["relu"]).unwrap();
        let both = payload.catalog(&["relu", "warp_sum"]).unwrap();
        assert_eq!(&relu_only[..8], b"\x7fTileIR\0");
        assert_eq!(&relu_only[8..12], &[13, 6, 1, 0]);
        assert!(both.len() > relu_only.len());
        let text = String::from_utf8_lossy(&both);
        assert!(
            text.contains("warp_sum_impl")
                && !String::from_utf8_lossy(&relu_only).contains("warp_sum")
        );
    }

    #[test]
    fn parametric_catalogs_carry_the_parameter() {
        let payload = parametric_payload();
        let catalog = payload.catalog(&["relu"]).unwrap();
        // The string table names the parameter and the type table holds `index`.
        assert!(String::from_utf8_lossy(&catalog).contains('N'));
        assert!(
            catalog.windows(2).any(|pair| pair == [0xD6, 0x01]),
            "index type entry"
        );
    }

    #[test]
    fn duplicates_bad_json_and_ltoir_for_parametric_exports_are_refused() {
        let twice = RELU_JSON.replacen("\"decl\":\"warp_sum\"", "\"decl\":\"relu\"", 1);
        let ltoir = || DeviceCode::Ltoir {
            bytes: vec![],
            arch: None,
        };
        let error = Payload::from_parts("lib", &twice, BytecodeVersion::new(13, 6), ltoir())
            .unwrap_err()
            .to_string();
        assert!(error.contains("share the declaration `relu`"), "{error}");
        assert!(matches!(
            Payload::from_parts("lib", "[]", BytecodeVersion::new(13, 6), ltoir()),
            Err(PayloadError::Empty)
        ));
        assert!(matches!(
            Payload::from_parts("lib", "nope", BytecodeVersion::new(13, 6), ltoir()),
            Err(PayloadError::Descriptions(_))
        ));
        let error = Payload::from_parts(
            "lib",
            &parametric_json(),
            BytecodeVersion::new(13, 6),
            ltoir(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("names no tile_shape"), "{error}");
    }
}
