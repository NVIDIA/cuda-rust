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
    /// symbol. `arch` is the compute capability it was compiled for, when
    /// known; a kernel for another architecture is refused.
    Ltoir {
        /// The LTOIR bytes.
        bytes: Vec<u8>,
        /// `sm_120` and the like, or `None` when unknown.
        arch: Option<String>,
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

    /// The implementation catalog for these declarations: a Tile IR module
    /// holding only their `preview$extern_impl` descriptions, which is what
    /// `provide_impls` returns.
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
    fn duplicates_and_bad_json_are_refused() {
        let twice = RELU_JSON.replacen("\"decl\":\"warp_sum\"", "\"decl\":\"relu\"", 1);
        let error = Payload::from_parts(
            "lib",
            &twice,
            BytecodeVersion::new(13, 6),
            DeviceCode::Ltoir {
                bytes: vec![],
                arch: None,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("share the declaration `relu`"), "{error}");
        assert!(matches!(
            Payload::from_parts(
                "lib",
                "[]",
                BytecodeVersion::new(13, 6),
                DeviceCode::Ltoir {
                    bytes: vec![],
                    arch: None
                }
            ),
            Err(PayloadError::Empty)
        ));
        assert!(matches!(
            Payload::from_parts(
                "lib",
                "nope",
                BytecodeVersion::new(13, 6),
                DeviceCode::Ltoir {
                    bytes: vec![],
                    arch: None
                }
            ),
            Err(PayloadError::Descriptions(_))
        ));
    }
}
