/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! From `#[tile_export]` descriptions to a [`Library`].
//!
//! Every parameter becomes a tile of the export's shape on the declaration
//! and a fragment on the implementation: a by-value scalar is the rank-0
//! `():()` buffer, everything else a contiguous footprint `(F):(1)`. All
//! parameters of one export share one fragmap, see [`fragmap`]. The same
//! mapping serves `cargo oxide tilelib`, which writes a static library ahead
//! of time, and the linkage provider runtime, which answers the assembler
//! with the implementation half while a kernel compiles.

use oxide_tile_export::{TileAccess, TileElem, TileExportDesc, TileSync};

use crate::Error;
use crate::model::{
    Access, Declaration, FragmentArg, FragmentType, Implementation, Layout, Library, MemorySpace,
    ScalarType, Synchronization, TileArg, TileType,
};

/// The library the descriptions make, verified.
pub fn library(name: impl Into<String>, exports: &[TileExportDesc]) -> Result<Library, Error> {
    let mut declarations = Vec::with_capacity(exports.len());
    let mut implementations = Vec::with_capacity(exports.len());
    for desc in exports {
        let (declaration, implementation) = export(desc)?;
        declarations.push(declaration);
        implementations.push(implementation);
    }
    let library = Library::new(name, declarations, implementations);
    library.verify()?;
    Ok(library)
}

/// One export as its declaration and its native implementation.
///
/// A `bf16` scalar passed by value needs the `bf16 = i16` frontend ABI entry
/// because cuda-device's bf16 is a `u16` wrapper, so the LLVM parameter is
/// `i16` while the tile element is `bf16`.
pub fn export(desc: &TileExportDesc) -> Result<(Declaration, Implementation), Error> {
    desc.validate()
        .map_err(|error| Error::Description(format!("export `{}`: {error}", desc.decl)))?;
    let tile_size = desc.shape[0];
    // The fragmaps of one implementation share their invocation count, so the
    // footprints have to agree: a 4-element window beside a scalar would need
    // 32 invocations for one argument and 128 for the other.
    let footprint = desc.params[0].footprint;
    if let Some(other) = desc
        .params
        .iter()
        .find(|param| param.footprint != footprint)
    {
        return Err(Error::Description(format!(
            "export `{}`: parameters `{}` (footprint {}) and `{}` (footprint {}) differ; every \
             parameter of one export shares the invocation count, so all footprints must be equal",
            desc.decl, desc.params[0].name, footprint, other.name, other.footprint
        )));
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
        let fragmap = fragmap(desc.sync, tile_size, footprint).map_err(|error| {
            Error::Description(format!(
                "export `{}`, parameter `{}`: {error}",
                desc.decl, param.name
            ))
        })?;
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
pub fn fragmap(sync: TileSync, tile_size: i64, footprint: i64) -> Result<Layout, String> {
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

/// The tile element type of a Rust element type. Unsigned integers share the
/// signless Tile IR integer of their width.
pub fn scalar_type(elem: TileElem) -> ScalarType {
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

fn synchronization(sync: TileSync) -> Option<Synchronization> {
    match sync {
        TileSync::None => None,
        TileSync::Warp => Some(Synchronization::Warp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_tile_export::TileParamDesc;

    fn relu() -> TileExportDesc {
        TileExportDesc {
            version: oxide_tile_export::CURRENT_VERSION,
            decl: "relu".into(),
            symbol: "relu_impl".into(),
            shape: vec![128],
            params: vec![
                TileParamDesc {
                    name: "x".into(),
                    elem: TileElem::F32,
                    access: TileAccess::In,
                    footprint: 1,
                    by_value: true,
                },
                TileParamDesc {
                    name: "relu_out".into(),
                    elem: TileElem::F32,
                    access: TileAccess::Out,
                    footprint: 1,
                    by_value: false,
                },
            ],
            sync: TileSync::None,
            arch: vec![],
        }
    }

    #[test]
    fn relu_becomes_an_elementwise_library() {
        let library = library("lib", &[relu()]).unwrap();
        let text = library.implementation_text();
        assert!(text.contains("@relu_impl implements @relu"), "{text}");
        assert!(text.contains("(128, 1, 1):(1, 0, 0)"), "{text}");
        assert!(
            text.contains("mem_layout = #cuda_tile.preview$layout<():()>"),
            "{text}"
        );
    }

    #[test]
    fn warp_sync_needs_whole_warps() {
        let mut desc = relu();
        desc.sync = TileSync::Warp;
        desc.shape = vec![100];
        let error = export(&desc).unwrap_err().to_string();
        assert!(error.contains("multiple of 32"), "{error}");
        assert!(fragmap(TileSync::Warp, 128, 1).is_ok());
        assert_eq!(
            fragmap(TileSync::None, 128, 4).unwrap().shapes,
            vec![32, 1, 4]
        );
    }

    #[test]
    fn footprints_must_agree() {
        let mut desc = relu();
        desc.params[1].footprint = 2;
        desc.params[1].by_value = false;
        let error = export(&desc).unwrap_err().to_string();
        assert!(error.contains("footprints must be equal"), "{error}");
    }
}
