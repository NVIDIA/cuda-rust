/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! From `#[tile_export]` descriptions to a [`Library`].
//!
//! Every parameter becomes a tile of the export's extent on the declaration
//! and a fragment on the implementation: a by-value scalar is the rank-0
//! `():()` buffer, everything else a contiguous footprint `(F):(1)`. All
//! parameters of one export share one fragmap, see [`fragmap`]. The extent
//! is either the literal the export names (`tile_shape = 128`, a static
//! library) or a constant parameter `N` the call supplies (a parametric
//! declaration, which only a linkage provider can serve). The same mapping
//! serves `cargo oxide tilelib`, which writes a static library ahead of
//! time, and the provider runtime, which answers the assembler with the
//! implementation half while a kernel compiles.

use oxide_tile_export::{TileAccess, TileElem, TileExportDesc, TileSync};

use crate::Error;
use crate::model::{
    Access, ConstParam, Declaration, Expr, FragmentArg, FragmentType, Implementation, Layout,
    Library, MemorySpace, ScalarType, Synchronization, TileArg, TileType,
};

/// The name every parametric declaration gives its tile extent.
pub const EXTENT_PARAM: &str = "N";

/// The library the descriptions make, at the extents they name, verified.
pub fn library(name: impl Into<String>, exports: &[TileExportDesc]) -> Result<Library, Error> {
    build_library(name, exports, export)
}

/// The library the descriptions make with every declaration parametric in
/// its extent, verified.
pub fn parametric_library(
    name: impl Into<String>,
    exports: &[TileExportDesc],
) -> Result<Library, Error> {
    build_library(name, exports, parametric_export)
}

fn build_library(
    name: impl Into<String>,
    exports: &[TileExportDesc],
    mut one: impl FnMut(&TileExportDesc) -> Result<(Declaration, Implementation), Error>,
) -> Result<Library, Error> {
    let mut declarations = Vec::with_capacity(exports.len());
    let mut implementations = Vec::with_capacity(exports.len());
    for desc in exports {
        let (declaration, implementation) = one(desc)?;
        declarations.push(declaration);
        implementations.push(implementation);
    }
    let library = Library::new(name, declarations, implementations);
    library.verify()?;
    Ok(library)
}

/// One export as its static declaration and implementation, at the extent
/// the description names.
pub fn export(desc: &TileExportDesc) -> Result<(Declaration, Implementation), Error> {
    desc.validate()
        .map_err(|error| Error::Description(format!("export `{}`: {error}", desc.decl)))?;
    build(desc, Expr::Int(desc.shape[0]))
}

/// One export as a parametric declaration `@decl<[N: index]>` over tiles of
/// `N` elements, with the implementation `@symbol<[N: index]> implements
/// @decl<[N]>` whose fragmaps are expressions in `N`.
///
/// The description's own extent is ignored; the shape-free rules are
/// checked on a copy with a probe extent that every layout divides.
pub fn parametric_export(desc: &TileExportDesc) -> Result<(Declaration, Implementation), Error> {
    let footprint = desc
        .params
        .first()
        .map_or(1, |param| i64::from(param.footprint.max(1)));
    let mut probe = desc.clone();
    probe.shape = vec![64 * footprint];
    probe
        .validate()
        .map_err(|error| Error::Description(format!("export `{}`: {error}", desc.decl)))?;
    build(desc, Expr::var(EXTENT_PARAM))
}

/// The common construction; `extent` is a literal or the parameter `N`.
///
/// A `bf16` scalar passed by value needs the `bf16 = i16` frontend ABI entry
/// because cuda-device's bf16 is a `u16` wrapper, so the LLVM parameter is
/// `i16` while the tile element is `bf16`.
fn build(desc: &TileExportDesc, extent: Expr) -> Result<(Declaration, Implementation), Error> {
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
            TileType::symbolic(vec![extent.clone()], element),
            access,
        ));
        let footprint = i64::from(param.footprint);
        let fragmap = fragmap(desc.sync, extent.clone(), footprint).map_err(|error| {
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
    let parametric = extent.as_int().is_none();
    let mut implementation = if parametric {
        Implementation::parametric(
            desc.symbol.clone(),
            desc.decl.clone(),
            vec![EXTENT_PARAM.to_string()],
            fragments,
        )
    } else {
        Implementation::new(desc.symbol.clone(), desc.decl.clone(), fragments)
    };
    implementation.synchronization = synchronization(desc.sync);
    implementation.arch = desc.arch.clone();
    if desc
        .params
        .iter()
        .any(|param| param.by_value && param.elem == TileElem::BF16)
    {
        implementation.frontend_abi = vec![("bf16".to_string(), "i16".to_string())];
    }
    let declaration = if parametric {
        Declaration::parametric(
            desc.decl.clone(),
            vec![ConstParam::index(EXTENT_PARAM)],
            args,
        )
    } else {
        Declaration::new(desc.decl.clone(), args)
    };
    Ok((declaration, implementation))
}

/// How one invocation's footprint maps onto a tile of `tile_size` elements.
///
/// Without synchronization every invocation owns `footprint` contiguous
/// elements: `(N, 1, 1):(1, 0, 0)` for a footprint of one, `(N/F, 1, F):(F,
/// 0, 1)` otherwise. With `sync = warp` the 32 lanes of a warp are the
/// participants of one invocation: `(N/32, 32, 1):(32, 1, 0)` for a footprint
/// of one, and `(N/(32F), 32, F):(32F, F, 1)` for a footprint of `F`, so each
/// lane keeps a contiguous footprint and adjacent lanes are adjacent in the
/// tile. A literal tile size must be a multiple of the footprint, and of 32
/// footprints for a warp; a parametric one is checked when the call binds it.
pub fn fragmap(sync: TileSync, tile_size: Expr, footprint: i64) -> Result<Layout, String> {
    if footprint <= 0 {
        return Err(format!("footprint {footprint} must be positive"));
    }
    if let Some(size) = tile_size.as_int()
        && size % footprint != 0
    {
        return Err(format!(
            "footprint {footprint} does not divide the tile size {size}"
        ));
    }
    match sync {
        TileSync::None if footprint == 1 => Ok(Layout::elementwise(tile_size)),
        TileSync::None => Ok(Layout::footprint_blocks(tile_size, footprint)),
        TileSync::Warp => {
            let per_warp = 32 * footprint;
            if let Some(size) = tile_size.as_int()
                && size % per_warp != 0
            {
                return Err(format!(
                    "sync = warp with footprint {footprint} needs a tile size that is a \
                     multiple of {per_warp}, found {size}"
                ));
            }
            if footprint == 1 {
                Ok(Layout::warp_lanes(tile_size))
            } else {
                Ok(Layout::warp_footprints(tile_size, footprint))
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
        assert!(text.contains("@relu_impl implements @relu("), "{text}");
        assert!(text.contains("(128, 1, 1):(1, 0, 0)"), "{text}");
        assert!(
            text.contains("mem_layout = #cuda_tile.preview$layout<():()>"),
            "{text}"
        );
        assert!(
            library
                .interface_text()
                .contains("@relu(%x: !cuda_tile.tile<128xf32>")
        );
    }

    #[test]
    fn relu_becomes_a_parametric_library() {
        let mut warp = relu();
        warp.decl = "warp_sum".into();
        warp.symbol = "warp_sum_impl".into();
        warp.sync = TileSync::Warp;
        let library = parametric_library("lib", &[relu(), warp]).unwrap();
        let interface = library.interface_text();
        assert!(
            interface.contains("@relu<[N: index]>(%x: !cuda_tile.tile<[N]xf32>"),
            "{interface}"
        );
        let implementation = library.implementation_text();
        assert!(
            implementation.contains("@relu_impl<[N: index]> implements @relu<[N]>("),
            "{implementation}"
        );
        assert!(
            implementation.contains("(N, 1, 1):(1, 0, 0)"),
            "{implementation}"
        );
        assert!(
            implementation.contains("(N / 32, 32, 1):(32, 1, 0)"),
            "{implementation}"
        );
    }

    #[test]
    fn warp_sync_needs_whole_warps() {
        let mut desc = relu();
        desc.sync = TileSync::Warp;
        desc.shape = vec![100];
        let error = export(&desc).unwrap_err().to_string();
        assert!(error.contains("multiple of 32"), "{error}");
        assert!(fragmap(TileSync::Warp, 128.into(), 1).is_ok());
        assert_eq!(
            fragmap(TileSync::None, 128.into(), 4).unwrap(),
            Layout::flat(&[32, 1, 4], &[4, 0, 1])
        );
        assert_eq!(
            fragmap(TileSync::Warp, Expr::var("N"), 4).unwrap().shapes[0].to_string(),
            "N / 128"
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
