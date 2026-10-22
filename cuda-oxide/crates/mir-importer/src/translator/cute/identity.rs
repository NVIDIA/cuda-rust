/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical defining paths and the closed source-abstraction boundary.

use super::registry::{CUTE_FNS, CuteFnSpec};
#[cfg(test)]
use super::registry::{CuteFn, SubstKind};
use rustc_public::{CrateDef, ty::FnDef};

/// Classification of one callee at the closed CuTe compiler boundary.
pub(super) enum CuteIdentity {
    /// Callee is not from the cute_rs crate; normal translation proceeds.
    NotCuteCrate,
    /// A recognized entry point.
    Known(#[doc = "Registered compiler-hook specification."] &'static CuteFnSpec),
    /// Ordinary zero-cost Rust whose body is intentionally translated.
    SourceAbstraction,
    /// In the cute_rs crate but not in the table: hard error (fall-through
    /// would body-translate and scalarize the layout).
    Unknown(#[doc = "Unrecognized defining path for the rejection diagnostic."] String),
}

/// Definition modules containing ordinary Rust source abstractions rather
/// than importer-owned recognition stubs.
///
/// This is deliberately a closed list. Registered [`CUTE_FNS`] entries are
/// matched first, including compiler stubs nested below these modules.
/// Every remaining function must have a real, device-safe body, even when an
/// ergonomic method can be recognized separately. An unregistered panic-only
/// stub could otherwise fall through to body translation. Adding a module
/// requires auditing that fallback and updating the exact-list test.
const CUTE_SOURCE_ABSTRACTION_MODULES: &[&str] = &[
    "arch",
    "mma::bf16",
    "block_scaled",
    "mma::block_scaled",
    "copy::cooperative",
    "epilogue::sm120",
    "numeric",
    "pipeline",
    "reduction",
    "scheduler",
    "sm100",
    "attention::sm100",
    "static_layout",
    "tensor",
    "copy::tiled",
];

/// Match a complete module path, including every namespace separator.
///
/// Auditing `mma::bf16` never admits `mma`, `mma::other`, or `mma::bf16_extra`.
fn path_is_within_module(relative: &str, module: &str) -> bool {
    relative
        .strip_prefix(module)
        .and_then(|suffix| suffix.strip_prefix("::"))
        .is_some_and(|item| !item.is_empty())
}

/// Accept only the explicitly listed source-helper module prefixes.
fn is_source_abstraction_path(path: &str, crate_name: &str) -> bool {
    let Some(rest) = path
        .strip_prefix(crate_name)
        .and_then(|rest| rest.strip_prefix("::"))
    else {
        return false;
    };
    CUTE_SOURCE_ABSTRACTION_MODULES
        .iter()
        .any(|module| path_is_within_module(rest, module))
}

/// Source-only modules whose function bodies are also callable through Rust's
/// `Fn` traits. The core `tensor` and `pipeline` modules intentionally remain
/// excluded: their ergonomic methods include compiler-owned stubs alongside
/// ordinary source helpers. The source-only `tensor::partition` child is
/// admitted separately.
const CUTE_SOURCE_CALLBACK_MODULES: &[&str] = &[
    "arch",
    "mma::bf16",
    "numeric",
    "tensor::partition",
    "reduction",
    "static_layout",
];

/// Whether a function item may bypass direct CuTe call recognition.
fn is_source_callback_path(path: &str, crate_name: &str) -> bool {
    if !matches!(crate_name, "cute_rs" | "cute-rs") || !is_source_abstraction_path(path, crate_name)
    {
        return false;
    }
    let Some(relative) = path
        .strip_prefix(crate_name)
        .and_then(|path| path.strip_prefix("::"))
    else {
        return false;
    };
    // Registered hooks always take precedence, including any future hook
    // added under a currently source-only module. Normalize the crate spelling
    // in the same way for both accepted crate identities.
    if CUTE_FNS
        .iter()
        .any(|spec| spec.canonical.strip_prefix("cute_rs::") == Some(relative))
    {
        return false;
    }
    CUTE_SOURCE_CALLBACK_MODULES
        .iter()
        .any(|module| path_is_within_module(relative, module))
}

/// Keep compiler-owned CuTe targets on the direct-call recognition path while
/// permitting audited real Rust helpers to serve as tensor-map callbacks.
pub(in crate::translator) fn function_item_requires_direct_dispatch(fn_def: &FnDef) -> bool {
    let crate_name = fn_def.krate().name.to_string();
    matches!(crate_name.as_str(), "cute_rs" | "cute-rs")
        && !is_source_callback_path(&canonical_path(fn_def, &crate_name), &crate_name)
}

/// Recover a callee's actual definition path, independent of public aliases.
pub(super) fn canonical_path(fn_def: &FnDef, crate_name: &str) -> String {
    canonical_def_path(fn_def.def_id(), crate_name)
}

/// Recover the defining identity shared by function and ADT recognition.
///
/// `DefId::name()` uses rustc's diagnostic printer, which can prefer a public
/// reexport even for individual parents. Combining those display names with
/// real parents can manufacture a path that never existed. Read the underlying
/// `DefPath` instead, as the backend's intrinsic classifier does. Its components
/// preserve impl owners and nested modules, including a module named like the
/// crate, and never substitute a compatibility alias.
pub(super) fn canonical_def_path(def: rustc_public::DefId, crate_name: &str) -> String {
    rustc_middle::ty::tls::with(|tcx| {
        let def = rustc_public::rustc_internal::internal(tcx, def);
        format!(
            "{crate_name}{}",
            tcx.def_path(def).to_string_no_crate_verbose()
        )
    })
}

#[cfg(test)]
mod definition_tests;

/// Classify a callee as a registered hook, approved source helper, foreign call or error.
pub(super) fn classify_cute_call(fn_def: &FnDef) -> CuteIdentity {
    let crate_name = fn_def.krate().name.to_string();
    if !matches!(crate_name.as_str(), "cute_rs" | "cute-rs") {
        return CuteIdentity::NotCuteCrate;
    }
    let path = canonical_path(fn_def, &crate_name);
    match CUTE_FNS.iter().find(|spec| spec.canonical == path) {
        Some(spec) => CuteIdentity::Known(spec),
        None if is_source_abstraction_path(&path, &crate_name) => CuteIdentity::SourceAbstraction,
        None => CuteIdentity::Unknown(path),
    }
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod canonical_path_tests {
    use super::super::classify::{
        block_scaled_method_name_from_path, pipeline_method_name_from_path,
        scheduler_method_name_from_path, smem_mma_method_name_from_path,
        tensor_method_name_from_path,
    };
    use super::{is_source_abstraction_path, is_source_callback_path};

    /// Higher-order tensor operations may call real numeric/helper bodies,
    /// but neither registered hooks nor ergonomic compiler stubs may escape
    /// the direct-call recognition boundary through a function item.
    #[test]
    fn function_items_allow_only_audited_source_bodies() {
        for path in [
            "cute_rs::numeric::silu_bf16",
            "cute_rs::numeric::{impl#3}::to_f32",
            "cute-rs::numeric::BFloat16::from_f32",
            "cute_rs::arch::exp2_f32",
            "cute_rs::tensor::partition::{impl#3}::get",
            "cute_rs::static_layout::{impl#1}::crd2idx",
            "cute_rs::mma::bf16::{impl#1}::get_slice",
            "cute_rs::reduction::{impl#0}::observe",
        ] {
            let crate_name = path.split_once("::").unwrap().0;
            assert!(is_source_callback_path(path, crate_name), "rejected {path}");
        }
        for path in [
            "cute_rs::copy::vector::load_tile",
            "cute_rs::tensor::make_tensor_read",
            "cute_rs::tensor::{impl#6}::local_tile",
            "cute_rs::tensor::{impl#7}::load",
            "cute_rs::pipeline::{impl#4}::advance",
            "cute_rs::sm100::tcgen05_alloc",
            "cute_rs::numeric_extra::silu_bf16",
            "cute_rs::other::numeric::silu_bf16",
            "other::numeric::silu_bf16",
        ] {
            assert!(!is_source_callback_path(path, "cute_rs"), "accepted {path}");
        }
        for spec in super::CUTE_FNS {
            assert!(
                !is_source_callback_path(spec.canonical, "cute_rs"),
                "registered compiler hook escaped direct dispatch: {}",
                spec.canonical
            );
        }
    }

    /// Grouped namespaces retain separate audited source and callback boundaries.
    /// Sibling MMA/copy/attention families must not inherit admission from a
    /// broadly allowed domain parent.
    #[test]
    fn grouped_domains_keep_source_and_callback_boundaries_separate() {
        for crate_name in ["cute_rs", "cute-rs"] {
            for module in ["mma::bf16", "tensor::partition"] {
                let path = format!("{crate_name}::{module}::{{impl#3}}::load");
                assert!(is_source_abstraction_path(&path, crate_name), "{path}");
                assert!(is_source_callback_path(&path, crate_name), "{path}");
                for relative in [
                    format!("{module}_extra::load"),
                    format!("other::{module}::load"),
                ] {
                    let path = format!("{crate_name}::{relative}");
                    assert!(!is_source_callback_path(&path, crate_name), "{path}");
                }
            }
            for module in [
                "mma::block_scaled",
                "copy::cooperative",
                "copy::tiled",
                "epilogue::sm120",
                "attention::sm100",
            ] {
                let path = format!("{crate_name}::{module}::{{impl#3}}::operation");
                assert!(is_source_abstraction_path(&path, crate_name), "{path}");
                assert!(!is_source_callback_path(&path, crate_name), "{path}");
                let sibling = format!("{crate_name}::{module}_extra::operation");
                assert!(
                    !is_source_abstraction_path(&sibling, crate_name),
                    "{sibling}"
                );
            }
            for relative in [
                "mma::unregistered_instruction",
                "mma::other::operation",
                "copy::vector::unregistered_copy",
                "copy::other::operation",
                "attention::metadata::operation",
                "attention::other::operation",
                "epilogue::other::operation",
                "tensor::{impl#0}::load",
            ] {
                let path = format!("{crate_name}::{relative}");
                assert!(!is_source_callback_path(&path, crate_name), "{path}");
            }
        }
    }

    /// Check that cache-aware stores match only their exact compiler entry point.
    #[test]
    fn cache_aware_global_store_has_an_exact_compiler_boundary() {
        let spec = super::CUTE_FNS
            .iter()
            .find(|spec| spec.canonical == "cute_rs::copy::vector::store_global_no_allocate")
            .expect("global store marker");
        assert_eq!(spec.func, super::CuteFn::StoreGlobalNoAllocate);
        assert_eq!(
            spec.schema,
            &[super::SubstKind::Type, super::SubstKind::Const]
        );
    }

    /// Check that only known source modules are recognized.
    #[test]
    fn source_abstraction_modules_are_exact_and_closed() {
        assert!(is_source_abstraction_path(
            "cute_rs::arch::make_warp_uniform",
            "cute_rs"
        ));
        assert_eq!(
            super::CUTE_SOURCE_ABSTRACTION_MODULES,
            &[
                "arch",
                "mma::bf16",
                "block_scaled",
                "mma::block_scaled",
                "copy::cooperative",
                "epilogue::sm120",
                "numeric",
                "pipeline",
                "reduction",
                "scheduler",
                "sm100",
                "attention::sm100",
                "static_layout",
                "tensor",
                "copy::tiled",
            ]
        );
        assert!(is_source_abstraction_path(
            "cute_rs::block_scaled::tensor::BlockScaledTensor<...>::thread_row",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute-rs::numeric::Float4E2M1FNx2::to_f32x2",
            "cute-rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::tensor::Tensor<...>::layout",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::copy::tiled::ThrCopy<...>::copy",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::mma::block_scaled::tiled::Mxf4TiledMma<...>::gemm",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::epilogue::sm120::Sm120EpilogueWarp128x128::store_atom",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::pipeline::PipelineTmaAsync<...>::consumer_wait",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::scheduler::StaticPersistentTileScheduler<...>::get_current_work",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::copy::vector::load_tile",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::block_scaled_extra::load",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "other::block_scaled::load",
            "cute_rs"
        ));
    }

    /// Tensor partitions and typed BF16 atoms have real Rust bodies; adjacent
    /// modules and same-named modules in other crates must remain rejected.
    #[test]
    fn typed_partition_and_bf16_modules_have_closed_source_boundaries() {
        for (module, method) in [
            ("copy::cooperative", "SharedRow<...>::zip_fold"),
            ("static_layout", "Layout<...>::crd2idx"),
            ("mma::bf16", "Bf16Mma16x8x16::get_slice"),
            ("reduction", "argmax_bf16"),
        ] {
            for crate_name in ["cute_rs", "cute-rs"] {
                assert!(is_source_abstraction_path(
                    &format!("{crate_name}::{module}::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("{crate_name}::{module}_extra::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("{crate_name}::other::{module}::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("other::{module}::{method}"),
                    crate_name
                ));
            }
        }
        // Hardware/compiler stubs continue to require explicit registration.
        assert!(!is_source_abstraction_path(
            "cute_rs::mma::unregistered_instruction",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::copy::vector::unregistered_copy",
            "cute_rs"
        ));
    }

    /// Check that tensor methods accept the supported rustc impl-owner spellings.
    #[test]
    fn tensor_method_path_accepts_rustc_impl_owner_spellings() {
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::Contiguous1D>::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::{impl#6}::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute-rs::tensor::Tile1D<TILE>>::store_at"),
            Some("store_at")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::from_slice"),
            None,
            "a free function is not an inherent method"
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::other::Tile1D<TILE>>::base"),
            None
        );
    }

    /// Check that block-scaled method matching ignores only the impl-owner spelling.
    #[test]
    fn block_scaled_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::tensor::KMajor<Mode>>::from_slices"
            ),
            Some("from_slices")
        );
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::gemv::{impl#21}::dot_accumulate"
            ),
            Some("dot_accumulate")
        );
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::__compiler::block_scaled_dot_k64"
            ),
            None,
            "compiler hooks are recognized only by the exact helper table"
        );
        assert_eq!(
            block_scaled_method_name_from_path("cute_rs::other::{impl#1}::load"),
            None
        );
    }

    /// Check that scheduler method matching ignores only the impl-owner spelling.
    #[test]
    fn scheduler_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            scheduler_method_name_from_path(
                "cute_rs::scheduler::StaticPersistentTileScheduler<...>>::has_work"
            ),
            Some("has_work")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::{impl#4}::tile_idx"),
            Some("tile_idx")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::__compiler::scheduler_has_work"),
            Some("scheduler_has_work"),
            "the exact helper table takes precedence over method classification"
        );
    }

    /// Check that pipeline method matching ignores only the impl-owner spelling.
    #[test]
    fn pipeline_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::PipelineTmaAsync<...>>::producer_expect_tx"
            ),
            Some("producer_expect_tx")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::{impl#4}::advance"),
            Some("advance")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::__compiler::pipeline_consumer_wait"),
            Some("pipeline_consumer_wait"),
            "the exact helper table takes precedence over method classification"
        );
    }

    /// Check that shared MMA matching requires the exact module and method.
    #[test]
    fn shared_mma_method_path_keeps_module_and_method_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::tiled::Mxf4TiledMma<...>>::load_a_128",
                "mma::block_scaled::tiled"
            ),
            Some("load_a_128")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute-rs::copy::tiled::{impl#7}::from_raw_parts",
                "copy::tiled"
            ),
            Some("from_raw_parts")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::other::Mxf4TiledMma<...>>::load_a_128",
                "mma::block_scaled::tiled"
            ),
            None
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::load_a_128",
                "mma::block_scaled::tiled"
            ),
            None,
            "a free function is not an inherent method"
        );
    }

    /// Check that the accumulator-tile `gemm` is keyed by the fragments module,
    /// so the one-cell `Mxf4TiledMma::gemm` in the tiled module is not taken for it.
    #[test]
    fn accumulator_gemm_is_keyed_by_the_fragments_module() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::fragments::{impl#4}::gemm",
                "mma::block_scaled::fragments"
            ),
            Some("gemm")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::fragments::Mxf4AccumulatorTile2x8::gemm",
                "mma::block_scaled::fragments"
            ),
            Some("gemm")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::tiled::Mxf4TiledMma<...>>::gemm",
                "mma::block_scaled::fragments"
            ),
            None
        );
    }

    /// Check that epilogue matching requires each exact owner module.
    #[test]
    fn epilogue_method_paths_keep_each_owner_module_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::sm120::Sm120Epilogue128x128::sync_ready_for_tma",
                "epilogue::sm120"
            ),
            Some("sync_ready_for_tma")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::mma::block_scaled::epilogue::{impl#9}::store_tile",
                "mma::block_scaled::epilogue"
            ),
            Some("store_tile")
        );
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::PipelineTmaStore<1>::producer_commit"
            ),
            Some("producer_commit")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::sm120::Sm120EpilogueWarp128x128::store_tile",
                "mma::block_scaled::epilogue"
            ),
            None,
            "store_tile is defined in mma::block_scaled::epilogue, not epilogue::sm120"
        );
    }
}
