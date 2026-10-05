/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use core::convert::Infallible;
use std::collections::HashMap;

use cuda_host::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelectionCache,
    KernelSelectionError, KernelSelector, KernelVariant, NoKernelSelectionCache, SelectionMode,
    SelectionSource,
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    A,
    B,
    C,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Metadata(&'static str);

type Variant = KernelVariant<VariantId, &'static str, Metadata>;
type Family<const N: usize> = KernelFamily<VariantId, &'static str, Metadata, N>;

fn variant(id: VariantId) -> Variant {
    let label = match id {
        VariantId::A => "a",
        VariantId::B => "b",
        VariantId::C => "c",
        VariantId::Unknown => "unknown",
    };
    KernelVariant::new(id, label, Metadata(label))
}

fn family() -> Family<3> {
    KernelFamily::try_new(
        "tests/family",
        7,
        [
            variant(VariantId::A),
            variant(VariantId::B),
            variant(VariantId::C),
        ],
    )
    .unwrap()
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Problem {
    key: u32,
    rejected: Vec<VariantId>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("variant {0:?} is ineligible")]
struct Rejection(VariantId);

impl KernelProblem<Variant> for Problem {
    type Rejection = Rejection;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection> {
        if self.rejected.contains(variant.id()) {
            Err(Rejection(*variant.id()))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("selector failed")]
struct SelectorFailure;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectorAnswer {
    Variant(VariantId),
    Fail,
}

struct SpySelector {
    answer: SelectorAnswer,
    calls: usize,
    families: Vec<KernelFamilyId>,
    eligible: Vec<Vec<VariantId>>,
}

impl SpySelector {
    fn returning(id: VariantId) -> Self {
        Self {
            answer: SelectorAnswer::Variant(id),
            calls: 0,
            families: Vec::new(),
            eligible: Vec::new(),
        }
    }

    fn failing() -> Self {
        Self {
            answer: SelectorAnswer::Fail,
            calls: 0,
            families: Vec::new(),
            eligible: Vec::new(),
        }
    }
}

impl KernelSelector<Problem, Variant, VariantId> for SpySelector {
    type Error = SelectorFailure;

    fn select(
        &mut self,
        family: KernelFamilyId,
        _problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<VariantId, Self::Error> {
        self.calls += 1;
        self.families.push(family);
        self.eligible
            .push(eligible.iter().map(|variant| *variant.id()).collect());
        match self.answer {
            SelectorAnswer::Variant(id) => Ok(id),
            SelectorAnswer::Fail => Err(SelectorFailure),
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("cache failed")]
struct CacheFailure;

struct SpyCache {
    lookup: Result<Option<VariantId>, CacheFailure>,
    fail_store: bool,
    lookup_calls: usize,
    lookup_keys: Vec<(KernelFamilyId, u32)>,
    stores: Vec<(KernelFamilyId, u32, VariantId)>,
}

impl SpyCache {
    fn with_lookup(id: Option<VariantId>) -> Self {
        Self {
            lookup: Ok(id),
            fail_store: false,
            lookup_calls: 0,
            lookup_keys: Vec::new(),
            stores: Vec::new(),
        }
    }
}

impl KernelSelectionCache<Problem, VariantId> for SpyCache {
    type Error = CacheFailure;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<VariantId>, Self::Error> {
        self.lookup_calls += 1;
        self.lookup_keys.push((family, problem.key));
        self.lookup
    }

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &VariantId,
    ) -> Result<(), Self::Error> {
        if self.fail_store {
            return Err(CacheFailure);
        }
        self.stores.push((family, problem.key, *variant));
        Ok(())
    }
}

#[derive(Default)]
struct KeyedCache {
    values: HashMap<(KernelFamilyId, u32), VariantId>,
}

impl KernelSelectionCache<Problem, VariantId> for KeyedCache {
    type Error = Infallible;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<VariantId>, Self::Error> {
        Ok(self.values.get(&(family, problem.key)).copied())
    }

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &VariantId,
    ) -> Result<(), Self::Error> {
        self.values.insert((family, problem.key), *variant);
        Ok(())
    }
}

#[test]
fn construction_rejects_empty_family_blank_name_and_duplicate_ids() {
    let empty = KernelFamily::<VariantId, (), (), 0>::try_new("empty", 1, []);
    assert_eq!(empty, Err(KernelFamilyBuildError::EmptyFamily));

    let blank = Family::try_new(
        "  ",
        1,
        [
            variant(VariantId::A),
            variant(VariantId::B),
            variant(VariantId::C),
        ],
    );
    assert_eq!(blank, Err(KernelFamilyBuildError::EmptyFamilyName));

    let duplicate = Family::try_new(
        "duplicate",
        1,
        [
            variant(VariantId::A),
            variant(VariantId::B),
            variant(VariantId::A),
        ],
    );
    assert_eq!(
        duplicate,
        Err(KernelFamilyBuildError::DuplicateVariantId {
            first_index: 0,
            duplicate_index: 2,
        })
    );
}

#[test]
fn force_validates_and_bypasses_cache_and_selector() {
    let family = family();
    let problem = Problem::default();
    let mut selector = SpySelector::returning(VariantId::A);
    let mut cache = SpyCache::with_lookup(Some(VariantId::A));

    let selected = family
        .select(
            &problem,
            SelectionMode::Force(VariantId::B),
            &mut selector,
            &mut cache,
        )
        .unwrap();

    assert_eq!(selected.variant().id(), &VariantId::B);
    assert_eq!(selected.source(), SelectionSource::Override);
    assert_eq!(selector.calls, 0);
    assert_eq!(cache.lookup_calls, 0);
    assert!(cache.stores.is_empty());
}

#[test]
fn force_reports_unknown_and_ineligible_ids() {
    let family = family();
    let mut selector = SpySelector::returning(VariantId::A);
    let mut cache = SpyCache::with_lookup(None);

    let unknown = family.select(
        &Problem::default(),
        SelectionMode::Force(VariantId::Unknown),
        &mut selector,
        &mut cache,
    );
    assert!(matches!(
        unknown,
        Err(KernelSelectionError::UnknownForcedVariant {
            id: VariantId::Unknown
        })
    ));

    let ineligible = family.select(
        &Problem {
            key: 0,
            rejected: vec![VariantId::B],
        },
        SelectionMode::Force(VariantId::B),
        &mut selector,
        &mut cache,
    );
    assert!(matches!(
        ineligible,
        Err(KernelSelectionError::IneligibleForcedVariant {
            id: VariantId::B,
            rejection: Rejection(VariantId::B),
        })
    ));
}

#[test]
fn eligible_cache_hit_wins_without_selector_or_store() {
    let family = family();
    let problem = Problem {
        key: 42,
        rejected: Vec::new(),
    };
    let mut selector = SpySelector::returning(VariantId::A);
    let mut cache = SpyCache::with_lookup(Some(VariantId::C));

    let selected = family
        .select(&problem, SelectionMode::Auto, &mut selector, &mut cache)
        .unwrap();

    assert_eq!(selected.variant().id(), &VariantId::C);
    assert_eq!(selected.source(), SelectionSource::Cache);
    assert_eq!(selector.calls, 0);
    assert!(cache.stores.is_empty());
    assert_eq!(cache.lookup_keys, vec![(family.id(), 42)]);
}

#[test]
fn stale_or_ineligible_cache_entries_fall_back_and_are_repaired() {
    for cached in [VariantId::Unknown, VariantId::B] {
        let family = family();
        let problem = Problem {
            key: 9,
            rejected: vec![VariantId::B],
        };
        let mut selector = SpySelector::returning(VariantId::C);
        let mut cache = SpyCache::with_lookup(Some(cached));

        let selected = family
            .select(&problem, SelectionMode::Auto, &mut selector, &mut cache)
            .unwrap();

        assert_eq!(selected.variant().id(), &VariantId::C);
        assert_eq!(selected.source(), SelectionSource::Selector);
        assert_eq!(selector.calls, 1);
        assert_eq!(cache.stores, vec![(family.id(), 9, VariantId::C)]);
    }
}

#[test]
fn selector_sees_only_eligible_variants_in_declaration_order() {
    let family = family();
    let problem = Problem {
        key: 5,
        rejected: vec![VariantId::B],
    };
    let mut selector = SpySelector::returning(VariantId::C);
    let mut cache = SpyCache::with_lookup(None);

    let selected = family
        .select(&problem, SelectionMode::Auto, &mut selector, &mut cache)
        .unwrap();

    assert_eq!(selected.variant().id(), &VariantId::C);
    assert_eq!(selector.eligible, vec![vec![VariantId::A, VariantId::C]]);
    assert_eq!(selector.families, vec![family.id()]);
    assert_eq!(cache.stores, vec![(family.id(), 5, VariantId::C)]);
}

#[test]
fn no_eligible_variants_skips_selector_and_store() {
    let family = family();
    let problem = Problem {
        key: 0,
        rejected: vec![VariantId::A, VariantId::B, VariantId::C],
    };
    let mut selector = SpySelector::returning(VariantId::A);
    let mut cache = SpyCache::with_lookup(None);

    let result = family.select(&problem, SelectionMode::Auto, &mut selector, &mut cache);

    assert!(matches!(
        result,
        Err(KernelSelectionError::NoEligibleVariants {
            family: rejected_family,
            variant_count: 3,
        }) if rejected_family == family.id()
    ));
    assert_eq!(selector.calls, 0);
    assert!(cache.stores.is_empty());
}

#[test]
fn selector_unknown_ineligible_and_failure_outputs_are_not_cached() {
    let family = family();
    let problem = Problem {
        key: 0,
        rejected: vec![VariantId::B],
    };

    let mut unknown_selector = SpySelector::returning(VariantId::Unknown);
    let mut unknown_cache = SpyCache::with_lookup(None);
    let unknown = family.select(
        &problem,
        SelectionMode::Auto,
        &mut unknown_selector,
        &mut unknown_cache,
    );
    assert!(matches!(
        unknown,
        Err(KernelSelectionError::SelectorReturnedUnknownVariant {
            id: VariantId::Unknown
        })
    ));
    assert!(unknown_cache.stores.is_empty());

    let mut ineligible_selector = SpySelector::returning(VariantId::B);
    let mut ineligible_cache = SpyCache::with_lookup(None);
    let ineligible = family.select(
        &problem,
        SelectionMode::Auto,
        &mut ineligible_selector,
        &mut ineligible_cache,
    );
    assert!(matches!(
        ineligible,
        Err(KernelSelectionError::SelectorReturnedIneligibleVariant { id: VariantId::B })
    ));
    assert!(ineligible_cache.stores.is_empty());

    let mut failing_selector = SpySelector::failing();
    let mut failing_cache = SpyCache::with_lookup(None);
    let failure = family.select(
        &problem,
        SelectionMode::Auto,
        &mut failing_selector,
        &mut failing_cache,
    );
    assert!(matches!(
        failure,
        Err(KernelSelectionError::SelectorFailed(SelectorFailure))
    ));
    assert!(failing_cache.stores.is_empty());
}

#[test]
fn cache_lookup_and_store_failures_are_distinct() {
    let family = family();
    let problem = Problem::default();
    let mut selector = SpySelector::returning(VariantId::A);
    let mut lookup_cache = SpyCache {
        lookup: Err(CacheFailure),
        fail_store: false,
        lookup_calls: 0,
        lookup_keys: Vec::new(),
        stores: Vec::new(),
    };
    let lookup = family.select(
        &problem,
        SelectionMode::Auto,
        &mut selector,
        &mut lookup_cache,
    );
    assert!(matches!(
        lookup,
        Err(KernelSelectionError::CacheLookupFailed(CacheFailure))
    ));
    assert_eq!(selector.calls, 0);

    let mut store_selector = SpySelector::returning(VariantId::A);
    let mut store_cache = SpyCache::with_lookup(None);
    store_cache.fail_store = true;
    let store = family.select(
        &problem,
        SelectionMode::Auto,
        &mut store_selector,
        &mut store_cache,
    );
    assert!(matches!(
        store,
        Err(KernelSelectionError::CacheStoreFailed(CacheFailure))
    ));
}

#[test]
fn cache_receives_family_name_revision_and_uses_stable_id_not_position() {
    let reordered = KernelFamily::try_new(
        "tests/family",
        8,
        [
            variant(VariantId::C),
            variant(VariantId::A),
            variant(VariantId::B),
        ],
    )
    .unwrap();
    let problem = Problem {
        key: 77,
        rejected: Vec::new(),
    };
    let mut selector = SpySelector::returning(VariantId::A);
    let mut cache = SpyCache::with_lookup(Some(VariantId::B));

    let selected = reordered
        .select(&problem, SelectionMode::Auto, &mut selector, &mut cache)
        .unwrap();

    assert_eq!(selected.variant().id(), &VariantId::B);
    assert_eq!(selected.variant().entry(), &"b");
    assert_eq!(cache.lookup_keys, vec![(reordered.id(), 77)]);
    assert_eq!(reordered.id().name(), "tests/family");
    assert_eq!(reordered.id().revision(), 8);
}

#[test]
fn family_revision_namespaces_real_cache_entries() {
    let variants = || {
        [
            variant(VariantId::A),
            variant(VariantId::B),
            variant(VariantId::C),
        ]
    };
    let revision_1 = KernelFamily::try_new("revisioned", 1, variants()).unwrap();
    let revision_2 = KernelFamily::try_new("revisioned", 2, variants()).unwrap();
    let problem = Problem {
        key: 11,
        rejected: Vec::new(),
    };
    let mut cache = KeyedCache::default();

    let mut first_selector = SpySelector::returning(VariantId::B);
    let first = revision_1
        .select(
            &problem,
            SelectionMode::Auto,
            &mut first_selector,
            &mut cache,
        )
        .unwrap();
    assert_eq!(first.variant().id(), &VariantId::B);
    assert_eq!(first.source(), SelectionSource::Selector);

    let mut revised_selector = SpySelector::returning(VariantId::C);
    let revised = revision_2
        .select(
            &problem,
            SelectionMode::Auto,
            &mut revised_selector,
            &mut cache,
        )
        .unwrap();
    assert_eq!(revised.variant().id(), &VariantId::C);
    assert_eq!(revised.source(), SelectionSource::Selector);

    let mut bypassed_selector = SpySelector::returning(VariantId::A);
    let cached = revision_2
        .select(
            &problem,
            SelectionMode::Auto,
            &mut bypassed_selector,
            &mut cache,
        )
        .unwrap();
    assert_eq!(cached.variant().id(), &VariantId::C);
    assert_eq!(cached.source(), SelectionSource::Cache);
    assert_eq!(bypassed_selector.calls, 0);
    assert_eq!(cache.values.len(), 2);
}

struct AlwaysEligible;

impl<Variant> KernelProblem<Variant> for AlwaysEligible {
    type Rejection = Infallible;

    fn validate(&self, _variant: &Variant) -> Result<(), Self::Rejection> {
        Ok(())
    }
}

#[test]
fn entries_and_metadata_do_not_need_to_be_copy() {
    let family = KernelFamily::try_new(
        "non-copy",
        1,
        [KernelVariant::new(
            VariantId::A,
            String::from("entry"),
            String::from("metadata"),
        )],
    )
    .unwrap();
    let mut selector =
        |_: KernelFamilyId,
         _: &AlwaysEligible,
         eligible: &[&KernelVariant<VariantId, String, String>]| {
            Ok::<_, Infallible>(*eligible[0].id())
        };
    let mut cache = NoKernelSelectionCache;

    let selected = family
        .select(
            &AlwaysEligible,
            SelectionMode::Auto,
            &mut selector,
            &mut cache,
        )
        .unwrap();

    fn assert_copy<T: Copy>(_: T) {}
    assert_copy(selected);
    assert_eq!(selected.variant().entry(), "entry");
    assert_eq!(selected.variant().metadata(), "metadata");
}

#[test]
fn ordinary_family_types_are_send_and_sync_without_unsafe_impls() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Family<3>>();
}
