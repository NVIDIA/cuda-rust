/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded families of ahead-of-time compiled kernel variants.
//!
//! A kernel family gives stable IDs to a small, fixed set of entries and keeps
//! selection policy outside the kernels themselves:
//!
//! ```text
//! Force(id) -> resolve + validate --------------------------> Override
//! Auto      -> eligible cache hit --------------------------> Cache
//!           -> selector(eligible variants) -> cache store --> Selector
//! ```
//!
//! The module is deliberately CPU-only. Callers decide which problem and
//! hardware facts matter, how to select among eligible variants, and whether
//! to keep an in-memory or persistent cache.

use core::{convert::Infallible, fmt};

use thiserror::Error;

/// Stable namespace for a kernel family and its cache entries.
///
/// Bump `revision` when variant membership, eligibility, selector/preference
/// policy, tuning methodology, or the meaning of a variant ID changes. A
/// reorder also requires a bump unless selection and tuning are explicitly
/// order-independent: caches store stable IDs, but selectors receive eligible
/// variants in declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelFamilyId {
    name: &'static str,
    revision: u32,
}

impl KernelFamilyId {
    /// Stable family name.
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Selection/cache contract revision.
    pub const fn revision(self) -> u32 {
        self.revision
    }
}

impl fmt::Display for KernelFamilyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.revision)
    }
}

/// One ahead-of-time compiled member of a [`KernelFamily`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelVariant<Id, Entry, Meta> {
    id: Id,
    entry: Entry,
    metadata: Meta,
}

impl<Id, Entry, Meta> KernelVariant<Id, Entry, Meta> {
    /// Creates a variant from its stable ID, callable entry, and policy data.
    pub const fn new(id: Id, entry: Entry, metadata: Meta) -> Self {
        Self {
            id,
            entry,
            metadata,
        }
    }

    /// Stable variant ID used by overrides and caches.
    pub const fn id(&self) -> &Id {
        &self.id
    }

    /// Callable entry associated with this variant.
    pub const fn entry(&self) -> &Entry {
        &self.entry
    }

    /// Caller-defined eligibility and selection metadata.
    pub const fn metadata(&self) -> &Meta {
        &self.metadata
    }

    /// Consumes the variant and returns its parts.
    pub fn into_parts(self) -> (Id, Entry, Meta) {
        (self.id, self.entry, self.metadata)
    }
}

/// Construction failure for a [`KernelFamily`].
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum KernelFamilyBuildError {
    /// A family must contain at least one compiled variant.
    #[error("kernel family must contain at least one variant")]
    EmptyFamily,

    /// The family name is a stable cache namespace and cannot be blank.
    #[error("kernel family name cannot be empty or whitespace-only")]
    EmptyFamilyName,

    /// Two entries used the same stable variant ID.
    #[error(
        "kernel family contains duplicate variant IDs at indices {first_index} and {duplicate_index}"
    )]
    DuplicateVariantId {
        /// Index of the first occurrence.
        first_index: usize,
        /// Index of the repeated occurrence.
        duplicate_index: usize,
    },
}

/// A small, immutable set of ahead-of-time compiled kernel variants.
///
/// `N` is part of the type so library authors can expose a bounded menu of
/// supported kernels without runtime registration or JIT compilation.
/// Automatic candidate filtering uses a fixed stack array and does not
/// allocate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFamily<Id, Entry, Meta, const N: usize> {
    id: KernelFamilyId,
    variants: [KernelVariant<Id, Entry, Meta>; N],
}

impl<Id: Eq, Entry, Meta, const N: usize> KernelFamily<Id, Entry, Meta, N> {
    /// Creates a validated family.
    ///
    /// Variant IDs must be stable semantic values. Array indices, pointers,
    /// `TypeId`s, and formatted debug strings are not stable cache identities.
    pub fn try_new(
        name: &'static str,
        revision: u32,
        variants: [KernelVariant<Id, Entry, Meta>; N],
    ) -> Result<Self, KernelFamilyBuildError> {
        if N == 0 {
            return Err(KernelFamilyBuildError::EmptyFamily);
        }
        if name.trim().is_empty() {
            return Err(KernelFamilyBuildError::EmptyFamilyName);
        }

        for duplicate_index in 1..N {
            if let Some(first_index) = (0..duplicate_index)
                .find(|&first_index| variants[first_index].id == variants[duplicate_index].id)
            {
                return Err(KernelFamilyBuildError::DuplicateVariantId {
                    first_index,
                    duplicate_index,
                });
            }
        }

        Ok(Self {
            id: KernelFamilyId { name, revision },
            variants,
        })
    }

    /// Stable name and revision passed to cache and selector implementations.
    pub const fn id(&self) -> KernelFamilyId {
        self.id
    }

    /// Variants in declaration order.
    pub const fn variants(&self) -> &[KernelVariant<Id, Entry, Meta>; N] {
        &self.variants
    }

    /// Looks up a variant by stable ID.
    pub fn get(&self, id: &Id) -> Option<&KernelVariant<Id, Entry, Meta>> {
        self.variants.iter().find(|variant| variant.id() == id)
    }
}

/// Problem-specific eligibility check for a compiled variant.
///
/// Implementations should be deterministic and side-effect free. Selection may
/// revalidate an untrusted cache entry before constructing the eligible list.
pub trait KernelProblem<Variant> {
    /// Why this variant cannot safely handle the problem. The thread-portable
    /// bound lets a complete selection error cross host worker/async boundaries.
    type Rejection: std::error::Error + Send + Sync + 'static;

    /// Returns `Ok(())` only when launching `variant` is valid for this problem.
    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection>;
}

/// Policy that chooses one ID from an already validated candidate slice.
pub trait KernelSelector<Problem, Variant, Id> {
    /// Selector failure unrelated to ordinary variant ineligibility. The
    /// thread-portable bound matches host worker/async selection use.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Chooses a stable variant ID.
    ///
    /// `eligible` preserves family declaration order and is never empty.
    fn select(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<Id, Self::Error>;
}

impl<Problem, Variant, Id, SelectorError, F> KernelSelector<Problem, Variant, Id> for F
where
    SelectorError: std::error::Error + Send + Sync + 'static,
    F: FnMut(KernelFamilyId, &Problem, &[&Variant]) -> Result<Id, SelectorError>,
{
    type Error = SelectorError;

    fn select(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<Id, Self::Error> {
        self(family, problem, eligible)
    }
}

/// Cache adapter for automatic family selection.
///
/// Implementations must include both `family.name()` and `family.revision()` in
/// their key. Cached IDs are treated as untrusted hints: unknown or newly
/// ineligible values fall back to the selector and are overwritten.
/// Best-effort persistent adapters can translate I/O failures into misses or
/// no-op stores; an error returned from this trait deliberately stops selection.
pub trait KernelSelectionCache<Problem, Id> {
    /// Cache backend failure. The thread-portable bound matches host
    /// worker/async selection use.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Looks up a cached stable variant ID for this family/problem pair.
    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<Id>, Self::Error>;

    /// Stores a selector result after the family has revalidated it.
    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &Id,
    ) -> Result<(), Self::Error>;
}

/// Cache adapter that always misses and never stores anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoKernelSelectionCache;

impl<Problem, Id> KernelSelectionCache<Problem, Id> for NoKernelSelectionCache {
    type Error = Infallible;

    fn lookup(
        &mut self,
        _family: KernelFamilyId,
        _problem: &Problem,
    ) -> Result<Option<Id>, Self::Error> {
        Ok(None)
    }

    fn store(
        &mut self,
        _family: KernelFamilyId,
        _problem: &Problem,
        _variant: &Id,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Whether selection is automatic or pinned by a caller override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode<Id> {
    /// Consult the cache, then the selector.
    Auto,
    /// Require this stable variant ID. Cache and selector are bypassed.
    Force(Id),
}

/// Why a variant was selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionSource {
    /// An explicit [`SelectionMode::Force`] override.
    Override,
    /// A validated cache hit.
    Cache,
    /// The automatic selector.
    Selector,
}

impl fmt::Display for SelectionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Override => f.write_str("override"),
            Self::Cache => f.write_str("cache"),
            Self::Selector => f.write_str("selector"),
        }
    }
}

/// Borrowed result of family selection.
#[derive(Debug, PartialEq, Eq)]
pub struct SelectedVariant<'family, Id, Entry, Meta> {
    variant: &'family KernelVariant<Id, Entry, Meta>,
    source: SelectionSource,
}

impl<Id, Entry, Meta> Copy for SelectedVariant<'_, Id, Entry, Meta> {}

impl<Id, Entry, Meta> Clone for SelectedVariant<'_, Id, Entry, Meta> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'family, Id, Entry, Meta> SelectedVariant<'family, Id, Entry, Meta> {
    /// Selected compiled variant.
    pub const fn variant(&self) -> &'family KernelVariant<Id, Entry, Meta> {
        self.variant
    }

    /// Override, cache, or selector provenance.
    pub const fn source(&self) -> SelectionSource {
        self.source
    }
}

/// Failure to select a valid member of a [`KernelFamily`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelSelectionError<Id, Rejection, SelectorError, CacheError>
where
    Id: fmt::Debug,
    Rejection: std::error::Error + 'static,
    SelectorError: std::error::Error + 'static,
    CacheError: std::error::Error + 'static,
{
    /// Explicit override did not name a family member.
    #[error("forced kernel variant {id:?} is not part of the family")]
    UnknownForcedVariant {
        /// Unknown requested ID.
        id: Id,
    },

    /// Explicit override named a member that cannot handle this problem.
    #[error("forced kernel variant {id:?} is ineligible: {rejection}")]
    IneligibleForcedVariant {
        /// Requested ID.
        id: Id,
        /// Problem-specific safety rejection.
        #[source]
        rejection: Rejection,
    },

    /// No compiled member can safely handle the problem.
    #[error("no variant in kernel family {family} is eligible ({variant_count} checked)")]
    NoEligibleVariants {
        /// Family whose variants were checked.
        family: KernelFamilyId,
        /// Number of rejected compiled variants.
        variant_count: usize,
    },

    /// Selector itself failed.
    #[error("kernel selector failed: {0}")]
    SelectorFailed(#[source] SelectorError),

    /// Selector returned an ID not owned by this family.
    #[error("kernel selector returned unknown variant {id:?}")]
    SelectorReturnedUnknownVariant {
        /// Unknown returned ID.
        id: Id,
    },

    /// Selector returned a known ID that was not in its eligible input slice.
    #[error("kernel selector returned ineligible variant {id:?}")]
    SelectorReturnedIneligibleVariant {
        /// Ineligible returned ID.
        id: Id,
    },

    /// Cache backend could not perform its lookup.
    #[error("kernel selection cache lookup failed: {0}")]
    CacheLookupFailed(#[source] CacheError),

    /// Cache backend could not store a validated selector result.
    #[error("kernel selection cache store failed: {0}")]
    CacheStoreFailed(#[source] CacheError),
}

/// Result type returned by [`KernelFamily::select`].
pub type KernelSelectionResult<'family, Id, Entry, Meta, Problem, Selector, Cache> = Result<
    SelectedVariant<'family, Id, Entry, Meta>,
    KernelSelectionError<
        Id,
        <Problem as KernelProblem<KernelVariant<Id, Entry, Meta>>>::Rejection,
        <Selector as KernelSelector<Problem, KernelVariant<Id, Entry, Meta>, Id>>::Error,
        <Cache as KernelSelectionCache<Problem, Id>>::Error,
    >,
>;

impl<Id, Entry, Meta, const N: usize> KernelFamily<Id, Entry, Meta, N>
where
    Id: Eq + fmt::Debug,
{
    /// Selects a valid variant with explicit provenance.
    ///
    /// `Force` bypasses both cache and selector. `Auto` validates cache hits,
    /// treats stale/unknown/ineligible IDs as misses, then gives the selector
    /// only eligible variants. Selector output is checked again before storage.
    pub fn select<'family, Problem, Selector, Cache>(
        &'family self,
        problem: &Problem,
        mode: SelectionMode<Id>,
        selector: &mut Selector,
        cache: &mut Cache,
    ) -> KernelSelectionResult<'family, Id, Entry, Meta, Problem, Selector, Cache>
    where
        Problem: KernelProblem<KernelVariant<Id, Entry, Meta>>,
        Selector: KernelSelector<Problem, KernelVariant<Id, Entry, Meta>, Id>,
        Cache: KernelSelectionCache<Problem, Id>,
    {
        if let SelectionMode::Force(id) = mode {
            let Some(variant) = self.get(&id) else {
                return Err(KernelSelectionError::UnknownForcedVariant { id });
            };
            problem.validate(variant).map_err(|rejection| {
                KernelSelectionError::IneligibleForcedVariant { id, rejection }
            })?;
            return Ok(SelectedVariant {
                variant,
                source: SelectionSource::Override,
            });
        }

        if let Some(cached_id) = cache
            .lookup(self.id, problem)
            .map_err(KernelSelectionError::CacheLookupFailed)?
            && let Some(variant) = self.get(&cached_id)
            && problem.validate(variant).is_ok()
        {
            return Ok(SelectedVariant {
                variant,
                source: SelectionSource::Cache,
            });
        }

        // Keep automatic selection allocation-free: the family is already a
        // fixed array, so compact eligible references into another fixed array
        // while preserving declaration order.
        let mut eligible: [&KernelVariant<Id, Entry, Meta>; N] =
            core::array::from_fn(|index| &self.variants[index]);
        let mut eligible_count = 0;
        for index in 0..N {
            let candidate = eligible[index];
            if problem.validate(candidate).is_ok() {
                eligible[eligible_count] = candidate;
                eligible_count += 1;
            }
        }
        let eligible = &eligible[..eligible_count];
        if eligible.is_empty() {
            return Err(KernelSelectionError::NoEligibleVariants {
                family: self.id,
                variant_count: N,
            });
        }

        let selected_id = selector
            .select(self.id, problem, eligible)
            .map_err(KernelSelectionError::SelectorFailed)?;
        let Some(selected) = self.get(&selected_id) else {
            return Err(KernelSelectionError::SelectorReturnedUnknownVariant { id: selected_id });
        };
        if !eligible.iter().any(|variant| variant.id() == &selected_id) {
            return Err(KernelSelectionError::SelectorReturnedIneligibleVariant {
                id: selected_id,
            });
        }

        cache
            .store(self.id, problem, &selected_id)
            .map_err(KernelSelectionError::CacheStoreFailed)?;
        Ok(SelectedVariant {
            variant: selected,
            source: SelectionSource::Selector,
        })
    }
}
