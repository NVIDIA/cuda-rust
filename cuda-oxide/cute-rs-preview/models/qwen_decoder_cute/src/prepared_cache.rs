/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Bounded reuse of independently prepared launch configurations.

use std::sync::Mutex;

/// Maximum distinct grid sizes retained; additional sizes still prepare successfully.
const CAPACITY: usize = 16;

/// Per-runner cache keyed by block count, with all other launch properties fixed by its owner.
pub(super) struct PreparedCache<T> {
    /// Independently cloneable prepared values; the lock never spans driver preparation.
    entries: Mutex<Vec<(u32, T)>>,
}

impl<T> Default for PreparedCache<T> {
    /// Start with no retained launch configurations.
    fn default() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
}

impl<T: Clone> PreparedCache<T> {
    /// Clone a cached grid or prepare it without holding the cache lock.
    ///
    /// The owner must use this cache for one kernel/context and fixed non-grid
    /// launch properties. Preparation errors propagate and remain retryable.
    /// Concurrent misses may prepare twice; full or poisoned caches simply stop
    /// retaining new entries. Returned values borrow no cache state.
    pub(super) fn get_or_prepare<E>(
        &self,
        blocks: u32,
        prepare: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if let Ok(entries) = self.entries.lock()
            && let Some((_, value)) = entries.iter().find(|(key, _)| *key == blocks)
        {
            return Ok(value.clone());
        }
        // Preparation may query the driver. No lock or borrowed cache entry
        // survives into preparation, kernel submission, or GPU completion.
        let prepared = prepare()?;
        if let Ok(mut entries) = self.entries.lock() {
            // Another host thread may have prepared the same grid meanwhile.
            if let Some((_, value)) = entries.iter().find(|(key, _)| *key == blocks) {
                return Ok(value.clone());
            }
            if entries.len() < CAPACITY {
                entries.push((blocks, prepared.clone()));
            }
        }
        // A full or poisoned cache only disables retention of this new value.
        Ok(prepared)
    }
}

/// CPU-only cache identity, capacity, error, and poisoning checks.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Verify a hit clones the matching grid's prepared value, not another grid's.
    #[test]
    fn reuses_only_the_requested_grid() {
        let cache = PreparedCache::default();
        let first = cache
            .get_or_prepare(4, || Ok::<_, ()>(Arc::new(4)))
            .unwrap();
        let second = cache
            .get_or_prepare(8, || Ok::<_, ()>(Arc::new(8)))
            .unwrap();
        let reused = cache
            .get_or_prepare(4, || Err("cached grid prepared again"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &reused));
        assert!(!Arc::ptr_eq(&second, &reused));
        assert_eq!(*second, 8);
    }

    /// Ensure an unsuccessful preparation does not occupy or poison an entry.
    #[test]
    fn failed_preparation_can_be_retried() {
        let cache = PreparedCache::default();
        assert_eq!(
            cache.get_or_prepare(4, || Err::<Arc<u32>, _>("failure")),
            Err("failure")
        );
        let prepared = cache
            .get_or_prepare(4, || Ok::<_, ()>(Arc::new(4)))
            .unwrap();
        let reused = cache
            .get_or_prepare(4, || Err("successful retry was not retained"))
            .unwrap();
        assert!(Arc::ptr_eq(&prepared, &reused));
    }

    /// Check bounded ownership while allowing arbitrary uncached grid sizes.
    #[test]
    fn bounds_retention_without_rejecting_new_grids() {
        let cache = PreparedCache::default();
        let mut values = Vec::new();
        for blocks in 1..=48 {
            let value = cache
                .get_or_prepare(blocks, || Ok::<_, ()>(Arc::new(blocks)))
                .unwrap();
            assert_eq!(*value, blocks);
            values.push(Arc::downgrade(&value));
        }
        let retained = values
            .iter()
            .filter(|value| value.strong_count() != 0)
            .count();
        assert!(retained > 0 && retained <= 16);
        drop(cache);
        assert!(values.iter().all(|value| value.strong_count() == 0));
    }

    /// Verify poisoned metadata disables reuse without preventing fresh preparation.
    #[test]
    fn poisoned_cache_falls_back_to_fresh_preparation() {
        let cache = PreparedCache::default();
        let first = cache
            .get_or_prepare(4, || Ok::<_, ()>(Arc::new(4)))
            .unwrap();
        let _ = std::panic::catch_unwind(|| {
            let _guard = cache.entries.lock().unwrap();
            panic!("poison cache metadata");
        });
        let fresh = cache
            .get_or_prepare(4, || Ok::<_, ()>(Arc::new(8)))
            .unwrap();
        assert_eq!(*fresh, 8);
        assert!(!Arc::ptr_eq(&first, &fresh));
    }
}
