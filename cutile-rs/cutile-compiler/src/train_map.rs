/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A persistent, copy-on-write hash map with parent-chain lookup.

#![allow(dead_code)]

use std::{borrow::Borrow, collections::HashMap, hash::Hash};

/// A scoped hash map that chains to a parent for fallback lookups.
#[derive(Clone, Debug)]
pub struct TrainMap<'a, K, V> {
    pub map: HashMap<K, V>,
    pub parent: Option<&'a TrainMap<'a, K, V>>,
}

impl<'a, K: Eq + Hash, V> TrainMap<'a, K, V> {
    /// Creates an empty `TrainMap` with no parent.
    pub fn new() -> Self {
        Self {
            map: Default::default(),
            parent: None,
        }
    }

    /// Creates an empty `TrainMap` with the given initial capacity and no parent.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            parent: None,
        }
    }

    /// Returns all keys from this map and its parent chain.
    pub fn keys(&self) -> Vec<&K> {
        let mut keys = self.map.keys().collect::<Vec<_>>();
        if let Some(parent) = self.parent {
            // TODO (hme): This is inefficient.
            keys.extend(parent.keys().into_iter().collect::<Vec<_>>())
        }
        keys
    }

    /// Looks up a key, falling back to parent maps if not found locally.
    pub fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        if let Some(value) = self.map.get(key) {
            Some(value)
        } else if let Some(parent) = self.parent {
            parent.get(key)
        } else {
            None
        }
    }

    /// Inserts a key-value pair into the local map.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.map.insert(key, value)
    }

    /// Creates a child map that delegates misses to `self`.
    pub fn fork(&'a self) -> Self {
        Self {
            map: Default::default(),
            parent: Some(self),
        }
    }
}

impl<'a, K: Eq + Hash, V> Default for TrainMap<'a, K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, K: Eq + Hash, V> Extend<(K, V)> for TrainMap<'a, K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iterator: T) {
        self.map.extend(iterator)
    }
}
