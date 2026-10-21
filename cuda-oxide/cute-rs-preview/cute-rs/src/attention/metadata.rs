/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked host metadata for BF16 attention and contiguous key/value caches.
//!
//! Query heads, KV heads and head width remain separate dimensions. In
//! particular, a model's hidden width need not equal its query projection width.
//! Element counts describe dense `[token, head, dimension]` storage, with
//! dimension coordinates contiguous. Cache capacities and append ranges count
//! tokens, while allocation lengths count BF16 elements in each of K and V.
//!
//! These types neither allocate buffers nor track initialized data. A validated
//! shape does not establish allocation size, context, readiness or ownership.
//! The cache owner must check those properties, preserve initialized contents,
//! and advance its cursor only after a successful append. Calling
//! [`CacheShape::append_range`] does not reserve or initialize any storage.
//!
//! ```
//! use cute_rs::attention::{AttentionShape, CacheShape};
//! let attention = AttentionShape::<16, 8, 128>::new().unwrap();
//! assert_eq!(attention.query_width(), 2048);
//! assert_eq!(attention.kv_width(), 1024);
//! let cache = CacheShape::<8, 128>::new(65).unwrap();
//! assert_eq!(cache.append_range(17, 3).unwrap(), 17..20);
//! ```

use core::{fmt, mem::size_of, ops::Range};

use crate::numeric::Bf16;

/// A rejected attention dimension, token window, BF16 extent or cache append.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionError {
    /// A head count or head width is zero.
    ZeroDimension,
    /// Query heads do not form complete, equally sized groups of KV heads.
    InvalidHeadGrouping,
    /// Token/element arithmetic overflowed or a BF16 extent exceeds `isize::MAX` bytes.
    ExtentOverflow,
    /// The supplied initialized-token cursor lies beyond cache capacity.
    InvalidCursor,
    /// An append would exceed the cache's token capacity.
    CapacityExceeded,
    /// A nonempty query sequence has no keys to attend to.
    MissingKeys,
    /// A causal query range lies outside the available key sequence.
    InvalidWindow,
    /// A query-row index lies outside the window's query sequence.
    InvalidQueryRow,
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroDimension => "attention head counts and head width must be positive",
            Self::InvalidHeadGrouping => "query head count must be a multiple of KV head count",
            Self::ExtentOverflow => {
                "attention extent overflowed or exceeds the BF16 addressable range"
            }
            Self::InvalidCursor => "initialized-token cursor exceeds cache capacity",
            Self::CapacityExceeded => "append exceeds cache token capacity",
            Self::MissingKeys => "nonempty attention queries require at least one key",
            Self::InvalidWindow => "causal query range exceeds the available key sequence",
            Self::InvalidQueryRow => "query row exceeds the attention window",
        })
    }
}

impl core::error::Error for AttentionError {}

/// Validate the complete byte extent of one BF16 allocation or slice.
fn check_extent(elements: usize) -> Result<usize, AttentionError> {
    match elements.checked_mul(size_of::<Bf16>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(elements),
        _ => Err(AttentionError::ExtentOverflow),
    }
}

/// Validate positive head dimensions and the BF16 extent of one token.
fn checked_width(heads: usize, head_dim: usize) -> Result<usize, AttentionError> {
    if heads == 0 || head_dim == 0 {
        return Err(AttentionError::ZeroDimension);
    }
    check_extent(
        heads
            .checked_mul(head_dim)
            .ok_or(AttentionError::ExtentOverflow)?,
    )
}

/// Validate token multiplication and the full BF16 slice byte extent.
fn checked_len(tokens: usize, width: usize) -> Result<usize, AttentionError> {
    check_extent(
        tokens
            .checked_mul(width)
            .ok_or(AttentionError::ExtentOverflow)?,
    )
}

/// Validated query/KV head geometry, independent of a model's hidden width.
///
/// `QH` must be a positive multiple of `KVH`, and `D` must be positive. No
/// architecture-specific head-width or tile-size restriction is imposed here.
/// Kernels must check any stronger specialization requirements at their boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape<const QH: usize, const KVH: usize, const D: usize> {
    /// BF16 elements per token in a query or attention output tensor.
    query_width: usize,
    /// BF16 elements per token in a key or value tensor.
    kv_width: usize,
}

impl<const QH: usize, const KVH: usize, const D: usize> AttentionShape<QH, KVH, D> {
    /// Check grouped-query geometry and both per-token BF16 extents.
    pub fn new() -> Result<Self, AttentionError> {
        if QH == 0 || KVH == 0 || D == 0 {
            return Err(AttentionError::ZeroDimension);
        }
        if !QH.is_multiple_of(KVH) {
            return Err(AttentionError::InvalidHeadGrouping);
        }
        Ok(Self {
            query_width: checked_width(QH, D)?,
            kv_width: checked_width(KVH, D)?,
        })
    }

    /// Number of query heads.
    #[must_use]
    pub const fn query_heads(self) -> usize {
        QH
    }

    /// Number of key/value heads.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        KVH
    }

    /// Coordinates in each query, key or value head.
    #[must_use]
    pub const fn head_dim(self) -> usize {
        D
    }

    /// BF16 elements per token in a query or attention output tensor.
    #[must_use]
    pub const fn query_width(self) -> usize {
        self.query_width
    }

    /// BF16 elements per token in a key or value tensor.
    #[must_use]
    pub const fn kv_width(self) -> usize {
        self.kv_width
    }

    /// Number of consecutive query heads sharing each KV head.
    #[must_use]
    pub const fn groups(self) -> usize {
        QH / KVH
    }

    /// BF16 elements in a dense query/output tensor, checking its byte extent.
    /// An empty token sequence needs zero elements.
    pub fn query_len(self, tokens: usize) -> Result<usize, AttentionError> {
        checked_len(tokens, self.query_width)
    }

    /// BF16 elements in one dense key/value tensor, checking its byte extent.
    /// An empty token sequence needs zero elements.
    pub fn kv_len(self, tokens: usize) -> Result<usize, AttentionError> {
        checked_len(tokens, self.kv_width)
    }
}

/// Checked query/key token counts and an optional absolute causal query offset.
///
/// A causal row at local query index `r` can attend to the key prefix ending at
/// absolute position `query_start + r`, inclusive. Noncausal rows see all keys.
/// Empty query sequences are permitted, but every nonempty query sequence must
/// have at least one key. A causal query range must fit entirely inside the key
/// sequence, including an empty range's start position.
///
/// This immutable metadata does not check tensor byte extents or establish that
/// key/value storage is initialized, owned or ready. Operators must validate
/// those properties separately, using initialized cache lengths rather than
/// allocation capacities as `key_tokens`.
///
/// ```
/// use cute_rs::attention::AttentionWindow;
/// let appended = AttentionWindow::causal(3, 19, 16).unwrap();
/// assert_eq!(appended.visible_keys(0).unwrap(), 17);
/// assert_eq!(appended.visible_keys(2).unwrap(), 19);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionWindow {
    query_tokens: usize,
    key_tokens: usize,
    causal_start: Option<usize>,
}

impl AttentionWindow {
    /// Validate a causal query range starting at an absolute key position.
    ///
    /// `query_start + query_tokens` must not overflow or exceed `key_tokens`.
    pub fn causal(
        query_tokens: usize,
        key_tokens: usize,
        query_start: usize,
    ) -> Result<Self, AttentionError> {
        let mut window = Self::noncausal(query_tokens, key_tokens)?;
        let query_end = query_start
            .checked_add(query_tokens)
            .ok_or(AttentionError::ExtentOverflow)?;
        if query_end > key_tokens {
            return Err(AttentionError::InvalidWindow);
        }
        window.causal_start = Some(query_start);
        Ok(window)
    }

    /// Validate a noncausal window in which every query sees all keys.
    /// Query and key sequence lengths may differ.
    pub fn noncausal(query_tokens: usize, key_tokens: usize) -> Result<Self, AttentionError> {
        if query_tokens != 0 && key_tokens == 0 {
            return Err(AttentionError::MissingKeys);
        }
        Ok(Self {
            query_tokens,
            key_tokens,
            causal_start: None,
        })
    }

    /// Number of query rows in the window.
    #[must_use]
    pub const fn query_tokens(self) -> usize {
        self.query_tokens
    }

    /// Number of available key/value token rows.
    #[must_use]
    pub const fn key_tokens(self) -> usize {
        self.key_tokens
    }

    /// Absolute key position of the first causal query, or `None` if noncausal.
    #[must_use]
    pub const fn causal_start(self) -> Option<usize> {
        self.causal_start
    }

    /// Count the visible key prefix for one local query row.
    ///
    /// Rejects `query_row >= query_tokens`, including every row of an empty
    /// query sequence. No causal row is fully masked by a validated window.
    pub fn visible_keys(self, query_row: usize) -> Result<usize, AttentionError> {
        if query_row >= self.query_tokens {
            return Err(AttentionError::InvalidQueryRow);
        }
        Ok(match self.causal_start {
            // The constructor checked start + query_tokens, and row is smaller.
            Some(start) => start + query_row + 1,
            None => self.key_tokens,
        })
    }
}

/// Validated capacity of each dense `[token, KV head, dimension]` cache tensor.
///
/// This is immutable metadata, not an initialized-cache handle. Head dimensions
/// remain part of its Rust type so checked callers can require compatible caches:
///
/// ```compile_fail,E0308
/// use cute_rs::attention::CacheShape;
/// fn needs_128_wide_heads(_: CacheShape<8, 128>) {}
/// let cache = CacheShape::<8, 64>::new(65).unwrap();
/// needs_128_wide_heads(cache);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheShape<const KVH: usize, const D: usize> {
    /// Maximum number of tokens per cache tensor.
    capacity: usize,
    /// BF16 elements per token.
    width: usize,
    /// BF16 elements in each complete key or value allocation.
    allocation_len: usize,
}

impl<const KVH: usize, const D: usize> CacheShape<KVH, D> {
    /// Check head dimensions and each cache tensor's full BF16 allocation extent.
    /// Zero token capacity is allowed; head dimensions must remain valid.
    pub fn new(capacity: usize) -> Result<Self, AttentionError> {
        let width = checked_width(KVH, D)?;
        Ok(Self {
            capacity,
            width,
            allocation_len: checked_len(capacity, width)?,
        })
    }

    /// Maximum number of tokens in each cache tensor.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }

    /// BF16 elements per token in each cache tensor.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// BF16 elements in each full allocation; K and V need this many separately.
    #[must_use]
    pub const fn allocation_len(self) -> usize {
        self.allocation_len
    }

    /// Validate an append and return its half-open **token** range.
    ///
    /// The caller supplies the number of initialized tokens. This checks its
    /// bounds, not initialization itself, and neither reserves storage nor
    /// advances a cursor. The owner must perform and complete the append before
    /// publishing the new cursor. An empty append at capacity is valid.
    pub fn append_range(
        self,
        valid_tokens: usize,
        new_tokens: usize,
    ) -> Result<Range<usize>, AttentionError> {
        if valid_tokens > self.capacity {
            return Err(AttentionError::InvalidCursor);
        }
        let end = valid_tokens
            .checked_add(new_tokens)
            .ok_or(AttentionError::ExtentOverflow)?;
        if end > self.capacity {
            return Err(AttentionError::CapacityExceeded);
        }
        Ok(valid_tokens..end)
    }
}
