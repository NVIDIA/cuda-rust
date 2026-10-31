/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked BF16 decoder arguments, embedding indices and normalization roles.
//!
//! Hidden width and attention width are independent: a decoder with hidden width
//! 1024 can produce 16 query heads of width 128. Projection roles, normalization
//! roles, BF16 storage and fixed widths stay in Rust types across composition.
//! Matrices retain checkpoint orientation `[output, input]`.
//!
//! These descriptors borrow existing allocations and do not launch work. Shapes
//! cannot prove checkpoint contents, device context or stream readiness. Operators
//! must check those properties and retain input borrows until launches complete.

use core::{fmt, marker::PhantomData};
use cuda_core::DeviceBuffer;

use crate::attention::metadata::{AttentionError, AttentionShape};
use crate::ffn::{
    AttentionOutput, FfnError, FfnShape, FfnWeights, Key, ProjectionRole, ProjectionWeights, Query,
    RowMajorShape, Value,
};
use crate::numeric::BFloat16;

/// A rejected decoder shape, vocabulary index or projection width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderError {
    /// A BF16 matrix or allocation extent is invalid.
    Matrix(FfnError),
    /// Attention heads or their element extents are invalid.
    Attention(AttentionError),
    /// Typed projection widths do not equal their head counts times head width.
    ProjectionWidthMismatch,
    /// An embedding vocabulary must contain at least one row.
    EmptyVocabulary,
    /// A token index is outside the embedding vocabulary.
    InvalidToken {
        /// Position of the invalid token in the supplied host sequence.
        index: usize,
        /// Supplied token index.
        token: u32,
        /// Exclusive vocabulary bound.
        vocabulary: usize,
    },
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(f),
            Self::Attention(error) => error.fmt(f),
            Self::ProjectionWidthMismatch => {
                f.write_str("projection widths do not match attention head geometry")
            }
            Self::EmptyVocabulary => f.write_str("embedding vocabulary must be nonempty"),
            Self::InvalidToken {
                index,
                token,
                vocabulary,
            } => write!(
                f,
                "token {token} at index {index} exceeds vocabulary bound {vocabulary}"
            ),
        }
    }
}

impl core::error::Error for DecoderError {}

impl From<FfnError> for DecoderError {
    fn from(error: FfnError) -> Self {
        Self::Matrix(error)
    }
}

impl From<AttentionError> for DecoderError {
    fn from(error: AttentionError) -> Self {
        Self::Attention(error)
    }
}

/// Checked activation geometry for a decoder layer with fixed model dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderShape<
    const H: usize,
    const I: usize,
    const QH: usize,
    const KVH: usize,
    const D: usize,
> {
    /// Hidden and intermediate activation shapes.
    ffn: FfnShape<H, I>,
    /// Independent query and key/value head geometry.
    attention: AttentionShape<QH, KVH, D>,
    /// Dense token-major query projection.
    query: RowMajorShape,
    /// Dense token-major key or value projection.
    key_value: RowMajorShape,
    /// Query heads flattened into independent normalization rows.
    query_norm: RowMajorShape,
    /// Key heads flattened into independent normalization rows.
    key_norm: RowMajorShape,
}

impl<const H: usize, const I: usize, const QH: usize, const KVH: usize, const D: usize>
    DecoderShape<H, I, QH, KVH, D>
{
    /// Check all activation extents before allocation; input rows may be padded.
    pub fn new(tokens: usize, input_stride: usize) -> Result<Self, DecoderError> {
        let attention = AttentionShape::new()?;
        let ffn = FfnShape::new(tokens, input_stride)?;
        let query = RowMajorShape::dense(tokens, attention.query_width())?;
        let key_value = RowMajorShape::dense(tokens, attention.kv_width())?;
        let query_rows = tokens.checked_mul(QH).ok_or(FfnError::ExtentOverflow)?;
        let key_rows = tokens.checked_mul(KVH).ok_or(FfnError::ExtentOverflow)?;
        Ok(Self {
            ffn,
            attention,
            query,
            key_value,
            query_norm: RowMajorShape::dense(query_rows, D)?,
            key_norm: RowMajorShape::dense(key_rows, D)?,
        })
    }

    /// Number of input tokens in the invocation.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.ffn.rows()
    }
    /// Checked input matrix, including row padding.
    #[must_use]
    pub const fn input(self) -> RowMajorShape {
        self.ffn.input()
    }
    /// Dense hidden-width activation shape.
    #[must_use]
    pub const fn hidden(self) -> RowMajorShape {
        self.ffn.hidden()
    }
    /// Dense intermediate-width FFN shape.
    #[must_use]
    pub const fn intermediate(self) -> RowMajorShape {
        self.ffn.intermediate()
    }
    /// Dense `[token, query_head * head_dim]` projection shape.
    #[must_use]
    pub const fn query(self) -> RowMajorShape {
        self.query
    }
    /// Dense `[token, kv_head * head_dim]` key or value shape.
    #[must_use]
    pub const fn key_value(self) -> RowMajorShape {
        self.key_value
    }
    /// Dense `[token * query_head, head_dim]` per-head normalization view.
    #[must_use]
    pub const fn query_norm(self) -> RowMajorShape {
        self.query_norm
    }
    /// Dense `[token * kv_head, head_dim]` per-head normalization view.
    #[must_use]
    pub const fn key_norm(self) -> RowMajorShape {
        self.key_norm
    }
    /// Query head count.
    #[must_use]
    pub const fn query_heads(self) -> usize {
        QH
    }
    /// Key/value head count.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        KVH
    }
    /// Coordinates per attention head.
    #[must_use]
    pub const fn head_dim(self) -> usize {
        D
    }
    /// FFN geometry for the same token rows and input pitch.
    #[must_use]
    pub const fn ffn(self) -> FfnShape<H, I> {
        self.ffn
    }

    /// Check the explicit const widths used by typed projection weights.
    ///
    /// Stable const generics cannot express products such as `QH * D` in field
    /// types, so this relation is checked once at the composition boundary.
    pub fn validate_projection_widths<const QWIDTH: usize, const KVWIDTH: usize>(
        self,
    ) -> Result<(), DecoderError> {
        if QWIDTH != self.attention.query_width() || KVWIDTH != self.attention.kv_width() {
            Err(DecoderError::ProjectionWidthMismatch)
        } else {
            Ok(())
        }
    }

    /// Validate head geometry against role- and width-checked decoder weights.
    pub fn validate_weights<const QWIDTH: usize, const KVWIDTH: usize>(
        self,
        _weights: &DecoderWeights<'_, H, I, QWIDTH, KVWIDTH, D>,
    ) -> Result<(), DecoderError> {
        self.validate_projection_widths::<QWIDTH, KVWIDTH>()
    }
}

/// Private supertrait restricting normalization roles.
mod sealed {
    pub trait Sealed {}
}

/// A sealed semantic role for one RMS normalization weight vector.
pub trait NormRole: sealed::Sealed {}

/// Hidden-state normalization before attention projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputNorm;
/// Hidden-state normalization before the FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostAttentionNorm;
/// Per-query-head normalization before RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryNorm;
/// Per-key-head normalization before RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyNorm;
/// Hidden-state normalization after the decoder stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalNorm;

impl sealed::Sealed for InputNorm {}
impl sealed::Sealed for PostAttentionNorm {}
impl sealed::Sealed for QueryNorm {}
impl sealed::Sealed for KeyNorm {}
impl sealed::Sealed for FinalNorm {}
impl NormRole for InputNorm {}
impl NormRole for PostAttentionNorm {}
impl NormRole for QueryNorm {}
impl NormRole for KeyNorm {}
impl NormRole for FinalNorm {}

/// Borrowed BF16 RMS normalization weights, distinguished by role and width.
///
/// A query norm cannot be substituted for the equally sized key norm:
/// ```compile_fail,E0308
/// use cute_rs::decoder::{KeyNorm, NormWeights, QueryNorm};
/// fn wrong_role(query: NormWeights<'_, QueryNorm, 128>) {
///     let _: NormWeights<'_, KeyNorm, 128> = query;
/// }
/// ```
/// The immutable borrow lasts through the checked view's final use:
/// ```compile_fail,E0502
/// use cuda_core::DeviceBuffer;
/// use cute_rs::{BFloat16, decoder::{InputNorm, NormWeights}};
/// fn mutate(_: &mut DeviceBuffer<BFloat16>) {}
/// fn alias(buffer: &mut DeviceBuffer<BFloat16>) {
///     let weights = NormWeights::<InputNorm, 1024>::new(buffer).unwrap();
///     mutate(buffer);
///     let _ = weights.buffer();
/// }
/// ```
pub struct NormWeights<'a, Role: NormRole, const WIDTH: usize> {
    buffer: &'a DeviceBuffer<BFloat16>,
    role: PhantomData<Role>,
}

impl<'a, Role: NormRole, const WIDTH: usize> NormWeights<'a, Role, WIDTH> {
    /// Check a positive logical width and BF16 allocation extent, allowing guards.
    pub fn new(buffer: &'a DeviceBuffer<BFloat16>) -> Result<Self, FfnError> {
        RowMajorShape::dense(1, WIDTH)?.validate_len(buffer.len())?;
        Ok(Self {
            buffer,
            role: PhantomData,
        })
    }
    /// Borrow the existing allocation for a checked kernel launch.
    #[must_use]
    pub const fn buffer(&self) -> &'a DeviceBuffer<BFloat16> {
        self.buffer
    }
    /// Logical number of normalization weights, excluding trailing padding.
    #[must_use]
    pub const fn width(&self) -> usize {
        WIDTH
    }
}

/// Borrowed BF16 bias for a projection with a specific role and output width.
pub struct BiasWeights<'a, Role: ProjectionRole, const WIDTH: usize> {
    buffer: &'a DeviceBuffer<BFloat16>,
    role: PhantomData<Role>,
}

impl<'a, Role: ProjectionRole, const WIDTH: usize> BiasWeights<'a, Role, WIDTH> {
    /// Check a positive logical width and BF16 allocation extent, allowing guards.
    pub fn new(buffer: &'a DeviceBuffer<BFloat16>) -> Result<Self, FfnError> {
        RowMajorShape::dense(1, WIDTH)?.validate_len(buffer.len())?;
        Ok(Self {
            buffer,
            role: PhantomData,
        })
    }
    /// Borrow the existing allocation for a checked kernel launch.
    #[must_use]
    pub const fn buffer(&self) -> &'a DeviceBuffer<BFloat16> {
        self.buffer
    }
    /// Logical number of bias elements, excluding trailing padding.
    #[must_use]
    pub const fn width(&self) -> usize {
        WIDTH
    }
}

/// Role-checked projections and per-head normalization weights for attention.
///
/// Query and key weights remain distinct even for multi-head attention where
/// their dimensions match:
/// ```compile_fail,E0308
/// use cute_rs::ffn::{Key, ProjectionWeights, Query};
/// fn wrong_role(query: ProjectionWeights<'_, Query, 8, 16>) {
///     let _: ProjectionWeights<'_, Key, 8, 16> = query;
/// }
/// ```
pub struct AttentionWeights<
    'a,
    const H: usize,
    const QWIDTH: usize,
    const KVWIDTH: usize,
    const D: usize,
> {
    query: ProjectionWeights<'a, Query, H, QWIDTH>,
    key: ProjectionWeights<'a, Key, H, KVWIDTH>,
    value: ProjectionWeights<'a, Value, H, KVWIDTH>,
    output: ProjectionWeights<'a, AttentionOutput, QWIDTH, H>,
    query_norm: NormWeights<'a, QueryNorm, D>,
    key_norm: NormWeights<'a, KeyNorm, D>,
}

impl<'a, const H: usize, const QWIDTH: usize, const KVWIDTH: usize, const D: usize>
    AttentionWeights<'a, H, QWIDTH, KVWIDTH, D>
{
    /// Compose already checked weights; head-count consistency is checked by
    /// [`DecoderShape::validate_weights`] before launch.
    #[must_use]
    pub const fn new(
        query: ProjectionWeights<'a, Query, H, QWIDTH>,
        key: ProjectionWeights<'a, Key, H, KVWIDTH>,
        value: ProjectionWeights<'a, Value, H, KVWIDTH>,
        output: ProjectionWeights<'a, AttentionOutput, QWIDTH, H>,
        query_norm: NormWeights<'a, QueryNorm, D>,
        key_norm: NormWeights<'a, KeyNorm, D>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
        }
    }
    /// Query projection weights.
    #[must_use]
    pub const fn query(&self) -> &ProjectionWeights<'a, Query, H, QWIDTH> {
        &self.query
    }
    /// Key projection weights.
    #[must_use]
    pub const fn key(&self) -> &ProjectionWeights<'a, Key, H, KVWIDTH> {
        &self.key
    }
    /// Value projection weights.
    #[must_use]
    pub const fn value(&self) -> &ProjectionWeights<'a, Value, H, KVWIDTH> {
        &self.value
    }
    /// Projection from query-head output width back to hidden width.
    #[must_use]
    pub const fn output(&self) -> &ProjectionWeights<'a, AttentionOutput, QWIDTH, H> {
        &self.output
    }
    /// Shared per-query-head RMS weights.
    #[must_use]
    pub const fn query_norm(&self) -> &NormWeights<'a, QueryNorm, D> {
        &self.query_norm
    }
    /// Shared per-key-head RMS weights.
    #[must_use]
    pub const fn key_norm(&self) -> &NormWeights<'a, KeyNorm, D> {
        &self.key_norm
    }
}

/// Borrowed weights for one pre-norm decoder layer.
///
/// Predictor hidden width 1024 cannot stand in for Talker hidden width 2048:
/// ```compile_fail,E0308
/// use cute_rs::decoder::{DecoderShape, DecoderWeights};
/// fn wrong_hidden(weights: &DecoderWeights<'_, 1024, 3072, 2048, 1024, 128>) {
///     let talker = DecoderShape::<2048, 6144, 16, 8, 128>::new(1, 2048).unwrap();
///     talker.validate_weights(weights).unwrap();
/// }
/// ```
pub struct DecoderWeights<
    'a,
    const H: usize,
    const I: usize,
    const QWIDTH: usize,
    const KVWIDTH: usize,
    const D: usize,
> {
    attention: AttentionWeights<'a, H, QWIDTH, KVWIDTH, D>,
    input_norm: NormWeights<'a, InputNorm, H>,
    post_attention_norm: NormWeights<'a, PostAttentionNorm, H>,
    ffn: FfnWeights<'a, H, I>,
}

impl<'a, const H: usize, const I: usize, const QWIDTH: usize, const KVWIDTH: usize, const D: usize>
    DecoderWeights<'a, H, I, QWIDTH, KVWIDTH, D>
{
    /// Compose role- and width-checked layer arguments without allocation.
    #[must_use]
    pub const fn new(
        attention: AttentionWeights<'a, H, QWIDTH, KVWIDTH, D>,
        input_norm: NormWeights<'a, InputNorm, H>,
        post_attention_norm: NormWeights<'a, PostAttentionNorm, H>,
        ffn: FfnWeights<'a, H, I>,
    ) -> Self {
        Self {
            attention,
            input_norm,
            post_attention_norm,
            ffn,
        }
    }
    /// Attention projections and per-head normalization weights.
    #[must_use]
    pub const fn attention(&self) -> &AttentionWeights<'a, H, QWIDTH, KVWIDTH, D> {
        &self.attention
    }
    /// Hidden-state normalization before attention.
    #[must_use]
    pub const fn input_norm(&self) -> &NormWeights<'a, InputNorm, H> {
        &self.input_norm
    }
    /// Hidden-state normalization before the FFN.
    #[must_use]
    pub const fn post_attention_norm(&self) -> &NormWeights<'a, PostAttentionNorm, H> {
        &self.post_attention_norm
    }
    /// Gate/up/down FFN projections.
    #[must_use]
    pub const fn ffn(&self) -> &FfnWeights<'a, H, I> {
        &self.ffn
    }
}

/// Checked vocabulary-table geometry independent of the token sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingShape<const VOCAB: usize, const WIDTH: usize> {
    table: RowMajorShape,
}

impl<const VOCAB: usize, const WIDTH: usize> EmbeddingShape<VOCAB, WIDTH> {
    /// Check a nonempty `[vocabulary, embedding_width]` table and its row pitch.
    pub fn new(row_stride: usize) -> Result<Self, DecoderError> {
        if VOCAB == 0 {
            return Err(DecoderError::EmptyVocabulary);
        }
        Ok(Self {
            table: RowMajorShape::new(VOCAB, WIDTH, row_stride)?,
        })
    }
    /// Logical table shape with checkpoint row orientation.
    #[must_use]
    pub const fn table(self) -> RowMajorShape {
        self.table
    }
    /// Dense embedding output for a checked number of tokens.
    pub fn output(self, tokens: usize) -> Result<RowMajorShape, DecoderError> {
        Ok(RowMajorShape::dense(tokens, WIDTH)?)
    }
}

/// Borrowed BF16 embedding table with checked vocabulary and embedding width.
pub struct EmbeddingWeights<'a, const VOCAB: usize, const WIDTH: usize> {
    buffer: &'a DeviceBuffer<BFloat16>,
    shape: EmbeddingShape<VOCAB, WIDTH>,
}

impl<'a, const VOCAB: usize, const WIDTH: usize> EmbeddingWeights<'a, VOCAB, WIDTH> {
    /// Borrow a `[vocabulary, width]` table, permitting trailing allocation guards.
    pub fn new(
        buffer: &'a DeviceBuffer<BFloat16>,
        row_stride: usize,
    ) -> Result<Self, DecoderError> {
        let shape = EmbeddingShape::new(row_stride)?;
        shape.table().validate_len(buffer.len())?;
        Ok(Self { buffer, shape })
    }
    /// Existing table allocation.
    #[must_use]
    pub const fn buffer(&self) -> &'a DeviceBuffer<BFloat16> {
        self.buffer
    }
    /// Checked table geometry.
    #[must_use]
    pub const fn shape(&self) -> EmbeddingShape<VOCAB, WIDTH> {
        self.shape
    }
    /// Dense output extent for IDs checked against this exact vocabulary bound.
    pub fn output_shape(&self, ids: TokenIds<'_, VOCAB>) -> Result<RowMajorShape, DecoderError> {
        self.shape.output(ids.len())
    }
}

/// Immutably borrowed host token IDs checked against an exclusive vocab bound.
///
/// These IDs are ready for an operator to copy to device memory. This type does
/// not certify an unrelated device ID buffer or a subsequent mutation to one.
///
/// The host indices cannot change while the checked view is in use:
/// ```compile_fail,E0506
/// use cute_rs::decoder::TokenIds;
/// let mut ids = [0, 1];
/// let checked = TokenIds::<2>::new(&ids).unwrap();
/// ids[0] = 99;
/// let _ = checked.as_slice();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenIds<'a, const VOCAB: usize> {
    ids: &'a [u32],
}

impl<'a, const VOCAB: usize> TokenIds<'a, VOCAB> {
    /// Reject an empty vocabulary or the first token outside `[0, VOCAB)`.
    pub fn new(ids: &'a [u32]) -> Result<Self, DecoderError> {
        if VOCAB == 0 {
            return Err(DecoderError::EmptyVocabulary);
        }
        for (index, &token) in ids.iter().enumerate() {
            if token as usize >= VOCAB {
                return Err(DecoderError::InvalidToken {
                    index,
                    token,
                    vocabulary: VOCAB,
                });
            }
        }
        Ok(Self { ids })
    }
    /// Checked host token indices.
    #[must_use]
    pub const fn as_slice(self) -> &'a [u32] {
        self.ids
    }
    /// Number of token indices.
    #[must_use]
    pub const fn len(self) -> usize {
        self.ids.len()
    }
    /// Whether the token sequence requires no embedding lookup.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.ids.is_empty()
    }
}
