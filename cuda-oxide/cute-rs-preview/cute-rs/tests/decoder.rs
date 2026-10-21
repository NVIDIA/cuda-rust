/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decoder geometry and embedding indices can be checked without a GPU.
#![cfg(feature = "host")]

use cute_rs::attention::AttentionError;
use cute_rs::decoder::{DecoderError, DecoderShape, EmbeddingShape, TokenIds};
use cute_rs::ffn::{FfnError, RowMajorShape};

#[test]
fn talker_and_predictor_keep_hidden_query_and_kv_widths_distinct() {
    let talker = DecoderShape::<2048, 6144, 16, 8, 128>::new(3, 2051).unwrap();
    let predictor = DecoderShape::<1024, 3072, 16, 8, 128>::new(3, 1024).unwrap();
    assert_eq!(talker.input(), RowMajorShape::new(3, 2048, 2051).unwrap());
    assert_eq!(talker.hidden(), RowMajorShape::dense(3, 2048).unwrap());
    assert_eq!(predictor.hidden(), RowMajorShape::dense(3, 1024).unwrap());
    assert_eq!(talker.query(), predictor.query());
    assert_eq!(talker.key_value(), predictor.key_value());
    assert_eq!(predictor.query(), RowMajorShape::dense(3, 2048).unwrap());
    assert_eq!(
        predictor.key_value(),
        RowMajorShape::dense(3, 1024).unwrap()
    );
    assert_eq!(talker.query_norm(), RowMajorShape::dense(48, 128).unwrap());
    assert_eq!(talker.key_norm(), RowMajorShape::dense(24, 128).unwrap());
    assert_eq!(
        talker.intermediate(),
        RowMajorShape::dense(3, 6144).unwrap()
    );
    assert_eq!(
        predictor.intermediate(),
        RowMajorShape::dense(3, 3072).unwrap()
    );
}

#[test]
fn empty_tokens_preserve_geometry_but_require_no_activation_storage() {
    let shape = DecoderShape::<1024, 3072, 16, 8, 128>::new(0, 1025).unwrap();
    for matrix in [
        shape.input(),
        shape.hidden(),
        shape.query(),
        shape.key_value(),
        shape.query_norm(),
        shape.key_norm(),
        shape.intermediate(),
    ] {
        assert_eq!(matrix.required_len(), 0);
    }
    assert_eq!(
        (
            shape.tokens(),
            shape.query_heads(),
            shape.kv_heads(),
            shape.head_dim()
        ),
        (0, 16, 8, 128)
    );
}

#[test]
fn invalid_head_grouping_and_matrix_extents_fail_before_allocation() {
    assert_eq!(
        DecoderShape::<8, 16, 3, 2, 4>::new(1, 8),
        Err(DecoderError::Attention(AttentionError::InvalidHeadGrouping))
    );
    assert_eq!(
        DecoderShape::<8, 16, 0, 2, 4>::new(1, 8),
        Err(DecoderError::Attention(AttentionError::ZeroDimension))
    );
    assert_eq!(
        DecoderShape::<0, 16, 4, 2, 4>::new(0, 8),
        Err(DecoderError::Matrix(FfnError::ZeroWidth))
    );
    assert_eq!(
        DecoderShape::<8, 0, 4, 2, 4>::new(0, 8),
        Err(DecoderError::Matrix(FfnError::ZeroWidth))
    );
    assert_eq!(
        DecoderShape::<8, 16, 4, 2, 4>::new(1, 7),
        Err(DecoderError::Matrix(FfnError::StrideTooSmall))
    );
    assert!(DecoderShape::<1, 1, 4, 2, 4>::new(isize::MAX as usize / 2, 1).is_err());
    assert!(DecoderShape::<1, 1, { usize::MAX }, 1, 2>::new(0, 1).is_err());
}

#[test]
fn projection_widths_are_checked_against_head_geometry() {
    let shape = DecoderShape::<1024, 3072, 16, 8, 128>::new(1, 1024).unwrap();
    assert_eq!(shape.validate_projection_widths::<2048, 1024>(), Ok(()));
    assert_eq!(
        shape.validate_projection_widths::<1024, 1024>(),
        Err(DecoderError::ProjectionWidthMismatch)
    );
    assert_eq!(
        shape.validate_projection_widths::<2048, 512>(),
        Err(DecoderError::ProjectionWidthMismatch)
    );
}

#[test]
fn embedding_tables_keep_vocab_width_and_pitch_independent() {
    let shape = EmbeddingShape::<17, 2048>::new(2050).unwrap();
    assert_eq!(shape.table(), RowMajorShape::new(17, 2048, 2050).unwrap());
    assert_eq!(
        shape.output(3).unwrap(),
        RowMajorShape::dense(3, 2048).unwrap()
    );
    assert_eq!(shape.output(0).unwrap().required_len(), 0);
    assert_eq!(
        EmbeddingShape::<0, 2048>::new(2048),
        Err(DecoderError::EmptyVocabulary)
    );
    assert_eq!(
        EmbeddingShape::<17, 0>::new(0),
        Err(DecoderError::Matrix(FfnError::ZeroWidth))
    );
    assert_eq!(
        EmbeddingShape::<17, 2048>::new(2047),
        Err(DecoderError::Matrix(FfnError::StrideTooSmall))
    );
    assert!(EmbeddingShape::<{ usize::MAX }, 2>::new(2).is_err());
    assert!(shape.output(usize::MAX).is_err());
}

#[test]
fn token_ids_validate_the_exact_vocab_boundary_and_retain_the_borrow() {
    let ids = [0, 16, 3, 16];
    let checked = TokenIds::<17>::new(&ids).unwrap();
    assert_eq!(checked.as_slice(), &ids);
    assert_eq!(checked.len(), 4);
    assert!(!checked.is_empty());
    assert!(TokenIds::<17>::new(&[]).unwrap().is_empty());
    assert_eq!(
        TokenIds::<17>::new(&[0, 17, 16]),
        Err(DecoderError::InvalidToken {
            index: 1,
            token: 17,
            vocabulary: 17
        })
    );
    assert_eq!(
        TokenIds::<17>::new(&[u32::MAX]),
        Err(DecoderError::InvalidToken {
            index: 0,
            token: u32::MAX,
            vocabulary: 17
        })
    );
    assert_eq!(TokenIds::<0>::new(&[]), Err(DecoderError::EmptyVocabulary));
}
