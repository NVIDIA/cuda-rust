// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![doc = "Checks const-indexed accumulator fragments and static instruction expansion order."]
use ctm_rs::mma::{self, Accumulator16x8, AccumulatorTile16x8};

/// Const-indexed accumulator access must preserve fragment bands, columns and register order.
#[test]
fn static_fragment_access_preserves_bands_columns_and_register_order() {
    let mut tile = AccumulatorTile16x8::<2, 8>::zero();
    mma::static_for!(ROW in [0, 1] {
        mma::static_for!(COL in [0, 1, 2, 3, 4, 5, 6, 7] {
            assert_eq!(tile.fragment::<ROW, COL>().registers(), [0.0; 4]);
            let tag = (ROW * 100 + COL * 10) as f32;
            *tile.fragment_mut::<ROW, COL>() =
                Accumulator16x8::from_registers([tag, tag + 1.0, tag + 2.0, tag + 3.0]);
        });
    });
    // Asymmetric tags detect a row/column swap or an aliased fragment update.
    assert_eq!(tile.fragment::<0, 0>().registers(), [0., 1., 2., 3.]);
    assert_eq!(tile.fragment::<0, 7>().registers(), [70., 71., 72., 73.]);
    assert_eq!(
        tile.fragment::<1, 0>().registers(),
        [100., 101., 102., 103.]
    );
    assert_eq!(
        tile.fragment::<1, 7>().registers(),
        [170., 171., 172., 173.]
    );
}

/// Nested static expansion must visit the requested order with usable const-generic indices.
#[test]
fn static_expansion_nests_in_source_order_with_const_generic_indices() {
    /// Return the statically supplied row and column so expansion order can be observed.
    fn coordinate<const ROW: usize, const COL: usize>() -> (usize, usize) {
        (ROW, COL)
    }
    let mut visited = Vec::new();
    mma::static_for!(ROW in [0, 1] {
        mma::static_for!(COL in [0, 1] {
            visited.push(coordinate::<ROW, COL>());
        });
    });
    assert_eq!(visited, [(0, 0), (0, 1), (1, 0), (1, 1)]);
}

/// Static expansion must preserve an explicitly supplied nonconsecutive index list.
#[test]
fn static_expansion_preserves_the_requested_index_list() {
    let mut visited = Vec::new();
    mma::static_for!(INDEX in [3, 1, 0, 1] {
        visited.push(INDEX);
    });
    assert_eq!(visited, [3, 1, 0, 1]);
}
