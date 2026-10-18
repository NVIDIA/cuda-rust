/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unit tests for shared GEMM mapping invariants.

use crate::cute::{
    epilogue::sm120::{
        epilogue_gep_element_type, epilogue_sync_operations, f16x2_intrinsic_operands,
    },
    mma::sm120::ldmatrix_swizzled_byte_offset,
    pipeline::sm120::mbarrier_init,
};
use crate::{MlirAttribute, MlirLocation, MlirType, MlirValueUse};
use dialect_cute::attributes::CuteEpilogueSyncPhaseAttr;

#[cfg(test)]
mod tests {
    //! Focused checks for GEMM byte offsets, packed lanes, and synchronization ordering.

    use super::*;
    use crate::{MlirFloatType, MlirValueId};

    /// Construct a target SSA operand with a chosen identifier and type for recipe tests.
    fn value(id: u64, ty: MlirType) -> MlirValueUse {
        MlirValueUse {
            id: MlirValueId(id),
            ty,
        }
    }

    /// Check the byte-domain permutation preserves each contiguous 16-byte matrix-load segment.
    #[test]
    fn byte_domain_ldmatrix_swizzle_preserves_sixteen_byte_segments() {
        for warp_row in 0..8_u64 {
            for warp_column in 0..2_u64 {
                for lane in 0..32_u64 {
                    let submatrix = lane / 8;
                    let row = warp_row * 16 + (submatrix % 2) * 8 + lane % 8;
                    let column = warp_column * 16 + (submatrix / 2) * 8;
                    let plain = row * 64 + column * 2;
                    assert_eq!(
                        ldmatrix_swizzled_byte_offset(plain) % 16,
                        0,
                        "warp_row={warp_row} warp_column={warp_column} lane={lane}"
                    );
                }
            }
        }

        // The old element-domain mask toggled bit 3 for lane 1 after the
        // layout was converted to bytes, reproducing the observed 0x6448
        // misaligned shared address.
        let lane_one_plain = 64_u64;
        let old = lane_one_plain ^ ((lane_one_plain & 0xc0) >> 3);
        assert_eq!(old, 72);
        assert_eq!(old % 16, 8);
        assert_eq!(ldmatrix_swizzled_byte_offset(lane_one_plain), 64);
    }

    /// Check low-first source lanes are reversed for the high-first native conversion intrinsic.
    #[test]
    fn packed_f16_intrinsic_reverses_low_first_source_operands() {
        let low = value(7, MlirType::Float(MlirFloatType::F32));
        let high = value(8, MlirType::Float(MlirFloatType::F32));
        let [first, second] = f16x2_intrinsic_operands(low.clone(), high.clone());
        assert_eq!(first, high);
        assert_eq!(second, low);
    }

    /// Check ReadyForTma emits proxy publication before counted sync and Reusable emits only counted sync.
    #[test]
    fn epilogue_ready_publishes_then_syncs_while_reusable_only_syncs() {
        let barrier_id = value(1, MlirType::Integer(32));
        let participants = value(2, MlirType::Integer(32));
        let ready = epilogue_sync_operations(
            CuteEpilogueSyncPhaseAttr::ReadyForTma,
            barrier_id.clone(),
            participants.clone(),
            &MlirLocation::Unknown,
        )
        .unwrap();
        assert_eq!(
            ready
                .iter()
                .map(|operation| operation.name.as_str())
                .collect::<Vec<_>>(),
            ["nvvm.fence.proxy", "nvvm.barrier.cta.sync"]
        );
        assert_eq!(
            ready[0].attributes.get("kind"),
            Some(&MlirAttribute::dialect("#nvvm.proxy_kind<async.shared>").unwrap())
        );
        assert_eq!(
            ready[0].attributes.get("space"),
            Some(&MlirAttribute::dialect("#nvvm.shared_space<cta>").unwrap())
        );
        assert_eq!(
            ready[1].operands,
            vec![barrier_id.clone(), participants.clone()]
        );

        let reusable = epilogue_sync_operations(
            CuteEpilogueSyncPhaseAttr::Reusable,
            barrier_id,
            participants,
            &MlirLocation::Unknown,
        )
        .unwrap();
        assert_eq!(reusable.len(), 1);
        assert_eq!(reusable[0].name, "nvvm.barrier.cta.sync");
        assert_eq!(
            epilogue_gep_element_type(),
            MlirType::Float(MlirFloatType::F16)
        );
    }

    /// Check barrier initialization remains unpredicated once source control flow has selected the initializer.
    #[test]
    fn pipeline_initialization_is_unpredicated_inside_the_source_branch() {
        let barrier = value(3, MlirType::dialect("!llvm.ptr<3>").unwrap());
        let count = value(4, MlirType::Integer(32));
        let init = mbarrier_init(barrier.clone(), count.clone(), &MlirLocation::Unknown).unwrap();
        assert_eq!(init.name, "nvvm.mbarrier.init");
        assert_eq!(init.operands, vec![barrier, count]);
    }
}
