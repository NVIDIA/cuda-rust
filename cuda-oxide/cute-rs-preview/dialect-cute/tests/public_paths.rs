/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public operation families keep their compatibility imports as identical types.

use std::any::TypeId;

use dialect_cute::{
    copy_ops, epilogue_ops, gemm_tma_ops, gemv_ops, ops, pipeline_ops, scheduler_ops,
    sm100_attention_ops, sm100_ops, smem_mma_ops, tensor_ops,
};

#[test]
fn operation_family_aliases_preserve_type_identity() {
    macro_rules! same_type {
        ($old:path, $grouped:path) => {
            assert_eq!(TypeId::of::<$old>(), TypeId::of::<$grouped>());
        };
    }

    same_type!(
        copy_ops::CuteStoreGlobalNoAllocateOp,
        ops::copy::CuteStoreGlobalNoAllocateOp
    );
    same_type!(tensor_ops::CuteTensorMakeOp, ops::tensor::CuteTensorMakeOp);
    same_type!(gemv_ops::CuteDotOp, ops::gemv::CuteDotOp);
    same_type!(
        gemm_tma_ops::CuteTmaCopy2dOp,
        ops::gemm_tma::CuteTmaCopy2dOp
    );
    same_type!(smem_mma_ops::CuteTiledGemmOp, ops::mma::CuteTiledGemmOp);
    same_type!(
        epilogue_ops::CuteEpilogueHalfOp,
        ops::epilogue::CuteEpilogueHalfOp
    );
    same_type!(
        pipeline_ops::CutePipelineStateNewOp,
        ops::pipeline::CutePipelineStateNewOp
    );
    same_type!(
        scheduler_ops::CuteSchedulerNew1dOp,
        ops::scheduler::CuteSchedulerNew1dOp
    );
    same_type!(
        sm100_ops::CuteSm100TiledMmaOp,
        ops::sm100::CuteSm100TiledMmaOp
    );
    same_type!(
        sm100_attention_ops::CuteSm100MmaSsOp,
        ops::sm100_attention::CuteSm100MmaSsOp
    );
    same_type!(ops::CuteCopyOp, ops::copy::CuteCopyOp);
    same_type!(ops::CuteCopyG2SOp, ops::copy::CuteCopyG2SOp);
    same_type!(ops::CuteLdmatrixOp, ops::mma::CuteLdmatrixOp);
}
