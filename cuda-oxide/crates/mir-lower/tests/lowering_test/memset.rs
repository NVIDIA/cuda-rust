/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops::MirMemsetOp;
use dialect_mir::types::MirPtrType;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, make_test_ctx};

/// Lower `mir.memset(dst: *mut u<elem_bits>, val: u8, count: u64)` and return
/// the exported LLVM IR.
fn lower_memset(elem_bits: u32) -> anyhow::Result<String> {
    let mut ctx = make_test_ctx();
    let elem = IntegerType::get(&ctx, elem_bits, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, elem.into(), true);
    let val_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let count_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (module, entry) = build_test_kernel(
        &mut ctx,
        vec![dst_ty.into(), val_ty.into(), count_ty.into()],
    );
    let dst = entry.deref(&ctx).get_argument(0);
    let val = entry.deref(&ctx).get_argument(1);
    let count = entry.deref(&ctx).get_argument(2);
    MirMemsetOp::build(&mut ctx, dst, val, count)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
    pliron::operation::verify_operation(module, &ctx).map_err(|e| anyhow::anyhow!("{e}"))?;
    let module = Operation::get_op::<ModuleOp>(module, &ctx).unwrap();
    llvm_export::export::export_module_to_string(&ctx, &module).map_err(|e| anyhow::anyhow!("{e}"))
}

#[test]
fn memset_scales_the_element_count_to_bytes() -> anyhow::Result<()> {
    let ir = lower_memset(32)?;
    assert!(ir.contains("call void @llvm.memset.p0.i64(ptr"), "{ir}");
    assert!(
        ir.contains("mul i64"),
        "a u32 element count is scaled by 4:\n{ir}"
    );
    assert!(
        ir.contains(", 4"),
        "a u32 element count is scaled by 4:\n{ir}"
    );
    Ok(())
}

#[test]
fn memset_of_bytes_passes_the_count_through() -> anyhow::Result<()> {
    let ir = lower_memset(8)?;
    assert!(ir.contains("call void @llvm.memset.p0.i64(ptr"), "{ir}");
    assert!(!ir.contains("mul "), "a byte count needs no scaling:\n{ir}");
    Ok(())
}
