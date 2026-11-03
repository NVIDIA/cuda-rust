//! Regression tests for CuTe cache-aware global-store width and pointer contracts.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dialect_cute::ops::copy::CuteStoreGlobalNoAllocateOp;
use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    r#type::TypeHandle,
};

/// Cache-aware stores require supported vector widths, matching storage, and a writable global destination.
#[test]
fn global_store_checks_width_storage_and_address_roles() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let word: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let half: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let mut ptr = |elem, mutable, space| -> TypeHandle {
        MirPtrType::get(&mut ctx, elem, mutable, space).into()
    };
    let types = vec![
        ptr(word, true, address_space::GLOBAL),
        ptr(word, false, address_space::LOCAL),
        ptr(word, false, address_space::GLOBAL),
        ptr(word, true, address_space::SHARED),
        ptr(half, false, address_space::LOCAL),
    ];
    let block = BasicBlock::new(&mut ctx, None, types);
    let a: Vec<_> = block.deref(&ctx).arguments().collect();
    for n in [1, 2, 4, 8] {
        assert!(
            CuteStoreGlobalNoAllocateOp::new(&mut ctx, a[0], a[1], n)
                .verify(&ctx)
                .is_ok()
        );
    }
    for n in [0, 3, 16, u64::MAX] {
        assert!(
            CuteStoreGlobalNoAllocateOp::new(&mut ctx, a[0], a[1], n)
                .verify(&ctx)
                .is_err()
        );
    }
    for (dst, src) in [(a[2], a[1]), (a[3], a[1]), (a[0], a[2]), (a[0], a[4])] {
        assert!(
            CuteStoreGlobalNoAllocateOp::new(&mut ctx, dst, src, 8)
                .verify(&ctx)
                .is_err()
        );
    }
}
