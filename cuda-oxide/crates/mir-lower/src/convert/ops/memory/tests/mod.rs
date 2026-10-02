/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end lowering tests for `dialect-mir` memory ops.
//!
//! The `convert_*` functions in this file take a live
//! `DialectConversionRewriter`, which is owned by pliron's conversion
//! driver and not constructible standalone. So each test builds a small
//! MIR module, runs the full `lower_mir_to_llvm` pass on it, and asserts
//! the lowered module contains the expected `dialect-llvm` shape — same
//! pattern as the `tests/lowering_test/` integration suite.

// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#![allow(clippy::disallowed_methods)]

mod access;
mod alignment;
mod debug_provenance;
mod device_global;
mod enums;
mod shared;

use crate::context::DeviceGlobalDeclaration;
use crate::convert::ops::test_util::*;
use crate::convert::types::convert_type;
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops as mir;
use dialect_mir::types::{
    MirArrayType, MirPointerKind, MirPtrType, MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::op_interfaces::PointerTypeResult;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::{
    ArrayType, PointerType, StructLayout, StructType, address_space as llvm_addr,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{StringAttr, TypeAttr};
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location, Source};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use std::path::PathBuf;

use super::device_global::relocated_initializer_storage_type;

fn ptr_addrspace(ctx: &Context, ty: TypeHandle) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<PointerType>()
        .expect("expected llvm.PointerType")
        .address_space()
}

fn src_location(ctx: &mut Context, file: &str, line: i32, column: i32) -> Location {
    Location::SrcPos {
        src: Source::new_from_file(ctx, PathBuf::from(file)),
        pos: combine::stream::position::SourcePosition { line, column },
    }
}

fn over_aligned_tuple_ty(ctx: &mut Context) -> TypeHandle {
    let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let marker: TypeHandle = MirStructType::get_with_full_layout(
        ctx,
        "Align32".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        32,
    )
    .into();
    MirTupleType::get_with_layout(ctx, vec![marker, byte], vec![0, 1], vec![0, 0], 32, 32).into()
}
