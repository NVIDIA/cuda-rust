/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect operations.
//!
//! This module defines and registers all MIR dialect operations by re-exporting them
//! from their respective sub-modules. MIR (Mid-level IR) operations represent Rust's
//! intermediate representation in Pliron IR form, preserving Rust-specific semantics
//! before lowering to LLVM.
//!
//! # Module Organization
//!
//! ```text
//! ┌──────────────┬──────────────────────────────────────┬─────┐
//! │ Module       │ Description                          │ Ops │
//! ├──────────────┼──────────────────────────────────────┼─────┤
//! │ function     │ Function definition                  │ 1   │
//! │ control_flow │ Terminators and branches             │ 5   │
//! │ memory       │ Memory access and allocation         │ 6   │
//! │ constants    │ Integer and float constants          │ 2   │
//! │ arithmetic   │ Math, bitwise, and shift operations  │ 13  │
//! │ comparison   │ Relational and equality comparisons  │ 6   │
//! │ aggregate    │ Struct and tuple manipulation        │ 4   │
//! │ enum_ops     │ Enum construction and inspection     │ 4   │
//! │ cast         │ Type conversions                     │ 1   │
//! │ storage      │ Variable lifetime markers            │ 2   │
//! │ call         │ Function calls                       │ 1   │
//! └──────────────┴──────────────────────────────────────┴─────┘
//! ```
//!
//! # Verification Strategy
//!
//! MIR dialect operations use **type consistency verification** that focuses on
//! ensuring operand/result types are compatible and Rust-specific invariants hold.
//!
//! ## Verification by Category
//!
//! ```text
//! ┌──────────────┬─────┬────────────────────────────────────────────────────────┐
//! │ Category     │ Ops │ Verification                                           │
//! ├──────────────┼─────┼────────────────────────────────────────────────────────┤
//! │ Function     │  1  │ ✅ Full: entry block args match function type inputs   │
//! │ Control Flow │  5  │ ✅ Full: condition is i1, successor args match block   │
//! │              │     │    params, operand counts verified                     │
//! │ Memory       │  6  │ ✅ Full: pointer types, pointee types, address spaces  │
//! │ Arithmetic   │ 13  │ ✅ Good: operand types match, result type matches      │
//! │ Comparison   │  6  │ ✅ Good: operand types match, result type matches      │
//! │ Aggregate    │  4  │ ✅ Good: struct/tuple type checking, index bounds      │
//! │ Enum         │  4  │ ✅ Good: discriminant type, payload type validation    │
//! │ Cast         │  1  │ ✅ Full: cast_kind required, operand/result types checked per kind │
//! │ Constants    │  2  │ ✅ Good: type attributes present and valid             │
//! │ Storage      │  2  │ ✅ Basic: structural verification                      │
//! │ Call         │  1  │ ✅ Good: callee attribute, argument count              │
//! └──────────────┴─────┴────────────────────────────────────────────────────────┘
//! ```
//!
//! ## What IS Verified
//!
//! MIR operations verify:
//!
//! - **Operand count**: Each operation has the correct number of operands.
//! - **Result count**: Each operation produces the expected number of results.
//! - **Type consistency**: Binary ops require matching operand types.
//! - **Result type matching**: Result type matches operand type where expected.
//! - **Attribute presence**: Required attributes (predicates, types) exist.
//! - **Rust-specific invariants**: Pointer address spaces, discriminant types.
//!
//! ## Key Verification Examples
//!
//! - **`mir.func`**: Entry block arguments must match function input types.
//! - **`mir.cond_branch`**: Condition must be `i1` (boolean).
//! - **`mir.goto`**: Target block arguments must type-match block parameters.
//! - **`mir.add`/`mir.sub`/etc.**: Both operands same type; result matches.
//! - **`mir.load`**: Operand must be pointer; result is pointee type.
//! - **`mir.store`**: First operand is value; second is pointer to that type.
//! - **`mir.ref`**: Result is pointer to operand type.
//! - **`mir.extract_field`**: Struct/tuple type checked; index in bounds.
//! - **`mir.get_discriminant`**: Result must be integer type.
//! - **`mir.cast`**: Must have `cast_kind` attribute; operand/result types must match the kind (e.g. IntToInt ⇒ both integer).
//!
//! ## Why Type Consistency Verification?
//!
//! 1. **Rust source is type-safe**: MIR is generated from type-checked Rust code.
//!    Verification confirms the importer preserved type safety.
//!
//! 2. **Foundation for lowering**: MIR → LLVM lowering relies on type information.
//!    Verification ensures types are correct before translation.
//!
//! 3. **Rust-specific semantics**: MIR preserves Rust concepts (references, enums,
//!    discriminants) that require specific type handling.
//!
//! 4. **Catch importer bugs**: The `mir-importer` translates rustc MIR to our MIR
//!    dialect. Verification catches translation errors.
//!
//! ## What is NOT Verified (deferred to lowering)
//!
//! - **Intrinsic signatures**: Call targets are symbols; signature matching
//!   happens at call resolution time.
//!
//! This balance provides meaningful validation while avoiding redundant checks
//! that `rustc` already performed on the source Rust code.

use pliron::context::Context;

pub mod aggregate;
pub mod arithmetic;
pub mod call;
pub mod cast;
pub mod comparison;
pub mod constants;
pub mod control_flow;
pub mod debug;
pub mod enum_ops;
pub mod function;
pub mod memory;
pub mod storage;

// Re-export all operations for convenient access
pub use aggregate::*;
pub use arithmetic::*;
pub use call::*;
pub use cast::*;
pub use comparison::*;
pub use constants::*;
pub use control_flow::*;
pub use debug::*;
pub use enum_ops::*;
pub use function::*;
pub use memory::*;
pub use storage::*;

/// Register all MIR dialect operations into the given context.
pub fn register(ctx: &mut Context) {
    function::register(ctx);
    control_flow::register(ctx);
    memory::register(ctx);
    constants::register(ctx);
    arithmetic::register(ctx);
    comparison::register(ctx);
    aggregate::register(ctx);
    enum_ops::register(ctx);
    debug::register(ctx);
    cast::register(ctx);
    storage::register(ctx);
    call::register(ctx);
}
