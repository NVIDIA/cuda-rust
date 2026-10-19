/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline PTX support.
//!
//! User code should use [`ptx_asm!`](crate::ptx_asm), not the hidden functions
//! in this module. The hidden functions are compiler markers: the MIR importer
//! recognizes calls to them and replaces those calls with inline PTX.

macro_rules! define_ptx_asm_out {
    ($name:ident; $($arg:ident : $ty:ident),*) => {
        #[doc(hidden)]
        #[inline(never)]
        #[allow(unused_variables)]
        #[allow(clippy::too_many_arguments)]
        /// # Safety
        ///
        /// Compiler marker for `ptx_asm!`; user code must not call this directly.
        pub unsafe fn $name<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            $($ty,)*
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            $($arg: $ty,)*
        ) -> T {
            unreachable!("ptx_asm marker called outside CUDA kernel context")
        }
    };
}

macro_rules! define_ptx_asm_void {
    ($name:ident; $($arg:ident : $ty:ident),*) => {
        #[doc(hidden)]
        #[inline(never)]
        #[allow(unused_variables)]
        #[allow(clippy::too_many_arguments)]
        /// # Safety
        ///
        /// Compiler marker for `ptx_asm!`; user code must not call this directly.
        pub unsafe fn $name<
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            $($ty,)*
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            $($arg: $ty,)*
        ) {
            unreachable!("ptx_asm marker called outside CUDA kernel context")
        }
    };
}

/// Typed identity helper for `in("C")` operands.
///
/// `ptx_asm!` wraps every `in("C")` operand in
/// `const { __ptx_asm_c(...) }`: the signature rejects anything that is not
/// a `&'static [u8; N]` byte string at type-check time, and the inline
/// const keeps the operand a compile-time constant so the MIR importer can
/// splice its text into the PTX template.
#[doc(hidden)]
#[inline(always)]
pub const fn __ptx_asm_c<const N: usize>(value: &'static [u8; N]) -> &'static [u8; N] {
    value
}

// Rust has no variadic generics, so expose marker stubs for fixed arities.
// Output markers support up to 32 arguments: 16 explicit inputs plus up to
// 16 hidden tied inputs generated for `inout` operands. Void markers require
// only 16 arguments because `inout` always produces an output.
define_ptx_asm_out!(__ptx_asm_out_0;);
define_ptx_asm_out!(__ptx_asm_out_1; a0: A0);
define_ptx_asm_out!(__ptx_asm_out_2; a0: A0, a1: A1);
define_ptx_asm_out!(__ptx_asm_out_3; a0: A0, a1: A1, a2: A2);
define_ptx_asm_out!(__ptx_asm_out_4; a0: A0, a1: A1, a2: A2, a3: A3);
define_ptx_asm_out!(__ptx_asm_out_5; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4);
define_ptx_asm_out!(__ptx_asm_out_6; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5);
define_ptx_asm_out!(__ptx_asm_out_7; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6);
define_ptx_asm_out!(__ptx_asm_out_8; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7);
define_ptx_asm_out!(__ptx_asm_out_9; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8);
define_ptx_asm_out!(__ptx_asm_out_10; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9);
define_ptx_asm_out!(__ptx_asm_out_11; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10);
define_ptx_asm_out!(__ptx_asm_out_12; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11);
define_ptx_asm_out!(__ptx_asm_out_13; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12);
define_ptx_asm_out!(__ptx_asm_out_14; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13);
define_ptx_asm_out!(__ptx_asm_out_15; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14);
define_ptx_asm_out!(__ptx_asm_out_16; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15);
define_ptx_asm_out!(__ptx_asm_out_17; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16);
define_ptx_asm_out!(__ptx_asm_out_18; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17);
define_ptx_asm_out!(__ptx_asm_out_19; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18);
define_ptx_asm_out!(__ptx_asm_out_20; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19);
define_ptx_asm_out!(__ptx_asm_out_21; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20);
define_ptx_asm_out!(__ptx_asm_out_22; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21);
define_ptx_asm_out!(__ptx_asm_out_23; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22);
define_ptx_asm_out!(__ptx_asm_out_24; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23);
define_ptx_asm_out!(__ptx_asm_out_25; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24);
define_ptx_asm_out!(__ptx_asm_out_26; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25);
define_ptx_asm_out!(__ptx_asm_out_27; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26);
define_ptx_asm_out!(__ptx_asm_out_28; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26, a27: A27);
define_ptx_asm_out!(__ptx_asm_out_29; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26, a27: A27, a28: A28);
define_ptx_asm_out!(__ptx_asm_out_30; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26, a27: A27, a28: A28, a29: A29);
define_ptx_asm_out!(__ptx_asm_out_31; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26, a27: A27, a28: A28, a29: A29, a30: A30);
define_ptx_asm_out!(__ptx_asm_out_32; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17, a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23, a24: A24, a25: A25, a26: A26, a27: A27, a28: A28, a29: A29, a30: A30, a31: A31);

define_ptx_asm_void!(__ptx_asm_void_0;);
define_ptx_asm_void!(__ptx_asm_void_1; a0: A0);
define_ptx_asm_void!(__ptx_asm_void_2; a0: A0, a1: A1);
define_ptx_asm_void!(__ptx_asm_void_3; a0: A0, a1: A1, a2: A2);
define_ptx_asm_void!(__ptx_asm_void_4; a0: A0, a1: A1, a2: A2, a3: A3);
define_ptx_asm_void!(__ptx_asm_void_5; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4);
define_ptx_asm_void!(__ptx_asm_void_6; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5);
define_ptx_asm_void!(__ptx_asm_void_7; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6);
define_ptx_asm_void!(__ptx_asm_void_8; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7);
define_ptx_asm_void!(__ptx_asm_void_9; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8);
define_ptx_asm_void!(__ptx_asm_void_10; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9);
define_ptx_asm_void!(__ptx_asm_void_11; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10);
define_ptx_asm_void!(__ptx_asm_void_12; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11);
define_ptx_asm_void!(__ptx_asm_void_13; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12);
define_ptx_asm_void!(__ptx_asm_void_14; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13);
define_ptx_asm_void!(__ptx_asm_void_15; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14);
define_ptx_asm_void!(__ptx_asm_void_16; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15);
