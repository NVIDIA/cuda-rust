/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level encoding primitives: varint, zigzag signed varint,
//! little-endian fixed-width, and alignment padding.
//!
//! Ported from `EncodingWriter` in `BytecodeWriter.cpp`.

#![allow(dead_code)]

use super::enums::ALIGNMENT_BYTE;

/// Byte-level encoder that writes into a `Vec<u8>` buffer.
pub struct EncodingWriter {
    buf: Vec<u8>,
    required_alignment: u64,
}

impl Default for EncodingWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            required_alignment: 1,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            required_alignment: 1,
        }
    }

    // ------ Raw writes ------

    pub fn write_byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    // ------ Variable-length integer (unsigned) ------

    /// Write an unsigned variable-length integer (LEB128-style, 7 bits per byte,
    /// high bit = continuation).
    pub fn write_varint(&mut self, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            self.buf.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    /// Write a signed variable-length integer using zigzag encoding.
    pub fn write_signed_varint(&mut self, value: i64) {
        let v = value as u64;
        let encoded = (v << 1) ^ ((v as i64 >> 63) as u64);
        self.write_varint(encoded);
    }

    // ------ Little-endian fixed-width ------

    pub fn write_le_u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    pub fn write_le_u16(&mut self, value: u16) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_i32(&mut self, value: i32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_i64(&mut self, value: i64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_f32(&mut self, value: f32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_le_f64(&mut self, value: f64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    // ------ APFloat encoding (mirrors C++ writeAPFloatRepresentation) ------

    /// Write a float value using the APInt bitcast representation, matching
    /// the C++ `writeAPFloatRepresentation` / `writeAPInt` encoding.
    ///
    /// The C++ rules:
    /// - bitWidth <= 8: writeByte
    /// - bitWidth <= 64: writeSignedVarInt
    /// - bitWidth > 64: writeVarInt(numWords) + writeSignedVarInt per word
    pub fn write_ap_float(&mut self, value: f64, ty: &crate::ir::Type) {
        use crate::ir::ScalarType;
        let scalar = match ty {
            crate::ir::Type::Scalar(s) => *s,
            _ => {
                // Fallback: treat as f64
                let bits = value.to_bits();
                self.write_signed_varint(bits as i64);
                return;
            }
        };
        match scalar {
            ScalarType::F16 => {
                let h = half::f16::from_f64(value);
                let bits = h.to_bits() as u64;
                self.write_signed_varint(bits as i64);
            }
            ScalarType::BF16 => {
                let h = half::bf16::from_f64(value);
                let bits = h.to_bits() as u64;
                self.write_signed_varint(bits as i64);
            }
            ScalarType::F32 => {
                let bits = (value as f32).to_bits() as u64;
                self.write_signed_varint(bits as i64);
            }
            ScalarType::F64 => {
                let bits = value.to_bits();
                self.write_signed_varint(bits as i64);
            }
            ScalarType::TF32 => {
                // TF32 uses f32 representation with lower bits zeroed
                let bits = (value as f32).to_bits() as u64;
                self.write_signed_varint(bits as i64);
            }
            ScalarType::F8E4M3FN => {
                // 8-bit float: sign(1) + exponent(4) + mantissa(3), bias=7
                let byte = f64_to_f8e4m3fn(value);
                self.write_byte(byte);
            }
            ScalarType::F8E5M2 => {
                // 8-bit float: sign(1) + exponent(5) + mantissa(2), bias=15
                let byte = f64_to_f8e5m2(value);
                self.write_byte(byte);
            }
            _ => {
                // Integer scalars shouldn't be used for float attrs
                let bits = value.to_bits();
                self.write_signed_varint(bits as i64);
            }
        }
    }

    // ------ Array helpers ------

    /// Write `count` as a varint, then the values as little-endian i64.
    pub fn write_le_var_size_i64(&mut self, values: &[i64]) {
        self.write_varint(values.len() as u64);
        for &v in values {
            self.write_le_i64(v);
        }
    }

    /// Write `count` as a varint, then the values as little-endian i32.
    pub fn write_le_var_size_i32(&mut self, values: &[i32]) {
        self.write_varint(values.len() as u64);
        for &v in values {
            self.write_le_i32(v);
        }
    }

    // ------ Alignment ------

    /// Pad the buffer so that the current position is aligned to `alignment`.
    pub fn align_to(&mut self, alignment: u64) {
        if alignment < 2 {
            return;
        }
        let pos = self.buf.len() as u64;
        let padding = (alignment - (pos % alignment)) % alignment;
        for _ in 0..padding {
            self.buf.push(ALIGNMENT_BYTE);
        }
        self.required_alignment = self.required_alignment.max(alignment);
    }

    // ------ Buffer access ------

    pub fn tell(&self) -> usize {
        self.buf.len()
    }

    pub fn required_alignment(&self) -> u64 {
        self.required_alignment
    }

    /// Consume the writer and return the underlying buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the underlying buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Get a mutable reference to the raw buffer (for patching offsets).
    pub fn buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }
}

/// Patch a `u32` value at `offset` in the buffer (little-endian).
pub fn patch_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Patch a `u64` value at `offset` in the buffer (little-endian).
pub fn patch_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------
// F8 float conversion
// ---------------------------------------------------------------------------

/// Convert f64 to F8E4M3FN bit pattern (sign:1, exp:4, man:3, bias=7).
/// NaN maps to 0x7F (S=0, E=1111, M=111). No infinities in this format.
fn f64_to_f8e4m3fn(value: f64) -> u8 {
    convert_to_f8(value, 4, 3, 7, true)
}

/// Convert f64 to F8E5M2 bit pattern (sign:1, exp:5, man:2, bias=15).
/// IEEE-like with NaN when E=all-ones and M!=0.
fn f64_to_f8e5m2(value: f64) -> u8 {
    convert_to_f8(value, 5, 2, 15, false)
}

/// Generic f64 → 8-bit float conversion.
/// `nan_only_all_ones`: if true, NaN is encoded as all-ones mantissa
/// (F8E4M3FN style — no infinities). If false, IEEE-style (F8E5M2).
fn convert_to_f8(
    value: f64,
    exp_bits: u32,
    man_bits: u32,
    bias: i32,
    nan_only_all_ones: bool,
) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 63) & 1) as u8;
    let f64_exp = ((bits >> 52) & 0x7FF) as i32;
    let f64_man = bits & ((1u64 << 52) - 1);

    let max_exp = (1i32 << exp_bits) - 1;
    let man_mask = (1u8 << man_bits) - 1;
    let max_finite = |sign: u8| -> u8 {
        if nan_only_all_ones {
            (sign << 7) | ((max_exp as u8) << man_bits) | (man_mask - 1)
        } else {
            (sign << 7) | (((max_exp - 1) as u8) << man_bits) | man_mask
        }
    };

    // Handle special values.
    if f64_exp == 0x7FF {
        // Inf or NaN
        if f64_man != 0 {
            // NaN
            if nan_only_all_ones {
                return (sign << 7) | ((max_exp as u8) << man_bits) | man_mask;
            } else {
                // IEEE NaN: all-ones exponent, non-zero mantissa
                return (sign << 7) | ((max_exp as u8) << man_bits) | 1;
            }
        }
        if !nan_only_all_ones {
            // Inf in IEEE-style format
            return (sign << 7) | ((max_exp as u8) << man_bits);
        }
        // Formats without inf: saturate to max finite
        return max_finite(sign);
    }

    if value == 0.0 || value == -0.0 {
        return sign << 7;
    }

    let (significand, unbiased) = if f64_exp == 0 {
        (f64_man, -1022)
    } else {
        ((1u64 << 52) | f64_man, f64_exp - 1023)
    };
    if significand == 0 {
        return sign << 7;
    }

    // Unbias f64 exponent, rebias for target.
    let mut target_exp = unbiased + bias;

    if target_exp > max_exp || (!nan_only_all_ones && target_exp >= max_exp) {
        // Overflow before rounding: clamp to max finite (or inf for IEEE-style).
        if !nan_only_all_ones {
            return (sign << 7) | ((max_exp as u8) << man_bits); // Inf
        }
        return max_finite(sign);
    }

    if target_exp <= 0 {
        // Subnormal or underflow to zero.
        let shift = 1 - target_exp;
        let subnormal_man =
            round_shift_right_ties_even(significand, (52 - man_bits as i32 + shift) as u32);
        if subnormal_man == 0 {
            return sign << 7; // Underflow to zero
        }
        if subnormal_man >= (1u64 << man_bits) {
            return (sign << 7) | (1u8 << man_bits); // Rounds up to the minimum normal.
        }
        return (sign << 7) | (subnormal_man as u8 & man_mask);
    }

    let rounded_significand = round_shift_right_ties_even(significand, 52 - man_bits);
    let mut mantissa = (rounded_significand as u8) & man_mask;
    if rounded_significand == (1u64 << (man_bits + 1)) {
        target_exp += 1;
        mantissa = 0;
    }

    if target_exp > max_exp || (nan_only_all_ones && target_exp == max_exp && mantissa == man_mask)
    {
        return max_finite(sign);
    }
    if !nan_only_all_ones && target_exp >= max_exp {
        return (sign << 7) | ((max_exp as u8) << man_bits); // Inf
    }

    (sign << 7) | ((target_exp as u8) << man_bits) | mantissa
}

fn round_shift_right_ties_even(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }

    let truncated = value >> shift;
    let remainder_mask = (1u64 << shift) - 1;
    let remainder = value & remainder_mask;
    let half = 1u64 << (shift - 1);
    let should_round_up = remainder > half || (remainder == half && (truncated & 1) == 1);
    truncated + u64::from(should_round_up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_zero() {
        let mut w = EncodingWriter::new();
        w.write_varint(0);
        assert_eq!(w.as_bytes(), &[0x00]);
    }

    #[test]
    fn varint_small() {
        let mut w = EncodingWriter::new();
        w.write_varint(127);
        assert_eq!(w.as_bytes(), &[0x7F]);
    }

    #[test]
    fn varint_multi_byte() {
        let mut w = EncodingWriter::new();
        w.write_varint(300);
        // 300 = 0b100101100 → [0xAC, 0x02]
        assert_eq!(w.as_bytes(), &[0xAC, 0x02]);
    }

    #[test]
    fn signed_varint_positive() {
        let mut w = EncodingWriter::new();
        w.write_signed_varint(1);
        // zigzag(1) = 2
        assert_eq!(w.as_bytes(), &[0x02]);
    }

    #[test]
    fn signed_varint_negative() {
        let mut w = EncodingWriter::new();
        w.write_signed_varint(-1);
        // zigzag(-1) = 1
        assert_eq!(w.as_bytes(), &[0x01]);
    }

    #[test]
    fn alignment_padding() {
        let mut w = EncodingWriter::new();
        w.write_byte(0x01); // pos=1
        w.align_to(4); // should pad 3 bytes
        assert_eq!(w.tell(), 4);
        assert_eq!(
            w.as_bytes(),
            &[0x01, ALIGNMENT_BYTE, ALIGNMENT_BYTE, ALIGNMENT_BYTE]
        );
    }

    #[test]
    fn le_u32_roundtrip() {
        let mut w = EncodingWriter::new();
        w.write_le_u32(0xDEADBEEF);
        assert_eq!(w.as_bytes(), &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn f8e4m3fn_rounds_normal_ties_to_even() {
        // 1.1875 is exactly halfway between mantissas 1 and 2 at exponent 0.
        // Ties-to-even must choose mantissa 2, while truncation chose 1.
        assert_eq!(f64_to_f8e4m3fn(1.1875), 0x3A);
    }

    #[test]
    fn f8e5m2_rounds_normal_ties_to_even() {
        // 1.375 is exactly halfway between mantissas 1 and 2 at exponent 0.
        // Ties-to-even must choose mantissa 2, while truncation chose 1.
        assert_eq!(f64_to_f8e5m2(1.375), 0x3E);
    }

    #[test]
    fn f8e4m3fn_rounds_subnormal_tie_to_min_normal() {
        // Halfway between the largest subnormal (0x07) and minimum normal
        // (0x08). The normal endpoint has an even encoded significand.
        assert_eq!(f64_to_f8e4m3fn(7.5 * 2f64.powi(-9)), 0x08);
    }

    #[test]
    fn f8e5m2_rounds_subnormal_tie_to_min_normal() {
        // Halfway between the largest subnormal (0x03) and minimum normal
        // (0x04). The normal endpoint has an even encoded significand.
        assert_eq!(f64_to_f8e5m2(3.5 * 2f64.powi(-16)), 0x04);
    }

    #[test]
    fn f8e4m3fn_preserves_finite_all_ones_exponent_values() {
        assert_eq!(f64_to_f8e4m3fn(256.0), 0x78);
        assert_eq!(f64_to_f8e4m3fn(448.0), 0x7E);
        assert_eq!(f64_to_f8e4m3fn(f64::INFINITY), 0x7E);
    }
}
