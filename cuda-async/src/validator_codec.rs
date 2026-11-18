/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Byte encoding of a [`Validator`], so a persistent kernel cache can serve the
//! launch checks of a kernel compiled by another process.
//!
//! Little-endian; strings and sequences carry a `u32` length, `usize` fields
//! travel as `u64`. Decoding accepts only what encoding produces: unknown tags,
//! out-of-range values, a term outside its normal form, a divisor below 1 and
//! trailing bytes all fail.

use crate::device_context::{
    PointerParamType, ScalarParamType, TensorParamType, ValidParamType, Validator,
};
use crate::predicate::{Atom, LaunchCheck, Predicate, Term};

/// Bump on any change to the encoding below.
const FORMAT_VERSION: u8 = 1;

impl Validator {
    /// The encoding [`Self::from_bytes`] reads back.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(vec![FORMAT_VERSION]);
        w.len(self.params.len());
        for param in &self.params {
            match param {
                ValidParamType::Scalar(ScalarParamType { element_type }) => {
                    w.u8(0);
                    w.str(element_type);
                }
                ValidParamType::Pointer(PointerParamType {
                    mutable,
                    element_type,
                }) => {
                    w.u8(1);
                    w.u8(u8::from(*mutable));
                    w.str(element_type);
                }
                ValidParamType::Tensor(TensorParamType {
                    element_type,
                    shape,
                }) => {
                    w.u8(2);
                    w.str(element_type);
                    w.len(shape.len());
                    for &dim in shape {
                        w.i32(dim);
                    }
                }
            }
        }
        w.len(self.launch_checks.len());
        for LaunchCheck { predicate, cause } in &self.launch_checks {
            w.predicate(predicate);
            w.str(cause);
        }
        w.0
    }

    /// Decodes [`Self::to_bytes`] output; `None` for anything else.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader(bytes);
        if r.u8()? != FORMAT_VERSION {
            return None;
        }
        let params = (0..r.len()?)
            .map(|_| match r.u8()? {
                0 => Some(ValidParamType::Scalar(ScalarParamType {
                    element_type: r.string()?,
                })),
                1 => Some(ValidParamType::Pointer(PointerParamType {
                    mutable: r.bool()?,
                    element_type: r.string()?,
                })),
                2 => Some(ValidParamType::Tensor(TensorParamType {
                    element_type: r.string()?,
                    shape: (0..r.len()?).map(|_| r.i32()).collect::<Option<_>>()?,
                })),
                _ => None,
            })
            .collect::<Option<_>>()?;
        let launch_checks = (0..r.len()?)
            .map(|_| {
                Some(LaunchCheck {
                    predicate: r.predicate()?,
                    cause: r.string()?,
                })
            })
            .collect::<Option<_>>()?;
        r.0.is_empty().then_some(Validator {
            params,
            launch_checks,
        })
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u64).to_le_bytes());
    }
    fn len(&mut self, n: usize) {
        self.u32(u32::try_from(n).expect("validator sequence longer than u32::MAX"));
    }
    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn atom(&mut self, atom: &Atom) {
        match *atom {
            Atom::Dim { param, axis } => {
                self.u8(0);
                self.usize(param);
                self.usize(axis);
            }
            Atom::Iv(id) => {
                self.u8(1);
                self.u32(id);
            }
            Atom::TileBlockId(axis) => {
                self.u8(2);
                self.usize(axis);
            }
            Atom::NumTileBlocks(axis) => {
                self.u8(3);
                self.usize(axis);
            }
            Atom::ViewExtent { param, axis } => {
                self.u8(4);
                self.usize(param);
                self.usize(axis);
            }
            Atom::TileCount { param, axis, tile } => {
                self.u8(5);
                self.usize(param);
                self.usize(axis);
                self.i32(tile);
            }
        }
    }
    fn term(&mut self, term: &Term) {
        self.len(term.coeffs().len());
        for (atom, &c) in term.coeffs() {
            self.atom(atom);
            self.i64(c);
        }
        self.i64(term.constant_part());
    }
    fn predicate(&mut self, predicate: &Predicate) {
        match predicate {
            Predicate::Zero(t) => {
                self.u8(0);
                self.term(t);
            }
            Predicate::Nonzero(t) => {
                self.u8(1);
                self.term(t);
            }
            Predicate::Positive(t) => {
                self.u8(2);
                self.term(t);
            }
            Predicate::DivisibleBy { term, divisor } => {
                self.u8(3);
                self.term(term);
                self.i64(*divisor);
            }
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|[b]| b)
    }
    fn bool(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn i32(&mut self) -> Option<i32> {
        self.take().map(i32::from_le_bytes)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn i64(&mut self) -> Option<i64> {
        self.take().map(i64::from_le_bytes)
    }
    fn usize(&mut self) -> Option<usize> {
        self.take().map(u64::from_le_bytes)?.try_into().ok()
    }
    /// A sequence length, bounded by the bytes left so a corrupt length
    /// cannot drive a huge loop.
    fn len(&mut self) -> Option<usize> {
        let n = usize::try_from(self.u32()?).ok()?;
        (n <= self.0.len()).then_some(n)
    }
    fn string(&mut self) -> Option<String> {
        let n = self.len()?;
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        String::from_utf8(head.to_vec()).ok()
    }
    fn atom(&mut self) -> Option<Atom> {
        Some(match self.u8()? {
            0 => Atom::Dim {
                param: self.usize()?,
                axis: self.usize()?,
            },
            1 => Atom::Iv(self.u32()?),
            2 => Atom::TileBlockId(self.usize()?),
            3 => Atom::NumTileBlocks(self.usize()?),
            4 => Atom::ViewExtent {
                param: self.usize()?,
                axis: self.usize()?,
            },
            5 => Atom::TileCount {
                param: self.usize()?,
                axis: self.usize()?,
                tile: self.i32()?,
            },
            _ => return None,
        })
    }
    fn term(&mut self) -> Option<Term> {
        let coeffs = (0..self.len()?)
            .map(|_| Some((self.atom()?, self.i64()?)))
            .collect::<Option<Vec<_>>>()?;
        Term::from_parts(coeffs, self.i64()?)
    }
    fn predicate(&mut self) -> Option<Predicate> {
        Some(match self.u8()? {
            0 => Predicate::Zero(self.term()?),
            1 => Predicate::Nonzero(self.term()?),
            2 => Predicate::Positive(self.term()?),
            3 => {
                let term = self.term()?;
                let divisor = self.i64()?;
                if divisor < 1 {
                    return None;
                }
                Predicate::DivisibleBy { term, divisor }
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_variant() -> Validator {
        let term = Term::atom(Atom::Dim { param: 0, axis: 1 })
            .add(&Term::affine(Atom::Iv(7), -3, 5))
            .unwrap()
            .add(&Term::atom(Atom::TileBlockId(2)))
            .unwrap()
            .add(&Term::atom(Atom::NumTileBlocks(1)))
            .unwrap()
            .add(&Term::atom(Atom::ViewExtent { param: 2, axis: 0 }))
            .unwrap()
            .add(&Term::atom(Atom::TileCount {
                param: 1,
                axis: 0,
                tile: 64,
            }))
            .unwrap();
        let check = |predicate, cause: &str| LaunchCheck {
            predicate,
            cause: cause.to_string(),
        };
        Validator {
            params: vec![
                ValidParamType::Scalar(ScalarParamType {
                    element_type: "f32".into(),
                }),
                ValidParamType::Pointer(PointerParamType {
                    mutable: true,
                    element_type: "u16".into(),
                }),
                ValidParamType::Tensor(TensorParamType {
                    element_type: "bf16".into(),
                    shape: vec![-1, 128, 0],
                }),
            ],
            launch_checks: vec![
                check(Predicate::Zero(term.clone()), "zero"),
                check(Predicate::Nonzero(term.clone()), "nonzero"),
                check(Predicate::Positive(Term::constant(-4)), "positive"),
                check(
                    Predicate::divisible_by(term, 16).unwrap(),
                    "divisible — ünïcode",
                ),
            ],
        }
    }

    #[test]
    fn roundtrip_covers_every_variant() {
        let v = every_variant();
        let decoded = Validator::from_bytes(&v.to_bytes()).unwrap();
        assert_eq!(format!("{decoded:?}"), format!("{v:?}"));
        let empty = Validator {
            params: vec![],
            launch_checks: vec![],
        };
        let decoded = Validator::from_bytes(&empty.to_bytes()).unwrap();
        assert_eq!(format!("{decoded:?}"), format!("{empty:?}"));
    }

    #[test]
    fn rejects_truncation_trailing_bytes_and_other_versions() {
        let bytes = every_variant().to_bytes();
        for n in 0..bytes.len() {
            assert!(Validator::from_bytes(&bytes[..n]).is_none(), "prefix {n}");
        }
        let mut long = bytes.clone();
        long.push(0);
        assert!(Validator::from_bytes(&long).is_none());
        let mut other = bytes;
        other[0] = FORMAT_VERSION + 1;
        assert!(Validator::from_bytes(&other).is_none());
    }

    fn one_check(predicate_bytes: &[u8]) -> Vec<u8> {
        let mut w = Writer(vec![FORMAT_VERSION]);
        w.len(0);
        w.len(1);
        w.0.extend_from_slice(predicate_bytes);
        w.str("c");
        w.0
    }

    #[test]
    fn rejects_terms_outside_normal_form_and_divisors_below_one() {
        let atom = |w: &mut Writer, axis| w.atom(&Atom::NumTileBlocks(axis));
        // Zero coefficient.
        let mut w = Writer(vec![0]);
        w.len(1);
        atom(&mut w, 0);
        w.i64(0);
        w.i64(1);
        assert!(Validator::from_bytes(&one_check(&w.0)).is_none());
        // Atoms out of order, and repeated.
        for axes in [[1, 0], [0, 0]] {
            let mut w = Writer(vec![0]);
            w.len(2);
            for axis in axes {
                atom(&mut w, axis);
                w.i64(1);
            }
            w.i64(0);
            assert!(Validator::from_bytes(&one_check(&w.0)).is_none());
        }
        // `DivisibleBy` with a divisor that `eval` cannot take.
        for divisor in [0, -8] {
            let mut w = Writer(vec![3]);
            w.term(&Term::atom(Atom::NumTileBlocks(0)));
            w.i64(divisor);
            assert!(Validator::from_bytes(&one_check(&w.0)).is_none());
        }
        let mut w = Writer(vec![3]);
        w.term(&Term::atom(Atom::NumTileBlocks(0)));
        w.i64(8);
        assert!(Validator::from_bytes(&one_check(&w.0)).is_some());
    }
}
