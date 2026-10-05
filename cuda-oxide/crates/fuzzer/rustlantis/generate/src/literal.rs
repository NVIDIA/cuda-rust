use mir::{
    syntax::{FloatTy, IntTy, Literal, TyId, TyKind, UintTy},
    tyctxt::TyCtxt,
};
use rand::{Rng, RngCore, seq::IndexedRandom};
use rand_distr::Distribution;

struct UsizeSombrero {
    small_values_upper_bound: usize,
}

impl Distribution<usize> for UsizeSombrero {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        if self.small_values_upper_bound == 0 {
            return rng.random_range(usize::MIN..=usize::MAX);
        }

        match rng.random_range(0..=1) {
            0 => rng.random_range(0..self.small_values_upper_bound),
            1 => rng.random_range(usize::MIN..=usize::MAX),
            _ => unreachable!(),
        }
    }
}

struct IsizeSombrero;

impl Distribution<isize> for IsizeSombrero {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> isize {
        match rng.random_range(0..=2) {
            0 => rng.random_range(-128i32..=127i32) as isize,
            1 => isize::MIN,
            2 => isize::MAX,
            _ => unreachable!(),
        }
    }
}

pub trait GenLiteral: Rng {
    fn is_literalble(ty: TyId, tcx: &TyCtxt) -> bool {
        match ty.kind(tcx) {
            TyKind::Unit => false,
            _ => ty.is_scalar(tcx),
        }
    }
    fn gen_literal(&mut self, ty: TyId, tcx: &TyCtxt) -> Option<Literal> {
        let lit: Literal = match ty.kind(tcx) {
            TyKind::Bool => self.random_bool(0.5).into(),
            TyKind::Char => {
                // There are 0xD7FF + 1 Unicode Scalar Values in the lower range, and 0x10FFFF - 0xE000 + 1
                // values in the upper range.
                let ordinal = self.random_range(0..((0xD7FF + 1) + (0x10FFFF - 0xE000 + 1)));
                if ordinal <= 0xD7FF {
                    char::from_u32(ordinal).unwrap().into()
                } else {
                    char::from_u32(ordinal - 0xD800 + 0xE000).unwrap().into()
                }
            }
            TyKind::Uint(UintTy::Usize) => {
                let distribution = UsizeSombrero {
                    small_values_upper_bound: tcx.config.array_max_len,
                };
                let i: usize = distribution.sample(self);
                i.try_into().expect("usize isn't greater than 128 bits")
            }
            TyKind::Uint(UintTy::U8) => self.random_range(u8::MIN..=u8::MAX).into(),
            TyKind::Uint(UintTy::U16) => self.random_range(u16::MIN..=u16::MAX).into(),
            TyKind::Uint(UintTy::U32) => self.random_range(u32::MIN..=u32::MAX).into(),
            TyKind::Uint(UintTy::U64) => self.random_range(u64::MIN..=u64::MAX).into(),
            TyKind::Uint(UintTy::U128) => self.random_range(u128::MIN..=u128::MAX).into(),
            TyKind::Int(IntTy::Isize) => {
                let i: isize = IsizeSombrero.sample(self);
                i.try_into().expect("isize isn't greater than 128 bits")
            }
            TyKind::Int(IntTy::I8) => self.random_range(i8::MIN..=i8::MAX).into(),
            TyKind::Int(IntTy::I16) => self.random_range(i16::MIN..=i16::MAX).into(),
            TyKind::Int(IntTy::I32) => self.random_range(i32::MIN..=i32::MAX).into(),
            TyKind::Int(IntTy::I64) => self.random_range(i64::MIN..=i64::MAX).into(),
            TyKind::Int(IntTy::I128) => self.random_range(i128::MIN..=i128::MAX).into(),
            TyKind::Float(FloatTy::F32) => generate_f32(self).into(),
            TyKind::Float(FloatTy::F64) => generate_f64(self).into(),
            _ => return None,
        };
        Some(lit)
    }
    fn gen_literal_non_zero(&mut self, ty: TyId, tcx: &TyCtxt) -> Option<Literal> {
        self.gen_literal(ty, tcx).map(|lit| match lit {
            Literal::Uint(n, t) => {
                if n == 0 {
                    Literal::Uint(n + 1, t)
                } else {
                    lit
                }
            }
            Literal::Int(n, t) => {
                if n == 0 {
                    Literal::Int(n + 1, t)
                } else {
                    lit
                }
            }
            Literal::Float(n, t) => {
                if n == 0. {
                    Literal::Float(n + 1., t)
                } else {
                    lit
                }
            }
            _ => lit,
        })
    }
}

impl<R: RngCore + ?Sized> GenLiteral for R {}

enum Category {
    Normal,
    Subnormal,
    Zero,
    Infinity,
    NaN,
}

const FLOAT_CATEGORIES: [Category; 5] = [
    Category::Normal,
    Category::Subnormal,
    Category::Zero,
    Category::Infinity,
    Category::NaN,
];

fn generate_f32<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let chosen = FLOAT_CATEGORIES.choose(rng).unwrap();
    match chosen {
        Category::Normal => {
            let sign: u32 = *[0 << 31, 1 << 31].choose(rng).unwrap();
            let exponent = rng.random_range(0x01..=0xfe);
            let fraction = rng.random_range(0..(1 << 23));
            f32::from_bits(sign | exponent | fraction)
        }
        Category::Subnormal => {
            let sign: u32 = *[0 << 31, 1 << 31].choose(rng).unwrap();
            let exponent = 0 << 23;
            let fraction = rng.random_range(1..(1 << 23));
            f32::from_bits(sign | exponent | fraction)
        }
        Category::Zero => *[0.0, -0.0].choose(rng).unwrap(),
        Category::Infinity => *[f32::INFINITY, f32::NEG_INFINITY].choose(rng).unwrap(),
        Category::NaN => f32::NAN,
    }
}

fn generate_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let chosen = FLOAT_CATEGORIES.choose(rng).unwrap();
    match chosen {
        Category::Normal => {
            let sign: u64 = *[0 << 63, 1 << 63].choose(rng).unwrap();
            let exponent = rng.random_range(0x001..=0x7fe);
            let fraction = rng.random_range(0..(1 << 52));
            f64::from_bits(sign | exponent | fraction)
        }
        Category::Subnormal => {
            let sign: u64 = *[0 << 63, 1 << 63].choose(rng).unwrap();
            let exponent = 0 << 52;
            let fraction = rng.random_range(1..(1 << 52));
            f64::from_bits(sign | exponent | fraction)
        }
        Category::Zero => *[0.0, -0.0].choose(rng).unwrap(),
        Category::Infinity => *[f64::INFINITY, f64::NEG_INFINITY].choose(rng).unwrap(),
        Category::NaN => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct SequenceRng {
        values: VecDeque<u64>,
    }

    impl SequenceRng {
        fn new(values: impl IntoIterator<Item = u64>) -> Self {
            Self {
                values: values.into_iter().collect(),
            }
        }
    }

    impl RngCore for SequenceRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.values.pop_front().expect("test RNG value")
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            for chunk in dst.chunks_mut(std::mem::size_of::<u64>()) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }
    }

    #[test]
    fn positive_bound_drives_small_value_sampling() {
        let distribution = UsizeSombrero {
            small_values_upper_bound: 3,
        };
        let mut rng = SequenceRng::new([0, u64::MAX]);

        assert_eq!(distribution.sample(&mut rng), 2);
    }

    #[test]
    fn positive_bound_keeps_the_full_width_bucket() {
        let distribution = UsizeSombrero {
            small_values_upper_bound: 3,
        };
        let mut rng = SequenceRng::new([u64::MAX, 0x1234]);

        assert_eq!(distribution.sample(&mut rng), 0x1234);
    }

    #[test]
    fn zero_bound_disables_the_small_value_bucket() {
        let distribution = UsizeSombrero {
            small_values_upper_bound: 0,
        };
        let mut rng = SequenceRng::new([0x1234]);

        assert_eq!(distribution.sample(&mut rng), 0x1234);
    }
}
