use std::borrow::BorrowMut;

use mir::{
    syntax::{Callee, Mutability, Operand, Place, TyId, TyKind},
    tyctxt::TyCtxt,
};
use rand::{Rng, seq::IteratorRandom};

use crate::{literal::GenLiteral, mem::BasicMemory, place_select::PlaceSelector};

use super::{GenerationCtx, Result, SelectionError};

pub trait CoreIntrinsic {
    fn name(&self) -> &'static str;

    fn dest_type(&self, ty: TyId, tcx: &TyCtxt) -> bool;

    fn choose_operands(&self, ctx: &GenerationCtx, dest: &Place) -> Option<Vec<Operand>>;

    fn generate_terminator(
        &self,
        ctx: &GenerationCtx,
        dest: &Place,
    ) -> Result<(Callee, Vec<Operand>)> {
        if !self.dest_type(dest.ty(ctx.current_decls(), &ctx.tcx), &ctx.tcx) {
            return Err(SelectionError::Exhausted);
        }
        let args = self
            .choose_operands(ctx, dest)
            .ok_or(SelectionError::Exhausted)?;
        Ok((Callee::Intrinsic(self.name()), args))
    }
}

struct Fmaf64;
impl CoreIntrinsic for Fmaf64 {
    fn name(&self) -> &'static str {
        "fmaf64"
    }

    fn dest_type(&self, ty: TyId, _: &TyCtxt) -> bool {
        ty == TyCtxt::F64
    }

    fn choose_operands(&self, ctx: &GenerationCtx, dest: &Place) -> Option<Vec<Operand>> {
        let a = ctx.choose_operand(&[TyCtxt::F64], dest).ok()?;
        let b = ctx.choose_operand(&[TyCtxt::F64], dest).ok()?;
        let c = ctx.choose_operand(&[TyCtxt::F64], dest).ok()?;
        Some(vec![a, b, c])
    }
}

pub(super) struct ArithOffset;
impl CoreIntrinsic for ArithOffset {
    fn name(&self) -> &'static str {
        "arith_offset"
    }

    fn dest_type(&self, ty: TyId, tcx: &TyCtxt) -> bool {
        matches!(ty.kind(tcx), TyKind::RawPtr(.., Mutability::Not))
    }

    fn choose_operands(&self, ctx: &GenerationCtx, dest: &Place) -> Option<Vec<Operand>> {
        let (ptrs, weights) = PlaceSelector::for_offsetee(ctx.tcx.clone())
            .of_ty(dest.ty(ctx.current_decls(), &ctx.tcx))
            .except(dest)
            .into_weighted(&ctx.pt)?;
        let ptr = ctx
            .make_choice_weighted(ptrs.into_iter(), weights, |ppath| {
                Ok(ppath.to_place(&ctx.pt, ctx.current_decls()))
            })
            .ok()?;

        let offset = ctx.pt.get_offset(&ptr);

        let mut rng = ctx.rng.borrow_mut();
        let new_offset = match offset {
            // Don't break roundtripped pointer
            Some(0) => {
                return None;
            }
            Some(existing) if existing.checked_neg().is_some() && rng.random_bool(0.5) => {
                Operand::Constant((-existing).try_into().unwrap())
            }
            _ => PlaceSelector::for_known_val(ctx.tcx.clone())
                .of_ty(TyCtxt::ISIZE)
                .into_iter_place(&ctx.pt, ctx.current_decls())
                .choose(&mut *rng)
                .map(Operand::Copy)
                .unwrap_or_else(|| {
                    Operand::Constant(
                        rng.borrow_mut()
                            .gen_literal(TyCtxt::ISIZE, &ctx.tcx)
                            .expect("can generate a literal"),
                    )
                }),
        };

        Some(vec![Operand::Copy(ptr), new_offset])
    }
}

struct Bswap;
impl CoreIntrinsic for Bswap {
    fn name(&self) -> &'static str {
        "bswap"
    }

    fn dest_type(&self, ty: TyId, tcx: &TyCtxt) -> bool {
        matches!(ty.kind(tcx), TyKind::Int(..) | TyKind::Uint(..))
    }

    fn choose_operands(&self, ctx: &GenerationCtx, dest: &Place) -> Option<Vec<Operand>> {
        let arg = ctx
            .choose_operand(&[dest.ty(ctx.current_decls(), &ctx.tcx)], dest)
            .ok()?;
        Some(vec![arg])
    }
}

pub(super) struct Transmute;
impl CoreIntrinsic for Transmute {
    fn name(&self) -> &'static str {
        "transmute"
    }

    fn dest_type(&self, ty: TyId, tcx: &TyCtxt) -> bool {
        if ty.contains(tcx, |tcx, ty| match ty.kind(tcx) {
            // Tys with value validity contstraints
            TyKind::Unit | TyKind::Bool | TyKind::Char | TyKind::RawPtr(..) | TyKind::Ref(..) => {
                true
            } // TODO: pointer transmute
            _ => false,
        }) {
            return false;
        }
        if BasicMemory::ty_size(ty, tcx).is_none() {
            return false;
        }
        true
    }

    fn choose_operands(&self, ctx: &GenerationCtx, dest: &Place) -> Option<Vec<Operand>> {
        let dest_size = BasicMemory::ty_size(dest.ty(ctx.current_decls(), &ctx.tcx), &ctx.tcx)
            .expect("dest must have known size");
        // Avoid pointer to int casts
        let allowed_tys: Vec<TyId> = ctx
            .tcx
            .indices()
            .filter(|ty| {
                !ty.contains(&ctx.tcx, |tcx, ty| {
                    // Avoid inspecting the bytes in fp as NaN payload is nd
                    ty.is_any_ptr(tcx) || ty == TyCtxt::F32 || ty == TyCtxt::F64
                })
            })
            .collect();

        let (srcs, weights) = PlaceSelector::for_argument(ctx.tcx.clone())
            .of_tys(&allowed_tys)
            .of_size(dest_size)
            .except(dest)
            .into_weighted(&ctx.pt)?;
        let src = ctx
            .make_choice_weighted(srcs.into_iter(), weights, |ppath| {
                Ok(ppath.to_place(&ctx.pt, ctx.current_decls()))
            })
            .ok()?;
        if src.ty(ctx.current_decls(), &ctx.tcx).is_copy(&ctx.tcx) {
            Some(vec![Operand::Copy(src)])
        } else {
            Some(vec![Operand::Move(src)])
        }
    }
}

impl GenerationCtx {
    pub fn choose_intrinsic(&self, dest: &Place) -> Result<(Callee, Vec<Operand>)> {
        let choices: [Box<dyn CoreIntrinsic>; 4] = [
            Box::new(Fmaf64),
            Box::new(ArithOffset),
            Box::new(Bswap),
            Box::new(Transmute),
        ];

        let intrinsic = self.make_choice(choices.iter(), Result::Ok)?;
        intrinsic.generate_terminator(self, dest)
    }
}
