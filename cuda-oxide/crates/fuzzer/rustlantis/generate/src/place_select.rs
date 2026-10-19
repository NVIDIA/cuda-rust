use std::{rc::Rc, vec};

use abi::size::Size;
use mir::{
    syntax::{Literal, LocalDecls, Place, TyId},
    tyctxt::TyCtxt,
};
use rand_distr::weighted::WeightedIndex;

use crate::{
    mem::BasicMemory,
    pgraph::{PlaceGraph, PlaceIndex, PlacePath, ToPlaceIndex},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlaceUsage {
    Operand,
    LHS,
    RET,
    SetDiscriminant,
    Pointee,
    Argument,
    KnownVal,
    NonZero,
    Offsetee,
}

#[derive(Clone)]
pub struct PlaceSelector {
    tys: Option<Vec<TyId>>,
    exclusions: Vec<Place>,
    moved: Vec<PlaceIndex>,
    refed: Vec<PlaceIndex>,
    size: Option<Size>,
    allow_uninit: bool,
    usage: PlaceUsage,
    tcx: Rc<TyCtxt>,
}

pub type Weight = usize;

const RET_LHS_WEIGHT_FACTOR: Weight = 2;
const UNINIT_WEIGHT_FACTOR: Weight = 2;
const DEREF_WEIGHT_FACTOR: Weight = 20;
const LIT_ARG_WEIGHT_FACTOR: Weight = 2;
const PTR_ARG_WEIGHT_FACTOR: Weight = 20;
const REF_ARG_WEIGHT_FACTOR: Weight = 20;
const OFFSETTED_PTR_WEIGHT_FACTOR: Weight = 10;
const ROUNDTRIPPED_PTR_WEIGHT_FACTOR: Weight = 100;

impl PlaceSelector {
    pub fn for_pointee(tcx: Rc<TyCtxt>, allow_uninit: bool) -> Self {
        Self {
            usage: PlaceUsage::Pointee,
            allow_uninit,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_operand(tcx: Rc<TyCtxt>) -> Self {
        Self {
            tys: None,
            size: None,
            usage: PlaceUsage::Operand,
            exclusions: vec![],
            allow_uninit: false,
            tcx,
            moved: vec![],
            refed: vec![],
        }
    }

    pub fn for_set_discriminant(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::SetDiscriminant,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_return_place(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::RET,
            ..Self::for_lhs(tcx)
        }
    }

    pub fn for_argument(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::Argument,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_lhs(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::LHS,
            allow_uninit: true,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_known_val(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::KnownVal,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_non_zero(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::NonZero,
            ..Self::for_operand(tcx)
        }
    }

    pub fn for_offsetee(tcx: Rc<TyCtxt>) -> Self {
        Self {
            usage: PlaceUsage::Offsetee,
            ..Self::for_operand(tcx)
        }
    }

    pub fn of_ty(self, ty: TyId) -> Self {
        let tys = Some(vec![ty]);
        Self { tys, ..self }
    }

    pub fn of_tys(self, types: &[TyId]) -> Self {
        let tys = Some(Vec::from(types));
        Self { tys, ..self }
    }

    pub fn of_size(self, size: Size) -> Self {
        Self {
            size: Some(size),
            ..self
        }
    }

    pub fn except(self, exclude: &Place) -> Self {
        let mut exclusions = self.exclusions;
        // TODO: More granular place discrimination
        exclusions.push(exclude.clone());
        Self { exclusions, ..self }
    }

    pub fn having_moved(self, place: PlaceIndex) -> Self {
        assert_eq!(self.usage, PlaceUsage::Argument);
        let mut moved = self.moved;
        moved.push(place.clone());
        Self { moved, ..self }
    }

    pub fn having_refed(self, target: PlaceIndex) -> Self {
        assert_eq!(self.usage, PlaceUsage::Argument);
        let mut refed = self.refed;
        refed.push(target.clone());
        Self { refed, ..self }
    }

    fn into_iter_path(self, pt: &PlaceGraph) -> impl Iterator<Item = PlacePath> + Clone + '_ {
        let exclusion_indicies: Vec<PlaceIndex> = self
            .exclusions
            .iter()
            .map(|place| place.to_place_index(pt).expect("excluded place exists"))
            .chain(pt.return_dest_stack()) // Don't touch anything that overlaps with any RET in the stack
            .chain(pt.moved_in_args_stack()) // Don't touch anything that overlaps with moved in args in the stack
            .collect();
        let moved: Vec<PlaceIndex> = self
            .moved
            .iter()
            .map(|place| place.to_place_index(pt).expect("place exists"))
            .collect();
        let refed: Vec<PlaceIndex> = self
            .refed
            .iter()
            .map(|place| place.to_place_index(pt).expect("place exists"))
            .collect();
        pt.reachable_nodes().filter(move |ppath| {
            let index = ppath.target_index();

            // Well-typedness
            if let Some(tys) = &self.tys
                && !tys.contains(&pt.ty(index))
            {
                return false;
            }

            // Liveness
            if !pt.is_place_live(index) {
                return false;
            }

            // Initness
            if !self.allow_uninit && !pt.is_place_init(index) {
                return false;
            };

            // Ref validity
            if self.usage != PlaceUsage::LHS && !pt.contains_only_valid_ref(index) {
                return false;
            }

            // Known val
            if self.usage == PlaceUsage::KnownVal && pt.known_val(index).is_none() {
                return false;
            }

            if self.usage == PlaceUsage::NonZero {
                let Some(known_val) = pt.known_val(index) else {
                    return false;
                };
                match known_val {
                    Literal::Uint(v, _) if *v != 0 => {}
                    Literal::Int(v, _) if *v != 0 => {}
                    Literal::Float(v, _) if *v != 0. => {}
                    _ => return false,
                }
            }

            // Avoid having ref in return type
            if self.usage == PlaceUsage::RET
                && pt.ty(index).contains(&self.tcx, |tcx, ty| ty.is_ref(tcx))
            {
                return false;
            }

            // Not excluded
            if exclusion_indicies
                .iter()
                .any(|excl| pt.overlap(index, excl))
            {
                return false;
            }

            // Has the right size
            if self.size.is_some() && BasicMemory::ty_size(pt.ty(index), &self.tcx) != self.size {
                return false;
            }

            // Check for aliasing rules
            match self.usage {
                // writes
                PlaceUsage::LHS | PlaceUsage::SetDiscriminant | PlaceUsage::RET => {
                    if !pt.can_write_through(ppath.source(), index) {
                        return false;
                    }
                }
                // reads that will be done as moves
                PlaceUsage::Operand | PlaceUsage::Argument if !pt.ty(index).is_copy(&self.tcx) => {
                    if !pt.can_write_through(ppath.source(), index) {
                        return false;
                    }
                }
                // reads
                _ => {
                    if !pt.can_read_through(ppath.source(), index) {
                        return false;
                    }
                }
            }

            // Function arguments
            if self.usage == PlaceUsage::Argument {
                if moved.iter().any(|excl| pt.overlap(index, excl)) {
                    return false;
                }
                // If this contains a ref, then it cannot point to an already-picked moved arg
                for m in &moved {
                    if pt.contains_ref_to(index, *m) {
                        return false;
                    }
                }

                if !pt.ty(index).is_copy(&self.tcx) {
                    // If this is a type that must be moved, then it must not be referenced by an already-picked reference
                    for r in &refed {
                        if pt.contains_ref_to(*r, index) {
                            return false;
                        }
                    }
                }
            }

            true
        })
    }

    pub fn into_weighted(self, pt: &PlaceGraph) -> Option<(Vec<PlacePath>, WeightedIndex<Weight>)> {
        let usage = self.usage;
        let tcx = self.tcx.clone();
        let (places, weights): (Vec<PlacePath>, Vec<Weight>) =
            self.into_iter_path(pt)
                .map(|ppath| {
                    let place = ppath.target_index();
                    let mut weight = match usage {
                        PlaceUsage::Argument => {
                            let mut weight = 1;
                            let index = ppath.target_index();
                            let ty = pt.ty(index);
                            if ty.contains(&tcx, |tcx, ty| ty.is_ref(tcx)) {
                                weight *= REF_ARG_WEIGHT_FACTOR;
                            }
                            if ty.contains(&tcx, |tcx, ty| ty.is_raw_ptr(tcx)) {
                                weight *= PTR_ARG_WEIGHT_FACTOR;
                            }
                            if pt.known_val(index).is_some() {
                                weight *= LIT_ARG_WEIGHT_FACTOR;
                            }
                            // Encourage isize for pointer offset
                            if ty.contains(&tcx, |_, ty| ty == TyCtxt::ISIZE) {
                                weight *= LIT_ARG_WEIGHT_FACTOR;
                            }
                            if ty.is_raw_ptr(&tcx) && pt.offseted(index) {
                                weight *= OFFSETTED_PTR_WEIGHT_FACTOR;
                            }
                            weight
                        }
                        PlaceUsage::LHS | PlaceUsage::SetDiscriminant | PlaceUsage::RET => {
                            let mut weight = if !pt.is_place_init(place) {
                                if ppath.is_return_proj(pt) {
                                    RET_LHS_WEIGHT_FACTOR
                                } else {
                                    UNINIT_WEIGHT_FACTOR
                                }
                            } else {
                                1
                            };
                            if pt.ty(place).is_raw_ptr(&tcx) && pt.get_offset(place).is_some() {
                                weight = 0;
                            }
                            weight
                        }
                        PlaceUsage::Operand => pt.get_complexity(place),
                        PlaceUsage::Pointee => 1,
                        PlaceUsage::KnownVal | PlaceUsage::NonZero => pt.get_complexity(place),
                        PlaceUsage::Offsetee => 1,
                    };

                    if ppath.projections(pt).any(|proj| proj.is_deref()) {
                        weight *= DEREF_WEIGHT_FACTOR;
                    }

                    if ppath.nodes(pt).any(|place| {
                        pt.ty(place).is_raw_ptr(&tcx) && pt.has_offset_roundtripped(place)
                    }) {
                        weight *= ROUNDTRIPPED_PTR_WEIGHT_FACTOR;
                    }

                    (ppath, weight)
                })
                .unzip();
        if let Ok(weighted_index) = WeightedIndex::new(weights) {
            Some((places, weighted_index))
        } else {
            None
        }
    }

    pub fn into_iter_place<'a>(
        self,
        pt: &'a PlaceGraph,
        local_decls: &'a LocalDecls,
    ) -> impl Iterator<Item = Place> + Clone + 'a {
        self.into_iter_path(pt)
            .map(move |ppath| ppath.to_place(pt, local_decls))
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use std::rc::Rc;

    use config::TyConfig;
    use mir::{
        syntax::{Local, LocalDecl, LocalDecls, Place},
        tyctxt::TyCtxt,
    };
    use rand::{
        Rng, SeedableRng,
        rngs::SmallRng,
        seq::{IndexedRandom, IteratorRandom},
    };
    use test::Bencher;

    use crate::{
        pgraph::PlaceGraph,
        ty::{TySelect, seed_tys},
    };

    use super::PlaceSelector;

    fn build_pt(rng: &mut impl Rng) -> (PlaceGraph, Rc<TyCtxt>, LocalDecls) {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        seed_tys(&mut tcx, rng);
        let tcx = Rc::new(tcx);
        let mut pt = PlaceGraph::new(tcx.clone());
        let ty_weights = TySelect::new(&tcx);
        let mut decls = LocalDecls::new();
        for i in 0..=32 {
            let ty = ty_weights.choose_ty(rng, &tcx);
            decls.push(LocalDecl::new_mut(ty));
            let pidx = pt.allocate_local(Local::new(i), ty);
            if i % 2 == 0 {
                pt.mark_place_init(pidx);
            }
        }
        (pt, tcx, decls)
    }

    #[bench]
    fn bench_select(b: &mut Bencher) {
        let mut rng = SmallRng::seed_from_u64(0);
        let (pt, tcx, decls) = build_pt(&mut rng);

        b.iter(|| {
            PlaceSelector::for_lhs(tcx.clone())
                .except(&Place::RETURN_SLOT)
                .into_iter_place(&pt, &decls)
                .choose(&mut rng)
                .expect("places not empty");
        })
    }

    #[bench]
    fn bench_materialise_into_vec(b: &mut Bencher) {
        let mut rng = SmallRng::seed_from_u64(0);
        let (pt, tcx, decls) = build_pt(&mut rng);

        b.iter(|| {
            let places: Vec<Place> = PlaceSelector::for_lhs(tcx.clone())
                .except(&Place::RETURN_SLOT)
                .into_iter_place(&pt, &decls)
                .collect();

            // places.choose(&mut rng).expect("not empty");
            places
                .choose_weighted(&mut rng, |p| p.projection().len())
                .expect("places not empty");
        })
    }
}
