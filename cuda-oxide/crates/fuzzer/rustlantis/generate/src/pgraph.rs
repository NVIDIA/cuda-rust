use std::{
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet},
    iter,
    rc::Rc,
    vec,
};

use bimap::BiHashMap;
use index_vec::IndexVec;
use mir::{
    syntax::{
        Body, FieldIdx, Literal, Local, LocalDecls, Mutability, Operand, Place, ProjectionElem,
        Rvalue, TyId, TyKind, UintTy, VariantIdx,
    },
    tyctxt::TyCtxt,
};
use petgraph::{
    Direction, prelude::EdgeIndex, prelude::NodeIndex, stable_graph::StableGraph, visit::EdgeRef,
};
use smallvec::{SmallVec, smallvec};

use crate::mem::{
    AbstractByte, AllocId, AllocationBuilder, BasicMemory, BorrowType, RunPointer, Tag,
};

type Graph = StableGraph<PlaceNode, ProjectionElem>;
pub type PlaceIndex = NodeIndex;
pub type ProjectionIndex = EdgeIndex;
pub type Path = SmallVec<[ProjectionIndex; 4]>;

pub const MAX_COMPLEXITY: usize = 100;

#[derive(Clone)]
struct Frame {
    locals: BiHashMap<Local, PlaceIndex>,

    // To maintain iteration order and determinism
    locals_ordered: BinaryHeap<PlaceIndex>,

    // return destination and moved in places cannot be read or written
    // while the frame is on stack
    return_destination: PlaceIndex,
    moved_in: SmallVec<[PlaceIndex; 4]>,
}

impl Frame {
    fn new(dest: PlaceIndex, moved_in: impl Iterator<Item = PlaceIndex>) -> Self {
        Self {
            locals: BiHashMap::new(),
            locals_ordered: BinaryHeap::new(),
            return_destination: dest,
            moved_in: SmallVec::from_iter(moved_in),
        }
    }

    fn add_local(&mut self, local: Local, pidx: PlaceIndex) {
        self.locals
            .insert_no_overwrite(local, pidx)
            .expect("did not insert existing local or place");
        self.locals_ordered.push(pidx);
    }

    fn get_by_index(&self, pidx: PlaceIndex) -> Option<Local> {
        self.locals.get_by_right(&pidx).copied()
    }

    fn get_by_local(&self, local: Local) -> Option<PlaceIndex> {
        self.locals.get_by_left(&local).copied()
    }
}

#[derive(Clone, Copy)]
pub enum PlaceOperand {
    Copy(PlaceIndex),
    Move(PlaceIndex),
    Constant(Literal),
}

impl PlaceOperand {
    pub fn from_operand(op: &Operand, pt: &PlaceGraph) -> Self {
        match op {
            Operand::Copy(place) => {
                PlaceOperand::Copy(place.to_place_index(pt).expect("arg exists"))
            }
            Operand::Move(place) => {
                let index = place.to_place_index(pt).expect("arg exists");
                PlaceOperand::Move(index)
            }
            Operand::Constant(lit) => PlaceOperand::Constant(*lit),
        }
    }
}

/// A data structure keeping track of all _syntactically expressible places_ in the program.
#[derive(Clone)]
pub struct PlaceGraph {
    /// The callstack
    frames: Vec<Frame>,
    index_candidates: HashMap<usize, SmallVec<[Local; 1]>>,
    pointer_tags: IndexVec<Tag, BTreeSet<PlaceIndex>>,

    places: Graph,
    memory: BasicMemory,
    tcx: Rc<TyCtxt>,
}

#[derive(Debug, Clone)]
pub struct PlaceNode {
    pub ty: TyId,
    alloc_id: AllocId,
    complexity: usize,

    subfields: Vec<PlaceIndex>,
    subfield_edges: Vec<EdgeIndex>,

    // Only Tys fitting into a single Run have these
    run_ptr: Option<RunPointer>,

    // Remember the value of simple literals
    val: Option<Literal>,

    // Offsetted raw pointer value
    offset: Option<isize>,

    // For enum types, the currently active variant
    active_variant: Option<VariantIdx>,

    // Tags of raw pointer or references
    tag: Option<Tag>,
}

pub trait ToPlaceIndex {
    fn to_place_index(&self, pt: &PlaceGraph) -> Option<PlaceIndex>;
}

impl ToPlaceIndex for Place {
    fn to_place_index(&self, pt: &PlaceGraph) -> Option<PlaceIndex> {
        pt.get_node(self)
    }
}

impl ToPlaceIndex for Local {
    fn to_place_index(&self, pt: &PlaceGraph) -> Option<PlaceIndex> {
        pt.get_node(&Place::from_local(*self))
    }
}

impl ToPlaceIndex for PlaceIndex {
    fn to_place_index(&self, _: &PlaceGraph) -> Option<PlaceIndex> {
        Some(*self)
    }
}

impl<T> ToPlaceIndex for &T
where
    T: ToPlaceIndex,
{
    fn to_place_index(&self, pt: &PlaceGraph) -> Option<PlaceIndex> {
        (*self).to_place_index(pt)
    }
}

/// VisitAction specifies what to do next after visiting
/// each node in {update, visit}_transitive_{sub, super}fields
#[derive(PartialEq, Eq, Clone, Copy)]
enum VisitAction {
    // Keep going in the current branch
    Continue,
    // Stop going in the current branch, but keep going in others
    Stop,
    // Stop going in all branches
    ShortCircuit,
}

impl PlaceGraph {
    pub fn new(tcx: Rc<TyCtxt>) -> Self {
        Self {
            frames: vec![Frame::new(
                /* fn0 dummy */ PlaceIndex::new(usize::MAX),
                iter::empty(),
            )],
            index_candidates: HashMap::new(),
            pointer_tags: IndexVec::new(),
            places: StableGraph::default(),
            memory: BasicMemory::new(),
            tcx,
        }
    }

    fn current_frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("call stack isn't empty")
    }

    fn current_frame(&self) -> &Frame {
        self.frames.last().expect("call stack isn't empty")
    }

    pub fn enter_fn0(&mut self, body: &Body) {
        // Declare return place
        self.allocate_local(Local::RET, body.return_ty());
        // Declare args
        body.args_decl_iter().for_each(|(local, decl)| {
            let pidx = self.allocate_local(local, decl.ty);
            // encourage use of args
            self.update_complexity(pidx, 5);
        });
    }

    /// Returns whether the selection of function call arguments will not cause
    /// UB. Must be called while we are still in the Caller
    #[allow(dead_code)]
    pub fn arguments_ok(&self, args: &[Operand], return_dest: &Place) -> bool {
        // Check if the move of any places would invalidate any reference
        let mut ok = true;

        let mut invalidated: HashSet<Tag> = HashSet::new();
        // Find all tags that will be invalidated
        for arg in args
            .iter()
            .filter_map(|arg| {
                if let Operand::Move(p) = arg {
                    Some(p)
                } else {
                    None
                }
            })
            .chain([return_dest])
        {
            self.visit_transitive_subfields(
                arg.to_place_index(self).expect("place exists"),
                |pid| {
                    let node = &self.places[pid];
                    if let Some(run_ptr) = node.run_ptr {
                        invalidated.extend(self.memory.above_first_ref(run_ptr));
                        VisitAction::Stop
                    } else {
                        VisitAction::Continue
                    }
                },
            );
        }
        // Find if any references uses an invalidated tag
        for arg in args {
            match arg {
                Operand::Copy(p) | Operand::Move(p) => {
                    self.visit_transitive_subfields(
                        p.to_place_index(self).expect("place exists"),
                        |pid| {
                            let node = &self.places[pid];
                            if node.ty.is_ref(&self.tcx) {
                                let tag = node.tag.expect("has tag");
                                if invalidated.contains(&tag) {
                                    ok = false;
                                    return VisitAction::ShortCircuit;
                                }
                            }
                            VisitAction::Continue
                        },
                    );
                }
                _ => {}
            }
        }
        ok
    }

    pub fn enter_fn(&mut self, body: &Body, args: &[Operand], return_dest: &Place) {
        // Get the PlaceIndices before frame switch

        let args: Vec<PlaceOperand> = args
            .iter()
            .map(|p| PlaceOperand::from_operand(p, self))
            .collect();

        let return_dest = return_dest
            .to_place_index(self)
            .expect("return dest exists");
        self.assign_literal(return_dest, None);

        let moved_in = args.iter().filter_map(|arg| match *arg {
            PlaceOperand::Move(pidx) => Some(pidx),
            _ => None,
        });

        // Frame switch
        self.frames.push(Frame::new(return_dest, moved_in));
        self.index_candidates.clear();

        self.allocate_local(Local::RET, body.return_ty());
        body.args_decl_iter()
            .zip(args)
            .for_each(|((local, decl), arg)| {
                let pidx = self.allocate_local(local, decl.ty);

                match &arg {
                    PlaceOperand::Copy(source_pidx) | PlaceOperand::Move(source_pidx) => {
                        debug_assert!(
                            self.is_place_init(source_pidx),
                            "function arguments must be init: arg {local:?} source {source_pidx:?}"
                        );
                        self.copy_place(pidx, source_pidx);
                        for r in self.refs_in(pidx) {
                            self.mark_ref_protected(r);
                        }
                    }
                    PlaceOperand::Constant(lit) => self.assign_literal(pidx, Some(*lit)),
                }
                if let PlaceOperand::Move(source_pidx) = arg {
                    self.mark_place_moved(source_pidx);
                }
            });
    }

    /// Checks if the value in RET would be valid upon return
    pub fn can_return(&self) -> bool {
        if !self.is_place_init(Local::RET) {
            return false;
        }

        let local_allocs: HashSet<AllocId> = self
            .current_frame()
            .locals_ordered
            .iter()
            .map(|local| self.places[*local].alloc_id)
            .collect();
        let mut has_invalid_ref = false;
        // Check if it contains any references that will be invalidated upon return
        self.visit_transitive_subfields(
            Local::RET.to_place_index(self).expect("ret exists"),
            |node| {
                if self.ty(node).is_ref(&self.tcx) {
                    if !self.is_ref_valid(node) {
                        has_invalid_ref = true;
                        return VisitAction::ShortCircuit;
                    }
                    if let Some(pointee) = self.pointee(node) {
                        if local_allocs.contains(&self.places[pointee].alloc_id) {
                            // Will dangle after return
                            has_invalid_ref = true;
                            return VisitAction::ShortCircuit;
                        }
                    }
                    return VisitAction::Stop;
                }
                VisitAction::Continue
            },
        );
        !has_invalid_ref
    }

    pub fn exit_fn(&mut self) {
        // FIXME: this is quite flimsy wrt. statement order
        let callee_ret = Place::RETURN_SLOT
            .to_place_index(self)
            .expect("place exists");
        // Frame switch
        let old_frame = self.frames.pop().expect("call stack isn't empty");
        self.index_candidates.clear(); // Invalidate cache

        // Copy ret
        self.copy_place(old_frame.return_destination, callee_ret);

        // Remove ref edges into places that are about to be deallocated.
        // This is necessary to prevent dangling references.
        let mut ref_edges = vec![];
        let mut alloc_ids = vec![];

        for pidx in old_frame.locals.right_values() {
            self.visit_transitive_subfields(*pidx, |node| {
                ref_edges.extend(self.pointers_to(node).iter().map(|(_, edge)| edge));
                VisitAction::Continue
            });
            alloc_ids.push(self.places[*pidx].alloc_id);
        }

        for edge in ref_edges {
            self.remove_edge(edge);
        }

        // Deallocate places.
        for alloc_id in alloc_ids {
            self.memory.deallocate(alloc_id);
        }
    }

    pub fn allocate_local(&mut self, local: Local, ty: TyId) -> PlaceIndex {
        let mut pidx = Default::default();
        self.memory.allocate_with_builder(|builder| {
            pidx = Self::add_place(&mut self.places, ty, &self.tcx, builder, None);
        });
        self.current_frame_mut().add_local(local, pidx);
        pidx
    }

    pub fn deallocate_local(&mut self, local: Local) {
        // FIXME: should we need to remove local from the frame?
        let pidx = local.to_place_index(self).expect("place exists");
        self.memory.deallocate(self.places[pidx].alloc_id);
    }

    fn add_place(
        places: &mut Graph,
        ty: TyId,
        tcx: &TyCtxt,
        alloc_builder: &mut AllocationBuilder,
        run_ptr: Option<RunPointer>,
    ) -> PlaceIndex {
        let alloc_id = alloc_builder.alloc_id();
        let pidx = if run_ptr.is_some() {
            // If this is called recursively, and our parent (array) already allocated a run
            places.add_node(PlaceNode {
                ty,
                alloc_id,
                complexity: 0,
                subfields: vec![],
                subfield_edges: vec![],
                run_ptr,
                val: None,
                offset: None,
                active_variant: None,
                tag: None,
            })
        } else if let Some(size) = BasicMemory::ty_size(ty, tcx) {
            let run_and_offset = alloc_builder.new_run(size);
            places.add_node(PlaceNode {
                ty,
                alloc_id,
                complexity: 0,
                subfields: vec![],
                subfield_edges: vec![],
                run_ptr: Some(RunPointer {
                    alloc_id,
                    run_and_offset,
                    size,
                }),
                val: None,
                offset: None,
                active_variant: None,
                tag: None,
            })
        } else {
            places.add_node(PlaceNode {
                ty,
                alloc_id,
                complexity: 0,
                subfields: vec![],
                subfield_edges: vec![],
                run_ptr: None,
                val: None,
                offset: None,
                active_variant: None,
                tag: None,
            })
        };
        match ty.kind(tcx) {
            TyKind::Tuple(elems) => elems.iter().enumerate().for_each(|(idx, elem)| {
                let sub_pidx = Self::add_place(places, *elem, tcx, alloc_builder, None);
                let edge_idx = places.add_edge(
                    pidx,
                    sub_pidx,
                    ProjectionElem::TupleField(FieldIdx::new(idx)),
                );
                places[pidx].subfields.push(sub_pidx);
                places[pidx].subfield_edges.push(edge_idx);
            }),
            TyKind::Array(elem_ty, len) => {
                for i in 0..*len {
                    let child_run_ptr = if let Some(run_ptr) = places[pidx].run_ptr {
                        let child_size =
                            BasicMemory::ty_size(*elem_ty, tcx).expect("ty has fixed size");
                        Some(RunPointer {
                            alloc_id,
                            run_and_offset: run_ptr
                                .run_and_offset
                                .offset(i as isize * child_size.bytes() as isize),
                            size: child_size,
                        })
                    } else {
                        None
                    };
                    let elem_pidx =
                        Self::add_place(places, *elem_ty, tcx, alloc_builder, child_run_ptr);
                    let edge_idx = places.add_edge(
                        pidx,
                        elem_pidx,
                        ProjectionElem::ConstantIndex { offset: i as u64 },
                    );
                    places[pidx].subfields.push(elem_pidx);
                    places[pidx].subfield_edges.push(edge_idx);
                }
            }
            TyKind::Adt(adt) if adt.is_enum() => {
                for (vid, var) in adt.variants.iter_enumerated() {
                    for (fid, ty) in var.fields.iter_enumerated() {
                        let field_pidx = Self::add_place(places, *ty, tcx, alloc_builder, None);
                        let edge_idx = places.add_edge(
                            pidx,
                            field_pidx,
                            ProjectionElem::DowncastField(vid, fid, *ty),
                        );
                        places[pidx].subfields.push(field_pidx);
                        places[pidx].subfield_edges.push(edge_idx);
                    }
                }
            }
            TyKind::Adt(adt) => {
                let fields = &adt.variants.first().expect("adt is a struct").fields;
                for (fid, ty) in fields.iter_enumerated() {
                    let field_pidx = Self::add_place(places, *ty, tcx, alloc_builder, None);
                    let edge_idx = places.add_edge(pidx, field_pidx, ProjectionElem::Field(fid));
                    places[pidx].subfields.push(field_pidx);
                    places[pidx].subfield_edges.push(edge_idx);
                }
            }
            TyKind::RawPtr(..) => { /* pointer has no subfields  */ }
            _ => { /* primitives, no projection */ }
        }
        pidx
    }

    pub fn transmute_place(&mut self, dst: impl ToPlaceIndex, src: impl ToPlaceIndex) {
        let dst = dst.to_place_index(self).expect("place exists");
        let src = src.to_place_index(self).expect("place exists");
        if dst == src {
            return;
        }
        self.update_complexity(dst, self.places[src].complexity);
        self.assign_literal(dst, None);
        let (dst_node, src_node) = self.places.index_twice_mut(dst, src);
        self.memory.copy(
            dst_node.run_ptr.expect("dst is packed"),
            src_node.run_ptr.expect("src is packed"),
        );

        if dst_node.ty.is_any_ptr(&self.tcx) {
            todo!()
        }
    }

    /// Assigns a discriminant to an enum-typed place, and invalidates all variant projections
    /// of the enum. If the old and new discriminant are equal, all variants are still invalidated.
    /// This is not necessary if the discriminant assignment originates from a SetDiscriminant statement,
    /// but it's needed if it originates from assigning a whole enum from another enum.
    pub fn assign_discriminant(&mut self, p: impl ToPlaceIndex, discriminant: Option<VariantIdx>) {
        let p = p.to_place_index(self).expect("place exists");
        assert!(self.ty(p).kind(&self.tcx).is_enum());
        self.places[p].active_variant = discriminant;

        // All inactive variant fields are uninit
        let invalidated = self.places[p].subfields.clone();

        for pidx in invalidated {
            self.invalidate_place(pidx);
        }
    }

    /// Invalidate place marks the place as uninit, it additionally removes any deref edges *into* the place + transitive subplaces
    fn invalidate_place(&mut self, p: impl ToPlaceIndex) {
        let pidx = p.to_place_index(self).unwrap();

        self.update_transitive_subfields(pidx, |this, place| {
            this.places[place].active_variant = None;
            let node = &this.places[place];
            if let Some(run_ptr) = node.run_ptr {
                this.memory.fill(run_ptr, AbstractByte::Uninit);
            }
            let refs: Vec<ProjectionIndex> = this
                .places
                .edges_directed(place, Direction::Outgoing)
                .chain(this.places.edges_directed(place, Direction::Incoming))
                .filter_map(|e| (e.weight() == &ProjectionElem::Deref).then_some(e.id()))
                .collect();
            for edge in refs {
                this.remove_edge(edge);
            }
            VisitAction::Continue
        });
    }

    pub fn copy_place(&mut self, dst: impl ToPlaceIndex, src: impl ToPlaceIndex) {
        let dst = dst.to_place_index(self).expect("place exists");
        let src = src.to_place_index(self).expect("place exists");
        if dst == src {
            return;
        }
        self.update_complexity(dst, self.places[src].complexity);
        self.assign_literal(dst, self.places[src].val);

        if self.places[dst].ty.kind(&self.tcx).is_enum() {
            self.assign_discriminant(dst, self.places[src].active_variant);
        }

        let (dst_node, src_node) = self.places.index_twice_mut(dst, src);
        assert_eq!(dst_node.ty, src_node.ty);

        if let Some(run_ptr) = src_node.run_ptr {
            self.memory
                .copy(dst_node.run_ptr.expect("dst is packed"), run_ptr);
        }

        if dst_node.ty.is_any_ptr(&self.tcx) {
            if let Some(pointee) = self.pointee(src) {
                self.set_ref(dst, pointee, Some(src));
            }
            if let Some(old) = self.ref_edge(dst) {
                self.remove_edge(old);
            }
            self.places[dst].offset = self.places[src].offset;
        }

        let projs: Vec<_> = self.places[dst].subfield_edges.iter().copied().collect();
        for eidx in projs {
            let proj = self.places[eidx];
            let new_dst = self
                .project_from_node(dst, proj)
                .expect("projection exists");
            let new_src = self
                .project_from_node(src, proj)
                .expect("projection exists");
            self.copy_place(new_dst, new_src);
        }
    }

    pub fn project_from_node(
        &self,
        pidx: PlaceIndex,
        mut proj: ProjectionElem,
    ) -> Option<PlaceIndex> {
        if let ProjectionElem::Index(local) = proj {
            let Some(Literal::Uint(i, UintTy::Usize)) = self.known_val(local) else {
                panic!("projection has a usize knownval");
            };
            proj = ProjectionElem::ConstantIndex { offset: *i as u64 };
        }
        self.places
            .edges_directed(pidx, Direction::Outgoing)
            .find(|edge| edge.weight() == &proj)
            .map(|e| e.target())
    }

    /// Get PlaceIndex from a Place
    fn get_node(&self, place: &Place) -> Option<PlaceIndex> {
        let mut node = self.current_frame().get_by_local(place.local())?;
        let proj_iter = place.projection().iter();
        for proj in proj_iter {
            let next = self.project_from_node(node, *proj);
            if let Some(next) = next {
                node = next;
            } else {
                return None;
            }
        }
        Some(node)
    }

    /// Call update on all transitive superfields of start, *excluding* start
    fn update_transitive_superfields<F>(&mut self, start: PlaceIndex, mut visit: F)
    where
        F: FnMut(&mut Self, PlaceIndex) -> VisitAction,
    {
        let mut to_visit: Vec<NodeIndex> = self.immediate_superfields(start).collect();
        while let Some(node) = to_visit.pop() {
            let todo = visit(self, node);
            match todo {
                VisitAction::Continue => to_visit.extend(self.immediate_superfields(node)),
                // These two should be the same as there is only one branch
                VisitAction::Stop => {}
                VisitAction::ShortCircuit => break,
            }
        }
    }

    /// Call visit on all transitive subfields of start, *including* start
    fn update_transitive_subfields<F>(&mut self, start: PlaceIndex, mut visit: F)
    where
        F: FnMut(&mut Self, PlaceIndex) -> VisitAction,
    {
        let mut to_visit: SmallVec<[PlaceIndex; 10]> = SmallVec::with_capacity(10);
        to_visit.push(start);
        while let Some(node) = to_visit.pop() {
            let todo = visit(self, node);
            match todo {
                VisitAction::Continue => to_visit.extend(self.immediate_subfields(node)),
                VisitAction::Stop => {}
                VisitAction::ShortCircuit => break,
            }
        }
    }

    fn visit_transitive_subfields<F>(&self, start: PlaceIndex, mut visit: F)
    where
        F: FnMut(PlaceIndex) -> VisitAction,
    {
        let mut to_visit: SmallVec<[PlaceIndex; 10]> = SmallVec::with_capacity(10);
        to_visit.push(start);
        while let Some(node) = to_visit.pop() {
            let todo = visit(node);
            match todo {
                VisitAction::Continue => to_visit.extend(self.immediate_subfields(node)),
                VisitAction::Stop => {}
                VisitAction::ShortCircuit => break,
            }
        }
    }

    pub fn update_complexity(&mut self, target: impl ToPlaceIndex, new_flow: usize) {
        let target = target.to_place_index(self).expect("place exists");
        let new_flow = new_flow.min(MAX_COMPLEXITY);

        // Subplaces' complexity is overwritten as target's new complexity
        self.update_transitive_subfields(target, |this, place| {
            this.places[place].complexity = new_flow;
            VisitAction::Continue
        });

        // Superplaces' complexity is updated to be the max of its children
        self.update_transitive_superfields(target, |this, place| {
            if let Some(max) = this
                .immediate_subfields(place)
                .map(|sub| this.places[sub].complexity)
                .max()
            {
                this.places[place].complexity = max;
            }
            VisitAction::Continue
        })
    }

    pub fn get_complexity(&self, p: impl ToPlaceIndex) -> usize {
        let pidx = p.to_place_index(self).unwrap();
        let node = &self.places[pidx];
        if node.ty.is_any_ptr(&self.tcx) {
            if let Some(pointee) = self.project_from_node(pidx, ProjectionElem::Deref) {
                self.get_complexity(pointee)
            } else {
                // Use the initial complexity
                node.complexity
            }
        } else {
            node.complexity
        }
    }

    fn mark_ref_protected(&mut self, p: impl ToPlaceIndex) {
        let p = p.to_place_index(&self).expect("place exists");
        assert!(self.ty(p).is_ref(&self.tcx));
        let run = self.places[p].run_ptr.expect("has run");
        self.memory
            .mark_protected(run, self.places[p].tag.expect("has tag"));
    }

    pub fn mark_place_moved(&mut self, p: impl ToPlaceIndex) {
        let p = p.to_place_index(&self).expect("place exists");
        self.mark_place_uninit(p);
        // We assume that we never move through a pointer
        self.place_written(p, None);
    }

    pub fn mark_place_uninit(&mut self, p: impl ToPlaceIndex) {
        let pidx = p.to_place_index(self).unwrap();

        // Remove Deref edges owned by the value being invalidated, including pointers nested in
        // aggregates. Incoming Deref edges from external pointers are intentionally preserved.
        let mut ref_edges = vec![];
        self.visit_transitive_subfields(pidx, |place| {
            if self.places[place].ty.is_any_ptr(&self.tcx)
                && let Some(edge) = self.ref_edge(place)
            {
                ref_edges.push(edge);
            }
            VisitAction::Continue
        });
        for edge in ref_edges {
            self.remove_edge(edge);
        }

        self.update_transitive_subfields(pidx, |this, place| {
            this.places[place].active_variant = None;
            let node = &this.places[place];
            if let Some(run_ptr) = node.run_ptr {
                this.memory.fill(run_ptr, AbstractByte::Uninit);
                VisitAction::Stop
            } else {
                VisitAction::Continue
            }
        });
    }

    /// Whether a reference value is legal to produce
    fn is_ref_valid(&self, r: PlaceIndex) -> bool {
        assert!(self.ty(r).is_ref(&self.tcx));
        let Some(target) = self.pointee(r) else {
            return false;
        };
        self.is_place_init(target)
    }

    /// Whether all refs contained in a place are all valid
    pub fn contains_only_valid_ref(&self, p: PlaceIndex) -> bool {
        let mut has_invalid = false;
        self.visit_transitive_subfields(p, |node| {
            if self.ty(node).is_ref(&self.tcx) && !self.is_ref_valid(node) {
                has_invalid = true;
                return VisitAction::ShortCircuit;
            }
            VisitAction::Continue
        });
        !has_invalid
    }

    pub fn mark_place_init(&mut self, p: impl ToPlaceIndex) {
        let pidx = p.to_place_index(self).unwrap();
        self.update_transitive_subfields(pidx, |this, place| {
            let node = &this.places[place];
            if let Some(run_ptr) = node.run_ptr {
                this.memory.fill(run_ptr, AbstractByte::Init);
                VisitAction::Stop
            } else {
                VisitAction::Continue
            }
        });
    }

    /// Returns all subfields that are reference type in a Place
    pub fn refs_in(&self, p: impl ToPlaceIndex) -> Vec<PlaceIndex> {
        self.subfields(p)
            .iter()
            .filter(|pidx| self.ty(pidx).is_ref(&self.tcx))
            .copied()
            .collect()
    }

    /// Whether a place contains a reference to a place and its subfields
    pub fn contains_ref_to(&self, r: impl ToPlaceIndex, dest: PlaceIndex) -> bool {
        let r = r.to_place_index(self).expect("place exists");
        let mut contains = false;
        self.visit_transitive_subfields(r.to_place_index(self).expect("place exists"), |pid| {
            if self.ty(pid).is_ref(&self.tcx) {
                if let Some(pointee) = self.pointee(pid)
                    && self.overlap(pointee, dest)
                {
                    contains = true;
                    return VisitAction::ShortCircuit;
                }
            }
            VisitAction::Continue
        });
        contains
    }

    /// Returns all pointers to a pointee, and the ref edge
    fn pointers_to(&self, pointee: PlaceIndex) -> Vec<(NodeIndex, ProjectionIndex)> {
        self.places
            .edges_directed(pointee, Direction::Incoming)
            .filter_map(|edge| {
                edge.weight()
                    .is_deref()
                    .then_some((edge.source(), edge.id()))
            })
            .collect()
    }

    /// Returns the pointee in pointer -[Deref]-> pointee, if one exists
    pub fn pointee(&self, pointer: PlaceIndex) -> Option<PlaceIndex> {
        assert!(self.places[pointer].ty.is_any_ptr(&self.tcx));
        self.places
            .edges_directed(pointer, Direction::Outgoing)
            .next()
            .map(|deref| deref.target())
    }

    /// Returns the edge in pointer -[edge: Deref]-> pointee, if one exists
    fn ref_edge(&self, pointer: PlaceIndex) -> Option<ProjectionIndex> {
        assert!(self.places[pointer].ty.is_any_ptr(&self.tcx));
        self.places
            .edges_directed(pointer, Direction::Outgoing)
            .next()
            .map(|deref| deref.id())
    }

    /// Creates an edge pointer -[Deref]-> pointee
    pub fn set_ref(
        &mut self,
        pointer: impl ToPlaceIndex,
        pointee: impl ToPlaceIndex,
        copied_from: Option<PlaceIndex>,
    ) {
        let pointer = pointer.to_place_index(self).expect("place exists");
        let pointee = pointee.to_place_index(self).expect("place exists");

        assert_eq!(
            self.places[pointer].ty.pointee_ty(&self.tcx).unwrap(),
            self.places[pointee].ty
        );

        let ref_type = match self.ty(pointer).kind(&self.tcx) {
            TyKind::RawPtr(_, _) => BorrowType::Raw,
            TyKind::Ref(_, Mutability::Mut) => BorrowType::Exclusive,
            TyKind::Ref(_, Mutability::Not) => BorrowType::Shared,
            _ => panic!("source must be of pointer type"),
        };

        if matches!(ref_type, BorrowType::Shared | BorrowType::Exclusive) {
            assert!(self.is_place_init(pointee));
        }

        if let Some(old) = self.ref_edge(pointer) {
            self.remove_edge(old);
        }

        self.places[pointer].offset = None;
        self.update_complexity(pointer, self.places[pointee].complexity);

        // Add new ref edge
        self.places
            .add_edge(pointer, pointee, ProjectionElem::Deref);

        if let Some(copied_from) = copied_from {
            let tag = self.places[copied_from].tag.expect("has tag");
            self.places[pointer].tag = Some(tag);
            self.pointer_tags[tag].insert(pointer);
        } else {
            let tag = self.pointer_tags.push(BTreeSet::from([pointer]));
            self.places[pointer].tag = Some(tag);
            self.update_transitive_subfields(pointee, |this, place| {
                if let Some(run) = this.places[place].run_ptr {
                    this.memory.add_ref(run, ref_type, tag);
                    VisitAction::Stop
                } else {
                    VisitAction::Continue
                }
            });
        }

        assert_eq!(
            self.places
                .edges_directed(pointer, Direction::Outgoing)
                .count(),
            1
        );
    }

    pub fn is_place_live(&self, p: impl ToPlaceIndex) -> bool {
        let Some(pidx) = p.to_place_index(self) else {
            return false;
        };
        let node = &self.places[pidx];
        self.memory.is_live(node.alloc_id)
    }

    pub fn is_place_init(&self, p: impl ToPlaceIndex) -> bool {
        if !self.is_place_live(&p) {
            return false;
        }
        let pidx = p.to_place_index(self).unwrap();
        let node = &self.places[pidx];
        if let Some(run_ptr) = node.run_ptr {
            // Leaf
            self.memory.bytes(run_ptr).iter().all(|b| b.is_init())
        } else if node.ty.kind(&self.tcx).is_enum() && node.active_variant.is_none() {
            // Uninit enum
            false
        } else {
            self.places[pidx]
                .subfield_edges
                .iter()
                .enumerate()
                .filter_map(|(i, &eidx)| {
                    if let ProjectionElem::DowncastField(vid, ..) = self.places[eidx]
                        && self.places[pidx].active_variant != Some(vid)
                    {
                        // Only check active variant
                        None
                    } else {
                        Some(self.places[pidx].subfields[i])
                    }
                })
                .all(|sub| self.is_place_init(sub))
        }
    }

    fn immediate_subfields(&self, pidx: PlaceIndex) -> impl Iterator<Item = PlaceIndex> + '_ {
        self.places[pidx].subfields.iter().copied()
    }

    fn immediate_superfields(&self, pidx: PlaceIndex) -> impl Iterator<Item = PlaceIndex> + '_ {
        self.places
            .edges_directed(pidx, Direction::Incoming)
            .filter_map(|e| (!e.weight().is_deref()).then_some(e.source()))
    }

    // Returns an iterator over all places reachable from node through projections
    fn reachable_from_node(&self, pidx: PlaceIndex) -> ProjectionIter<'_> {
        ProjectionIter::new(self, pidx)
    }

    pub fn reachable_nodes(&self) -> impl Iterator<Item = PlacePath> + Clone + '_ {
        let local_iter = self.current_frame().locals_ordered.iter();
        local_iter.flat_map(|&pidx| self.reachable_from_node(pidx))
    }

    /// Returns all transitive subfields of a place
    fn subfields(&self, p: impl ToPlaceIndex) -> Vec<PlaceIndex> {
        let mut subs: Vec<PlaceIndex> = vec![];
        self.visit_transitive_subfields(p.to_place_index(self).expect("place exists"), |sub| {
            subs.push(sub);
            VisitAction::Continue
        });
        subs
    }

    /// Whether two places overlap or alias
    #[inline(always)]
    pub fn overlap(&self, a: impl ToPlaceIndex, b: impl ToPlaceIndex) -> bool {
        let a = a.to_place_index(self).expect("place exists");
        let b = b.to_place_index(self).expect("place exists");

        if a == b {
            return true;
        }

        if self.places[a].alloc_id != self.places[b].alloc_id {
            return false;
        }

        let a_sub = self.subfields(a);
        let b_sub: HashSet<PlaceIndex> = self.subfields(b).into_iter().collect();

        a_sub.into_iter().any(|a_node| b_sub.contains(&a_node))
    }

    pub fn assign_literal(&mut self, p: impl ToPlaceIndex, val: Option<Literal>) {
        let p = p.to_place_index(self).expect("place exists");
        if let Some(local) = self.current_frame().get_by_index(p) {
            // If place is a local
            if let Some(&Literal::Uint(i, UintTy::Usize)) = self.known_val(p)
                && let Some(old) = self.index_candidates.get_mut(&(i as usize))
                && let Some(to_remove) = old.iter().position(|&l| l == local)
            {
                // unconditionally remove the old entry if it exists
                old.remove(to_remove);
            }
            if let Some(Literal::Uint(i, UintTy::Usize)) = val {
                self.index_candidates
                    .entry(i as usize)
                    .or_default()
                    .push(local)
            }
        }

        if let Some(val) = val {
            self.places[p].val = Some(val);
        } else {
            self.update_transitive_subfields(p, |this, node| {
                this.places[node].val = None;
                VisitAction::Continue
            });
            self.update_transitive_superfields(p, |this, node| {
                this.places[node].val = None;
                VisitAction::Continue
            });
        }
    }

    /// Return destinations of all currently active frames
    pub fn return_dest_stack(&self) -> impl Iterator<Item = PlaceIndex> + '_ {
        // Skip fn0 which is a dummy
        self.frames.iter().skip(1).map(|f| f.return_destination)
    }

    /// Moved in places of all currently active frames
    pub fn moved_in_args_stack(&self) -> impl Iterator<Item = PlaceIndex> + '_ {
        // Skip fn0 which is a dummy
        self.frames
            .iter()
            .skip(1)
            .flat_map(|f| f.moved_in.iter().copied())
    }

    pub fn ty(&self, p: impl ToPlaceIndex) -> TyId {
        self.places[p.to_place_index(self).expect("place exists")].ty
    }

    pub fn known_val(&self, p: impl ToPlaceIndex) -> Option<&Literal> {
        self.places[p.to_place_index(self).expect("place exists")]
            .val
            .as_ref()
    }

    pub fn known_variant(&self, p: impl ToPlaceIndex) -> Option<VariantIdx> {
        self.places[p.to_place_index(self).expect("place exists")].active_variant
    }

    // Whether the pointer has been offsetted (and therefore unusable)
    pub fn offseted(&self, p: impl ToPlaceIndex) -> bool {
        let p = p.to_place_index(self).expect("place exists");
        assert!(self.places[p].ty.is_raw_ptr(&self.tcx));

        !matches!(self.places[p].offset, None | Some(0))
    }

    pub fn get_offset(&self, p: impl ToPlaceIndex) -> Option<isize> {
        let p = p.to_place_index(self).expect("place exists");
        assert!(self.places[p].ty.is_raw_ptr(&self.tcx));

        self.places[p].offset
    }

    pub fn offset_ptr(&mut self, p: impl ToPlaceIndex, offset: isize) {
        let p = p.to_place_index(self).expect("place exists");
        assert!(self.places[p].ty.is_raw_ptr(&self.tcx));

        self.places[p].offset = match self.places[p].offset {
            None => Some(offset),
            Some(o) => Some(offset.wrapping_add(o)),
        };
    }

    /// Whether a pointer has had multiple offsets summing up to zero (therefore usable)
    pub fn has_offset_roundtripped(&self, p: impl ToPlaceIndex) -> bool {
        let p = p.to_place_index(self).expect("place exists");
        assert!(self.places[p].ty.is_raw_ptr(&self.tcx));

        self.places[p].offset == Some(0)
    }

    fn locals_with_val(&self, val: usize) -> Vec<Local> {
        if let Some(locals) = self.index_candidates.get(&val) {
            locals
                .iter()
                .copied()
                .filter(|local| {
                    if self.is_place_init(local)
                        && let Some(Literal::Uint(v, UintTy::Usize)) = self.known_val(local)
                        && *v as usize == val
                    {
                        true
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    #[allow(dead_code)]
    pub fn place_count(&self) -> usize {
        self.places.node_count()
    }

    /// Returns the tag used to access a Place. Returns None if the place is not accessed through a pointer
    /// Panics if the place is behind a pointer but the pointer has no tag
    pub fn accessing_tag(&self, place: &Place) -> Option<Tag> {
        // This assumes that Deref is always the first projection
        if let Some(ProjectionElem::Deref) = place.projection().get(0) {
            Some(
                self.places[place.local().to_place_index(&self).expect("place exists")]
                    .tag
                    .expect("has tag"),
            )
        } else {
            None
        }
    }

    /// To be called when a place is written to. Invalidates (removes) all references and raw pointers after the first
    /// shared reference on the stack, and tags after the mutable reference tag used to write
    pub fn place_written(&mut self, p: impl ToPlaceIndex, tag: Option<Tag>) {
        let target = p.to_place_index(&self).expect("place exists");
        self.update_transitive_subfields(target, |this, place| {
            if let Some(run) = this.places[place].run_ptr {
                let invalidated = if let Some(tag) = tag {
                    this.memory.tags_above(run, tag)
                } else {
                    this.memory.above_first_ref(run)
                };
                // TODO: don't copy partially invalidated refs
                for tag in invalidated {
                    this.memory.remove_tag_run_ptr(tag, run);
                }
                VisitAction::Stop
            } else {
                VisitAction::Continue
            }
        });
    }

    fn is_mutably_borrowed(&self, p: PlaceIndex) -> bool {
        let mut is = false;
        self.visit_transitive_subfields(p, |node| {
            if let Some(run) = self.places[node].run_ptr {
                if self.memory.has_mut(run) {
                    is = true;
                    return VisitAction::ShortCircuit;
                }
                VisitAction::Stop
            } else {
                VisitAction::Continue
            }
        });
        is
    }

    /// Checks if a local can be used to read a place
    pub fn can_read_through(&self, local: PlaceIndex, p: PlaceIndex) -> bool {
        if self.is_mutably_borrowed(local) {
            // We cannot access a local if it is currently mutably borrowed
            return false;
        }
        if local == p {
            return true;
        }
        // Access not through deref
        if !self.ty(local).is_any_ptr(&self.tcx) {
            true
        } else {
            let Some(tag) = self.places[local].tag else {
                // Has no permission
                return false;
            };
            let mut can = true;
            self.visit_transitive_subfields(p, |node| {
                if let Some(run) = self.places[node].run_ptr {
                    if !self.memory.can_read_with(run, tag) {
                        can = false;
                        return VisitAction::ShortCircuit;
                    }
                    VisitAction::Stop
                } else {
                    VisitAction::Continue
                }
            });
            can
        }
    }

    /// Checks if a local can be used to write a place
    pub fn can_write_through(&self, local: PlaceIndex, p: PlaceIndex) -> bool {
        if self.is_mutably_borrowed(local) {
            // We cannot access a local if it is currently mutably borrowed
            return false;
        }
        if local == p {
            return true;
        }
        if !self.ty(local).is_any_ptr(&self.tcx) {
            true
        } else {
            let Some(tag) = self.places[local].tag else {
                // Has no permission
                return false;
            };
            let mut can = true;
            self.visit_transitive_subfields(p, |node| {
                if let Some(run) = self.places[node].run_ptr {
                    if !self.memory.can_write_with(run, tag) {
                        can = false;
                        return VisitAction::ShortCircuit;
                    }
                    VisitAction::Stop
                } else {
                    VisitAction::Continue
                }
            });
            can
        }
    }

    // We need to mark reference uninit if the edge is removed (though not raw pointers)
    fn remove_edge(&mut self, e: ProjectionIndex) {
        let (source, _) = self.places.edge_endpoints(e).expect("edge exists");
        let tag = self.places[source].tag.expect("has tag");
        let edges: &mut BTreeSet<NodeIndex> = &mut self.pointer_tags[tag];
        edges.remove(&source);

        let removed = self.places.remove_edge(e).expect("edge exists");
        assert!(removed.is_deref());
    }

    // Returns if a ConstantIndex can be turned into an Index with local.
    // Conditions are that a local with a known val equal to the index exists, and this local
    // can be read
    // HACK: ConstantIndex isn't supported in custom MIR syntax
    fn usable_offset(&self, offset: u64) -> bool {
        let locals = self.locals_with_val(offset as usize);
        let has_usable = locals.iter().any(|local| {
            let pidx = local.to_place_index(&self).expect("exists");
            self.can_read_through(pidx, pidx)
        });
        has_usable
    }
}

#[derive(Debug, Clone)]
pub struct PlacePath {
    source: NodeIndex,
    path: Path,

    target: NodeIndex,
}

impl PlacePath {
    pub fn source(&self) -> NodeIndex {
        self.source
    }

    /// The MIR place this path names in the current frame, validated against
    /// that frame's `local_decls`. A path the graph cannot express as a valid
    /// place is a generator bug, so it panics here rather than surfacing later
    /// as a rustc error on the emitted program.
    pub fn to_place(&self, pt: &PlaceGraph, local_decls: &LocalDecls) -> Place {
        let projs: SmallVec<[ProjectionElem; 8]> = self
            .path
            .iter()
            .map(|&proj| {
                let mut proj = pt.places[proj];
                if let ProjectionElem::ConstantIndex { offset } = proj {
                    let locals = pt.locals_with_val(offset as usize);
                    let local = locals
                        .iter()
                        .filter(|local| {
                            let pidx = local.to_place_index(pt).expect("exists");
                            pt.can_read_through(pidx, pidx)
                        })
                        .next()
                        .expect("has a usable local as index");
                    proj = ProjectionElem::Index(*local);
                }
                proj
            })
            .collect();
        Place::from_projected(
            pt.current_frame().get_by_index(self.source).unwrap(),
            &projs,
            local_decls,
            &pt.tcx,
        )
        .unwrap_or_else(|err| panic!("place path does not name a valid place: {err}"))
    }

    pub fn projections<'pt>(
        &'pt self,
        pt: &'pt PlaceGraph,
    ) -> impl Iterator<Item = ProjectionElem> + 'pt {
        self.path.iter().map(|e| pt.places[*e])
    }

    pub fn nodes<'pt>(&'pt self, pt: &'pt PlaceGraph) -> impl Iterator<Item = PlaceIndex> + 'pt {
        [self.source].into_iter().chain(
            self.path
                .iter()
                .map(|e| pt.places.edge_endpoints(*e).expect("edge exists").1),
        )
    }

    pub fn is_return_proj(&self, pt: &PlaceGraph) -> bool {
        pt.current_frame()
            .get_by_index(self.source)
            .expect("source exists")
            == Local::RET
    }

    pub fn target_index(&self) -> PlaceIndex {
        self.target
    }
}

/// A depth-first iterator over all reachable projection paths from a local variable.
///
/// The root may be followed through any outgoing projection, including `Deref`.
/// Subsequent traversal only follows `subfield_edges`, which contain structural
/// projections created by `add_place`. Reference cycles therefore cannot be
/// re-entered by this iterator.
#[derive(Clone)]
pub struct ProjectionIter<'pt> {
    pt: &'pt PlaceGraph,
    root: PlaceIndex,
    path: Path,
    // Stack of nodes to visit and their depth (number of projections from root)
    to_visit: Vec<(ProjectionIndex, usize)>,

    root_visited: bool,
}

impl<'pt> ProjectionIter<'pt> {
    fn new(pt: &'pt PlaceGraph, root: PlaceIndex) -> Self {
        ProjectionIter {
            pt,
            root,
            path: smallvec![],
            to_visit: pt
                .places
                .edges_directed(root, Direction::Outgoing)
                .filter_map(|e| {
                    // Only downcast to current variants
                    if let ProjectionElem::DowncastField(vid, _, _) = e.weight()
                        && pt.known_variant(e.source()) != Some(*vid)
                    {
                        return None;
                    }
                    if let ProjectionElem::ConstantIndex { offset } = e.weight()
                        && !pt.usable_offset(*offset)
                    {
                        return None;
                    }

                    if pt.ty(e.source()).is_raw_ptr(&pt.tcx) && pt.offseted(e.source()) {
                        return None;
                    }

                    Some((e.id(), 1))
                })
                .collect(),
            root_visited: false,
        }
    }
}

impl<'pt> Iterator for ProjectionIter<'pt> {
    type Item = PlacePath;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.root_visited {
            self.root_visited = true;
            return Some(PlacePath {
                source: self.root,
                path: smallvec![],
                target: self.root,
            });
        }
        if let Some((edge, depth)) = self.to_visit.pop() {
            let (_, target) = self.pt.places.edge_endpoints(edge).unwrap();
            self.path.truncate(depth - 1);
            self.path.push(edge);

            let new_edges = self.pt.places[target].subfield_edges.iter();
            self.to_visit.extend(new_edges.filter_map(|&eidx| {
                let e = self.pt.places[eidx];
                debug_assert!(
                    !e.is_deref(),
                    "subfield_edges must not contain Deref projections"
                );

                // Only downcast to current variants
                if let ProjectionElem::DowncastField(vid, _, _) = e
                    && self.pt.known_variant(target) != Some(vid)
                {
                    return None;
                }

                if let ProjectionElem::ConstantIndex { offset } = e
                    && !self.pt.usable_offset(offset)
                {
                    return None;
                }
                Some((eidx, depth + 1))
            }));

            Some(PlacePath {
                source: self.root,
                path: self.path.clone(),
                target,
            })
        } else {
            None
        }
    }
}

pub trait HasComplexity {
    fn complexity(&self, pt: &PlaceGraph) -> usize;
}

impl HasComplexity for Place {
    fn complexity(&self, pt: &PlaceGraph) -> usize {
        pt.places[self.to_place_index(pt).expect("place exists")].complexity
    }
}

impl HasComplexity for Operand {
    fn complexity(&self, pt: &PlaceGraph) -> usize {
        match self {
            Operand::Copy(place) | Operand::Move(place) => place.complexity(pt),
            Operand::Constant(_) => 1,
        }
    }
}

impl HasComplexity for Rvalue {
    fn complexity(&self, pt: &PlaceGraph) -> usize {
        match self {
            Rvalue::Use(operand) | Rvalue::Cast(operand, _) | Rvalue::UnaryOp(_, operand) => {
                operand.complexity(pt)
            }
            Rvalue::BinaryOp(_, l, r) | Rvalue::CheckedBinaryOp(_, l, r) => {
                l.complexity(pt) + r.complexity(pt)
            }
            Rvalue::Aggregate(_, elems) => elems.iter().map(|op| op.complexity(pt)).sum(),
            Rvalue::Len(_) => 1,
            Rvalue::Discriminant(place) => place.complexity(pt),
            Rvalue::AddressOf(_, place) => place.complexity(pt),
            Rvalue::Ref(_, place) => place.complexity(pt),
        }
    }
}

impl<T> HasComplexity for &T
where
    T: HasComplexity,
{
    fn complexity(&self, pt: &PlaceGraph) -> usize {
        (*self).complexity(pt)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use config::TyConfig;
    use index_vec::IndexVec;
    use mir::{
        syntax::{
            Adt, BinOp, FieldIdx, Literal, Local, LocalDecl, LocalDecls, Mutability, Operand,
            Place, ProjectionElem, Rvalue, TyId, TyKind, UintTy, VariantDef, VariantIdx,
        },
        tyctxt::{AdtMeta, TyCtxt},
    };

    use crate::{
        mem::BasicMemory,
        pgraph::{HasComplexity, PlaceIndex, ToPlaceIndex},
    };

    use super::PlaceGraph;

    /// The `LocalDecls` of a test body declaring `locals`; the return slot and
    /// any undeclared gap are unit. Test places are built against it, as the
    /// generator builds them against the current body.
    fn decls(locals: &[(Local, TyId)]) -> LocalDecls {
        let len = locals
            .iter()
            .map(|(local, _)| local.index() + 1)
            .max()
            .unwrap_or(1);
        let mut decls: LocalDecls = (0..len).map(|_| LocalDecl::new_mut(TyCtxt::UNIT)).collect();
        for &(local, ty) in locals {
            decls[local].ty = ty;
        }
        decls
    }

    fn place(pt: &PlaceGraph, decls: &LocalDecls, local: Local, projs: &[ProjectionElem]) -> Place {
        Place::from_projected(local, projs, decls, &pt.tcx).expect("test place is valid")
    }

    fn prepare_t() -> (PlaceGraph, Local, Place, Place, Place, Place, Place) {
        /*
            ┌──────┬──────┐
            │      │      │
            a      │      c
                ┌──b──┐
                │     │
                d     e
        */

        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let t_i16_i32 = tcx.push(TyKind::Tuple(vec![TyCtxt::I16, TyCtxt::I32]));
        let t_root = tcx.push(TyKind::Tuple(vec![TyCtxt::I8, t_i16_i32, TyCtxt::I64]));

        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let local = Local::new(1);
        pt.allocate_local(local, t_root);
        let decls = decls(&[(local, t_root)]);

        let a = place(
            &pt,
            &decls,
            local,
            &[ProjectionElem::TupleField(FieldIdx::new(0))],
        );
        let b = place(
            &pt,
            &decls,
            local,
            &[ProjectionElem::TupleField(FieldIdx::new(1))],
        );
        let c = place(
            &pt,
            &decls,
            local,
            &[ProjectionElem::TupleField(FieldIdx::new(2))],
        );

        let d = b
            .clone()
            .project(
                ProjectionElem::TupleField(FieldIdx::new(0)),
                &decls,
                &pt.tcx,
            )
            .unwrap()
            .clone();
        let e = b
            .clone()
            .project(
                ProjectionElem::TupleField(FieldIdx::new(1)),
                &decls,
                &pt.tcx,
            )
            .unwrap()
            .clone();
        (pt, local, a, b, c, d, e)
    }

    #[test]
    fn nested_tuple() {
        let (pt, local, _, b, ..) = prepare_t();

        let visited: Vec<TyId> = pt
            .reachable_from_node(local.to_place_index(&pt).unwrap())
            .map(|ppath| pt.ty(ppath.target_index()))
            .collect();
        let root = pt.get_node(&Place::from_local(local)).unwrap();
        let root_ty = pt.places[root].ty;
        let b = pt.get_node(&b).unwrap();
        let b_ty = pt.places[b].ty;
        assert_eq!(
            &visited,
            &[
                root_ty,     // (i8, (i16, i32), i64)
                TyCtxt::I8,  // i8
                b_ty,        // (i16, 132)
                TyCtxt::I32, // i32
                TyCtxt::I16, // i16
                TyCtxt::I64, // i64
            ]
        );
    }

    #[test]
    fn overlap_check() {
        let (pt, local, a, b, c, d, _) = prepare_t();

        assert!(pt.overlap(&b, local));
        assert!(pt.overlap(local, &b));
        assert!(pt.overlap(local, &d));

        assert!(!pt.overlap(&a, &c));
        assert!(!pt.overlap(&a, &d))
    }

    #[test]
    fn pointers() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        // *const i32
        let ptr = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        // (*const i32,)
        let inner_ty = tcx.push(TyKind::Tuple(vec![ptr]));
        // *const (*const i32,)
        let ty = tcx.push(TyKind::RawPtr(inner_ty, Mutability::Not));

        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let root = Local::new(1);
        pt.allocate_local(root, ty);

        let tuple = Local::new(2);
        pt.allocate_local(tuple, inner_ty);

        let int = Local::new(3);
        pt.allocate_local(int, TyCtxt::I32);

        let decls = decls(&[(root, ty), (tuple, inner_ty), (int, TyCtxt::I32)]);

        // root -[Deref]-> tuple -[Field(0)]-> tuple.0 -[Deref]-> int
        let tuple_0 = place(
            &pt,
            &decls,
            tuple,
            &[ProjectionElem::TupleField(FieldIdx::from_usize(0))],
        );
        pt.set_ref(tuple_0.clone(), int, None);

        pt.set_ref(root, tuple, None);

        let visited: Vec<PlaceIndex> = pt
            .reachable_from_node(root.to_place_index(&pt).unwrap())
            .map(|ppath| ppath.target_index())
            .collect();

        // int is not reachable because it is behind a Field projection
        assert_eq!(
            &visited,
            &[
                root.to_place_index(&pt).unwrap(),
                tuple.to_place_index(&pt).unwrap(),
                tuple_0.to_place_index(&pt).unwrap(),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "subfield_edges must not contain Deref projections")]
    fn deref_in_subfield_edges_panics() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        // *const i32
        let ptr = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        // (*const i32,)
        let ty = tcx.push(TyKind::Tuple(vec![ptr]));

        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let root = Local::new(1);
        let root_pidx = pt.allocate_local(root, ty);

        let int = Local::new(2);
        pt.allocate_local(int, TyCtxt::I32);

        let decls = decls(&[(root, ty), (int, TyCtxt::I32)]);

        // root -[TupleField(0)]-> root.0 -[Deref]-> int
        let root_0 = place(
            &pt,
            &decls,
            root,
            &[ProjectionElem::TupleField(FieldIdx::new(0))],
        );
        pt.set_ref(root_0.clone(), int, None);

        // Violate the ProjectionIter invariant: subfield_edges must only
        // contain structural projections created by add_place, but we inject
        // the Deref edge created by set_ref.
        let root_0_pidx = root_0.to_place_index(&pt).unwrap();
        let deref_edge = pt.ref_edge(root_0_pidx).unwrap();
        pt.places[root_0_pidx].subfield_edges.push(deref_edge);

        // Traversing past root.0 consumes its subfield_edges and must trip
        // the debug_assert.
        let _ = pt.reachable_from_node(root_pidx).count();
    }

    #[test]
    fn tuple_projection() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let ty = tcx.push(TyKind::Tuple(vec![TyCtxt::I8, TyCtxt::I32]));

        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let local = Local::new(1);
        pt.allocate_local(local, ty);
        let decls = decls(&[(local, ty)]);

        let place = pt
            .reachable_from_node(local.to_place_index(&pt).unwrap())
            .filter(|ppath| !ppath.path.is_empty())
            .map(|ppath| ppath.to_place(&pt, &decls))
            .next()
            .unwrap();

        assert!(matches!(
            place.projection()[0],
            ProjectionElem::TupleField(..)
        ));
    }

    #[test]
    fn recursive_init() {
        let (mut pt, local, a, b, c, d, e) = prepare_t();

        pt.mark_place_init(&a);
        assert!(pt.is_place_init(&a));
        assert!(!pt.is_place_init(local));

        pt.mark_place_init(&b);
        assert!(pt.is_place_init(&d));
        assert!(pt.is_place_init(&e));
        assert!(!pt.is_place_init(local));

        pt.mark_place_init(&c);
        assert!(pt.is_place_init(local));
    }

    #[test]
    fn recursive_uninit() {
        let (mut pt, local, a, b, c, d, e) = prepare_t();
        pt.mark_place_init(local);

        pt.mark_place_uninit(&d);
        assert!(!pt.is_place_init(&d));

        pt.mark_place_uninit(&e);
        assert!(!pt.is_place_init(&b));

        pt.mark_place_uninit(local);
        assert!(!pt.is_place_init(&a));
        assert!(!pt.is_place_init(&c));
    }

    #[test]
    fn uninit_removes_nested_deref_edges_but_preserves_incoming_refs() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let t_ref_i32 = tcx.push(TyKind::Ref(TyCtxt::I32, Mutability::Not));
        let t_ptr_i32 = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        let t_inner = tcx.push(TyKind::Tuple(vec![t_ref_i32, t_ptr_i32]));
        let t_root = tcx.push(TyKind::Tuple(vec![TyCtxt::U32, t_inner]));
        let t_root_ref = tcx.push(TyKind::Ref(t_root, Mutability::Not));

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let root = Local::new(1);
        let root_pidx = pt.allocate_local(root, t_root);
        pt.mark_place_init(root);

        let target_ref = Local::new(2);
        let target_ref_pidx = pt.allocate_local(target_ref, TyCtxt::I32);
        pt.mark_place_init(target_ref);

        let target_ptr = Local::new(3);
        let target_ptr_pidx = pt.allocate_local(target_ptr, TyCtxt::I32);
        pt.mark_place_init(target_ptr);

        let decls = decls(&[
            (root, t_root),
            (target_ref, TyCtxt::I32),
            (target_ptr, TyCtxt::I32),
        ]);

        let nested_ref = place(
            &pt,
            &decls,
            root,
            &[
                ProjectionElem::TupleField(FieldIdx::new(1)),
                ProjectionElem::TupleField(FieldIdx::new(0)),
            ],
        );
        let nested_ref_pidx = nested_ref.to_place_index(&pt).unwrap();
        pt.set_ref(nested_ref_pidx, target_ref_pidx, None);

        let nested_ptr = place(
            &pt,
            &decls,
            root,
            &[
                ProjectionElem::TupleField(FieldIdx::new(1)),
                ProjectionElem::TupleField(FieldIdx::new(1)),
            ],
        );
        let nested_ptr_pidx = nested_ptr.to_place_index(&pt).unwrap();
        pt.set_ref(nested_ptr_pidx, target_ptr_pidx, None);

        let external_ref = Local::new(4);
        let external_ref_pidx = pt.allocate_local(external_ref, t_root_ref);
        pt.set_ref(external_ref_pidx, root_pidx, None);

        assert_eq!(pt.pointee(nested_ref_pidx), Some(target_ref_pidx));
        assert_eq!(pt.pointee(nested_ptr_pidx), Some(target_ptr_pidx));
        assert_eq!(pt.pointee(external_ref_pidx), Some(root_pidx));

        pt.mark_place_uninit(root_pidx);

        assert!(!pt.is_place_init(root_pidx));
        assert_eq!(pt.pointee(nested_ref_pidx), None);
        assert_eq!(pt.pointee(nested_ptr_pidx), None);
        assert_eq!(pt.pointee(external_ref_pidx), Some(root_pidx));
    }

    #[test]
    fn uninit_array_of_pointers_removes_element_deref_edges() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let t_ptr_i32 = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        let t_arr = tcx.push(TyKind::Array(t_ptr_i32, 2));
        let t_arr_ref = tcx.push(TyKind::Ref(t_arr, Mutability::Not));

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let arr = Local::new(1);
        let arr_pidx = pt.allocate_local(arr, t_arr);
        pt.mark_place_init(arr);

        let target_a = Local::new(2);
        let target_a_pidx = pt.allocate_local(target_a, TyCtxt::I32);
        pt.mark_place_init(target_a);

        let target_b = Local::new(3);
        let target_b_pidx = pt.allocate_local(target_b, TyCtxt::I32);
        pt.mark_place_init(target_b);

        let decls = decls(&[
            (arr, t_arr),
            (target_a, TyCtxt::I32),
            (target_b, TyCtxt::I32),
        ]);

        let elem_0 = place(
            &pt,
            &decls,
            arr,
            &[ProjectionElem::ConstantIndex { offset: 0 }],
        );
        let elem_0_pidx = elem_0.to_place_index(&pt).unwrap();
        pt.set_ref(elem_0_pidx, target_a_pidx, None);

        let elem_1 = place(
            &pt,
            &decls,
            arr,
            &[ProjectionElem::ConstantIndex { offset: 1 }],
        );
        let elem_1_pidx = elem_1.to_place_index(&pt).unwrap();
        pt.set_ref(elem_1_pidx, target_b_pidx, None);

        let external_ref = Local::new(4);
        let external_ref_pidx = pt.allocate_local(external_ref, t_arr_ref);
        pt.set_ref(external_ref_pidx, arr_pidx, None);

        assert_eq!(pt.pointee(elem_0_pidx), Some(target_a_pidx));
        assert_eq!(pt.pointee(elem_1_pidx), Some(target_b_pidx));
        assert_eq!(pt.pointee(external_ref_pidx), Some(arr_pidx));

        pt.mark_place_uninit(arr_pidx);

        assert!(!pt.is_place_init(arr_pidx));
        // The array node owns the RunPointer, so the memory-bounded walk in
        // mark_place_uninit stops at the array itself; only the structural
        // subfield walk reaches the pointer elements. Their outgoing Deref
        // edges must still be removed.
        assert_eq!(pt.pointee(elem_0_pidx), None);
        assert_eq!(pt.pointee(elem_1_pidx), None);
        // Incoming Deref edges from external pointers are preserved.
        assert_eq!(pt.pointee(external_ref_pidx), Some(arr_pidx));
    }

    #[test]
    fn uninit_enum_removes_variant_field_deref_edge() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let t_ptr_i32 = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        let t_enum = tcx.push_adt(
            Adt {
                variants: IndexVec::from_iter([
                    VariantDef {
                        fields: IndexVec::from_iter([t_ptr_i32]),
                    },
                    VariantDef {
                        fields: IndexVec::from_iter([TyCtxt::U32]),
                    },
                ]),
            },
            AdtMeta { copy: true },
        );

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let root = Local::new(1);
        let root_pidx = pt.allocate_local(root, t_enum);
        pt.assign_discriminant(root_pidx, Some(VariantIdx::new(0)));

        let target = Local::new(2);
        let target_pidx = pt.allocate_local(target, TyCtxt::I32);
        pt.mark_place_init(target);

        let decls = decls(&[(root, t_enum), (target, TyCtxt::I32)]);

        let variant_ptr = place(
            &pt,
            &decls,
            root,
            &[ProjectionElem::DowncastField(
                VariantIdx::new(0),
                FieldIdx::new(0),
                t_ptr_i32,
            )],
        );
        let variant_ptr_pidx = variant_ptr.to_place_index(&pt).unwrap();
        pt.mark_place_init(variant_ptr_pidx);
        pt.set_ref(variant_ptr_pidx, target_pidx, None);

        assert!(pt.is_place_init(root_pidx));
        assert_eq!(pt.pointee(variant_ptr_pidx), Some(target_pidx));

        pt.mark_place_uninit(root_pidx);

        assert!(!pt.is_place_init(root_pidx));
        assert_eq!(pt.pointee(variant_ptr_pidx), None);
    }

    #[test]
    fn moved_place_invalidates_nested_deref_edge() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let t_ptr_i32 = tcx.push(TyKind::RawPtr(TyCtxt::I32, Mutability::Not));
        let t_root = tcx.push(TyKind::Tuple(vec![TyCtxt::U32, t_ptr_i32]));

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let root = Local::new(1);
        let root_pidx = pt.allocate_local(root, t_root);
        pt.mark_place_init(root);

        let target = Local::new(2);
        let target_pidx = pt.allocate_local(target, TyCtxt::I32);
        pt.mark_place_init(target);

        let decls = decls(&[(root, t_root), (target, TyCtxt::I32)]);

        let nested_ptr = place(
            &pt,
            &decls,
            root,
            &[ProjectionElem::TupleField(FieldIdx::new(1))],
        );
        let nested_ptr_pidx = nested_ptr.to_place_index(&pt).unwrap();
        pt.set_ref(nested_ptr_pidx, target_pidx, None);

        assert_eq!(pt.pointee(nested_ptr_pidx), Some(target_pidx));

        // mark_place_moved routes through mark_place_uninit; moving out of a
        // place containing a pointer must invalidate the nested Deref edge.
        pt.mark_place_moved(root_pidx);

        assert!(!pt.is_place_init(root_pidx));
        assert_eq!(pt.pointee(nested_ptr_pidx), None);
    }

    #[test]
    fn complexity() {
        let (mut pt, local, a, b, c, d, e) = prepare_t();

        pt.update_complexity(&a, Rvalue::Use(Operand::Constant(1.into())).complexity(&pt));
        assert_eq!(a.complexity(&pt), 1);
        assert_eq!(Place::from(local).complexity(&pt), 1);
        pt.update_complexity(&c, Rvalue::Use(Operand::Constant(1.into())).complexity(&pt));
        assert_eq!(c.complexity(&pt), 1);
        assert_eq!(Place::from(local).complexity(&pt), 1);

        pt.update_complexity(
            &d,
            Rvalue::BinaryOp(
                BinOp::Add,
                Operand::Copy(a.clone()),
                Operand::Copy(c.clone()),
            )
            .complexity(&pt),
        );
        assert_eq!(d.complexity(&pt), 2);

        pt.update_complexity(&e, Rvalue::Use(Operand::Constant(1.into())).complexity(&pt));
        assert_eq!(b.complexity(&pt), 2);

        assert_eq!(Place::from(local).complexity(&pt), 2);
    }

    #[test]
    fn prim_arrays() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let ty = tcx.push(TyKind::Array(TyCtxt::I32, 4));

        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let local = Local::new(1);
        let local_pidx = pt.allocate_local(local, ty);

        let one = Local::new(2);
        pt.allocate_local(one, TyCtxt::USIZE);
        pt.assign_literal(one, Some(Literal::Uint(1, UintTy::Usize)));

        let proj = ProjectionElem::Index(one);

        let local_one = pt.project_from_node(local_pidx, proj).unwrap();

        assert_eq!(
            pt.places[local_pidx].alloc_id,
            pt.places[local_one].alloc_id
        );

        assert_eq!(
            pt.places[local_one].run_ptr.unwrap().run_and_offset,
            pt.places[local_pidx]
                .run_ptr
                .unwrap()
                .run_and_offset
                .offset(
                    BasicMemory::ty_size(TyCtxt::I32, &pt.tcx)
                        .unwrap()
                        .bytes_usize() as isize
                )
        )
    }
    #[test]
    fn composite_arrays() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let elem_ty = tcx.push(TyKind::Tuple(vec![TyCtxt::I32, TyCtxt::I64]));
        let ty = tcx.push(TyKind::Array(elem_ty, 4));
        let mut pt = PlaceGraph::new(Rc::new(tcx));
        let local = Local::new(1);
        let local_pidx = pt.allocate_local(local, ty);

        let one = Local::new(2);
        pt.allocate_local(one, TyCtxt::USIZE);
        pt.assign_literal(one, Some(Literal::Uint(1, UintTy::Usize)));

        let decls = decls(&[(local, ty), (one, TyCtxt::USIZE)]);

        // local[one].0
        let one_zero = place(
            &pt,
            &decls,
            local,
            &[
                ProjectionElem::Index(one),
                ProjectionElem::TupleField(FieldIdx::new(0)),
            ],
        );
        let one_zero = pt.get_node(&one_zero).unwrap();
        assert_eq!(pt.places[one_zero].ty, TyCtxt::I32);
        assert_eq!(pt.places[local_pidx].alloc_id, pt.places[one_zero].alloc_id);
    }

    /// `arr: [i32; 2]` and `one: usize = 1` in the graph. The graph offers the
    /// `arr[1]` path, which `PlacePath::to_place` spells `arr[one]` through
    /// the local known to hold 1. The body declares `one` as `one_decl`, or
    /// not at all.
    fn index_path_place(one_decl: Option<TyId>) -> Place {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let arr_ty = tcx.push(TyKind::Array(TyCtxt::I32, 2));
        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let arr = Local::new(1);
        let arr_pidx = pt.allocate_local(arr, arr_ty);
        pt.mark_place_init(arr);
        let one = Local::new(2);
        pt.allocate_local(one, TyCtxt::USIZE);
        pt.assign_literal(one, Some(Literal::Uint(1, UintTy::Usize)));
        pt.mark_place_init(one);

        let decls = match one_decl {
            Some(ty) => decls(&[(arr, arr_ty), (one, ty)]),
            None => decls(&[(arr, arr_ty)]),
        };
        pt.reachable_from_node(arr_pidx)
            .find(|ppath| {
                ppath
                    .projections(&pt)
                    .eq([ProjectionElem::ConstantIndex { offset: 1 }])
            })
            .expect("the graph offers arr[1]")
            .to_place(&pt, &decls)
    }

    #[test]
    fn to_place_indexes_through_a_declared_usize_local() {
        let place = index_path_place(Some(TyCtxt::USIZE));
        assert_eq!(place.projection(), &[ProjectionElem::Index(Local::new(2))]);
    }

    #[test]
    #[should_panic(expected = "NonUsizeIndexLocal")]
    fn to_place_rejects_an_index_local_declared_non_usize() {
        index_path_place(Some(TyCtxt::I32));
    }

    #[test]
    #[should_panic(expected = "MissingIndexLocal")]
    fn to_place_rejects_an_undeclared_index_local() {
        index_path_place(None);
    }

    #[test]
    fn shared_reference() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());

        let t_i16_i32 = tcx.push(TyKind::Tuple(vec![TyCtxt::I16, TyCtxt::I32]));
        let t_ref = tcx.push(TyKind::Ref(t_i16_i32, Mutability::Not));
        let t_ptr = tcx.push(TyKind::RawPtr(t_i16_i32, Mutability::Not));

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let root = Local::new(1);
        pt.allocate_local(root, t_i16_i32);
        pt.mark_place_init(root);
        let root_ptr1 = Local::new(2);
        let decls = decls(&[(root, t_i16_i32), (root_ptr1, t_ptr)]);

        let root_0 = place(
            &pt,
            &decls,
            root,
            &[ProjectionElem::TupleField(FieldIdx::new(0))],
        )
        .to_place_index(&pt)
        .unwrap();
        let root_1 = place(
            &pt,
            &decls,
            root,
            &[ProjectionElem::TupleField(FieldIdx::new(1))],
        )
        .to_place_index(&pt)
        .unwrap();

        // root_ptr1 = addr_of!(root)
        pt.allocate_local(root_ptr1, t_ptr);
        pt.set_ref(root_ptr1, root, None);
        let root_ptr1_p = root_ptr1.to_place_index(&pt).unwrap();

        // root_ref = &root
        let root_ref = Local::new(3);
        pt.allocate_local(root_ref, t_ref);
        pt.set_ref(root_ref, root, None);
        let root_ref_p = root_ref.to_place_index(&pt).unwrap();

        // (*root_ref).0 can be read
        assert!(pt.can_read_through(root_ref_p, root_0));

        // (*root_ref).0 cannot be written to
        assert!(!pt.can_write_through(root_ref_p, root_0));

        // root_ptr2 = addr_of!(root)
        let root_ptr2 = Local::new(4);
        pt.allocate_local(root_ptr2, t_ptr);
        pt.set_ref(root_ptr2, root, None);
        let root_ptr2_p = root_ptr2.to_place_index(&pt).unwrap();

        // (*root_ptr2).0 cannot be written to
        assert!(!pt.can_write_through(root_ptr2_p, root_0));

        // Writing through (*root_ptr1).0
        let root_ptr1_0 = place(
            &pt,
            &decls,
            root_ptr1,
            &[ProjectionElem::Deref, ProjectionElem::TupleField(0.into())],
        );
        pt.place_written(&root_ptr1_0, pt.places[root_ptr1_p].tag);

        // (*root_ref).0 is invalidated
        assert!(!pt.can_read_through(root_ref_p, root_0));

        // (*root_ptr2).0 is invalidated
        assert!(!pt.can_read_through(root_ptr2_p, root_0));

        // (*root_ref).1 is not invalidated
        assert!(pt.can_read_through(root_ref_p, root_1));

        // (*root_ptr2).1 is not invalidated
        assert!(pt.can_read_through(root_ptr2_p, root_1));

        // *root_ptr1 is not invalidated
        assert!(pt.can_read_through(root_ptr1_p, root.to_place_index(&pt).unwrap(),));
    }

    #[test]
    fn reborrow() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());

        let t_ref = tcx.push(TyKind::Ref(TyCtxt::I32, Mutability::Not));

        let mut pt = PlaceGraph::new(Rc::new(tcx));

        let int = Local::new(1);
        pt.allocate_local(int, TyCtxt::I32);
        pt.mark_place_init(int);

        // int_ref = &int
        let int_ref = Local::new(2);
        pt.allocate_local(int_ref, t_ref);
        let int_ref_p = int_ref.to_place_index(&pt).unwrap();
        pt.set_ref(int_ref, int, None);
        assert_eq!(pt.pointer_tags.first().unwrap().len(), 1);

        // int_ref = &*int_ref
        pt.set_ref(int_ref, int, Some(int_ref_p));
        assert_eq!(pt.pointer_tags.first().unwrap().len(), 1);

        pt.place_written(&Place::from_local(int), None);
        assert!(!pt.can_read_through(int_ref_p, int.to_place_index(&pt).unwrap()));
        assert!(!pt.can_write_through(int_ref_p, int.to_place_index(&pt).unwrap()));
    }
}
