use std::num::TryFromIntError;

use index_vec::{IndexVec, define_index_type};
use smallvec::SmallVec;

use crate::tyctxt::TyCtxt;

#[derive(Clone)]
pub struct Program {
    pub functions: IndexVec<Function, Body>,
    pub entry_args: Vec<Literal>,
    pub use_debug_dumper: bool,
}

pub type LocalDecls = IndexVec<Local, LocalDecl>;

define_index_type! {pub struct Function = u32;}
#[derive(Clone)]
pub struct Body {
    pub basic_blocks: IndexVec<BasicBlock, BasicBlockData>,
    pub local_decls: LocalDecls,
    arg_count: usize,
    pub public: bool,
}

define_index_type! {pub struct BasicBlock = u32;}
#[derive(Clone)]
pub struct BasicBlockData {
    pub(crate) statements: Vec<Statement>,
    pub(crate) terminator: Terminator,
}

impl BasicBlockData {
    /// An empty basic block
    pub fn new() -> Self {
        Self {
            statements: vec![],
            terminator: Terminator::Hole,
        }
    }

    pub fn insert_statement(&mut self, stmt: Statement) {
        self.statements.push(stmt);
    }

    pub fn set_terminator(&mut self, term: Terminator) {
        assert!(matches!(self.terminator, Terminator::Hole));
        self.terminator = term;
    }

    pub fn terminator(&self) -> &Terminator {
        &self.terminator
    }
}

define_index_type! {pub struct Local = u32;}
#[derive(Clone)]
pub struct LocalDecl {
    /// Whether this is a mutable binding (i.e., `let x` or `let mut x`).
    ///
    /// Temporaries and the return place are always mutable.
    pub mutability: Mutability,

    /// If this local is a temporary and `is_block_tail` is `Some`,
    /// The type of this local.
    pub ty: TyId,
}

define_index_type! {pub struct FieldIdx = u32;}
define_index_type! {pub struct VariantIdx = u32;}
impl FieldIdx {
    pub fn identifier(&self) -> String {
        format!("fld{}", self.index())
    }
}
impl VariantIdx {
    pub fn identifier(&self) -> String {
        format!("Variant{}", self.index())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Place {
    local: Local,
    projection: SmallVec<[ProjectionElem; 4]>,
}
impl Place {
    pub const fn from_local(local: Local) -> Self {
        Place {
            local,
            projection: SmallVec::new_const(),
        }
    }

    /// Builds `local` projected by `projections`, validated against the
    /// body's `local_decls`: the base local and every `Index` local must be
    /// declared, and each projection must apply to the type it projects.
    pub fn from_projected(
        local: Local,
        projections: &[ProjectionElem],
        local_decls: &LocalDecls,
        tcx: &TyCtxt,
    ) -> Result<Self, ProjectionError> {
        let place = Place {
            local,
            projection: SmallVec::from(projections),
        };
        place.checked_ty(local_decls, tcx)?;
        Ok(place)
    }

    pub fn local(&self) -> Local {
        self.local
    }

    pub fn projection(&self) -> &[ProjectionElem] {
        &self.projection
    }

    /// Appends `proj`, validated against the body's `local_decls` like
    /// [`Place::from_projected`]. On error the place is left unchanged.
    pub fn project(
        &mut self,
        proj: ProjectionElem,
        local_decls: &LocalDecls,
        tcx: &TyCtxt,
    ) -> Result<&mut Self, ProjectionError> {
        let base = self.checked_ty(local_decls, tcx)?;
        project_elem(base, proj, local_decls, tcx).map_err(|reason| ProjectionError::Invalid {
            position: self.projection.len(),
            elem: proj,
            base,
            reason,
        })?;
        self.projection.push(proj);
        Ok(self)
    }

    /// The type of this place, or the first reason its projection chain is
    /// invalid for `local_decls`.
    pub fn checked_ty(
        &self,
        local_decls: &LocalDecls,
        tcx: &TyCtxt,
    ) -> Result<TyId, ProjectionError> {
        let decl = local_decls
            .get(self.local)
            .ok_or(ProjectionError::MissingLocal(self.local))?;
        self.projection
            .iter()
            .enumerate()
            .try_fold(decl.ty, |base, (position, &elem)| {
                project_elem(base, elem, local_decls, tcx).map_err(|reason| {
                    ProjectionError::Invalid {
                        position,
                        elem,
                        base,
                        reason,
                    }
                })
            })
    }

    pub fn ty(&self, local_decl: &LocalDecls, tcx: &TyCtxt) -> TyId {
        self.checked_ty(local_decl, tcx)
            .unwrap_or_else(|err| panic!("invalid place {self:?}: {err}"))
    }
}

/// One projection step with the checks that need the body: an `Index` local
/// must be a declared `usize`. Type semantics stay in [`TyId::project_one`].
fn project_elem(
    base: TyId,
    elem: ProjectionElem,
    local_decls: &LocalDecls,
    tcx: &TyCtxt,
) -> Result<TyId, InvalidProjection> {
    if let ProjectionElem::Index(index) = elem {
        let decl = local_decls
            .get(index)
            .ok_or(InvalidProjection::MissingIndexLocal(index))?;
        if decl.ty != TyCtxt::USIZE {
            return Err(InvalidProjection::NonUsizeIndexLocal {
                local: index,
                ty: decl.ty,
            });
        }
    }
    base.project_one(tcx, elem)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionError {
    /// The place's base local is not declared in the body.
    MissingLocal(Local),
    /// `elem`, at `position` in the chain, cannot project `base`.
    Invalid {
        position: usize,
        elem: ProjectionElem,
        base: TyId,
        reason: InvalidProjection,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidProjection {
    DerefOfNonPointer,
    TupleFieldOfNonTuple,
    TupleFieldOutOfBounds { len: usize },
    FieldOfNonStruct,
    FieldOutOfBounds { len: usize },
    IndexOfNonArray,
    MissingIndexLocal(Local),
    NonUsizeIndexLocal { local: Local, ty: TyId },
    ConstantIndexOutOfBounds { len: usize },
    DowncastOfNonEnum,
    VariantOutOfBounds { count: usize },
    VariantFieldOutOfBounds { len: usize },
    DowncastFieldTyMismatch { stored: TyId, actual: TyId },
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectionError::MissingLocal(local) => {
                write!(f, "base local {local:?} is not declared")
            }
            ProjectionError::Invalid {
                position,
                elem,
                base,
                reason,
            } => write!(
                f,
                "projection {position} ({elem:?}) cannot project {base:?}: {reason:?}"
            ),
        }
    }
}

impl std::error::Error for ProjectionError {}

impl From<Local> for Place {
    fn from(value: Local) -> Self {
        Self::from_local(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjectionElem {
    Deref,
    /// This should be the same as Field, but to allow for context free serialization
    /// we separate it out
    TupleField(FieldIdx),
    Field(FieldIdx),
    Index(Local),
    // We need the field ty because serialisaion needs it
    DowncastField(VariantIdx, FieldIdx, TyId),
    ConstantIndex {
        offset: u64,
    },
    // TODO: Subslice
}

impl ProjectionElem {
    pub fn is_deref(&self) -> bool {
        *self == ProjectionElem::Deref
    }
}

#[derive(Clone, Copy)]
pub enum Callee {
    Generated(Function),
    Named(&'static str),
    Intrinsic(&'static str),
}

#[derive(Clone)]
pub enum Terminator {
    Hole,
    // define!("mir_return", fn Return() -> BasicBlock);
    Return,
    // define!("mir_goto", fn Goto(destination: BasicBlock) -> BasicBlock);
    Goto {
        target: BasicBlock,
    },
    // define!("mir_unreachable", fn Unreachable() -> BasicBlock);
    Unreachable,
    // define!("mir_drop", fn Drop<T>(place: T, goto: BasicBlock));
    Drop {
        place: Place,
        target: BasicBlock,
    },
    // define!("mir_call", fn Call<T>(place: T, goto: BasicBlock, call: T));
    Call {
        callee: Callee,
        destination: Place,
        target: BasicBlock,
        args: Vec<Operand>,
    },
    /// Switches based on the computed value.
    ///
    /// First, evaluates the `discr` operand. The type of the operand must be a signed or unsigned
    /// integer, char, or bool, and must match the given type. Then, if the list of switch targets
    /// contains the computed value, continues execution at the associated basic block. Otherwise,
    /// continues execution at the "otherwise" basic block.
    ///
    /// Target values may not appear more than once.
    SwitchInt {
        /// The discriminant value being tested.
        discr: Operand,
        targets: SwitchTargets,
    },
}

#[derive(Clone)]
pub struct SwitchTargets {
    pub branches: Vec<(u128, BasicBlock)>,
    pub otherwise: BasicBlock,
}

#[derive(Clone)]
pub enum Rvalue {
    Use(Operand),
    UnaryOp(UnOp, Operand),
    BinaryOp(BinOp, Operand, Operand),
    Cast(Operand, TyId),
    // define!("mir_checked", fn Checked<T>(binop: T) -> (T, bool));
    CheckedBinaryOp(BinOp, Operand, Operand),
    // define!("mir_len", fn Len<T>(place: T) -> usize);
    Len(Place),
    // define!("mir_discriminant",fn Discriminant<T>(place: T) -> <T as ::core::marker::DiscriminantKind>::Discriminant);
    Discriminant(Place),
    AddressOf(Mutability, Place),
    Aggregate(AggregateKind, IndexVec<FieldIdx, Operand>),
    Ref(Mutability, Place),
}

#[derive(Clone, Copy)]
pub enum AggregateKind {
    /// The type is of the element
    Array(TyId),
    Tuple,

    Adt(TyId, VariantIdx),
}

#[derive(Debug, Clone, Copy)]
pub enum Literal {
    Uint(u128, UintTy),
    Int(i128, IntTy),
    Bool(bool),
    Char(char),
    // Every f32 can be expressed exactly as f64
    Float(f64, FloatTy),
}

#[derive(Clone, Debug)]
pub enum Operand {
    Copy(Place),
    // define!("mir_move", fn Move<T>(place: T) -> T);
    Move(Place),
    Constant(Literal),
    // TODO: the following
    // define!("mir_static", fn Static<T>(s: T) -> &'static T);
    // define!("mir_static_mut", fn StaticMut<T>(s: T) -> *mut T);
}

impl Operand {
    pub fn ty(&self, local_decls: &LocalDecls, tcx: &TyCtxt) -> TyId {
        match self {
            Operand::Copy(place) | Operand::Move(place) => place.ty(local_decls, tcx),
            Operand::Constant(lit) => lit.ty(),
        }
    }

    pub fn place(&self) -> Option<&Place> {
        match self {
            Operand::Copy(place) | Operand::Move(place) => Some(place),
            Operand::Constant(..) => None,
        }
    }
}

#[derive(Clone)]
pub enum Statement {
    Assign(Place, Rvalue),
    // define!("mir_storage_live", fn StorageLive<T>(local: T));
    StorageLive(Local),
    // define!("mir_storage_dead", fn StorageDead<T>(local: T));
    StorageDead(Local),
    // define!("mir_storage_live", fn StorageLive<T>(local: T));
    Deinit(Place),
    // define!("mir_set_discriminant", fn SetDiscriminant<T>(place: T, index: u32));
    SetDiscriminant(Place, u32),
    // define!("mir_retag", fn Retag<T>(place: T));
    Retag(Place),
    Nop,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mutability {
    // N.B. Order is deliberate, so that Not < Mut
    Not,
    Mut,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatTy {
    F32,
    F64,
}

define_index_type! {pub struct TyId = u32;}
impl TyId {
    pub fn type_name(&self) -> String {
        format!("Adt{}", self.index())
    }
}
impl TyId {
    pub fn kind(self, tcx: &TyCtxt) -> &TyKind {
        tcx.kind(self)
    }
    pub fn tuple_elems(self, tcx: &TyCtxt) -> Option<&[TyId]> {
        match self.kind(tcx) {
            TyKind::Tuple(tys) => Some(tys),
            _ => None,
        }
    }

    pub fn is_checked_binary_op_lhs(self, tcx: &TyCtxt) -> bool {
        match self.kind(tcx) {
            TyKind::Tuple(elems) => matches!(
                (
                    elems.get(0).map(|ty| ty.kind(tcx)),
                    elems.get(1).map(|ty| ty.kind(tcx))
                ),
                (
                    Some(TyKind::Int(..) | TyKind::Float(..) | TyKind::Uint(..)),
                    Some(TyKind::Bool)
                )
            ),
            _ => false,
        }
    }

    // TODO: are pointers scalar?
    pub fn is_scalar(self, tcx: &TyCtxt) -> bool {
        self.kind(tcx).is_scalar()
    }

    pub fn projected_ty(self, tcx: &TyCtxt, projs: &[ProjectionElem]) -> Self {
        projs.iter().fold(self, |ty, &elem| {
            ty.project_one(tcx, elem)
                .unwrap_or_else(|reason| panic!("{elem:?} cannot project {ty:?}: {reason:?}"))
        })
    }

    /// The type `elem` projects out of `self`. The single owner of projection
    /// type semantics, shared by [`TyId::projected_ty`] and [`Place`]
    /// construction. An `Index` local's own type is checked by `Place`, which
    /// has the body's `LocalDecls`.
    pub fn project_one(
        self,
        tcx: &TyCtxt,
        elem: ProjectionElem,
    ) -> Result<Self, InvalidProjection> {
        match (elem, self.kind(tcx)) {
            (ProjectionElem::Deref, TyKind::RawPtr(pointee, ..) | TyKind::Ref(pointee, ..)) => {
                Ok(*pointee)
            }
            (ProjectionElem::Deref, _) => Err(InvalidProjection::DerefOfNonPointer),
            (ProjectionElem::TupleField(fid), TyKind::Tuple(elems)) => elems
                .get(fid.index())
                .copied()
                .ok_or(InvalidProjection::TupleFieldOutOfBounds { len: elems.len() }),
            (ProjectionElem::TupleField(_), _) => Err(InvalidProjection::TupleFieldOfNonTuple),
            (ProjectionElem::Index(_), TyKind::Array(elem_ty, _)) => Ok(*elem_ty),
            (ProjectionElem::ConstantIndex { offset }, TyKind::Array(elem_ty, len)) => {
                if offset < *len as u64 {
                    Ok(*elem_ty)
                } else {
                    Err(InvalidProjection::ConstantIndexOutOfBounds { len: *len })
                }
            }
            (ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. }, _) => {
                Err(InvalidProjection::IndexOfNonArray)
            }
            (ProjectionElem::Field(fid), TyKind::Adt(adt)) if !adt.is_enum() => {
                let fields = &adt
                    .variants
                    .first()
                    .ok_or(InvalidProjection::FieldOfNonStruct)?
                    .fields;
                fields
                    .get(fid)
                    .copied()
                    .ok_or(InvalidProjection::FieldOutOfBounds { len: fields.len() })
            }
            (ProjectionElem::Field(_), _) => Err(InvalidProjection::FieldOfNonStruct),
            (ProjectionElem::DowncastField(vid, fid, stored), TyKind::Adt(adt))
                if adt.is_enum() =>
            {
                let variant =
                    adt.variants
                        .get(vid)
                        .ok_or(InvalidProjection::VariantOutOfBounds {
                            count: adt.variants.len(),
                        })?;
                let actual = variant.fields.get(fid).copied().ok_or(
                    InvalidProjection::VariantFieldOutOfBounds {
                        len: variant.fields.len(),
                    },
                )?;
                if stored == actual {
                    Ok(actual)
                } else {
                    Err(InvalidProjection::DowncastFieldTyMismatch { stored, actual })
                }
            }
            (ProjectionElem::DowncastField(..), _) => Err(InvalidProjection::DowncastOfNonEnum),
        }
    }

    pub fn contains<P>(self, tcx: &TyCtxt, predicate: P) -> bool
    where
        P: Fn(&TyCtxt, TyId) -> bool + Copy,
    {
        if predicate(tcx, self) {
            return true;
        }
        match self.kind(tcx) {
            TyKind::Tuple(elems) => elems.iter().any(|ty| ty.contains(tcx, predicate)),
            TyKind::RawPtr(pointee, _) => pointee.contains(tcx, predicate),
            TyKind::Array(ty, ..) => ty.contains(tcx, predicate),
            TyKind::Adt(adt) => adt
                .variants
                .iter()
                .any(|variant| variant.fields.iter().any(|ty| ty.contains(tcx, predicate))),
            _ => false,
        }
    }

    pub fn is_ref(self, tcx: &TyCtxt) -> bool {
        matches!(self.kind(tcx), TyKind::Ref(..))
    }

    pub fn is_raw_ptr(self, tcx: &TyCtxt) -> bool {
        matches!(self.kind(tcx), TyKind::RawPtr(..))
    }

    pub fn is_any_ptr(self, tcx: &TyCtxt) -> bool {
        matches!(self.kind(tcx), TyKind::RawPtr(..) | TyKind::Ref(..))
    }

    pub fn pointee_ty(self, tcx: &TyCtxt) -> Option<Self> {
        match self.kind(tcx) {
            TyKind::RawPtr(ty, ..) | TyKind::Ref(ty, ..) => Some(*ty),
            _ => None,
        }
    }

    // If doesn't contain printer
    pub fn determ_printable(self, tcx: &TyCtxt) -> bool {
        !self.contains(tcx, |tcx, ty| ty.is_any_ptr(tcx))
    }

    pub fn hashable(self, tcx: &TyCtxt) -> bool {
        // TODO: hash Adts maybe
        self.kind(tcx).is_structural()
            && self.determ_printable(tcx)
            && !self.contains(tcx, |_, ty| {
                ty == TyCtxt::F32 || ty == TyCtxt::F64 || ty.kind(tcx).is_adt()
            })
            && self != TyCtxt::UNIT
    }

    pub fn is_copy(self, tcx: &TyCtxt) -> bool {
        let kind = self.kind(tcx);
        match kind {
            TyKind::Unit
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_) => true,
            TyKind::RawPtr(_, _) | TyKind::Ref(_, _) => false,
            TyKind::Tuple(tys) => tys.iter().all(|ty| ty.is_copy(tcx)),
            TyKind::Array(ty, _) => ty.is_copy(tcx),
            TyKind::Adt(_) => tcx.meta(self).copy,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TyKind {
    // Scalars
    Unit,
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    // Composite
    RawPtr(TyId, Mutability),
    Ref(TyId, Mutability),
    Tuple(Vec<TyId>),
    // User-defined
    Adt(Adt),
    Array(TyId, usize),
    // TODO: more types
}

impl TyKind {
    pub const ISIZE: Self = TyKind::Int(IntTy::Isize);
    pub const I8: Self = TyKind::Int(IntTy::I8);
    pub const I16: Self = TyKind::Int(IntTy::I16);
    pub const I32: Self = TyKind::Int(IntTy::I32);
    pub const I64: Self = TyKind::Int(IntTy::I64);
    pub const I128: Self = TyKind::Int(IntTy::I128);
    pub const USIZE: Self = TyKind::Uint(UintTy::Usize);
    pub const U8: Self = TyKind::Uint(UintTy::U8);
    pub const U16: Self = TyKind::Uint(UintTy::U16);
    pub const U32: Self = TyKind::Uint(UintTy::U32);
    pub const U64: Self = TyKind::Uint(UintTy::U64);
    pub const U128: Self = TyKind::Uint(UintTy::U128);
    pub const F32: Self = TyKind::Float(FloatTy::F32);
    pub const F64: Self = TyKind::Float(FloatTy::F64);

    pub const INTS: [Self; 6] = [
        Self::ISIZE,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::I128,
    ];

    pub const UINTS: [Self; 6] = [
        Self::USIZE,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::U128,
    ];

    pub const FLOATS: [Self; 2] = [Self::F32, Self::F64];

    pub fn is_scalar(&self) -> bool {
        match self {
            &TyKind::Int(..)
            | &TyKind::Uint(..)
            | &TyKind::Bool
            | &TyKind::Char
            | &TyKind::Unit => true,
            _ => false,
        }
    }

    pub fn is_structural(&self) -> bool {
        !matches!(self, TyKind::Adt(..))
    }

    pub fn is_adt(&self) -> bool {
        matches!(self, TyKind::Adt(..))
    }

    pub fn is_enum(&self) -> bool {
        match self {
            TyKind::Adt(adt) => adt.is_enum(),
            _ => false,
        }
    }
}

impl PartialEq for TyKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Uint(l0), Self::Uint(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::RawPtr(l0, l1), Self::RawPtr(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Ref(l0, l1), Self::Ref(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Tuple(l0), Self::Tuple(r0)) => l0 == r0,
            (Self::Array(l0, l1), Self::Array(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Adt(..), Self::Adt(..)) => false,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub struct VariantDef {
    /// Fields of this variant.
    pub fields: IndexVec<FieldIdx, TyId>,
}

#[derive(Clone, Hash, Debug)]
pub struct Adt {
    pub variants: IndexVec<VariantIdx, VariantDef>,
}
impl Adt {
    pub fn copy_derivable(&self, tcx: &TyCtxt) -> bool {
        self.variants
            .iter()
            .all(|variant| variant.fields.iter().all(|ty| ty.is_copy(tcx)))
    }

    pub fn is_enum(&self) -> bool {
        self.variants.len() > 1
    }
}

#[derive(Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Offset,
}

#[derive(Clone, Copy)]
pub enum UnOp {
    Not,
    Neg,
}

impl Literal {
    // TODO: this doesn't need tcx
    pub fn ty(&self) -> TyId {
        match self {
            Literal::Int(_, IntTy::I8) => TyCtxt::I8,
            Literal::Int(_, IntTy::I16) => TyCtxt::I16,
            Literal::Int(_, IntTy::I32) => TyCtxt::I32,
            Literal::Int(_, IntTy::I64) => TyCtxt::I64,
            Literal::Int(_, IntTy::I128) => TyCtxt::I128,
            Literal::Int(_, IntTy::Isize) => TyCtxt::ISIZE,
            Literal::Uint(_, UintTy::U8) => TyCtxt::U8,
            Literal::Uint(_, UintTy::U16) => TyCtxt::U16,
            Literal::Uint(_, UintTy::U32) => TyCtxt::U32,
            Literal::Uint(_, UintTy::U64) => TyCtxt::U64,
            Literal::Uint(_, UintTy::U128) => TyCtxt::U128,
            Literal::Uint(_, UintTy::Usize) => TyCtxt::USIZE,
            Literal::Bool(_) => TyCtxt::BOOL,
            Literal::Char(_) => TyCtxt::CHAR,
            Literal::Float(_, FloatTy::F32) => TyCtxt::F32,
            Literal::Float(_, FloatTy::F64) => TyCtxt::F64,
        }
    }
}

impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<char> for Literal {
    fn from(value: char) -> Self {
        Self::Char(value)
    }
}

impl From<u8> for Literal {
    fn from(value: u8) -> Self {
        Self::Uint(value as u128, UintTy::U8)
    }
}
impl From<u16> for Literal {
    fn from(value: u16) -> Self {
        Self::Uint(value as u128, UintTy::U16)
    }
}
impl From<u32> for Literal {
    fn from(value: u32) -> Self {
        Self::Uint(value as u128, UintTy::U32)
    }
}
impl From<u64> for Literal {
    fn from(value: u64) -> Self {
        Self::Uint(value as u128, UintTy::U64)
    }
}
impl From<u128> for Literal {
    fn from(value: u128) -> Self {
        Self::Uint(value, UintTy::U128)
    }
}
impl TryFrom<usize> for Literal {
    type Error = TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::Uint(value.try_into()?, UintTy::Usize))
    }
}
impl From<i8> for Literal {
    fn from(value: i8) -> Self {
        Self::Int(value as i128, IntTy::I8)
    }
}
impl From<i16> for Literal {
    fn from(value: i16) -> Self {
        Self::Int(value as i128, IntTy::I16)
    }
}
impl From<i32> for Literal {
    fn from(value: i32) -> Self {
        Self::Int(value as i128, IntTy::I32)
    }
}
impl From<i64> for Literal {
    fn from(value: i64) -> Self {
        Self::Int(value as i128, IntTy::I64)
    }
}
impl From<i128> for Literal {
    fn from(value: i128) -> Self {
        Self::Int(value, IntTy::I128)
    }
}
impl TryFrom<isize> for Literal {
    type Error = TryFromIntError;

    fn try_from(value: isize) -> Result<Self, Self::Error> {
        Ok(Self::Int(value.try_into()?, IntTy::Isize))
    }
}

impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Self::Float(value as f64, FloatTy::F32)
    }
}

impl From<f64> for Literal {
    fn from(value: f64) -> Self {
        Self::Float(value, FloatTy::F64)
    }
}

impl Program {
    pub const FUNCTION_ATTRIBUTE: &'static str =
        "#[custom_mir(dialect = \"runtime\", phase = \"initial\")]";
    pub const HEADER: &'static str = "#![recursion_limit = \"1024\"]
    #![feature(custom_mir, core_intrinsics, lazy_get)]
    #![allow(unused_parens, unused_assignments, overflowing_literals)]
    extern crate core;
    use core::intrinsics::mir::*;\n";

    pub const DUMPER: &'static str = r#"
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::LazyLock;

    static mut H: LazyLock<DefaultHasher> = LazyLock::new(|| DefaultHasher::new());

    #[inline(never)]
    fn dump_var(
        val0: impl Hash,
        val1: impl Hash,
        val2: impl Hash,
        val3: impl Hash,
    ) {
        unsafe {
            val0.hash(LazyLock::force_mut(&mut H));
            val1.hash(LazyLock::force_mut(&mut H));
            val2.hash(LazyLock::force_mut(&mut H));
            val3.hash(LazyLock::force_mut(&mut H));
        }
    }
    "#;

    pub const DEBUG_DUMPER: &'static str = r#"
    use std::fmt::Debug;

    #[inline(never)]
    fn dump_var(
        f: usize,
        var0: usize, val0: impl Debug,
        var1: usize, val1: impl Debug,
        var2: usize, val2: impl Debug,
        var3: usize, val3: impl Debug,
    ) {
        println!("fn{f}:_{var0} = {val0:?}\n_{var1} = {val1:?}\n_{var2} = {val2:?}\n_{var3} = {val3:?}");
    }
    "#;

    // Fake "intrinsic"
    pub const DUMPER_CALL: Callee = Callee::Named("dump_var");
    pub const DUMPER_ARITY: usize = 4;

    // A new, empty function
    pub fn new(debug: bool) -> Self {
        Self {
            functions: IndexVec::default(),
            entry_args: vec![],
            use_debug_dumper: debug,
        }
    }

    pub fn push_fn(&mut self, body: Body) -> Function {
        self.functions.push(body)
    }

    pub fn set_entry_args(&mut self, args: &[Literal]) {
        self.entry_args = Vec::from(args);
    }
}

impl Function {
    pub fn identifier(&self) -> String {
        format!("fn{}", self.index())
    }
}

impl SwitchTargets {
    pub fn match_arms(&self) -> String {
        let mut arms: String = self
            .branches
            .iter()
            .map(|(val, bb)| format!("{val} => {},\n", bb.identifier()))
            .collect();
        arms.push_str(&format!("_ => {}", self.otherwise.identifier()));
        arms
    }
}

impl LocalDecl {
    pub fn new_mut(ty: TyId) -> Self {
        Self {
            mutability: Mutability::Mut,
            ty,
        }
    }
}

impl Local {
    pub const RET: Self = Self::from_raw_unchecked(0);
    pub fn identifier(&self) -> String {
        if *self == Self::RET {
            "RET".to_owned()
        } else {
            format!("_{}", self.index())
        }
    }
}

impl Body {
    pub fn new(args: &[TyId], return_ty: TyId, public: bool) -> Self {
        let mut locals = IndexVec::new();
        locals.push(LocalDecl::new_mut(return_ty));
        // TODO: args shouldn't always be mut
        args.iter().for_each(|ty| {
            locals.push(LocalDecl::new_mut(*ty));
        });
        Self {
            basic_blocks: IndexVec::new(),
            local_decls: locals,
            public,
            arg_count: args.len(),
        }
    }

    pub fn is_arg(&self, local: Local) -> bool {
        (1..self.arg_count).contains(&local.index())
    }

    /// Returns an iterator over function arguments
    pub fn args_iter(&self) -> impl ExactSizeIterator<Item = Local> {
        (1..self.arg_count + 1).map(Local::new)
    }

    pub fn args_decl_iter(&self) -> impl ExactSizeIterator<Item = (Local, &LocalDecl)> {
        self.args_iter()
            .map(|local| (local, &self.local_decls[local]))
    }

    /// Returns an iterator over locals defined in the function body
    pub fn vars_iter(&self) -> impl ExactSizeIterator<Item = Local> {
        (self.arg_count + 1..self.local_decls.len()).map(Local::new)
    }

    pub fn vars_decl_iter(&self) -> impl ExactSizeIterator<Item = (Local, &LocalDecl)> {
        self.vars_iter()
            .map(|local| (local, &self.local_decls[local]))
    }

    pub fn return_ty(&self) -> TyId {
        self.local_decls[Local::RET].ty
    }

    pub fn declare_new_var(&mut self, mutability: Mutability, ty: TyId) -> Local {
        self.local_decls.push(LocalDecl { mutability, ty })
    }

    pub fn new_basic_block(&mut self, bb: BasicBlockData) -> BasicBlock {
        self.basic_blocks.push(bb)
    }
}

impl BasicBlock {
    pub fn identifier(&self) -> String {
        format!("bb{}", self.index())
    }
}

impl Mutability {
    /// Returns `""` (empty string) or `"mut "` depending on the mutability.
    pub fn prefix_str(&self) -> &'static str {
        match self {
            Mutability::Mut => "mut ",
            Mutability::Not => "",
        }
    }

    pub fn ptr_prefix_str(&self) -> &'static str {
        match self {
            Mutability::Mut => "*mut ",
            Mutability::Not => "*const ",
        }
    }
}

impl Place {
    pub const RETURN_SLOT: Self = Self::from_local(Local::RET);
}

impl BinOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Ne => "!=",
            BinOp::Ge => ">=",
            BinOp::Gt => ">",
            BinOp::Offset => panic!("offset is not a real infix operator"),
        }
    }
}

impl UnOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            UnOp::Not => "!",
            UnOp::Neg => "-",
        }
    }
}

#[cfg(test)]
mod place_construction {
    use config::TyConfig;
    use index_vec::IndexVec;

    use super::*;
    use crate::tyctxt::AdtMeta;

    struct Body {
        tcx: TyCtxt,
        decls: LocalDecls,
        root: Local,
        index: Local,
        signed: Local,
        arr: TyId,
    }

    /// `root: *mut (u32, [S; 2])` with `struct S { i16, E }` and
    /// `enum E { V0(u64), V1(i8) }`, plus `index: usize` and `signed: i32`.
    fn body() -> Body {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let e = tcx.push_adt(
            Adt {
                variants: IndexVec::from_iter([
                    VariantDef {
                        fields: IndexVec::from_iter([TyCtxt::U64]),
                    },
                    VariantDef {
                        fields: IndexVec::from_iter([TyCtxt::I8]),
                    },
                ]),
            },
            AdtMeta { copy: true },
        );
        let s = tcx.push_adt(
            Adt {
                variants: IndexVec::from_iter([VariantDef {
                    fields: IndexVec::from_iter([TyCtxt::I16, e]),
                }]),
            },
            AdtMeta { copy: true },
        );
        let arr = tcx.push(TyKind::Array(s, 2));
        let tuple = tcx.push(TyKind::Tuple(vec![TyCtxt::U32, arr]));
        let ptr = tcx.push(TyKind::RawPtr(tuple, Mutability::Mut));

        let mut decls = LocalDecls::new();
        decls.push(LocalDecl::new_mut(TyCtxt::UNIT));
        let root = decls.push(LocalDecl::new_mut(ptr));
        let index = decls.push(LocalDecl::new_mut(TyCtxt::USIZE));
        let signed = decls.push(LocalDecl::new_mut(TyCtxt::I32));
        Body {
            tcx,
            decls,
            root,
            index,
            signed,
            arr,
        }
    }

    fn reason(err: ProjectionError) -> InvalidProjection {
        match err {
            ProjectionError::Invalid { reason, .. } => reason,
            other => panic!("expected an invalid projection, got {other:?}"),
        }
    }

    /// `(*root).1[index].1 as V1 .0`, and the same chain with a constant index.
    fn valid_chain(index: ProjectionElem) -> [ProjectionElem; 5] {
        [
            ProjectionElem::Deref,
            ProjectionElem::TupleField(FieldIdx::new(1)),
            index,
            ProjectionElem::Field(FieldIdx::new(1)),
            ProjectionElem::DowncastField(VariantIdx::new(1), FieldIdx::new(0), TyCtxt::I8),
        ]
    }

    #[test]
    fn valid_chain_builds_the_same_place_either_way() {
        let b = body();
        for index in [
            ProjectionElem::Index(b.index),
            ProjectionElem::ConstantIndex { offset: 1 },
        ] {
            let chain = valid_chain(index);
            let built = Place::from_projected(b.root, &chain, &b.decls, &b.tcx).unwrap();
            assert_eq!(built.ty(&b.decls, &b.tcx), TyCtxt::I8);

            let mut stepped = Place::from_local(b.root);
            for elem in chain {
                stepped.project(elem, &b.decls, &b.tcx).unwrap();
            }
            assert_eq!(stepped, built);
        }
    }

    #[test]
    fn base_local_must_be_declared() {
        let b = body();
        let missing = Local::new(b.decls.len());
        assert_eq!(
            Place::from_projected(missing, &[ProjectionElem::Deref], &b.decls, &b.tcx),
            Err(ProjectionError::MissingLocal(missing))
        );
    }

    #[test]
    fn index_local_must_be_declared() {
        let b = body();
        let missing = Local::new(b.decls.len());
        let err = Place::from_projected(
            b.root,
            &valid_chain(ProjectionElem::Index(missing)),
            &b.decls,
            &b.tcx,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ProjectionError::Invalid {
                position: 2,
                elem: ProjectionElem::Index(missing),
                base: b.arr,
                reason: InvalidProjection::MissingIndexLocal(missing),
            }
        );
    }

    #[test]
    fn index_local_must_be_usize() {
        let b = body();
        let signed = b.signed;
        let err = Place::from_projected(
            b.root,
            &valid_chain(ProjectionElem::Index(signed)),
            &b.decls,
            &b.tcx,
        )
        .unwrap_err();
        assert_eq!(
            reason(err),
            InvalidProjection::NonUsizeIndexLocal {
                local: signed,
                ty: TyCtxt::I32,
            }
        );
    }

    #[test]
    fn each_projection_must_fit_its_base_type() {
        let b = body();
        let deref = [ProjectionElem::Deref];
        let at_tuple = |elem| [ProjectionElem::Deref, elem];
        let at_s = |elem| {
            [
                ProjectionElem::Deref,
                ProjectionElem::TupleField(FieldIdx::new(1)),
                ProjectionElem::ConstantIndex { offset: 0 },
                elem,
            ]
        };
        let at_e = |elem| {
            [
                ProjectionElem::Deref,
                ProjectionElem::TupleField(FieldIdx::new(1)),
                ProjectionElem::ConstantIndex { offset: 0 },
                ProjectionElem::Field(FieldIdx::new(1)),
                elem,
            ]
        };
        let cases: Vec<(Vec<ProjectionElem>, InvalidProjection)> = vec![
            (
                [deref, deref].concat(),
                InvalidProjection::DerefOfNonPointer,
            ),
            (
                at_tuple(ProjectionElem::TupleField(FieldIdx::new(2))).to_vec(),
                InvalidProjection::TupleFieldOutOfBounds { len: 2 },
            ),
            (
                at_tuple(ProjectionElem::Field(FieldIdx::new(0))).to_vec(),
                InvalidProjection::FieldOfNonStruct,
            ),
            (
                at_tuple(ProjectionElem::Index(b.index)).to_vec(),
                InvalidProjection::IndexOfNonArray,
            ),
            (
                [
                    ProjectionElem::Deref,
                    ProjectionElem::TupleField(FieldIdx::new(1)),
                    ProjectionElem::ConstantIndex { offset: 2 },
                ]
                .to_vec(),
                InvalidProjection::ConstantIndexOutOfBounds { len: 2 },
            ),
            (
                at_s(ProjectionElem::Field(FieldIdx::new(2))).to_vec(),
                InvalidProjection::FieldOutOfBounds { len: 2 },
            ),
            (
                at_s(ProjectionElem::DowncastField(
                    VariantIdx::new(0),
                    FieldIdx::new(0),
                    TyCtxt::I16,
                ))
                .to_vec(),
                InvalidProjection::DowncastOfNonEnum,
            ),
            (
                at_e(ProjectionElem::Field(FieldIdx::new(0))).to_vec(),
                InvalidProjection::FieldOfNonStruct,
            ),
            (
                at_e(ProjectionElem::DowncastField(
                    VariantIdx::new(2),
                    FieldIdx::new(0),
                    TyCtxt::I8,
                ))
                .to_vec(),
                InvalidProjection::VariantOutOfBounds { count: 2 },
            ),
            (
                at_e(ProjectionElem::DowncastField(
                    VariantIdx::new(1),
                    FieldIdx::new(1),
                    TyCtxt::I8,
                ))
                .to_vec(),
                InvalidProjection::VariantFieldOutOfBounds { len: 1 },
            ),
            (
                at_e(ProjectionElem::DowncastField(
                    VariantIdx::new(1),
                    FieldIdx::new(0),
                    TyCtxt::U64,
                ))
                .to_vec(),
                InvalidProjection::DowncastFieldTyMismatch {
                    stored: TyCtxt::U64,
                    actual: TyCtxt::I8,
                },
            ),
        ];
        for (chain, expected) in cases {
            let err = Place::from_projected(b.root, &chain, &b.decls, &b.tcx).unwrap_err();
            let ProjectionError::Invalid {
                position, reason, ..
            } = err
            else {
                panic!("{chain:?}: expected an invalid projection, got {err:?}");
            };
            assert_eq!(reason, expected, "{chain:?}");
            assert_eq!(position, chain.len() - 1, "{chain:?}");
        }
    }

    #[test]
    fn rejected_projection_leaves_the_place_unchanged() {
        let b = body();
        let mut place = Place::from_local(b.root);
        place
            .project(ProjectionElem::Deref, &b.decls, &b.tcx)
            .unwrap();
        let before = place.clone();

        let err = place
            .project(ProjectionElem::Deref, &b.decls, &b.tcx)
            .unwrap_err();
        assert_eq!(reason(err), InvalidProjection::DerefOfNonPointer);
        assert_eq!(place, before);
    }
}
