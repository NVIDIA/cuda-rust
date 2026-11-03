use std::{
    collections::{BTreeSet, HashMap},
    fmt, mem,
    ops::Range,
};

use abi::size::Size;
use index_vec::{IndexVec, define_index_type};
use mir::{
    syntax::{TyId, TyKind},
    tyctxt::TyCtxt,
};
use rangemap::RangeMap;
use smallvec::SmallVec;

define_index_type! {pub struct Tag = u32;}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AbstractByte {
    /// An uninitialized byte.
    Uninit,
    /// An initialized byte, optionally with some provenance (if it is encoding a pointer).
    Init,
}

impl fmt::Debug for AbstractByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninit => write!(f, "UU"),
            Self::Init => write!(f, "II"),
        }
    }
}

impl AbstractByte {
    pub fn is_init(&self) -> bool {
        self == &AbstractByte::Init
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowType {
    Raw,
    Shared,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Borrow {
    borrow_type: BorrowType,
    tag: Tag,
    protected: bool,
}

/// A Run represents a contiguous region of memory free of padding
#[derive(Debug, Clone)]
pub struct Run {
    bytes: Box<[AbstractByte]>,
    ref_stack: RangeMap<Vec<Borrow>>,

    // A cache on if any borrow stack in the run has a mutable borrow
    _has_mut: Option<bool>,
}

impl Run {
    pub fn new_uninit(size: Size) -> Self {
        let bytes = vec![AbstractByte::Uninit; size.bytes() as usize].into_boxed_slice();
        let ref_stack = RangeMap::new(size, vec![]);
        Self {
            bytes,
            ref_stack,
            _has_mut: Some(false),
        }
    }

    #[allow(dead_code)]
    pub fn size(&self) -> Size {
        Size::from_bytes(self.bytes.len())
    }

    pub fn add_borrow(&mut self, offset: Size, len: Size, borrow_type: BorrowType, tag: Tag) {
        if borrow_type == BorrowType::Exclusive {
            self._has_mut = Some(true);
        }
        for (_, stack) in self.ref_stack.iter_mut(offset, len) {
            stack.push(Borrow {
                borrow_type,
                tag,
                protected: false,
            });
        }
    }

    pub fn remove_borrow(&mut self, offset: Size, len: Size, tag: Tag) {
        self._has_mut = None;
        for (_, stack) in self.ref_stack.iter_mut(offset, len) {
            if let Some(i) = stack.iter().position(|b| b.tag == tag) {
                let removed = stack.remove(i);
                assert!(!removed.protected);
            }
        }
    }

    pub fn protect(&mut self, offset: Size, len: Size, tag: Tag) {
        for (_, stack) in self.ref_stack.iter_mut(offset, len) {
            if let Some(i) = stack.iter().position(|b| b.tag == tag) {
                stack[i].protected = true;
            }
        }
    }

    /// Returns a list of tags above a certain tag
    pub fn tags_above(&self, offset: Size, len: Size, tag: Tag) -> Vec<Tag> {
        let mut edges = BTreeSet::new();
        for (_, stack) in self.ref_stack.iter(offset, len) {
            let index = stack
                .iter()
                .position(|borrow| borrow.tag == tag)
                .expect("tag exists");
            edges.extend(stack[index + 1..].iter().map(|borrow| borrow.tag));
        }
        edges.iter().copied().collect()
    }

    pub fn above_first_ref(&self, offset: Size, len: Size) -> Vec<Tag> {
        let mut edges = BTreeSet::new();
        for (_, stack) in self.ref_stack.iter(offset, len) {
            let first_shared = stack.iter().position(|borrow| {
                matches!(
                    borrow.borrow_type,
                    BorrowType::Shared | BorrowType::Exclusive
                )
            });
            if let Some(first_shared) = first_shared {
                edges.extend(stack[first_shared..].iter().map(|borrow| borrow.tag));
            }
        }
        edges.iter().copied().collect()
    }

    fn has_mut(&self, offset: Size, len: Size) -> bool {
        if let Some(has_mut) = self._has_mut {
            return has_mut;
        }
        for (_, stack) in self.ref_stack.iter(offset, len) {
            if stack
                .iter()
                .any(|borrow| borrow.borrow_type == BorrowType::Exclusive)
            {
                return true;
            }
        }
        false
    }

    /// Checks if a tag is below the last exclusive reference
    fn is_below_last_mut(&self, offset: Size, len: Size, tag: Tag) -> bool {
        for (_, stack) in self.ref_stack.iter(offset, len) {
            let last_mut = stack
                .iter()
                .rposition(|borrow| borrow.borrow_type == BorrowType::Exclusive);
            if let Some(pos) = last_mut {
                if stack[..pos]
                    .iter()
                    .find(|borrow| borrow.tag == tag)
                    .is_some()
                {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_read_with(&self, offset: Size, len: Size, tag: Tag) -> bool {
        if self.is_below_last_mut(offset, len, tag) {
            return false;
        }
        self.ref_stack
            .iter(offset, len)
            .all(|(_, stack)| stack.iter().any(|borrow| borrow.tag == tag))
    }

    pub fn can_write_with(&self, offset: Size, len: Size, tag: Tag) -> bool {
        if self.is_below_last_mut(offset, len, tag) {
            return false;
        }
        for (_, stack) in self.ref_stack.iter(offset, len) {
            let first_shared = stack
                .iter()
                .position(|borrow| borrow.borrow_type == BorrowType::Shared);
            let mut tag_search_limit = stack.len();
            if let Some(first_shared) = first_shared {
                tag_search_limit = first_shared;
                // nothing above first shared (which will be popped off)
                // is protected
                if stack[first_shared..].iter().any(|borrow| borrow.protected) {
                    return false;
                }
            }
            // and the tag exists below first shared (or anywhere, if there is no shared)
            if stack[..tag_search_limit]
                .iter()
                .find(|borrow| borrow.tag == tag)
                .is_none()
            {
                return false;
            }
        }
        true
    }
}

define_index_type! {pub struct RunId = u32;}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunAndOffset(RunId, Size);

impl RunAndOffset {
    pub fn same_run(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    pub fn offset(&self, offset: isize) -> Self {
        Self(self.0, Size::from_bytes(self.1.bytes() as isize + offset))
    }
}

#[derive(Clone)]
struct Allocation {
    /// The data stored in this allocation.
    runs: IndexVec<RunId, Run>,
    /// The alignment that was requested for this allocation.
    // align: Align,
    /// Whether this allocation is still live.
    live: bool,
}

#[allow(dead_code)]
impl Allocation {
    fn runs_and_sizes(&self) -> impl Iterator<Item = (RunId, Size)> + '_ {
        self.runs
            .iter_enumerated()
            .map(|(run_id, run)| (run_id, run.size()))
    }

    fn run(&self, run_and_offset: RunAndOffset) -> &Run {
        &self.runs[run_and_offset.0]
    }
}

pub struct AllocationBuilder {
    alloc_id: AllocId,
    runs: IndexVec<RunId, Run>,
}

impl AllocationBuilder {
    pub fn new_run(&mut self, size: Size) -> RunAndOffset {
        let run = Run::new_uninit(size);
        let run_id = self.runs.push(run);
        RunAndOffset(run_id, Size::ZERO)
    }

    pub fn alloc_id(&self) -> AllocId {
        self.alloc_id
    }

    fn build(self) -> Allocation {
        Allocation {
            runs: self.runs,
            live: true,
        }
    }
}

trait RangeExt: Sized {
    fn overlap(&self, other: &Self) -> bool;
    fn subtract(&self, other: &Self) -> [Option<Self>; 2];
}

impl RangeExt for Range<usize> {
    fn overlap(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }
    fn subtract(&self, other: &Self) -> [Option<Self>; 2] {
        assert!(self.overlap(other));
        let left = if self.start < other.start {
            Some(self.start..other.start)
        } else {
            None
        };
        let right = if other.end < self.end {
            Some(other.end..self.end)
        } else {
            None
        };
        [left, right]
    }
}

define_index_type! {pub struct AllocId = u32;}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunPointer {
    pub alloc_id: AllocId,
    pub run_and_offset: RunAndOffset,
    pub size: Size,
}

impl RunPointer {
    pub fn from_bytes_range(range: Range<usize>, alloc_id: AllocId, run: RunId) -> Self {
        RunPointer {
            alloc_id,
            run_and_offset: RunAndOffset(run, Size::from_bytes(range.start)),
            size: Size::from_bytes(range.count()),
        }
    }

    pub fn run(&self) -> RunId {
        self.run_and_offset.0
    }

    pub fn offset(&self) -> Size {
        self.run_and_offset.1
    }

    #[allow(dead_code)]
    pub fn len(&self) -> Size {
        self.size
    }

    pub fn bytes_range(&self) -> Range<usize> {
        self.run_and_offset.1.bytes_usize()
            ..self.run_and_offset.1.bytes_usize() + self.size.bytes_usize()
    }

    pub fn overlap(&self, other: &Self) -> bool {
        if self.alloc_id != other.alloc_id {
            return false;
        }
        if !self.run_and_offset.same_run(&other.run_and_offset) {
            return false;
        }
        self.bytes_range().overlap(&other.bytes_range())
    }
}

#[derive(Clone)]
pub struct BasicMemory {
    allocations: IndexVec<AllocId, Allocation>,

    // a lookup table to aid removal from borrow stacks
    // a tag may cover multiple runs, e.g. &(u32, u32),
    pointers: HashMap<Tag, SmallVec<[RunPointer; 4]>>,
}

impl BasicMemory {
    const PTR_SIZE: Size = Size::from_bytes_const(mem::size_of::<*const ()>() as u64);

    pub fn new() -> Self {
        Self {
            allocations: IndexVec::new(),
            pointers: HashMap::new(),
        }
    }

    pub fn allocate_with_builder<F>(&mut self, build: F) -> AllocId
    where
        F: FnOnce(&mut AllocationBuilder),
    {
        let alloc_id = self.allocations.len_idx();
        let mut builder = AllocationBuilder {
            alloc_id,
            runs: IndexVec::new(),
        };
        build(&mut builder);
        self.allocations.push(builder.build())
    }

    pub fn deallocate(&mut self, alloc_id: AllocId) {
        self.allocations[alloc_id].live = false;
    }

    pub fn is_live(&self, alloc_id: AllocId) -> bool {
        self.allocations[alloc_id].live
    }

    pub fn bytes(&self, run_ptr: RunPointer) -> &[AbstractByte] {
        assert!(
            self.allocations[run_ptr.alloc_id].live,
            "can't access dead bytes"
        );
        &self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].bytes[run_ptr.bytes_range()]
    }

    pub fn fill(&mut self, run_ptr: RunPointer, val: AbstractByte) {
        self.bytes_mut(run_ptr).fill(val);
    }

    pub fn bytes_mut(&mut self, run_ptr: RunPointer) -> &mut [AbstractByte] {
        assert!(
            self.allocations[run_ptr.alloc_id].live,
            "can't access dead bytes"
        );
        &mut self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].bytes[run_ptr.bytes_range()]
    }

    pub fn copy(&mut self, dst: RunPointer, src: RunPointer) {
        assert_eq!(dst.size, src.size);
        let tmp = self.bytes(src).to_vec();
        self.bytes_mut(dst).copy_from_slice(&tmp)
    }

    /// Returns Size for types with guaranteed size.
    /// Composite types under the default layout has no guaranteed size,
    /// as the AM is free to insert arbitarily large paddings.
    pub fn ty_size(ty: TyId, tcx: &TyCtxt) -> Option<Size> {
        Some(match ty {
            TyCtxt::UNIT => Size::ZERO,
            TyCtxt::BOOL => Size::from_bytes(1),
            TyCtxt::CHAR => Size::from_bytes(4),
            TyCtxt::I8 | TyCtxt::U8 => Size::from_bits(8),
            TyCtxt::I16 | TyCtxt::U16 => Size::from_bits(16),
            TyCtxt::I32 | TyCtxt::U32 => Size::from_bits(32),
            TyCtxt::I64 | TyCtxt::U64 => Size::from_bits(64),
            TyCtxt::I128 | TyCtxt::U128 => Size::from_bits(128),
            TyCtxt::F32 => Size::from_bits(32),
            TyCtxt::F64 => Size::from_bits(64),
            TyCtxt::ISIZE | TyCtxt::USIZE => Self::PTR_SIZE,
            _ => match ty.kind(tcx) {
                TyKind::RawPtr(..) => Self::PTR_SIZE,
                TyKind::Ref(..) => Self::PTR_SIZE,
                TyKind::Array(ty, len) => {
                    return Self::ty_size(*ty, tcx)
                        .map(|elem| Size::from_bytes(elem.bytes_usize() * len));
                }
                _ => return None,
            },
        })
    }

    pub fn add_ref(&mut self, run_ptr: RunPointer, borrow_type: BorrowType, tag: Tag) {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].add_borrow(
            run_ptr.offset(),
            run_ptr.size,
            borrow_type,
            tag,
        );
        self.pointers
            .entry(tag)
            .and_modify(|ptrs| ptrs.push(run_ptr))
            .or_insert(SmallVec::from([run_ptr].as_slice()));
    }

    /// Remove a range (run_ptr) from the lookup table.
    fn derange(&mut self, tag: Tag, run_ptr: RunPointer) {
        if let Some(all_run_ptrs) = self.pointers.get(&tag) {
            // Check if the run_ptr we removed overlaps with ones cached, then remove/split them as necessary
            let mut updated = SmallVec::new();
            for stored in all_run_ptrs {
                if stored.overlap(&run_ptr) {
                    let left_and_right = stored.bytes_range().subtract(&run_ptr.bytes_range());
                    for range in left_and_right {
                        if let Some(range) = range {
                            updated.push(RunPointer::from_bytes_range(
                                range,
                                stored.alloc_id,
                                stored.run(),
                            ));
                        }
                    }
                } else {
                    updated.push(*stored);
                }
            }

            let empty = updated.is_empty();
            if empty {
                self.pointers.remove(&tag);
            } else {
                self.pointers.insert(tag, updated);
            }
        }
    }

    /// Remove tag for a run ptr. Returns true if the ref is no longer present in any
    /// borrow stack
    pub fn remove_tag_run_ptr(&mut self, tag: Tag, run_ptr: RunPointer) -> bool {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].remove_borrow(
            run_ptr.offset(),
            run_ptr.size,
            tag,
        );

        self.derange(tag, run_ptr);
        !self.pointers.contains_key(&tag)
    }

    pub fn tags_above(&self, run_ptr: RunPointer, tag: Tag) -> Vec<Tag> {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].tags_above(
            run_ptr.offset(),
            run_ptr.size,
            tag,
        )
    }

    pub fn above_first_ref(&self, run_ptr: RunPointer) -> Vec<Tag> {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()]
            .above_first_ref(run_ptr.offset(), run_ptr.size)
    }

    pub fn mark_protected(&mut self, run_ptr: RunPointer, tag: Tag) {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].protect(
            run_ptr.offset(),
            run_ptr.size,
            tag,
        )
    }

    pub fn can_read_with(&self, run_ptr: RunPointer, tag: Tag) -> bool {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].can_read_with(
            run_ptr.offset(),
            run_ptr.size,
            tag,
        )
    }

    pub fn can_write_with(&self, run_ptr: RunPointer, tag: Tag) -> bool {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()].can_write_with(
            run_ptr.offset(),
            run_ptr.size,
            tag,
        )
    }

    pub fn has_mut(&self, run_ptr: RunPointer) -> bool {
        self.allocations[run_ptr.alloc_id].runs[run_ptr.run()]
            .has_mut(run_ptr.offset(), run_ptr.size)
    }
}
