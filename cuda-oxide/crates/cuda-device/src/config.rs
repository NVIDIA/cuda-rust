/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-time vocabulary for describing kernel policies.
//!
//! These types are metadata. They let a kernel library give names to choices
//! such as shape, layout, memory space, and execution scope without owning
//! memory or implying that every combination is executable.
//!
//! ```text
//! Shape2<128, 256> + RowMajor + Shared + Block
//!                         |
//!                         v
//! Tile<..., ..., ..., ...>       (description only)
//! ```
//!
//! A higher-level library decides which descriptions it supports and gives
//! them behavior. For example, a GEMM library can require a shared-memory tile
//! to have an alignment and shape accepted by its copy instruction.
//! Using one of these types as a generic kernel policy creates a separate
//! specialization. The policy is not passed to the GPU as a runtime argument.
//!
//! # Example
//!
//! ```
//! use cuda_device::config::{
//!     Atom, AtomKind, Block, Policy, PolicyId, RowMajor, Shape2, Shape3, Shared,
//!     Tile, WarpGroup,
//! };
//!
//! enum WgmmaF16 {}
//! impl AtomKind for WgmmaF16 {}
//!
//! type OutputTile = Tile<Shape2<128, 256>, RowMajor, Shared, Block>;
//! type MmaAtom = Atom<WgmmaF16, Shape3<128, 256, 64>, WarpGroup>;
//!
//! enum Gemm128x256 {}
//! impl Policy for Gemm128x256 {
//!     // Explicit project namespace + policy value; not a Rust TypeId or hash.
//!     const ID: PolicyId = PolicyId::new(0x6375_6461_6f78_6964, 1);
//! }
//!
//! # let _: core::marker::PhantomData<OutputTile> = core::marker::PhantomData;
//! # let _: core::marker::PhantomData<MmaAtom> = core::marker::PhantomData;
//! assert_eq!(Gemm128x256::ID.value(), 1);
//! ```

use core::marker::PhantomData;

/// A statically known, one-, two-, or three-dimensional shape.
///
/// `EXTENTS` lists dimensions in declaration order. A domain decides what
/// each axis means: a tile can use `(rows, columns)`, while an MMA atom can
/// use `(m, n, k)`. Unused trailing entries are one, so generic code can
/// inspect all shapes uniformly.
///
/// This trait is open because descriptors grant no behavior or safety claim.
/// Libraries may define domain-specific shapes; code that turns a shape into
/// memory access or an instruction must still validate the associated values.
/// Implementations should use rank `1..=3`, pad unused extents with one, and
/// report the checked product of meaningful extents in `ELEMENTS`.
pub trait Shape: 'static {
    /// Number of meaningful entries in [`Self::EXTENTS`].
    const RANK: u8;

    /// Extents in declaration order, padded to three entries with ones.
    const EXTENTS: [usize; 3];

    /// Product of the meaningful extents, or `None` if it overflows `usize`.
    const ELEMENTS: Option<usize>;
}

/// A one-dimensional shape.
pub enum Shape1<const D0: usize> {}

impl<const D0: usize> Shape1<D0> {
    /// Extent of the first axis.
    pub const D0: usize = D0;
}

impl<const D0: usize> Shape for Shape1<D0> {
    const RANK: u8 = 1;
    const EXTENTS: [usize; 3] = [D0, 1, 1];
    const ELEMENTS: Option<usize> = Some(D0);
}

/// A two-dimensional shape.
pub enum Shape2<const D0: usize, const D1: usize> {}

impl<const D0: usize, const D1: usize> Shape2<D0, D1> {
    /// Extent of the first axis.
    pub const D0: usize = D0;

    /// Extent of the second axis.
    pub const D1: usize = D1;
}

impl<const D0: usize, const D1: usize> Shape for Shape2<D0, D1> {
    const RANK: u8 = 2;
    const EXTENTS: [usize; 3] = [D0, D1, 1];
    const ELEMENTS: Option<usize> = D0.checked_mul(D1);
}

/// A three-dimensional shape.
pub enum Shape3<const D0: usize, const D1: usize, const D2: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize> Shape3<D0, D1, D2> {
    /// Extent of the first axis.
    pub const D0: usize = D0;

    /// Extent of the second axis.
    pub const D1: usize = D1;

    /// Extent of the third axis.
    pub const D2: usize = D2;
}

impl<const D0: usize, const D1: usize, const D2: usize> Shape for Shape3<D0, D1, D2> {
    const RANK: u8 = 3;
    const EXTENTS: [usize; 3] = [D0, D1, D2];
    const ELEMENTS: Option<usize> = match D0.checked_mul(D1) {
        Some(prefix) => prefix.checked_mul(D2),
        None => None,
    };
}

/// Memory ordering attached to a tile description.
///
/// This trait is open and metadata-only. Libraries can name swizzled or
/// distributed layouts without asking cuda-oxide to understand their meaning.
pub trait Layout: 'static {}

/// The rightmost (column) coordinate is contiguous.
pub enum RowMajor {}

/// The leftmost (row) coordinate is contiguous.
pub enum ColumnMajor {}

impl Layout for RowMajor {}

impl Layout for ColumnMajor {}

/// A CUDA storage location attached to a tile description.
///
/// A memory-space marker does not allocate memory and does not validate that
/// an operation can access that space.
///
/// This trait is open and metadata-only. Implementing it does not create a
/// CUDA address space or make an access valid.
pub trait MemorySpace: 'static {}

/// Device global memory.
pub enum Global {}

/// Per-block shared memory.
pub enum Shared {}

/// Thread-local registers.
pub enum Register {}

/// Hardware tensor memory.
pub enum TensorMemory {}

impl MemorySpace for Global {}

impl MemorySpace for Shared {}

impl MemorySpace for Register {}

impl MemorySpace for TensorMemory {}

/// The group of CUDA threads that cooperates on an operation.
///
/// A scope marker describes intent only. It does not synchronize threads or
/// prove that all threads in the group participate.
///
/// This trait is open and metadata-only. A library can describe a custom
/// cooperation group, but implementing the trait does not synchronize it.
pub trait Scope: 'static {}

/// One CUDA thread.
pub enum Thread {}

/// One CUDA warp.
pub enum Warp {}

/// A hardware warpgroup.
pub enum WarpGroup {}

/// One CUDA thread block (CTA).
pub enum Block {}

/// A cluster of CUDA thread blocks.
pub enum Cluster {}

impl Scope for Thread {}

impl Scope for Warp {}

impl Scope for WarpGroup {}

impl Scope for Block {}

impl Scope for Cluster {}

/// A metadata-only tile description.
///
/// `Tile` owns no storage and exposes no pointer or indexing operations. It is
/// a zero-sized type used in policy associated types. A domain library must
/// validate combinations before making them operational.
pub struct Tile<S: Shape, L: Layout, M: MemorySpace, Q: Scope> {
    _shape: PhantomData<fn() -> S>,
    _layout: PhantomData<fn() -> L>,
    _memory_space: PhantomData<fn() -> M>,
    _scope: PhantomData<fn() -> Q>,
}

/// Type-level access to the parts of a [`Tile`] description.
///
/// This trait is open and metadata-only, so a library may wrap [`Tile`] or
/// supply a richer descriptor. Consumers must validate a specification before
/// using it to perform memory operations.
pub trait TileSpec: 'static {
    /// Tile shape.
    type Shape: Shape;

    /// Tile memory ordering.
    type Layout: Layout;

    /// Tile storage location.
    type MemorySpace: MemorySpace;

    /// Threads that collectively own or operate on the tile.
    type Scope: Scope;
}

impl<S: Shape, L: Layout, M: MemorySpace, Q: Scope> TileSpec for Tile<S, L, M, Q> {
    type Shape = S;
    type Layout = L;
    type MemorySpace = M;
    type Scope = Q;
}

/// An operation identity used by an [`Atom`] description.
///
/// This marker is intentionally open: libraries can name operations such as
/// a particular MMA, copy, or reduction instruction. Implementing it grants
/// no behavior or safety property.
pub trait AtomKind: 'static {}

/// A metadata-only description of one indivisible operation.
///
/// `K` names the operation, `S` describes its logical footprint, and `Q`
/// describes the participating threads. Operand layouts and memory spaces are
/// domain-specific because an operation can have multiple inputs and outputs;
/// higher-level policy traits should describe those explicitly.
pub struct Atom<K: AtomKind, S: Shape, Q: Scope> {
    _kind: PhantomData<fn() -> K>,
    _shape: PhantomData<fn() -> S>,
    _scope: PhantomData<fn() -> Q>,
}

/// Type-level access to the parts of an [`Atom`] description.
///
/// This trait is open and metadata-only, so instruction libraries may wrap an
/// [`Atom`] or provide a richer descriptor. Implementing it grants no ability
/// to emit an instruction.
pub trait AtomSpec: 'static {
    /// Operation identity supplied by the domain library.
    type Kind: AtomKind;

    /// Logical operation footprint.
    type Shape: Shape;

    /// Threads that collectively execute the operation.
    type Scope: Scope;
}

impl<K: AtomKind, S: Shape, Q: Scope> AtomSpec for Atom<K, S, Q> {
    type Kind = K;
    type Shape = S;
    type Scope = Q;
}

/// Explicit, stable identity for one policy configuration.
///
/// The two 64-bit fields are supplied by the policy author. They are not
/// derived from Rust's [`core::any::TypeId`], a type name, compiler mangling,
/// or a hash, all of which can change between builds. Use a project-specific
/// namespace and keep the value stable while the configuration's generated
/// behavior is unchanged. Allocate a new value when that behavior changes.
///
/// cuda-oxide does not maintain a global namespace registry or detect
/// duplicate IDs; policy libraries own that responsibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PolicyId {
    namespace: u64,
    value: u64,
}

impl PolicyId {
    /// Creates an ID from a project namespace and project-local value.
    pub const fn new(namespace: u64, value: u64) -> Self {
        Self { namespace, value }
    }

    /// Returns the project namespace.
    pub const fn namespace(self) -> u64 {
        self.namespace
    }

    /// Returns the project-local value.
    pub const fn value(self) -> u64 {
        self.value
    }
}

/// Base trait for a named compile-time kernel policy.
///
/// This trait is intentionally open and minimal. Domain libraries extend it
/// with associated types and constants for the choices they understand:
///
/// ```
/// use cuda_device::config::{Policy, TileSpec};
///
/// trait GemmPolicy: Policy {
///     type OutputTile: TileSpec;
///     const PIPELINE_STAGES: usize;
/// }
/// ```
///
/// Keeping the base trait small avoids putting a GEMM-specific or
/// architecture-specific policy model into cuda-oxide.
pub trait Policy: 'static {
    /// Stable identity used by selection, reporting, and cache layers.
    const ID: PolicyId;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn shapes_report_rank_extents_and_checked_size() {
        assert_eq!(Shape1::<8>::RANK, 1);
        assert_eq!(Shape1::<8>::EXTENTS, [8, 1, 1]);
        assert_eq!(Shape1::<8>::ELEMENTS, Some(8));

        assert_eq!(Shape2::<4, 8>::RANK, 2);
        assert_eq!(Shape2::<4, 8>::EXTENTS, [4, 8, 1]);
        assert_eq!(Shape2::<4, 8>::ELEMENTS, Some(32));

        assert_eq!(Shape3::<2, 4, 8>::RANK, 3);
        assert_eq!(Shape3::<2, 4, 8>::EXTENTS, [2, 4, 8]);
        assert_eq!(Shape3::<2, 4, 8>::ELEMENTS, Some(64));
        assert_eq!(Shape2::<{ usize::MAX }, 2>::ELEMENTS, None);
    }

    #[test]
    fn descriptors_are_zero_sized_metadata() {
        type T = Tile<Shape2<16, 32>, RowMajor, Shared, Block>;

        enum CopyKind {}
        impl AtomKind for CopyKind {}
        type A = Atom<CopyKind, Shape2<16, 32>, Warp>;

        assert_eq!(size_of::<T>(), 0);
        assert_eq!(size_of::<A>(), 0);
    }

    #[test]
    fn built_in_markers_cover_the_public_vocabulary() {
        fn assert_layout<L: Layout>() {}
        fn assert_space<M: MemorySpace>() {}
        fn assert_scope<Q: Scope>() {}

        assert_layout::<RowMajor>();
        assert_layout::<ColumnMajor>();

        assert_space::<Global>();
        assert_space::<Shared>();
        assert_space::<Register>();
        assert_space::<TensorMemory>();

        assert_scope::<Thread>();
        assert_scope::<Warp>();
        assert_scope::<WarpGroup>();
        assert_scope::<Block>();
        assert_scope::<Cluster>();
    }

    #[test]
    fn tile_and_atom_parts_remain_visible_to_generic_libraries() {
        fn assert_tile<T>()
        where
            T: TileSpec<
                    Shape = Shape2<16, 32>,
                    Layout = ColumnMajor,
                    MemorySpace = Global,
                    Scope = WarpGroup,
                >,
        {
        }

        enum MmaKind {}
        impl AtomKind for MmaKind {}

        fn assert_atom<A>()
        where
            A: AtomSpec<Kind = MmaKind, Shape = Shape3<16, 8, 16>, Scope = Warp>,
        {
        }

        assert_tile::<Tile<Shape2<16, 32>, ColumnMajor, Global, WarpGroup>>();
        assert_atom::<Atom<MmaKind, Shape3<16, 8, 16>, Warp>>();
    }

    #[test]
    fn policy_ids_are_explicit_values() {
        enum ExamplePolicy {}
        impl Policy for ExamplePolicy {
            const ID: PolicyId = PolicyId::new(0x1122_3344_5566_7788, 7);
        }

        assert_eq!(ExamplePolicy::ID.namespace(), 0x1122_3344_5566_7788);
        assert_eq!(ExamplePolicy::ID.value(), 7);
    }
}
