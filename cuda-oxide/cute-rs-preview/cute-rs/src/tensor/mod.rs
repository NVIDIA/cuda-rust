/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed views of existing tensor memory.
//!
//! A [`Tensor`] combines three things:
//!
//! ```text
//! storage: where the bits are
//! layout:  how coordinates find those bits
//! element: what the bits mean
//!              │
//!              ▼
//!            Tensor
//! ```
//!
//! A tensor view does not allocate, copy, or load memory. Operations such as
//! [`Tensor::zipped_divide`] and [`Tensor::slice`] only change how the same
//! storage is viewed. [`Tensor::load`] is the first operation that reads it.
//!
//! [`partition`] describes strided views, subranges and cyclic thread
//! partitions while retaining their element and mutable-ownership contracts.

pub mod partition;

use core::marker::PhantomData;
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use cuda_device::DisjointSlice;

use crate::atom::vector::{assume, load_tile, store_tile};

/// Says which Rust storage type may hold one or more logical elements.
///
/// For example, `Float4E2M1FN: TensorElement<u8>` means an `u8` may carry two packed
/// FP4 values. A missing implementation rejects an invalid pair at compile
/// time, such as E2M1 data stored in `f32`.
pub trait TensorElement<Storage: Copy> {}

// Plain f16 and f32 values use the same type for meaning and storage.
impl TensorElement<f16> for f16 {}
impl TensorElement<f32> for f32 {}
impl TensorElement<u8> for u8 {}
impl TensorElement<u16> for u16 {}
impl TensorElement<u32> for u32 {}
impl TensorElement<u64> for u64 {}
impl TensorElement<u128> for u128 {}
impl TensorElement<usize> for usize {}
impl TensorElement<i8> for i8 {}
impl TensorElement<i16> for i16 {}
impl TensorElement<i32> for i32 {}
impl TensorElement<i64> for i64 {}
impl TensorElement<i128> for i128 {}
impl TensorElement<isize> for isize {}

/// One line of adjacent elements.
///
/// ```text
/// index:   0   1   2   3
/// memory: [a] [b] [c] [d]
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contiguous1D {
    /// Number of logical elements in the contiguous row.
    len: usize,
}

impl Contiguous1D {
    /// Create a layout containing `len` adjacent elements.
    #[must_use]
    #[inline(always)]
    pub const fn new(len: usize) -> Self {
        Self { len }
    }

    /// Return the number of elements.
    #[must_use]
    #[inline(always)]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return `true` when the length is zero.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// A one-dimensional tensor split into fixed-size tiles.
///
/// For `TILE = 4`:
///
/// ```text
/// original: [0 1 2 3 | 4 5 6 7 | 8 9]
/// tile:          0          1        2
/// inside:    0 1 2 3    0 1 2 3    0 1
/// ```
///
/// The final tile may be shorter than `TILE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zipped1D<const TILE: usize> {
    /// Full tensor length, used to retain a possibly partial final tile.
    len: usize,
}

impl<const TILE: usize> Zipped1D<TILE> {
    /// Return the number of tiles, including a short final tile.
    #[must_use]
    #[inline(always)]
    pub const fn tile_count(self) -> usize {
        const { assert!(TILE > 0) };
        self.len.div_ceil(TILE)
    }
}

/// One selected tile from a [`Zipped1D`] view.
///
/// It stores the tile's first index and the full tensor length. It does not
/// contain or load the tile values.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile1D<const TILE: usize> {
    /// Absolute first element index of the selected tile.
    base: usize,
    /// Full tensor length used to clip the selected tile.
    len: usize,
}

impl<const TILE: usize> Tile1D<TILE> {
    /// Return the tile's first index in the original tensor.
    #[must_use]
    #[inline(always)]
    pub const fn base(self) -> usize {
        self.base
    }

    /// Return `true` when all `TILE` positions are in bounds.
    #[must_use]
    #[inline(always)]
    pub const fn is_full(self) -> bool {
        self.valid_len() == TILE
    }

    /// Return the number of positions that are in bounds.
    #[must_use]
    #[inline(always)]
    pub const fn valid_len(self) -> usize {
        if self.base >= self.len {
            0
        } else {
            let remaining = self.len - self.base;
            if remaining < TILE { remaining } else { TILE }
        }
    }
}

/// The `N` values from one tile, held in thread-local registers.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisterTile<T: Copy, const N: usize>(
    #[doc = "Per-thread register values in increasing tile-coordinate order."] [T; N],
);

impl<T: Copy, const N: usize> RegisterTile<T, N> {
    /// Construct a register tile in increasing tile-coordinate order.
    #[must_use]
    #[inline(always)]
    pub const fn from_array(values: [T; N]) -> Self {
        Self(values)
    }

    /// Broadcast one scalar into every lane of the register tile
    /// (CuTeDSL `cute.full((N,), value)` / `vector.broadcast`).
    #[must_use]
    #[inline(always)]
    pub fn splat(value: T) -> Self {
        if N == 0 {
            return Self(core::array::from_fn(|_| value));
        }
        __compiler::register_tile_broadcast::<T, N>(value)
    }

    /// Construct registers by visiting each coordinate exactly once in order.
    #[must_use]
    #[inline(always)]
    pub fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        if N == 0 {
            return Self(core::array::from_fn(f));
        }
        let first = f(0);
        let mut values = [first; N];
        let mut i = 1;
        while i < N {
            cuda_device::thread::__unroll_config::<0>();
            values[i] = f(i);
            i += 1;
        }
        Self(values)
    }

    /// Transform each register value, preserving the tile's element count.
    #[must_use]
    #[inline(always)]
    pub fn map<U: Copy>(self, mut f: impl FnMut(T) -> U) -> RegisterTile<U, N> {
        RegisterTile::from_fn(|i| f(self.0[i]))
    }

    /// Transform corresponding values of two equally sized register tiles.
    #[must_use]
    #[inline(always)]
    pub fn zip_map<U: Copy, V: Copy>(
        self,
        rhs: RegisterTile<U, N>,
        mut f: impl FnMut(T, U) -> V,
    ) -> RegisterTile<V, N> {
        RegisterTile::from_fn(|i| f(self.0[i], rhs.0[i]))
    }

    /// Reduce this thread's registers in increasing element order.
    #[must_use]
    #[inline(always)]
    pub fn fold<U>(self, initial: U, mut f: impl FnMut(U, T) -> U) -> U {
        let mut accumulator = initial;
        let mut i = 0;
        while i < N {
            cuda_device::thread::__unroll_config::<0>();
            accumulator = f(accumulator, self.0[i]);
            i += 1;
        }
        accumulator
    }

    /// Gather register coordinates into another fixed-size register tile.
    #[must_use]
    #[inline(always)]
    pub fn gather<const M: usize>(self, coordinates: [usize; M]) -> RegisterTile<T, M> {
        RegisterTile::from_fn(|i| self.0[coordinates[i]])
    }

    /// Borrow the values as a fixed-size array.
    #[must_use]
    #[inline(always)]
    pub const fn as_array(&self) -> &[T; N] {
        &self.0
    }

    /// Consume the tile and return its fixed-size array.
    #[must_use]
    #[inline(always)]
    pub fn into_array(self) -> [T; N] {
        self.0
    }

    /// Convert this register tile to target numeric format `Dst` (CuTeDSL `TensorSSA.to(dtype)`).
    #[must_use]
    #[inline(always)]
    pub fn to<Dst: ToNumeric<T, N>>(self) -> Dst::Output {
        Dst::convert_tile(self)
    }

    /// Reinterpret the bits of this register tile as `RegisterTile<Dst, M>` of equal total bit width
    /// (CuTeDSL `TensorSSA.bitcast(dtype)`).
    #[must_use]
    #[inline(always)]
    pub fn bitcast<Dst: BitcastNumeric<T, N, M>, const M: usize>(self) -> RegisterTile<Dst, M> {
        Dst::bitcast_tile(self)
    }

    /// Extract a contiguous `M`-element sub-vector starting at `OFFSET`
    /// (CuTeDSL `TensorSSA[OFFSET:OFFSET + M]` / `vector.extract_strided_slice`).
    #[must_use]
    #[inline(always)]
    pub fn slice<const OFFSET: usize, const M: usize>(&self) -> RegisterTile<T, M> {
        const { assert!(OFFSET + M <= N) };
        if M == 0 {
            return RegisterTile(core::array::from_fn(|i| self.0[OFFSET + i]));
        }
        __compiler::register_tile_slice::<T, N, OFFSET, M>(self)
    }
}

/// CuTeDSL-aligned alias (`cutlass.cute.TensorSSA`) for [`RegisterTile`].
pub type TensorSSA<T, const N: usize> = RegisterTile<T, N>;

/// Reduction operator for [`RegisterTile::reduce`] (`cutlass.cute.ReductionOp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOp {
    /// Elementwise sum reduction (`ReductionOp.ADD`).
    Add,
    /// Elementwise product reduction (`ReductionOp.MUL`).
    Mul,
    /// Elementwise maximum reduction (`ReductionOp.MAX`).
    Max,
    /// Elementwise minimum reduction (`ReductionOp.MIN`).
    Min,
}

impl<T: Copy, const N: usize> Index<usize> for RegisterTile<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Copy, const N: usize> IndexMut<usize> for RegisterTile<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize> RegisterTile<f32, N> {
    /// Elementwise absolute value (CuTeDSL `abs(TensorSSA)` / `math.absf : vector<Nxf32>`).
    #[must_use]
    #[inline(always)]
    pub fn abs(self) -> Self {
        if N == 0 {
            return self;
        }
        __compiler::register_tile_unary::<f32, N, 1>(&self)
    }

    /// Elementwise maximum with a broadcast scalar (`cute.arch.fmax` / `arith.maxnumf : vector<Nxf32>`).
    #[must_use]
    #[inline(always)]
    pub fn fmax(self, rhs: f32) -> Self {
        if N == 0 {
            return self;
        }
        __compiler::register_tile_scalar::<f32, N, 4>(&self, rhs)
    }

    /// Reduce all elements using `op` starting from `init_val`
    /// (CuTeDSL `TensorSSA.reduce(op, init_val)` / `vector.reduction`).
    #[must_use]
    #[inline(always)]
    pub fn reduce(self, op: ReductionOp, init_val: f32) -> f32 {
        if N == 0 {
            return init_val;
        }
        match op {
            ReductionOp::Add => __compiler::register_tile_reduce::<N, 0>(&self, init_val),
            ReductionOp::Mul => __compiler::register_tile_reduce::<N, 1>(&self, init_val),
            ReductionOp::Max => __compiler::register_tile_reduce::<N, 2>(&self, init_val),
            ReductionOp::Min => __compiler::register_tile_reduce::<N, 3>(&self, init_val),
        }
    }
}

impl<T: Copy> RegisterTile<T, 2> {
    /// Concatenate two 2-element register tiles into one 4-element [`RegisterTile<T, 4>`]
    /// (`vector.shuffle`).
    #[must_use]
    #[inline(always)]
    pub fn concat(self, high: Self) -> RegisterTile<T, 4> {
        __compiler::register_tile_concat::<T, 2, 4>(&self, &high)
    }
}

impl<T: Copy> RegisterTile<T, 4> {
    /// Concatenate two 4-element register tiles into one 8-element [`RegisterTile<T, 8>`]
    /// (`vector.shuffle`).
    #[must_use]
    #[inline(always)]
    pub fn concat(self, high: Self) -> RegisterTile<T, 8> {
        __compiler::register_tile_concat::<T, 4, 8>(&self, &high)
    }
}

impl<T: Copy> RegisterTile<T, 8> {
    /// Concatenate two 8-element register tiles into one 16-element [`RegisterTile<T, 16>`]
    /// (`vector.shuffle`).
    #[must_use]
    #[inline(always)]
    pub fn concat(self, high: Self) -> RegisterTile<T, 16> {
        __compiler::register_tile_concat::<T, 8, 16>(&self, &high)
    }
}

/// Target numeric format for [`RegisterTile::to`] (CuTeDSL `TensorSSA.to(dtype)`).
pub trait ToNumeric<Src: Copy, const N: usize> {
    /// Resulting [`RegisterTile`] representation.
    type Output;

    /// Convert `tile` into `Self::Output`.
    #[must_use]
    fn convert_tile(tile: RegisterTile<Src, N>) -> Self::Output;
}

impl ToNumeric<f32, 8> for crate::numeric::Float4E2M1FN {
    type Output = RegisterTile<crate::numeric::Float4E2M1FNx8, 1>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f32, 8>) -> Self::Output {
        let [v0, v1, v2, v3, v4, v5, v6, v7] = tile.into_array();
        let bits = crate::numeric::__compiler::cvt_f32x8_to_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
        RegisterTile([crate::numeric::Float4E2M1FNx8::from_bits(bits)])
    }
}

impl ToNumeric<f32, 16> for crate::numeric::Float4E2M1FN {
    type Output = RegisterTile<crate::numeric::Float4E2M1FNx8, 2>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f32, 16>) -> Self::Output {
        let [
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
        ] = tile.into_array();
        let lo = crate::numeric::__compiler::cvt_f32x8_to_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
        let hi =
            crate::numeric::__compiler::cvt_f32x8_to_e2m1x8(v8, v9, v10, v11, v12, v13, v14, v15);
        RegisterTile([
            crate::numeric::Float4E2M1FNx8::from_bits(lo),
            crate::numeric::Float4E2M1FNx8::from_bits(hi),
        ])
    }
}

impl<const N: usize> ToNumeric<f16, N> for f32 {
    type Output = RegisterTile<f32, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f16, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| 0.0f32));
        }
        __compiler::register_tile_cast::<f16, f32, N, 5>(&tile)
    }
}

impl<const N: usize> ToNumeric<f32, N> for f16 {
    type Output = RegisterTile<f16, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f32, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| 0.0f16));
        }
        __compiler::register_tile_cast::<f32, f16, N, 4>(&tile)
    }
}

/// Bit-preserving reinterpretation target for [`RegisterTile::bitcast`]
/// (CuTeDSL `TensorSSA.bitcast(dtype)` / `vector.bitcast`).
pub trait BitcastNumeric<Src: Copy, const N: usize, const M: usize>: Copy {
    /// Reinterpret the bits of `tile` (`N` elements of `Src`) as `M` elements of `Self`.
    #[must_use]
    fn bitcast_tile(tile: RegisterTile<Src, N>) -> RegisterTile<Self, M>;
}

impl BitcastNumeric<crate::numeric::Float4E2M1FNx8, 1, 1> for u32 {
    #[inline(always)]
    fn bitcast_tile(tile: RegisterTile<crate::numeric::Float4E2M1FNx8, 1>) -> RegisterTile<u32, 1> {
        __compiler::register_tile_bitcast::<crate::numeric::Float4E2M1FNx8, u32, 1, 1>(&tile)
    }
}

impl BitcastNumeric<crate::numeric::Float4E2M1FNx8, 2, 1> for u64 {
    #[inline(always)]
    fn bitcast_tile(tile: RegisterTile<crate::numeric::Float4E2M1FNx8, 2>) -> RegisterTile<u64, 1> {
        __compiler::register_tile_bitcast::<crate::numeric::Float4E2M1FNx8, u64, 2, 1>(&tile)
    }
}

impl BitcastNumeric<crate::numeric::Float4E2M1FNx8, 2, 8> for u8 {
    #[inline(always)]
    fn bitcast_tile(tile: RegisterTile<crate::numeric::Float4E2M1FNx8, 2>) -> RegisterTile<u8, 8> {
        __compiler::register_tile_bitcast::<crate::numeric::Float4E2M1FNx8, u8, 2, 8>(&tile)
    }
}

impl BitcastNumeric<crate::numeric::Float4E2M1FNx8, 4, 1> for u128 {
    #[inline(always)]
    fn bitcast_tile(
        tile: RegisterTile<crate::numeric::Float4E2M1FNx8, 4>,
    ) -> RegisterTile<u128, 1> {
        __compiler::register_tile_bitcast::<crate::numeric::Float4E2M1FNx8, u128, 4, 1>(&tile)
    }
}

impl BitcastNumeric<crate::numeric::Float8E4M3FN, 4, 1> for u32 {
    #[inline(always)]
    fn bitcast_tile(tile: RegisterTile<crate::numeric::Float8E4M3FN, 4>) -> RegisterTile<u32, 1> {
        __compiler::register_tile_bitcast::<crate::numeric::Float8E4M3FN, u32, 4, 1>(&tile)
    }
}

impl BitcastNumeric<crate::numeric::BFloat16, 8, 1> for u128 {
    #[inline(always)]
    fn bitcast_tile(tile: RegisterTile<crate::numeric::BFloat16, 8>) -> RegisterTile<u128, 1> {
        __compiler::register_tile_bitcast::<crate::numeric::BFloat16, u128, 8, 1>(&tile)
    }
}

impl<const N: usize> ToNumeric<f32, N> for crate::numeric::BFloat16 {
    type Output = RegisterTile<crate::numeric::BFloat16, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f32, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| {
                crate::numeric::BFloat16::from_bits(0)
            }));
        }
        __compiler::register_tile_cast::<f32, crate::numeric::BFloat16, N, 0>(&tile)
    }
}

impl<const N: usize> ToNumeric<crate::numeric::BFloat16, N> for f32 {
    type Output = RegisterTile<f32, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<crate::numeric::BFloat16, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| 0.0f32));
        }
        __compiler::register_tile_cast::<crate::numeric::BFloat16, f32, N, 1>(&tile)
    }
}

impl<const N: usize> ToNumeric<f32, N> for crate::numeric::Float8E4M3FN {
    type Output = RegisterTile<crate::numeric::Float8E4M3FN, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<f32, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| {
                crate::numeric::Float8E4M3FN::from_bits(0)
            }));
        }
        __compiler::register_tile_cast::<f32, crate::numeric::Float8E4M3FN, N, 2>(&tile)
    }
}

impl<const N: usize> ToNumeric<crate::numeric::Float8E4M3FN, N> for f32 {
    type Output = RegisterTile<f32, N>;

    #[inline(always)]
    fn convert_tile(tile: RegisterTile<crate::numeric::Float8E4M3FN, N>) -> Self::Output {
        if N == 0 {
            return RegisterTile(core::array::from_fn(|_| 0.0f32));
        }
        __compiler::register_tile_cast::<crate::numeric::Float8E4M3FN, f32, N, 3>(&tile)
    }
}

/// Elementwise sum of two equally sized register tiles (CuTeDSL `TensorSSA + TensorSSA`).
impl<T: Copy + Add<Output = T>, const N: usize> Add for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_binary::<T, N, 0>(&self, &rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
        }
    }
}

/// Elementwise sum of a register tile and a broadcast scalar (CuTeDSL `TensorSSA + scalar`).
impl<T: Copy + Add<Output = T>, const N: usize> Add<T> for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: T) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_scalar::<T, N, 0>(&self, rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] + rhs))
        }
    }
}

/// Elementwise difference of two equally sized register tiles (CuTeDSL `TensorSSA - TensorSSA`).
impl<T: Copy + Sub<Output = T>, const N: usize> Sub for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_binary::<T, N, 1>(&self, &rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
        }
    }
}

/// Elementwise difference of a register tile and a broadcast scalar (CuTeDSL `TensorSSA - scalar`).
impl<T: Copy + Sub<Output = T>, const N: usize> Sub<T> for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: T) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_scalar::<T, N, 1>(&self, rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] - rhs))
        }
    }
}

/// Elementwise product of two equally sized register tiles (CuTeDSL `TensorSSA * TensorSSA`).
impl<T: Copy + Mul<Output = T>, const N: usize> Mul for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_binary::<T, N, 2>(&self, &rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] * rhs.0[i]))
        }
    }
}

/// Elementwise product of a register tile and a broadcast scalar (CuTeDSL `TensorSSA * scalar`).
impl<T: Copy + Mul<Output = T>, const N: usize> Mul<T> for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_scalar::<T, N, 2>(&self, rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] * rhs))
        }
    }
}

/// Elementwise quotient of two equally sized register tiles (CuTeDSL `TensorSSA / TensorSSA`
/// for floating-point lanes).
impl<T: Copy + Div<Output = T>, const N: usize> Div for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_binary::<T, N, 3>(&self, &rhs)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| self.0[i] / rhs.0[i]))
        }
    }
}

/// Elementwise negation of a register tile (CuTeDSL `-TensorSSA`).
impl<T: Copy + Neg<Output = T>, const N: usize> Neg for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        if N == 0 {
            return self;
        }
        #[cfg(target_os = "cuda")]
        {
            __compiler::register_tile_unary::<T, N, 0>(&self)
        }
        #[cfg(not(target_os = "cuda"))]
        {
            Self(core::array::from_fn(|i| -self.0[i]))
        }
    }
}

/// Compiler hooks lowering [`RegisterTile`] (`TensorSSA`) operations to `dialect-cute` vector ops
/// (`vector.broadcast`, `arith.*`, `math.absf`, `vector.reduction`, `vector.bitcast`,
/// `vector.extract_strided_slice`, `vector.shuffle`).
#[doc(hidden)]
pub mod __compiler {
    use super::RegisterTile;

    /// Broadcast `value` across `N` lanes (`cute.vector_broadcast` -> `vector.broadcast`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_broadcast<T: Copy, const N: usize>(value: T) -> RegisterTile<T, N> {
        RegisterTile([value; N])
    }

    /// Elementwise binary op (`0=Add`, `1=Sub`, `2=Mul`, `3=Div`) (`cute.vector_binary`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_binary<T: Copy, const N: usize, const OP: u8>(
        lhs: &RegisterTile<T, N>,
        _rhs: &RegisterTile<T, N>,
    ) -> RegisterTile<T, N> {
        *lhs
    }

    /// Elementwise vector-scalar op (`0=Add`, `1=Sub`, `2=Mul`, `3=Div`, `4=Max`) (`cute.vector_scalar`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_scalar<T: Copy, const N: usize, const OP: u8>(
        lhs: &RegisterTile<T, N>,
        rhs: T,
    ) -> RegisterTile<T, N> {
        RegisterTile(core::array::from_fn(|i| {
            let size = core::mem::size_of::<T>();
            if OP == 4 && size == 4 {
                let a = unsafe { core::mem::transmute_copy::<T, f32>(&lhs.0[i]) };
                let b = unsafe { core::mem::transmute_copy::<T, f32>(&rhs) };
                let r = a.max(b);
                unsafe { core::mem::transmute_copy::<f32, T>(&r) }
            } else {
                lhs.0[i]
            }
        }))
    }

    /// Elementwise unary op (`0=Neg`, `1=Abs`) (`cute.vector_unary`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_unary<T: Copy, const N: usize, const OP: u8>(
        src: &RegisterTile<T, N>,
    ) -> RegisterTile<T, N> {
        RegisterTile(core::array::from_fn(|i| {
            if OP == 1 && core::mem::size_of::<T>() == 4 {
                let a = unsafe { core::mem::transmute_copy::<T, f32>(&src.0[i]) }.abs();
                unsafe { core::mem::transmute_copy::<f32, T>(&a) }
            } else {
                src.0[i]
            }
        }))
    }

    /// Horizontal reduction (`0=Add`, `1=Mul`, `2=Max`, `3=Min`) (`cute.vector_reduce` -> `vector.reduction`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_reduce<const N: usize, const OP: u8>(
        src: &RegisterTile<f32, N>,
        init: f32,
    ) -> f32 {
        src.0.iter().copied().fold(init, |acc, v| match OP {
            0 => acc + v,
            1 => acc * v,
            2 => acc.max(v),
            _ => acc.min(v),
        })
    }

    /// Elementwise numeric cast (`cute.vector_cast` -> `arith.truncf` / `arith.extf`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_cast<Src: Copy, Dst: Copy, const N: usize, const KIND: u8>(
        src: &RegisterTile<Src, N>,
    ) -> RegisterTile<Dst, N> {
        RegisterTile(core::array::from_fn(|i| match KIND {
            0 => {
                let v = unsafe { core::mem::transmute_copy::<Src, f32>(&src.0[i]) };
                let b = crate::numeric::BFloat16::from_f32(v);
                unsafe { core::mem::transmute_copy::<crate::numeric::BFloat16, Dst>(&b) }
            }
            1 => {
                let b = unsafe {
                    core::mem::transmute_copy::<Src, crate::numeric::BFloat16>(&src.0[i])
                };
                let v = b.to_f32();
                unsafe { core::mem::transmute_copy::<f32, Dst>(&v) }
            }
            2 => {
                let v = unsafe { core::mem::transmute_copy::<Src, f32>(&src.0[i]) };
                let b = crate::numeric::Float8E4M3FN::from_f32(v);
                unsafe { core::mem::transmute_copy::<crate::numeric::Float8E4M3FN, Dst>(&b) }
            }
            3 => {
                let b = unsafe {
                    core::mem::transmute_copy::<Src, crate::numeric::Float8E4M3FN>(&src.0[i])
                };
                let v = b.to_f32();
                unsafe { core::mem::transmute_copy::<f32, Dst>(&v) }
            }
            4 => {
                let v = unsafe { core::mem::transmute_copy::<Src, f32>(&src.0[i]) } as f16;
                unsafe { core::mem::transmute_copy::<f16, Dst>(&v) }
            }
            _ => {
                let v = unsafe { core::mem::transmute_copy::<Src, f16>(&src.0[i]) } as f32;
                unsafe { core::mem::transmute_copy::<f32, Dst>(&v) }
            }
        }))
    }

    /// Bit-preserving vector reinterpretation (`cute.vector_bitcast` -> `vector.bitcast`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_bitcast<Src: Copy, Dst: Copy, const N: usize, const M: usize>(
        src: &RegisterTile<Src, N>,
    ) -> RegisterTile<Dst, M> {
        const {
            assert!(N * core::mem::size_of::<Src>() == M * core::mem::size_of::<Dst>());
        }
        unsafe { core::mem::transmute_copy(src) }
    }

    /// Contiguous sub-vector extraction (`cute.vector_slice` -> `vector.extract_strided_slice`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_slice<T: Copy, const N: usize, const OFFSET: usize, const M: usize>(
        src: &RegisterTile<T, N>,
    ) -> RegisterTile<T, M> {
        RegisterTile(core::array::from_fn(|i| src.0[OFFSET + i]))
    }

    /// Concatenate two `N`-element register tiles into one `M`-element register tile (`cute.vector_concat` -> `vector.shuffle`).
    #[must_use]
    #[inline(never)]
    pub fn register_tile_concat<T: Copy, const N: usize, const M: usize>(
        low: &RegisterTile<T, N>,
        high: &RegisterTile<T, N>,
    ) -> RegisterTile<T, M> {
        RegisterTile(core::array::from_fn(|i| {
            if i < N { low.0[i] } else { high.0[i - N] }
        }))
    }
}

/// A read-only view of storage with a typed element format and layout.
///
/// The view owns no memory and performs no allocation:
///
/// ```text
/// Tensor<'a, Element, Layout, Storage>
///             │        │        │
///             │        │        └─ Rust type holding the bits
///             │        └────────── coordinate-to-offset rule
///             └─────────────────── meaning of the bits
/// ```
///
/// For ordinary values, `Storage` defaults to `Element`, so
/// `Tensor<f32, Layout>` is enough. Packed data names both types; for example,
/// `Tensor<Float4E2M1FN, Layout, u8>` means two logical FP4 values share each byte.
/// [`TensorElement`] checks that the pair is allowed.
///
/// Copying this value copies only the view. The underlying storage is still
/// borrowed. This view is used inside Rust kernel code; it is not passed
/// directly as a raw kernel argument.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a, Element, Layout, Storage = Element>
where
    Element: TensorElement<Storage>,
    Storage: Copy,
{
    /// Borrowed storage carrying the logical elements.
    pub(crate) storage: &'a [Storage],
    /// Coordinate map or tile position attached to the storage.
    pub(crate) layout: Layout,
    /// Logical element meaning, which may differ from the storage type.
    element: PhantomData<Element>,
}

impl<'a, Element: TensorElement<Storage>, Layout, Storage: Copy>
    Tensor<'a, Element, Layout, Storage>
{
    /// View `storage` through `layout` without copying or loading it.
    #[must_use]
    #[inline(always)]
    pub const fn from_storage(storage: &'a [Storage], layout: Layout) -> Self {
        Self {
            storage,
            layout,
            element: PhantomData,
        }
    }

    /// Return the slice that holds the physical bits.
    #[must_use]
    #[inline(always)]
    pub const fn storage(&self) -> &'a [Storage] {
        self.storage
    }

    /// Return the layout value stored by this view.
    #[must_use]
    #[inline(always)]
    pub const fn layout(&self) -> Layout
    where
        Layout: Copy,
    {
        self.layout
    }
}

impl<'a, T: Copy + TensorElement<T>> Tensor<'a, T, Contiguous1D, T> {
    /// View a slice as one contiguous row.
    #[must_use]
    #[inline(always)]
    pub const fn from_slice(storage: &'a [T]) -> Self {
        make_tensor_read(storage)
    }

    /// Split the row into `(position_in_tile, tile_number)` coordinates.
    ///
    /// ```text
    /// [0 1 2 3 | 4 5 6 7]  ── zipped_divide::<4>() ──► 2 tiles of 4
    /// ```
    ///
    /// This only changes the view. It does not read memory.
    #[must_use]
    #[inline(always)]
    pub const fn zipped_divide<const TILE: usize>(self) -> Tensor<'a, T, Zipped1D<TILE>, T> {
        zipped_divide_read::<T, TILE>(self)
    }

    /// Select tile number `tile` of the row split into `TILE`-element tiles.
    ///
    /// Equivalent to `zipped_divide::<TILE>().slice(tile)`: CuTe's
    /// `local_tile` for a one-dimensional tiler and a single tile
    /// coordinate. This only changes the view. It does not read memory.
    #[must_use]
    #[inline(always)]
    pub const fn local_tile<const TILE: usize>(
        self,
        tile: usize,
    ) -> Tensor<'a, T, Tile1D<TILE>, T> {
        self.zipped_divide::<TILE>().slice(tile)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> Tensor<'a, T, Zipped1D<TILE>, T> {
    /// Select tile number `tile`.
    ///
    /// This only computes the tile's starting index. Call [`Tensor::load`] to
    /// read its values.
    #[must_use]
    #[inline(always)]
    pub const fn slice(self, tile: usize) -> Tensor<'a, T, Tile1D<TILE>, T> {
        slice_read::<T, TILE>(self, tile)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> Tensor<'a, T, Tile1D<TILE>, T> {
    /// Return `true` when every position in this tile is in bounds.
    ///
    /// This is the tensor-level form of [`Tile1D::is_full`]. Keeping the
    /// question on the view makes the complete CuTe flow visible to the
    /// compiler:
    ///
    /// ```text
    /// Tensor -> zipped_divide -> slice -> is_full
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn is_full(self) -> bool {
        tensor_tile_is_full::<T, TILE>(self)
    }

    /// Return this tile's first index in the original tensor.
    ///
    /// This is useful for a short final tile, where the kernel falls back to
    /// scalar loads and stores.
    #[must_use]
    #[inline(always)]
    pub const fn base(self) -> usize {
        tensor_tile_base::<T, TILE>(self)
    }

    /// Load the full tile into one [`RegisterTile`] using one vector load.
    ///
    /// # Safety
    ///
    /// - [`Tile1D::is_full`] must be `true`.
    /// - The tile's first address must be aligned to
    ///   `TILE * size_of::<T>()` bytes.
    /// - That total byte size must be a supported power of two from 4 through
    ///   16.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load(self) -> RegisterTile<T, TILE> {
        unsafe { tensor_load_tile::<T, TILE>(self) }
    }
}

/// A writable view of storage with a typed element format and layout.
///
/// This is separate from [`Tensor`] so a read-only slice can never gain a
/// store method. `TensorMut` is not `Copy` or `Clone`: there should be one
/// owner of the writable view.
///
/// GPU threads may still calculate overlapping indices, so store methods are
/// unsafe and state the non-overlap rules. This view is used inside Rust
/// kernel code; it is not passed directly as a raw kernel argument.
#[derive(Debug)]
pub struct TensorMut<'a, Element, Layout, Storage = Element>
where
    Element: TensorElement<Storage>,
    Storage: Copy,
{
    /// Non-owning mutable storage base.
    storage: *mut Storage,
    /// Storage element count available from the base.
    storage_len: usize,
    /// Coordinate map or selected tile position.
    layout: Layout,
    /// Logical element meaning preserved across view transformations.
    element: PhantomData<Element>,
    /// Retains the exclusive storage borrow without manufacturing a mutable slice.
    borrow: PhantomData<&'a mut [Storage]>,
}

impl<'a, Element: TensorElement<Storage>, Layout, Storage: Copy>
    TensorMut<'a, Element, Layout, Storage>
{
    /// Build a writable tensor view from a raw pointer, length, and layout.
    ///
    /// # Safety
    ///
    /// - `storage` must remain valid and correctly aligned for `storage_len`
    ///   elements throughout `'a`.
    /// - Every offset produced by `layout` must be less than `storage_len`.
    /// - No other live reference may read or write the same memory in a way
    ///   that breaks Rust's exclusive-access rules.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        storage: *mut Storage,
        storage_len: usize,
        layout: Layout,
    ) -> Self {
        Self {
            storage,
            storage_len,
            layout,
            element: PhantomData,
            borrow: PhantomData,
        }
    }

    /// Return the layout value stored by this view.
    #[must_use]
    #[inline(always)]
    pub const fn layout(&self) -> Layout
    where
        Layout: Copy,
    {
        self.layout
    }
}

impl<'a, T: Copy + TensorElement<T>> TensorMut<'a, T, Contiguous1D, T> {
    /// View a kernel output slice as one contiguous row.
    #[must_use]
    #[inline(always)]
    pub fn from_disjoint_slice(storage: &'a mut DisjointSlice<'_, T>) -> Self {
        let len = storage.len();
        unsafe { make_tensor_write(storage.as_mut_ptr(), len) }
    }

    /// Write one value at its index in the original contiguous tensor.
    ///
    /// # Safety
    ///
    /// - `index` must be less than the tensor length.
    /// - No concurrently running thread may access the same element in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store_at(&mut self, index: usize, value: T) {
        unsafe { *self.storage.add(index) = value };
    }

    /// Split the row into `(position_in_tile, tile_number)` coordinates.
    ///
    /// This only changes the view. It does not write memory.
    #[must_use]
    #[inline(always)]
    pub fn zipped_divide<const TILE: usize>(self) -> TensorMut<'a, T, Zipped1D<TILE>, T> {
        zipped_divide_write::<T, TILE>(self)
    }

    /// Select writable tile number `tile` of the row split into `TILE`-element tiles.
    ///
    /// Equivalent to `zipped_divide::<TILE>().slice(tile)`: CuTe's
    /// `local_tile` for a one-dimensional tiler and a single tile
    /// coordinate. This only changes the view. It does not write memory.
    #[must_use]
    #[inline(always)]
    pub fn local_tile<const TILE: usize>(self, tile: usize) -> TensorMut<'a, T, Tile1D<TILE>, T> {
        self.zipped_divide::<TILE>().slice(tile)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> TensorMut<'a, T, Zipped1D<TILE>, T> {
    /// Select writable tile number `tile`.
    ///
    /// This only computes the tile's starting index. Call [`TensorMut::store`]
    /// to write it.
    #[must_use]
    #[inline(always)]
    pub fn slice(self, tile: usize) -> TensorMut<'a, T, Tile1D<TILE>, T> {
        slice_write::<T, TILE>(self, tile)
    }
}

/// Compiler boundary for a read-only contiguous tensor view.
///
/// This stays public only so device compilation can identify the exact
/// definition across crate boundaries. Use [`Tensor::from_slice`] in kernels.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn make_tensor_read<'a, T: Copy + TensorElement<T>>(
    storage: &'a [T],
) -> Tensor<'a, T, Contiguous1D, T> {
    Tensor::from_storage(storage, Contiguous1D::new(storage.len()))
}

/// Compiler boundary for a writable contiguous tensor view.
///
/// Use [`TensorMut::from_disjoint_slice`] instead. The safety rules are the
/// same as [`TensorMut::from_raw_parts`].
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const unsafe fn make_tensor_write<'a, T: Copy + TensorElement<T>>(
    storage: *mut T,
    storage_len: usize,
) -> TensorMut<'a, T, Contiguous1D, T> {
    unsafe { TensorMut::from_raw_parts(storage, storage_len, Contiguous1D::new(storage_len)) }
}

/// Compiler boundary for splitting a read-only row into fixed-size tiles.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn zipped_divide_read<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Contiguous1D, T>,
) -> Tensor<'a, T, Zipped1D<TILE>, T> {
    const { assert!(TILE > 0) };
    Tensor::from_storage(
        tensor.storage,
        Zipped1D {
            len: tensor.layout.len,
        },
    )
}

/// Compiler boundary for splitting a writable row into fixed-size tiles.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub fn zipped_divide_write<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Contiguous1D, T>,
) -> TensorMut<'a, T, Zipped1D<TILE>, T> {
    const { assert!(TILE > 0) };
    TensorMut {
        storage: tensor.storage,
        storage_len: tensor.storage_len,
        layout: Zipped1D {
            len: tensor.layout.len,
        },
        element: PhantomData,
        borrow: PhantomData,
    }
}

/// Compiler boundary for selecting one read-only tile.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn slice_read<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Zipped1D<TILE>, T>,
    tile: usize,
) -> Tensor<'a, T, Tile1D<TILE>, T> {
    Tensor::from_storage(
        tensor.storage,
        Tile1D {
            base: tile.saturating_mul(TILE),
            len: tensor.layout.len,
        },
    )
}

/// Compiler boundary for selecting one writable tile.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub fn slice_write<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Zipped1D<TILE>, T>,
    tile: usize,
) -> TensorMut<'a, T, Tile1D<TILE>, T> {
    TensorMut {
        storage: tensor.storage,
        storage_len: tensor.storage_len,
        layout: Tile1D {
            base: tile.saturating_mul(TILE),
            len: tensor.layout.len,
        },
        element: PhantomData,
        borrow: PhantomData,
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> TensorMut<'a, T, Tile1D<TILE>, T> {
    /// Write one value at its absolute index in the original tensor.
    ///
    /// Use this for a short final tile; it is the same scalar store verb as
    /// `store_at` on an undivided row. Use [`TensorMut::store`] for a full
    /// tile so all values are written by one vector store.
    ///
    /// # Safety
    ///
    /// - `index` must be between this tile's [`Tile1D::base`] and the end of
    ///   its valid range.
    /// - No concurrently running thread may access the same element in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store_at(&mut self, index: usize, value: T) {
        unsafe { tensor_store_element_abs::<T, TILE>(self, index, value) };
    }

    /// Write a complete [`RegisterTile`] using one vector store.
    ///
    /// # Safety
    ///
    /// - [`Tile1D::is_full`] must be `true`.
    /// - The tile's first address must meet the alignment required by
    ///   [`crate::store_tile`].
    /// - `TILE * size_of::<T>()` must be a supported power of two from 4
    ///   through 16.
    /// - No concurrently running thread may access the same elements in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store(self, values: RegisterTile<T, TILE>) {
        unsafe { tensor_store_tile::<T, TILE>(self, values) };
    }
}

/// Compiler boundary for asking whether a selected tile is complete.
///
/// This stays public only so device compilation can identify the exact
/// definition across crate boundaries. Use [`Tensor::is_full`] in kernels.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn tensor_tile_is_full<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> bool {
    tensor.layout.is_full()
}

/// Compiler boundary for reading a selected tile's absolute base index.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn tensor_tile_base<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> usize {
    tensor.layout.base()
}

/// Compiler boundary for loading one complete selected tile.
///
/// # Safety
///
/// The rules are the same as [`Tensor::load`].
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub unsafe fn tensor_load_tile<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> RegisterTile<T, TILE> {
    let base = unsafe { assume::<TILE>(tensor.layout.base) };
    RegisterTile(unsafe { load_tile::<T, TILE>(tensor.storage, base) })
}

/// Compiler boundary for storing one complete selected tile.
///
/// # Safety
///
/// The rules are the same as [`TensorMut::store`].
#[doc(hidden)]
#[inline(never)]
pub unsafe fn tensor_store_tile<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Tile1D<TILE>, T>,
    values: RegisterTile<T, TILE>,
) {
    let base = unsafe { assume::<TILE>(tensor.layout.base) };
    unsafe { store_tile::<T, TILE>(tensor.storage, base, values.as_array()) };
}

/// Compiler boundary for one scalar store in a short final tile.
///
/// # Safety
///
/// The rules are the same as [`TensorMut::store_at`] on a selected tile.
#[doc(hidden)]
#[inline(never)]
pub unsafe fn tensor_store_element_abs<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: &mut TensorMut<'a, T, Tile1D<TILE>, T>,
    index: usize,
    value: T,
) {
    unsafe { *tensor.storage.add(index) = value };
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct logical encoding used to test a tensor backed by u8 storage.
    #[derive(Clone, Copy)]
    struct TestElement;

    impl TensorElement<u32> for TestElement {}

    /// A logical-element view must borrow the original slice without allocating or copying it.
    #[test]
    fn tensor_view_borrows_storage_without_allocating() {
        let storage = [0u32; 8];
        let tensor = Tensor::<TestElement, _, u32>::from_storage(&storage, (2usize, 4usize));

        assert_eq!(tensor.storage().as_ptr(), storage.as_ptr());
        assert_eq!(tensor.storage().len(), storage.len());
        assert_eq!(tensor.layout(), (2, 4));
    }

    /// Tiling and slicing must preserve the borrowed storage and expose a partial final tile.
    #[test]
    fn zipped_divide_and_slice_are_views_over_the_same_storage() {
        let storage = [0.0f32; 10];
        let tensor: Tensor<'_, f32, Contiguous1D> = Tensor::from_slice(&storage);
        let tiled = tensor.zipped_divide::<4>();

        assert_eq!(tiled.layout().tile_count(), 3);

        let middle = tiled.slice(1);
        assert_eq!(middle.storage().as_ptr(), storage.as_ptr());
        assert_eq!(middle.layout().base(), 4);
        assert!(middle.layout().is_full());

        let tail = tiled.slice(2);
        assert_eq!(tail.layout().base(), 8);
        assert!(!tail.layout().is_full());
    }

    /// The fused tile selection must agree with the two-step chain on a full and a ragged tile.
    #[test]
    fn local_tile_equals_zipped_divide_then_slice() {
        let storage = [0.0f32; 10];
        let tensor: Tensor<'_, f32, Contiguous1D> = Tensor::from_slice(&storage);
        let tiled = tensor.zipped_divide::<4>();

        // Tile 1 is full; tile 2 is the two-element tail.
        for tile in [1, 2] {
            let direct = tensor.local_tile::<4>(tile);
            let chained = tiled.slice(tile);
            assert_eq!(direct.storage().as_ptr(), chained.storage().as_ptr());
            assert_eq!(direct.base(), chained.base());
            assert_eq!(direct.is_full(), chained.is_full());
        }
        assert!(tensor.local_tile::<4>(1).is_full());
        assert!(!tensor.local_tile::<4>(2).is_full());

        let mut output = [0.0f32; 10];
        for tile in [1, 2] {
            // Each writable view is built, queried and dropped before the
            // next one, so no two views over `output` are alive together.
            let direct = unsafe { make_tensor_write(output.as_mut_ptr(), output.len()) }
                .local_tile::<4>(tile)
                .layout();
            let chained = unsafe { make_tensor_write(output.as_mut_ptr(), output.len()) }
                .zipped_divide::<4>()
                .slice(tile)
                .layout();
            assert_eq!(direct.base(), chained.base());
            assert_eq!(direct.is_full(), chained.is_full());
            assert_eq!(direct.is_full(), tile == 1);
        }
    }

    /// Register-tile addition must preserve element order and produce per-element sums.
    #[test]
    fn register_tiles_add_elementwise() {
        let a = RegisterTile([1.0f32, 2.0, 3.0, 4.0]);
        let b = RegisterTile([10.0f32, 20.0, 30.0, 40.0]);

        assert_eq!((a + b).into_array(), [11.0, 22.0, 33.0, 44.0]);
    }

    /// Mutable tensor markers and lifetime tracking must add no bytes beyond pointer, length and layout.
    #[test]
    fn mutable_tensor_is_not_larger_than_its_runtime_fields() {
        assert_eq!(
            core::mem::size_of::<TensorMut<'static, f32, Contiguous1D, f32>>(),
            3 * core::mem::size_of::<usize>()
        );
    }
}
