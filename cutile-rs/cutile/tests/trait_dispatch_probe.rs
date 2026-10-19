/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Probe for the trait-dispatch emission pattern used by `cutile-macro`.
//!
//! Validates that the "single rank-polymorphic trait + per-rank impls + free-fn
//! wrapper" shape compiles cleanly and lets rustc resolve calls through
//! nested expressions. Uses toy types (not cuTile's real ones) so the
//! pattern can be exercised in isolation as a regression test against the
//! emitter design itself.
//!
//! Covers:
//! - Rank-preserving same-shape op (`addf`) — representative of case 3a
//! - Rank-changing op (`reshape`) — representative of case 3b
//! - Nested expressions `addf(addf(a, b, ...), c, ...)`
//! - Mixed ops `reshape(addf(a, b, ...), shape)`
//! - Method-form calls (`a.addf(b, ...)`) resolving via trait dispatch

#![allow(non_camel_case_types)]

use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Toy per-rank types (what the variadic-struct macro stamps out)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct Tile_1<E, const D0: i32>(PhantomData<E>);
#[derive(Copy, Clone)]
pub struct Tile_2<E, const D0: i32, const D1: i32>(PhantomData<E>);
#[derive(Copy, Clone)]
pub struct Tile_3<E, const D0: i32, const D1: i32, const D2: i32>(PhantomData<E>);

#[derive(Copy, Clone)]
pub struct Shape_1<const D0: i32>;
#[derive(Copy, Clone)]
pub struct Shape_2<const D0: i32, const D1: i32>;
#[derive(Copy, Clone)]
pub struct Shape_3<const D0: i32, const D1: i32, const D2: i32>;

#[derive(Copy, Clone)]
pub struct Tensor_1<E, const D0: i32>(PhantomData<E>);
#[derive(Copy, Clone)]
pub struct Tensor_2<E, const D0: i32, const D1: i32>(PhantomData<E>);
#[derive(Copy, Clone)]
pub struct Tensor_3<E, const D0: i32, const D1: i32, const D2: i32>(PhantomData<E>);

// Constructors so tests can build values without needing real DSL machinery.
impl<E, const D0: i32> Default for Tile_1<E, D0> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32> Tile_1<E, D0> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<E, const D0: i32, const D1: i32> Default for Tile_2<E, D0, D1> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32, const D1: i32> Tile_2<E, D0, D1> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<E, const D0: i32, const D1: i32, const D2: i32> Default for Tile_3<E, D0, D1, D2> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32, const D1: i32, const D2: i32> Tile_3<E, D0, D1, D2> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<E, const D0: i32> Default for Tensor_1<E, D0> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32> Tensor_1<E, D0> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<E, const D0: i32, const D1: i32> Default for Tensor_2<E, D0, D1> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32, const D1: i32> Tensor_2<E, D0, D1> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<E, const D0: i32, const D1: i32, const D2: i32> Default for Tensor_3<E, D0, D1, D2> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, const D0: i32, const D1: i32, const D2: i32> Tensor_3<E, D0, D1, D2> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

// Mock of `cuda_tile.get_tile_block_id()` — in real kernels this would
// return (blockIdx.x, blockIdx.y, blockIdx.z) at runtime.
fn get_tile_block_id() -> (i32, i32, i32) {
    (0, 0, 0)
}

// ---------------------------------------------------------------------------
// ZST modifier modules
// ---------------------------------------------------------------------------

pub mod rounding {
    pub trait Mode {}
    pub struct NearestEven;
    pub struct Zero;
    impl Mode for NearestEven {}
    impl Mode for Zero {}
}

pub mod ftz {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

// ---------------------------------------------------------------------------
// Case 3a: rank-preserving same-shape (addf)
// ---------------------------------------------------------------------------

pub trait AddF<R, F>
where
    R: rounding::Mode,
    F: ftz::Mode,
{
    fn addf(self, other: Self, r: R, f: F) -> Self;
}

impl<E: Copy, R: rounding::Mode, F: ftz::Mode, const D0: i32> AddF<R, F> for Tile_1<E, D0> {
    fn addf(self, _other: Self, _r: R, _f: F) -> Self {
        self
    }
}

impl<E: Copy, R: rounding::Mode, F: ftz::Mode, const D0: i32, const D1: i32> AddF<R, F>
    for Tile_2<E, D0, D1>
{
    fn addf(self, _other: Self, _r: R, _f: F) -> Self {
        self
    }
}

impl<E: Copy, R: rounding::Mode, F: ftz::Mode, const D0: i32, const D1: i32, const D2: i32>
    AddF<R, F> for Tile_3<E, D0, D1, D2>
{
    fn addf(self, _other: Self, _r: R, _f: F) -> Self {
        self
    }
}

pub fn addf<T, R, F>(a: T, b: T, r: R, f: F) -> T
where
    T: AddF<R, F>,
    R: rounding::Mode,
    F: ftz::Mode,
{
    a.addf(b, r, f)
}

// ---------------------------------------------------------------------------
// Case 3b: rank-changing (reshape)
// ---------------------------------------------------------------------------

pub trait Reshape<Sh> {
    type Out;
    fn reshape(self, shape: Sh) -> Self::Out;
}

// Rank 1 → Rank 2
impl<E: Copy, const S0: i32, const T0: i32, const T1: i32> Reshape<Shape_2<T0, T1>>
    for Tile_1<E, S0>
{
    type Out = Tile_2<E, T0, T1>;
    fn reshape(self, _shape: Shape_2<T0, T1>) -> Tile_2<E, T0, T1> {
        Tile_2::new()
    }
}

// Rank 2 → Rank 1
impl<E: Copy, const S0: i32, const S1: i32, const T0: i32> Reshape<Shape_1<T0>>
    for Tile_2<E, S0, S1>
{
    type Out = Tile_1<E, T0>;
    fn reshape(self, _shape: Shape_1<T0>) -> Tile_1<E, T0> {
        Tile_1::new()
    }
}

// Rank 2 → Rank 2
impl<E: Copy, const S0: i32, const S1: i32, const T0: i32, const T1: i32> Reshape<Shape_2<T0, T1>>
    for Tile_2<E, S0, S1>
{
    type Out = Tile_2<E, T0, T1>;
    fn reshape(self, _shape: Shape_2<T0, T1>) -> Tile_2<E, T0, T1> {
        Tile_2::new()
    }
}

// Rank 2 → Rank 3
impl<E: Copy, const S0: i32, const S1: i32, const T0: i32, const T1: i32, const T2: i32>
    Reshape<Shape_3<T0, T1, T2>> for Tile_2<E, S0, S1>
{
    type Out = Tile_3<E, T0, T1, T2>;
    fn reshape(self, _shape: Shape_3<T0, T1, T2>) -> Tile_3<E, T0, T1, T2> {
        Tile_3::new()
    }
}

pub fn reshape<Src, Sh>(src: Src, shape: Sh) -> Src::Out
where
    Src: Reshape<Sh>,
{
    src.reshape(shape)
}

// ---------------------------------------------------------------------------
// load_tile_like: rank-specific BODIES per impl
//
// This is the case that motivated the old rank-specific load_tile_like helpers
// in _core.rs. The body genuinely differs per rank (different number of pid
// components in the index) — not a simple substitution. Each rank impl is
// hand-written with the rank-appropriate body.
//
// The call site uses one name (`load_tile_like`) and rustc dispatches to the
// right rank based on the input tensor's concrete type.
// ---------------------------------------------------------------------------

pub trait LoadTileLike<Y> {
    type Out;
    fn load_tile_like(x: &Self, y: &Y) -> Self::Out;
}

// Rank 1: dynamic-shape input `Tensor_1<E1, -1>`, specific-shape reference
// `Tensor_1<E2, S0>`. Body indexes with a single pid component.
impl<E1: Copy, E2: Copy, const S0: i32> LoadTileLike<Tensor_1<E2, S0>> for Tensor_1<E1, -1> {
    type Out = Tile_1<E1, S0>;
    fn load_tile_like(_x: &Self, _y: &Tensor_1<E2, S0>) -> Tile_1<E1, S0> {
        let _pid = get_tile_block_id();
        // In the real DSL this would call `load_view_tko(&partition, [_pid.0], ordering::Weak, scope::TileBlock, None, tma::Enabled)`.
        // The rank-1 body uses a 1-element index array.
        let _idx: [i32; 1] = [_pid.0];
        Tile_1::new()
    }
}

// Rank 2: dynamic `Tensor_2<E1, -1, -1>`, specific `Tensor_2<E2, S0, S1>`.
// Body indexes with two pid components.
impl<E1: Copy, E2: Copy, const S0: i32, const S1: i32> LoadTileLike<Tensor_2<E2, S0, S1>>
    for Tensor_2<E1, -1, -1>
{
    type Out = Tile_2<E1, S0, S1>;
    fn load_tile_like(_x: &Self, _y: &Tensor_2<E2, S0, S1>) -> Tile_2<E1, S0, S1> {
        let _pid = get_tile_block_id();
        // Rank-2 body uses a 2-element index array — different from rank 1.
        let _idx: [i32; 2] = [_pid.0, _pid.1];
        Tile_2::new()
    }
}

// Rank 3: body indexes with three pid components.
impl<E1: Copy, E2: Copy, const S0: i32, const S1: i32, const S2: i32>
    LoadTileLike<Tensor_3<E2, S0, S1, S2>> for Tensor_3<E1, -1, -1, -1>
{
    type Out = Tile_3<E1, S0, S1, S2>;
    fn load_tile_like(_x: &Self, _y: &Tensor_3<E2, S0, S1, S2>) -> Tile_3<E1, S0, S1, S2> {
        let _pid = get_tile_block_id();
        let _idx: [i32; 3] = [_pid.0, _pid.1, _pid.2];
        Tile_3::new()
    }
}

pub fn load_tile_like<X, Y>(x: &X, y: &Y) -> <X as LoadTileLike<Y>>::Out
where
    X: LoadTileLike<Y>,
{
    <X as LoadTileLike<Y>>::load_tile_like(x, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn simple_call_resolves() {
    let a: Tile_2<f32, 128, 256> = Tile_2::new();
    let b: Tile_2<f32, 128, 256> = Tile_2::new();
    let c = addf(a, b, rounding::NearestEven, ftz::Disabled);
    // rustc inferred T = Tile_2<f32, 128, 256>; result type matches.
    let _: Tile_2<f32, 128, 256> = c;
}

#[test]
fn nested_addf_resolves_without_annotations() {
    // This is the case that today's variadic expansion breaks on:
    // a macro-level rewriter can't see through the nested call to infer the
    // rank of the inner addf's result. Trait dispatch handles it via type
    // inference at rustc level.
    let a: Tile_2<f32, 64, 64> = Tile_2::new();
    let b: Tile_2<f32, 64, 64> = Tile_2::new();
    let c: Tile_2<f32, 64, 64> = Tile_2::new();
    let result = addf(
        addf(a, b, rounding::NearestEven, ftz::Disabled),
        c,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let _: Tile_2<f32, 64, 64> = result;
}

#[test]
fn reshape_then_addf_nested() {
    // Mixing rank-changing and same-shape ops in a nested chain — rustc
    // threads through the associated-type output of reshape into addf's
    // Self-constrained arg.
    let a: Tile_1<f32, 8> = Tile_1::new();
    let b: Tile_2<f32, 4, 2> = Tile_2::new();
    let shape: Shape_2<4, 2> = Shape_2;
    let result = addf(reshape(a, shape), b, rounding::NearestEven, ftz::Disabled);
    let _: Tile_2<f32, 4, 2> = result;
}

#[test]
fn method_form_chains_through_ranks() {
    // Method chains via trait dispatch: each link's return type is the
    // driver for the next method's Self.
    let a: Tile_1<f32, 8> = Tile_1::new();
    let shape_2: Shape_2<4, 2> = Shape_2;
    let shape_3: Shape_3<2, 2, 2> = Shape_3;
    let result = a.reshape(shape_2).reshape(shape_3);
    let _: Tile_3<f32, 2, 2, 2> = result;
}

#[test]
fn deeply_nested_mixed_ops() {
    // Worst case for inference: four levels of nested calls mixing addf and
    // reshape across multiple ranks. If this compiles, the pattern holds.
    let a: Tile_1<f32, 8> = Tile_1::new();
    let b: Tile_1<f32, 8> = Tile_1::new();
    let c: Tile_2<f32, 4, 2> = Tile_2::new();
    let d: Tile_2<f32, 4, 2> = Tile_2::new();
    let shape: Shape_2<4, 2> = Shape_2;

    let result = addf(
        reshape(addf(a, b, rounding::NearestEven, ftz::Disabled), shape),
        addf(c, d, rounding::NearestEven, ftz::Disabled),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let _: Tile_2<f32, 4, 2> = result;
}

#[test]
fn load_tile_like_rank_1() {
    let input: Tensor_1<f32, -1> = Tensor_1::new();
    let output: Tensor_1<f32, 128> = Tensor_1::new();
    let tile = load_tile_like(&input, &output);
    // rustc resolves via (X = Tensor_1<f32, -1>, Y = Tensor_1<f32, 128>),
    // picks the rank-1 impl, output type is Tile_1<f32, 128>.
    let _: Tile_1<f32, 128> = tile;
}

#[test]
fn load_tile_like_rank_2() {
    let input: Tensor_2<f32, -1, -1> = Tensor_2::new();
    let output: Tensor_2<f32, 64, 128> = Tensor_2::new();
    let tile = load_tile_like(&input, &output);
    let _: Tile_2<f32, 64, 128> = tile;
}

#[test]
fn load_tile_like_rank_3() {
    let input: Tensor_3<f32, -1, -1, -1> = Tensor_3::new();
    let output: Tensor_3<f32, 8, 16, 32> = Tensor_3::new();
    let tile = load_tile_like(&input, &output);
    let _: Tile_3<f32, 8, 16, 32> = tile;
}

#[test]
fn load_tile_like_different_element_types_resolve() {
    // Element types may differ; output tile uses the source tensor element.
    let input: Tensor_2<f32, -1, -1> = Tensor_2::new();
    let output: Tensor_2<i32, 64, 128> = Tensor_2::new();
    let tile = load_tile_like(&input, &output);
    let _: Tile_2<f32, 64, 128> = tile;
}

#[test]
fn load_tile_like_composed_with_addf() {
    // A realistic nested use: load a tile from each input tensor, add them,
    // and get a concrete typed result. Tests that load_tile_like's output
    // type flows through trait dispatch of addf without annotations.
    let x: Tensor_2<f32, -1, -1> = Tensor_2::new();
    let y: Tensor_2<f32, -1, -1> = Tensor_2::new();
    let out: Tensor_2<f32, 64, 128> = Tensor_2::new();

    let tx = load_tile_like(&x, &out);
    let ty = load_tile_like(&y, &out);
    let result = addf(tx, ty, rounding::NearestEven, ftz::Disabled);
    let _: Tile_2<f32, 64, 128> = result;
}

#[test]
fn shape_mismatch_fails_at_compile_time() {
    // Positive side of the mismatch check: if we MATCH shapes, same-shape
    // resolves. The negative side (mismatch = compile error) can't be
    // asserted in a regular #[test] — it would fail the build. This test
    // exists to document that the positive path works; see the commented-
    // out code below for what would fail if uncommented.
    let a: Tile_2<f32, 128, 256> = Tile_2::new();
    let b: Tile_2<f32, 128, 256> = Tile_2::new();
    let _ = addf(a, b, rounding::NearestEven, ftz::Disabled);

    // The following would fail to compile with a type mismatch error:
    //
    //   let a: Tile_2<f32, 128, 256> = Tile_2::new();
    //   let b: Tile_2<f32, 64, 64>   = Tile_2::new();
    //   let _ = addf(a, b, rounding::NearestEven, ftz::Disabled);
    //   // error: expected `Tile_2<f32, 128, 256>`, found `Tile_2<f32, 64, 64>`
    //
    // This confirms case-3a same-shape unification is preserved through the
    // trait-dispatch pattern: `fn addf<T>(a: T, b: T, ...) -> T` forces both
    // args to the same concrete type.
}
