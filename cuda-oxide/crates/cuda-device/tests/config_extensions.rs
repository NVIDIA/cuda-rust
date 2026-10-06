/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::config::{
    AtomKind, AtomSpec, Global, Layout, MemorySpace, Scope, Shape, Shape1, Shape2, Tile, TileSpec,
};

/// An abstraction library can expose a layout knob cuda-oxide does not know.
enum Swizzled<const BYTES: usize> {}
impl<const BYTES: usize> Layout for Swizzled<BYTES> {}

/// The same is true for a domain-specific cooperation group.
enum WarpPair {}
impl Scope for WarpPair {}

/// Richer library descriptors can still participate in generic policy traits.
struct NamedTile<T>(core::marker::PhantomData<T>);

impl<T: TileSpec> TileSpec for NamedTile<T> {
    type Shape = T::Shape;
    type Layout = T::Layout;
    type MemorySpace = T::MemorySpace;
    type Scope = T::Scope;
}

enum LibraryAtom {}
impl AtomKind for LibraryAtom {}

struct NamedAtom;
impl AtomSpec for NamedAtom {
    type Kind = LibraryAtom;
    type Shape = Shape1<1>;
    type Scope = WarpPair;
}

#[test]
fn metadata_vocabulary_is_open_to_abstraction_libraries() {
    type LibraryTile = NamedTile<Tile<Shape2<16, 32>, Swizzled<128>, Global, WarpPair>>;

    fn assert_tile<T>()
    where
        T: TileSpec<
                Shape = Shape2<16, 32>,
                Layout = Swizzled<128>,
                MemorySpace = Global,
                Scope = WarpPair,
            >,
    {
    }

    fn assert_atom<A: AtomSpec<Kind = LibraryAtom, Shape = Shape1<1>, Scope = WarpPair>>() {}

    assert_tile::<LibraryTile>();
    assert_atom::<NamedAtom>();
    _accept_library_space::<LibrarySpace>();
    assert_eq!(
        <<LibraryTile as TileSpec>::Shape as Shape>::ELEMENTS,
        Some(512)
    );
}

// Keep all four extension traits exercised from outside the cuda_device crate.
enum LibrarySpace {}
impl MemorySpace for LibrarySpace {}

fn _accept_library_space<M: MemorySpace>() {}
