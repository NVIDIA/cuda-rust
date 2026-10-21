/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device tests for the aggregation capabilities, `AddTensor` through
//! `MinTensor`.
//!
//! The compile-only suite (`tests/aggregation_tensor.rs`) proves each checked
//! operation emits the intended `cuda_tile.atomic_red_view_tko` with the
//! right mode. These tests prove the emitted op does the intended arithmetic — every program's contribution
//! lands, none is lost to a racing read-modify-write, and the sum matches a
//! host oracle — through both host bindings, the exclusive borrow
//! (`tensor.atomic()`) and the owned one (`Atomic::owned`).
//!
//! The value kernels are split-K: the grid's last axis carries several
//! programs per output tile, which is exactly the overlap an ordinary
//! `&mut Tensor` output would forbid.
//!
//! The last two tests are about the host's within-launch alias rule rather
//! than arithmetic: one allocation bound both atomically and ordinarily is
//! rejected before any GPU work, while two atomic bindings over one
//! allocation stay accepted.

use crate::common;
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use std::sync::Arc;

const BM: usize = 16;
const BN: usize = 16;
const BK: usize = 32;

#[cutile::module]
mod atomic_value_kernels {
    use cutile::core::*;

    /// Every program adds a ones-tile into output tile `[m, n]`. With a grid
    /// of `(tiles_m, tiles_n, splits)` each tile must end at exactly
    /// `splits`; a non-atomic store would lose updates and land lower.
    #[cutile::entry()]
    fn accumulate_ones(out: &AddTensor<f32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let ones: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([bid.0, bid.1], ones);
    }

    #[cutile::entry()]
    fn accumulate_f16_ones(out: &AddTensor<f16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let ones: Tile<f16, { [16, 16] }> = constant(f16::ONE, shape![16, 16]);
        let acc: AddPartition<f16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([bid.0, bid.1], ones);
    }

    #[cutile::entry()]
    fn accumulate_bf16_ones(out: &AddTensor<bf16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let ones: Tile<bf16, { [16, 16] }> = constant(bf16::ONE, shape![16, 16]);
        let acc: AddPartition<bf16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([bid.0, bid.1], ones);
    }

    /// Split-K matmul: program `(m, n, k)` contributes `lhs[m, k] @ rhs[k, n]`
    /// to output tile `[m, n]`. The K axis is divided across the grid, so the
    /// output tile is only correct once every program's partial product has
    /// been accumulated.
    #[cutile::entry()]
    fn split_k_matmul(
        out: &AddTensor<f32, { [-1, -1] }>,
        lhs: &Tensor<f32, { [-1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let m: i32 = bid.0;
        let n: i32 = bid.1;
        let k: i32 = bid.2;

        let lhs_part: Partition<f32, { [16, 32] }> = lhs.partition(shape![16, 32]);
        let rhs_part: Partition<f32, { [32, 16] }> = rhs.partition(shape![32, 16]);
        let a: Tile<f32, { [16, 32] }> = lhs_part.load([m, k]);
        let b: Tile<f32, { [32, 16] }> = rhs_part.load([k, n]);
        let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        let partial: Tile<f32, { [16, 16] }> = mma(a, b, zero);

        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([m, n], partial);
    }
}

use atomic_value_kernels::{
    accumulate_bf16_ones, accumulate_f16_ones, accumulate_ones, split_k_matmul,
};

/// Lost updates are the failure this capability exists to prevent, so the
/// split count is high enough that a racing read-modify-write loses some.
const SPLITS: u32 = 64;

fn assert_atomic_add_ones<T, L>(mut out: Tensor<T>, grid: (u32, u32, u32), expected: T, launch: L)
where
    T: DType + PartialEq + std::fmt::Debug + Copy,
    L: FnOnce(Atomic<&mut Tensor<T>>, (u32, u32, u32)),
{
    launch(out.atomic(), grid);
    let host: Vec<T> = out.dup().to_host_vec().sync().expect("to_host");
    assert_uniform(&host, expected, "atomic add");
}

#[test]
fn f16_atomic_add_accumulates_every_contribution() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles = (2u32, 3u32);
        let out = api::zeros::<f16>(&[tiles.0 as usize * BM, tiles.1 as usize * BN])
            .sync()
            .expect("zeros");
        assert_atomic_add_ones(
            out,
            (tiles.0, tiles.1, SPLITS),
            f16::from_f32(SPLITS as f32),
            |out, grid| {
                let _ = accumulate_f16_ones(out)
                    .grid(grid)
                    .sync()
                    .expect("f16 atomic add");
            },
        );
    });
}

#[test]
fn bf16_atomic_add_accumulates_every_contribution() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[
            cutile_ir::requirements::Feature::AtomicRedView,
            cutile_ir::requirements::Feature::Bf16AtomicAdd,
        ]) {
            return;
        }
        let tiles = (2u32, 3u32);
        let out = api::zeros::<bf16>(&[tiles.0 as usize * BM, tiles.1 as usize * BN])
            .sync()
            .expect("zeros");
        assert_atomic_add_ones(
            out,
            (tiles.0, tiles.1, SPLITS),
            bf16::from_f32(SPLITS as f32),
            |out, grid| {
                let _ = accumulate_bf16_ones(out)
                    .grid(grid)
                    .sync()
                    .expect("bf16 atomic add");
            },
        );
    });
}

#[test]
fn every_contribution_lands() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles_m = 2u32;
        let tiles_n = 3u32;
        let rows = tiles_m as usize * BM;
        let cols = tiles_n as usize * BN;

        // `atomic_add` accumulates; the identity has to be there first.
        let mut out = api::zeros::<f32>(&[rows, cols]).sync().expect("zeros");
        // The borrowed binding writes in place; dropping the recovered
        // `&mut` here ends the borrow so `out` can be read below.
        let _ = accumulate_ones(out.atomic())
            .grid((tiles_m, tiles_n, SPLITS))
            .sync()
            .expect("launch");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        let expected = SPLITS as f32;
        // Integers up to 2^24 are exact in f32, so this is an equality, not a
        // tolerance: any deviation is a lost or duplicated update.
        let wrong = host.iter().filter(|&&v| v != expected).count();
        assert_eq!(
            wrong,
            0,
            "{wrong} of {} elements are not {expected}; first few: {:?}",
            host.len(),
            &host[..8.min(host.len())]
        );
    });
}

#[test]
fn split_k_matmul_matches_a_host_oracle() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles_m = 2usize;
        let tiles_n = 2usize;
        let splits = 4usize;
        let (m, n, k) = (tiles_m * BM, tiles_n * BN, splits * BK);

        // Small signed values keep every partial product exact in f32, so the
        // only inexactness left is the order the atomics add them in.
        let lhs_host: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let rhs_host: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for step in 0..k {
                    acc += lhs_host[row * k + step] * rhs_host[step * n + col];
                }
                expected[row * n + col] = acc;
            }
        }

        let lhs: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(lhs_host))
            .reshape(&[m, k])
            .sync()
            .expect("lhs")
            .into();
        let rhs: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(rhs_host))
            .reshape(&[k, n])
            .sync()
            .expect("rhs")
            .into();

        let mut out = api::zeros::<f32>(&[m, n]).sync().expect("zeros");
        let _ = split_k_matmul(out.atomic(), lhs, rhs)
            .grid((tiles_m as u32, tiles_n as u32, splits as u32))
            .sync()
            .expect("launch");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), m * n);

        // Atomic float addition has no fixed summation order, so this is a
        // tolerance and not an equality — and deliberately so: the capability
        // does not promise bit-reproducibility. A lost contribution is a whole
        // tile's worth of error and cannot hide inside this bound.
        let mut worst = 0.0f32;
        for (got, want) in host.iter().zip(expected.iter()) {
            let scale = want.abs().max(1.0);
            worst = worst.max((got - want).abs() / scale);
        }
        assert!(
            worst < 1e-5,
            "worst relative error {worst} exceeds tolerance; \
             got {:?} want {:?}",
            &host[..4.min(host.len())],
            &expected[..4.min(expected.len())]
        );
    });
}

// ── The owned binding ───────────────────────────────────────────────────────

/// The owned binding, end to end on a device.
///
/// `Atomic::owned` is the other exclusive host authority: ownership rather
/// than an exclusive borrow. Until now it was only typechecked
/// (`tests/aggregation_tensor.rs::owned_host_binding_typechecks`), so nothing
/// exercised preparation, launch, and recovery through it, and nothing checked
/// that the tensor the launch hands back holds the accumulated values. The
/// launch *takes* the tensor, so recovery is the only way back to the data —
/// which makes the recovered `Tensor<f32>` the thing under test, not a
/// convenience.
#[test]
fn an_owned_binding_accumulates_and_recovers_the_tensor() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles_m = 2u32;
        let tiles_n = 3u32;
        let rows = tiles_m as usize * BM;
        let cols = tiles_n as usize * BN;

        let out = api::zeros::<f32>(&[rows, cols]).sync().expect("zeros");
        // A one-parameter kernel recovers a one-tuple; the annotation pins
        // that the owned binding comes back as the plain `Tensor<f32>` it was,
        // not as the `Atomic<..>` wrapper.
        let (recovered,): (Tensor<f32>,) = accumulate_ones(Atomic::owned(out))
            .grid((tiles_m, tiles_n, SPLITS))
            .sync()
            .expect("launch");

        let host: Vec<f32> = recovered.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        let expected = SPLITS as f32;
        // Integers up to 2^24 are exact in f32, so this is an equality, not a
        // tolerance: any deviation is a lost or duplicated update.
        let wrong = host.iter().filter(|&&v| v != expected).count();
        assert_eq!(
            wrong,
            0,
            "{wrong} of {} elements are not {expected}; first few: {:?}",
            host.len(),
            &host[..8.min(host.len())]
        );
    });
}

// ── One allocation, two bindings ────────────────────────────────────────────

#[cutile::module]
mod alias_kernels {
    use cutile::core::*;

    /// One atomic destination and one ordinary input. Binding both to the
    /// same allocation is the race the host must reject: the two parameters
    /// are independent views, so no token ordering inside the kernel fixes
    /// it.
    ///
    /// The contribution is `other + 1`, which is nonzero whatever `other`
    /// holds, so a launch that reached the device leaves a mark in `out`.
    #[cutile::entry()]
    fn atomic_and_ordinary(out: &AddTensor<f32, { [-1, -1] }>, other: &Tensor<f32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [16, 16] }> = other.partition(shape![16, 16]);
        let seen: Tile<f32, { [16, 16] }> = part.load([bid.0, bid.1]);
        let ones: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([bid.0, bid.1], seen + ones);
    }

    /// Two atomic destinations. Binding both to one allocation is exactly
    /// what the capability is for, because overlapping atomic updates
    /// compose, so the alias check must leave it alone.
    #[cutile::entry()]
    fn two_atomic_destinations(
        first: &AddTensor<f32, { [-1, -1] }>,
        second: &AddTensor<f32, { [-1, -1] }>,
    ) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let one: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let two: Tile<f32, { [16, 16] }> = constant(2.0f32, shape![16, 16]);
        let a: AddPartition<f32, { [16, 16] }> = first.partition(shape![16, 16]);
        let b: AddPartition<f32, { [16, 16] }> = second.partition(shape![16, 16]);
        let _x: Token = a.atomic_add([bid.0, bid.1], one);
        let _y: Token = b.atomic_add([bid.0, bid.1], two);
    }
}

use alias_kernels::{atomic_and_ordinary, two_atomic_destinations};

/// One allocation bound both atomically and ordinarily is rejected, and
/// rejected before anything runs.
///
/// "Before any GPU work" is asserted, not inferred. Two independent facts
/// carry it:
///
/// 1. The error is *that* rejection — it names both parameters and says one
///    is bound atomically — so a launch that failed for some other reason
///    cannot pass this test.
/// 2. The destination is unchanged. It was zeroed, and every program of this
///    kernel adds a strictly positive tile to its output tile, so any program
///    that reached the device would have left `out` non-zero.
///
/// Neither fact depends on a valid kernel executing afterwards: the test
/// still fails if nothing on this machine could ever launch.
#[test]
fn one_allocation_bound_atomically_and_ordinarily_is_rejected_before_any_gpu_work() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles = 2u32;
        let rows = tiles as usize * BM;
        let cols = tiles as usize * BN;
        let mut out = api::zeros::<f32>(&[rows, cols]).sync().expect("zeros");

        // Deliberate aliasing: one allocation, two bindings. `into_shared_alias`
        // is `unsafe` for exactly the reason under test, and the launch is
        // supposed to be what catches it.
        let alias: Arc<Tensor<f32>> = unsafe { out.into_shared_alias() };
        assert_eq!(
            alias.storage_id(),
            out.storage_id(),
            "the alias must name the same allocation, or this test proves nothing"
        );

        let result = atomic_and_ordinary(out.atomic(), alias)
            .grid((tiles, tiles, SPLITS))
            .sync();
        let Err(error) = result else {
            panic!("binding one allocation atomically and ordinarily must fail the launch");
        };
        let message = error.to_string();
        for fragment in ["`out`", "`other`", "same allocation", "is bound atomically"] {
            assert!(
                message.contains(fragment),
                "the rejection must be the alias check and must explain itself; \
                 missing {fragment:?} in: {message}"
            );
        }

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        let touched = host.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            touched,
            0,
            "{touched} elements changed, so the launch reached the device before \
             being rejected; first few: {:?}",
            &host[..8.min(host.len())]
        );
    });
}

/// Two atomic bindings over one allocation stay accepted — the alias check
/// rejects a mixed pair, not aliasing itself.
///
/// Both programs' contributions land in the same storage, so each element
/// ends at `1.0 + 2.0`.
#[test]
fn two_atomic_bindings_over_one_allocation_are_accepted() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let tiles = 2u32;
        let rows = tiles as usize * BM;
        let cols = tiles as usize * BN;
        let mut out = api::zeros::<f32>(&[rows, cols]).sync().expect("zeros");

        // A second *owned* tensor over the same storage, which is what the
        // owned atomic binding needs. A freshly minted alias is the only
        // reference to its own `Arc`, so unwrapping it always succeeds.
        let Ok(second) = Arc::try_unwrap(unsafe { out.into_shared_alias() }) else {
            panic!("a freshly minted alias is the only reference to itself");
        };
        assert_eq!(
            second.storage_id(),
            out.storage_id(),
            "both bindings must name the same allocation"
        );

        // The recovered bindings are dropped here, which ends the borrow of
        // `out` so it can be read below.
        let _ = two_atomic_destinations(out.atomic(), Atomic::owned(second))
            .grid((tiles, tiles, 1))
            .sync()
            .expect("two atomic bindings over one allocation must be accepted");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        let wrong = host.iter().filter(|&&v| v != 3.0).count();
        assert_eq!(
            wrong,
            0,
            "{wrong} of {} elements are not 3.0; first few: {:?}",
            host.len(),
            &host[..8.min(host.len())]
        );
    });
}

#[cutile::module]
mod per_operation_kernels {
    use cutile::core::*;

    /// OR is idempotent, so any number of overlapping programs must leave
    /// exactly `seed | 0x00A5_0F3C`.
    #[cutile::entry()]
    fn or_pattern(out: &OrTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(0x00A5_0F3Cu32, shape![16, 16]);
        let acc: OrPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }

    /// AND is idempotent, so starting from all-ones the result must be
    /// exactly the mask.
    #[cutile::entry()]
    fn and_mask(out: &AndTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(0x0F0F_F0F0u32, shape![16, 16]);
        let acc: AndPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }

    /// Every program XORs the same value; an even number of programs must
    /// restore the seed exactly, so a lost or duplicated update shows.
    #[cutile::entry()]
    fn xor_toggle(out: &XorTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(0xA5A5_A5A5u32, shape![16, 16]);
        let acc: XorPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }

    /// Unsigned maximum where contribution and seed straddle the high bit:
    /// the seed is `0x7FFF_FFF0` and the contribution `0x8000_0010`. Under
    /// the correct unsigned comparison the high-bit value wins; a signed
    /// comparison would read it as negative and keep the seed, a visibly
    /// wrong result.
    #[cutile::entry()]
    fn max_straddles_the_high_bit(out: &MaxTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(0x8000_0010u32, shape![16, 16]);
        let acc: MaxPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    /// Unsigned minimum across the same straddle, reversed: the seed is
    /// `0x8000_0010` and the contribution `0x7FFF_FFF0`, which must win; a
    /// signed comparison would keep the "negative" seed.
    #[cutile::entry()]
    fn min_straddles_the_high_bit(out: &MinTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(0x7FFF_FFF0u32, shape![16, 16]);
        let acc: MinPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }

    /// Signed maximum of seed `2` and contribution `-5`: the signed answer
    /// keeps `2`; an unsigned comparison would read `-5` as `0xFFFF_FFFB`
    /// and take it.
    #[cutile::entry()]
    fn max_signed_negatives(out: &MaxTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(-5i32, shape![16, 16]);
        let acc: MaxPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    /// Integer addition is exact at any count, unlike the float split-K
    /// above, so the per-element count is an equality.
    #[cutile::entry()]
    fn add_u32_ones(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
}

/// Seeds a `(2 tiles, 3 tiles)` u32 destination with `seed`, launches over a
/// `(2, 3, splits)` grid, and returns the device result.
fn run_u32<L>(seed: u32, splits: u32, launch: L) -> Vec<u32>
where
    L: FnOnce(Atomic<&mut Tensor<u32>>, (u32, u32, u32)),
{
    let (tiles_m, tiles_n) = (2u32, 3u32);
    let rows = tiles_m as usize * BM;
    let cols = tiles_n as usize * BN;
    let mut out = api::full::<u32>(seed, &[rows, cols]).sync().expect("seed");
    launch(out.atomic(), (tiles_m, tiles_n, splits));
    out.dup().to_host_vec().sync().expect("to_host")
}

/// Every element of `host` must equal `expected`; reports how many differ
/// and shows the first few, so a wrong comparison or lost update is legible.
fn assert_uniform<T: PartialEq + std::fmt::Debug + Copy>(host: &[T], expected: T, what: &str) {
    let wrong = host.iter().filter(|&&v| v != expected).count();
    assert_eq!(
        wrong,
        0,
        "{what}: {wrong} of {} elements are not {expected:?}; first few: {:?}",
        host.len(),
        &host[..8.min(host.len())]
    );
}

/// OR, AND and XOR against bit-exact references: idempotent OR and AND land
/// exactly on `seed | v` and `seed & v`, and an even XOR toggle count
/// restores the seed, so any lost or duplicated update is visible.
#[test]
fn bitwise_operations_match_bit_exact_references() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let or = run_u32(0x5000_0001, 4, |out, grid| {
            let _ = per_operation_kernels::or_pattern(out)
                .grid(grid)
                .sync()
                .expect("or");
        });
        assert_uniform(&or, 0x5000_0001 | 0x00A5_0F3C, "atomic_or");

        let and = run_u32(!0u32, 4, |out, grid| {
            let _ = per_operation_kernels::and_mask(out)
                .grid(grid)
                .sync()
                .expect("and");
        });
        assert_uniform(&and, 0x0F0F_F0F0, "atomic_and");

        let xor = run_u32(0x5A5A_5A5A, 8, |out, grid| {
            let _ = per_operation_kernels::xor_toggle(out)
                .grid(grid)
                .sync()
                .expect("xor");
        });
        assert_uniform(&xor, 0x5A5A_5A5A, "atomic_xor: 8 toggles must cancel");
    });
}

/// The comparisons respect the element's signedness on real hardware, with
/// seed and contribution straddling the high bit so the wrong comparison
/// gives a visibly wrong value rather than a coincidence.
#[test]
fn comparisons_respect_signedness_on_device() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let max = run_u32(0x7FFF_FFF0, 1, |out, grid| {
            let _ = per_operation_kernels::max_straddles_the_high_bit(out)
                .grid(grid)
                .sync()
                .expect("max u32");
        });
        assert_uniform(&max, 0x8000_0010u32, "atomic_max u32: unsigned comparison");

        let min = run_u32(0x8000_0010, 1, |out, grid| {
            let _ = per_operation_kernels::min_straddles_the_high_bit(out)
                .grid(grid)
                .sync()
                .expect("min u32");
        });
        assert_uniform(&min, 0x7FFF_FFF0u32, "atomic_min u32: unsigned comparison");

        let (tiles_m, tiles_n) = (2u32, 3u32);
        let rows = tiles_m as usize * BM;
        let cols = tiles_n as usize * BN;
        let mut out = api::full::<i32>(2, &[rows, cols]).sync().expect("seed");
        let _ = per_operation_kernels::max_signed_negatives(out.atomic())
            .grid((tiles_m, tiles_n, 1))
            .sync()
            .expect("max i32");
        let host: Vec<i32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_uniform(&host, 2i32, "atomic_max i32: signed comparison");
    });
}

/// Integer addition is exact, so the count is an equality at any size.
#[test]
fn integer_addition_is_exact_on_device() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let splits = 64u32;
        let host = run_u32(0, splits, |out, grid| {
            let _ = per_operation_kernels::add_u32_ones(out)
                .grid(grid)
                .sync()
                .expect("add");
        });
        assert_uniform(&host, splits, "atomic_add u32");
    });
}

#[cutile::module]
mod mixed_operation_kernels {
    use cutile::core::*;

    /// Two different aggregation operations over what the host binds as one
    /// allocation: the launcher must reject the launch, so this body is
    /// never reached on any device.
    #[cutile::entry()]
    fn add_and_max(a: &AddTensor<i32, { [-1, -1] }>, b: &MaxTensor<i32, { [-1, -1] }>) {
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let add: AddPartition<i32, { [16, 16] }> = a.partition(shape![16, 16]);
        let max: MaxPartition<i32, { [16, 16] }> = b.partition(shape![16, 16]);
        let _t: Token = add.atomic_add([0i32, 0i32], v);
        let _t: Token = max.atomic_max([0i32, 0i32], v);
    }
}

/// One allocation bound to two *different* aggregation operations is
/// rejected before any GPU work, naming both parameters and both operations.
#[test]
fn mixed_operations_on_one_allocation_are_rejected_before_any_gpu_work() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let mut out = api::zeros::<i32>(&[32, 48]).sync().expect("zeros");
        // A second owned tensor over the same storage, as in the
        // two-atomic-bindings test above.
        let Ok(second) = Arc::try_unwrap(unsafe { out.into_shared_alias() }) else {
            panic!("a freshly minted alias is the only reference to itself");
        };
        let err = mixed_operation_kernels::add_and_max(out.atomic(), Atomic::owned(second))
            .grid((2, 3, 1))
            .sync()
            .expect_err("two operations over one allocation must not launch");
        let message = err.to_string();
        assert!(
            message.contains("addition") && message.contains("maximum"),
            "the rejection must name both operations: {message}"
        );
    });
}

#[cutile::module]
mod matrix_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn matrix_add_i32(out: &AddTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_add_i64(out: &AddTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: AddPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_add_u64(out: &AddTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: AddPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_add_f64(out: &AddTensor<f64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f64, { [16, 16] }> = constant(1.0f64, shape![16, 16]);
        let acc: AddPartition<f64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_or_i32(out: &OrTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(0x0F0F_0F0Fi32, shape![16, 16]);
        let acc: OrPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_and_i32(out: &AndTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(0x0F0F_0F0Fi32, shape![16, 16]);
        let acc: AndPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_xor_i32(out: &XorTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(0x0F0F_0F0Fi32, shape![16, 16]);
        let acc: XorPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_or_i64(out: &OrTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fi64, shape![16, 16]);
        let acc: OrPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_and_i64(out: &AndTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fi64, shape![16, 16]);
        let acc: AndPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_xor_i64(out: &XorTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fi64, shape![16, 16]);
        let acc: XorPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_or_u64(out: &OrTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fu64, shape![16, 16]);
        let acc: OrPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_and_u64(out: &AndTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fu64, shape![16, 16]);
        let acc: AndPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_xor_u64(out: &XorTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(0x0F0F_0F0F_0F0F_0F0Fu64, shape![16, 16]);
        let acc: XorPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_max_i64(out: &MaxTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(-5i64, shape![16, 16]);
        let acc: MaxPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_max_u64(out: &MaxTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(0x8000_0000_0000_0010u64, shape![16, 16]);
        let acc: MaxPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_min_i32(out: &MinTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(-5i32, shape![16, 16]);
        let acc: MinPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_min_i64(out: &MinTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(-5i64, shape![16, 16]);
        let acc: MinPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }
    #[cutile::entry()]
    fn matrix_min_u64(out: &MinTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(0x7FFF_FFFF_FFFF_FFF0u64, shape![16, 16]);
        let acc: MinPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }
}

/// One device value test per remaining operation/element cell, completing
/// hardware coverage of the legal matrix (f16/bf16/f32/u32/i32 cells are
/// covered above). Exact references throughout: counting for addition,
/// idempotent masks for OR and AND, an even toggle count for XOR, and
/// seed/contribution pairs that distinguish the comparison's signedness,
/// including 64-bit values straddling the high bit.
#[test]
fn the_remaining_matrix_cells_hold_on_device() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let (tiles_m, tiles_n) = (2u32, 2u32);
        let shape = [tiles_m as usize * BM, tiles_n as usize * BN];
        let grid = (tiles_m, tiles_n, 1u32);
        let grid8 = (tiles_m, tiles_n, 8u32);

        fn check<T: cutile::prelude::DType + PartialEq + std::fmt::Debug + Copy>(
            what: &str,
            out: &mut Tensor<T>,
            expected: T,
        ) {
            let host: Vec<T> = out.dup().to_host_vec().sync().expect("to_host");
            let wrong = host.iter().filter(|&&v| v != expected).count();
            assert_eq!(wrong, 0, "{what}: {wrong} of {} wrong", host.len());
        }

        // Addition counts exactly at 8 overlapping programs per tile.
        let mut o = api::zeros::<i32>(&shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_add_i32(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("add i32", &mut o, 8i32);
        let mut o = api::zeros::<i64>(&shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_add_i64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("add i64", &mut o, 8i64);
        let mut o = api::zeros::<u64>(&shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_add_u64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("add u64", &mut o, 8u64);
        let mut o = api::zeros::<f64>(&shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_add_f64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("add f64", &mut o, 8.0f64);

        // OR and AND are idempotent; XOR's even toggle count restores the seed.
        let mut o = api::full::<i32>(0x5000_0001, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_or_i32(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("or i32", &mut o, 0x5000_0001 | 0x0F0F_0F0F);
        let mut o = api::full::<i32>(-1, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_and_i32(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("and i32", &mut o, 0x0F0F_0F0F);
        let mut o = api::full::<i32>(0x5A5A_5A5A, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_xor_i32(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("xor i32 (8 toggles cancel)", &mut o, 0x5A5A_5A5A);

        let mut o = api::full::<i64>(0x5000_0000_0000_0001, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_or_i64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check(
            "or i64",
            &mut o,
            0x5000_0000_0000_0001 | 0x0F0F_0F0F_0F0F_0F0F,
        );
        let mut o = api::full::<i64>(-1, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_and_i64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("and i64", &mut o, 0x0F0F_0F0F_0F0F_0F0F);
        let mut o = api::full::<i64>(0x5A5A_5A5A_5A5A_5A5A, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_xor_i64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("xor i64", &mut o, 0x5A5A_5A5A_5A5A_5A5A);

        let mut o = api::full::<u64>(0x9000_0000_0000_0001, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_or_u64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check(
            "or u64",
            &mut o,
            0x9000_0000_0000_0001 | 0x0F0F_0F0F_0F0F_0F0F,
        );
        let mut o = api::full::<u64>(!0u64, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_and_u64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("and u64", &mut o, 0x0F0F_0F0F_0F0F_0F0F);
        let mut o = api::full::<u64>(0xA5A5_A5A5_A5A5_A5A5, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_xor_u64(o.atomic())
            .grid(grid8)
            .sync()
            .expect("launch");
        check("xor u64", &mut o, 0xA5A5_A5A5_A5A5_A5A5);

        // Comparisons: signed keeps 2 over -5; unsigned 64-bit straddles the
        // high bit in both directions.
        let mut o = api::full::<i64>(2, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_max_i64(o.atomic())
            .grid(grid)
            .sync()
            .expect("launch");
        check("max i64 (signed)", &mut o, 2i64);
        let mut o = api::full::<u64>(0x7FFF_FFFF_FFFF_FFF0, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_max_u64(o.atomic())
            .grid(grid)
            .sync()
            .expect("launch");
        check("max u64 (unsigned)", &mut o, 0x8000_0000_0000_0010u64);
        let mut o = api::full::<i32>(2, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_min_i32(o.atomic())
            .grid(grid)
            .sync()
            .expect("launch");
        check("min i32 (signed)", &mut o, -5i32);
        let mut o = api::full::<i64>(2, &shape).sync().expect("seed");
        let _ = matrix_kernels::matrix_min_i64(o.atomic())
            .grid(grid)
            .sync()
            .expect("launch");
        check("min i64 (signed)", &mut o, -5i64);
        let mut o = api::full::<u64>(0x8000_0000_0000_0010, &shape)
            .sync()
            .expect("seed");
        let _ = matrix_kernels::matrix_min_u64(o.atomic())
            .grid(grid)
            .sync()
            .expect("launch");
        check("min u64 (unsigned)", &mut o, 0x7FFF_FFFF_FFFF_FFF0u64);
    });
}
