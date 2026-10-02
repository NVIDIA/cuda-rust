/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reuse each CUDA block (CTA) for several output tiles.
//!
//! A thread is one GPU worker. A warp is 32 threads that run together. A
//! block, also called a CTA, is a group of warps. A persistent kernel launches
//! only as many blocks as the GPU can keep active. Each block finishes one
//! output tile, then takes another.
//!
//! First, number every `(M tile, N tile, batch)` position. M changes first:
//!
//! ```text
//! for M_TILES=3, N_TILES=2:
//!
//! linear 0 -> (m=0, n=0, batch=0)
//! linear 1 -> (m=1, n=0, batch=0)
//! linear 2 -> (m=2, n=0, batch=0)
//! linear 3 -> (m=0, n=1, batch=0)
//! ```
//!
//! Then block `i` visits `i`, `i + block_count`, and so on:
//!
//! ```text
//! 3 blocks, 8 tiles
//! block 0: 0 -> 3 -> 6
//! block 1: 1 -> 4 -> 7
//! block 2: 2 -> 5
//! ```
//!
//! Every tile is visited once, without a shared counter.
//! [`StaticPersistentTileScheduler::new_1d`] uses only the X launch dimension,
//! so launch `(active_blocks, 1, 1)`.

/// Coordinates of one output tile selected by the scheduler.
///
/// This value describes a location; it does not enforce exclusive access.
/// Warps and threads inside the block must still write different elements of
/// the selected tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkTile<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize> {
    linear: usize,
    m_tile: usize,
    n_tile: usize,
    batch: usize,
}

impl<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize>
    WorkTile<M_TILES, N_TILES, BATCHES>
{
    /// Return `(linear, M tile, N tile, batch)` together.
    ///
    /// Kernels normally unpack this once at the start of a tile iteration.
    #[must_use]
    #[inline(always)]
    pub fn coordinates(self) -> (usize, usize, usize, usize) {
        __compiler::work_tile_coordinates(self)
    }

    /// Tile number when M changes first, then N, then batch.
    #[must_use]
    #[inline(always)]
    pub const fn linear(self) -> usize {
        self.linear
    }

    /// Tile position along M.
    #[must_use]
    #[inline(always)]
    pub const fn m_tile(self) -> usize {
        self.m_tile
    }

    /// Tile position along N.
    #[must_use]
    #[inline(always)]
    pub const fn n_tile(self) -> usize {
        self.n_tile
    }

    /// Batch position, also called logical L.
    #[must_use]
    #[inline(always)]
    pub const fn batch(self) -> usize {
        self.batch
    }
}

/// Assign a fixed output-tile grid across persistent CUDA blocks.
///
/// Every block starts at `blockIdx.x`. Each call to [`Self::advance`] adds
/// `gridDim.x`, so blocks visit different tile numbers without a shared
/// counter:
///
/// ```text
/// start = blockIdx.x
/// next  = current + gridDim.x
/// stop  = current >= TOTAL_TILES
/// ```
///
/// The caller owns accumulator and pipeline state. A multi-stage pipeline's
/// slot and phase bit must continue across output tiles; that bit marks each
/// new reuse of a buffer. Only the K-loop's tile counter starts over for each
/// output tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticPersistentTileScheduler<
    const M_TILES: usize,
    const N_TILES: usize,
    const BATCHES: usize,
> {
    current_linear: usize,
    stride: usize,
}

impl<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize>
    StaticPersistentTileScheduler<M_TILES, N_TILES, BATCHES>
{
    /// Total output tiles: `M_TILES * N_TILES * BATCHES`.
    ///
    /// Using this constant rejects zero sizes and multiplication overflow at
    /// compile time.
    pub const TOTAL_TILES: usize = {
        assert!(M_TILES > 0, "M_TILES must be positive");
        assert!(N_TILES > 0, "N_TILES must be positive");
        assert!(BATCHES > 0, "BATCHES must be positive");
        let mn = match M_TILES.checked_mul(N_TILES) {
            Some(value) => value,
            None => panic!("M_TILES * N_TILES overflows usize"),
        };
        match mn.checked_mul(BATCHES) {
            Some(value) => value,
            None => panic!("tile count overflows usize"),
        }
    };

    /// Create a scheduler from this block's X index and the X grid size.
    ///
    /// CUDA guarantees `blockIdx.x < gridDim.x` and `gridDim.x > 0`, so this
    /// path needs no runtime error check.
    #[must_use]
    #[inline(always)]
    pub fn new_1d() -> Self {
        __compiler::scheduler_new_1d::<M_TILES, N_TILES, BATCHES>()
    }

    /// Create one block's schedule from explicit numbers.
    ///
    /// This form is useful in CPU tests and emulators. It returns `None` when
    /// `worker_count` is zero or `worker` is outside `0..worker_count`.
    #[must_use]
    #[inline(always)]
    pub const fn for_worker(worker: usize, worker_count: usize) -> Option<Self> {
        let _ = Self::TOTAL_TILES;
        if worker_count == 0 || worker >= worker_count {
            None
        } else {
            Some(Self {
                current_linear: worker,
                stride: worker_count,
            })
        }
    }

    /// Return the current tile coordinates, or `None` when this block is done.
    #[must_use]
    #[inline(always)]
    pub fn current(&self) -> Option<WorkTile<M_TILES, N_TILES, BATCHES>> {
        let linear = self.current_linear;
        if linear >= Self::TOTAL_TILES {
            return None;
        }

        // M changes first, then N, then batch. Compute each remainder with
        // multiply/subtract so LLVM can reuse the matching division result.
        let n_batch = linear / M_TILES;
        let m_tile = linear - n_batch * M_TILES;
        let batch = n_batch / N_TILES;
        let n_tile = n_batch - batch * N_TILES;

        Some(WorkTile {
            linear,
            m_tile,
            n_tile,
            batch,
        })
    }

    /// True while this block still has an output tile to process.
    #[must_use]
    #[inline(always)]
    pub fn has_work(&self) -> bool {
        __compiler::scheduler_has_work(self)
    }

    /// Return the current tile after [`Self::has_work`] returned true.
    ///
    /// This split keeps the bounds check outside the coordinate arithmetic in
    /// the kernel loop.
    #[must_use]
    #[inline(always)]
    pub fn current_tile(&self) -> WorkTile<M_TILES, N_TILES, BATCHES> {
        __compiler::scheduler_current_tile(self)
    }

    /// Move to this block's next output tile.
    #[inline(always)]
    pub fn advance(&mut self) {
        __compiler::scheduler_advance(self);
    }

    /// Choose the number of blocks for a one-dimensional persistent launch.
    ///
    /// The result is the smaller of the output tile count and the number of
    /// blocks the GPU can keep active:
    ///
    /// ```text
    /// active capacity = SM count * active blocks per SM
    /// launch blocks   = min(total tiles, active capacity)
    /// ```
    ///
    /// An SM is one streaming multiprocessor on the GPU. Get
    /// `active_ctas_per_sm` from CUDA's occupancy query, which reports how
    /// many blocks fit on each SM for this compiled kernel. Register and
    /// dynamic shared-memory use change that number. This returns `None` for
    /// zero capacity or an X grid larger than CUDA's `u32` launch limit.
    #[must_use]
    #[inline(always)]
    pub const fn resident_grid_size(sm_count: u32, active_ctas_per_sm: u32) -> Option<u32> {
        let total = Self::TOTAL_TILES as u64;
        let resident = (sm_count as u64) * (active_ctas_per_sm as u64);
        if resident == 0 {
            return None;
        }
        let workers = if total < resident { total } else { resident };
        if workers == 0 || workers > u32::MAX as u64 {
            None
        } else {
            Some(workers as u32)
        }
    }
}

/// Stable compiler boundaries for the five scheduler operations.
///
/// The public methods above stay pleasant to use. Device compilation
/// recognizes these exact functions; pure coordinate/state operations remain
/// ordinary testable Rust.
#[doc(hidden)]
pub mod __compiler {
    use super::{StaticPersistentTileScheduler, WorkTile};

    #[inline(never)]
    pub fn scheduler_new_1d<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize>()
    -> StaticPersistentTileScheduler<M_TILES, N_TILES, BATCHES> {
        let _ = StaticPersistentTileScheduler::<M_TILES, N_TILES, BATCHES>::TOTAL_TILES;
        unreachable!("cute-rs scheduler creation executed outside recognized device compilation")
    }

    #[inline(never)]
    pub fn scheduler_has_work<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize>(
        scheduler: &StaticPersistentTileScheduler<M_TILES, N_TILES, BATCHES>,
    ) -> bool {
        scheduler.current_linear
            < StaticPersistentTileScheduler::<M_TILES, N_TILES, BATCHES>::TOTAL_TILES
    }

    #[inline(never)]
    pub fn scheduler_current_tile<
        const M_TILES: usize,
        const N_TILES: usize,
        const BATCHES: usize,
    >(
        scheduler: &StaticPersistentTileScheduler<M_TILES, N_TILES, BATCHES>,
    ) -> WorkTile<M_TILES, N_TILES, BATCHES> {
        let linear = scheduler.current_linear;
        let n_batch = linear / M_TILES;
        let m_tile = linear - n_batch * M_TILES;
        let batch = n_batch / N_TILES;
        let n_tile = n_batch - batch * N_TILES;
        WorkTile {
            linear,
            m_tile,
            n_tile,
            batch,
        }
    }

    #[inline(never)]
    pub fn work_tile_coordinates<
        const M_TILES: usize,
        const N_TILES: usize,
        const BATCHES: usize,
    >(
        tile: WorkTile<M_TILES, N_TILES, BATCHES>,
    ) -> (usize, usize, usize, usize) {
        (tile.linear, tile.m_tile, tile.n_tile, tile.batch)
    }

    #[inline(never)]
    pub fn scheduler_advance<const M_TILES: usize, const N_TILES: usize, const BATCHES: usize>(
        scheduler: &mut StaticPersistentTileScheduler<M_TILES, N_TILES, BATCHES>,
    ) {
        scheduler.current_linear = scheduler.current_linear.saturating_add(scheduler.stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn m_fastest_mapping_crosses_n_and_batch_boundaries() {
        type Scheduler = StaticPersistentTileScheduler<3, 2, 2>;
        let expected = [
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            (2, 2, 0, 0),
            (3, 0, 1, 0),
            (5, 2, 1, 0),
            (6, 0, 0, 1),
            (11, 2, 1, 1),
        ];

        for (linear, m_tile, n_tile, batch) in expected {
            let scheduler = Scheduler::for_worker(linear, Scheduler::TOTAL_TILES).unwrap();
            let tile = scheduler.current().unwrap();
            assert_eq!(
                (tile.linear(), tile.m_tile(), tile.n_tile(), tile.batch()),
                (linear, m_tile, n_tile, batch)
            );
        }
    }

    #[test]
    fn grid_stride_170_covers_the_full_16_by_16_problem_once() {
        type Scheduler = StaticPersistentTileScheduler<16, 16, 1>;
        const WORKERS: usize = 170;
        let mut seen = [false; Scheduler::TOTAL_TILES];

        for worker in 0..WORKERS {
            let mut scheduler = Scheduler::for_worker(worker, WORKERS).unwrap();
            while let Some(tile) = scheduler.current() {
                let linear = tile.linear();
                assert!(!seen[linear], "tile {linear} was scheduled twice");
                seen[linear] = true;
                assert_eq!(
                    linear,
                    tile.batch() * 16 * 16 + tile.n_tile() * 16 + tile.m_tile()
                );
                scheduler.advance();
            }
        }

        assert!(seen.into_iter().all(|visited| visited));
    }

    #[test]
    fn uneven_final_wave_has_the_expected_worker_lengths() {
        type Scheduler = StaticPersistentTileScheduler<16, 16, 1>;

        for worker in 0..170 {
            let mut scheduler = Scheduler::for_worker(worker, 170).unwrap();
            let mut count = 0;
            while scheduler.current().is_some() {
                count += 1;
                scheduler.advance();
            }
            assert_eq!(count, if worker < 86 { 2 } else { 1 });
        }
    }

    #[test]
    fn worker_constructor_rejects_invalid_physical_grids() {
        type Scheduler = StaticPersistentTileScheduler<1, 1, 1>;
        assert!(Scheduler::for_worker(0, 0).is_none());
        assert!(Scheduler::for_worker(1, 1).is_none());
        assert!(Scheduler::for_worker(2, 1).is_none());
    }

    #[test]
    fn resident_grid_is_checked_and_capped() {
        type Scheduler = StaticPersistentTileScheduler<16, 16, 1>;
        assert_eq!(Scheduler::resident_grid_size(170, 1), Some(170));
        assert_eq!(Scheduler::resident_grid_size(170, 2), Some(256));
        assert_eq!(Scheduler::resident_grid_size(512, 8), Some(256));
        assert_eq!(Scheduler::resident_grid_size(0, 1), None);
        assert_eq!(Scheduler::resident_grid_size(170, 0), None);

        type TooWide = StaticPersistentTileScheduler<{ u32::MAX as usize }, 2, 1>;
        assert_eq!(TooWide::resident_grid_size(u32::MAX, u32::MAX), None);
    }

    #[test]
    fn completed_scheduler_stays_invalid() {
        type Scheduler = StaticPersistentTileScheduler<1, 1, 1>;
        let mut scheduler = Scheduler::for_worker(0, 1).unwrap();
        assert!(scheduler.current().is_some());
        scheduler.advance();
        assert!(scheduler.current().is_none());
        scheduler.advance();
        assert!(scheduler.current().is_none());
    }

    #[test]
    fn scheduler_state_is_only_current_and_stride() {
        type Scheduler = StaticPersistentTileScheduler<16, 16, 1>;
        assert_eq!(size_of::<Scheduler>(), 2 * size_of::<usize>());
        assert_eq!(Scheduler::TOTAL_TILES, 256);
    }
}
