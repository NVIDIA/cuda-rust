/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-event kernel benchmarking, API-compatible in spirit and naming
//! with Triton's `testing.do_bench` (`warmup`/`rep` are the same
//! millisecond budgets, with the same defaults).
//!
//! [`do_bench`] times a closure with CUDA events on a stream: warmup by time
//! budget (absorbing first-launch JIT), rep count derived from a measurement
//! budget, and the L2 cache cleared between reps so every rep sees cold
//! caches instead of the previous rep's footprint. Results report medians and
//! quantiles rather than means: on machines without locked clocks, means
//! drift with sustained load while medians stay stable.
//!
//! [`do_bench_paired`] measures two closures in strict A/B/A/B alternation.
//! Use it whenever two configurations are *compared*: back-to-back sequential
//! runs let clock and thermal drift masquerade as multi-percent differences.
//!
//! The closure must enqueue its GPU work on the stream it is given — for
//! cutile ops, `op.sync_on(stream)`. Work sent to other streams is not
//! covered by the timing events.

use crate::error::Error;
use cuda_core::Stream;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Options for [`do_bench`] and [`do_bench_paired`].
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// Wall-clock budget for untimed warmup iterations (absorbs JIT and
    /// clock ramp). At least one warmup iteration always runs.
    pub warmup: Duration,
    /// Wall-clock budget the timed reps should roughly fill (Triton's `rep`);
    /// the timed-rep count is `rep / estimated_rep_time`, clamped to
    /// [`min_reps`](Self::min_reps)..=[`max_reps`](Self::max_reps).
    pub rep: Duration,
    /// Lower bound on timed reps (medians need at least a few samples).
    pub min_reps: usize,
    /// Upper bound on timed reps.
    pub max_reps: usize,
    /// Clear the device's L2 cache before every timed rep, so reps measure
    /// cold-cache behavior instead of the previous rep's residency.
    ///
    /// Allocates a device buffer of about twice the L2 size (64 MiB to
    /// 512 MiB) for the duration of the run; like all cutile allocations,
    /// running out of device memory panics.
    pub clear_l2: bool,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            warmup: Duration::from_millis(25),
            rep: Duration::from_millis(100),
            min_reps: 5,
            max_reps: 1000,
            clear_l2: true,
        }
    }
}

/// Timings from a [`do_bench`] run, in milliseconds per rep (the analogue
/// of `torch.utils.benchmark`'s `Measurement`).
#[derive(Debug, Clone)]
pub struct Measurement {
    times_ms: Vec<f32>,
}

impl Measurement {
    /// Crate-internal constructor for the tuner's unit tests.
    #[cfg(all(test, feature = "experimental-tune"))]
    pub(crate) fn from_times_ms(times_ms: Vec<f32>) -> Self {
        Self { times_ms }
    }

    /// Number of timed reps.
    pub fn reps(&self) -> usize {
        self.times_ms.len()
    }

    /// All rep times, in the order measured.
    pub fn times_ms(&self) -> &[f32] {
        &self.times_ms
    }

    /// Fastest rep.
    pub fn min_ms(&self) -> f32 {
        self.times_ms.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Arithmetic mean. Prefer [`median_ms`](Self::median_ms) for
    /// comparisons; the mean is reported for completeness.
    pub fn mean_ms(&self) -> f32 {
        self.times_ms.iter().sum::<f32>() / self.times_ms.len() as f32
    }

    /// Median rep time — the headline number.
    pub fn median_ms(&self) -> f32 {
        self.quantile_ms(0.5)
    }

    /// Linearly-interpolated quantile, `q` in `[0, 1]`.
    pub fn quantile_ms(&self, q: f32) -> f32 {
        let mut sorted = self.times_ms.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let q = q.clamp(0.0, 1.0);
        let pos = q * (sorted.len() - 1) as f32;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = pos - lo as f32;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}

/// A device buffer sized past the L2 cache, memset before each timed rep to
/// evict prior residency. Freed asynchronously on the stream when dropped.
struct L2Clear {
    dptr: cuda_core::sys::CUdeviceptr,
    num_bytes: usize,
    stream: Arc<Stream>,
}

impl L2Clear {
    /// Twice the L2 size flushes reliably; clamp keeps pathological
    /// attribute readings from allocating absurd buffers.
    fn new(stream: &Arc<Stream>) -> Result<Self, Error> {
        let l2 = stream.device().l2_cache_size_bytes().unwrap_or(0);
        let num_bytes = (l2 * 2).clamp(64 << 20, 512 << 20);
        // Safety: freed in Drop on the same stream; never exposed.
        let dptr = unsafe { cuda_core::malloc_async(num_bytes, stream) }?;
        Ok(Self {
            dptr,
            num_bytes,
            stream: stream.clone(),
        })
    }

    fn clear(&self) -> Result<(), Error> {
        // Safety: dptr covers num_bytes and outlives the enqueued memset
        // (freed on the same stream, so stream order protects it).
        unsafe { cuda_core::memset_d8_async(self.dptr, 0, self.num_bytes, &self.stream) }?;
        Ok(())
    }
}

impl Drop for L2Clear {
    fn drop(&mut self) {
        // Safety: allocated by us with malloc_async; stream order guarantees
        // the free lands after every enqueued memset. A refused free cannot be
        // reported from a destructor and must not panic here; the scratch
        // buffer then lives until the context is released.
        let _ = unsafe { cuda_core::free_async(self.dptr, &self.stream) };
    }
}

/// One event-timed execution of `f` on `stream`.
fn time_one<F>(stream: &Arc<Stream>, f: &mut F) -> Result<f32, Error>
where
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    let device = stream.device();
    let start = device.new_event()?;
    let end = device.new_event()?;
    start.record(stream)?;
    f(stream)?;
    end.record(stream)?;
    end.synchronize()?;
    Ok(start.elapsed_time(&end)?)
}

/// Times `f` on `stream` with CUDA events. See the module docs for the
/// protocol; `f` must enqueue its work on the stream it receives.
pub fn do_bench<F>(
    stream: &Arc<Stream>,
    opts: &BenchOptions,
    mut f: F,
) -> Result<Measurement, Error>
where
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    // Warmup: at least once, then until the budget is spent. Absorbs the
    // first-launch JIT so it never lands in a timed rep.
    let warmup_start = Instant::now();
    f(stream)?;
    while warmup_start.elapsed() < opts.warmup {
        f(stream)?;
    }

    // Derive the rep count from one timed estimate.
    let est_ms = time_one(stream, &mut f)?.max(1e-4);
    let target_ms = opts.rep.as_secs_f64() * 1e3;
    let reps = ((target_ms / est_ms as f64).round() as usize).clamp(opts.min_reps, opts.max_reps);

    let l2 = opts.clear_l2.then(|| L2Clear::new(stream)).transpose()?;
    let mut times_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        if let Some(l2) = &l2 {
            l2.clear()?;
        }
        times_ms.push(time_one(stream, &mut f)?);
    }
    Ok(Measurement { times_ms })
}

/// Times two closures in strict A/B/A/B alternation on `stream` — the
/// protocol for *comparing* two configurations, immune to the clock and
/// thermal drift that corrupts sequential comparisons. Both arms run the
/// same number of reps, derived from arm A's estimate.
pub fn do_bench_paired<A, B>(
    stream: &Arc<Stream>,
    opts: &BenchOptions,
    mut a: A,
    mut b: B,
) -> Result<(Measurement, Measurement), Error>
where
    A: FnMut(&Arc<Stream>) -> Result<(), Error>,
    B: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    let warmup_start = Instant::now();
    a(stream)?;
    b(stream)?;
    while warmup_start.elapsed() < opts.warmup {
        a(stream)?;
        b(stream)?;
    }

    let est_ms = time_one(stream, &mut a)?.max(1e-4);
    let target_ms = opts.rep.as_secs_f64() * 1e3;
    // Each pair runs both arms; halve the budget-derived count per arm.
    let reps =
        (((target_ms / est_ms as f64) / 2.0).round() as usize).clamp(opts.min_reps, opts.max_reps);

    let l2 = opts.clear_l2.then(|| L2Clear::new(stream)).transpose()?;
    let (mut times_a, mut times_b) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
    for _ in 0..reps {
        if let Some(l2) = &l2 {
            l2.clear()?;
        }
        times_a.push(time_one(stream, &mut a)?);
        if let Some(l2) = &l2 {
            l2.clear()?;
        }
        times_b.push(time_one(stream, &mut b)?);
    }
    Ok((
        Measurement { times_ms: times_a },
        Measurement { times_ms: times_b },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(times: &[f32]) -> Measurement {
        Measurement {
            times_ms: times.to_vec(),
        }
    }

    #[test]
    fn quantiles_interpolate_and_clamp() {
        let r = result(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(r.min_ms(), 1.0);
        assert_eq!(r.median_ms(), 2.5);
        assert_eq!(r.quantile_ms(0.0), 1.0);
        assert_eq!(r.quantile_ms(1.0), 4.0);
        assert_eq!(r.quantile_ms(-1.0), 1.0);
        assert_eq!(r.quantile_ms(2.0), 4.0);
        assert!((r.mean_ms() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn single_sample_quantiles_are_that_sample() {
        let r = result(&[7.0]);
        assert_eq!(r.median_ms(), 7.0);
        assert_eq!(r.quantile_ms(0.25), 7.0);
    }

    #[test]
    fn median_is_robust_to_one_outlier() {
        let r = result(&[1.0, 1.0, 1.0, 1.0, 100.0]);
        assert_eq!(r.median_ms(), 1.0);
        assert!(r.mean_ms() > 20.0);
    }
}
