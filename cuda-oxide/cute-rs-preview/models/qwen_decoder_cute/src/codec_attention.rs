/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Full-sequence D64 causal sliding attention for the waveform decoder.
use crate::{model::Batch, operator::Result, rotary::HostTables};
use ctm_rs::{cuda_module, kernel, launch_contract, memory::DisjointSlice, thread};
use cuda_core::{CudaStream, LaunchConfig};
use cute_rs::{BFloat16, attention::AttentionShape};
use std::sync::Arc;

/// Return the first included key for a causal window ending at `query`.
///
/// The positive window counts the current token; subtraction saturates at zero.
/// Returns an error for a zero window.
fn window_start(query: usize, window: usize) -> Result<usize> {
    if window == 0 {
        return Err("sliding attention requires a positive window".into());
    }
    Ok(query.saturating_sub(window - 1))
}

/// Device implementation of full-sequence causal attention over 64-value heads.
#[cuda_module]
pub mod kernels {
    use super::*;
    use cute_rs::RegisterTile;
    use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
    /// Attend to up to `window` keys ending at each query in `[rows, heads, 64]`.
    ///
    /// The host must supply equal dense Q/K/V/output extents, a positive window,
    /// and a separate output allocation. Each complete warp owns one query/head;
    /// all 32 lanes must execute each dot-product reduction together. Scores use
    /// 1/sqrt(64) scaling and an FP32 online softmax; only the final output is BF16.
    #[launch_contract(domain=1, block=(128,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn window_attention_bf16(
        q: &[BFloat16],
        k: &[BFloat16],
        v: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        rows: usize,
        heads: usize,
        window: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let work = thread::blockIdx_x() as usize * 4 + tid / 32;
        let row = work / heads;
        let head = work % heads;
        // Whole warps share this predicate; no collective sees partial lanes.
        if row >= rows {
            return;
        }
        let base = (row * heads + head) * 64;
        // SAFETY: the checked dense head row contains 64 values; the full
        // warp partitions it into two ordered coordinates per lane.
        let query = unsafe { StridedTensor::from_slice(q).subview_unchecked(base, 64) }
            .local_partition::<32>(lane)
            .load::<2>()
            .map(BFloat16::to_f32);
        let start = (row + 1).saturating_sub(window);
        let mut key = start;
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator = 0.0f32;
        let mut sums = RegisterTile::from_array([0.0_f32; 2]);
        while key <= row {
            let at = (key * heads + head) * 64;
            // SAFETY: key is inside the checked sequence and head is valid.
            let keys = unsafe { StridedTensor::from_slice(k).subview_unchecked(at, 64) }
                .local_partition::<32>(lane)
                .load::<2>()
                .map(BFloat16::to_f32);
            let q = query.as_array();
            let k = keys.as_array();
            // Keep the original two-product expression: introducing a zero-
            // seeded fold or materializing products would change contraction.
            let score = q[0] * k[0] + q[1] * k[1];
            // SAFETY: all 32 lanes execute this fixed XOR sum convergently.
            let mut score = unsafe { cute_rs::arch::warp_reduction_sum_f32(score) };
            score *= 0.125 * core::f32::consts::LOG2_E;
            let next = if maximum > score { maximum } else { score };
            let old = cute_rs::arch::exp2_f32(maximum - next);
            let weight = cute_rs::arch::exp2_f32(score - next);
            // SAFETY: V has the same checked geometry as K.
            let values = unsafe { StridedTensor::from_slice(v).subview_unchecked(at, 64) }
                .local_partition::<32>(lane)
                .load::<2>()
                .map(BFloat16::to_f32);
            sums = sums.zip_map(values, |sum, value| sum * old + weight * value);
            denominator = denominator * old + weight;
            maximum = next;
            key += 1;
        }
        let reciprocal = cute_rs::arch::reciprocal_ftz_f32(denominator);
        // SAFETY: one warp per query/head and one lane per pair of coordinates;
        // output is a separate dense allocation with exactly rows * heads * 64 slots.
        unsafe {
            let mut output = StridedTensorMut::from_disjoint_slice(&mut output)
                .subview_unchecked(base, 64)
                .local_partition::<32>(lane);
            output.store(sums.map(|sum| BFloat16::from_f32(sum * reciprocal)));
        }
    }
}

/// Stream-bound rotary and sliding-attention executor for the waveform decoder.
pub struct Runner {
    /// Stream used for uploads, rotations, attention and completion.
    stream: Arc<CudaStream>,
    /// Embedded D64 sliding-attention entry point.
    module: kernels::LoadedModule,
    /// Embedded rotary entry point shared with the language decoder.
    rotary: crate::kernel::kernels::LoadedModule,
}
impl Runner {
    /// Load both embedded modules into the stream context, propagating CUDA errors.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            // SAFETY: this module is the embedded artifact generated from the
            // kernel source in this example; prepared launches verify its contract.
            module: unsafe { kernels::load(stream.context())? },
            rotary: crate::kernel::kernels::load(stream.context())?,
            stream,
        })
    }
    /// Rotate Q/K and compute causal sliding attention with `W = H * 64`.
    ///
    /// Rows are sequence positions and each row contains H consecutive heads.
    /// Returns an error for unequal row counts, incompatible rotary tables, a zero
    /// window, mixed CUDA contexts, grid overflow, or CUDA failures. Temporary
    /// buffers remain live through stream completion, including launch errors.
    pub fn run<const H: usize, const W: usize>(
        &self,
        q: &Batch<W>,
        k: &Batch<W>,
        v: &Batch<W>,
        tables: &HostTables<64>,
        window: usize,
    ) -> Result<Batch<W>> {
        let shape = AttentionShape::<H, H, 64>::new()?;
        if shape.query_len(1)? != W
            || q.rows() != k.rows()
            || q.rows() != v.rows()
            || tables.tokens() != q.rows()
        {
            return Err("codec attention tensor geometry mismatch".into());
        }
        window_start(0, window)?;
        let work = q.rows().checked_mul(H).ok_or("attention grid overflow")?;
        let blocks = work.div_ceil(4);
        if blocks > i32::MAX as usize {
            return Err("attention grid exceeds CUDA range".into());
        }
        for b in [q.buffer(), k.buffer(), v.buffer()] {
            if !Arc::ptr_eq(b.context(), self.stream.context()) {
                return Err("codec attention context mismatch".into());
            }
        }
        self.stream.context().synchronize()?;
        let mut out = Batch::<W>::allocate(&self.stream, q.rows())?;
        if work == 0 {
            return Ok(out);
        }
        let mut rq = Batch::<W>::allocate(&self.stream, q.rows())?;
        let mut rk = Batch::<W>::allocate(&self.stream, q.rows())?;
        let cos = cuda_core::DeviceBuffer::from_host(&self.stream, tables.cos())?;
        let sin = cuda_core::DeviceBuffer::from_host(&self.stream, tables.sin())?;
        let pairs = q.buffer().len() / 2;
        let rotate_blocks = pairs.div_ceil(128);
        if rotate_blocks > i32::MAX as usize {
            return Err("rotary grid exceeds CUDA range".into());
        }
        let rotate = LaunchConfig {
            grid_dim: (rotate_blocks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let attention = self
            .module
            .prepare_window_attention_bf16(cuda_core::LaunchConfig1D::new(blocks as u32, 128, 0))?;
        // SAFETY: checked dense geometry and contexts; complete warps own one
        // query/head row and each lane owns two output coordinates. Rotation
        // buffers are distinct and kept alive through unconditional completion.
        let launched = (|| -> Result<()> {
            unsafe {
                self.rotary.rotate_bf16(
                    &self.stream,
                    rotate,
                    q.buffer(),
                    &cos,
                    &sin,
                    rq.buffer_mut(),
                    q.rows(),
                    H,
                    64,
                )?;
                self.rotary.rotate_bf16(
                    &self.stream,
                    rotate,
                    k.buffer(),
                    &cos,
                    &sin,
                    rk.buffer_mut(),
                    q.rows(),
                    H,
                    64,
                )?;
                self.module.window_attention_bf16(
                    &self.stream,
                    &attention,
                    rq.buffer(),
                    rk.buffer(),
                    v.buffer(),
                    out.buffer_mut(),
                    q.rows(),
                    H,
                    window,
                )?;
            }
            Ok(())
        })();
        let completed = self.stream.synchronize();
        launched?;
        completed?;
        Ok(out)
    }
}

/// Numerical window-boundary and head isolation checks against the FP64 oracle.
pub fn validate(stream: Arc<CudaStream>) -> Result<()> {
    /// Sequence length crossing the 72-token sliding-window boundary.
    const ROWS: usize = 75;
    /// Two independent D64 heads per synthetic token.
    const WIDTH: usize = 128;
    let runner = Runner::new(stream.clone())?;
    let positions: Vec<_> = (0..ROWS).map(|i| [i as u32; 3]).collect();
    let tables = crate::rotary::RotarySpec::<64>::standard(10_000.0, 1000)?.tables(&positions)?;
    let mut checked = 0;
    for uniform in [true, false] {
        let values = |salt: usize| -> Vec<BFloat16> {
            (0..ROWS * WIDTH)
                .map(|i| crate::reference::round(((i * 19 + salt * 31) % 127) as f64 / 64.0 - 1.0))
                .collect()
        };
        let q = if uniform {
            vec![BFloat16::from_bits(0); ROWS * WIDTH]
        } else {
            values(1)
        };
        let k = if uniform { q.clone() } else { values(3) };
        let v = if uniform {
            (0..ROWS * WIDTH)
                .map(|i| crate::reference::round((i / WIDTH + i % WIDTH / 64 * 80) as f64))
                .collect()
        } else {
            values(5)
        };
        let dq = Batch::<WIDTH>::from_host(&stream, ROWS, &q)?;
        let dk = Batch::<WIDTH>::from_host(&stream, ROWS, &k)?;
        let dv = Batch::<WIDTH>::from_host(&stream, ROWS, &v)?;
        let rq = crate::rotary_reference::rotate(&q, tables.cos(), tables.sin(), 2, 64);
        let rk = crate::rotary_reference::rotate(&k, tables.cos(), tables.sin(), 2, 64);
        for window in [1, 72, 100] {
            let actual = runner
                .run::<2, WIDTH>(&dq, &dk, &dv, &tables, window)?
                .read(&stream)?;
            for row in 0..ROWS {
                let start = window_start(row, window)?;
                let expected = crate::attention_reference::attention(
                    &rq[row * WIDTH..(row + 1) * WIDTH],
                    &rk[start * WIDTH..(row + 1) * WIDTH],
                    &v[start * WIDTH..(row + 1) * WIDTH],
                    2,
                    2,
                    64,
                    1,
                    row + 1 - start,
                    None,
                );
                for (col, expected) in expected.iter().enumerate() {
                    let a = crate::reference::widen(actual[row * WIDTH + col]);
                    let e = crate::reference::widen(*expected);
                    if !a.is_finite() || (a - e).abs() > 0.001 + 0.008 * e.abs() {
                        return Err(format!("codec attention mismatch uniform={uniform}, window={window}, row={row}, col={col}: {a} != {e}").into());
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("codec D64 sliding attention: {checked} outputs passed");
    Ok(())
}

/// Host-only checks for causal window bounds, including integer limits.
#[cfg(test)]
mod tests {
    use super::*;
    /// Verify the current token is included and stale keys are excluded without overflow.
    #[test]
    fn sliding_window_includes_current_token_and_excludes_old_keys() {
        assert_eq!(window_start(0, 72).unwrap(), 0);
        assert_eq!(window_start(71, 72).unwrap(), 0);
        assert_eq!(window_start(72, 72).unwrap(), 1);
        assert_eq!(window_start(99, 1).unwrap(), 99);
        assert!(window_start(0, 0).is_err());
        assert_eq!(window_start(usize::MAX, 72).unwrap(), usize::MAX - 71);
    }
}
