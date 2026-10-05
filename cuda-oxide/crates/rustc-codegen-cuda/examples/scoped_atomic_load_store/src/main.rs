// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scoped atomic load/store and fences, exercised end to end.
//!
//! This is a regression example for a path that previously could not be
//! compiled at all. `DeviceAtomic*::load` and `::store` lowered to LLVM
//! `load atomic` / `store atomic`, and any `AcqRel` or `SeqCst` ordering
//! emitted an LLVM `fence`; libNVVM rejects all three:
//!
//! ```text
//! context:   %v48 = load atomic i32, ptr %v47 syncscope("device") monotonic
//!   Atomic loads/stores are not supported
//! context:   fence syncscope("block") release
//!   Illegal instruction: fence
//! ```
//!
//! So every call to these documented APIs was a build failure under
//! `--materialize-cubin`, which is to say in any configuration that produces a
//! real kernel. They now lower to the PTX instructions that have existed for
//! this purpose since sm_70 (`ld.acquire.gpu`, `st.release.gpu`, `fence.acq_rel.gpu`).
//!
//! The example deliberately uses the release/acquire *publication* idiom rather
//! than a synthetic load and store, because that is what the feature is for:
//! one thread writes a payload and publishes a flag with a release store;
//! another observes the flag with an acquire load and is then guaranteed to see
//! the payload.
//!
//! It uses a barrier rather than a spin loop to pair writer and reader. A spin
//! would be the more literal test, but a hung example on a scheduling quirk is
//! a much worse failure mode than a slightly weaker one, and the instruction
//! coverage is identical.
//!
//! Build and run with:
//!   cargo oxide run scoped_atomic_load_store --materialize-cubin

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;
    use core::sync::atomic::{Ordering, compiler_fence, fence};
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};

    /// Each thread publishes a payload with a release store, then reads its
    /// neighbour's with an acquire load.
    ///
    /// Exercises, in one kernel: `st.release.gpu.b32`, `ld.acquire.gpu.b32`,
    /// `st.relaxed.gpu.b64`, `ld.relaxed.gpu.b64`, source-level
    /// `fence.acq_rel.sys` / `llvm.nvvm.membar.sys`, and the fused-fence
    /// SeqCst forms `fence.sc.gpu; st.release.gpu.b32` and `fence.sc.gpu;
    /// ld.acquire.gpu.b32`.
    #[kernel]
    pub fn publish_and_observe(
        flags: &[u32],
        wide: &[u64],
        mut observed: DisjointSlice<u32>,
        mut observed_wide: DisjointSlice<u64>,
        mut fence_ok: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if tid >= N {
            return;
        }
        let next = (tid + 1) % N;

        // SAFETY: `tid` and `next` are both < N, the buffers have N elements,
        // and every access to them in this kernel is atomic.
        let mine = unsafe { DeviceAtomicU32::from_ptr((flags.as_ptr() as *mut u32).add(tid)) };
        let mine_wide = unsafe { DeviceAtomicU64::from_ptr((wide.as_ptr() as *mut u64).add(tid)) };
        let neighbour =
            unsafe { DeviceAtomicU32::from_ptr((flags.as_ptr() as *mut u32).add(next)) };
        let neighbour_wide =
            unsafe { DeviceAtomicU64::from_ptr((wide.as_ptr() as *mut u64).add(next)) };

        // Publish. The release store is what an acquire reader synchronises with.
        mine.store(tid as u32 + 1, AtomicOrdering::Release);
        mine_wide.store((tid as u64 + 1) << 32, AtomicOrdering::Relaxed);

        // Exercise the source-level core fence path from issue #723. Core
        // atomics use system scope, so Release lowers to fence.acq_rel.sys.
        fence(Ordering::Release);

        // `compiler_fence` is the weaker sibling (issue #781): it constrains
        // only the optimizer and must emit no hardware barrier at all, so it
        // lowers to an empty side-effecting asm carrying a `~{memory}` clobber.
        // Placed between two stores, which is the ordering it exists to keep.
        compiler_fence(Ordering::Release);

        // Pair writers with readers without spinning: a hung example is a far
        // worse failure mode than a slightly weaker test, and the instruction
        // coverage is identical either way.
        thread::sync_threads();

        let seen = neighbour.load(AtomicOrdering::Acquire);
        let seen_wide = neighbour_wide.load(AtomicOrdering::Relaxed);

        if let Some(out) = observed.get_mut(thread::index_1d()) {
            *out = seen;
        }
        if let Some(out) = observed_wide.get_mut(thread::index_1d()) {
            *out = seen_wide;
        }

        // Exercise the other source-level lowering route. SeqCst core fences
        // use the typed llvm.nvvm.membar.sys intrinsic rather than LLVM fence.
        fence(Ordering::SeqCst);

        // SeqCst store and load: PTX has no sequentially consistent access,
        // so these lower to the libcu++ mapping with the fence fused into
        // one asm template: `fence.sc.gpu; st.release.gpu.b32` and
        // `fence.sc.gpu; ld.acquire.gpu.b32`.
        mine.store(tid as u32 + 1, AtomicOrdering::SeqCst);
        let reread = mine.load(AtomicOrdering::SeqCst);
        if let Some(out) = fence_ok.get_mut(thread::index_1d()) {
            *out = u32::from(reread == tid as u32 + 1);
        }
    }
}

fn main() {
    println!("=== Scoped atomic load/store and fences ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let flags = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let wide = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut observed = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut observed_wide = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut fence_ok = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block of N threads, and every buffer has N elements.
    unsafe {
        module.publish_and_observe(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (N as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &flags,
            &wide,
            &mut observed,
            &mut observed_wide,
            &mut fence_ok,
        )
    }
    .expect("Kernel launch failed");

    let observed = observed.to_host_vec(&stream).unwrap();
    let observed_wide = observed_wide.to_host_vec(&stream).unwrap();
    let fence_ok = fence_ok.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for i in 0..N {
        let next = (i + 1) % N;
        let want = next as u32 + 1;
        let want_wide = (next as u64 + 1) << 32;
        if observed[i] != want {
            if errors < 5 {
                eprintln!(
                    "  acquire load: thread {i} saw {}, expected {want}",
                    observed[i]
                );
            }
            errors += 1;
        }
        if observed_wide[i] != want_wide {
            if errors < 5 {
                eprintln!(
                    "  relaxed 64-bit load: thread {i} saw {:#x}, expected {want_wide:#x}",
                    observed_wide[i]
                );
            }
            errors += 1;
        }
        if fence_ok[i] != 1 {
            if errors < 5 {
                eprintln!("  SeqCst store/load round trip failed on thread {i}");
            }
            errors += 1;
        }
    }

    println!("  release store / acquire load   {} threads", N);
    println!("  relaxed 64-bit store / load    {} threads", N);
    println!("  source-level core fences        Release + SeqCst");
    println!("  SeqCst store / load (fence.sc)  {} threads", N);

    if errors == 0 {
        println!("\n=== SUCCESS: scoped atomic load/store and fences all correct ===");
    } else {
        eprintln!("\n=== FAILURE: {errors} mismatches ===");
        std::process::exit(1);
    }
}
