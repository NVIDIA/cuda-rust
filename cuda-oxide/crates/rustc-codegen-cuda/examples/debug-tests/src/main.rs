/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug-Info Assessment Example
//!
//! Exercises DWARF / line-table features that cuda-gdb consumes in full debug
//! mode. The kernels group compatible debugger features around stable source
//! stops; together they let the assessment script probe:
//!
//! * Scalar locals (i32, u32, f32, bool, u64)
//! * Nested struct types (Point, Rect, NavState)
//! * C-style enum types (#[repr(u32)] Direction)
//! * Struct-typed args / locals (ThreadIndex, DisjointSlice)
//! * Fixed-size array locals ([i32; 4])
//! * Inlined-function frames (kernel calls a helper)
//! * Loop-variable liveness (accumulator inside a while-loop)
//! * Raw-pointer locals (*const i32, *const f32)
//! * Tuple locals ((usize, f32))
//! * Shared-memory static array (SharedArray<i32, 32>)
//! * Device-global static variable (static mut GLOBAL_COUNTER)
//! * Constant memory (ConstantMemory<f32> + #[constant])
//! * Exception signals: gpu_assert! failure, debug::trap()
//! * Null pointer dereference (MMU fault / illegal address)
//!
//! Each run launches exactly one kernel, selected by an optional CLI argument
//! (the safe `values` scenario is the default):
//!   cargo oxide run debug-tests
//!   cargo oxide run debug-tests -- <kernel>
//!   CUDA_OXIDE_DEBUG=full cargo oxide run debug-tests -- <kernel>
//!   ./debug-tests <kernel>
//!
//! kernels: values, aggregates, loop, memory_spaces, deep_stack, assert_fail,
//!          trap_fail, breakpoint, oob_index, null_deref
//!
//! The last five are "crash modes" (for cuda-gdb exception / coredump tests):
//!   ./debug-tests assert_fail      triggers gpu_assert! failure
//!   ./debug-tests trap_fail        triggers debug::trap()
//!   ./debug-tests breakpoint       triggers debug::breakpoint() (use under cuda-gdb)
//!   ./debug-tests oob_index        triggers a Rust bounds-check trap
//!   ./debug-tests null_deref       triggers a null-pointer dereference

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{
    ConstantMemory, DisjointSlice, SharedArray, constant, debug, gpu_assert, gpu_printf, kernel,
    thread,
};
use cuda_host::cuda_module;

// =============================================================================
// Shared types (visible to both host and device)
// =============================================================================

/// Simple 2-D point using Rust's native aggregate layout.
#[derive(Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// Axis-aligned bounding box; tests nested struct inspection.
#[derive(Clone, Copy)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

/// Cardinal direction; #[repr(u32)] makes the discriminant a plain integer so
/// cuda-gdb can display it as an enum value.
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

/// Compound struct embedding an enum — tests mixed struct+enum DWARF.
#[derive(Clone, Copy)]
pub struct NavState {
    pub pos: Point,
    pub heading: Direction,
    pub speed: f32,
}

// =============================================================================
// Device-side global memory variable
// =============================================================================

/// Device-side global counter — accessible by name from cuda-gdb when stopped
/// inside a kernel that uses it.
static mut GLOBAL_COUNTER: u64 = 0;

// =============================================================================
// Device-side helpers
// =============================================================================

/// Scale-and-bias helper used alongside the other primitive-value checks.
fn compute_scaled(v: i32, scale: i32) -> i32 {
    v.wrapping_mul(scale).wrapping_add(1)
}

// ---------------------------------------------------------------------------
// Mixed physical/inlined call-stack helpers (corner case 13)
//
// These are defined OUTSIDE `#[cuda_module]` so they represent the
// "function passed in to the kernel" scenario — external callees whose
// DWARF shows a separate compilation scope from the kernel itself.
//
// The leaf and outer helpers remain physical calls, while the middle helper is
// forced inline. cuda-gdb should reconstruct the inline frame between the two
// physical frames.
// ---------------------------------------------------------------------------

/// Leaf function: the deepest frame in the callstack.
///
/// In cuda-gdb, break via `break debug_tests__deep_leaf` (the PTX symbol name —
/// cuda-oxide prefixes extern helpers with `debug_tests__`).
///
/// The `fn` signature is the only source line with a direct `.loc` entry in the
/// PTX; the body maps through inlined `wrapping_mul` / `wrapping_add` entries.
// <<<GDB BREAK DEEP>>> use `break debug_tests__deep_leaf`; inspect v and bias
#[inline(never)]
fn deep_leaf(v: i32, bias: i32) -> i32 {
    let doubled: i32 = v.wrapping_mul(2);
    let result: i32 = doubled.wrapping_add(bias);
    result
}

/// Inline middle link — its logical frame and `scaled` local must remain visible.
#[inline(always)]
fn deep_middle(v: i32, scale: i32, bias: i32) -> i32 {
    let scaled: i32 = v.wrapping_mul(scale);
    deep_leaf(scaled, bias)
}

/// Outer link — calls `deep_middle`, one step below the kernel.
/// In cuda-gdb: frame 2. `offset` must be visible with `info locals`.
#[inline(never)]
fn deep_outer(v: i32, scale: i32) -> i32 {
    let offset: i32 = scale.wrapping_add(1);
    deep_middle(v, scale, offset)
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    // -------------------------------------------------------------------------
    // Constant memory declaration used by debuginfo_memory_spaces.
    // -------------------------------------------------------------------------
    #[constant]
    static COEFF: ConstantMemory<f32> = ConstantMemory::UNINIT;

    // -------------------------------------------------------------------------
    // Primitive values: scalars, fixed array, raw pointers, inline helper, and
    // successful gpu_printf!/gpu_assert! expansion in one lexical scope.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_values(input: &[i32], fdata: &[f32], scale: i32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let tid: u32 = idx.get() as u32;
        if let Some(slot) = out.get_mut(idx) {
            let x: i32 = input[tid as usize];
            let y: f32 = x as f32 * 1.5_f32;
            let flag: bool = x > 0;
            let big: u64 = tid as u64 * 1000_u64;
            let base: i32 = tid as i32;
            let buf: [i32; 4] = [base, base + 10, base + 20, base + 30];
            let ptr: *const i32 = &input[tid as usize] as *const i32;
            let fptr: *const f32 = &fdata[tid as usize] as *const f32;
            let scaled: i32 = compute_scaled(x, scale);
            if tid == 0 {
                gpu_printf!("thread 0: v = {}\n", x);
            }
            gpu_assert!(x >= 0, "debuginfo_values: negative value");
            // <<<GDB BREAK VALUES>>> all primitive locals are live here
            *slot = if flag {
                x.wrapping_add(y as i32)
                    .wrapping_add(big as i32)
                    .wrapping_add(buf[0])
                    .wrapping_add(buf[1])
                    .wrapping_add(buf[2])
                    .wrapping_add(buf[3])
                    .wrapping_add(scaled)
                    .wrapping_add(unsafe { *ptr })
                    .wrapping_add(unsafe { *fptr } as i32)
            } else {
                0
            };
        }
    }

    // -------------------------------------------------------------------------
    // Aggregate values: kernel/runtime structs, tuple, user structs, and enum.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_aggregates(data: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();
        if tid >= data.len() {
            return;
        }
        let p: Point = Point {
            x: data[tid],
            y: data[tid] * 2.0_f32,
        };
        let r: Rect = Rect {
            min: p,
            max: Point {
                x: p.x + 10.0_f32,
                y: p.y + 10.0_f32,
            },
        };
        let state: NavState = NavState {
            pos: p,
            heading: Direction::East,
            speed: 42.0_f32,
        };
        let dir_a: Direction = Direction::East;
        let dir_b: Direction = Direction::West;
        let slot_idx = thread::index_1d();
        if let Some(slot) = out.get_mut(slot_idx) {
            let pair: (usize, f32) = (tid, data[tid]);
            // <<<GDB BREAK AGGREGATES>>> all aggregate locals are live here
            *slot = pair.1 * 2.0_f32
                + p.x
                + r.max.x
                + state.speed
                + dir_a as u32 as f32
                + dir_b as u32 as f32
                + idx.get() as f32;
        }
    }

    // -------------------------------------------------------------------------
    // Loop variable liveness and source stepping.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_loop(n: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let mut acc: u64 = 0_u64;
            let mut i: u32 = 0_u32;
            while i < n {
                // <<<GDB BREAK (first iteration)>>> inspect i and acc
                acc = acc.wrapping_add(i as u64);
                i = i.wrapping_add(1);
            }
            *slot = acc;
        }
    }

    // -------------------------------------------------------------------------
    // CUDA address spaces: shared, global, and constant memory at one stop.
    // Must be launched with all threads in one block for the neighbor read.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_memory_spaces(data: &[i32], mut out: DisjointSlice<f32>) {
        static mut TILE: SharedArray<i32, 32> = SharedArray::UNINIT;

        let tid: usize = thread::threadIdx_x() as usize;
        let idx = thread::index_1d();
        let i: usize = idx.get();
        // This fixture launches one block of eight threads. Each thread owns
        // four disjoint cells; initialize all 32 values the debugger prints.
        // The raw receiver avoids overlapping mutable references to TILE.
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        unsafe {
            tile.add(tid).write(data[i]);
            tile.add(tid + 8).write(0);
            tile.add(tid + 16).write(0);
            tile.add(tid + 24).write(0);
        }
        thread::sync_threads();
        let coeff_val: f32 = COEFF.get();
        if tid == 0 {
            unsafe {
                GLOBAL_COUNTER = 7;
            }
        }
        thread::sync_threads();
        let global_val: u64 = unsafe { GLOBAL_COUNTER };
        // <<<GDB BREAK MEMORY SPACES>>> all address spaces are inspectable here
        if let Some(slot) = out.get_mut(idx) {
            let neighbor: i32 = unsafe { tile.add((tid + 1) % 8).read() };
            *slot = neighbor as f32 + coeff_val + global_val as f32 + i as f32;
        }
    }

    // -------------------------------------------------------------------------
    // Corner case 13: mixed physical/inlined callstack (4 logical frames)
    //
    // Call chain (outermost → innermost):
    //   debuginfo_deep_stack  (frame 3 / kernel)
    //     → deep_outer        (frame 2 / extern fn, "passed in to kernel")
    //       → deep_middle     (frame 1 / forced inline)
    //         → deep_leaf     (frame 0 / extern fn, deepest; has GDB break)
    //
    // All three callees are defined OUTSIDE `#[cuda_module]`. deep_leaf and
    // deep_outer are physical calls; deep_middle must appear from inline DWARF.
    //
    // GDB tests:
    //   backtrace → 4 frames
    //   frame 0 info locals → doubled, result
    //   frame 1 info locals → scaled (inline frame)
    //   frame 2 info locals → offset
    //   frame 3 info locals → tid, idx (kernel frame)
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_deep_stack(input: &[i32], scale: i32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();
        if tid >= input.len() {
            return;
        }
        // kernel (frame 3) → deep_outer (frame 2) → deep_middle (inline frame 1)
        //   → deep_leaf (physical frame 0, has <<<GDB BREAK DEEP>>>)
        if let Some(slot) = out.get_mut(idx) {
            *slot = deep_outer(input[tid], scale);
        }
    }

    // -------------------------------------------------------------------------
    // Crash kernel A: gpu_assert! with message (--assert-fail mode)
    //
    // Generates CUDA_ERROR_ASSERT; cuda-gdb sees $GDB_SIGNAL_CUDA_WARP_ASSERT.
    // The assertion message and call-site metadata are printed to stderr.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_assert_fail(input: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let v: i32 = input[tid];
            // This assertion is intentionally false for all threads.
            gpu_assert!(v < 0, "debuginfo: intentional assertion failure");
            *slot = v;
        }
    }

    // -------------------------------------------------------------------------
    // Crash kernel B: debug::trap() (--trap-fail mode)
    //
    // Generates CUDA_ERROR_ILLEGAL_INSTRUCTION;
    // cuda-gdb sees $GDB_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION.
    // -------------------------------------------------------------------------
    // The post-trap assignment intentionally keeps the debugger fixture's
    // lexical scope intact even though `trap` never returns.
    #[allow(unreachable_code, unused_variables)]
    #[kernel]
    pub fn debuginfo_trap_fail(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            // <<<GDB BREAK TRAP>>> cuda-gdb catches WARP_ILLEGAL_INSTRUCTION here
            debug::trap();
            *slot = 0;
        }
    }

    // -------------------------------------------------------------------------
    // Crash kernel B2: debug::breakpoint() (--breakpoint mode)
    //
    // Inserts a software breakpoint (PTX `brkpt`); cuda-gdb catches it as
    // SIGTRAP and stops at the breakpoint site with a correct source location.
    // Unlike debug::trap(), this is designed for interactive debugging rather
    // than fatal termination.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_breakpoint(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            // <<<GDB BREAK BRKPT>>> cuda-gdb catches software breakpoint here
            debug::breakpoint();
            *slot = 0;
        }
    }

    // -------------------------------------------------------------------------
    // Crash kernel C2: out-of-bounds slice index (--oob-index mode)
    //
    // `input` is intentionally shorter than `out`, so threads 4-7 execute
    // `input[tid]` with tid >= input.len().  The Rust bounds check fires and
    // calls debug::trap(), producing CUDA_ERROR_ILLEGAL_INSTRUCTION.
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_oob_index(input: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // This line triggers the bounds-check trap when tid >= input.len().
            let v: i32 = input[tid]; // ← OOB for tid >= input.len()
            *slot = v.wrapping_add(1);
        }
    }

    // -------------------------------------------------------------------------
    // Crash kernel C: null pointer dereference (--null-deref mode)
    //
    // Dereferences a null *const i32 → MMU fault or illegal address.
    // cuda-gdb sees $GDB_SIGNAL_CUDA_WARP_MMU_FAULT (or
    // $GDB_SIGNAL_CUDA_WARP_ILLEGAL_ADDRESS on older drivers).
    // -------------------------------------------------------------------------
    #[kernel]
    pub fn debuginfo_null_deref(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let null_ptr: *const i32 = core::ptr::null();
            // <<<GDB BREAK NULL DEREF>>> cuda-gdb catches MMU fault here
            // SAFETY: intentionally UB — this is a crash test.
            *slot = unsafe { *null_ptr };
        }
    }
}

// =============================================================================
// Host code
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args.get(1).map_or("values", String::as_str);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = 8;

    match kernel {
        "values" => {
            println!("--- debuginfo_values ---");
            let input: Vec<i32> = (1..=N as i32).collect();
            let fdata: Vec<f32> = (0..N).map(|i| i as f32).collect();
            let input_dev = DeviceBuffer::from_host(&stream, &input)?;
            let fdata_dev = DeviceBuffer::from_host(&stream, &fdata)?;
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            unsafe {
                module.debuginfo_values(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &input_dev,
                    &fdata_dev,
                    3,
                    &mut out_dev,
                )
            }?;
            stream.synchronize()?;
            // Thread 0: x=1, y=1.5, big=0, buf sum=60, scaled=4,
            // integer/floating pointer values=1/0 → 1+1+0+60+4+1+0=67.
            assert_eq!(out_dev.to_host_vec(&stream)?[0], 67);
            println!("  PASS");
            Ok(())
        }

        "aggregates" => {
            println!("--- debuginfo_aggregates ---");
            let data: Vec<f32> = (1..=N as u32).map(|i| i as f32).collect();
            let data_dev = DeviceBuffer::from_host(&stream, &data)?;
            let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            unsafe {
                module.debuginfo_aggregates(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &data_dev,
                    &mut out_dev,
                )
            }?;
            stream.synchronize()?;
            // Thread 0: pair=1, p.x=1, r.max.x=11, speed=42,
            // directions East/West=1/3 → 2+1+11+42+1+3=60.
            let result = out_dev.to_host_vec(&stream)?;
            assert!((result[0] - 60.0_f32).abs() < 1e-4, "got {}", result[0]);
            println!("  PASS");
            Ok(())
        }

        "loop" => {
            println!("--- debuginfo_loop ---");
            let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
            unsafe {
                module.debuginfo_loop(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    5u32,
                    &mut out_dev,
                )
            }?;
            stream.synchronize()?;
            assert_eq!(out_dev.to_host_vec(&stream)?[0], 10); // 0+1+2+3+4
            println!("  PASS");
            Ok(())
        }

        "memory_spaces" => {
            println!("--- debuginfo_memory_spaces ---");
            let data: Vec<i32> = (0..N as i32).collect();
            let data_dev = DeviceBuffer::from_host(&stream, &data)?;
            module.set_coeff(&stream, &2.5_f32)?;
            let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (N as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { module.debuginfo_memory_spaces(&stream, cfg, &data_dev, &mut out_dev) }?;
            stream.synchronize()?;
            let result = out_dev.to_host_vec(&stream)?;
            for (i, actual) in result.into_iter().enumerate() {
                let expected = ((i + 1) % N) as f32 + 2.5 + 7.0 + i as f32;
                assert_eq!(actual, expected, "thread {i}");
            }
            println!("  PASS");
            Ok(())
        }

        "deep_stack" => {
            println!("--- debuginfo_deep_stack ---");
            let input: Vec<i32> = (1..=N as i32).collect();
            let scale: i32 = 3;
            let input_dev = DeviceBuffer::from_host(&stream, &input)?;
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            unsafe {
                module.debuginfo_deep_stack(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &input_dev,
                    scale,
                    &mut out_dev,
                )
            }?;
            stream.synchronize()?;
            // Thread 0: v=1, scale=3
            //   deep_outer: offset = 3+1 = 4
            //   deep_middle: scaled = 1*3 = 3
            //   deep_leaf: doubled = 3*2 = 6, result = 6+4 = 10
            let result = out_dev.to_host_vec(&stream)?;
            assert_eq!(result[0], 10, "got {}", result[0]);
            println!("  PASS");
            Ok(())
        }

        "assert_fail" => {
            println!("--- debuginfo_assert_fail (intentional crash) ---");
            // All positive input → all threads hit `v < 0` assert → CUDA_ERROR_ASSERT
            let input: Vec<i32> = (1..=N as i32).collect();
            let input_dev = DeviceBuffer::from_host(&stream, &input)?;
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            let result = unsafe {
                module.debuginfo_assert_fail(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &input_dev,
                    &mut out_dev,
                )
            }
            .and_then(|_| stream.synchronize());
            match result {
                Err(e) => {
                    println!("✓ assertion failure surfaced: {e:?}");
                    Ok(())
                }
                Ok(()) => {
                    eprintln!("✗ expected assertion failure but kernel succeeded");
                    std::process::exit(1);
                }
            }
        }

        "trap_fail" => {
            println!("--- debuginfo_trap_fail (intentional crash) ---");
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            let result = unsafe {
                module.debuginfo_trap_fail(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &mut out_dev,
                )
            }
            .and_then(|_| stream.synchronize());
            match result {
                Err(e) => {
                    println!("✓ trap failure surfaced: {e:?}");
                    Ok(())
                }
                Ok(()) => {
                    eprintln!("✗ expected trap failure but kernel succeeded");
                    std::process::exit(1);
                }
            }
        }

        "breakpoint" => {
            println!("--- debuginfo_breakpoint (software breakpoint, run under cuda-gdb) ---");
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            let result = unsafe {
                module.debuginfo_breakpoint(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &mut out_dev,
                )
            }
            .and_then(|_| stream.synchronize());
            match result {
                Err(e) => {
                    println!("✓ breakpoint failure surfaced (expected outside debugger): {e:?}");
                    Ok(())
                }
                Ok(()) => {
                    println!("✓ breakpoint kernel completed (debugger must have resumed it)");
                    Ok(())
                }
            }
        }

        "oob_index" => {
            println!("--- debuginfo_oob_index (intentional OOB crash) ---");
            // Launch with input.len() < out.len() so threads [input.len()..out.len()-1]
            // hit the Rust bounds-check trap inside input[tid].
            let input: Vec<i32> = vec![10, 20, 30, 40]; // only 4 elements
            let n_out: usize = 8; // 8 output slots → threads 4-7 OOB
            let input_dev = DeviceBuffer::from_host(&stream, &input)?;
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, n_out)?;
            let result = unsafe {
                module.debuginfo_oob_index(
                    &stream,
                    LaunchConfig::for_num_elems(n_out as u32),
                    &input_dev,
                    &mut out_dev,
                )
            }
            .and_then(|_| stream.synchronize());
            match result {
                Err(e) => {
                    println!("✓ OOB trap surfaced: {e:?}");
                    Ok(())
                }
                Ok(()) => {
                    eprintln!("✗ expected OOB trap but kernel succeeded");
                    std::process::exit(1);
                }
            }
        }

        "null_deref" => {
            println!("--- debuginfo_null_deref (intentional crash) ---");
            let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
            let result = unsafe {
                module.debuginfo_null_deref(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &mut out_dev,
                )
            }
            .and_then(|_| stream.synchronize());
            match result {
                Err(e) => {
                    println!("✓ null-deref failure surfaced: {e:?}");
                    Ok(())
                }
                Ok(()) => {
                    eprintln!("✗ expected null-deref failure but kernel succeeded");
                    std::process::exit(1);
                }
            }
        }

        other => {
            eprintln!("unknown kernel: {other}");
            std::process::exit(2);
        }
    }
}
