/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in removal of indexing bounds checks
//! (`#[kernel(unchecked_indexing)]`).
//!
//! Normally every `a[i]` in a kernel compiles to "is `i` inside the slice?
//! if not, stop the kernel" (a compare plus a branch to a `trap;` block in
//! PTX). The flag deletes that check: faster, but an out-of-bounds index
//! becomes undefined behavior, the same deal as `get_unchecked`.
//!
//! Two kernels here have byte-identical bodies and index with a raw thread
//! id the compiler cannot relate to the slice lengths, so it cannot remove
//! the checks on its own:
//!
//! - `indexed_sum_checked` uses plain `#[kernel]`: its PTX entry contains
//!   the guarded branches and `trap;` blocks.
//! - `indexed_sum_unchecked` adds `unchecked_indexing`: its PTX entry
//!   contains no `trap;` at all. The host sizes the buffers so every access
//!   really is in bounds.
//!
//! A third scenario guards against the flag leaking. A generic opted kernel
//! (`scaled_gather<T>`) expands into a user-named helper function plus a
//! generated kernel entry; only the entry may carry the flag. The kernel
//! `gather_then_check` never opts in but calls that helper and then does its
//! own unprovable indexing. If the flag ever traveled with the helper, the
//! caller would silently lose ALL of its own bounds checks without asking
//! for it. The assertion here pins the fix: the caller's entry keeps its
//! traps.
//!
//! The host runs the kernels, asserts identical results against a CPU
//! reference, and then inspects the generated PTX: traps present in the
//! checked entry and in the non-opted caller entry, zero traps in both
//! opted entries, and no leftover `__unchecked_indexing_config` marker
//! anywhere in the module.
//!
//! Build and run with:
//!   cargo oxide run unchecked_indexing

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 1024;

// =============================================================================
// KERNELS - identical bodies; only the attribute differs
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Baseline: bounds checks stay (idx is unrelated to the slice lengths,
    /// so rustc keeps a compare + trap for each indexing expression).
    #[kernel]
    pub fn indexed_sum_checked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[idx_raw] + b[idx_raw] + a[2 * idx_raw + 1];
        }
    }

    /// Same body, opted in: every slice bounds check is elided. Out-of-bounds
    /// indexing is UB here, exactly like `get_unchecked`.
    #[kernel(unchecked_indexing)]
    pub fn indexed_sum_unchecked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[idx_raw] + b[idx_raw] + a[2 * idx_raw + 1];
        }
    }

    /// Generic kernel, opted in: elision composes with generics. The macro
    /// expands this into a user-named `#[inline(always)]` implementation
    /// helper plus a generated entry wrapper; only the wrapper carries the
    /// `__unchecked_indexing_config` marker, so the helper stays ordinary
    /// callable (and bounds-checked) Rust.
    #[kernel(unchecked_indexing)]
    pub fn scaled_gather<T: Copy + core::ops::Add<Output = T>>(
        a: &[T],
        b: &[T],
        mut c: DisjointSlice<T>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[idx_raw] + b[idx_raw];
        }
    }

    /// Leak regression: this kernel never opts in, but it calls the generic
    /// opted kernel's user-named helper and then performs its own unprovable
    /// indexing. rustc MIR-inlines the `#[inline(always)]` helper into this
    /// body; if the helper carried the elision marker, the MIR importer would
    /// silently elide THIS kernel's bounds checks too. Its PTX entry must
    /// keep its compare-and-trap lowering.
    #[kernel(launch_context = lc)]
    pub fn gather_then_check(
        a: &[f32],
        b: &[f32],
        c: DisjointSlice<f32>,
        mut d: DisjointSlice<f32>,
    ) {
        // Reuse the opted generic kernel's implementation: c[i] = a[i] + b[i].
        scaled_gather::<f32>(a, b, c, lc);
        // This kernel's own unprovable indexing must keep its bounds checks.
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out) = d.get_mut(idx) {
            *out = a[2 * idx_raw + 1] + b[idx_raw];
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unchecked Indexing (bounds-check elision) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // `a` is sized 2N so the strided access a[2*idx+1] stays in bounds for
    // every launched thread (idx in 0..N).
    let a_host: Vec<f32> = (0..2 * N).map(|i| (i % 97) as f32 * 0.5).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i % 61) as f32 * 2.0).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_checked_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut c_unchecked_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // SAFETY: launch shape/resources match the kernels; `a` (2N), `b` (N) and
    // `c` (N) cover every access made by the N launched threads.
    unsafe {
        module.indexed_sum_checked(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_checked_dev,
        )
    }
    .expect("checked kernel launch failed");

    // SAFETY: same shape and buffers as above; additionally, every index is
    // in bounds, which the unchecked kernel requires (UB otherwise).
    unsafe {
        module.indexed_sum_unchecked(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_unchecked_dev,
        )
    }
    .expect("unchecked kernel launch failed");

    // Leak-regression scenario: the opted generic kernel plus the non-opted
    // caller that reuses its implementation helper.
    let mut c_gather_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut c_caller_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut d_caller_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; every index is in
    // bounds, which the opted (unchecked) generic kernel requires.
    unsafe {
        module.scaled_gather::<f32>(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_gather_dev,
        )
    }
    .expect("scaled_gather::<f32> launch failed");

    // SAFETY: same shape; `a` (2N), `b` (N), `c`/`d` (N) cover every access
    // made by the N launched threads (the kernel itself is bounds-checked).
    unsafe {
        module.gather_then_check(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_caller_dev,
            &mut d_caller_dev,
        )
    }
    .expect("gather_then_check launch failed");

    let c_checked = c_checked_dev.to_host_vec(&stream).unwrap();
    let c_unchecked = c_unchecked_dev.to_host_vec(&stream).unwrap();
    let c_gather = c_gather_dev.to_host_vec(&stream).unwrap();
    let c_caller = c_caller_dev.to_host_vec(&stream).unwrap();
    let d_caller = d_caller_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i] + a_host[2 * i + 1];
        if (c_checked[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  checked kernel error at [{i}]: expected {expected}, got {}",
                    c_checked[i]
                );
            }
            errors += 1;
        }
        if c_unchecked[i].to_bits() != c_checked[i].to_bits() {
            if errors < 5 {
                eprintln!(
                    "  kernels disagree at [{i}]: checked {} vs unchecked {}",
                    c_checked[i], c_unchecked[i]
                );
            }
            errors += 1;
        }
        let gather_expected = a_host[i] + b_host[i];
        if (c_gather[i] - gather_expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  scaled_gather error at [{i}]: expected {gather_expected}, got {}",
                    c_gather[i]
                );
            }
            errors += 1;
        }
        if c_caller[i].to_bits() != c_gather[i].to_bits() {
            if errors < 5 {
                eprintln!(
                    "  helper reuse disagrees at [{i}]: entry {} vs caller {}",
                    c_gather[i], c_caller[i]
                );
            }
            errors += 1;
        }
        let d_expected = a_host[2 * i + 1] + b_host[i];
        if (d_caller[i] - d_expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  gather_then_check error at [{i}]: expected {d_expected}, got {}",
                    d_caller[i]
                );
            }
            errors += 1;
        }
    }
    if errors != 0 {
        eprintln!("\nFAILED: {errors} mismatches between kernels/reference");
        std::process::exit(1);
    }
    println!("Both kernels produced identical, correct results.");

    if let Err(error) = verify_ptx() {
        eprintln!("\nFAILED PTX verification: {error}");
        std::process::exit(1);
    }
    println!("PTX structure verified: traps in checked entries, none in opted-in entries.");

    println!("\nSUCCESS: unchecked_indexing elides bounds checks; default is unchanged");
}

// =============================================================================
// PTX STRUCTURAL ASSERTIONS
// =============================================================================

fn verify_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("unchecked_indexing.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    let document = ptx_parse::Document::parse(&ptx)?;

    // The compiler marker must never survive into the generated module,
    // neither as a call nor as a stray declaration.
    if ptx.contains("__unchecked_indexing_config") {
        return Err("compile-time marker `__unchecked_indexing_config` leaked into PTX".into());
    }

    let checked = entry(&document, "indexed_sum_checked")?;
    let unchecked = entry(&document, "indexed_sum_unchecked")?;

    // Default behavior: the checked kernel keeps its bounds checks, meaning
    // at least one trap block plus a guarded branch that jumps around/into it.
    let checked_traps = count_traps(checked);
    if checked_traps == 0 {
        return Err("checked entry has no `trap;`, default bounds checks disappeared".into());
    }
    if !has_guarded_branch(checked) {
        return Err("checked entry has no guarded `bra` for its bounds checks".into());
    }

    // Opt-in behavior: the unchecked kernel must contain zero traps.
    let unchecked_traps = count_traps(unchecked);
    if unchecked_traps != 0 {
        return Err(format!(
            "unchecked entry still contains {unchecked_traps} `trap;` instruction(s)"
        )
        .into());
    }

    // Generic opted kernel: the entry wrapper (named `scaled_gather_TID_<hash>`
    // by the generic-kernel naming scheme) carries the marker, so its entry
    // must contain zero traps.
    let gather = entry_by_prefix(&document, "scaled_gather")?;
    let gather_traps = count_traps(gather);
    if gather_traps != 0 {
        return Err(format!(
            "generic opted entry (scaled_gather) still contains {gather_traps} `trap;` instruction(s)"
        )
        .into());
    }

    // Leak regression: the NON-opted caller reuses the generic kernel's
    // user-named implementation helper, which rustc MIR-inlines into the
    // caller. The helper body must not carry the elision marker, so the
    // caller's own (and the inlined helper's) bounds checks must survive.
    let caller = entry(&document, "gather_then_check")?;
    let caller_traps = count_traps(caller);
    if caller_traps == 0 {
        return Err(
            "non-opted `gather_then_check` entry has no `trap;`: the unchecked_indexing \
             marker leaked out of the generic implementation helper into a caller that \
             never opted in"
                .into(),
        );
    }
    if !has_guarded_branch(caller) {
        return Err("gather_then_check entry has no guarded `bra` for its bounds checks".into());
    }

    println!(
        "  {checked_traps} trap(s) in indexed_sum_checked, {unchecked_traps} in \
         indexed_sum_unchecked, {gather_traps} in scaled_gather (generic, opted), \
         {caller_traps} in gather_then_check (non-opted caller)"
    );
    Ok(())
}

fn entry<'document, 'source>(
    document: &'document ptx_parse::Document<'source>,
    name: &str,
) -> Result<ptx_parse::CallableDefinition<'document, 'source>, Box<dyn std::error::Error>> {
    entry_from(
        document
            .definitions_named(name)
            .find(|definition| definition.callable().kind() == ptx_parse::CallableKind::Entry),
        name,
    )
}

/// Like [`entry`], but matches on an entry-name prefix. Generic kernel
/// entries are exported as `<name>_TID_<hex32>`, where the hash depends on
/// the concrete instantiation.
fn entry_by_prefix<'document, 'source>(
    document: &'document ptx_parse::Document<'source>,
    prefix: &str,
) -> Result<ptx_parse::CallableDefinition<'document, 'source>, Box<dyn std::error::Error>> {
    entry_from(
        document.definitions().find(|definition| {
            definition.callable().kind() == ptx_parse::CallableKind::Entry
                && definition.callable().name().starts_with(prefix)
        }),
        prefix,
    )
}

fn entry_from<'document, 'source>(
    definition: Option<ptx_parse::CallableDefinition<'document, 'source>>,
    name: &str,
) -> Result<ptx_parse::CallableDefinition<'document, 'source>, Box<dyn std::error::Error>> {
    definition.ok_or_else(|| format!("missing or incomplete PTX entry `{name}`").into())
}

fn count_traps(definition: ptx_parse::CallableDefinition<'_, '_>) -> usize {
    definition
        .instructions()
        .filter(|instruction| instruction.base_opcode() == "trap")
        .count()
}

/// A predicated branch such as `@%p1 bra $L__BB0_4;`, the shape of a bounds
/// guard (and of the `get_mut` Option branch).
fn has_guarded_branch(definition: ptx_parse::CallableDefinition<'_, '_>) -> bool {
    definition
        .instructions()
        .any(|instruction| instruction.base_opcode() == "bra" && instruction.predicate().is_some())
}
