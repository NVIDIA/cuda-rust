/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-crate merged-bundle symbol regression test (#1277).
//!
//! Two crates each define a static `SharedArray`, an ordinary `static mut`
//! device global, and a kernel that reaches the dynamic shared-memory pool
//! through one shared helper (`merged_symbols_lib::pool_base_addr`). Before
//! module-scope symbols carried the compiling crate's stable id, both
//! bundles named their first shared allocation `__shared_mem_0` and their
//! first device global `__device_global_0`, and
//! `load_all_ptx_bundles_merged` produced a module the driver rejected:
//!
//! ```text
//! DriverError(218, "a PTX JIT compilation failed")
//! ```
//!
//! The shared dynamic-pool helper is subtler: its `.extern .shared` symbol
//! appeared in both bundles under one name with *different* alignments (128
//! from this library kernel's launch contract, 16 from the binary kernel's),
//! and the merged module silently kept whichever declaration came first —
//! so the stronger alignment could be lost depending on bundle order.
//!
//! This example checks both properties in BOTH merge orders: the merged
//! module namespace has no colliding family symbols, both orders JIT-load,
//! every kernel reads its own crate's values, and each crate's pool keeps
//! its own alignment.
//!
//! Run: cargo oxide run cross_crate_merged_symbols

use std::error::Error;
use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::embedded::artifact_bundles_from_current_exe;
use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, launch_kernel_on_stream};
use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::{load_all_ptx_bundles_merged, merge_ptx_bundles};
use merged_symbols_lib::LIB_TAG;

/// Value published through the binary crate's device global.
pub const BIN_TAG: u32 = 0x33CC_44DD;

static mut BIN_SLOT: u32 = BIN_TAG;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn bin_shared_roundtrip(mut out: DisjointSlice<f64>) {
        static mut S: SharedArray<f64, 1> = SharedArray::UNINIT;
        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 accesses slot 0, before the block barrier.
            unsafe { S[0] = 2.5 };
        }
        thread::sync_threads();
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked.
            unsafe { *out.as_mut_ptr() = S[0] };
        }
    }

    #[kernel]
    pub fn bin_global_read(mut out: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked; the static is only read.
            unsafe { *out.as_mut_ptr() = BIN_SLOT };
        }
    }

    /// Same pool helper as the library's `lib_pool_addr`, weaker alignment:
    /// this is the pair whose merged extern declarations used to resolve by
    /// bundle order.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 256,
        dynamic_shared_alignment = 16,
    )]
    pub fn bin_pool_addr(mut out: DisjointSlice<u64>) {
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked.
            unsafe { *out.as_mut_ptr() = merged_symbols_lib::pool_base_addr() };
        }
    }
}

/// All whole identifiers in `ptx` that start with `family`.
fn distinct_family_symbols(ptx: &str, family: &str) -> std::collections::BTreeSet<String> {
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'$';
    let bytes = ptx.as_bytes();
    let mut found = std::collections::BTreeSet::new();
    let mut start = 0;
    while let Some(pos) = ptx[start..].find(family) {
        let begin = start + pos;
        // Whole-identifier boundary on the left; extend to the right edge.
        if begin == 0 || !is_ident(bytes[begin - 1]) {
            let mut end = begin;
            while end < bytes.len() && is_ident(bytes[end]) {
                end += 1;
            }
            found.insert(ptx[begin..end].to_string());
            start = end;
        } else {
            start = begin + family.len();
        }
    }
    found
}

/// The merged module must define each crate's symbols under distinct names,
/// and both dynamic-pool extern declarations must survive with their own
/// alignments — in every bundle order.
fn assert_merged_namespace(label: &str, ptx: &str) {
    for family in ["__shared_mem_", "__device_global_"] {
        let symbols = distinct_family_symbols(ptx, family);
        assert_eq!(
            symbols.len(),
            2,
            "{label}: expected the two crates' {family} symbols to stay \
             distinct in the merged module, got {symbols:?}"
        );
    }

    let extern_shared: Vec<&str> = ptx
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with(".extern .shared"))
        .collect();
    assert_eq!(
        extern_shared.len(),
        2,
        "{label}: expected one dynamic-pool extern per crate, got {extern_shared:?}"
    );
    // Trailing space keeps ".align 16 " from matching inside ".align 128 ".
    for align in [".align 128 ", ".align 16 "] {
        assert_eq!(
            extern_shared
                .iter()
                .filter(|line| line.contains(align))
                .count(),
            1,
            "{label}: expected exactly one dynamic-pool extern with {align}, \
             got {extern_shared:?}"
        );
    }
    let pool_symbols = distinct_family_symbols(ptx, "__dynamic_smem_");
    assert_eq!(
        pool_symbols.len(),
        2,
        "{label}: expected the shared pool helper's symbols to stay distinct \
         across crates, got {pool_symbols:?}"
    );
}

/// Launch a single-thread-block kernel writing one `$ty` and return it.
macro_rules! read_single {
    ($ty:ty, $stream:expr, $module:expr, $name:literal, $smem:expr) => {{
        let function = $module.load_function($name)?;
        let out = DeviceBuffer::<$ty>::zeroed($stream, 1)?;
        let mut ptr = out.cu_deviceptr();
        let mut len: u64 = 1;
        // SAFETY: the argument list mirrors the kernel's (DisjointSlice<$ty>)
        // parameter as (pointer, length); the buffer outlives the launch and
        // the stream belongs to the module's context.
        unsafe {
            launch_kernel_on_stream(
                &function,
                (1, 1, 1),
                (32, 1, 1),
                $smem,
                $stream,
                &mut [
                    (&raw mut ptr).cast::<c_void>(),
                    (&raw mut len).cast::<c_void>(),
                ],
            )?;
        }
        out.to_host_vec($stream)?[0]
    }};
}

/// Run all six kernels from one merged module and check each reads its own
/// crate's values and pool.
fn check_merged_module(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let lib_shared = read_single!(f64, stream, module, "lib_shared_roundtrip", 0);
    let bin_shared = read_single!(f64, stream, module, "bin_shared_roundtrip", 0);
    assert_eq!(
        lib_shared, 1.5,
        "{label}: lib kernel read foreign shared slot"
    );
    assert_eq!(
        bin_shared, 2.5,
        "{label}: bin kernel read foreign shared slot"
    );

    let lib_tag = read_single!(u32, stream, module, "lib_global_read", 0);
    let bin_tag = read_single!(u32, stream, module, "bin_global_read", 0);
    assert_eq!(lib_tag, LIB_TAG, "{label}: lib kernel read foreign global");
    assert_eq!(bin_tag, BIN_TAG, "{label}: bin kernel read foreign global");

    // The launch contracts promise 256 pool bytes; the 128-contract kernel's
    // pool base must actually be 128-aligned regardless of bundle order.
    let lib_pool = read_single!(u64, stream, module, "lib_pool_addr", 256);
    let bin_pool = read_single!(u64, stream, module, "bin_pool_addr", 256);
    assert_eq!(
        lib_pool % 128,
        0,
        "{label}: 128-aligned pool contract lost in merge (base {lib_pool:#x})"
    );
    assert_eq!(
        bin_pool % 16,
        0,
        "{label}: 16-aligned pool contract lost in merge (base {bin_pool:#x})"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Anchor the library's artifact bundle into this executable through its
    // generated host loader, and check the library's own load path works.
    // SAFETY: the library's bundle is embedded in this executable and its
    // entry definitions are generated by that crate's #[cuda_module].
    let _lib_module = unsafe { merged_symbols_lib::kernels::load(&ctx)? };

    // Both merge orders must produce a collision-free namespace...
    let bundles = artifact_bundles_from_current_exe()?;
    let forward = merge_ptx_bundles(bundles.iter())?;
    let reversed = merge_ptx_bundles(bundles.iter().rev())?;
    assert_merged_namespace("forward order", &forward);
    assert_merged_namespace("reversed order", &reversed);

    // ...and both must JIT-load. The forward order is the public loader path
    // that #1277 reported failing with CUDA_ERROR_INVALID_PTX.
    let merged = load_all_ptx_bundles_merged(&ctx)?;
    let merged_reversed = ctx.load_module_from_image(reversed.as_bytes())?;

    check_merged_module(&stream, &merged, "forward order")?;
    check_merged_module(&stream, &merged_reversed, "reversed order")?;

    println!(
        "SUCCESS: merged bundles keep per-crate shared, global, and dynamic-pool symbols distinct in both orders"
    );
    Ok(())
}
