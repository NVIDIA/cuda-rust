/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]
// MIR-shaped bodies carry rustc's own local names and redundant temps, and
// custom MIR's `Call(..)` terminator reads to clippy as a unit argument.
#![allow(
    clippy::just_underscores_and_digits,
    clippy::similar_names,
    clippy::unit_arg
)]

//! Calls writing through prepared projected destinations.
//!
//! rustc normally lowers an ordinary surface-Rust call whose destination
//! carries a projection into a temporary followed by a store, so chained call
//! destinations rarely survive to code generation. Custom MIR keeps shapes
//! such as `RET.1[i]`, `(*ptr).field`, and `RET[i].field` intact so the importer
//! has to translate the actual destination chain.
//!
//! Three translation paths build or store the call result themselves and so
//! have store sites of their own, exercised separately: a float-math
//! placeholder (`sqrtf32`), whose result must be typed from the projected
//! place rather than the whole local; a plain function call, whose result
//! the function-item path stores itself; and `libm::sincosf`, which packs a
//! `(sin, cos)` tuple and must write it through the projection. Those cases
//! are written so the shape actually reaches the importer under full
//! optimization: inlining rewrites a projected call destination into a
//! temporary plus a store, and GVN sees through a locally-taken pointer, so
//! the bodies are `inline(never)` and the sincos pointer arrives as an
//! opaque argument.
//!
//! The ordinary-call cases run on the device and on the host from the same
//! bodies and must agree. The generated-SREG case is device-only because the
//! host implementation intentionally traps; its observed value is checked
//! against the launch configuration. The `through_mutating_index` case is
//! deliberately adversarial: its callee mutates the runtime index after MIR
//! has fixed the destination, so #1164 requires the store to keep using the
//! pre-call address.
//!
//! Build and run with:
//!   cargo oxide run mir_projected_call_destination

use core::intrinsics::mir::*;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// `(*p) = bswap(x)`, the pointer being to this function's own argument.
///
/// The result has to land in the pointee. Writing it to `_2` instead would
/// leave `_1` untouched, which the returned value reports.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_deref(mut _1: i32) -> i32 {
    mir! {
        type RET = i32;
        let _2: *mut i32;
        {
            _2 = core::ptr::addr_of_mut!(_1);
            Call((*_2) = core::intrinsics::bswap(451059808_i32), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET = _1;
            Return()
        }
    }
}

/// `RET.1 = bswap(x)` on a tuple whose other field is eight bytes wide.
///
/// The result has to land in the second field. Writing it to the whole tuple
/// asks for a cast from a byte to `{ double, i8, [7 x i8] }`, which is the
/// shape LLVM refuses.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_field() -> (f64, u8) {
    mir! {
        type RET = (f64, u8);
        {
            Call(RET.1 = core::intrinsics::bswap(7_u8), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 1.5_f64;
            Return()
        }
    }
}

/// `RET[i] = bswap(x)` with a runtime index.
///
/// The result has to land in the indexed element, leaving the other two at
/// the value the array was initialised with.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_index(mut _1: usize) -> [i32; 3] {
    mir! {
        type RET = [i32; 3];
        {
            RET = [11_i32, 22_i32, 33_i32];
            Call(RET[_1] = core::intrinsics::bswap(451059808_i32), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Callee for the destination-ordering regression.
///
/// The call mutates the runtime index local. MIR semantics require the call
/// destination to be evaluated before entering this callee.
#[inline(never)]
fn mutate_index(_1: *mut usize) -> i32 {
    unsafe { *_1 = 2 };
    7
}

/// `RET[i] = mutate_index(&mut i)`: destination evaluation must precede the call.
///
/// With `_1 = 0`, rustc fixes the destination as `RET[0]` before the callee
/// mutates `_1` to 2, so the correct result is `[7, 0, 0]`. Before #1164 the
/// importer materialized the destination after the call and wrote to `RET[2]`,
/// producing `[0, 0, 7]`; this case pins the corrected ordering.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_mutating_index(mut _1: usize) -> [i32; 3] {
    mir! {
        type RET = [i32; 3];
        let _2: *mut usize;
        {
            RET = [0_i32, 0_i32, 0_i32];
            _2 = core::ptr::addr_of_mut!(_1);
            Call(RET[_1] = mutate_index(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// `RET.1 = sqrtf32(x)` on a tuple whose other field is eight bytes wide.
///
/// Unlike `bswap`, a float-math intrinsic keeps a placeholder call in the
/// translation, and the placeholder's result is typed from the destination.
/// Typed from the whole tuple local instead of the projected field, the
/// store would aim a `{ i64, float }` at the field's `float` slot, which the
/// verifier refuses. `sqrt` is correctly rounded on host and device alike,
/// so the bit-exact comparison below is safe.
///
/// `inline(never)` matters: inlining rewrites a projected call destination
/// into a fresh temporary plus an ordinary store, which would erase the very
/// shape this case exists to reach.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_field_float(_1: f32) -> (u64, f32) {
    mir! {
        type RET = (u64, f32);
        {
            Call(RET.1 = core::intrinsics::sqrtf32(_1), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 3_u64;
            Return()
        }
    }
}

/// `RET[i] = sqrtf32(x)` with a runtime index, on a float array.
///
/// The index spelling of the same float-math shape. Unlike the field one it
/// survives even inlining (the inliner keeps index projections), so it
/// reaches the importer under every optimization decision.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_index_float(mut _1: usize, _2: f32) -> [f32; 3] {
    mir! {
        type RET = [f32; 3];
        {
            RET = [1.0_f32, 1.0_f32, 1.0_f32];
            Call(RET[_1] = core::intrinsics::sqrtf32(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Callee for the plain-function case; `inline(never)` keeps the call (and
/// with it the projected destination) alive in the caller's MIR.
#[inline(never)]
fn double_it(x: u32) -> u32 {
    x.wrapping_mul(2)
}

/// `RET.1 = double_it(x)`: a plain function call, not an intrinsic.
///
/// Ordinary surface-Rust calls lower a projected destination to a temporary
/// before codegen, but custom MIR hands the projection straight to the
/// importer. The function-item path (and its closure sibling) has a store
/// site of its own, separate from the intrinsic ones, and must write the
/// call result through the projection there too.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_field_fn(_1: u32) -> (u64, u32) {
    mir! {
        type RET = (u64, u32);
        {
            Call(RET.1 = double_it(_1), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 5_u64;
            Return()
        }
    }
}

/// `(*p) = sincosf(x)` through a pointer this function receives opaquely.
///
/// `sincos` packs the `(sin, cos)` pair itself before storing, so it has a
/// store site of its own: the pair is typed from the pointee and has to be
/// written through the pointer. Written to the pointer's local instead, a
/// tuple would be aimed at a pointer slot. The pointer must arrive as an
/// argument: were it materialized here from a local, GVN would see through
/// it and rewrite `(*p)` back to the plain local. The angle 0 keeps the
/// comparison exact: both sides produce `(+0.0, 1.0)` bitwise.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_deref_sincos(_1: *mut (f32, f32), _2: f32) {
    mir! {
        {
            Call((*_1) = libm::sincosf(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// `RET.1[i] = double_it(x)`: Field -> Index chain.
///
/// The first projection selects the array field and the second selects one
/// runtime element. Both projections must be materialized before the callee
/// executes, then the result is written through that prepared address.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_field_index(mut _1: usize, _2: u32) -> (u64, [u32; 3]) {
    mir! {
        type RET = (u64, [u32; 3]);
        {
            RET.1 = [3_u32, 5_u32, 7_u32];
            Call(RET.1[_1] = double_it(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 11_u64;
            Return()
        }
    }
}

/// `(*p).1 = double_it(x)`: Deref -> Field chain.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_deref_field(_1: *mut (u64, u32), _2: u32) {
    mir! {
        {
            Call((*_1).1 = double_it(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// `RET[i].1 = double_it(x)`: Index -> Field chain.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_index_field(mut _1: usize, _2: u32, _3: [(u64, u32); 2]) -> [(u64, u32); 2] {
    mir! {
        type RET = [(u64, u32); 2];
        {
            RET = _3;
            Call(RET[_1].1 = double_it(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Bare-local control: the prepared-destination abstraction must leave the
/// established `ValueMap::store_local` path unchanged.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_bare_local(_1: u32) -> u32 {
    mir! {
        type RET = u32;
        {
            Call(RET = double_it(_1), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Generated intrinsic acceptance case.
///
/// `thread::blockDim_x()` is generated from the SREG catalog and is replaced
/// by the importer during device compilation. Its host body intentionally
/// traps, so this helper is called only from the kernel. The nested
/// `RET.1[i]` destination proves that generated dispatch consumes the same
/// prepared-destination abstraction as ordinary calls.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_generated_sreg(mut _1: usize) -> (u64, [u32; 3]) {
    mir! {
        type RET = (u64, [u32; 3]);
        {
            RET.1 = [0_u32, 0_u32, 0_u32];
            Call(RET.1[_1] = thread::blockDim_x(), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 29_u64;
            Return()
        }
    }
}

fn fold_case_results(results: &[u64; 12]) -> u64 {
    let mut folded = 0_u64;
    let mut i = 0_usize;
    while i < results.len() {
        folded ^= results[i].rotate_left((i as u32 * 5) & 63);
        i += 1;
    }
    folded
}

fn fold_generated_sreg(result: (u64, [u32; 3])) -> u64 {
    result.0
        ^ u64::from(result.1[0]).rotate_left(8)
        ^ u64::from(result.1[1]).rotate_left(24)
        ^ u64::from(result.1[2]).rotate_left(40)
}

fn expected_generated_sreg(block_dim_x: u32) -> u64 {
    fold_generated_sreg((29_u64, [0_u32, block_dim_x, 0_u32]))
}

/// Folds twelve host/device-comparable results into one word per case.
fn case_results() -> [u64; 12] {
    let deref = through_deref(0) as u32 as u64;

    let field = through_field();
    let field = field.0.to_bits() ^ u64::from(field.1);

    let indexed = through_index(1);
    let index = (indexed[0] as u32 as u64)
        ^ ((indexed[1] as u32 as u64) << 8)
        ^ ((indexed[2] as u32 as u64) << 16);

    let mutated_index = through_mutating_index(0);
    let mutating_index = (mutated_index[0] as u32 as u64)
        ^ ((mutated_index[1] as u32 as u64) << 16)
        ^ ((mutated_index[2] as u32 as u64) << 32);

    let field_float = through_field_float(2.0_f32);
    let field_float = field_float.0 ^ u64::from(field_float.1.to_bits());

    let index_float = through_index_float(1, 2.0_f32);
    let index_float = u64::from(index_float[0].to_bits())
        ^ u64::from(index_float[1].to_bits()).rotate_left(8)
        ^ u64::from(index_float[2].to_bits()).rotate_left(16);

    let field_fn = through_field_fn(21);
    let field_fn = field_fn.0 ^ u64::from(field_fn.1);

    let mut pair = (9.0_f32, 9.0_f32);
    through_deref_sincos(&raw mut pair, 0.0_f32);
    let sincos = u64::from(pair.0.to_bits()) ^ (u64::from(pair.1.to_bits()) << 32);

    let field_index_value = through_field_index(2, 13);
    let field_index = field_index_value.0
        ^ u64::from(field_index_value.1[0]).rotate_left(8)
        ^ u64::from(field_index_value.1[1]).rotate_left(24)
        ^ u64::from(field_index_value.1[2]).rotate_left(40);

    let mut deref_field_value = (23_u64, 3_u32);
    through_deref_field(&raw mut deref_field_value, 17);
    let deref_field = deref_field_value.0 ^ u64::from(deref_field_value.1).rotate_left(32);

    let index_field_value = through_index_field(1, 19, [(17_u64, 1_u32), (19_u64, 2_u32)]);
    let index_field = index_field_value[0].0
        ^ u64::from(index_field_value[0].1).rotate_left(8)
        ^ index_field_value[1].0.rotate_left(24)
        ^ u64::from(index_field_value[1].1).rotate_left(48);

    let bare_local = u64::from(through_bare_local(23));

    [
        deref,
        field,
        index,
        mutating_index,
        field_float,
        index_float,
        field_fn,
        sincos,
        field_index,
        deref_field,
        index_field,
        bare_local,
    ]
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn projected_destinations(mut out: DisjointSlice<u64>) {
        let host_comparable = fold_case_results(&case_results());
        let generated = fold_generated_sreg(through_generated_sreg(1));
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = host_comparable ^ generated.rotate_left(17);
        }
    }
}

fn main() {
    let host = case_results();
    let host_comparable = fold_case_results(&host);

    println!("=== calls writing through prepared projected destinations ===\n");
    println!("host  deref case:          0x{:016x}", host[0]);
    println!("host  field case:          0x{:016x}", host[1]);
    println!("host  index case:          0x{:016x}", host[2]);
    println!("host  mutating index case: 0x{:016x}", host[3]);
    println!("host  float field case:    0x{:016x}", host[4]);
    println!("host  float index case:    0x{:016x}", host[5]);
    println!("host  fn field case:       0x{:016x}", host[6]);
    println!("host  sincos case:         0x{:016x}", host[7]);
    println!("host  Field -> Index:      0x{:016x}", host[8]);
    println!("host  Deref -> Field:      0x{:016x}", host[9]);
    println!("host  Index -> Field:      0x{:016x}", host[10]);
    println!("host  bare-local control:  0x{:016x}", host[11]);

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("alloc out");
    let module = kernels::load(&ctx).expect("failed to load device module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let generated_expected = expected_generated_sreg(cfg.block_dim.0);
    let host_folded = host_comparable ^ generated_expected.rotate_left(17);

    // SAFETY: the one argument matches `projected_destinations`' single slice
    // parameter, and `out` is a live DeviceBuffer allocated above.
    unsafe { module.projected_destinations(&stream, cfg, &mut out) }.expect("kernel launch failed");

    let device_folded = out.to_host_vec(&stream).expect("readback")[0];
    println!("host  generated SREG expected: 0x{generated_expected:016x}");
    println!("\nhost  folded:     0x{host_folded:016x}");
    println!("device folded:    0x{device_folded:016x}");

    if device_folded == host_folded {
        println!(
            "\nPASS: host/device agree on twelve call-destination cases and the generated SREG projection"
        );
    } else {
        println!("\nFAIL: device and host disagree");
        std::process::exit(1);
    }
}
