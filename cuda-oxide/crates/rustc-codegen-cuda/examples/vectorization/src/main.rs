/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Alignment-driven vectorization across the CUDA built-in vector types.
//!
//! Each type is the Rust equivalent of a CUDA vector type (e.g. `f32x4` =
//! `float4`), with its exact CUDA size and alignment. A per-type copy kernel
//! (`output[i] = input[i]`) shows how alignment governs whether the whole-
//! element load/store fuses into a vectorized `ld/st.global.v*` or stays
//! scalar. For every type, `main` checks the Rust layout matches CUDA, the
//! round-trip copy is bit-correct on the GPU, and reports/asserts the codegen.
//!
//! Run: `cargo oxide run vectorization`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{access, cuda_module};

/// Elements (or quads, for the view kernel) per launch; one per thread.
const N: usize = 256;

/// One row of the per-type report.
struct Row {
    name: &'static str,
    cuda: &'static str,
    size: usize,
    align: usize,
    kernel: &'static str,
    copy_ok: bool,
}

/// Generate, from a single table, the CUDA-style vector types, the
/// `#[cuda_module]` of per-type copy kernels, and `run_all` (launch + verify
/// each). The macro emits the *whole* `#[cuda_module]` so the kernel repetition
/// expands before the attribute macro runs (a `macro_rules!` *inside* the
/// module would be invisible to it).
macro_rules! vector_suite {
    ($($ty:ident: $base:ty; $n:literal; align $align:literal; size $size:literal; $cuda:literal,)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, PartialEq, Debug)]
            pub struct $ty([$base; $n]);
            // Plain POD aggregate, no pointers: safe to memcpy to/from the device.
            unsafe impl cuda_core::DeviceCopy for $ty {}
        )*

        #[cuda_module]
        mod kernels {
            use cuda_device::{kernel, thread, DisjointSlice};
            $(
                #[kernel]
                pub fn $ty(input: &[super::$ty], mut output: DisjointSlice<super::$ty>) {
                    let idx = thread::index_1d();
                    let i = idx.get();
                    if let Some(o) = output.get_mut(idx) {
                        *o = input[i];
                    }
                }
            )*
        }

        /// Launch every kernel, asserting each type's Rust layout matches CUDA
        /// and its copy round-trips.
        fn run_all(
            module: &kernels::LoadedModule,
            stream: &cuda_core::CudaStream,
            cfg: LaunchConfig,
            n: usize,
        ) -> Vec<Row> {
            let mut rows = Vec::new();
            $(
                {
                    assert_eq!(core::mem::size_of::<$ty>(), $size, concat!(stringify!($ty), " size"));
                    assert_eq!(core::mem::align_of::<$ty>(), $align, concat!(stringify!($ty), " align"));
                    let input: Vec<$ty> = (0..n)
                        .map(|i| $ty(core::array::from_fn(|j| (i * $n + j + 1) as $base)))
                        .collect();
                    let in_dev = DeviceBuffer::from_host(stream, &input).unwrap();
                    let mut out_dev = DeviceBuffer::<$ty>::zeroed(stream, n).unwrap();
                    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                    unsafe { module
                        .$ty(stream, cfg, &in_dev, &mut out_dev) }
                        .expect(concat!(stringify!($ty), " launch"));
                    let out = out_dev.to_host_vec(stream).unwrap();
                    rows.push(Row {
                        name: stringify!($ty),
                        cuda: $cuda,
                        size: $size,
                        align: $align,
                        kernel: stringify!($ty),
                        copy_ok: out == input,
                    });
                }
            )*
            rows
        }
    };
}

vector_suite! {
    i8x1: i8; 1; align 1; size 1; "char1",
    i8x2: i8; 2; align 2; size 2; "char2",
    i8x3: i8; 3; align 1; size 3; "char3",
    i8x4: i8; 4; align 4; size 4; "char4",
    i16x1: i16; 1; align 2; size 2; "short1",
    i16x2: i16; 2; align 4; size 4; "short2",
    i16x3: i16; 3; align 2; size 6; "short3",
    i16x4: i16; 4; align 8; size 8; "short4",
    i32x1: i32; 1; align 4; size 4; "int1",
    i32x2: i32; 2; align 8; size 8; "int2",
    i32x3: i32; 3; align 4; size 12; "int3",
    i32x4: i32; 4; align 16; size 16; "int4",
    i64x1: i64; 1; align 8; size 8; "longlong1",
    i64x2: i64; 2; align 16; size 16; "longlong2",
    i64x3: i64; 3; align 8; size 24; "longlong3",
    i64x4: i64; 4; align 16; size 32; "longlong4",
    i64x4_a32: i64; 4; align 32; size 32; "longlong4_32a",
    f32x1: f32; 1; align 4; size 4; "float1",
    f32x2: f32; 2; align 8; size 8; "float2",
    f32x3: f32; 3; align 4; size 12; "float3",
    f32x4: f32; 4; align 16; size 16; "float4",
    f64x1: f64; 1; align 8; size 8; "double1",
    f64x2: f64; 2; align 16; size 16; "double2",
    f64x3: f64; 3; align 8; size 24; "double3",
    f64x4: f64; 4; align 16; size 32; "double4",
    f64x4_a32: f64; 4; align 32; size 32; "double4_32a",
}

/// Kernels over *flat* `f32` buffers: no over-aligned host type at all. The
/// view is taken inside the kernel with `cuda_device::vector::as_vectors`,
/// which is the intended path when the host allocation is flat.
#[cuda_module]
mod view_kernels {
    use cuda_device::vector::{self, F32x4};
    use cuda_device::{DisjointSlice, kernel, thread};

    #[repr(C, packed)]
    #[allow(dead_code)]
    struct PackedEmptyElement {
        tag: u8,
        value: u32,
    }

    #[allow(dead_code)]
    union UnionEmptyElement {
        pointer: *mut u32,
        bits: u64,
    }

    #[inline(never)]
    fn promoted_empty_ptr<T>() -> *mut T {
        let empty: &mut [T; 0] = &mut [];
        empty.as_mut_ptr()
    }

    /// Copy one `F32x4` quad per thread between flat `f32` buffers through
    /// checked [`vector::as_vectors`] views. The quad is moved whole, never
    /// decomposed into lanes, so both the load and the store fuse into
    /// 128-bit transactions (`ld/st.global.v4.f32`).
    #[kernel]
    pub fn f32x4_view_copy(input: &[f32], mut output: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let out_len = output.len();
        // SAFETY: `DisjointSlice` grants this launch exclusive access to the
        // whole output buffer; the flat view aliases nothing else.
        let out_flat = unsafe { core::slice::from_raw_parts_mut(output.as_mut_ptr(), out_len) };
        let Some(quads) = vector::as_vectors::<F32x4>(input) else {
            return;
        };
        let Some(out_quads) = vector::as_vectors_mut::<F32x4>(out_flat) else {
            return;
        };
        if i < quads.len() && i < out_quads.len() {
            out_quads[i] = quads[i];
        }
    }

    /// Compile-only coverage for rustc-promoted `&mut [T; 0]` when `T` has a
    /// packed layout, a union, a unit, or a fat-pointer value. No element
    /// storage exists, but the promoted pointer must retain `T`'s alignment
    /// and exact semantic type.
    #[kernel]
    pub fn promoted_empty_array_kinds(mut output: DisjointSlice<usize>) {
        let Some(slot) = output.get_mut(thread::index_1d()) else {
            return;
        };
        let packed = promoted_empty_ptr::<PackedEmptyElement>() as usize;
        let union = promoted_empty_ptr::<UnionEmptyElement>() as usize;
        let unit = promoted_empty_ptr::<()>() as usize;
        let slice_value = promoted_empty_ptr::<&'static [u8]>() as usize;
        *slot = packed ^ union ^ unit ^ slice_value;
    }
}

/// The shape `f32x4_view_copy` relies on, derived by [`access::plan`] instead
/// of by hand: a 128-bit transaction of `f32` is one 4-element quad per
/// thread and needs `F32x4`'s 16-byte alignment, and the flat `N * 4`-element
/// buffer is exactly one pass for the `N`-thread block `main` launches.
const _: () = {
    let plan = match access::plan::<f32>(access::TXN_128, N) {
        Some(p) => p,
        None => panic!("128 bits is a whole number of f32"),
    };
    assert!(plan.elems_per_thread == 4, "one F32x4 quad per thread");
    assert!(plan.align == core::mem::align_of::<cuda_device::vector::F32x4>());
    match plan.passes_for_tile(N * 4) {
        Some(passes) => assert!(passes == 1, "whole buffer in one block-wide pass"),
        None => panic!("N * 4 must be a whole number of block-wide accesses"),
    }

    // The same shape checked for global coalescing, which `plan` cannot see:
    // one 16-byte access per lane laid end to end, so a warp covers 512
    // contiguous bytes and touches the four lines that is the floor for them.
    // A strided variant with the identical plan would waste three of four.
    let mut lanes = [0usize; cuda_device::swizzle::WARP_LANES];
    let mut lane = 0;
    while lane < cuda_device::swizzle::WARP_LANES {
        lanes[lane] = lane;
        lane += 1;
    }
    let elem = core::mem::size_of::<cuda_device::vector::F32x4>();
    assert!(
        access::lines_touched(&lanes, elem) == 4,
        "a warp of F32x4 spans exactly four cache lines"
    );
    assert!(
        access::is_fully_coalesced(&lanes, elem),
        "the view-kernel access pattern must not waste bandwidth"
    );
};

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let rows = run_all(&module, &stream, cfg, N);

    // Flat-buffer path: allocate and fill as plain `f32`, take the `F32x4`
    // view inside the kernel. One quad per thread.
    let flat: Vec<f32> = (0..N * 4).map(|i| i as f32 * 0.25 + 1.0).collect();
    let flat_in = DeviceBuffer::from_host(&stream, &flat).expect("flat input");
    let mut flat_out = DeviceBuffer::<f32>::zeroed(&stream, N * 4).expect("flat output");
    let view_module = view_kernels::load(&ctx).expect("Failed to load embedded view module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { view_module.f32x4_view_copy(&stream, cfg, &flat_in, &mut flat_out) }
        .expect("f32x4_view_copy launch");
    let flat_round_trip = flat_out.to_host_vec(&stream).expect("flat readback") == flat;

    // Inspect the code we just launched: read the PTX back from the artifact
    // bundle embedded in this binary. Those are the same bytes `kernels::load`
    // handed to the driver, so the shape assertion can never diverge from what
    // actually ran (a loose `vectorization.ptx` on disk can). A
    // `--materialize-cubin` build embeds a cubin with no PTX text; only the
    // textual shape check is skipped then, and says so.
    let ptx = embedded_ptx_text();

    let mut errors = 0;
    for r in &rows {
        if !r.copy_ok {
            errors += 1;
            println!("  !! {} round-trip copy mismatch", r.name);
        }
    }
    if !flat_round_trip {
        errors += 1;
        println!("  !! f32x4_view_copy round-trip copy mismatch");
    }

    if let Some(ptx) = &ptx {
        let document = ptx_parse::Document::parse(ptx).expect("parse embedded PTX");
        println!(
            "{:<11} {:<14} {:>4} {:>5}  {:<22} vectorized",
            "rust", "cuda", "size", "align", "ptx load"
        );
        for r in &rows {
            let body = kernel_body(&document, r.kernel);
            let load = first_mem_op(body);
            let vectorized =
                load.contains(".v2.") || load.contains(".v4.") || load.contains(".v8.");
            // The robust, alignment-gated invariant this example exists to show: a
            // type aligned to the 128-bit vector width (or wider) always fuses into
            // a vector `ld/st.global.v*`. (Some smaller types also vectorize, and
            // the 8-byte ones coalesce into a single `b64` -- reported, not asserted.)
            let expect_vec = r.align >= 16;
            if expect_vec && !vectorized {
                errors += 1;
                println!("  !! {} expected to vectorize but did not", r.name);
            }
            println!(
                "{:<11} {:<14} {:>4} {:>5}  {:<22} {}",
                r.name,
                r.cuda,
                r.size,
                r.align,
                load,
                if vectorized { "yes" } else { "no" }
            );
        }

        // The in-kernel `as_vectors::<F32x4>` view over a flat `&[f32]` parameter
        // must reach the same 128-bit transactions as the over-aligned parameter
        // types above: this is the checked-view path the `vector` module ships.
        let view_body = kernel_body(&document, "f32x4_view_copy");
        for dir in ["ld", "st"] {
            match find_128bit_global(view_body, dir) {
                Some(line) => println!("f32x4_view_copy: {line}"),
                None => {
                    errors += 1;
                    // Deliberately not "has no 128-bit vector op". When this fires,
                    // the likely state is not a scalarized transaction but a
                    // widened one that lost its state space: measured on sm_86
                    // while #657 was open, the view path did reach 128 bits, as a
                    // generic `LD.E.128` / `ST.E.128` rather than the `LDG.E.128` /
                    // `STG.E.128` the direct path gets, and paid four extra
                    // `LDL.64`/`STL.64` local accesses on the way. What was missing
                    // was the *global* window, not the width. Say that, so whoever
                    // sees this next looks at the address space first.
                    println!(
                        "  !! f32x4_view_copy: no 128-bit {dir}.global vector op \
                         (the 128-bit access is there, but in the generic window)"
                    );
                }
            }
        }
    } else {
        println!(
            "PTX shape check skipped: the embedded bundle has no PTX text to \
             inspect (a --materialize-cubin build embeds a cubin)"
        );
    }

    if errors == 0 {
        let scope = if ptx.is_some() {
            "layout, copy, and codegen all correct"
        } else {
            "layout and copy correct (codegen shape not inspected)"
        };
        println!(
            "\n\u{2713} SUCCESS: {} CUDA vector types -- {}",
            rows.len(),
            scope
        );
    } else {
        println!("\n\u{2717} FAILED: {} problem(s)", errors);
        std::process::exit(1);
    }
}

/// PTX text of this binary's embedded device bundle, if the bundle carries
/// PTX. These are the bytes `kernels::load` hands to the driver, so asserting
/// on them asserts on the code that ran. Returns `None` when the bundle holds
/// a cubin instead (a `--materialize-cubin` build), which has no PTX text.
fn embedded_ptx_text() -> Option<String> {
    use cuda_host::embedded::{ArtifactPayloadKind, artifact_bundles_from_current_exe};

    let bundles = artifact_bundles_from_current_exe()
        .expect("failed to read embedded artifact bundles from the current executable");
    let bundle = bundles
        .into_iter()
        .find(|bundle| bundle.name == env!("CARGO_PKG_NAME"))
        .expect("embedded device bundle not found in the current executable");
    let ptx = bundle.payload(ArtifactPayloadKind::Ptx)?;
    Some(
        std::str::from_utf8(ptx)
            .expect("embedded PTX payload is not valid UTF-8")
            .trim_end_matches('\0')
            .to_string(),
    )
}

/// PTX text of a `.entry <name>` kernel, header through closing brace.
fn kernel_body<'source>(document: &ptx_parse::Document<'source>, name: &str) -> &'source str {
    document
        .callables_named(name)
        .find(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
        .and_then(|callable| callable.body_text().map(|_| callable.text()))
        .unwrap_or_else(|| panic!("kernel `{name}` not found or incomplete in PTX"))
}

/// First 128-bit vector global memory op of direction `dir` (`ld`/`st`) in a
/// kernel body. llc's spelling of the 128-bit transaction varies by version
/// (`ld.global.v4.f32` on older llc, `ld.global.v2.b64` on llc-21/22); every
/// accepted form is a single 128-bit access.
fn find_128bit_global<'a>(body: &'a str, dir: &str) -> Option<&'a str> {
    let prefix = format!("{dir}.global.v");
    body.lines().map(str::trim).find(|t| {
        t.starts_with(&prefix)
            && [".v4.f32", ".v4.b32", ".v2.f64", ".v2.b64"]
                .iter()
                .any(|w| t.contains(w))
    })
}

/// First global memory op mnemonic in a kernel body (e.g. `ld.global.v2.b64`).
fn first_mem_op(body: &str) -> String {
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t
            .strip_prefix("ld.global.")
            .or_else(|| t.strip_prefix("st.global."))
        {
            let mnem: String = rest.split_whitespace().next().unwrap_or("").to_string();
            let kind = if t.starts_with("ld") {
                "ld.global."
            } else {
                "st.global."
            };
            return format!("{kind}{mnem}");
        }
    }
    "(none)".to_string()
}
