/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static shared-memory access through `llvm.addressof` (guards issue #54).
//!
//! The kernel does `OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight` on a static
//! `SharedArray<f32, 1>`. Before the fix in PR #55, the llvm-export textual exporter
//! gave the `addressof @__shared_mem_N` result a `%vN` SSA name even though
//! `addressof` is virtual in textual LLVM IR (it has no instruction form,
//! only a symbol reference at use sites). When the use printed before the
//! addressof's block, the GEP referenced a `%vN` no instruction defined and
//! libNVVM rejected the IR.
//!
//! The same kernel verifies that exposing the address of the first static
//! shared allocation does not turn its valid shared-space offset zero into a
//! null Rust address. Named-space pointers must become CUDA generic pointers
//! before pointer-to-integer conversion, and the exposed integer must cast
//! back into the shared space through the generic space, so the
//! expose/recover round trip stores through the recovered pointer.
//!
//! It also constructs and matches enum payloads containing shared pointers:
//! one nested through two ordinary Rust structs, and one `Option` containing a
//! bounded array of two shared references. Their physical storage must
//! recursively replace semantic shared-pointer leaves with target-stable
//! generic pointers, then restore address space 3 when extracting the payload.
//!
//! The kernel also round-trips shared pointers through non-inlined device
//! functions as struct, tuple, array, and enum arguments and return values,
//! then dereferences each returned pointer as shared memory. This keeps the
//! aggregate call/return ABI boundary observable instead of letting inlining
//! erase it.
//!
//! Finally the kernel validates `SharedArray::as_raw_mut_ptr`: 32 threads
//! derive pointers from one raw shared-array address, write disjoint elements,
//! synchronize, and let thread 0 verify the complete allocation.
//!
//! This example launches the kernel through `cuda_host::ltoir::load_kernel_module`,
//! which compiles the cuda-oxide-emitted NVVM IR via libNVVM and links the
//! cubin via nvJitLink. A dangling SSA reference in the `.ll` would fail at
//! libNVVM's verifier before the kernel could run, so a regression of #54
//! is now a hard runtime failure instead of a silent build artifact.
//!
//! Run: `cargo oxide run addressof_sharedarray`

#![allow(static_mut_refs)]
#![allow(clippy::assign_op_pattern)] // Expanded assignment preserves the addressof repro CFG.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, thread};
use cuda_host::{cuda_module, ltoir};

struct SharedPointerInner {
    pointer: *mut SharedArray<f32, 1>,
    cookie: usize,
}

struct SharedPointerOuter {
    inner: SharedPointerInner,
    guard: u32,
}

enum SharedPointerPayload {
    Empty,
    Pointer(SharedPointerOuter),
}

struct SharedPointerArrayWrapper {
    pointers: [&'static SharedArray<f32, 1>; 2],
}

#[cuda_module]
mod kernels {
    use super::*;

    const THREADS: usize = 32;
    const ENUM_COOKIE: usize = 0xC0DE;
    const ENUM_GUARD: u32 = 0xA55A;

    #[inline(never)]
    #[device]
    fn shared_pointer_enum_address(use_pointer: bool, pointer: *mut SharedArray<f32, 1>) -> usize {
        let payload = if use_pointer {
            SharedPointerPayload::Pointer(SharedPointerOuter {
                inner: SharedPointerInner {
                    pointer,
                    cookie: ENUM_COOKIE,
                },
                guard: ENUM_GUARD,
            })
        } else {
            SharedPointerPayload::Empty
        };

        match payload {
            SharedPointerPayload::Empty => 0,
            SharedPointerPayload::Pointer(extracted)
                if extracted.inner.cookie == ENUM_COOKIE && extracted.guard == ENUM_GUARD =>
            {
                extracted.inner.pointer.addr()
            }
            SharedPointerPayload::Pointer(_) => 0,
        }
    }

    #[inline(never)]
    #[device]
    fn shared_pointer_array_enum_matches(
        use_pointer: bool,
        pointer: *mut SharedArray<f32, 1>,
    ) -> usize {
        let shared: &'static SharedArray<f32, 1> = unsafe { &*pointer };
        let payload = if use_pointer {
            Some(SharedPointerArrayWrapper {
                pointers: [shared, shared],
            })
        } else {
            None
        };

        match payload {
            Some(extracted) => {
                let expected = pointer.addr();
                let first = (extracted.pointers[0] as *const SharedArray<f32, 1>).addr();
                let second = (extracted.pointers[1] as *const SharedArray<f32, 1>).addr();
                if first == expected && second == expected {
                    1
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    #[inline(never)]
    #[device]
    fn bounce_struct(value: SharedPointerOuter) -> SharedPointerOuter {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_tuple(value: (*mut SharedArray<f32, 1>, usize)) -> (*mut SharedArray<f32, 1>, usize) {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_array(value: [*mut SharedArray<f32, 1>; 2]) -> [*mut SharedArray<f32, 1>; 2] {
        value
    }

    #[inline(never)]
    #[device]
    fn bounce_enum(value: SharedPointerPayload) -> SharedPointerPayload {
        value
    }

    #[kernel]
    pub fn sharedarray_late_use(seed: f32, mut out: DisjointSlice<f32>) {
        static mut OUTPUT_NORM: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut SCRATCH: SharedArray<u32, THREADS> = SharedArray::UNINIT;

        if thread::index_1d().get() == 0 {
            unsafe {
                OUTPUT_NORM[0] = seed;
                let weight = repro_weight();
                // Issue #54 repro shape: load addressof[0], multiply, store.
                OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight;
                *out.get_unchecked_mut(0) = OUTPUT_NORM[0];

                // The first static shared allocation has local shared offset
                // zero, but its CUDA generic address must not be null.
                // Exercise all three compiler-recognized public pointer
                // boundaries: shared/unique references become const/mut raw
                // pointers, while the raw receiver remains RawMut.
                let borrowed_const = OUTPUT_NORM.as_ptr();
                let borrowed_mut = OUTPUT_NORM.as_mut_ptr();
                let raw = &raw mut OUTPUT_NORM;
                let raw_address = raw.addr();
                *out.get_unchecked_mut(1) = if raw.is_null()
                    || borrowed_const.is_null()
                    || borrowed_mut.is_null()
                    || raw_address == 0
                    || borrowed_const.addr() != raw_address
                    || borrowed_mut.addr() != raw_address
                {
                    0.0
                } else {
                    1.0
                };

                // The enum payload nests the shared pointer through two
                // structs. Lowering recursively rebuilds the payload with a
                // generic physical pointer, then restores shared space during
                // extraction. The runtime condition keeps construction,
                // discriminant inspection, and extraction observable.
                let enum_address = shared_pointer_enum_address(seed != 0.0, raw);
                *out.get_unchecked_mut(2) = if enum_address == raw_address {
                    1.0
                } else {
                    0.0
                };

                // A bounded array of shared references inside an Option uses
                // the same generic physical representation recursively. Both
                // array elements must survive construction and extraction.
                *out.get_unchecked_mut(5) =
                    shared_pointer_array_enum_matches(seed != 0.0, raw) as f32;

                // The exposed integer is a generic address, so casting it
                // back into the shared space must recover the original
                // pointer (inttoptr to generic, then cvta back to shared).
                // Write through the recovered pointer and observe the store
                // through the original allocation.
                // The recovered pointer must equal the original as a value,
                // not merely reach the same memory: hardware masks the
                // shared-window base out of st.shared addresses, so a wrong
                // pointer value can still store to the right slot.
                let round_trip = recover_shared_pointer(raw_address);
                (&mut (*round_trip))[0] = OUTPUT_NORM[0] + 1.0;
                *out.get_unchecked_mut(3) = if core::ptr::eq(round_trip, raw)
                    && OUTPUT_NORM[0] == seed * repro_weight() + 1.0
                {
                    1.0
                } else {
                    0.0
                };

                // Round-trip address-space-3 pointers through ordinary device-call
                // ABI boundaries, then dereference the returned pointer as shared
                // memory. #[inline(never)] on the helpers keeps the call/return
                // boundary observable to lowering and code generation.
                let struct_round_trip = bounce_struct(SharedPointerOuter {
                    inner: SharedPointerInner {
                        pointer: raw,
                        cookie: ENUM_COOKIE,
                    },
                    guard: ENUM_GUARD,
                });
                (&mut (*struct_round_trip.inner.pointer))[0] = 31.0;
                *out.get_unchecked_mut(6) = if core::ptr::eq(struct_round_trip.inner.pointer, raw)
                    && struct_round_trip.inner.cookie == ENUM_COOKIE
                    && struct_round_trip.guard == ENUM_GUARD
                    && OUTPUT_NORM[0] == 31.0
                {
                    1.0
                } else {
                    0.0
                };

                let tuple_round_trip = bounce_tuple((raw, ENUM_COOKIE));
                (&mut (*tuple_round_trip.0))[0] = 32.0;
                *out.get_unchecked_mut(7) = if core::ptr::eq(tuple_round_trip.0, raw)
                    && tuple_round_trip.1 == ENUM_COOKIE
                    && OUTPUT_NORM[0] == 32.0
                {
                    1.0
                } else {
                    0.0
                };

                let array_round_trip = bounce_array([raw, raw]);
                (&mut (*array_round_trip[1]))[0] = 33.0;
                *out.get_unchecked_mut(8) = if core::ptr::eq(array_round_trip[0], raw)
                    && core::ptr::eq(array_round_trip[1], raw)
                    && OUTPUT_NORM[0] == 33.0
                {
                    1.0
                } else {
                    0.0
                };

                let enum_round_trip =
                    bounce_enum(SharedPointerPayload::Pointer(SharedPointerOuter {
                        inner: SharedPointerInner {
                            pointer: raw,
                            cookie: ENUM_COOKIE,
                        },
                        guard: ENUM_GUARD,
                    }));
                *out.get_unchecked_mut(9) = match enum_round_trip {
                    SharedPointerPayload::Pointer(extracted)
                        if extracted.inner.cookie == ENUM_COOKIE
                            && extracted.guard == ENUM_GUARD =>
                    {
                        (&mut (*extracted.inner.pointer))[0] = 34.0;
                        if core::ptr::eq(extracted.inner.pointer, raw) && OUTPUT_NORM[0] == 34.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                };
            }
        }

        // Every thread starts from the raw address of one shared allocation,
        // then writes only its own element. No thread constructs an
        // `&mut SharedArray` spanning elements owned by other threads.
        let lane = thread::threadIdx_x() as usize;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        if lane < THREADS {
            unsafe { scratch.add(lane).write((lane + 1) as u32) };
        }
        thread::sync_threads();

        if lane == 0 {
            let mut sum = 0_u32;
            for index in 0..THREADS {
                sum += unsafe { scratch.add(index).read() };
            }
            unsafe { *out.get_unchecked_mut(4) = sum as f32 };
        }
    }

    #[inline(never)]
    #[device]
    fn repro_weight() -> f32 {
        3.0
    }

    /// Recover a shared pointer from its exposed integer address.
    ///
    /// `#[inline(never)]` keeps the `inttoptr` in a function that cannot
    /// see the matching `ptrtoint`, so LLVM's InferAddressSpaces cannot
    /// fold the pair away and the lowering's own re-entry rule is what
    /// executes.
    #[inline(never)]
    #[device]
    fn recover_shared_pointer(address: usize) -> *mut SharedArray<f32, 1> {
        address as *mut SharedArray<f32, 1>
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== addressof_sharedarray (issue #54 regression) ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Forces the cuda-oxide-emitted `.ll` through libNVVM + nvJitLink.
    // A dangling SSA reference in the IR would fail libNVVM's verifier here.
    let raw_module = ltoir::load_kernel_module(&ctx, "addressof_sharedarray")?;
    let module = kernels::from_module(raw_module).expect("typed module init failed");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 10)?;
    let seed: f32 = 7.0;

    // SAFETY: one 32-thread block writes 32 distinct shared elements, then
    // thread 0 reads them after a block-wide barrier. Only thread 0 writes the
    // ten output elements.
    unsafe { module.sharedarray_late_use(stream.as_ref(), cfg, seed, &mut out) }?;

    let result = out.to_host_vec(&stream)?;
    let expected: f32 = 21.0; // seed * repro_weight() == 7.0 * 3.0

    if (result[0] - expected).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: got {}, expected {expected}",
            result[0]
        );
        std::process::exit(1);
    }
    if (result[1] - 1.0).abs() >= f32::EPSILON {
        eprintln!("FAIL addressof_sharedarray: shared offset zero exposed as null");
        std::process::exit(1);
    }
    if (result[2] - 1.0).abs() >= f32::EPSILON {
        eprintln!("FAIL addressof_sharedarray: nested shared pointer enum did not round-trip");
        std::process::exit(1);
    }
    if (result[3] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: integer address did not cast back to the shared pointer"
        );
        std::process::exit(1);
    }
    if (result[5] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: bounded shared-pointer array enum did not round-trip"
        );
        std::process::exit(1);
    }

    if (result[6] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: shared pointer struct did not survive device call + return"
        );
        std::process::exit(1);
    }
    if (result[7] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: shared pointer tuple did not survive device call + return"
        );
        std::process::exit(1);
    }
    if (result[8] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: shared pointer array did not survive device call + return"
        );
        std::process::exit(1);
    }
    if (result[9] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: shared pointer enum did not survive device call + return"
        );
        std::process::exit(1);
    }
    println!(
        "PASS addressof_sharedarray: seed={seed}, result={}, shared address is non-null, enum storage and struct/tuple/array/enum device-call round trips preserved shared pointers",
        result[0]
    );

    let raw_expected = (1_u32..=32).sum::<u32>() as f32;
    if (result[4] - raw_expected).abs() > f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray raw receiver: got {}, expected {raw_expected}",
            result[4]
        );
        std::process::exit(1);
    }
    println!(
        "PASS addressof_sharedarray raw receiver: result={}",
        result[4]
    );
    Ok(())
}
