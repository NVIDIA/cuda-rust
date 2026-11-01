/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Host data packing, CPU verification, and CUDA-event timing for instruction-level MXFP4 GEMM.

#![feature(f16)]
mod kernel;
mod physical;
use ctm_rs::host::tma::{EncodeOptions, L2Promotion, create_tensor_map_tiled};
use ctm_rs::{
    memory::{F16Tile128, Mxf4ScaleTile, Mxf4Tile128},
    tma::TensorMap,
};
use cuda_core::{CudaContext, LaunchConfig1D};
use kernel::{DYNAMIC_SMEM_BYTES, K, K_STAGES, KB, M, N, THREADS, TOTAL_TILES, kernels};

/// The eight non-negative e2m1 values, indexed by nibble 0x0..=0x7;
/// nibbles 0x8..=0xF are their negations (0x8 is -0.0).
const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode the sign and three-bit magnitude of one FP4 nibble for the CPU oracle.
fn e2m1_decode(nibble: u8) -> f32 {
    let magnitude = E2M1_VALUES[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Small deterministic hash used only to build the host fixture.
///
/// Mixing both coordinates prevents tile-sized row/column permutations from
/// hiding behind a short periodic input pattern.
fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Deterministic data covering all 16 nibbles (zeros, negatives, both -0).
fn a_nibble(m: usize, k: usize) -> u8 {
    (mix32((m as u32).wrapping_mul(0x9e37_79b9) ^ k as u32 ^ 0xa511_e9b3) & 0xf) as u8
}

/// Generate a deterministic B value from its K/N coordinates, including signed zeros.
fn b_nibble(k: usize, n: usize) -> u8 {
    (mix32((n as u32).wrapping_mul(0x85eb_ca6b) ^ k as u32 ^ 0x63d8_3595) & 0xf) as u8
}

/// ue8m0 scale bytes: 2^(byte-127). 126..=128 keeps every product a small
/// binary fraction that `f32` represents exactly.
fn sfa_byte(m: usize, kb: usize) -> u8 {
    126 + (mix32((m as u32).wrapping_mul(0xc2b2_ae35) ^ kb as u32) % 3) as u8
}

/// Generate a deterministic B scale exponent in 126..=128 from its row and K-group indices.
fn sfb_byte(n: usize, kb: usize) -> u8 {
    126 + (mix32((n as u32).wrapping_mul(0x27d4_eb2f) ^ kb as u32 ^ 0x1656_67b1) % 3) as u8
}

/// Decode a finite UE8M0 exponent byte as 2 raised to byte minus 127.
fn ue8m0_decode(byte: u8) -> f32 {
    2f32.powi(byte as i32 - 127)
}

/// Pack four increasing-K FP4 nibbles into each 16-bit transport carrier.
fn pack_mxf4_view(nibbles: &[u8]) -> Vec<f16> {
    nibbles
        .as_chunks::<4>()
        .0
        .iter()
        .map(|q| {
            f16::from_bits(
                (q[0] as u16 & 0xf)
                    | ((q[1] as u16 & 0xf) << 4)
                    | ((q[2] as u16 & 0xf) << 8)
                    | ((q[3] as u16 & 0xf) << 12),
            )
        })
        .collect()
}

/// Reorder dense `[row][K/32]` scales into SM120's 128-row x four-group
/// atoms, then expose the exact bytes as the u16 carrier used by TMA.
fn pack_canonical_scales(dense: &[u8], rows: usize) -> Vec<u16> {
    let mut packed = vec![0u8; rows * KB];
    for row in 0..rows {
        for group in 0..KB {
            packed[physical::scale_offset(row, group, KB)] = dense[row * KB + group];
        }
    }
    packed
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// Select a rounded percentile index from nonempty sorted timings using an integer fraction.
fn percentile(sorted: &[f32], numerator: usize, denominator: usize) -> f32 {
    assert!(!sorted.is_empty());
    assert!(denominator > 0 && numerator <= denominator);
    let scaled = (sorted.len() - 1)
        .checked_mul(numerator)
        .expect("percentile index overflow");
    sorted[(scaled + denominator / 2) / denominator]
}

/// Parse launch/timing options, prepare packed tensors, verify GPU results, and write timing reports.
fn main() {
    let mut ctas = None;
    let mut warmup = 100usize;
    let mut samples = 31usize;
    let mut json_path = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--ctas" => {
                ctas = Some(
                    arguments
                        .next()
                        .expect("--ctas value")
                        .parse::<u32>()
                        .expect("positive CTA count"),
                )
            }
            "--warmup" => {
                warmup = arguments
                    .next()
                    .expect("--warmup value")
                    .parse()
                    .expect("warmup count")
            }
            "--samples" => {
                samples = arguments
                    .next()
                    .expect("--samples value")
                    .parse()
                    .expect("sample count")
            }
            "--json" => json_path = Some(arguments.next().expect("--json path")),
            _ => panic!("unknown option {argument}; use --ctas, --warmup, --samples, --json"),
        }
    }
    assert!(samples > 0, "at least one timing sample is required");
    if let Some(ctas) = ctas {
        assert!(
            ctas > 0 && ctas <= TOTAL_TILES as u32,
            "CTA count out of range"
        );
    }
    if let Some(path) = &json_path {
        match std::fs::remove_file(path) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => panic!("cannot remove stale report: {error}"),
        }
    }
    println!("=== Block-scaled MXFP4 GEMM with Rust CTM ===");
    println!("C({M}x{N}) = A({M}x{K} e2m1) x B({K}x{N} e2m1), ue8m0 scales per 32-element block\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // Accelerated MXFP4 block-scaled MMA exists on SM120/121 targets.
    let (major, minor) = ctx.compute_capability().expect("query compute capability");
    println!("GPU Compute Capability: sm_{major}{minor}");
    if (major, minor) != (12, 0) {
        eprintln!("mxf4 primitives require an SM120 GPU");
        std::process::exit(1);
    }

    // Host-side packed FP4 payloads; B is stored transposed as N x K.
    let a_nibbles: Vec<u8> = (0..M * K).map(|i| a_nibble(i / K, i % K)).collect();
    let b_nibbles: Vec<u8> = (0..N * K).map(|i| b_nibble(i % K, i / K)).collect();
    let sfa: Vec<u8> = (0..M * KB).map(|i| sfa_byte(i / KB, i % KB)).collect();
    let sfb: Vec<u8> = (0..N * KB).map(|i| sfb_byte(i / KB, i % KB)).collect();

    // Exact f32 accumulator reference: per K-block, an exact 32-term dot times the two
    // power-of-two scales. Every partial sum is an exact binary fraction well
    // inside f32's exact range, so accumulation order cannot matter.
    println!("computing exact host reference ({M}x{N}x{K})...");
    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        let a_row = &a_nibbles[m * K..(m + 1) * K];
        let sfa_row = &sfa[m * KB..(m + 1) * KB];
        for n in 0..N {
            let b_row = &b_nibbles[n * K..(n + 1) * K];
            let sfb_row = &sfb[n * KB..(n + 1) * KB];
            let mut total = 0.0f32;
            for kb in 0..KB {
                let mut block = 0.0f32;
                for j in 0..32 {
                    let k = kb * 32 + j;
                    block += e2m1_decode(a_row[k]) * e2m1_decode(b_row[k]);
                }
                total += ue8m0_decode(sfa_row[kb]) * ue8m0_decode(sfb_row[kb]) * block;
            }
            expected[m * N + n] = total;
        }
    }
    let expected_f16: Vec<f16> = expected.iter().copied().map(|value| value as f16).collect();

    let stream = ctx.default_stream();
    // SAFETY: this executable is the sole owner of the embedded device module.
    let module = unsafe { kernels::load(&ctx) }.expect("Failed to load embedded CUDA module");

    let a_packed = pack_mxf4_view(&a_nibbles);
    let b_packed = pack_mxf4_view(&b_nibbles);
    let sfa_packed = pack_canonical_scales(&sfa, M);
    let sfb_packed = pack_canonical_scales(&sfb, N);
    let a_dev = cuda_core::DeviceBuffer::from_host(&stream, &a_packed).unwrap();
    let b_dev = cuda_core::DeviceBuffer::from_host(&stream, &b_packed).unwrap();
    let sfa_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfa_packed).unwrap();
    let sfb_dev = cuda_core::DeviceBuffer::from_host(&stream, &sfb_packed).unwrap();

    // The descriptor types are the host/kernel agreement: A/B land in the
    // B64 byte layout, while each flattened scale row is one canonical atom.
    let a_desc: TensorMap<u8, Mxf4Tile128> = create_tensor_map_tiled(
        a_dev.cu_deviceptr() as *mut core::ffi::c_void,
        M as u64,
        (K / 2) as u64,
        (K / 2) as u64,
        EncodeOptions {
            l2_promotion: L2Promotion::B128,
        },
    )
    .expect("A TMA descriptor encoding failed");
    let b_desc: TensorMap<u8, Mxf4Tile128> = create_tensor_map_tiled(
        b_dev.cu_deviceptr() as *mut core::ffi::c_void,
        N as u64,
        (K / 2) as u64,
        (K / 2) as u64,
        EncodeOptions {
            l2_promotion: L2Promotion::B128,
        },
    )
    .expect("B TMA descriptor encoding failed");
    let sfa_atom_rows = (M / 128) * K_STAGES;
    let sfb_atom_rows = (N / 128) * K_STAGES;
    let sfa_desc: TensorMap<u16, Mxf4ScaleTile> = create_tensor_map_tiled(
        sfa_dev.cu_deviceptr() as *mut core::ffi::c_void,
        sfa_atom_rows as u64,
        256,
        256,
        EncodeOptions {
            l2_promotion: L2Promotion::B128,
        },
    )
    .expect("SFA TMA descriptor encoding failed");
    let sfb_desc: TensorMap<u16, Mxf4ScaleTile> = create_tensor_map_tiled(
        sfb_dev.cu_deviceptr() as *mut core::ffi::c_void,
        sfb_atom_rows as u64,
        256,
        256,
        EncodeOptions {
            l2_promotion: L2Promotion::B128,
        },
    )
    .expect("SFB TMA descriptor encoding failed");
    let poison = vec![f16::NAN; M * N];
    let out = cuda_core::DeviceBuffer::from_host(&stream, &poison).unwrap();
    let out_desc: TensorMap<f16, F16Tile128> = create_tensor_map_tiled(
        out.cu_deviceptr() as *mut core::ffi::c_void,
        M as u64,
        N as u64,
        N as u64,
        EncodeOptions {
            l2_promotion: L2Promotion::B128,
        },
    )
    .expect("C TMA descriptor encoding failed");
    // Ask the driver how many copies of this CTA fit on each SM. The
    // persistent scheduler needs that number to choose its physical grid.
    let probe = module
        .prepare_blockscale_gemm(LaunchConfig1D::new(
            1,
            THREADS as u32,
            DYNAMIC_SMEM_BYTES as u32,
        ))
        .expect("launch contract validation failed");
    let active_ctas = probe
        .function()
        .max_active_blocks_per_multiprocessor(THREADS as u32, DYNAMIC_SMEM_BYTES as u32)
        .expect("query active CTAs per SM");
    let sm_count = ctx.multiprocessor_count().expect("query SM count");
    let persistent_ctas = sm_count
        .checked_mul(active_ctas)
        .expect("resident grid overflow")
        .min(TOTAL_TILES as u32)
        .max(1);
    drop(probe);
    let persistent_ctas = ctas.unwrap_or(persistent_ctas);
    println!("Persistent grid: {persistent_ctas} CTAs");

    let config = LaunchConfig1D::new(persistent_ctas, THREADS as u32, DYNAMIC_SMEM_BYTES as u32);
    let launch = module
        .prepare_blockscale_gemm(config)
        .expect("persistent launch contract validation failed");
    // SAFETY: 288-thread blocks contain eight compute warps and one producer
    // warp. The grid-stride scheduler partitions all 256 output tiles into
    // disjoint CTA streams, and the warp/lane mapping partitions each tile.
    // SAFETY: all descriptor allocations and tensor buffers remain live through stream completion.
    unsafe {
        module.blockscale_gemm(
            &stream, &launch, a_desc, b_desc, sfa_desc, sfb_desc, out_desc,
        )
    }
    .expect("kernel launch failed");
    let got = out.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    let mut nonzero = 0;
    for m in 0..M {
        for n in 0..N {
            if expected[m * N + n] != 0.0 {
                nonzero += 1;
            }
            if got[m * N + n].to_bits() != expected_f16[m * N + n].to_bits() {
                if errors < 5 {
                    eprintln!(
                        "  C({m},{n}): expected {} (0x{:04x}), got {} (0x{:04x})",
                        expected_f16[m * N + n],
                        expected_f16[m * N + n].to_bits(),
                        got[m * N + n],
                        got[m * N + n].to_bits(),
                    );
                }
                errors += 1;
            }
        }
    }
    if errors != 0 {
        println!("\n✗ FAILED: {errors} mismatches");
        std::process::exit(1);
    }
    println!(
        "Numeric check: all {} FP16 output bit patterns exact ({nonzero} nonzero FP32 accumulators), {} K-blocks",
        M * N,
        KB,
    );

    // Single-kernel device timing: every sample places exactly one launch
    // between CUDA events. The throughput count is 2*M*N*K FLOPs per launch.

    for _ in 0..warmup {
        // SAFETY: all descriptor allocations and tensor buffers remain live through stream completion.
        unsafe {
            module.blockscale_gemm(
                &stream, &launch, a_desc, b_desc, sfa_desc, sfb_desc, out_desc,
            )
        }
        .expect("warmup launch failed");
    }
    stream.synchronize().expect("warmup sync failed");

    let mut times_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = ctx
            .new_event_with_flags(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("create start event");
        let end = ctx
            .new_event_with_flags(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("create end event");

        start.record(&stream).expect("record start event");
        // SAFETY: all descriptor allocations and tensor buffers remain live through stream completion.
        unsafe {
            module.blockscale_gemm(
                &stream, &launch, a_desc, b_desc, sfa_desc, sfb_desc, out_desc,
            )
        }
        .expect("timed launch failed");
        end.record(&stream).expect("record end event");
        times_ms.push(start.elapsed_ms(&end).expect("event timing"));
    }
    times_ms.sort_by(f32::total_cmp);
    let median_ms = percentile(&times_ms, 1, 2);
    let p10_ms = percentile(&times_ms, 1, 10);
    let p90_ms = percentile(&times_ms, 9, 10);
    let tflops = (2.0 * (M * N * K) as f64) / (f64::from(median_ms) * 1e-3) / 1e12;
    println!(
        "Single-kernel device time: median={median_ms:.6} ms, p10={p10_ms:.6} ms, \
         p90={p90_ms:.6} ms ({samples} samples, {warmup} warmups); {tflops:.2} TFLOP/s"
    );

    let replay = out
        .to_host_vec(&stream)
        .expect("copy repeated-launch output");
    assert!(
        replay
            .iter()
            .zip(&expected_f16)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "repeated launches changed numerical results"
    );
    if let Some(path) = json_path {
        let report = serde_json::json!({"verification":"bit-exact-pass", "mnk":[M,N,K],
            "compute_capability":[major,minor], "ctas":persistent_ctas,
            "warmup":warmup,"samples":samples,"median_ms":median_ms,
            "p10_ms":p10_ms,"p90_ms":p90_ms,"tflops":tflops,
            "checked_outputs":M*N,"replay_verified":true});
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
    }
    println!(
        "\n✓ SUCCESS: block-scaled MXFP4 GEMM is bit-exact through Rust instruction primitives"
    );
}
