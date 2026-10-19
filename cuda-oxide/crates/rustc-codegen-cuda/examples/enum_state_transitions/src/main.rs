/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conformance coverage for `Option` state transitions and `Result` combinators.
//!
//! Runtime-selected variants keep enum construction, discriminant reads, payload
//! extraction, and payload mutation observable in both optimized and low-MIR-opt
//! device builds.
//!
//! Usage:
//!   cargo oxide run enum_state_transitions
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run enum_state_transitions

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    const SOME_TAG: u64 = 1u64 << 32;
    const ERR_TAG: u64 = 1u64 << 32;

    #[kernel]
    pub fn option_transitions(seed: u32, start_some: u32, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let mut take_state = if start_some != 0 { Some(seed) } else { None };
        let taken = take_state.take();

        let mut replace_state = if start_some != 0 {
            Some(seed + 10)
        } else {
            None
        };
        let replaced = replace_state.replace(seed + 20);

        let mut entry_state = if start_some != 0 {
            Some(seed + 40)
        } else {
            None
        };
        let entry_payload = entry_state.get_or_insert(seed + 30);
        let entry_before_mutation = *entry_payload;
        *entry_payload += 1;

        let taken_packed = match taken {
            None => 0,
            Some(value) => SOME_TAG | value as u64,
        };
        let take_state_packed = match take_state {
            None => 0,
            Some(value) => SOME_TAG | value as u64,
        };
        let replaced_packed = match replaced {
            None => 0,
            Some(value) => SOME_TAG | value as u64,
        };
        let replace_state_packed = match replace_state {
            None => 0,
            Some(value) => SOME_TAG | value as u64,
        };
        let entry_state_packed = match entry_state {
            None => 0,
            Some(value) => SOME_TAG | value as u64,
        };

        unsafe {
            // One active thread owns the entire output region.
            let ptr = out.as_mut_ptr();
            ptr.write(taken_packed);
            ptr.add(1).write(take_state_packed);
            ptr.add(2).write(replaced_packed);
            ptr.add(3).write(replace_state_packed);
            ptr.add(4).write(entry_before_mutation as u64);
            ptr.add(5).write(entry_state_packed);
        }
    }

    #[kernel]
    pub fn result_combinators(seed: u32, start_ok: u32, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let base: Result<u32, u32> = if start_ok != 0 {
            Ok(seed)
        } else {
            Err(seed + 100)
        };

        let mapped: Result<u32, u32> = base.map(|value| value + 1);
        let mapped_err: Result<u32, u32> = base.map_err(|error| error + 2);
        let chained: Result<u32, u32> = base.and_then(|value| Err::<u32, u32>(value + 3));
        let recovered: Result<u32, u32> = base.or_else(|error| Ok::<u32, u32>(error + 4));

        let base_packed = match base {
            Ok(value) => value as u64,
            Err(error) => ERR_TAG | error as u64,
        };
        let mapped_packed = match mapped {
            Ok(value) => value as u64,
            Err(error) => ERR_TAG | error as u64,
        };
        let mapped_err_packed = match mapped_err {
            Ok(value) => value as u64,
            Err(error) => ERR_TAG | error as u64,
        };
        let chained_packed = match chained {
            Ok(value) => value as u64,
            Err(error) => ERR_TAG | error as u64,
        };
        let recovered_packed = match recovered {
            Ok(value) => value as u64,
            Err(error) => ERR_TAG | error as u64,
        };

        unsafe {
            // One active thread owns the entire output region.
            let ptr = out.as_mut_ptr();
            ptr.write(base_packed);
            ptr.add(1).write(mapped_packed);
            ptr.add(2).write(mapped_err_packed);
            ptr.add(3).write(chained_packed);
            ptr.add(4).write(recovered_packed);
        }
    }
}

const TAG: u64 = 1u64 << 32;

fn some(value: u32) -> u64 {
    TAG | value as u64
}

fn err(value: u32) -> u64 {
    TAG | value as u64
}

fn main() {
    println!("=== enum_state_transitions ===");

    const SEED: u32 = 10;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    let mut option_some_out =
        DeviceBuffer::<u64>::zeroed(&stream, 6).expect("Option Some output allocation");
    unsafe { module.option_transitions(&stream, cfg, SEED, 1, &mut option_some_out) }
        .expect("option_transitions Some launch");
    let option_some = option_some_out
        .to_host_vec(&stream)
        .expect("copy Option Some results");

    assert_eq!(
        option_some[0],
        some(SEED),
        "Option::take returned Some payload"
    );
    assert_eq!(option_some[1], 0, "Option::take left None");
    assert_eq!(
        option_some[2],
        some(SEED + 10),
        "Option::replace returned old Some payload"
    );
    assert_eq!(
        option_some[3],
        some(SEED + 20),
        "Option::replace stored replacement payload"
    );
    assert_eq!(
        option_some[4],
        (SEED + 40) as u64,
        "Option::get_or_insert preserved existing payload"
    );
    assert_eq!(
        option_some[5],
        some(SEED + 41),
        "Option::get_or_insert returned mutable existing payload"
    );

    let mut option_none_out =
        DeviceBuffer::<u64>::zeroed(&stream, 6).expect("Option None output allocation");
    unsafe { module.option_transitions(&stream, cfg, SEED, 0, &mut option_none_out) }
        .expect("option_transitions None launch");
    let option_none = option_none_out
        .to_host_vec(&stream)
        .expect("copy Option None results");

    assert_eq!(option_none[0], 0, "Option::take returned None");
    assert_eq!(option_none[1], 0, "Option::take kept None");
    assert_eq!(option_none[2], 0, "Option::replace returned old None");
    assert_eq!(
        option_none[3],
        some(SEED + 20),
        "Option::replace constructed Some replacement"
    );
    assert_eq!(
        option_none[4],
        (SEED + 30) as u64,
        "Option::get_or_insert inserted payload"
    );
    assert_eq!(
        option_none[5],
        some(SEED + 31),
        "Option::get_or_insert returned mutable inserted payload"
    );

    println!("PASS: Option::take");
    println!("PASS: Option::replace");
    println!("PASS: Option::get_or_insert");

    let mut result_ok_out =
        DeviceBuffer::<u64>::zeroed(&stream, 5).expect("Result Ok output allocation");
    unsafe { module.result_combinators(&stream, cfg, SEED, 1, &mut result_ok_out) }
        .expect("result_combinators Ok launch");
    let result_ok = result_ok_out
        .to_host_vec(&stream)
        .expect("copy Result Ok results");

    assert_eq!(result_ok[0], SEED as u64, "Result base Ok");
    assert_eq!(
        result_ok[1],
        (SEED + 1) as u64,
        "Result::map transformed Ok"
    );
    assert_eq!(result_ok[2], SEED as u64, "Result::map_err preserved Ok");
    assert_eq!(
        result_ok[3],
        err(SEED + 3),
        "Result::and_then transitioned Ok to Err"
    );
    assert_eq!(result_ok[4], SEED as u64, "Result::or_else preserved Ok");

    let mut result_err_out =
        DeviceBuffer::<u64>::zeroed(&stream, 5).expect("Result Err output allocation");
    unsafe { module.result_combinators(&stream, cfg, SEED, 0, &mut result_err_out) }
        .expect("result_combinators Err launch");
    let result_err = result_err_out
        .to_host_vec(&stream)
        .expect("copy Result Err results");

    assert_eq!(result_err[0], err(SEED + 100), "Result base Err");
    assert_eq!(result_err[1], err(SEED + 100), "Result::map preserved Err");
    assert_eq!(
        result_err[2],
        err(SEED + 102),
        "Result::map_err transformed Err"
    );
    assert_eq!(
        result_err[3],
        err(SEED + 100),
        "Result::and_then preserved Err"
    );
    assert_eq!(
        result_err[4],
        (SEED + 104) as u64,
        "Result::or_else transitioned Err to Ok"
    );

    println!("PASS: Result::map");
    println!("PASS: Result::map_err");
    println!("PASS: Result::and_then");
    println!("PASS: Result::or_else");

    println!("PASS: enum_state_transitions");
}
