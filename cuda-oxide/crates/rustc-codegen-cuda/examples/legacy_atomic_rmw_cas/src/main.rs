// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{
    DisjointSlice,
    atomic::{
        AtomicOrdering, BlockAtomicU32, BlockAtomicU64, DeviceAtomicU32, DeviceAtomicU64,
        SystemAtomicU32, SystemAtomicU64,
    },
    kernel, thread,
};
use cuda_host::cuda_module;

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn integer_rmw(
        counter_u32: &[DeviceAtomicU32],
        counter_u64: &[DeviceAtomicU64],
        mut old_values: DisjointSlice<(u32, u64)>,
    ) {
        let index = thread::index_1d();
        if index.get() >= N {
            return;
        }

        // AcqRel deliberately exercises the fence-splitting path: the LLVM
        // atomicrmw that reaches legacy NVVM is monotonic, while the fences
        // preserve the source ordering.
        let old_u32 = counter_u32[0].fetch_add(1, AtomicOrdering::AcqRel);
        let old_u64 = counter_u64[0].fetch_add(1, AtomicOrdering::AcqRel);
        if let Some(slot) = old_values.get_mut(index) {
            *slot = (old_u32, old_u64);
        }
    }

    #[allow(clippy::manual_unwrap_or)]
    #[kernel]
    pub fn integer_cas(
        counter_u32: &[DeviceAtomicU32],
        counter_u64: &[DeviceAtomicU64],
        mut observed_u32: DisjointSlice<(u32, u32)>,
        mut observed_u64: DisjointSlice<(u64, u64)>,
    ) {
        let index = thread::index_1d();
        if index.get() != 0 {
            return;
        }

        // Relaxed on both sides keeps this CAS on the native legacy lane:
        // libNVVM lowers ordered cmpxchg to a bare, unordered atom.cas, so
        // any ordered success or failure ordering routes to the scoped
        // inline-PTX rewrite instead.
        let success_u32 = match counter_u32[0].compare_exchange(
            7,
            11,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let failure_u32 = match counter_u32[0].compare_exchange(
            7,
            13,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        let success_u64 = match counter_u64[0].compare_exchange(
            7,
            11,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(old) => old,
            Err(_) => u64::MAX,
        };
        let failure_u64 = match counter_u64[0].compare_exchange(
            7,
            13,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => u64::MAX,
            Err(old) => old,
        };

        if let Some(slot) = observed_u32.get_mut(thread::index_1d()) {
            *slot = (success_u32, failure_u32);
        }
        if let Some(slot) = observed_u64.get_mut(thread::index_1d()) {
            *slot = (success_u64, failure_u64);
        }
    }

    #[kernel]
    pub fn scoped_integer_rmw(
        block_u32: &[BlockAtomicU32],
        block_u64: &[BlockAtomicU64],
        system_u32: &[SystemAtomicU32],
        system_u64: &[SystemAtomicU64],
        mut block_old_values: DisjointSlice<(u32, u64)>,
        mut system_old_values: DisjointSlice<(u32, u64)>,
    ) {
        let index = thread::index_1d();
        if index.get() >= N {
            return;
        }

        // Block/System scopes cannot be represented faithfully by the legacy
        // LLVM atomic form used by libNVVM. The legacy legalization pass
        // rewrites these monotonic RMWs to scoped inline PTX, while the
        // existing fence-splitting path preserves the AcqRel source ordering.
        let block_old_u32 = block_u32[0].fetch_add(1, AtomicOrdering::AcqRel);
        let block_old_u64 = block_u64[0].fetch_add(1, AtomicOrdering::AcqRel);
        let system_old_u32 = system_u32[0].fetch_add(1, AtomicOrdering::AcqRel);
        let system_old_u64 = system_u64[0].fetch_add(1, AtomicOrdering::AcqRel);

        if let Some(slot) = block_old_values.get_mut(thread::index_1d()) {
            *slot = (block_old_u32, block_old_u64);
        }
        if let Some(slot) = system_old_values.get_mut(thread::index_1d()) {
            *slot = (system_old_u32, system_old_u64);
        }
    }

    #[allow(clippy::manual_unwrap_or, clippy::too_many_arguments)]
    #[kernel]
    pub fn scoped_integer_cas(
        block_counter: &[BlockAtomicU32],
        system_counter: &[SystemAtomicU32],
        device_counter: &[DeviceAtomicU32],
        device_ordered_counter: &[DeviceAtomicU32],
        mut block_observed: DisjointSlice<(u32, u32)>,
        mut system_observed: DisjointSlice<(u32, u32)>,
        mut device_observed: DisjointSlice<(u32, u32)>,
        mut device_ordered_observed: DisjointSlice<(u32, u32)>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let block_success = match block_counter[0].compare_exchange(
            7,
            11,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let block_failure = match block_counter[0].compare_exchange(
            7,
            13,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        let system_success = match system_counter[0].compare_exchange(
            7,
            11,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let system_failure = match system_counter[0].compare_exchange(
            7,
            13,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        // Device scope with monotonic orderings stays a native LLVM cmpxchg.
        // Stronger failure ordering forces the scoped PTX rewrite because
        // legacy libNVVM accepts but ignores LLVM's failure-ordering field.
        let device_success = match device_counter[0].compare_exchange(
            7,
            11,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let device_failure = match device_counter[0].compare_exchange(
            7,
            13,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        // Ordered success with relaxed failure also routes to the scoped PTX
        // rewrite: libNVVM would lower the ordered cmpxchg to a bare,
        // unordered atom.cas, so the legalizer emits atom.acq_rel.gpu.cas
        // instead of keeping the native form.
        let device_ordered_success = match device_ordered_counter[0].compare_exchange(
            7,
            11,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let device_ordered_failure = match device_ordered_counter[0].compare_exchange(
            7,
            13,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        if let Some(slot) = block_observed.get_mut(thread::index_1d()) {
            *slot = (block_success, block_failure);
        }
        if let Some(slot) = system_observed.get_mut(thread::index_1d()) {
            *slot = (system_success, system_failure);
        }
        if let Some(slot) = device_observed.get_mut(thread::index_1d()) {
            *slot = (device_success, device_failure);
        }
        if let Some(slot) = device_ordered_observed.get_mut(thread::index_1d()) {
            *slot = (device_ordered_success, device_ordered_failure);
        }
    }
}

fn old_values_are_permutations(values: &[(u32, u64)]) -> bool {
    let mut u32_values = values.iter().map(|&(value, _)| value).collect::<Vec<_>>();
    let mut u64_values = values.iter().map(|&(_, value)| value).collect::<Vec<_>>();
    u32_values.sort_unstable();
    u64_values.sort_unstable();

    u32_values
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as u32)
        && u64_values
            .iter()
            .enumerate()
            .all(|(index, &value)| value == index as u64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;

    let rmw_u32 = DeviceBuffer::<u32>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicU32>();
    let rmw_u64 = DeviceBuffer::<u64>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicU64>();
    let mut old_values = DeviceBuffer::<(u32, u64)>::zeroed(&stream, N)?;

    // SAFETY: the launch covers exactly N unique one-dimensional indices.
    // `old_values` has N elements, and both one-element counters use atomic
    // wrapper pointees so all shared updates are atomic.
    unsafe {
        module.integer_rmw(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &rmw_u32,
            &rmw_u64,
            &mut old_values,
        )?;
    }
    stream.synchronize()?;

    let got_u32 = rmw_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let got_u64 = rmw_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    let old_values = old_values.to_host_vec(&stream)?;
    if got_u32 != N as u32 || got_u64 != N as u64 || !old_values_are_permutations(&old_values) {
        return Err(format!(
            "legacy integer RMW mismatch: u32={got_u32}, u64={got_u64}, old_values_permutations={}",
            old_values_are_permutations(&old_values)
        )
        .into());
    }

    let cas_u32 = DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<DeviceAtomicU32>();
    let cas_u64 = DeviceBuffer::from_host(&stream, &[7_u64])?.cast_elem::<DeviceAtomicU64>();
    let mut observed_u32 = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;
    let mut observed_u64 = DeviceBuffer::<(u64, u64)>::zeroed(&stream, 1)?;

    // SAFETY: exactly one thread executes the two CAS probes for each
    // one-element atomic counter and writes one element in each output buffer.
    unsafe {
        module.integer_cas(
            &stream,
            LaunchConfig::for_num_elems(1),
            &cas_u32,
            &cas_u64,
            &mut observed_u32,
            &mut observed_u64,
        )?;
    }
    stream.synchronize()?;

    let observed_u32 = observed_u32.to_host_vec(&stream)?[0];
    let observed_u64 = observed_u64.to_host_vec(&stream)?[0];
    let final_u32 = cas_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let final_u64 = cas_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    if observed_u32 != (7, 11) || observed_u64 != (7, 11) || final_u32 != 11 || final_u64 != 11 {
        return Err(format!(
            "legacy integer CAS mismatch: observed_u32={observed_u32:?}, observed_u64={observed_u64:?}, final_u32={final_u32}, final_u64={final_u64}"
        )
        .into());
    }

    let block_rmw_u32 = DeviceBuffer::<u32>::zeroed(&stream, 1)?.cast_elem::<BlockAtomicU32>();
    let block_rmw_u64 = DeviceBuffer::<u64>::zeroed(&stream, 1)?.cast_elem::<BlockAtomicU64>();
    let system_rmw_u32 = DeviceBuffer::<u32>::zeroed(&stream, 1)?.cast_elem::<SystemAtomicU32>();
    let system_rmw_u64 = DeviceBuffer::<u64>::zeroed(&stream, 1)?.cast_elem::<SystemAtomicU64>();
    let mut block_old_values = DeviceBuffer::<(u32, u64)>::zeroed(&stream, N)?;
    let mut system_old_values = DeviceBuffer::<(u32, u64)>::zeroed(&stream, N)?;
    let scoped_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: exactly one CTA participates, satisfying BlockAtomic's scope
    // contract. Every thread owns one output slot and all counter updates are
    // atomic. System-scoped counters are only touched by the device during the
    // launch; host inspection happens after stream synchronization.
    unsafe {
        module.scoped_integer_rmw(
            &stream,
            scoped_cfg,
            &block_rmw_u32,
            &block_rmw_u64,
            &system_rmw_u32,
            &system_rmw_u64,
            &mut block_old_values,
            &mut system_old_values,
        )?;
    }
    stream.synchronize()?;

    let block_final_u32 = block_rmw_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let block_final_u64 = block_rmw_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    let system_final_u32 = system_rmw_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let system_final_u64 = system_rmw_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    let block_old_values = block_old_values.to_host_vec(&stream)?;
    let system_old_values = system_old_values.to_host_vec(&stream)?;
    if block_final_u32 != N as u32
        || block_final_u64 != N as u64
        || system_final_u32 != N as u32
        || system_final_u64 != N as u64
        || !old_values_are_permutations(&block_old_values)
        || !old_values_are_permutations(&system_old_values)
    {
        return Err(format!(
            "legacy scoped RMW mismatch: block=({block_final_u32},{block_final_u64}), system=({system_final_u32},{system_final_u64}), block_old_values_permutations={}, system_old_values_permutations={}",
            old_values_are_permutations(&block_old_values),
            old_values_are_permutations(&system_old_values)
        )
        .into());
    }

    let block_cas = DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<BlockAtomicU32>();
    let system_cas = DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<SystemAtomicU32>();
    let device_strong_cas =
        DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<DeviceAtomicU32>();
    let device_ordered_cas =
        DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<DeviceAtomicU32>();
    let mut block_observed = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;
    let mut system_observed = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;
    let mut device_observed = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;
    let mut device_ordered_observed = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;

    // SAFETY: a single thread executes all CAS probes. The block-scoped target
    // is therefore accessed only inside one CTA, and all host reads occur after
    // synchronization.
    unsafe {
        module.scoped_integer_cas(
            &stream,
            LaunchConfig::for_num_elems(1),
            &block_cas,
            &system_cas,
            &device_strong_cas,
            &device_ordered_cas,
            &mut block_observed,
            &mut system_observed,
            &mut device_observed,
            &mut device_ordered_observed,
        )?;
    }
    stream.synchronize()?;

    let block_observed = block_observed.to_host_vec(&stream)?[0];
    let system_observed = system_observed.to_host_vec(&stream)?[0];
    let device_observed = device_observed.to_host_vec(&stream)?[0];
    let device_ordered_observed = device_ordered_observed.to_host_vec(&stream)?[0];
    let block_cas_final = block_cas.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let system_cas_final = system_cas.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let device_cas_final = device_strong_cas.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let device_ordered_final = device_ordered_cas.cast_elem::<u32>().to_host_vec(&stream)?[0];
    if block_observed != (7, 11)
        || system_observed != (7, 11)
        || device_observed != (7, 11)
        || device_ordered_observed != (7, 11)
        || block_cas_final != 11
        || system_cas_final != 11
        || device_cas_final != 11
        || device_ordered_final != 11
    {
        return Err(format!(
            "legacy scoped CAS mismatch: block={block_observed:?}/{block_cas_final}, system={system_observed:?}/{system_cas_final}, device={device_observed:?}/{device_cas_final}, device_ordered={device_ordered_observed:?}/{device_ordered_final}"
        )
        .into());
    }

    println!(
        "legacy_atomic_rmw_cas: PASS (device_rmw=({got_u32},{got_u64}), device_cas=({final_u32},{final_u64}), scoped_rmw=block({block_final_u32},{block_final_u64})/system({system_final_u32},{system_final_u64}), scoped_cas=block({block_cas_final})/system({system_cas_final})/device_strong_failure({device_cas_final})/device_ordered_success({device_ordered_final}))"
    );
    Ok(())
}
