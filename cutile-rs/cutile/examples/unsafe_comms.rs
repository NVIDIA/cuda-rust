/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unsafe two-GPU communication with explicit CUDA VMM mappings and device
//! release/acquire signaling. Run from cutile-rs with CUDA 13.5 selected:
//! cargo run -p cutile --example unsafe_comms -- unicast
//! cargo run -p cutile --example unsafe_comms -- multicast
//! Each mode checks 256 rounds, then repeats with the launch order reversed.
//! Unicast needs peer access; multicast additionally needs NVLink SHARP.
use cuda_core::{vmm, Context, IntoResult};
use cutile::prelude::*;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;
    use cutile::tileir::*;
    // Each GPU writes the peer's inbox and polls its own. Sequence numbers
    // increase at every phase, so the inbox never needs resetting in a kernel.
    // The release/acquire pair supplies inter-GPU ordering; the token carries
    // that ordering to subsequent Tile IR operations on this GPU.
    #[allow(unreachable_code)]
    unsafe fn exchange(local: *mut i32, peer_inbox: *mut i32, phase: i32, before: Token) -> Token {
        let inbox: PointerTile<*mut i32, { [] }> = pointer_to_tile(local);
        let inbox: PointerTile<*mut i32, { [] }> = addptr(inbox, 16i32);
        let peer: PointerTile<*mut i32, { [] }> = pointer_to_tile(peer_inbox);
        let peer: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_unicast(peer) };
        let value: Tile<i32, { [] }> = scalar_to_tile(phase);
        let mut token = unsafe {
            store_ptr_tko(
                peer,
                value,
                ordering::Release,
                Some(scope::System),
                None,
                Some(before),
                Latency::<0>,
            )
        };
        let mut observed = 0i32;
        let mut polls = 0i32;
        while observed < phase && polls < 10000000i32 {
            let (value, loaded): (Tile<i32, { [] }>, Token) = unsafe {
                load_ptr_tko(
                    inbox,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            observed = tile_to_scalar(value);
            token = loaded;
            polls = polls + 1i32;
        }
        cuda_tile_assert!(observed >= phase, "two-GPU handshake timed out");
        token
    }

    unsafe fn snapshot(
        output: *mut i32,
        offset: i32,
        value: Tile<i32, { [4] }>,
        before: Token,
    ) -> Token {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(output);
        let lanes: Tile<i32, { [4] }> = iota(shape![4]);
        let base: PointerTile<*mut i32, { [] }> = addptr(base, offset);
        let one: PointerTile<*mut i32, { [1] }> = reshape_ptr(base, shape![1]);
        let pointers: PointerTile<*mut i32, { [4] }> = broadcast_ptr(one, shape![4]);
        let pointers: PointerTile<*mut i32, { [4] }> = addptr_tile(pointers, lanes);
        unsafe {
            store_ptr_tko(
                pointers,
                value,
                ordering::Weak,
                None::<scope::System>,
                None,
                Some(before),
                Latency::<0>,
            )
        }
    }

    #[cutile::entry()]
    unsafe fn nvls_concurrent(
        local: *mut i32,
        multicast: *mut i32,
        peer_inbox: *mut i32,
        output: *mut i32,
        rank: i32,
        rounds: i32,
    ) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(multicast);
        let base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_multicast(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<i32, { [4] }> =
            unsafe { make_tensor_view(base, shape![4], strides, new_token_unordered()) };
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                &tensor,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let local_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(local);
        let lanes: Tile<i32, { [4] }> = iota(shape![4]);
        let one: PointerTile<*mut i32, { [1] }> = reshape_ptr(local_base, shape![1]);
        let local_pointers: PointerTile<*mut i32, { [4] }> = broadcast_ptr(one, shape![4]);
        let local_pointers: PointerTile<*mut i32, { [4] }> = addptr_tile(local_pointers, lanes);
        let mut token = new_token_unordered();
        for round in 0i32..rounds {
            let phase = round * 6i32;
            let seed: Tile<i32, { [4] }> =
                broadcast_scalar(1i32 + round * 100i32 + rank * 10i32, shape![4]);
            token = unsafe {
                store_ptr_tko(
                    local_pointers,
                    seed + lanes,
                    ordering::Weak,
                    None::<scope::System>,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            // The local and multicast addresses alias the same physical bytes.
            token = unsafe { memory_fence_alias_tko(token) };
            token = unsafe { exchange(local, peer_inbox, phase + 1i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (sum, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                multimem_load_reduce_view_tko(
                    &view,
                    [0i32],
                    multimem::Add,
                    ordering::Acquire,
                    scope::System,
                    Some(token),
                    [true],
                    shape![4],
                )
            };
            token = unsafe { snapshot(output, round * 12i32, sum, loaded) };
            token = unsafe { exchange(local, peer_inbox, phase + 2i32, token) };

            // Both GPUs contribute to every backing allocation concurrently.
            let increment: Tile<i32, { [4] }> = broadcast_scalar(rank + 1i32, shape![4]);
            token = unsafe {
                multimem_reduce_view_tko(
                    &view,
                    increment,
                    [0i32],
                    multimem::Add,
                    ordering::Release,
                    scope::System,
                    Some(token),
                    [true],
                )
            };
            token = unsafe { memory_fence_alias_tko(token) };
            token = unsafe { exchange(local, peer_inbox, phase + 3i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (reduced, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    local_pointers,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { snapshot(output, round * 12i32 + 4i32, reduced, loaded) };
            token = unsafe { exchange(local, peer_inbox, phase + 4i32, token) };

            // Alternate the broadcast producer, with one writer per round.
            if rank == round % 2i32 {
                let seed: Tile<i32, { [4] }> =
                    broadcast_scalar(100000i32 + round * 100i32, shape![4]);
                token = unsafe {
                    multimem_store_view_tko(
                        &view,
                        seed + lanes,
                        [0i32],
                        ordering::Release,
                        scope::System,
                        Some(token),
                        [true],
                    )
                };
            }
            token = unsafe { memory_fence_alias_tko(token) };
            token = unsafe { exchange(local, peer_inbox, phase + 5i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (broadcast, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    local_pointers,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { snapshot(output, round * 12i32 + 8i32, broadcast, loaded) };
            // Neither producer may reuse its input before both consumers finish.
            token = unsafe { exchange(local, peer_inbox, phase + 6i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
        }
    }

    #[cutile::entry()]
    unsafe fn unicast_concurrent(
        local: *mut i32,
        peer: *mut i32,
        peer_inbox: *mut i32,
        output: *mut i32,
        rank: i32,
        rounds: i32,
    ) {
        let lanes: Tile<i32, { [4] }> = iota(shape![4]);
        let local_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(local);
        let one: PointerTile<*mut i32, { [1] }> = reshape_ptr(local_base, shape![1]);
        let local_pointers: PointerTile<*mut i32, { [4] }> = broadcast_ptr(one, shape![4]);
        let local_pointers: PointerTile<*mut i32, { [4] }> = addptr_tile(local_pointers, lanes);
        let peer_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(peer);
        let peer_base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_unicast(peer_base) };
        let one: PointerTile<*mut i32, { [1] }> = reshape_ptr(peer_base, shape![1]);
        let peer_pointers: PointerTile<*mut i32, { [4] }> = broadcast_ptr(one, shape![4]);
        let peer_pointers: PointerTile<*mut i32, { [4] }> = addptr_tile(peer_pointers, lanes);
        let mut token = new_token_unordered();
        for round in 0i32..rounds {
            let phase = round * 6i32;
            let seed: Tile<i32, { [4] }> =
                broadcast_scalar(1i32 + round * 100i32 + rank * 10i32, shape![4]);
            token = unsafe {
                store_ptr_tko(
                    local_pointers,
                    seed + lanes,
                    ordering::Weak,
                    None::<scope::System>,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { memory_fence_alias_tko(token) };
            token = unsafe { exchange(local, peer_inbox, phase + 1i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (received, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    peer_pointers,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { snapshot(output, round * 12i32, received, loaded) };
            token = unsafe { exchange(local, peer_inbox, phase + 2i32, token) };

            // Each GPU writes the other GPU's allocation through NVLink.
            let seed: Tile<i32, { [4] }> =
                broadcast_scalar(10000i32 + round * 100i32 + rank * 10i32, shape![4]);
            token = unsafe {
                store_ptr_tko(
                    peer_pointers,
                    seed + lanes,
                    ordering::Release,
                    Some(scope::System),
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { memory_fence_alias_tko(token) };
            token = unsafe { exchange(local, peer_inbox, phase + 3i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (received, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    local_pointers,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { snapshot(output, round * 12i32 + 4i32, received, loaded) };
            token = unsafe { exchange(local, peer_inbox, phase + 4i32, token) };

            // Each backing receives one local and one remote atomic add.
            // They can overlap, and system scope makes their updates atomic
            // with respect to the other GPU's accesses.
            let increment: Tile<i32, { [4] }> = broadcast_scalar(rank + 1i32, shape![4]);
            let (_old_peer, updated_peer): (Tile<i32, { [4] }>, Token) = unsafe {
                atomic_rmw_tko(
                    peer_pointers,
                    increment,
                    atomic::Add,
                    ordering::AcqRel,
                    scope::System,
                    None,
                    Some(token),
                )
            };
            let (_old_local, updated_local): (Tile<i32, { [4] }>, Token) = unsafe {
                atomic_rmw_tko(
                    local_pointers,
                    increment,
                    atomic::Add,
                    ordering::AcqRel,
                    scope::System,
                    None,
                    Some(updated_peer),
                )
            };
            token = unsafe { memory_fence_alias_tko(updated_local) };
            token = unsafe { exchange(local, peer_inbox, phase + 5i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
            let (received, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
                load_ptr_tko(
                    local_pointers,
                    ordering::Acquire,
                    Some(scope::System),
                    None,
                    None,
                    Some(token),
                    Latency::<0>,
                )
            };
            token = unsafe { snapshot(output, round * 12i32 + 8i32, received, loaded) };
            token = unsafe { exchange(local, peer_inbox, phase + 6i32, token) };
            token = unsafe { memory_fence_alias_tko(token) };
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Unicast,
    Multicast,
}

fn write_words(address: u64, data: &[i32; 4]) {
    unsafe {
        cuda_core::sys::cuMemcpyHtoD_v2(address, data.as_ptr().cast(), std::mem::size_of_val(data))
            .result()
            .unwrap();
    }
}

fn read_words(address: u64) -> [i32; 4] {
    let mut data = [0i32; 4];
    unsafe {
        cuda_core::sys::cuMemcpyDtoH_v2(
            data.as_mut_ptr().cast(),
            address,
            std::mem::size_of_val(&data),
        )
        .result()
        .unwrap();
    }
    data
}

/// Run the example and check every snapshot on both devices. Hardware tests
/// call this same function, so the example exercises the tested implementation.
pub fn run(mode: Mode, rounds: i32) {
    assert!((1..=10000).contains(&rounds), "rounds must be in 1..=10000");
    const INBOX_OFFSET: u64 = 16 * 4;
    let contexts = [
        Context::new(0).expect("GPU 0 required"),
        Context::new(1).expect("GPU 1 required"),
    ];
    let devices = contexts.each_ref().map(|ctx| ctx.cu_device());
    // These checks fail; an explicit hardware test never silently skips.
    for i in 0..2 {
        let mut peer_access = 0;
        unsafe {
            cuda_core::sys::cuDeviceCanAccessPeer(&mut peer_access, devices[i], devices[1 - i])
                .result()
                .unwrap();
        }
        assert_ne!(peer_access, 0, "GPU {i} cannot access its peer");
        if mode == Mode::Unicast {
            let mut native_atomics = 0;
            unsafe {
                cuda_core::sys::cuDeviceGetP2PAttribute(
                    &mut native_atomics,
                    cuda_core::sys::CUdevice_P2PAttribute_enum_CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                    devices[i],
                    devices[1 - i],
                )
                .result()
                .unwrap();
            }
            assert_ne!(native_atomics, 0, "GPU {i} needs native peer atomics");
        }
        if mode == Mode::Multicast {
            assert!(
                vmm::multicast_supported(devices[i]).unwrap(),
                "NVLink SHARP multicast required"
            );
        }
    }
    let mut granularity = vmm::allocation_granularity(devices[0])
        .unwrap()
        .max(vmm::allocation_granularity(devices[1]).unwrap());
    if mode == Mode::Multicast {
        granularity = granularity.max(
            vmm::multicast_granularity(2, 1 << 20, vmm::MulticastGranularity::Recommended).unwrap(),
        );
    }
    let size = vmm::align_size(1 << 20, granularity);
    let team = if mode == Mode::Multicast {
        let team = vmm::MulticastObject::new(2, size).unwrap();
        for &dev in &devices {
            team.add_device(dev).unwrap();
        }
        Some(team)
    } else {
        None
    };
    let mut physical = Vec::new();
    let mut bindings = Vec::new();
    for (ctx, &dev) in contexts.iter().zip(&devices) {
        ctx.bind_to_thread().unwrap();
        let allocation = vmm::PhysicalAllocation::new(dev, size).unwrap();
        if let Some(team) = &team {
            bindings.push(team.bind_mem(0, &allocation, 0, size).unwrap());
        }
        physical.push(allocation);
    }
    // Both devices access each ordinary mapping. The inbox at word 16 has
    // exactly one writer (the peer); data uses words 0..4. Mapping objects must
    // be dropped before their corresponding virtual address reservations.
    let mut mappings = Vec::new();
    for (i, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().unwrap();
        let uc = vmm::VirtualReservation::new(size, granularity).unwrap();
        let uc_map = vmm::Mapping::new(uc.base(), size, &physical[i], 0).unwrap();
        vmm::set_access(uc.base(), size, &devices).unwrap();
        let mc = team.as_ref().map(|team| {
            let mc = vmm::VirtualReservation::new(size, granularity).unwrap();
            let mc_map = vmm::Mapping::new_multicast(mc.base(), size, team, 0).unwrap();
            vmm::set_access(mc.base(), size, &[devices[i]]).unwrap();
            (mc_map, mc)
        });
        mappings.push((uc_map, uc, mc));
    }
    let streams = contexts.each_ref().map(|ctx| ctx.new_stream().unwrap());
    let outputs = streams.each_ref().map(|stream| {
        stream.context().bind_to_thread().unwrap();
        std::sync::Arc::new(
            api::zeros::<i32>(&[(rounds * 12) as usize])
                .sync_on(stream)
                .unwrap(),
        )
    });
    let dispatch = |i: usize, compile_only: bool| {
        contexts[i].bind_to_thread()?;
        let remote = match mode {
            Mode::Unicast => mappings[1 - i].1.base(),
            Mode::Multicast => mappings[i].2.as_ref().unwrap().1.base(),
        };
        macro_rules! launch {
            ($kernel:path) => {{
                // SAFETY: the mappings and buffers below outlive every launch.
                let operation = unsafe {
                    $kernel(
                        cutile::tile_kernel::DevicePointer::from_cu_deviceptr(mappings[i].1.base()),
                        cutile::tile_kernel::DevicePointer::from_cu_deviceptr(remote),
                        cutile::tile_kernel::DevicePointer::from_cu_deviceptr(
                            mappings[1 - i].1.base() + INBOX_OFFSET,
                        ),
                        outputs[i].device_pointer(),
                        i as i32,
                        rounds,
                    )
                }
                .grid((1, 1, 1));
                if compile_only {
                    operation.compile_on(&streams[i])
                } else {
                    unsafe { operation.async_on(&streams[i]) }.map(|_| ())
                }
            }};
        }
        match mode {
            Mode::Unicast => launch!(kernels::unicast_concurrent),
            Mode::Multicast => launch!(kernels::nvls_concurrent),
        }
    };
    // Avoid JIT/module-loading delays while the first GPU waits for its peer.
    for i in 0..2 {
        dispatch(i, true).unwrap();
    }
    for order in [[0usize, 1usize], [1usize, 0usize]] {
        for (i, ctx) in contexts.iter().enumerate() {
            ctx.bind_to_thread().unwrap();
            write_words(mappings[i].1.base() + INBOX_OFFSET, &[0; 4]);
            // Poison all snapshots on every launch to detect missing stores.
            unsafe {
                cuda_core::sys::cuMemsetD32_v2(
                    outputs[i].device_pointer().cu_deviceptr(),
                    u32::MAX,
                    (rounds * 12) as usize,
                )
                .result()
                .unwrap();
            }
        }
        // Both kernels are enqueued before waiting for either stream. There
        // are no host copies, waits, or launches between device-side rounds.
        let launches = order.map(|i| dispatch(i, false));
        // Drain BOTH streams before propagating a launch or execution error.
        let completed = streams.each_ref().map(|stream| {
            stream.context().bind_to_thread()?;
            stream.synchronize()
        });
        for result in launches {
            result.unwrap();
        }
        for result in completed {
            result.unwrap();
        }
        for (rank, stream) in streams.iter().enumerate() {
            contexts[rank].bind_to_thread().unwrap();
            let actual = (&outputs[rank]).to_host_vec().sync_on(stream).unwrap();
            for round in 0..rounds {
                for lane in 0..4i32 {
                    let expected = match mode {
                        Mode::Multicast => [
                            2 + round * 200 + 10 + lane * 2,
                            1 + round * 100 + rank as i32 * 10 + lane + 3,
                            100000 + round * 100 + lane,
                        ],
                        Mode::Unicast => [
                            1 + round * 100 + (1 - rank) as i32 * 10 + lane,
                            10000 + round * 100 + (1 - rank) as i32 * 10 + lane,
                            10000 + round * 100 + (1 - rank) as i32 * 10 + lane + 3,
                        ],
                    };
                    for (phase, value) in expected.into_iter().enumerate() {
                        let index = round as usize * 12 + phase * 4 + lane as usize;
                        assert_eq!(actual[index], value,
                            "{mode:?}, launch order {order:?}, GPU {rank}, round {round}, phase {phase}, lane {lane}");
                    }
                }
            }
            contexts[rank].bind_to_thread().unwrap();
            assert_eq!(
                read_words(mappings[rank].1.base() + INBOX_OFFSET),
                [rounds * 6, 0, 0, 0]
            );
            let last = match mode {
                Mode::Multicast => 100000 + (rounds - 1) * 100,
                Mode::Unicast => 10000 + (rounds - 1) * 100 + (1 - rank) as i32 * 10 + 3,
            };
            assert_eq!(
                read_words(mappings[rank].1.base()),
                [last, last + 1, last + 2, last + 3]
            );
        }
        println!("{mode:?}: {rounds} rounds passed on both GPUs, launch order {order:?}");
    }
    for (ctx, mapping) in contexts.iter().zip(mappings) {
        ctx.bind_to_thread().unwrap();
        drop(mapping);
    }
    drop(bindings);
    drop(team);
    drop(physical);
}

// The integration test includes this example as a module and calls run().
#[allow(dead_code)]
fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "all".into());
    let rounds = args
        .next()
        .map(|s| s.parse::<i32>().expect("integer round count"))
        .unwrap_or(256);
    assert!(
        args.next().is_none(),
        "usage: unsafe_comms [unicast|multicast|all] [rounds]"
    );
    // Compilation needs more stack than Rust's default spawned-thread size.
    std::thread::Builder::new()
        .stack_size(8_000_000)
        .spawn(move || match mode.as_str() {
            "unicast" => run(Mode::Unicast, rounds),
            "multicast" => run(Mode::Multicast, rounds),
            "all" => {
                run(Mode::Unicast, rounds);
                run(Mode::Multicast, rounds);
            }
            _ => panic!("usage: unsafe_comms [unicast|multicast|all] [rounds]"),
        })
        .unwrap()
        .join()
        .unwrap();
}
