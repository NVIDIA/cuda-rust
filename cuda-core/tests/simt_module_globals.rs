/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `ConstantHandle` resolves a module-scope global with its driver-reported
//! size, keeps the module loaded, and refuses writes past the symbol.
//!
//! The module is hand-written PTX with two data symbols and no kernel, so
//! this needs the driver's JIT but no device-code toolchain.

use cuda_core::{ConstantHandle, CudaContext, CudaModule, CudaStream, DriverError, IntoResult};
use std::mem::MaybeUninit;
use std::sync::Arc;

const PTX: &str = r#"
.version 7.8
.target sm_75
.address_size 64

.global .align 4 .b8 payload[16];
.const .align 4 .b8 tuning[8];
"#;

const INVALID_VALUE: DriverError =
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);

fn load() -> (Arc<CudaContext>, Arc<CudaStream>, Arc<CudaModule>) {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let module = ctx
        .load_module_from_ptx_src(PTX)
        .expect("failed to JIT the test PTX");
    (ctx, stream, module)
}

/// Reads `len` bytes back from the global `name` resolves to, synchronously.
fn read_back(module: &Arc<CudaModule>, name: &str, len: usize) -> Vec<u8> {
    let (dptr, size) = module.get_global(name).expect("get_global failed");
    assert!(len <= size, "test reads within the symbol");
    let mut host = vec![0_u8; len];
    module.context().bind_to_thread().expect("bind failed");
    unsafe { cuda_bindings::cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), dptr, len) }
        .result()
        .expect("cuMemcpyDtoH failed");
    host
}

fn staged(bytes: &[u8]) -> Box<[MaybeUninit<u8>]> {
    bytes.iter().map(|&b| MaybeUninit::new(b)).collect()
}

#[test]
fn constant_handle_carries_the_driver_reported_size() {
    let (_ctx, _stream, module) = load();

    let payload = ConstantHandle::new(&module, "payload").expect("payload must resolve");
    assert_eq!(payload.size(), 16);
    assert!(Arc::ptr_eq(payload.module(), &module));

    let tuning = ConstantHandle::new(&module, "tuning").expect("tuning must resolve");
    assert_eq!(tuning.size(), 8);

    assert_eq!(
        ConstantHandle::new(&module, "missing").map(|handle| handle.size()),
        Err(DriverError(
            cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_FOUND
        )),
        "an unknown symbol is the driver's error, not a handle"
    );
}

#[test]
fn write_async_staged_rejects_oversize_writes_before_touching_the_device() {
    let (_ctx, stream, module) = load();
    let handle = ConstantHandle::new(&module, "payload").expect("payload must resolve");

    assert_eq!(
        handle.write_async_staged(&stream, staged(&[0xAB; 17])),
        Err(INVALID_VALUE),
        "one byte past the symbol must be refused"
    );
    stream.synchronize().expect("sync failed");
    assert_eq!(
        read_back(&module, "payload", 16),
        vec![0_u8; 16],
        "a refused write must leave the symbol untouched"
    );

    let bytes: Vec<u8> = (1..=16).collect();
    handle
        .write_async_staged(&stream, staged(&bytes))
        .expect("an exact-size write is accepted");
    stream.synchronize().expect("sync failed");
    assert_eq!(read_back(&module, "payload", 16), bytes);

    handle
        .write_async_staged(&stream, staged(&[7, 7]))
        .expect("a shorter write is accepted");
    stream.synchronize().expect("sync failed");
    assert_eq!(&read_back(&module, "payload", 16)[..2], &[7, 7]);
}

#[test]
fn raw_writers_reject_oversize_lengths() {
    let (_ctx, stream, module) = load();
    let handle = ConstantHandle::new(&module, "tuning").expect("tuning must resolve");
    let src = [0x5A_u8; 9];

    // SAFETY: `src` outlives the calls and is long enough for every length
    // passed; the oversize lengths are rejected before the pointer is used.
    unsafe {
        assert_eq!(
            handle.write_async(&stream, src.as_ptr(), 9),
            Err(INVALID_VALUE)
        );
        assert_eq!(handle.write_blocking(src.as_ptr(), 9), Err(INVALID_VALUE));
        handle
            .write_async(&stream, src.as_ptr(), 8)
            .expect("exact-size async write");
    }
    stream.synchronize().expect("sync failed");
    assert_eq!(read_back(&module, "tuning", 8), vec![0x5A; 8]);

    let blocking = [0x3C_u8; 8];
    // SAFETY: as above.
    unsafe { handle.write_blocking(blocking.as_ptr(), 8) }.expect("exact-size blocking write");
    assert_eq!(read_back(&module, "tuning", 8), vec![0x3C; 8]);
}

#[test]
fn handle_keeps_its_module_loaded() {
    let (_ctx, stream, module) = load();
    let handle = ConstantHandle::new(&module, "payload").expect("payload must resolve");
    let clone = handle.clone();
    assert_eq!(
        Arc::strong_count(&module),
        3,
        "each handle holds the module"
    );
    drop(module);

    // The module's only owners are now the handles; the write must still
    // land in a loaded module rather than in an unloaded one's address.
    handle
        .write_async_staged(&stream, staged(&[9; 16]))
        .expect("write through the sole owner of the module");
    stream.synchronize().expect("sync failed");
    assert_eq!(read_back(clone.module(), "payload", 16), vec![9; 16]);
    drop(handle);
    assert_eq!(Arc::strong_count(clone.module()), 1);
}

/// The context check compares driver handles, not `Arc` identity: a second
/// `CudaContext` for the same device retains the same primary context and
/// its streams must be accepted. (A genuinely different context needs a
/// second GPU, which this suite does not assume.)
#[test]
fn writes_accept_a_stream_from_another_handle_to_the_same_context() {
    let (ctx, _stream, module) = load();
    let handle = ConstantHandle::new(&module, "payload").expect("payload must resolve");

    let same_device = CudaContext::new(ctx.ordinal()).expect("second context object");
    let stream = same_device
        .new_stream()
        .expect("stream on second context object");
    handle
        .write_async_staged(&stream, staged(&[1; 16]))
        .expect("the same primary context is the same context");
    stream.synchronize().expect("sync failed");
}
