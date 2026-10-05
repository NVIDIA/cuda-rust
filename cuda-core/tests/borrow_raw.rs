/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verifies the ownership contract of `Device::borrow_raw` and
//! `Stream::borrow_raw`: dropping a borrowed wrapper must not release the
//! primary context or destroy the stream, so the source handles keep
//! working afterward.

use core::ffi::{c_int, c_void};
use cuda_core::{Device, Function, Module, Stream};

const NOOP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry noop()
{
    ret;
}
";

fn has_gpu() -> bool {
    Device::device_count().map(|n| n > 0).unwrap_or(false)
}

fn borrow_device(src: &Device) -> std::sync::Arc<Device> {
    unsafe {
        Device::borrow_raw(
            src.cu_ctx() as *mut c_void,
            src.cu_device() as c_int,
            src.ordinal(),
        )
    }
}

fn borrow_stream(src: &Stream, dev: &std::sync::Arc<Device>) -> std::sync::Arc<Stream> {
    unsafe { Stream::borrow_raw(src.cu_stream() as *mut c_void, dev) }
}

#[test]
fn borrowed_drop_leaves_source_device_usable() {
    if !has_gpu() {
        return;
    }
    let source = Device::new(0).unwrap();

    {
        let borrowed = borrow_device(&source);
        borrowed.bind_to_thread().unwrap();
        unsafe { borrowed.synchronize() }.unwrap();
    }

    source.bind_to_thread().unwrap();
    unsafe { source.synchronize() }.unwrap();
    let _stream = source.new_stream().unwrap();
}

#[test]
fn borrowed_drop_leaves_source_stream_usable() {
    if !has_gpu() {
        return;
    }
    let source_dev = Device::new(0).unwrap();
    let source_stream = source_dev.new_stream().unwrap();

    {
        let borrowed_dev = borrow_device(&source_dev);
        let borrowed_stream = borrow_stream(&source_stream, &borrowed_dev);
        unsafe { borrowed_stream.synchronize() }.unwrap();
    }

    unsafe { source_stream.synchronize() }.unwrap();
}

#[test]
fn borrowed_module_drop_does_not_unload() {
    if !has_gpu() {
        return;
    }
    let source_dev = Device::new(0).unwrap();
    let source_module = source_dev.load_module_from_ptx_src(NOOP_PTX).unwrap();

    {
        let borrowed =
            unsafe { Module::borrow_raw(source_module.cu_module() as *mut c_void, &source_dev) };
        // Looking up a function through the borrowed wrapper exercises the
        // raw CUmodule handle.
        let _f = borrowed.load_function("noop").unwrap();
    }

    // If the borrowed drop had unloaded the module, this lookup would fail.
    let _f = source_module.load_function("noop").unwrap();
}

#[test]
fn borrowed_function_uses_parent_module() {
    if !has_gpu() {
        return;
    }
    let source_dev = Device::new(0).unwrap();
    let source_module = source_dev.load_module_from_ptx_src(NOOP_PTX).unwrap();
    let source_fn = source_module.load_function("noop").unwrap();

    let raw_fn = unsafe { source_fn.cu_function() } as *mut c_void;
    let borrowed_fn = unsafe { Function::borrow_raw(raw_fn, &source_module) };
    assert_eq!(unsafe { borrowed_fn.cu_function() }, unsafe {
        source_fn.cu_function()
    });
}

#[test]
fn many_borrowed_drops_do_not_invalidate_source() {
    if !has_gpu() {
        return;
    }
    let source_dev = Device::new(0).unwrap();
    let source_stream = source_dev.new_stream().unwrap();

    for _ in 0..8 {
        let borrowed_dev = borrow_device(&source_dev);
        let borrowed_stream = borrow_stream(&source_stream, &borrowed_dev);
        unsafe { borrowed_stream.synchronize() }.unwrap();
    }

    unsafe { source_stream.synchronize() }.unwrap();
    let _another = source_dev.new_stream().unwrap();
}
