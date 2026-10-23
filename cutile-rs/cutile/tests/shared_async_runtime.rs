/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile and SIMT launches share allocation ownership, ordering and graph replay.

mod common;

use cuda_async::device_box::DeviceBox;
use cuda_async::launch::{AsyncKernelLaunch, AsyncKernelLaunchBuilder};
use cuda_core::{Context, DeviceBuffer, Function, LaunchConfig};
use cutile::prelude::*;

const SCALE_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry scale(.param .u64 ptr, .param .u32 len) {
    .reg .u64 %p, %offset;
    .reg .u32 %i, %n;
    .reg .f32 %v;
    .reg .pred %done;
    ld.param.u64 %p, [ptr];
    ld.param.u32 %n, [len];
    mov.u32 %i, %tid.x;
    setp.ge.u32 %done, %i, %n;
    @%done bra END;
    mul.wide.u32 %offset, %i, 4;
    add.u64 %p, %p, %offset;
    ld.global.f32 %v, [%p];
    mul.f32 %v, %v, 2.0;
    st.global.f32 [%p], %v;
END:
    ret;
}
"#;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry]
    fn increment<const S: [i32; 1]>(out: &mut Tensor<f32, S>) {
        let x = out.load();
        out.store(x + broadcast_scalar(1.0f32, out.shape()));
    }
}

fn scale(tensor: &Tensor<f32>, function: Arc<Function>) -> AsyncKernelLaunch<'static> {
    let view = tensor.as_buffer().unwrap();
    let mut builder = AsyncKernelLaunchBuilder::new(function);
    builder.retain_buffer(view.storage().clone(), true);
    builder.push_args((view.cu_deviceptr(), view.len() as u32));
    // SAFETY: this PTX accepts pointer/length, covers at most 32 f32 values,
    // and each of 32 threads updates its own element. Storage is retained.
    unsafe {
        builder.finalize_unchecked(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

#[test]
fn mixed_launches_and_graph_replays_use_the_same_allocation() {
    common::with_test_stack(|| {
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let other = context.new_stream().unwrap();
        let buffer = DeviceBuffer::from_host(&stream, &[1.0f32; 32]).unwrap();
        let identity = buffer.storage().allocation_id();
        let pointer = buffer.cu_deviceptr();
        let buffer = DeviceBox::from_buffer(buffer).into_buffer();
        assert_eq!(buffer.storage().allocation_id(), identity);
        let mut tensor = Arc::new(Tensor::from_buffer(buffer, vec![32]));
        let function = Arc::new(
            context
                .load_module_from_ptx_src(SCALE_PTX)
                .unwrap()
                .load_function("scale")
                .unwrap(),
        );

        // Warm compilation before capture, and verify Tile -> SIMT ordering.
        kernels::increment(Arc::get_mut(&mut tensor).unwrap().partition([32]))
            .then(|(part,)| scale(part.unpartition(), function.clone()))
            .sync_on(&stream)
            .unwrap();
        assert_eq!(
            tensor.as_buffer().unwrap().storage().allocation_id(),
            identity
        );
        assert_eq!(tensor.device_pointer().cu_deviceptr(), pointer);
        assert_eq!(
            (&tensor).to_host_vec().sync_on(&stream).unwrap(),
            vec![4.0; 32]
        );

        let graph = kernels::increment(Arc::get_mut(&mut tensor).unwrap().partition([32]))
            .then(|(part,)| scale(part.unpartition(), function.clone()))
            .graph_on(stream.clone())
            .unwrap();
        graph.launch().sync_on(&other).unwrap();
        graph.launch().sync_on(&stream).unwrap();
        drop(graph);
        assert_eq!(
            tensor.as_buffer().unwrap().storage().allocation_id(),
            identity
        );
        assert_eq!(
            (&tensor).to_host_vec().sync_on(&stream).unwrap(),
            vec![22.0; 32]
        );
    });
}
