/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Graph-capture input discipline, passing side: launchers over pre-allocated
//! inputs — `&Tensor`, `Arc<Tensor>`, `&TensorView`, and `&mut` partitions —
//! are `GraphNode`s and record into a `CudaGraph::scope`, and the replayed
//! graph computes into the pre-allocated buffers. The rejected side (an
//! allocating input op) is the compile-fail case in
//! `tests/ui/graph_scope_rejects_allocating_input.rs`.

use cutile::api;
use cutile::cuda_async::cuda_graph::CudaGraph;
use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod graph_scope_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let ta: Tile<f32, { [B] }> = load_tile_like(a, out);
        let tb: Tile<f32, { [B] }> = load_tile_like(b, out);
        out.store(ta + tb);
    }
}

use graph_scope_module::add;

#[test]
fn pre_allocated_inputs_record_and_replay() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        let a = api::arange::<f32>(8).sync_on(&stream).expect("a");
        let ones = Arc::new(api::ones::<f32>(&[8]).sync_on(&stream).expect("ones"));
        let mut out = api::zeros::<f32>(&[8]).sync_on(&stream).expect("out");
        let mut out2 = api::zeros::<f32>(&[8]).sync_on(&stream).expect("out2");

        let graph = CudaGraph::scope(&stream, |s| {
            // `&mut` partition output, borrowed `&Tensor` and `Arc<Tensor>` inputs.
            s.record(add((&mut out).partition([8]), &a, ones.clone()))?;
            // A `&TensorView` input over a buffer written by the previous node.
            let view = out.view(&[8])?;
            s.record(add((&mut out2).partition([8]), &view, &a))?;
            Ok(())
        })
        .expect("scope capture");

        // Capture records; replay computes.
        graph.launch().sync_on(&stream).expect("graph launch");

        let host: Vec<f32> = out.to_host_vec().sync_on(&stream).expect("out host");
        let expected: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        assert_eq!(host, expected, "out = arange + 1");

        let host2: Vec<f32> = out2.to_host_vec().sync_on(&stream).expect("out2 host");
        let expected2: Vec<f32> = (0..8).map(|i| 2.0 * i as f32 + 1.0).collect();
        assert_eq!(host2, expected2, "out2 = out + arange");
    });
}
