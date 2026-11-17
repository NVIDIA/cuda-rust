/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Event::record_with_flags` and `Stream::wait_event_with_flags`: an external
//! record or wait during stream capture becomes an event-record or event-wait
//! node of the graph; a plain record during capture adds no node.

use cuda_core::sys;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, IntoResult};
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Captures `issue(stream)` on `stream` and returns the graph and the number of
/// its nodes of type `kind`.
fn capture(
    stream: &CudaStream,
    kind: sys::CUgraphNodeType,
    issue: impl FnOnce(&CudaStream) -> Result<(), DriverError>,
) -> (sys::CUgraph, usize) {
    unsafe {
        stream
            .begin_capture(sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .expect("begin capture");
    }
    let issued = issue(stream);
    let graph = unsafe { stream.end_capture() }.expect("end capture");
    issued.expect("issue during capture");

    let mut n = 0usize;
    unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) }
        .result()
        .expect("node count");
    let mut nodes = vec![std::ptr::null_mut(); n];
    if n > 0 {
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n) }
            .result()
            .expect("nodes");
    }
    let matching = nodes
        .iter()
        .filter(|&&node| {
            let mut ty = MaybeUninit::uninit();
            unsafe { sys::cuGraphNodeGetType(node, ty.as_mut_ptr()) }
                .result()
                .expect("node type");
            let ty = unsafe { ty.assume_init() };
            ty == kind
        })
        .count();
    (graph, matching)
}

const EVENT_RECORD: sys::CUgraphNodeType =
    sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EVENT_RECORD;
const WAIT_EVENT: sys::CUgraphNodeType = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_WAIT_EVENT;

fn setup() -> (Arc<CudaContext>, Arc<CudaStream>, CudaEvent) {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    let event = ctx
        .new_event_with_flags(None)
        .expect("failed to create event");
    (ctx, stream, event)
}

#[test]
fn an_external_record_is_a_graph_node_that_each_launch_records() {
    let (_ctx, stream, event) = setup();
    let (graph, records) = capture(&stream, EVENT_RECORD, |s| {
        event.record_with_flags(s, sys::CUevent_record_flags_enum_CU_EVENT_RECORD_EXTERNAL)
    });
    assert_eq!(records, 1, "one event-record node");

    let mut exec = MaybeUninit::uninit();
    unsafe {
        sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0)
            .result()
            .expect("instantiate");
        let exec = exec.assume_init();
        for _ in 0..2 {
            sys::cuGraphLaunch(exec, stream.cu_stream())
                .result()
                .expect("launch");
            event.synchronize().expect("the launch records the event");
            assert!(event.query().expect("query"));
        }
        sys::cuGraphExecDestroy(exec)
            .result()
            .expect("destroy exec");
        sys::cuGraphDestroy(graph).result().expect("destroy graph");
    }
}

#[test]
fn a_plain_record_during_capture_adds_no_node() {
    let (_ctx, stream, event) = setup();
    let (graph, records) = capture(&stream, EVENT_RECORD, |s| {
        event.record_with_flags(s, sys::CUevent_record_flags_enum_CU_EVENT_RECORD_DEFAULT)
    });
    assert_eq!(records, 0);
    unsafe { sys::cuGraphDestroy(graph) }
        .result()
        .expect("destroy graph");
}

#[test]
fn an_external_wait_is_a_graph_node() {
    let (ctx, stream, event) = setup();
    let producer = ctx.new_stream().expect("failed to create stream");
    event.record(&producer).expect("record outside the capture");
    let (graph, waits) = capture(&stream, WAIT_EVENT, |s| {
        s.wait_event_with_flags(&event, sys::CUevent_wait_flags_enum_CU_EVENT_WAIT_EXTERNAL)
    });
    assert_eq!(waits, 1, "one event-wait node");
    unsafe { sys::cuGraphDestroy(graph) }
        .result()
        .expect("destroy graph");
}

#[test]
fn outside_capture_default_flags_record_and_wait() {
    let (ctx, stream, event) = setup();
    let consumer = ctx.new_stream().expect("failed to create stream");
    event
        .record_with_flags(
            &stream,
            sys::CUevent_record_flags_enum_CU_EVENT_RECORD_DEFAULT,
        )
        .expect("record");
    consumer
        .wait_event_with_flags(&event, sys::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT)
        .expect("wait");
    consumer.synchronize().expect("synchronize");
    assert!(event.query().expect("query"));
}

#[test]
fn an_external_record_outside_capture_is_rejected() {
    let (_ctx, stream, event) = setup();
    assert!(event
        .record_with_flags(
            &stream,
            sys::CUevent_record_flags_enum_CU_EVENT_RECORD_EXTERNAL
        )
        .is_err());
}
