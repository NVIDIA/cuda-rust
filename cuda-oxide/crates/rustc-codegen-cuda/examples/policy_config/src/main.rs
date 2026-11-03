/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::{
    BlockRequirement, CudaContext, DeviceBuffer, KernelLaunchContract, LaunchConfig1D,
};
use cuda_device::config::{
    Atom, AtomKind, AtomSpec, Block, Global, Policy, PolicyId, RowMajor, Shape, Shape1, Thread,
    Tile, TileSpec,
};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract, thread};

/// The vocabulary this vector kernel library understands.
///
/// `cuda_device::config` supplies the metadata types. This domain trait gives
/// those descriptions meaning for this particular kernel.
trait VectorPolicy: Policy {
    type BlockTile: TileSpec<Layout = RowMajor, MemorySpace = Global, Scope = Block>;
    type ElementAtom: AtomSpec<Kind = XorRotate, Scope = Thread>;

    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const ITEMS_PER_THREAD: u32;
    const UNROLL: u32;
    const TAG: u32;
}

enum XorRotate {}
impl AtomKind for XorRotate {}

enum SmallTilePolicy {}

impl Policy for SmallTilePolicy {
    // Explicit library namespace + versioned policy value.
    const ID: PolicyId = PolicyId::new(0x706f_6c69_6379_5f63, 1);
}

impl VectorPolicy for SmallTilePolicy {
    type BlockTile = Tile<Shape1<1024>, RowMajor, Global, Block>;
    type ElementAtom = Atom<XorRotate, Shape1<1>, Thread>;

    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const ITEMS_PER_THREAD: u32 = 16;
    const UNROLL: u32 = 2;
    const TAG: u32 = 0x1357_9bdf;
}

enum WideTilePolicy {}

impl Policy for WideTilePolicy {
    const ID: PolicyId = PolicyId::new(0x706f_6c69_6379_5f63, 2);
}

impl VectorPolicy for WideTilePolicy {
    type BlockTile = Tile<Shape1<4096>, RowMajor, Global, Block>;
    type ElementAtom = Atom<XorRotate, Shape1<1>, Thread>;

    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const ITEMS_PER_THREAD: u32 = 16;
    const UNROLL: u32 = 4;
    const TAG: u32 = 0x2468_ace0;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub unsafe fn transform<P: VectorPolicy>(input: *const u32, output: *mut u32, count: u32) {
        let base = thread::index_1d().get() * P::ITEMS_PER_THREAD as usize;
        let mut lane = 0;
        #[unroll(P::UNROLL)]
        while lane < count {
            let index = base + lane as usize;
            // SAFETY: the caller must keep `count <= ITEMS_PER_THREAD` and
            // provide that full readable/writable tile for every thread.
            let value = unsafe { input.add(index).read_volatile() } ^ P::TAG;
            unsafe { output.add(index).write_volatile(value) };
            lane += 1;
        }
    }
}

fn specialization_names() -> [&'static str; 2] {
    use cuda_host::GenericCudaKernel;
    [
        <kernels::__transform_CudaKernel<SmallTilePolicy> as GenericCudaKernel>::ptx_name(),
        <kernels::__transform_CudaKernel<WideTilePolicy> as GenericCudaKernel>::ptx_name(),
    ]
}

fn entry_body<'source>(document: &ptx_parse::Document<'source>, name: &str) -> &'source str {
    document
        .callables_named(name)
        .find(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
        .and_then(|callable| callable.body_text().map(|_| callable.text()))
        .unwrap_or_else(|| panic!("missing or incomplete PTX entry `{name}`"))
}

fn is_generic_entry_name(name: &str) -> bool {
    name.strip_prefix("transform_TID_").is_some_and(|suffix| {
        suffix.len() == 32 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn entry_parameter_count(body: &str) -> usize {
    body.split_once(')')
        .expect("PTX entry has a closing parameter list")
        .0
        .matches(".param")
        .count()
}

fn contains_u32_immediate(body: &str, value: u32) -> bool {
    let unsigned = value.to_string();
    let signed = (value as i32).to_string();
    let hex = format!("0x{value:x}");
    body.contains(&unsigned) || body.contains(&signed) || body.contains(&hex)
}

fn llvm_function_with_immediate(llvm_ir: &str, value: u32) -> &str {
    let decimal = value.to_string();
    let matches: Vec<_> = llvm_ir
        .split("\ndefine ")
        .filter_map(|definition| {
            let (_, body_and_rest) = definition.split_once("{\n")?;
            let (body, _) = body_and_rest.split_once("\n}")?;
            body.contains(&decimal).then_some(body)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one raw LLVM function containing policy immediate {value}"
    );
    matches[0]
}

fn verify_raw_unroll_factor<P: VectorPolicy>(llvm_ir: &str) {
    let body = llvm_function_with_immediate(llvm_ir, P::TAG);
    // Before LLVM optimization, cuda-oxide's partial unroller emits N copies
    // in the grouped loop plus one copy in the remainder loop. A missing or
    // defaulted policy marker would leave only the original single body.
    let expected_body_copies = P::UNROLL as usize + 1;
    assert_eq!(
        body.matches("load volatile i32").count(),
        expected_body_copies,
        "raw LLVM did not preserve policy unroll factor {}:\n{body}",
        P::UNROLL
    );
    assert_eq!(
        body.matches("store volatile i32").count(),
        expected_body_copies,
        "raw LLVM did not preserve policy unroll factor {}:\n{body}",
        P::UNROLL
    );
}

fn verify_policy_metadata<P: VectorPolicy>() {
    type TileShape<P> = <<P as VectorPolicy>::BlockTile as TileSpec>::Shape;
    type AtomShape<P> = <<P as VectorPolicy>::ElementAtom as AtomSpec>::Shape;

    assert_eq!(
        TileShape::<P>::ELEMENTS,
        (P::MAX_THREADS as usize).checked_mul(P::ITEMS_PER_THREAD as usize)
    );
    assert_eq!(AtomShape::<P>::ELEMENTS, Some(1));
}

fn verify_generated_ptx() {
    let llvm_ir = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/policy_config.ll"))
        .expect("read policy_config.ll; run `cargo oxide build policy_config` first");
    let ptx = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/policy_config.ptx"))
        .expect("read policy_config.ptx; run `cargo oxide build policy_config` first");
    let document = ptx_parse::Document::parse(&ptx).expect("parse generated PTX");
    for marker in ["__launch_bounds_config", "__unroll_config"] {
        assert!(
            !ptx.contains(marker),
            "compile-time marker `{marker}` leaked into the PTX module"
        );
    }
    let [small_name, wide_name] = specialization_names();

    assert_eq!(
        <kernels::__transform_CudaKernel<SmallTilePolicy> as KernelLaunchContract>::SPEC.block(),
        BlockRequirement::MaxThreads(64),
    );
    assert_eq!(
        <kernels::__transform_CudaKernel<WideTilePolicy> as KernelLaunchContract>::SPEC.block(),
        BlockRequirement::MaxThreads(256),
    );

    assert_ne!(SmallTilePolicy::ID, WideTilePolicy::ID);
    assert_eq!(SmallTilePolicy::ID.namespace(), 0x706f_6c69_6379_5f63);
    assert_eq!(WideTilePolicy::ID.namespace(), 0x706f_6c69_6379_5f63);
    assert_eq!(SmallTilePolicy::ID.value(), 1);
    assert_eq!(WideTilePolicy::ID.value(), 2);
    assert_ne!(small_name, wide_name);
    assert_eq!(specialization_names(), [small_name, wide_name]);
    assert!(is_generic_entry_name(small_name), "{small_name}");
    assert!(is_generic_entry_name(wide_name), "{wide_name}");

    verify_policy_metadata::<SmallTilePolicy>();
    verify_policy_metadata::<WideTilePolicy>();
    verify_raw_unroll_factor::<SmallTilePolicy>(&llvm_ir);
    verify_raw_unroll_factor::<WideTilePolicy>(&llvm_ir);

    let small = entry_body(&document, small_name);
    let wide = entry_body(&document, wide_name);
    assert!(small.contains(".maxntid 64, 1, 1"), "{small}");
    assert!(small.contains(".minnctapersm 2"), "{small}");
    assert!(wide.contains(".maxntid 256, 1, 1"), "{wide}");
    assert!(wide.contains(".minnctapersm 1"), "{wide}");
    assert_eq!(entry_parameter_count(small), 3, "{small}");
    assert_eq!(entry_parameter_count(wide), 3, "{wide}");

    // The policy is a type parameter, not a runtime kernel argument. Its tag
    // must be folded into each specialization. `count` is ordinary workload
    // data and deliberately remains a runtime kernel argument.
    assert!(
        contains_u32_immediate(small, SmallTilePolicy::TAG),
        "{small}"
    );
    assert!(contains_u32_immediate(wide, WideTilePolicy::TAG), "{wide}");
}

fn launch_on_gpu() {
    const COUNT: u32 = 11;
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    // SAFETY: cargo-oxide built the embedded artifact from this cuda_module.
    let module = unsafe { kernels::load(&context) }.expect("load policy_config module");
    let input = DeviceBuffer::<u32>::zeroed(&stream, 4096).expect("allocate input");
    let small_output = DeviceBuffer::<u32>::zeroed(&stream, 1024).expect("allocate small output");
    let wide_output = DeviceBuffer::<u32>::zeroed(&stream, 4096).expect("allocate wide output");

    let small_launch = module
        .prepare_transform::<SmallTilePolicy>(LaunchConfig1D::new(
            1,
            SmallTilePolicy::MAX_THREADS,
            0,
        ))
        .expect("prepare SmallTilePolicy launch");
    let wide_launch = module
        .prepare_transform::<WideTilePolicy>(LaunchConfig1D::new(1, WideTilePolicy::MAX_THREADS, 0))
        .expect("prepare WideTilePolicy launch");

    // Geometry was checked against each policy's maximum above. SAFETY: the
    // raw pointers cover each policy's full block tile and stay valid until
    // the stream synchronizes. COUNT fits each thread's tile.
    unsafe {
        module
            .transform::<SmallTilePolicy>(
                &stream,
                &small_launch,
                input.cu_deviceptr() as *const u32,
                small_output.cu_deviceptr() as *mut u32,
                COUNT,
            )
            .expect("launch SmallTilePolicy");
        module
            .transform::<WideTilePolicy>(
                &stream,
                &wide_launch,
                input.cu_deviceptr() as *const u32,
                wide_output.cu_deviceptr() as *mut u32,
                COUNT,
            )
            .expect("launch WideTilePolicy");
    }

    let small = small_output
        .to_host_vec(&stream)
        .expect("copy small output");
    let wide = wide_output.to_host_vec(&stream).expect("copy wide output");
    for (output, threads, tag) in [
        (&small, SmallTilePolicy::MAX_THREADS, SmallTilePolicy::TAG),
        (&wide, WideTilePolicy::MAX_THREADS, WideTilePolicy::TAG),
    ] {
        for thread in 0..threads as usize {
            let tile = &output[thread * 16..(thread + 1) * 16];
            assert!(tile[..COUNT as usize].iter().all(|value| *value == tag));
            assert!(tile[COUNT as usize..].iter().all(|value| *value == 0));
        }
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--launch") {
        launch_on_gpu();
        println!("PASS: both policy specializations ran on the GPU");
        return;
    }

    verify_generated_ptx();
    let [small_name, wide_name] = specialization_names();
    println!("small-tile policy: {small_name}");
    println!("wide-tile policy:  {wide_name}");
    println!("PASS: policy metadata and unroll factors produced two PTX specializations");
}
