/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogFile, ClusterBarrierOrdering, ClusterMemoryOperation, CpAsyncControlOperation,
    CpAsyncMbarrierOperation, CpAsyncSourceSize, ExecutionControlOperation, LdmatrixParticipation,
    LdmatrixShape, MbarrierBasicOperation, MbarrierExtendedAdapter, RegisterMmaAdapter,
    RegisterMmaOverflow, SparseMmaOverflow, Tcgen05LdShape, Tcgen05Operation, TmaAdapter,
    WarpShuffleValueKind, WgmmaControlMode,
};
use crate::render::common::{hardware_target_label, rust_header, source_label};
use crate::render::families::{
    ClcSafetyArgNames, is_blackwell_ldmatrix, render_clc_safety_lines, sparse_mma_fragment_counts,
    sparse_mma_metadata_rule, sparse_mma_selector_description, stmatrix_variant, tcgen05_is_commit,
    tcgen05_is_multicast_commit, tcgen05_is_shift, tcgen05_participation_doc,
};
use crate::render::reference::modules;
use anyhow::Result;
use std::fmt::Write as _;

pub(super) fn render_raw_mod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    let abi_module = format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi);
    writeln!(
        output,
        "#[doc(hidden)]\n#[path = \"abi_v{}.rs\"]\npub mod {abi_module};\n",
        catalog.intrinsic_abi
    )
    .unwrap();
    for module in modules(catalog) {
        writeln!(
            output,
            "/// Generated `{module}` intrinsic source API.\npub mod {module} {{"
        )
        .unwrap();
        for record in catalog
            .intrinsics
            .iter()
            .filter(|record| record.rust.module == module)
        {
            writeln!(
                output,
                "    pub use crate::{abi_module}::{} as {};",
                record.rust.abi_id, record.rust.name
            )
            .unwrap();
        }
        output.push_str("}\n\n");
    }
    output.push_str("#[cfg(test)]\n#[allow(clippy::type_complexity)]\nmod tests {\n");
    for record in &catalog.intrinsics {
        let arguments = record.rust.arguments.join(", ");
        writeln!(
            output,
            "    #[test]\n    fn public_{}_reexports_abi_{}() {{",
            record.rust.name, record.rust.abi_id
        )
        .unwrap();
        writeln!(
            output,
            "        let public: {}fn({}) -> {} = super::{}::{};",
            if record.rust.safe { "" } else { "unsafe " },
            arguments,
            record.rust.result,
            record.rust.module,
            record.rust.name
        )
        .unwrap();
        writeln!(
            output,
            "        let canonical: {}fn({}) -> {} = super::{abi_module}::{};",
            if record.rust.safe { "" } else { "unsafe " },
            arguments,
            record.rust.result,
            record.rust.abi_id
        )
        .unwrap();
        output.push_str("        assert_eq!(public as usize, canonical as usize);\n    }\n");
    }
    output.push_str("}\n");
    output
}

pub(super) fn render_raw_abi(catalog: &CatalogFile, hash: &str) -> Result<String> {
    let mut output = rust_header(catalog, hash);
    output.push_str("//! Raw ABI functions recognized by cuda-oxide.\n\n");
    for record in &catalog.intrinsics {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "///").unwrap();
        writeln!(
            output,
            "/// Catalog ID: `{}`. Source: {}; expects PTX `{}`.",
            record.id,
            source_label(record),
            record.expected_ptx
        )
        .unwrap();
        if is_blackwell_ldmatrix(record) {
            writeln!(
                output,
                "/// Available on `{}` targets. Instruction floor PTX {}; the selected target may require a newer PTX version.",
                hardware_target_label(&record.target.hardware),
                record.target.minimum_ptx
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "/// Available on `{}` targets from PTX {}.",
                hardware_target_label(&record.target.hardware),
                record.target.minimum_ptx
            )
            .unwrap();
        }
        if let Some(operation) = record.tcgen05.as_ref().map(|tcgen05| tcgen05.operation)
            && let Some(participation) = tcgen05_participation_doc(operation)
        {
            writeln!(output, "/// {participation}").unwrap();
        }
        if record.tcgen05.is_some() {
            output.push_str(
                "/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n",
            );
        }
        if let Some(reason) = &record.rust.safe_allowlist_reason {
            writeln!(output, "/// Safe because {reason}").unwrap();
        }
        if !record.rust.safe {
            output.push_str("///\n/// # Safety\n");
            if let Some(ldmatrix) = &record.ldmatrix {
                let variant = &ldmatrix.variant;
                let contributing_lanes = variant.multiplicity.register_count() * 8;
                let readable_bytes = if variant.shape == LdmatrixShape::M16n16 {
                    32
                } else {
                    16
                };
                let participation = match ldmatrix.safety.participation {
                    LdmatrixParticipation::AllWarpLanesSameInstruction => {
                        "All 32 warp lanes must execute the same instruction."
                    }
                    LdmatrixParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes => {
                        "All 32 warp lanes must execute the same instruction and qualifiers; no lane may have exited."
                    }
                };
                writeln!(
                    output,
                    "/// {participation} PTX maps {contributing_lanes} lane-provided row addresses for this {:?} variant; addresses may alias, but each used address must be 16-byte aligned and have {readable_bytes} readable shared-memory bytes.",
                    variant.multiplicity
                )
                .unwrap();
                if matches!(
                    ldmatrix.safety.address_contract,
                    crate::model::LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication
                ) && contributing_lanes < 32
                {
                    output.push_str("/// For portable sm_75 behavior, otherwise-unused lanes must also carry valid addresses replicated from the contributing lanes.\n");
                }
                output.push_str(
                    "/// This weak memory operation does not replace a required barrier or fence.\n",
                );
            } else if let Some((multiplicity, _)) = stmatrix_variant(record) {
                let address_lanes = multiplicity.register_count() * 8;
                writeln!(
                    output,
                    "/// All 32 warp lanes must execute the same instruction. The first {address_lanes} lanes must provide valid, 16-byte-aligned shared-memory row addresses."
                )
                .unwrap();
                output.push_str(
                    "/// Register operands must contain the packed b16 fragments required by the selected layout.\n/// This weak memory operation does not replace a required barrier or fence.\n",
                );
            } else if let Some(mma) = &record.sparse_mma {
                let (c_count, a_count, b_count, _) = sparse_mma_fragment_counts(record);
                output.push_str(
                    "/// All 32 warp lanes must execute the same sparse MMA instruction with the same qualifiers, and no lane may have exited.\n",
                );
                writeln!(
                    output,
                    "/// `_arg0`, `_arg1`, and `_arg2` hold this lane's {c_count}-register C, {a_count}-register A, and {b_count}-register B fragments; `_arg3` holds its sparse metadata."
                )
                .unwrap();
                writeln!(
                    output,
                    "/// `_arg4` must be {}.",
                    sparse_mma_selector_description(record)
                )
                .unwrap();
                output.push_str("/// The operation is register-only and is not a memory fence.\n");
                writeln!(output, "/// {}", sparse_mma_metadata_rule(mma)).unwrap();
                writeln!(
                    output,
                    "/// See the [PTX sparse MMA fragment layouts]({}).",
                    record.target.ptx_isa_url
                )
                .unwrap();
                match mma.overflow {
                    SparseMmaOverflow::NotApplicable => {}
                    SparseMmaOverflow::Wrapping => {
                        output.push_str(
                            "/// Signed accumulator overflow wraps because this form omits `.satfinite`.\n",
                        );
                    }
                    SparseMmaOverflow::Satfinite => output.push_str(
                        "/// Signed accumulator overflow clamps to the finite `i32` range.\n",
                    ),
                }
            } else if let Some(mma) = &record.register_mma {
                output.push_str(
                    "/// All 32 warp lanes must execute the same `mma.sync.aligned` instruction with the same qualifiers, and no lane may have exited.\n\
                     /// `_arg0`, `_arg1`, and `_arg2` must contain this lane's C, A, and B fragments in the documented PTX layout.\n\
                     /// The operation is register-only and is not a memory fence.\n",
                );
                if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
                {
                    output.push_str(
                        "/// `_arg3` and `_arg6` contain this lane's packed A and B scale data. `_arg4`/`_arg5` and `_arg7`/`_arg8` are the corresponding byte/thread selectors.\n\
                         /// For `scale_vec::1X`, A and B byte selectors must be in `0..=3`, the A thread selector in `0..=1`, and the B thread selector in `0..=3`; other values make the PTX operation undefined.\n",
                    );
                }
                writeln!(
                    output,
                    "/// See the [PTX MMA fragment layouts]({}).",
                    record.target.ptx_isa_url
                )
                .unwrap();
                if mma.overflow == RegisterMmaOverflow::Wrapping {
                    output.push_str("/// Signed accumulator overflow wraps because this form omits `.satfinite`.\n");
                } else if mma.overflow == RegisterMmaOverflow::Satfinite {
                    output.push_str(
                        "/// Signed accumulator overflow clamps to the finite `i32` range.\n",
                    );
                }
            } else if record.movmatrix.is_some() {
                output.push_str(
                    "/// All 32 warp lanes must execute the same instruction, and no lane may have exited.\n\
                     /// `_arg0` must contain this lane's two packed b16 input elements.\n\
                     /// The operation is register-only and is not a memory fence.\n",
                );
            } else if record.redux.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `redux.sync` operation with the same qualifiers and mask.\n\
                     /// The instruction waits for those lanes; violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.warp_barrier.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `bar.warp.sync` operation with the same mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute the barrier in convergence, and no lane outside `mask` may be active when it executes.\n\
                     /// The barrier orders memory accesses among participating lanes; violating the participation contract makes the PTX operation undefined.\n",
                );
            } else if record.warp_shuffle.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `shfl.sync` operation with the same qualifiers and mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute in convergence, and no lane outside `mask` may be active.\n\
                     /// If the computed source lane is in range, it must be active and named in `mask`; otherwise the result is undefined. If PTX marks the computed source out of range, the calling lane's input is copied.\n",
                );
                if record
                    .warp_shuffle
                    .as_ref()
                    .is_some_and(|shuffle| shuffle.value_kind == WarpShuffleValueKind::I64)
                {
                    output.push_str(
                        "/// The 64-bit value is moved by two `b32` shuffles in one convergent block.\n",
                    );
                }
            } else if record.warp_match.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `match.sync` operation with the same qualifiers and mask.\n\
                     /// Violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.family == "elect" {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `elect.sync` operation with the same mask.\n\
                     /// Violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.vote.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `vote.sync` operation with the same qualifiers and mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute in convergence, and no lane outside `mask` may be active.\n\
                     /// Violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.family == "sync" {
                output.push_str(
                    "/// Every active thread in the CTA must reach the same barrier. Calling it from divergent control flow can deadlock the CTA.\n",
                );
            } else if record.family == "cluster_barrier" {
                let barrier = record
                    .cluster_barrier
                    .as_ref()
                    .expect("cluster-barrier contract");
                output.push_str(
                    "/// Each non-exited cluster thread must arrive exactly once before the barrier completes, then execute the matching wait.\n",
                );
                if barrier.aligned {
                    output.push_str(
                        "/// Every non-exited thread in the warp must execute this aligned instruction with identical control flow.\n",
                    );
                }
                if barrier.ordering == ClusterBarrierOrdering::Relaxed {
                    output.push_str(
                        "/// This relaxed arrival does not publish earlier memory accesses; add the required cluster-scope fence before it.\n",
                    );
                }
            } else if let Some(cluster) = &record.cluster_memory {
                match cluster.operation {
                    ClusterMemoryOperation::MapSharedRank => output.push_str(
                        "/// `_arg0` must point into CTA shared memory, and `_arg1` must name a rank in the same live cluster.\n\
                         /// The result is a cluster-shared pointer in address space 7. Dereferencing it performs a remote DSMEM access; ordinary loads and stores compile to `ld.shared::cluster` and `st.shared::cluster`. The target CTA must remain live and synchronization must make the access race-free.\n",
                    ),
                    ClusterMemoryOperation::ReadU32 => output.push_str(
                        "/// `_arg0` must point to an aligned readable `u32` in CTA shared memory, and `_arg1` must name a rank in the same live cluster.\n\
                         /// The target CTA must publish the value with the required cluster synchronization before this weak load.\n",
                    ),
                }
            } else if let Some(control) = &record.wgmma_control {
                output.push_str(
                    "/// All four warps in the warpgroup must execute this instruction in convergence with the same control flow.\n",
                );
                match control.mode {
                    WgmmaControlMode::Fence => output.push_str(
                        "/// Issue it after register writes and before the WGMMA operations that consume those registers.\n",
                    ),
                    WgmmaControlMode::CommitGroup => output.push_str(
                        "/// Issue it only after the warpgroup has submitted the WGMMA operations to commit.\n",
                    ),
                    WgmmaControlMode::WaitGroup => output.push_str(
                        "/// `_arg0` must be a compile-time constant shared by the whole warpgroup.\n",
                    ),
                }
            } else if let Some(tma) = &record.tma {
                match tma.adapter {
                    TmaAdapter::G2sPointersCoordinatesBarrierInjectDefaults
                    | TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults => output
                        .push_str(
                            "/// The destination and barrier must be valid shared-memory objects, and the tensor map must be a live descriptor for this dimensionality.\n\
                             /// Keep every object alive until the asynchronous copy completes. Only the designated issuing thread may start this transfer.\n",
                        ),
                    TmaAdapter::S2gPointersCoordinatesInjectDefaults => output.push_str(
                        "/// The source must name a live shared-memory tile, and the tensor map must be a live descriptor for this dimensionality.\n\
                         /// Keep both objects alive until the committed bulk-copy group completes.\n",
                    ),
                    TmaAdapter::ReductionPointersCoordinatesInjectDefaults => output.push_str(
                        "/// The source must name a live shared-memory tile, and the tensor map must describe a compatible global tensor destination for this dimensionality.\n\
                         /// Keep both objects alive until the committed asynchronous reduction completes.\n",
                    ),
                    TmaAdapter::DescriptorPointer
                    | TmaAdapter::DescriptorCoordinatesInjectDefaults
                    | TmaAdapter::DescriptorCoordinatesCacheHintInjectFlag => output.push_str(
                        "/// The tensor-map pointer must name a live, correctly encoded descriptor for this operation.\n",
                    ),
                    TmaAdapter::DescriptorAndAddressPointers
                    | TmaAdapter::DescriptorOrdinalAndU32
                    | TmaAdapter::DescriptorOrdinalAndU64
                    | TmaAdapter::DescriptorAndImmediateU32
                    | TmaAdapter::DescriptorAndRuntimeU32 => output.push_str(
                        "/// The tensor-map pointer must name a writable, 128-byte descriptor in global memory, and every replacement value must satisfy the PTX field contract.\n",
                    ),
                    TmaAdapter::DescriptorPointerInjectBytes => output.push_str(
                        "/// The tensor-map pointer must name a live 128-byte descriptor covered by the matching generic-proxy release fence.\n",
                    ),
                    TmaAdapter::NoOperands | TmaAdapter::CompileTimeConstantMaxPending => {}
                }
            } else if let Some(operation) = ExecutionControlOperation::from_catalog_id(&record.id) {
                match operation {
                    ExecutionControlOperation::BarrierCtaSync
                    | ExecutionControlOperation::BarrierCtaSyncAligned
                    | ExecutionControlOperation::BarrierCtaArrive
                    | ExecutionControlOperation::BarrierCtaArriveAligned => output.push_str(
                        "/// `_arg0` must identify a CTA barrier and `_arg1` must be the compatible expected thread count used by every participant.\n",
                    ),
                    ExecutionControlOperation::GridDependencyLaunchDependents
                    | ExecutionControlOperation::GridDependencyWait => output.push_str(
                        "/// The kernel launch must participate in a valid programmatic dependent-launch protocol.\n",
                    ),
                    ExecutionControlOperation::SetMaxNRegInc
                    | ExecutionControlOperation::SetMaxNRegDec => output.push_str(
                        "/// `_arg0` must be a compile-time multiple of eight in `24..=256`, and every thread in the warpgroup must execute the same operation and count.\n",
                    ),
                }
            } else if let Some(bridge) = &record.cp_async_mbarrier {
                output.push_str(
                    "/// `_arg0` must point to a live, initialized, eight-byte-aligned mbarrier object in shared memory.\n\
                     /// The issuing thread must have prior `cp.async` operations, and the object must remain valid until they complete.\n",
                );
                match bridge.operation {
                    CpAsyncMbarrierOperation::Arrive => output.push_str(
                        "/// This instruction increments the pending count before scheduling the asynchronous arrival; that increment must not exceed the barrier's pending-count limit.\n",
                    ),
                    CpAsyncMbarrierOperation::ArriveNoInc => output.push_str(
                        "/// The barrier's initial pending count must already include the asynchronous arrival because this form does not increment it.\n",
                    ),
                }
            } else if let Some(mbarrier) = &record.mbarrier_basic {
                output.push_str(
                    "/// `_arg0` must point to a live, eight-byte-aligned mbarrier object in shared memory.\n",
                );
                match mbarrier.operation {
                    MbarrierBasicOperation::Init => output.push_str(
                        "/// Exactly one thread may initialize the object. `_arg1` must be in `1..=0xFFFFF` and include every arrival in the phase.\n\
                         /// The object must be uninitialized or invalidated, with no concurrent barrier operation.\n",
                    ),
                    MbarrierBasicOperation::Arrive => output.push_str(
                        "/// The object must be initialized, and this arrival must be included in the current phase's expected count.\n\
                         /// Use the returned token only with the same object and phase.\n",
                    ),
                    MbarrierBasicOperation::ArriveNoComplete => output.push_str(
                        "/// The object must be initialized. `_arg1` must be a valid PTX arrival count and this operation must not complete the current phase.\n\
                         /// Use the returned opaque state only with the same object and phase.\n",
                    ),
                    MbarrierBasicOperation::TestWait => output.push_str(
                        "/// The object must be initialized. `_arg1` must be a token returned for the same object and phase.\n",
                    ),
                    MbarrierBasicOperation::Inval => output.push_str(
                        "/// The object must be initialized and no thread or asynchronous operation may still use it.\n\
                         /// Exactly one thread may invalidate the object.\n",
                    ),
                }
            } else if let Some(mbarrier) = &record.mbarrier_extended {
                output.push_str(
                    "/// This is a convergent, side-effecting operation with a compiler memory clobber.\n",
                );
                match mbarrier.adapter {
                    MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount => {
                        output.push_str(
                            "/// `_arg0` must point to a live, initialized, eight-byte-aligned mbarrier in shared memory. `_arg2` must match the asynchronous transaction bytes.\n\
                             /// `_arg1` is retained for compatibility and is not a PTX operand.\n",
                        );
                    }
                    MbarrierExtendedAdapter::RawClusterAddressToVoid => output.push_str(
                        "/// `_arg0` must be a live remote shared-memory barrier address obtained through cluster address mapping.\n",
                    ),
                    MbarrierExtendedAdapter::PointerTokenToPredicate => output.push_str(
                        "/// `_arg0` must point to a live initialized barrier. `_arg1` must be a token for that barrier and phase.\n",
                    ),
                    MbarrierExtendedAdapter::PointerParityToPredicate => output.push_str(
                        "/// `_arg0` must point to a live initialized barrier. `_arg1` must be the expected phase parity.\n",
                    ),
                    MbarrierExtendedAdapter::ZeroOperandsToVoid => output.push_str(
                        "/// The call site must satisfy the scope and proxy-ordering contract named by the operation.\n",
                    ),
                    MbarrierExtendedAdapter::NanosecondsToVoid => output.push_str(
                        "/// `_arg0` is an approximate suspension duration; hardware may resume the thread earlier.\n",
                    ),
                }
            } else if let Some(copy) = &record.cp_async_copy {
                let bytes = copy.copy_size.bytes();
                writeln!(
                    output,
                    "/// `_arg0` must point to {bytes} writable bytes in shared memory and be aligned to {bytes} bytes."
                )
                .unwrap();
                if copy.source_size == CpAsyncSourceSize::Runtime {
                    writeln!(
                        output,
                        "/// `_arg2` must be at most {bytes}; `_arg1` must point to that many readable bytes in global memory and be aligned to {bytes} bytes."
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "/// `_arg1` must point to {bytes} readable bytes in global memory and be aligned to {bytes} bytes."
                    )
                    .unwrap();
                }
                output.push_str(
                    "/// Both ranges must remain valid, the source must remain unchanged, and the destination must not be accessed until this copy completes.\n\
                     /// The issuing thread must use a matching `cp.async` completion operation. Synchronize threads after completion before another thread accesses the destination.\n\
                     /// User-authored completion assembly must include a compiler memory clobber.\n",
                );
            } else if let Some(control) = &record.cp_async_control {
                if control.operation == CpAsyncControlOperation::WaitGroup {
                    output.push_str(
                        "/// `_arg0` must be a compile-time constant. Access only destinations whose copy groups this wait completes.\n",
                    );
                } else if control.operation == CpAsyncControlOperation::WaitAll {
                    output.push_str(
                        "/// This waits only for copies issued by the executing thread. Synchronize threads before another thread accesses a completed destination.\n",
                    );
                } else {
                    output.push_str(
                        "/// This commits only copies issued by the executing thread and does not wait for completion.\n",
                    );
                }
            } else if let Some(tcgen05) = &record.tcgen05 {
                if tcgen05_is_commit(tcgen05.operation) {
                    if tcgen05_is_multicast_commit(tcgen05.operation) {
                        output.push_str(
                            "/// `_arg0` must point to a live initialized cluster-shared mbarrier. `_arg1` must select valid CTA ranks in its cluster.\n",
                        );
                    } else {
                        output.push_str(
                            "/// `_arg0` must point to a live initialized mbarrier valid for this CTA-group mode.\n",
                        );
                    }
                    output.push_str(
                        "/// The same thread that issued the tracked asynchronous tcgen05 operations must issue this commit.\n",
                    );
                } else if tcgen05_is_shift(tcgen05.operation) {
                    output.push_str(
                        "/// `_arg0` must name a live tensor-memory allocation, and its lane component must be a multiple of 32.\n\
                         /// Completion must be tracked by a matching commit from that same thread and observed through the selected mbarrier before relying on shifted data.\n",
                    );
                } else if tcgen05.operation == Tcgen05Operation::Ld {
                    output.push_str(
                        "/// `_arg0` must name a live tensor-memory allocation covering the selected tile.\n\
                         /// All active warp lanes must execute convergently with the same address.\n\
                         /// Complete the matching tensor-memory load wait before consuming the returned registers.\n",
                    );
                    if tcgen05
                        .ld
                        .is_some_and(|ld| ld.shape == Tcgen05LdShape::M16x32bx2)
                    {
                        output.push_str(
                            "/// `_arg1` must be a compile-time constant; inline PTX encodes its low 32 bits.\n",
                        );
                    }
                } else if tcgen05.operation == Tcgen05Operation::St {
                    output.push_str(
                        "/// `_arg0` must name a live tensor-memory allocation covering the selected tile.\n\
                         /// All active warp lanes must execute convergently with the same address.\n\
                         /// Complete the matching tensor-memory store wait before relying on completion or reusing the affected storage.\n",
                    );
                    if tcgen05
                        .st
                        .is_some_and(|st| st.shape == Tcgen05LdShape::M16x32bx2)
                    {
                        output.push_str(
                            "/// `_arg1` must be a compile-time constant; inline PTX encodes its low 32 bits.\n",
                        );
                    }
                } else {
                    output.push_str(
                        "/// The caller must satisfy the tcgen05 address, lifetime, and participation rules.\n",
                    );
                }
            } else if record.packed_atomic.is_some() {
                output.push_str(
                    "/// `addr` must designate four writable bytes in global memory and be naturally aligned to four bytes.\n\
                     /// Do not overlap this operation with a whole-word atomic or any non-atomic access to either 16-bit lane.\n\
                     /// Racing atomics must use scopes that include each other; this relaxed GPU-scope operation is not atomic with host/system access.\n",
                );
            } else if let Some(clc) = &record.clc {
                render_clc_safety_lines(&mut output, clc.operation, ClcSafetyArgNames::RawAbi);
            } else {
                anyhow::bail!(
                    "{} ({}) is unsafe but has no dedicated family safety renderer",
                    record.id,
                    record.family
                );
            }
        }
        if record.rust.arguments.len() > 7 {
            output.push_str("#[allow(clippy::too_many_arguments)]\n");
        }
        if record.rust.must_use {
            output.push_str("#[must_use]\n");
        }
        output.push_str("#[inline(never)]\n");
        let safety = if record.rust.safe { "" } else { "unsafe " };
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("_arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "pub {safety}fn {}({arguments}) -> {} {{",
            record.rust.abi_id, record.rust.result,
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{}` executed outside device compilation\")",
            record.rust.canonical_path
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    Ok(output)
}
