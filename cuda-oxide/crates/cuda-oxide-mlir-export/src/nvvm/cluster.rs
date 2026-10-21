/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit cluster, mbarrier, and TMA mappings for the CUTLASS MLIR consumer.
//!
//! Native intrinsic markers remain part of the contract. Pointer conversions
//! preserve the native address spaces, and try-wait preserves its boolean
//! result rather than becoming the similarly named blocking NVVM operation.

use crate::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};
use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};

/// Register the native primitives used by persistent two-CTA TMA kernels.
pub fn register_nvvm_cluster_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    /// Register one source operation with its marker, selector recipe, and source verifier.
    macro_rules! register {
        ($op:ty, $marker:literal, $kind:expr) => {
            registry.register_operation::<$op>(Recipe {
                marker: $marker,
                kind: $kind,
                verify: |ctx, source| {
                    <$op>::new(source)
                        .verify(ctx)
                        .map_err(|error| error.to_string())
                },
            })?;
        };
    }
    use Kind::*;
    register!(TrapOp, "v1:i0295", NoOperands("llvm.intr.trap"));
    register!(
        ReadPtxSregClusterCtarankOp,
        "v1:i0275",
        Sreg("nvvm.read.ptx.sreg.cluster.ctarank")
    );
    register!(
        ReadPtxSregClusterNctarankOp,
        "v1:i0276",
        Sreg("nvvm.read.ptx.sreg.cluster.nctarank")
    );
    register!(
        ReadPtxSregClusterCtaidXOp,
        "v1:i0263",
        Sreg("nvvm.read.ptx.sreg.cluster.ctaid.x")
    );
    register!(
        ReadPtxSregClusterCtaidYOp,
        "v1:i0264",
        Sreg("nvvm.read.ptx.sreg.cluster.ctaid.y")
    );
    register!(
        ReadPtxSregClusterCtaidZOp,
        "v1:i0265",
        Sreg("nvvm.read.ptx.sreg.cluster.ctaid.z")
    );
    register!(
        ReadPtxSregClusterNctaidXOp,
        "v1:i0266",
        Sreg("nvvm.read.ptx.sreg.cluster.nctaid.x")
    );
    register!(
        ReadPtxSregClusterNctaidYOp,
        "v1:i0267",
        Sreg("nvvm.read.ptx.sreg.cluster.nctaid.y")
    );
    register!(
        ReadPtxSregClusterNctaidZOp,
        "v1:i0268",
        Sreg("nvvm.read.ptx.sreg.cluster.nctaid.z")
    );
    register!(
        ReadPtxSregClusterIdXOp,
        "v1:i0269",
        Sreg("nvvm.read.ptx.sreg.clusterid.x")
    );
    register!(
        ReadPtxSregClusterIdYOp,
        "v1:i0270",
        Sreg("nvvm.read.ptx.sreg.clusterid.y")
    );
    register!(
        ReadPtxSregClusterIdZOp,
        "v1:i0271",
        Sreg("nvvm.read.ptx.sreg.clusterid.z")
    );
    register!(
        ReadPtxSregNclusterIdXOp,
        "v1:i0272",
        Sreg("nvvm.read.ptx.sreg.nclusterid.x")
    );
    register!(
        ReadPtxSregNclusterIdYOp,
        "v1:i0273",
        Sreg("nvvm.read.ptx.sreg.nclusterid.y")
    );
    register!(
        ReadPtxSregNclusterIdZOp,
        "v1:i0274",
        Sreg("nvvm.read.ptx.sreg.nclusterid.z")
    );
    registry.register_attribute::<ClusterBarrierModeAttr>(DropAttribute)?;
    registry.register_operation::<ClusterBarrierOp>(ClusterBarrierTranslation)?;
    register!(
        FenceMbarrierInitReleaseClusterOp,
        "v1:i0313",
        NoOperands("nvvm.fence.mbarrier.init")
    );
    register!(FenceProxyAsyncSharedCtaOp, "v1:i0312", AsyncFence);
    register!(
        FenceAcquireSharedClusterClusterOp,
        "v1:i1042",
        SharedAcquireFence
    );
    register!(
        BarrierCtaSyncCountOp,
        "v1:i0883",
        CountedBarrier {
            arrive: false,
            aligned: false
        }
    );
    register!(
        BarrierCtaSyncAlignedCountOp,
        "v1:i0884",
        CountedBarrier {
            arrive: false,
            aligned: true
        }
    );
    register!(
        BarrierCtaArriveCountOp,
        "v1:i0885",
        CountedBarrier {
            arrive: true,
            aligned: false
        }
    );
    register!(
        BarrierCtaArriveAlignedCountOp,
        "v1:i0886",
        CountedBarrier {
            arrive: true,
            aligned: true
        }
    );
    register!(MbarrierInitSharedOp, "v1:i0097", BarrierInit);
    register!(MbarrierInvalSharedOp, "v1:i0100", BarrierInval);
    register!(MbarrierArriveSharedOp, "v1:i0098", BarrierArrive);
    register!(
        MbarrierArriveExpectTxSharedOp,
        "v1:i0306",
        ExpectTx { cluster: false }
    );
    register!(
        MbarrierArriveExpectTxClusterOp,
        "v1:i0307",
        ExpectTx { cluster: true }
    );
    register!(
        MbarrierTryWaitParitySharedOp,
        "v1:i0310",
        WaitParity {
            cluster: false,
            relaxed: false
        }
    );
    register!(
        MbarrierTryWaitParityClusterOp,
        "v1:i0311",
        WaitParity {
            cluster: true,
            relaxed: false
        }
    );
    register!(
        MbarrierArriveClusterOp,
        "v1:i0308",
        RemoteArrive { relaxed: false }
    );
    register!(
        MbarrierArriveClusterRelaxedOp,
        "v1:i1040",
        RemoteArrive { relaxed: true }
    );
    register!(
        MbarrierTryWaitParityClusterRelaxedOp,
        "v1:i1041",
        WaitParity {
            cluster: true,
            relaxed: true
        }
    );
    register!(MapaSharedClusterOp, "v1:i0320", MapShared);
    register!(ElectSyncOp, "v1:i0367", Elect);
    register!(PrefetchTensorMapOp, "v1:i0887", Prefetch);
    register!(
        CpAsyncBulkTensorG2sTile2dMulticastCg2Op,
        "v1:i0331",
        TmaLoadCg2
    );
    register!(CpAsyncBulkTensorS2gTile2dOp, "v1:i0336", TmaStore);
    register!(
        CpAsyncBulkCommitGroupOp,
        "v1:i0340",
        NoOperands("nvvm.cp.async.bulk.commit.group")
    );
    register!(
        CpAsyncBulkWaitGroupOp,
        "v1:i0341",
        WaitGroup { read: false }
    );
    register!(
        CpAsyncBulkWaitGroupReadOp,
        "v1:i0342",
        WaitGroup { read: true }
    );
    Ok(())
}

/// Select one reviewed cluster instruction contract and its structural selectors.
enum Kind {
    /// Read the named cluster special register.
    Sreg(#[doc = "Target NVVM operation spelling for the selected cluster register."] &'static str),
    /// Emit the named result-free synchronization instruction.
    NoOperands(#[doc = "Target NVVM synchronization operation spelling."] &'static str),
    /// Publish generic shared accesses to the asynchronous proxy.
    AsyncFence,
    /// Acquire ordering restricted to cluster shared-memory accesses.
    SharedAcquireFence,
    /// Emit arrival or synchronization on a named CTA barrier with explicit participation.
    CountedBarrier {
        #[doc = "Emit nonblocking arrival instead of blocking synchronization."]
        arrive: bool,
        #[doc = "Preserve the source aligned-participation instruction variant."]
        aligned: bool,
    },
    /// Initialize an mbarrier with its expected arrival count.
    BarrierInit,
    /// Invalidate an mbarrier without adding synchronization.
    BarrierInval,
    /// Arrive on a local mbarrier and preserve its result contract.
    BarrierArrive,
    /// Announce asynchronous transaction bytes with the selected barrier scope.
    ExpectTx {
        #[doc = "Use cluster scope instead of CTA scope."]
        cluster: bool,
    },
    /// Probe barrier phase completion and preserve its boolean predicate result.
    WaitParity {
        #[doc = "Use cluster scope instead of CTA scope."]
        cluster: bool,
        #[doc = "Select relaxed memory ordering rather than acquire/release ordering."]
        relaxed: bool,
    },
    /// Arrive on a cluster-mapped barrier using the selected memory ordering.
    RemoteArrive {
        #[doc = "Select relaxed memory ordering rather than acquire/release ordering."]
        relaxed: bool,
    },
    /// Preserve CTA rank when mapping a shared address across the cluster.
    MapShared,
    /// Return both the elected lane and its participation predicate.
    Elect,
    /// Prefetch the tensor-map descriptor without changing its address space contract.
    Prefetch,
    /// Issue a two-CTA tensor load with remote-barrier completion routing.
    TmaLoadCg2,
    /// Issue shared-to-global tensor stores in NVVM operand order.
    TmaStore,
    /// Wait for a legal compile-time number of pending bulk groups.
    WaitGroup {
        #[doc = "Wait only until shared source reads complete when set."]
        read: bool,
    },
}

/// Combine intrinsic identity, source verification, and one concrete cluster lowering.
struct Recipe {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// Concrete cluster instruction and selector contract for this recipe.
    kind: Kind,
    /// Check the source operation's selectors and operand contract.
    verify: fn(&Context, Ptr<Operation>) -> Result<(), String>,
}

impl OperationTranslation for Recipe {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        check_marker(&mut input, self.marker)?;
        let location = &input.location;
        let mut operations = vec![];
        use TypeSpec::*;
        let mut target = match self.kind {
            Kind::Sreg(name) => {
                signature(&input, &[], &[Int(32)])?;
                operation(name, input.results, input.operands, location)?
            }
            Kind::NoOperands(name) => {
                signature(&input, &[], &[])?;
                operation(name, vec![], vec![], location)?
            }
            Kind::SharedAcquireFence => {
                signature(&input, &[], &[])?;
                let mut op = operation("nvvm.fence.sync_restrict", vec![], vec![], location)?;
                dialect_property(&mut op, "order", "#nvvm.mem_order<acquire>")?;
                op
            }
            Kind::AsyncFence => {
                signature(&input, &[], &[])?;
                let mut op = operation("nvvm.fence.proxy", vec![], vec![], location)?;
                dialect_property(&mut op, "kind", "#nvvm.proxy_kind<async.shared>")?;
                dialect_property(&mut op, "space", "#nvvm.shared_space<cta>")?;
                op
            }
            Kind::CountedBarrier { arrive, aligned } => {
                signature(&input, &[Int(32), Int(32)], &[])?;
                let name = if arrive {
                    "nvvm.barrier.cta.arrive"
                } else {
                    "nvvm.barrier.cta.sync"
                };
                let mut op = operation(name, vec![], input.operands, location)?;
                op.properties
                    .insert("aligned".into(), MlirAttribute::Bool(aligned));
                op
            }
            Kind::BarrierInit => {
                signature(&input, &[Pointer(&[0, 3]), Int(32)], &[])?;
                operation("nvvm.mbarrier.init", vec![], input.operands, location)?
            }
            Kind::BarrierInval => {
                signature(&input, &[Pointer(&[0, 3])], &[])?;
                operation("nvvm.mbarrier.inval", vec![], input.operands, location)?
            }
            Kind::BarrierArrive => {
                signature(&input, &[Pointer(&[0, 3])], &[Int(64)])?;
                operation(
                    "nvvm.mbarrier.arrive",
                    input.results,
                    input.operands,
                    location,
                )?
            }
            Kind::ExpectTx { cluster } => {
                // The native semantic op has already discarded the compatibility
                // tx_count argument. PTX's arrival decrement is always one.
                signature(&input, &[Pointer(&[0, 3]), Int(32)], &[Int(64)])?;
                let mut op = operation(
                    "nvvm.mbarrier.arrive.expect_tx",
                    input.results,
                    input.operands,
                    location,
                )?;
                if cluster {
                    dialect_property(&mut op, "scope", "#nvvm.mem_scope<cluster>")?;
                    op.properties
                        .insert("relaxed".into(), MlirAttribute::Bool(true));
                }
                op
            }
            Kind::WaitParity { cluster, relaxed } => {
                signature(&input, &[Pointer(&[0, 3]), Int(32)], &[Int(1)])?;
                let mut op = operation(
                    "nvvm.mbarrier.wait.parity",
                    input.results,
                    input.operands,
                    location,
                )?;
                dialect_property(&mut op, "kind", "#nvvm.mbar_wait<try>")?;
                if cluster {
                    dialect_property(&mut op, "scope", "#nvvm.mbar_scope<cluster>")?;
                    dialect_property(
                        &mut op,
                        "order",
                        if relaxed {
                            "#nvvm.mem_order<relaxed>"
                        } else {
                            "#nvvm.mem_order<acquire>"
                        },
                    )?;
                }
                op
            }
            Kind::RemoteArrive { relaxed } => {
                signature(&input, &[Int(64)], &[])?;
                let pointer = cast(
                    session,
                    &mut operations,
                    "llvm.inttoptr",
                    input.operands[0].clone(),
                    pointer_type(7)?,
                    location,
                )?;
                let mut op = operation("nvvm.mbarrier.arrive", vec![], vec![pointer], location)?;
                dialect_property(&mut op, "scope", "#nvvm.mem_scope<cluster>")?;
                if relaxed {
                    op.properties
                        .insert("relaxed".into(), MlirAttribute::Bool(true));
                }
                op
            }
            Kind::MapShared => {
                signature(&input, &[Pointer(&[0, 3]), Int(32)], &[Pointer(&[7])])?;
                let shared = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    3,
                    location,
                )?;
                operation(
                    "nvvm.mapa",
                    input.results,
                    vec![shared, input.operands[1].clone()],
                    location,
                )?
            }
            Kind::Elect => {
                signature(&input, &[Int(32)], &[Int(32), Int(1)])?;
                // NVVM dialect elect.sync only exposes the predicate. The LLVM
                // intrinsic preserves both native results, including lane ID.
                let ty = MlirType::dialect("!llvm.struct<(i32, i1)>")?;
                let (result, value) = fresh(session, ty);
                let mut call = operation(
                    "llvm.call_intrinsic",
                    vec![result],
                    input.operands,
                    location,
                )?;
                call.properties.insert(
                    "intrin".into(),
                    MlirAttribute::String("llvm.nvvm.elect.sync".into()),
                );
                call.properties.insert(
                    "op_bundle_sizes".into(),
                    MlirAttribute::DenseI32Array(vec![]),
                );
                call.properties.insert(
                    "operandSegmentSizes".into(),
                    MlirAttribute::DenseI32Array(vec![1, 0]),
                );
                operations.push(call);
                for (index, result) in input.results.into_iter().enumerate() {
                    let mut extract = operation(
                        "llvm.extractvalue",
                        vec![result],
                        vec![value.clone()],
                        location,
                    )?;
                    extract.properties.insert(
                        "position".into(),
                        MlirAttribute::DenseI64Array(vec![index as i64]),
                    );
                    operations.push(extract);
                }
                return Ok(operations);
            }
            Kind::Prefetch => {
                signature(&input, &[Pointer(&[0, 1, 4])], &[])?;
                let descriptor = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    0,
                    location,
                )?;
                let mut op = operation("nvvm.prefetch", vec![], vec![descriptor], location)?;
                op.properties
                    .insert("tensormap".into(), MlirAttribute::Unit);
                op
            }
            Kind::TmaLoadCg2 => {
                // Importer order follows LLVM: dst, barrier, descriptor, x, y,
                // multicast mask, zero cache hint. This differs from the API.
                signature(
                    &input,
                    &[
                        Pointer(&[0, 3, 7]),
                        Pointer(&[0, 3, 7]),
                        Pointer(&[0, 1, 4]),
                        Int(32),
                        Int(32),
                        Int(16),
                        Int(64),
                    ],
                    &[],
                )?;
                if constant_integer(ctx, source, 6)? != 0 {
                    return Err(
                        "cg2 TMA native mapping requires the importer's zero cache hint".into(),
                    );
                }
                // Cluster TMA's LLVM intrinsic requires addrspace(7), including
                // when the Rust API began with the local CTA shared pointer.
                let dst = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    7,
                    location,
                )?;
                let descriptor = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[2].clone(),
                    0,
                    location,
                )?;
                let barrier = if input.operands[1].ty == pointer_type(7)? {
                    // mapa(rank=0) carries a cluster-space address. The CG2
                    // intrinsic encodes its barrier as a shared-space pointer;
                    // retype the raw address without changing its rank bits.
                    let address = cast(
                        session,
                        &mut operations,
                        "llvm.ptrtoint",
                        input.operands[1].clone(),
                        MlirType::Integer(32),
                        location,
                    )?;
                    cast(
                        session,
                        &mut operations,
                        "llvm.inttoptr",
                        address,
                        pointer_type(3)?,
                        location,
                    )?
                } else {
                    pointer_cast(
                        session,
                        &mut operations,
                        input.operands[1].clone(),
                        3,
                        location,
                    )?
                };
                let mut op = operation(
                    "nvvm.cp.async.bulk.tensor.shared.cluster.global",
                    vec![],
                    vec![
                        dst,
                        descriptor,
                        input.operands[3].clone(),
                        input.operands[4].clone(),
                        barrier,
                        input.operands[5].clone(),
                    ],
                    location,
                )?;
                op.properties.insert(
                    "operandSegmentSizes".into(),
                    MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 1, 0, 0]),
                );
                dialect_property(&mut op, "group", "#nvvm.cta_group<cta_2>")?;
                // In pinned CUTLASS 4.7, false selects LLVM's g2s.tile intrinsic
                // (addrspace 7). True selects the older CUDA gmem.to.smem
                // intrinsic, whose destination contract is addrspace 3.
                op.properties
                    .insert("useIntrinsic".into(), MlirAttribute::Bool(false));
                op
            }
            Kind::TmaStore => {
                signature(
                    &input,
                    &[Pointer(&[0, 3]), Pointer(&[0, 1, 4]), Int(32), Int(32)],
                    &[],
                )?;
                let src = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    3,
                    location,
                )?;
                let descriptor = pointer_cast(
                    session,
                    &mut operations,
                    input.operands[1].clone(),
                    0,
                    location,
                )?;
                // Native API is source-first; NVVM's operation is descriptor-first.
                let mut op = operation(
                    "nvvm.cp.async.bulk.tensor.global.shared.cta",
                    vec![],
                    vec![
                        descriptor,
                        src,
                        input.operands[2].clone(),
                        input.operands[3].clone(),
                    ],
                    location,
                )?;
                op.properties.insert(
                    "operandSegmentSizes".into(),
                    MlirAttribute::DenseI32Array(vec![1, 1, 2, 0, 0]),
                );
                op
            }
            Kind::WaitGroup { read } => {
                signature(&input, &[Int(32)], &[])?;
                let group = constant_group(ctx, source)?;
                let mut op = operation("nvvm.cp.async.bulk.wait_group", vec![], vec![], location)?;
                op.properties.insert(
                    "group".into(),
                    MlirAttribute::Integer {
                        value: group.into(),
                        ty: MlirType::Integer(32),
                    },
                );
                if read {
                    op.properties.insert("read".into(), MlirAttribute::Unit);
                }
                op
            }
        };
        target.location = input.location;
        operations.push(target);
        Ok(operations)
    }
}

/// Preserve source cluster-barrier mode and native intrinsic identity.
struct ClusterBarrierTranslation;

impl OperationTranslation for ClusterBarrierTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let op = ClusterBarrierOp::new(source);
        op.verify(ctx).map_err(|error| error.to_string())?;
        let mode = op
            .get_attr_nvvm_cluster_barrier_mode(ctx)
            .ok_or("missing cluster barrier mode")?;
        use ClusterBarrierModeAttr::*;
        let (marker, name, aligned) = match *mode {
            Arrive => ("v1:i0277", "nvvm.cluster.arrive", false),
            ArriveAligned => ("v1:i0278", "nvvm.cluster.arrive", true),
            ArriveRelaxed => ("v1:i0279", "nvvm.cluster.arrive.relaxed", false),
            ArriveRelaxedAligned => ("v1:i0280", "nvvm.cluster.arrive.relaxed", true),
            Wait => ("v1:i0281", "nvvm.cluster.wait", false),
            WaitAligned => ("v1:i0282", "nvvm.cluster.wait", true),
        };
        check_marker(&mut input, marker)?;
        signature(&input, &[], &[])?;
        let mut target = operation(name, vec![], vec![], &input.location)?;
        if aligned {
            target
                .properties
                .insert("aligned".into(), MlirAttribute::Unit);
        }
        Ok(vec![target])
    }
}

/// Consume and validate the canonical intrinsic marker before target emission.
fn check_marker(input: &mut OperationInput, expected: &str) -> Result<(), String> {
    let marker = input.attributes.remove("cuda_oxide_intrinsic_marker");
    if marker != Some(MlirAttribute::String(expected.into())) {
        return Err(format!(
            "expected intrinsic marker {expected}, got {marker:?}"
        ));
    }
    if !input.attributes.is_empty() || !input.regions.is_empty() || !input.successors.is_empty() {
        return Err(
            "native NVVM primitive has unexpected attributes, regions, or successors".into(),
        );
    }
    Ok(())
}

/// Operand/result type constraint used by explicit NVVM signatures.
#[derive(Clone, Copy, Debug)]
enum TypeSpec {
    /// An integer scalar with the required bit width.
    Int(#[doc = "Required integer width in bits."] u32),
    /// An opaque pointer in one of the accepted numeric address spaces.
    Pointer(#[doc = "Accepted numeric LLVM pointer address spaces."] &'static [u32]),
}

impl TypeSpec {
    /// Check a translated operand type against this signature constraint.
    fn matches(self, ty: &MlirType) -> bool {
        match self {
            Self::Int(bits) => *ty == MlirType::Integer(bits),
            Self::Pointer(spaces) => spaces
                .iter()
                .any(|space| pointer_type(*space).as_ref() == Ok(ty)),
        }
    }
}

/// Check operand/result counts and each translated type against the reviewed signature.
fn signature(
    input: &OperationInput,
    operands: &[TypeSpec],
    results: &[TypeSpec],
) -> Result<(), String> {
    if input.operands.len() != operands.len() || input.results.len() != results.len() {
        return Err(format!(
            "expected {} operands and {} results, got {} and {}",
            operands.len(),
            results.len(),
            input.operands.len(),
            input.results.len()
        ));
    }
    for (index, (expected, actual)) in operands.iter().zip(&input.operands).enumerate() {
        if !expected.matches(&actual.ty) {
            return Err(format!(
                "operand {index} expected {expected:?}, got {:?}",
                actual.ty
            ));
        }
    }
    for (index, (expected, actual)) in results.iter().zip(&input.results).enumerate() {
        if !expected.matches(&actual.ty) {
            return Err(format!(
                "result {index} expected {expected:?}, got {:?}",
                actual.ty
            ));
        }
    }
    Ok(())
}

/// Require a legal compile-time bulk wait-group count from the source operand.
fn constant_group(ctx: &Context, source: Ptr<Operation>) -> Result<u32, String> {
    let parsed = constant_integer(ctx, source, 0)
        .map_err(|_| "bulk wait group must be a constant in 0..=7")?;
    if parsed > 7 {
        return Err("bulk wait group must be a constant in 0..=7".into());
    }
    Ok(parsed as u32)
}

/// Read an unsigned source integer constant or reject runtime/noninteger values.
fn constant_integer(ctx: &Context, source: Ptr<Operation>, index: usize) -> Result<u64, String> {
    let operand = source.deref(ctx).get_operand(index);
    let definition = operand
        .defining_op()
        .ok_or("expected a constant integer operand")?;
    if Operation::get_opid(definition, ctx) != MirConstantOp::get_opid_static() {
        return Err("expected a constant integer operand".into());
    }
    let value = MirConstantOp::new(definition)
        .get_attr_value(ctx)
        .ok_or("constant has no value")?;
    value
        .value()
        .to_string_decimal(false)
        .parse::<u64>()
        .map_err(|_| "constant integer exceeds u64".into())
}

/// Spell an opaque LLVM pointer for a required NVVM address space.
fn pointer_type(space: u32) -> Result<MlirType, String> {
    MlirType::dialect(if space == 0 {
        "!llvm.ptr".into()
    } else {
        format!("!llvm.ptr<{space}>")
    })
}

/// Build a target operation with the supplied operands, results, and source location.
fn operation(
    name: &str,
    results: Vec<MlirResult>,
    operands: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut op = MlirOperation::new(name)?;
    op.results = results;
    op.operands = operands;
    op.location = location.clone();
    Ok(op)
}

/// Attach a parsed dialect attribute as a target operation property.
fn dialect_property(op: &mut MlirOperation, name: &str, value: &str) -> Result<(), String> {
    op.properties
        .insert(name.into(), MlirAttribute::dialect(value)?);
    Ok(())
}

/// Allocate one SSA identifier and return matching result and operand handles.
fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

/// Append a one-operand conversion with a fresh result of the requested type.
fn cast(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    name: &str,
    operand: MlirValueUse,
    ty: MlirType,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, value) = fresh(session, ty);
    operations.push(operation(name, vec![result], vec![operand], location)?);
    Ok(value)
}

/// Convert pointer space explicitly while retaining any required integer-address bridge.
fn pointer_cast(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    operand: MlirValueUse,
    space: u32,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let ty = pointer_type(space)?;
    if operand.ty == ty {
        Ok(operand)
    } else {
        cast(
            session,
            operations,
            "llvm.addrspacecast",
            operand,
            ty,
            location,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Contract and rejection tests for native cluster, barrier, and TMA mappings.

    use super::*;
    use crate::{
        CutlassFullCuteMlir22, MlirConsumerProfile,
        profile::render_mapping_module_without_cutlass_envelope,
    };
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::MirPtrType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{IntegerAttr, StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        identifier::Identifier,
        r#type::{TypeHandle, TypedHandle},
        utils::apint::APInt,
        value::Value,
    };
    use std::num::NonZeroUsize;

    /// Typed source function exposing descriptor, shared-address, parity, and coordinate test operands.
    struct Fixture {
        /// Source IR context owning the fixture's registered types and operations.
        ctx: Context,
        /// Source module containing the operations under test.
        module: ModuleOp,
        /// Entry block where the fixture appends native source instructions.
        entry: Ptr<BasicBlock>,
        /// Argument carrier types used to construct well-typed source instructions.
        types: Vec<TypeHandle>,
        /// Unsigned 32-bit type used for narrow immediate operands.
        u32: TypedHandle<IntegerType>,
        /// Unsigned 64-bit type used for wide immediate operands.
        u64: TypedHandle<IntegerType>,
    }

    impl Fixture {
        /// Build the canonical cluster test function with distinct address-space and scalar arguments.
        fn new() -> Self {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            dialect_nvvm::register(&mut ctx);
            let u8 = IntegerType::get(&ctx, 8, Signedness::Unsigned);
            let u32 = IntegerType::get(&ctx, 32, Signedness::Unsigned);
            let u64 = IntegerType::get(&ctx, 64, Signedness::Unsigned);
            // Arguments: descriptor, shared, remote shared, parity, coordinate,
            // multicast mask, raw remote address, predicate.
            let types = vec![
                MirPtrType::get(&mut ctx, u8.into(), true, 0).into(),
                MirPtrType::get(&mut ctx, u8.into(), true, 3).into(),
                MirPtrType::get(&mut ctx, u8.into(), true, 7).into(),
                u32.into(),
                IntegerType::get(&ctx, 32, Signedness::Signed).into(),
                IntegerType::get(&ctx, 16, Signedness::Unsigned).into(),
                u64.into(),
                IntegerType::get(&ctx, 1, Signedness::Signless).into(),
            ];
            let module = ModuleOp::new(&mut ctx, Identifier::try_from("cluster_test").unwrap());
            let function_type = FunctionType::get(&ctx, types.clone(), vec![]);
            let function_op = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let function =
                MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
            function.set_symbol_name(&mut ctx, Identifier::try_from("kernel_primitives").unwrap());
            module.append_operation(&mut ctx, function_op, 0);
            let region = function_op.deref(&ctx).get_region(0);
            let entry = BasicBlock::new(&mut ctx, None, types.clone());
            entry.insert_at_back(region, &ctx);
            Self {
                ctx,
                module,
                entry,
                types,
                u32,
                u64,
            }
        }

        /// Return the indexed source function argument for constructing a test operation.
        fn arg(&self, index: usize) -> Value {
            self.entry.deref(&self.ctx).get_argument(index)
        }

        /// Append a typed source operation to the fixture block and return its operation handle.
        fn append<O: Op>(
            &mut self,
            marker: &str,
            operands: Vec<Value>,
            result_types: Vec<TypeHandle>,
        ) -> Ptr<Operation> {
            let op = Operation::new(
                &mut self.ctx,
                O::get_concrete_op_info(),
                result_types,
                operands,
                vec![],
                0,
            );
            op.deref_mut(&self.ctx).attributes.set(
                Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
                StringAttr::new(marker.into()),
            );
            op.insert_at_back(self.entry, &self.ctx);
            op
        }

        /// Append a u32 or u64 source constant according to the requested width.
        fn constant(&mut self, value: u64, wide: bool) -> Value {
            let ty = if wide { self.u64 } else { self.u32 };
            let width = if wide { 64 } else { 32 };
            let op = Operation::new(
                &mut self.ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![ty.into()],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(op).set_attr_value(
                &self.ctx,
                IntegerAttr::new(
                    ty,
                    APInt::from_u64(value, NonZeroUsize::new(width).unwrap()),
                ),
            );
            op.insert_at_back(self.entry, &self.ctx);
            op.deref(&self.ctx).get_result(0)
        }

        /// Terminate, export, and render the fixture without the separately tested GPU envelope.
        fn render(mut self) -> Result<String, String> {
            let ret = Operation::new(
                &mut self.ctx,
                MirReturnOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            ret.insert_at_back(self.entry, &self.ctx);
            let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
            profile
                .translate_module(&self.ctx, &self.module)
                .map(|target| {
                    render_mapping_module_without_cutlass_envelope(&target, "cluster_test")
                })
                .map_err(|error| error.to_string())
        }
    }

    /// Check native descriptor prefetch accepts generic storage and rejects a shared descriptor pointer.
    #[test]
    fn descriptor_prefetch_is_native_and_rejects_shared_storage() {
        let mut fixture = Fixture::new();
        fixture.append::<PrefetchTensorMapOp>("v1:i0887", vec![fixture.arg(0)], vec![]);
        let text = fixture.render().unwrap();
        assert!(text.contains("\"nvvm.prefetch\"(%v0)"), "{text}");
        assert!(text.contains("tensormap = unit"), "{text}");
        assert!(
            !text.contains("inline_ptx") && !text.contains("inline_asm"),
            "{text}"
        );
        for argument in [1, 2] {
            let mut fixture = Fixture::new();
            fixture.append::<PrefetchTensorMapOp>("v1:i0887", vec![fixture.arg(argument)], vec![]);
            assert!(fixture.render().is_err());
        }
    }

    /// Check invalidation retains its barrier address without adding an unrelated barrier wait.
    #[test]
    fn barrier_invalidation_preserves_pointer_and_has_no_hidden_sync() {
        for argument in [0, 1] {
            let mut fixture = Fixture::new();
            fixture.append::<MbarrierInvalSharedOp>(
                "v1:i0100",
                vec![fixture.arg(argument)],
                vec![],
            );
            let text = fixture.render().unwrap();
            assert_eq!(text.matches("\"nvvm.mbarrier.inval\"").count(), 1, "{text}");
            assert!(
                !text.contains("nvvm.barrier.cta") && !text.contains("nvvm.fence"),
                "{text}"
            );
            assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
        }
        let mut fixture = Fixture::new();
        fixture.append::<MbarrierInvalSharedOp>("v1:i0097", vec![fixture.arg(1)], vec![]);
        assert!(
            fixture
                .render()
                .unwrap_err()
                .contains("expected intrinsic marker v1:i0100")
        );
        let mut fixture = Fixture::new();
        fixture.append::<MbarrierInvalSharedOp>("v1:i0100", vec![fixture.arg(2)], vec![]);
        assert!(fixture.render().is_err());
    }

    /// Check counted arrival/sync forms retain their aligned-participation selectors.
    #[test]
    fn counted_barrier_forms_preserve_instruction_and_alignment() {
        let mut fixture = Fixture::new();
        let id = fixture.constant(3, false);
        let count = fixture.constant(128, false);
        fixture.append::<BarrierCtaSyncCountOp>("v1:i0883", vec![id, count], vec![]);
        fixture.append::<BarrierCtaSyncAlignedCountOp>("v1:i0884", vec![id, count], vec![]);
        fixture.append::<BarrierCtaArriveCountOp>("v1:i0885", vec![id, count], vec![]);
        fixture.append::<BarrierCtaArriveAlignedCountOp>("v1:i0886", vec![id, count], vec![]);
        let text = fixture.render().unwrap();
        assert_eq!(
            text.matches("\"nvvm.barrier.cta.sync\"").count(),
            2,
            "{text}"
        );
        assert_eq!(
            text.matches("\"nvvm.barrier.cta.arrive\"").count(),
            2,
            "{text}"
        );
        assert_eq!(text.matches("aligned = true").count(), 2, "{text}");
        for line in text
            .lines()
            .filter(|line| line.contains("\"nvvm.barrier.cta."))
        {
            assert!(line.contains("(%v8, %v9)"), "{line}");
        }
    }

    /// Check cluster synchronization modes and catalog identities are validated and preserved.
    #[test]
    fn cluster_sync_preserves_mode_and_intrinsic_identity() {
        for (mode, marker, name) in [
            (
                ClusterBarrierModeAttr::ArriveRelaxed,
                "v1:i0279",
                "nvvm.cluster.arrive.relaxed",
            ),
            (
                ClusterBarrierModeAttr::Wait,
                "v1:i0281",
                "nvvm.cluster.wait",
            ),
        ] {
            let mut fixture = Fixture::new();
            let op = fixture.append::<ClusterBarrierOp>(marker, vec![], vec![]);
            ClusterBarrierOp::new(op).set_attr_nvvm_cluster_barrier_mode(&fixture.ctx, mode);
            let text = fixture.render().unwrap();
            assert!(text.contains(name), "{text}");
        }
        let mut fixture = Fixture::new();
        let op = fixture.append::<ClusterBarrierOp>("v1:i0281", vec![], vec![]);
        ClusterBarrierOp::new(op).set_attr_nvvm_cluster_barrier_mode(
            &fixture.ctx,
            ClusterBarrierModeAttr::ArriveRelaxed,
        );
        assert!(
            fixture
                .render()
                .unwrap_err()
                .contains("expected intrinsic marker v1:i0279")
        );
    }

    /// Check cluster try-wait produces the completion predicate and retains acquire ordering.
    #[test]
    fn barrier_wait_returns_predicate_and_preserves_cluster_acquire() {
        let mut fixture = Fixture::new();
        fixture.append::<MbarrierTryWaitParityClusterOp>(
            "v1:i0311",
            vec![fixture.arg(1), fixture.arg(3)],
            vec![fixture.types[7]],
        );
        fixture.append::<MbarrierArriveExpectTxClusterOp>(
            "v1:i0307",
            vec![fixture.arg(1), fixture.arg(3)],
            vec![fixture.types[6]],
        );
        fixture.append::<MbarrierArriveClusterOp>("v1:i0308", vec![fixture.arg(6)], vec![]);
        let text = fixture.render().unwrap();
        assert!(text.contains("\"nvvm.mbarrier.wait.parity\""), "{text}");
        assert!(text.contains("kind = #nvvm.mbar_wait<try>"), "{text}");
        assert!(text.contains("order = #nvvm.mem_order<acquire>"), "{text}");
        assert!(text.contains("scope = #nvvm.mbar_scope<cluster>"), "{text}");
        assert!(text.contains("relaxed = true"), "{text}");
        assert!(
            text.contains("\"llvm.inttoptr\"(%v6) : (i64) -> (!llvm.ptr<7>)"),
            "{text}"
        );
        assert!(!text.contains("\"nvvm.mbarrier.try_wait.parity\""));
    }

    /// Check relaxed barrier instructions retain cluster scope without gaining stronger ordering.
    #[test]
    fn relaxed_cluster_arrival_and_wait_keep_scope_without_acquire_release() {
        let mut fixture = Fixture::new();
        fixture.append::<MbarrierArriveClusterRelaxedOp>("v1:i1040", vec![fixture.arg(6)], vec![]);
        fixture.append::<MbarrierTryWaitParityClusterRelaxedOp>(
            "v1:i1041",
            vec![fixture.arg(1), fixture.arg(3)],
            vec![fixture.types[7]],
        );
        let text = fixture.render().unwrap();
        assert!(
            text.contains("\"llvm.inttoptr\"(%v6) : (i64) -> (!llvm.ptr<7>)"),
            "{text}"
        );
        assert!(text.contains("\"nvvm.mbarrier.arrive\""), "{text}");
        assert!(text.contains("scope = #nvvm.mem_scope<cluster>"), "{text}");
        assert!(text.contains("relaxed = true"), "{text}");
        assert!(text.contains("order = #nvvm.mem_order<relaxed>"), "{text}");
        assert!(text.contains("scope = #nvvm.mbar_scope<cluster>"), "{text}");
        assert!(!text.contains("mem_order<acquire>"), "{text}");
        assert!(!text.contains("inline_ptx"), "{text}");
    }

    /// Check restricted acquire fences do not become unrestricted memory-order operations.
    #[test]
    fn restricted_acquire_fence_retains_shared_only_ordering() {
        let mut fixture = Fixture::new();
        fixture.append::<FenceAcquireSharedClusterClusterOp>("v1:i1042", vec![], vec![]);
        let text = fixture.render().unwrap();
        assert!(text.contains("\"nvvm.fence.sync_restrict\""), "{text}");
        assert!(text.contains("order = #nvvm.mem_order<acquire>"), "{text}");
        assert!(!text.contains("inline_ptx"), "{text}");
        assert!(!text.contains("fence.proxy"), "{text}");
    }

    /// Check lane election retains both its lane ID and elected-thread predicate.
    #[test]
    fn elect_preserves_both_lane_and_predicate_results() {
        let mut fixture = Fixture::new();
        fixture.append::<ElectSyncOp>(
            "v1:i0367",
            vec![fixture.arg(3)],
            vec![fixture.types[3], fixture.types[7]],
        );
        let text = fixture.render().unwrap();
        assert!(text.contains("intrin = \"llvm.nvvm.elect.sync\""), "{text}");
        assert!(text.contains("position = array<i64: 0>"), "{text}");
        assert!(text.contains("position = array<i64: 1>"), "{text}");
        assert!(text.contains("!llvm.struct<(i32, i1)>"), "{text}");
    }

    /// Check tensor loads use NVVM operand order while retaining remote-barrier rank bits.
    #[test]
    fn tma_reorders_operands_and_preserves_remote_barrier_address() {
        let mut fixture = Fixture::new();
        let zero = fixture.constant(0, true);
        fixture.append::<CpAsyncBulkTensorG2sTile2dMulticastCg2Op>(
            "v1:i0331",
            vec![
                fixture.arg(1),
                fixture.arg(2),
                fixture.arg(0),
                fixture.arg(4),
                fixture.arg(4),
                fixture.arg(5),
                zero,
            ],
            vec![],
        );
        fixture.append::<CpAsyncBulkTensorS2gTile2dOp>(
            "v1:i0336",
            vec![
                fixture.arg(1),
                fixture.arg(0),
                fixture.arg(4),
                fixture.arg(4),
            ],
            vec![],
        );
        let text = fixture.render().unwrap();
        assert!(text.contains("group = #nvvm.cta_group<cta_2>"), "{text}");
        assert!(text.contains("useIntrinsic = false"), "{text}");
        assert!(
            text.contains("operandSegmentSizes = array<i32: 1, 1, 2, 1, 0, 1, 0, 0>"),
            "{text}"
        );
        assert!(
            text.contains("\"llvm.ptrtoint\"(%v2) : (!llvm.ptr<7>) -> (i32)"),
            "{text}"
        );
        assert!(
            text.contains("\"nvvm.cp.async.bulk.tensor.global.shared.cta\"(%v0, %v1, %v4, %v4)"),
            "{text}"
        );
    }

    /// Reject malformed tensor transfer signatures and unsupported cache selectors.
    #[test]
    fn tma_rejects_incorrect_shape_and_hidden_cache_policy() {
        for cache_hint in [None, Some(1)] {
            let mut fixture = Fixture::new();
            let zero = fixture.constant(cache_hint.unwrap_or(0), true);
            let mut operands = vec![
                fixture.arg(1),
                fixture.arg(2),
                fixture.arg(0),
                fixture.arg(4),
                fixture.arg(4),
                fixture.arg(5),
                zero,
            ];
            if cache_hint.is_none() {
                operands.pop();
            }
            fixture.append::<CpAsyncBulkTensorG2sTile2dMulticastCg2Op>(
                "v1:i0331",
                operands,
                vec![],
            );
            let error = fixture.render().unwrap_err();
            assert!(
                error.contains(if cache_hint.is_none() {
                    "Expected 7 operand(s), but found 6"
                } else {
                    "zero cache hint"
                }),
                "{error}"
            );
        }
    }

    /// Reject runtime or out-of-range bulk wait counts while preserving legal immediates.
    #[test]
    fn bulk_wait_requires_legal_compile_time_group_count() {
        for group in [0, 1, 7, 8] {
            let mut fixture = Fixture::new();
            let count = fixture.constant(group, false);
            fixture.append::<CpAsyncBulkWaitGroupReadOp>("v1:i0342", vec![count], vec![]);
            let result = fixture.render();
            if group <= 7 {
                let text = result.unwrap();
                assert!(text.contains(&format!("group = {group} : i32")), "{text}");
                assert!(text.contains("read = unit"), "{text}");
            } else {
                assert!(result.unwrap_err().contains("constant in 0..=7"));
            }
        }
        let mut fixture = Fixture::new();
        fixture.append::<CpAsyncBulkWaitGroupReadOp>("v1:i0342", vec![fixture.arg(3)], vec![]);
        assert!(fixture.render().unwrap_err().contains("constant in 0..=7"));
    }
}
