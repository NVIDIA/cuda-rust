/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{CatalogFile, Tcgen05Adapter};
use crate::render::common::rust_header;
use crate::render::families::{
    clc_intrinsics, cluster_barriers, cluster_memory, debug_controls, execution_control_family,
    execution_controls, render_tcgen05_carrier_runs, tcgen05_intrinsics, tcgen05_mma_intrinsics,
    tcgen05_non_mma_intrinsics, tma_intrinsics, wgmma_controls,
};
use std::fmt::Write as _;

pub(in crate::render) fn render_dialect_cluster_barrier(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    if cluster_barriers(catalog).next().is_none() {
        return output;
    }
    output.push_str(
        r##"//! Generated operations for the closed cluster-barrier family.

use pliron::{
    attribute::Attribute,
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.cluster_barrier_mode", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ClusterBarrierModeAttr {
    Arrive,
    ArriveAligned,
    ArriveRelaxed,
    ArriveRelaxedAligned,
    Wait,
    WaitAligned,
}

/// Cluster synchronization whose exact spelling is carried by an attribute.
#[pliron_op(
    name = "nvvm.cluster_barrier",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (nvvm_cluster_barrier_mode: ClusterBarrierModeAttr)
)]
pub struct ClusterBarrierOp;

impl ClusterBarrierOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, mode: ClusterBarrierModeAttr) -> Ptr<Operation> {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let this = Self { op };
        this.set_attr_nvvm_cluster_barrier_mode(ctx, mode);
        this.get_operation()
    }
}

impl Verify for ClusterBarrierOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if self.get_attr_nvvm_cluster_barrier_mode(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.cluster_barrier requires a mode attribute");
        }
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "nvvm.cluster_barrier takes no operands or results");
        }
        Ok(())
    }
}

/// Compatibility operation for a complete aligned cluster synchronization.
#[pliron_op(
    name = "nvvm.cluster_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ClusterSyncOp;

impl ClusterSyncOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    ClusterBarrierModeAttr::register(ctx);
    ClusterBarrierOp::register(ctx);
    ClusterSyncOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_cluster_memory(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    assert_eq!(cluster_memory(catalog).count(), 2);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for cluster address mapping and remote shared reads.

use dialect_mir::types::{MirPointerKind, MirPtrType, address_space};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

fn verify_pointer_rank(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    if op.get_num_operands() != 2 || op.get_num_results() != 1 {
        return verify_err!(op.loc(), "{name} requires two operands and one result");
    }
    if op
        .get_operand(0)
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_none()
    {
        return verify_err!(op.loc(), "{name} source must be a MIR pointer");
    }
    let rank_ty = op.get_operand(1).get_type(ctx);
    let rank_ty = rank_ty.deref(ctx);
    if !rank_ty
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.width() == 32 && ty.signedness() == Signedness::Unsigned)
    {
        return verify_err!(op.loc(), "{name} rank must be u32");
    }
    Ok(())
}

#[pliron_op(
    name = "nvvm.mapa_shared_cluster",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MapaSharedClusterOp;

impl MapaSharedClusterOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    // Kind-preserving addrspace retype: copies the source pointer's kind
    // verbatim; concrete kinds are minted only in mir-importer's facts.rs.
    #[allow(clippy::disallowed_methods)]
    pub fn build(ctx: &mut Context, source: Value, rank: Value) -> Ptr<Operation> {
        let source_ty = source.get_type(ctx);
        let (pointee, is_mutable, pointer_kind) = {
            let source_ty_obj = source_ty.deref(ctx);
            let source_ptr = source_ty_obj
                .downcast_ref::<MirPtrType>()
                .expect("nvvm.mapa_shared_cluster source must be a MIR pointer");
            (
                source_ptr.pointee,
                source_ptr.is_mutable(),
                source_ptr.pointer_kind(),
            )
        };
        let result_ty = MirPtrType::get_with_kind(
            ctx,
            pointee,
            is_mutable,
            address_space::CLUSTER_SHARED,
            pointer_kind,
        );
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![source, rank],
            vec![],
            0,
        )
    }
}

impl Verify for MapaSharedClusterOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation();
        verify_pointer_rank(ctx, operation, "nvvm.mapa_shared_cluster")?;
        let op = operation.deref(ctx);
        let source_ty = op.get_operand(0).get_type(ctx);
        let result_ty = op.get_result(0).get_type(ctx);
        let source_ty_obj = source_ty.deref(ctx);
        let result_ty_obj = result_ty.deref(ctx);
        let source_ptr = source_ty_obj.downcast_ref::<MirPtrType>().unwrap();
        let Some(result_ptr) = result_ty_obj.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "nvvm.mapa_shared_cluster result must be a MIR pointer"
            );
        };
        if !matches!(
            source_ptr.pointer_kind(),
            MirPointerKind::RawConst | MirPointerKind::RawMut
        )
            || result_ptr.pointee != source_ptr.pointee
            || result_ptr.is_mutable() != source_ptr.is_mutable()
            || !result_ptr.is_cluster_shared()
            || result_ptr.pointer_kind() != source_ptr.pointer_kind()
        {
            return verify_err!(
                op.loc(),
                "nvvm.mapa_shared_cluster requires a raw source pointer and must preserve its pointee, mutability, and raw kind while returning addrspace(7)"
            );
        }
        Ok(())
    }
}

#[pliron_op(
    name = "nvvm.dsmem_read_u32",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct DsmemReadU32Op;

impl DsmemReadU32Op {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(ctx: &mut Context, source: Value, rank: Value) -> Ptr<Operation> {
        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.to_handle()],
            vec![source, rank],
            vec![],
            0,
        )
    }
}

impl Verify for DsmemReadU32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation();
        verify_pointer_rank(ctx, operation, "nvvm.dsmem_read_u32")?;
        let op = operation.deref(ctx);
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty = result_ty.deref(ctx);
        if !result_ty
            .downcast_ref::<IntegerType>()
            .is_some_and(|ty| ty.width() == 32 && ty.signedness() == Signedness::Unsigned)
        {
            return verify_err!(op.loc(), "nvvm.dsmem_read_u32 result must be u32");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    MapaSharedClusterOp::register(ctx);
    DsmemReadU32Op::register(ctx);
}
"#,
    );
    output
}

pub(in crate::render) fn render_dialect_wgmma_control(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(wgmma_controls(catalog).count(), 3);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! Structural operations for generated WGMMA controls.

use pliron::{
    builtin::{op_interfaces::{NOpdsInterface, NResultsInterface}, types::IntegerType},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

#[pliron_op(
    name = "nvvm.wgmma_fence_sync_aligned",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct WgmmaFenceSyncAlignedOp;

impl WgmmaFenceSyncAlignedOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(ctx: &mut Context) -> Ptr<Operation> {
        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)
    }
}

#[pliron_op(
    name = "nvvm.wgmma_commit_group_sync_aligned",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct WgmmaCommitGroupSyncAlignedOp;

impl WgmmaCommitGroupSyncAlignedOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(ctx: &mut Context) -> Ptr<Operation> {
        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)
    }
}

#[pliron_op(
    name = "nvvm.wgmma_wait_group_sync_aligned",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct WgmmaWaitGroupSyncAlignedOp;

impl WgmmaWaitGroupSyncAlignedOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(ctx: &mut Context, max_pending: Value) -> Ptr<Operation> {
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![max_pending],
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaWaitGroupSyncAlignedOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "nvvm.wgmma_wait_group_sync_aligned takes one operand and no results");
        }
        if !is_i64(ctx, op.get_operand(0).get_type(ctx)) {
            return verify_err!(op.loc(), "nvvm.wgmma_wait_group_sync_aligned operand must be i64");
        }
        Ok(())
    }
}

fn is_i64(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64)
}

pub(super) fn register(ctx: &mut Context) {
    WgmmaFenceSyncAlignedOp::register(ctx);
    WgmmaCommitGroupSyncAlignedOp::register(ctx);
    WgmmaWaitGroupSyncAlignedOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_debug_control(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(debug_controls(catalog).count(), 3);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated PTX debug controls.\n\n\
         use pliron::{\n\
             builtin::{\n\
                 attributes::IntegerAttr,\n\
                 op_interfaces::{NOpdsInterface, NResultsInterface},\n\
                 types::{IntegerType, Signedness},\n\
             },\n\
             common_traits::Verify,\n\
             context::{Context, Ptr},\n\
             identifier::Identifier,\n\
             location::Located,\n\
             op::Op,\n\
             operation::Operation,\n\
             result::Error,\n\
             verify_err,\n\
         };\n\
         use pliron::utils::apint::APInt;\n\
         use pliron_derive::pliron_op;\n\
         use std::num::NonZeroUsize;\n\n\
         #[pliron_op(\n\
             name = \"nvvm.trap\",\n\
             format,\n\
             verifier = \"succ\",\n\
             interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n\
         )]\n\
         pub struct TrapOp;\n\n\
         impl TrapOp {\n\
             pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\
             pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n\
                 Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)\n\
             }\n\
         }\n\n\
         #[pliron_op(\n\
             name = \"nvvm.brkpt\",\n\
             format,\n\
             verifier = \"succ\",\n\
             interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n\
         )]\n\
         pub struct BreakpointOp;\n\n\
         impl BreakpointOp {\n\
             pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\
             pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n\
                 Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)\n\
             }\n\
         }\n\n\
         #[pliron_op(\n\
             name = \"nvvm.pmevent\",\n\
             format,\n\
             interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n\
         )]\n\
         pub struct PmEventOp;\n\n\
         impl PmEventOp {\n\
             pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n\
             pub fn build(ctx: &mut Context, event_id: u32) -> Ptr<Operation> {\n\
                 let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);\n\
                 let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n\
                 let value = APInt::from_u64(event_id.into(), NonZeroUsize::new(32).unwrap());\n\
                 op.deref_mut(ctx).attributes.set(\n\
                     Identifier::try_from(\"event_id\").unwrap(),\n\
                     IntegerAttr::new(ty, value),\n\
                 );\n\
                 op\n\
             }\n\n\
             pub fn new_with_event_id(ctx: &mut Context, event_id: u32) -> Ptr<Operation> {\n\
                 Self::build(ctx, event_id)\n\
             }\n\n\
             pub fn event_id(&self, ctx: &Context) -> Option<u32> {\n\
                 let key = Identifier::try_from(\"event_id\").unwrap();\n\
                 let operation = self.get_operation().deref(ctx);\n\
                 let attribute: &IntegerAttr = operation.attributes.get(&key)?;\n\
                 let ty_handle = attribute.get_type();\n\
                 let ty = ty_handle.deref(ctx);\n\
                 if ty.width() != 32 || ty.signedness() != Signedness::Unsigned {\n\
                     return None;\n\
                 }\n\
                 u32::try_from(attribute.value().to_u64()).ok().filter(|value| *value <= 15)\n\
             }\n\
\n\
             pub fn get_event_id(&self, ctx: &Context) -> Option<u32> {\n\
                 self.event_id(ctx)\n\
             }\n\
         }\n\n\
         impl Verify for PmEventOp {\n\
             fn verify(&self, ctx: &Context) -> Result<(), Error> {\n\
                 let op = self.get_operation().deref(ctx);\n\
                 if self.event_id(ctx).is_none() {\n\
                     return verify_err!(op.loc(), \"nvvm.pmevent requires a u32 event ID in 0..=15\");\n\
                 }\n\
                 Ok(())\n\
             }\n\
         }\n\n\
         pub(super) fn register(ctx: &mut Context) {\n\
             TrapOp::register(ctx);\n\
             BreakpointOp::register(ctx);\n\
             PmEventOp::register(ctx);\n\
         }\n",
    );
    output
}

pub(in crate::render) fn render_dialect_clc(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(clc_intrinsics(catalog).count(), 6);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated Cluster Launch Control operations.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::IntegerType,\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::{TypeHandle, Typed},\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in clc_intrinsics(catalog) {
        let result_count = record.dialect.results.len();
        let verifier = if result_count == 0 {
            "    verifier = \"succ\",\n"
        } else {
            ""
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n{verifier}    interfaces = [NOpdsInterface<2>, NResultsInterface<{result_count}>],\n)]",
            record.dialect.op_name,
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n}\n\n");
        if result_count == 1 {
            writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
            writeln!(
                output,
                "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_query_shape(ctx, self.get_operation(), {:?})\n    }}\n}}\n",
                record.dialect.op_name,
            )
            .unwrap();
        }
    }
    output.push_str(
        "fn verify_query_shape(\n    ctx: &Context,\n    operation: Ptr<Operation>,\n    name: &str,\n) -> Result<(), Error> {\n    let op = operation.deref(ctx);\n    let valid = op.get_num_operands() == 2\n        && op.get_num_results() == 1\n        && is_integer_width(ctx, op.get_operand(0).get_type(ctx), 64)\n        && is_integer_width(ctx, op.get_operand(1).get_type(ctx), 64)\n        && is_integer_width(ctx, op.get_result(0).get_type(ctx), 32);\n    if !valid {\n        return verify_err!(\n            op.loc(),\n            \"{} requires two i64 operands and one i32 result\",\n            name\n        );\n    }\n    Ok(())\n}\n\nfn is_integer_width(ctx: &Context, ty: TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\npub(super) fn register(ctx: &mut Context) {\n",
    );
    for record in clc_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_execution_control(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    assert_eq!(execution_controls(catalog).count(), 8);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated counted-barrier, grid-dependency, and register-control operations.\n\nuse pliron::{\n    builtin::{attributes::IntegerAttr, op_interfaces::{NOpdsInterface, NResultsInterface}, types::{IntegerType, Signedness}},\n    common_traits::Verify,\n    context::{Context, Ptr},\n    identifier::Identifier,\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    verify_err,\n};\nuse pliron::utils::apint::APInt;\nuse pliron_derive::pliron_op;\nuse std::num::NonZeroUsize;\n\n",
    );
    for record in execution_controls(catalog).filter(|record| record.family != "register_control") {
        let operand_count = record.dialect.operands.len();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    verifier = \"succ\",\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n}\n\n");
    }
    for record in execution_control_family(catalog, "register_control") {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n    pub fn build(ctx: &mut Context, register_count: u32) -> Ptr<Operation> {\n        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);\n        let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        let value = APInt::from_u64(register_count.into(), NonZeroUsize::new(32).unwrap());\n        op.deref_mut(ctx).attributes.set(\n            Identifier::try_from(\"register_count\").unwrap(),\n            IntegerAttr::new(ty, value),\n        );\n        op\n    }\n\n    pub fn register_count(&self, ctx: &Context) -> Option<u32> {\n        let key = Identifier::try_from(\"register_count\").unwrap();\n        let operation = self.get_operation().deref(ctx);\n        let attribute: &IntegerAttr = operation.attributes.get(&key)?;\n        let ty_handle = attribute.get_type();\n        let ty = ty_handle.deref(ctx);\n        if ty.width() != 32 || ty.signedness() != Signedness::Unsigned {\n            return None;\n        }\n        u32::try_from(attribute.value().to_u64()).ok().filter(|value| (24..=256).contains(value) && value % 8 == 0)\n    }\n}\n\n",
        );
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {\n        let op = self.get_operation().deref(ctx);\n        if self.register_count(ctx).is_none() {\n            return verify_err!(op.loc(), \"setmaxnreg requires an immediate register count in 24..=256 divisible by 8\");\n        }\n        Ok(())\n    }\n}\n\n",
        );
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in execution_controls(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_tma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated Tensor Memory Accelerator operations.\n\nuse pliron::{\n    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},\n    context::{Context, Ptr},\n    op::Op,\n    operation::Operation,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in tma_intrinsics(catalog) {
        let operand_count = record.dialect.operands.len();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    verifier = \"succ\",\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n}\n\n");
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in tma_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_tcgen05(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(tcgen05_intrinsics(catalog).count(), 233);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Generated Tensor Core Generation 5 operations.

use dialect_mir::{ops::MirConstantOp, types::MirPtrType};
use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        ops::ConstantOp,
        types::{FP32Type, IntegerType},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[derive(Clone, Copy)]
enum Tcgen05Carrier {
    Ptr,
    I1,
    I16,
    I32,
    I64,
    F32,
}

impl Tcgen05Carrier {
    fn matches(self, ctx: &Context, ty: TypeHandle) -> bool {
        match self {
            Self::Ptr => ty.deref(ctx).downcast_ref::<MirPtrType>().is_some(),
            Self::I1 => is_integer_width(ctx, ty, 1),
            Self::I16 => is_integer_width(ctx, ty, 16),
            Self::I32 => is_integer_width(ctx, ty, 32),
            Self::I64 => is_integer_width(ctx, ty, 64),
            Self::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ptr => "MIR pointer",
            Self::I1 => "i1",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
        }
    }
}

fn is_integer_width(ctx: &Context, ty: TypeHandle, width: u32) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == width)
}

fn expected_carrier_count(runs: &[(Tcgen05Carrier, usize)]) -> usize {
    runs.iter().map(|(_, count)| *count).sum()
}

fn verify_tcgen05_signature(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    operand_runs: &[(Tcgen05Carrier, usize)],
    result_runs: &[(Tcgen05Carrier, usize)],
    constant_operand: Option<usize>,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let expected_operands = expected_carrier_count(operand_runs);
    let expected_results = expected_carrier_count(result_runs);
    if op.get_num_operands() != expected_operands || op.get_num_results() != expected_results {
        return verify_err!(
            op.loc(),
            "{name} requires {expected_operands} operands and {expected_results} results"
        );
    }

    let mut index = 0;
    for (carrier, count) in operand_runs {
        for _ in 0..*count {
            if !carrier.matches(ctx, op.get_operand(index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "{name} operand {index} must be {}",
                    carrier.name()
                );
            }
            index += 1;
        }
    }

    index = 0;
    for (carrier, count) in result_runs {
        for _ in 0..*count {
            if !carrier.matches(ctx, op.get_result(index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "{name} result {index} must be {}",
                    carrier.name()
                );
            }
            index += 1;
        }
    }

    if let Some(index) = constant_operand {
        let value = op.get_operand(index);
        let is_constant = value.defining_op().is_some_and(|defining_op| {
            Operation::get_op::<MirConstantOp>(defining_op, ctx).is_some()
                || Operation::get_op::<ConstantOp>(defining_op, ctx).is_some()
        });
        if !is_constant {
            return verify_err!(
                op.loc(),
                "{name} operand {index} must be a compile-time constant"
            );
        }
    }

    Ok(())
}

"#,
    );
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        output.push_str(
            r#"#[pliron_attr(name = "nvvm.tcgen05_mma_form", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaFormAttr {
    Shared, Tensor, TensorAshift, SpShared, SpTensor, SpTensorAshift,
    WsShared, WsSharedZeroColMask, WsSpShared, WsSpSharedZeroColMask,
    WsSpTensor, WsSpTensorZeroColMask, WsTensor, WsTensorZeroColMask,
}

#[pliron_attr(name = "nvvm.tcgen05_mma_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaKindAttr { F16, Tf32, F8f6f4, I8 }

#[pliron_attr(name = "nvvm.tcgen05_mma_cta_group", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaCtaGroupAttr { Cg1, Cg2 }

#[pliron_attr(name = "nvvm.tcgen05_mma_collector_a", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaCollectorAAttr { Discard, LastUse, Fill, Use }

#[pliron_attr(name = "nvvm.tcgen05_mma_b_buffer", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaBBufferAttr { B0, B1, B2, B3 }

#[pliron_attr(name = "nvvm.tcgen05_mma_b_usage", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Tcgen05MmaBUsageAttr { Discard, LastUse, Fill, Use }

#[pliron_op(
    name = "nvvm.tcgen05_mma",
    format,
    attributes = (
        nvvm_tcgen05_mma_form: Tcgen05MmaFormAttr,
        nvvm_tcgen05_mma_kind: Tcgen05MmaKindAttr,
        nvvm_tcgen05_mma_cta_group: Tcgen05MmaCtaGroupAttr,
        nvvm_tcgen05_mma_collector_a: Tcgen05MmaCollectorAAttr,
        nvvm_tcgen05_mma_b_buffer: Tcgen05MmaBBufferAttr,
        nvvm_tcgen05_mma_b_usage: Tcgen05MmaBUsageAttr
    )
)]
pub struct Tcgen05MmaOp;

impl Tcgen05MmaOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }
}

impl Verify for Tcgen05MmaOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        use Tcgen05Carrier::{I1, I32, I64};
        let op = self.get_operation();
        let form = self.get_attr_nvvm_tcgen05_mma_form(ctx);
        let Some(form) = form.as_deref() else {
            return verify_err!(op.deref(ctx).loc(), "nvvm.tcgen05_mma requires a form");
        };
        if self.get_attr_nvvm_tcgen05_mma_kind(ctx).is_none() {
            return verify_err!(op.deref(ctx).loc(), "nvvm.tcgen05_mma requires a kind");
        }
        let (operands, base): (&[(Tcgen05Carrier, usize)], bool) = match form {
            Tcgen05MmaFormAttr::Shared | Tcgen05MmaFormAttr::WsShared =>
                (&[(I32, 1), (I64, 2), (I32, 1), (I1, 1)], matches!(form, Tcgen05MmaFormAttr::Shared)),
            Tcgen05MmaFormAttr::Tensor | Tcgen05MmaFormAttr::TensorAshift | Tcgen05MmaFormAttr::WsTensor =>
                (&[(I32, 2), (I64, 1), (I32, 1), (I1, 1)], !matches!(form, Tcgen05MmaFormAttr::WsTensor)),
            Tcgen05MmaFormAttr::SpShared | Tcgen05MmaFormAttr::WsSpShared =>
                (&[(I32, 1), (I64, 2), (I32, 1), (I1, 1), (I32, 1)], matches!(form, Tcgen05MmaFormAttr::SpShared)),
            Tcgen05MmaFormAttr::SpTensor | Tcgen05MmaFormAttr::SpTensorAshift | Tcgen05MmaFormAttr::WsSpTensor =>
                (&[(I32, 2), (I64, 1), (I32, 1), (I1, 1), (I32, 1)], !matches!(form, Tcgen05MmaFormAttr::WsSpTensor)),
            Tcgen05MmaFormAttr::WsSharedZeroColMask =>
                (&[(I32, 1), (I64, 2), (I32, 1), (I1, 1), (I64, 1)], false),
            Tcgen05MmaFormAttr::WsSpSharedZeroColMask =>
                (&[(I32, 1), (I64, 2), (I32, 1), (I1, 1), (I32, 1), (I64, 1)], false),
            Tcgen05MmaFormAttr::WsSpTensorZeroColMask =>
                (&[(I32, 2), (I64, 1), (I32, 1), (I1, 1), (I32, 1), (I64, 1)], false),
            Tcgen05MmaFormAttr::WsTensorZeroColMask =>
                (&[(I32, 2), (I64, 1), (I32, 1), (I1, 1), (I64, 1)], false),
        };
        verify_tcgen05_signature(ctx, op, "nvvm.tcgen05_mma", operands, &[], None)?;

        let cta_group = self.get_attr_nvvm_tcgen05_mma_cta_group(ctx);
        let collector_a = self.get_attr_nvvm_tcgen05_mma_collector_a(ctx);
        let b_buffer = self.get_attr_nvvm_tcgen05_mma_b_buffer(ctx);
        let b_usage = self.get_attr_nvvm_tcgen05_mma_b_usage(ctx);
        if base {
            if cta_group.is_none() || collector_a.is_none() || b_buffer.is_some() || b_usage.is_some() {
                return verify_err!(op.deref(ctx).loc(), "base tcgen05 MMA requires CTA-group and collector-A selectors");
            }
            if matches!(form, Tcgen05MmaFormAttr::TensorAshift | Tcgen05MmaFormAttr::SpTensorAshift)
                && !matches!(collector_a.as_deref(), Some(Tcgen05MmaCollectorAAttr::Discard | Tcgen05MmaCollectorAAttr::LastUse))
            {
                return verify_err!(op.deref(ctx).loc(), "ashift tcgen05 MMA only supports discard or last-use collector-A");
            }
        } else if cta_group.is_some() || collector_a.is_some() || b_buffer.is_none() || b_usage.is_none() {
            return verify_err!(op.deref(ctx).loc(), "warp-specialized tcgen05 MMA requires B-buffer and B-usage selectors");
        }
        Ok(())
    }
}

"#,
        );
    }
    for record in tcgen05_non_mma_intrinsics(catalog) {
        let operand_count = record.dialect.operands.len();
        let result_count = record.dialect.results.len();
        let operand_runs = render_tcgen05_carrier_runs(&record.dialect.operands);
        let result_runs = render_tcgen05_carrier_runs(&record.dialect.results);
        let constant_operand = record
            .tcgen05
            .as_ref()
            .is_some_and(|tcgen05| {
                matches!(
                    tcgen05.adapter,
                    Tcgen05Adapter::TmemHalfSplitOffsetInjectPack16ToU32Registers
                        | Tcgen05Adapter::TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid
                )
            })
            .then_some(1);
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<{result_count}>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        output.push_str("    fn verify(&self, ctx: &Context) -> Result<(), Error> {\n");
        output.push_str("        verify_tcgen05_signature(\n");
        output.push_str("            ctx,\n");
        output.push_str("            self.get_operation(),\n");
        writeln!(output, "            {:?},", record.dialect.op_name).unwrap();
        writeln!(output, "            {operand_runs},").unwrap();
        writeln!(output, "            {result_runs},").unwrap();
        writeln!(output, "            {constant_operand:?},").unwrap();
        output.push_str("        )\n");
        output.push_str("    }\n");
        output.push_str("}\n\n");
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        output.push_str(
            "    Tcgen05MmaFormAttr::register(ctx);\n    Tcgen05MmaKindAttr::register(ctx);\n    Tcgen05MmaCtaGroupAttr::register(ctx);\n    Tcgen05MmaCollectorAAttr::register(ctx);\n    Tcgen05MmaBBufferAttr::register(ctx);\n    Tcgen05MmaBUsageAttr::register(ctx);\n    Tcgen05MmaOp::register(ctx);\n",
        );
    }
    for record in tcgen05_non_mma_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}
