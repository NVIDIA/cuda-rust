/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent SM100 MMA and rank-three tiled-copy contracts.
//!
//! Each operation owns its layout/shape contract. QK and PV may have different
//! shapes and layouts within one kernel. Pointer lifetimes, barrier phases and
//! asynchronous alias handoffs are unsafe source obligations, not implied by
//! the order of operations in a single thread's IR.
//!
//! MMA shared operands carry allocation bases plus trailing u32 byte offsets
//! (A then B for SS, B for TS). Offsets select complete tiles and must preserve
//! the layout's full swizzle period. Their validity within the live allocation
//! is an unsafe source obligation, like shared-pointer validity; the typed API
//! constructs these offsets only from bounded, complete-tile array indices.
//!
//! Every operation has zero SSA results; the operand lists below describe
//! `new`'s positional values. TMEM operands are encoded `u32` addresses.
//! TMA's three coordinates address the descriptor's physical axes: within-span
//! column, row, and column-span index. For a 64-element span, physical
//! `(row, column) = (7, 128)` becomes `(c0, c1, c2) = (0, 7, 2)`.
use crate::{
    attributes::CuteComposedLayoutAttr,
    layout::{ComposedLayout, Layout, OffsetUnit, Swizzle},
};
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::{
    attribute::Attribute,
    builtin::types::IntegerType,
    common_traits::Verify,
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err, verify_err_noloc,
};
use pliron_derive::{pliron_attr, pliron_op};

/// Canonical tiled FP16 layout, with all offsets expressed in elements.
///
/// `mn = false` makes K/columns contiguous; `true` makes M-or-N/rows contiguous.
/// The contiguous extent is 32 or a positive multiple of 64, and the other
/// extent is a positive multiple of eight. A span of 32 elements uses a
/// 64-byte swizzle; larger extents use tiled spans of 64 and a 128-byte swizzle.
///
/// ```
/// use dialect_cute::ops::sm100_attention::{CuteSm100MmaAttr, shared_layout};
/// use pliron::{common_traits::Verify, context::Context};
///
/// // A collective M128/N128/K64 product: each CTA stores 64 rows of A and B.
/// let layout = shared_layout(64, 64, false).unwrap();
/// let plan = CuteSm100MmaAttr {
///     m: 128, n: 128, k: 64, cta_group: 2,
///     a_mn: false, b_mn: false,
///     a_layout: layout.clone(), b_layout: layout,
/// };
/// plan.verify(&Context::new()).unwrap();
/// ```
pub fn shared_layout(rows: u32, cols: u32, mn: bool) -> Result<CuteComposedLayoutAttr, String> {
    let (contiguous, outer) = if mn { (rows, cols) } else { (cols, rows) };
    if contiguous < 32
        || (contiguous != 32 && !contiguous.is_multiple_of(64))
        || outer == 0
        || !outer.is_multiple_of(8)
    {
        return Err(
            "SM100 shared layout needs contiguous 32 or a multiple of 64 and outer multiple of 8"
                .into(),
        );
    }
    let span = contiguous.min(64);
    let (shape, stride) = if mn {
        (
            format!("(({span},{}),(8,{}))", rows / span, cols / 8),
            format!("((1,{}),({span},{}))", span * cols, span * 8),
        )
    } else {
        (
            format!("((8,{}),({span},{}))", rows / 8, cols / span),
            format!("(({span},{}),(1,{}))", span * 8, span * rows),
        )
    };
    let outer: Layout = format!("{shape}:{stride}")
        .parse()
        .map_err(|e| format!("{e}"))?;
    ComposedLayout::new(
        Swizzle::new(if span == 32 { 2 } else { 3 }, 3, 3),
        0,
        outer,
        OffsetUnit::Elements,
    )
    .map(CuteComposedLayoutAttr)
    .map_err(|e| e.to_string())
}

/// One independent FP16 MMA product with FP32 accumulation.
///
/// Dimensions describe the whole CTA group: `M = 128`, N is a multiple of 16
/// in `16..=256`, and K is a positive multiple of 16. Each CTA's shared A/B
/// layouts have `M / cta_group` and `N / cta_group` rows, respectively.
/// `a_mn`/`b_mn` select contiguous rows instead of contiguous K; both layouts
/// must match [`shared_layout`]. TMEM-A operations additionally require group
/// one and K-major A; their Rust importer accepts K=32 or a multiple of 64.
#[pliron_attr(
    name = "cute.sm100_mma",
    format = "`<` $m `,` $n `,` $k `,` $cta_group `,` $a_mn `,` $b_mn `,` $a_layout `,` $b_layout `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100MmaAttr {
    /// Collective output-row extent M.
    pub m: u32,
    /// Collective output-column extent N.
    pub n: u32,
    /// Reduction extent K, in logical elements.
    pub k: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// Whether A's M dimension, rather than K, is physically contiguous.
    pub a_mn: bool,
    /// Whether B's N dimension, rather than K, is physically contiguous.
    pub b_mn: bool,
    /// Physical shared placement of each CTA's A tile.
    pub a_layout: CuteComposedLayoutAttr,
    /// Physical shared placement of each CTA's B tile.
    pub b_layout: CuteComposedLayoutAttr,
}
impl Verify for CuteSm100MmaAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        if !matches!(self.cta_group, 1 | 2)
            || self.m != 128
            || !(16..=256).contains(&self.n)
            || !self.n.is_multiple_of(16)
            || self.k == 0
            || !self.k.is_multiple_of(16)
        {
            return verify_err_noloc!(
                "SM100 independent FP16 MMA requires M128, N16..256 by16, K by16, CTA group1 or2"
            );
        }
        for (layout, rows, mn) in [
            (&self.a_layout, self.m / self.cta_group, self.a_mn),
            (&self.b_layout, self.n / self.cta_group, self.b_mn),
        ] {
            layout.verify(ctx)?;
            let expected =
                shared_layout(rows, self.k, mn).map_err(|e| pliron::verify_error_noloc!("{e}"))?;
            if *layout != expected {
                return verify_err_noloc!(
                    "SM100 MMA physical operand layout disagrees with shape or major order"
                );
            }
        }
        Ok(())
    }
}
/// One FP16 TMA tile's physical shared placement and participating CTA group.
///
/// `layout` is one CTA's tile, in element offsets, as produced by
/// [`shared_layout`]. Group one or two is supported; the descriptor's three
/// axes encode the tile's contiguous span, rows, and remaining columns.
#[pliron_attr(name = "cute.sm100_tma", format = "`<` $layout `,` $cta_group `>`")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100TmaAttr {
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteComposedLayoutAttr,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
}
impl Verify for CuteSm100TmaAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        if !matches!(self.cta_group, 1 | 2) {
            return verify_err_noloc!("SM100 TMA requires CTA group1 or2");
        }
        let sw = self.layout.0.inner();
        if !matches!(sw.num_bits, 2 | 3)
            || sw.num_base != 3
            || sw.num_shift != 3
            || self.layout.0.offset() != 0
        {
            return verify_err_noloc!("SM100 TMA requires zero-offset tiled B64/B128 FP16 layout");
        }
        let modes = self.layout.0.outer().modes();
        if modes.len() != 2 {
            return verify_err_noloc!("SM100 TMA layout must have two logical modes");
        }
        let extent = |index: usize| {
            modes[index]
                .checked_size()
                .and_then(|n| u32::try_from(n).ok())
        };
        let (Some(rows), Some(cols)) = (extent(0), extent(1)) else {
            return verify_err_noloc!("SM100 TMA layout extents must fit u32");
        };
        if ![false, true]
            .into_iter()
            .any(|mn| shared_layout(rows, cols, mn).is_ok_and(|canonical| canonical == self.layout))
        {
            return verify_err_noloc!(
                "SM100 TMA layout must match a canonical FP16 tiled shared layout"
            );
        }
        Ok(())
    }
}
/// Scalar and address-space requirements for one independent attention operand.
#[derive(Clone, Copy)]
enum Carrier {
    /// Unsigned 32-bit index or encoded tensor-memory address.
    U32,
    /// Signed 32-bit physical tensor-map coordinate.
    I32,
    /// One-bit predicate selecting whether MMA retains the previous accumulator.
    Bool,
    /// FP16 pointer in CTA shared memory, with either read-only or writable access.
    Shared,
    /// Writable FP16 pointer in CTA shared memory.
    SharedWrite,
    /// Immutable tensor-map pointer in generic, global, or constant space.
    Descriptor,
    /// Writable CTA-shared pointer to a 64-bit completion barrier.
    Barrier,
}
/// Check scalar width or shared/descriptor pointer requirements for an operand.
fn carrier(ctx: &Context, value: Value, want: Carrier) -> bool {
    let ty = value.get_type(ctx).deref(ctx);
    match want {
        Carrier::U32 | Carrier::I32 | Carrier::Bool => ty
            .downcast_ref::<IntegerType>()
            .is_some_and(|i| i.width() == if matches!(want, Carrier::Bool) { 1 } else { 32 }),
        _ => {
            let Some(p) = ty.downcast_ref::<MirPtrType>() else {
                return false;
            };
            if matches!(want, Carrier::Descriptor) {
                return !p.is_mutable
                    && matches!(
                        p.address_space,
                        address_space::GENERIC | address_space::GLOBAL | address_space::CONSTANT
                    );
            }
            if p.address_space != address_space::SHARED {
                return false;
            }
            if matches!(want, Carrier::Barrier) {
                return p.is_mutable
                    && p.pointee
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .is_some_and(|i| i.width() == 64);
            }
            (!matches!(want, Carrier::SharedWrite) || p.is_mutable)
                && p.pointee.deref(ctx).downcast_ref::<MirFP16Type>().is_some()
        }
    }
}
/// Generate independent zero-result MMA/TMA operations with typed plans and carrier checks.
macro_rules! operation{
    ($(#[$doc:meta])* $ty:ident,$name:literal,$key:ident,$attr:ty,[$($arg:ident),*])=>{
        #[pliron_op(name=$name,format,attributes=($key:$attr))]pub struct $ty;
        // Keep contracts on the impl and constructor: pliron's generated
        // attribute-table links do not compose correctly with local struct docs.
        $(#[$doc])*
        impl $ty{
            $(#[$doc])*
            pub fn new(ctx:&mut Context,operands:Vec<Value>,plan:$attr)->Self{let op=Self{op:Operation::new(ctx,Self::get_concrete_op_info(),vec![],operands,vec![],0)};op.get_operation().deref_mut(ctx).attributes.set(stringify!($key).try_into().unwrap(),plan);op}
            /// View an existing operation without allocating or verifying it.
            pub fn wrap(op:Ptr<Operation>)->Self{Self{op}}
            /// Clone the typed MMA/TMA plan, or return `None` if the attribute is missing.
            pub fn plan(&self,ctx:&Context)->Option<$attr>{self.get_operation().deref(ctx).attributes.get::<$attr>(&stringify!($key).try_into().unwrap()).cloned()}
        }
        impl Verify for $ty{
            fn verify(&self,ctx:&Context)->Result<(),Error>{
                let op=self.get_operation().deref(ctx);let expected=&[$(Carrier::$arg),*];
                if op.get_num_results()!=0||op.get_num_operands()!=expected.len(){return verify_err!(op.loc(),"{} has invalid operand/result count",$name);}
                let Some(plan)=self.plan(ctx)else{return verify_err!(op.loc(),"{} lost its typed plan",$name)};plan.verify(ctx)?;
                for (i,c) in expected.iter().enumerate(){if !carrier(ctx,op.get_operand(i),*c){return verify_err!(op.loc(),"{} operand {} violates its carrier/address-space contract",$name,i);}}
                Ok(())
            }
        }
    }
}
operation!(
    /// Issue shared A × shared B into FP32 TMEM from one elected leader lane.
    ///
    /// Operands: `[destination, a_base, b_base, accumulate, a_bytes, b_bytes]`.
    /// Byte offsets select complete shared tiles and preserve the full swizzle
    /// period. `accumulate` controls the first K step; later steps accumulate.
    /// Issue after input completion and keep all allocations live until MMA completes.
    CuteSm100MmaSsOp,
    "cute.sm100_mma_ss",
    sm100_attention_mma_ss_plan,
    CuteSm100MmaAttr,
    [U32, Shared, Shared, Bool, U32, U32]
);
operation!(
    /// Issue TMEM A × shared B into distinct FP32 TMEM from one elected lane.
    ///
    /// Operands: `[destination, a_tmem, b_base, accumulate, b_bytes]`.
    /// A contains packed FP16 values; B's byte offset selects a complete tile.
    /// A and destination must not overlap. Inputs must be visible to MMA, and
    /// all allocations must remain live until completion.
    CuteSm100MmaTsOp,
    "cute.sm100_mma_ts",
    sm100_attention_mma_ts_plan,
    CuteSm100MmaAttr,
    [U32, U32, Shared, Bool, U32]
);
operation!(
    /// Start a rank-three global-to-shared transfer from all producer-warp lanes.
    ///
    /// Operands: `[descriptor, destination, c0, c1, c2, barrier, rank]`.
    /// Coordinates are signed element indices on the descriptor's physical
    /// axes. Rank is the local CTA rank; group-two completion routes to rank
    /// zero, whose barrier expectation covers the whole group's copies.
    CuteSm100TmaLoadOp,
    "cute.sm100_attention_tma_load",
    sm100_attention_tma_load_plan,
    CuteSm100TmaAttr,
    [Descriptor, SharedWrite, I32, I32, I32, Barrier, U32]
);
operation!(
    /// Start a rank-three shared-to-global transfer from one elected issuer.
    ///
    /// Operands: `[descriptor, source, c0, c1, c2]`, using signed element
    /// coordinates on the descriptor's physical axes. Shared writes must be
    /// visible to the async proxy; keep the source live and unchanged until
    /// its committed store group finishes reading it.
    CuteSm100TmaStoreOp,
    "cute.sm100_attention_tma_store",
    sm100_attention_tma_store_plan,
    CuteSm100TmaAttr,
    [Descriptor, Shared, I32, I32, I32]
);
/// Apply a callback to the closed independent attention operation set.
macro_rules! each_op {
    ($m:ident) => {
        $m!(
            CuteSm100MmaSsOp,
            CuteSm100MmaTsOp,
            CuteSm100TmaLoadOp,
            CuteSm100TmaStoreOp
        )
    };
}
/// Register independent attention operations and their typed MMA/TMA attributes.
pub fn register(ctx: &mut Context) {
    /// Register each listed attention operation in the supplied IR context.
    macro_rules! reg{($($t:ty),*)=>{$(<$t>::register(ctx);)*}}
    each_op!(reg);
    CuteSm100MmaAttr::register(ctx);
    CuteSm100TmaAttr::register(ctx);
}
/// Return the operation IDs included in the independent attention family.
pub fn semantic_ids() -> Vec<OpId> {
    /// Collect the registered operation IDs from the supplied type list.
    macro_rules! ids{($($t:ty),*)=>{vec![$(<$t>::get_opid_static()),*]}}
    each_op!(ids)
}
/// Run this family's local verifier, returning `None` for an operation from another family.
pub fn verify_local(ctx: &Context, op: Ptr<Operation>) -> Option<Result<(), Error>> {
    let id = Operation::get_opid(op, ctx);
    /// Dispatch a family-local verifier when the operation ID matches.
    macro_rules! check{($($t:ty),*)=>{$(if id==<$t>::get_opid_static(){return Some(<$t>::wrap(op).verify(ctx));})*}}
    each_op!(check);
    None
}
/// Enforce one hardware CTA-group mode while permitting distinct per-resource
/// shapes within one function. Unlike the paired GEMM family, QK does not
/// constrain PV's layout. Group consistency across helper calls and allocation
/// lifetimes remain unsafe caller obligations.
pub fn verify_story(ctx: &Context, ops: &[Ptr<Operation>]) -> Result<(), String> {
    let mut group = None;
    for &op in ops {
        let id = Operation::get_opid(op, ctx);
        let plan = if id == CuteSm100MmaSsOp::get_opid_static() {
            CuteSm100MmaSsOp::wrap(op).plan(ctx)
        } else if id == CuteSm100MmaTsOp::get_opid_static() {
            CuteSm100MmaTsOp::wrap(op).plan(ctx)
        } else {
            None
        };
        if let Some(p) = plan {
            if group.is_some_and(|g| g != p.cta_group) {
                return Err("SM100 MMA operations in one kernel must agree on CTA group".into());
            }
            group = Some(p.cta_group);
            if id == CuteSm100MmaTsOp::get_opid_static() {
                if p.cta_group != 1 || p.a_mn {
                    return Err(
                        "SM100 TMEM-A attention supports group1 K-major probabilities".into(),
                    );
                }
                let o = op.deref(ctx);
                if o.get_operand(0) == o.get_operand(1) {
                    return Err("SM100 TMEM-A source cannot be its destination".into());
                }
            }
        }
    }
    Ok(())
}
