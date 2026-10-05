/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Workspace-internal requirements shared by the compiler, writer and runners.
//!
//! These are Tile IR compatibility levels, not CUDA driver versions. The
//! selected assembler determines which level we can emit. Requirements follow
//! the versioned dialect and its target restrictions:
//! <https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html>.
//! Operand/shape verification remains in `verify_target`; it is not a feature
//! availability test. Wire-layout branches remain in the bytecode writer.

use crate::bytecode::{BytecodeVersion, Opcode};
use crate::capabilities::{CapabilityError, TargetCapabilities};
use crate::ir::{Attribute, Location, ScalarType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Any,
    AtLeast(u32),
    Only(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirement {
    pub since: BytecodeVersion,
    pub architecture: Architecture,
}

impl Requirement {
    pub fn check(
        self,
        target: &TargetCapabilities,
        name: &str,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        target.require_version(name, self.since, location)?;
        self.check_architecture(target, name, location)
    }

    pub(crate) fn check_architecture(
        self,
        target: &TargetCapabilities,
        name: &str,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        match self.architecture {
            Architecture::Any => Ok(()),
            Architecture::AtLeast(sm) => target.require_sm(name, sm, location),
            Architecture::Only(sm) if target.sm() == Some(sm) => Ok(()),
            Architecture::Only(sm) => Err(target.error(name, &format!("sm_{sm}"), location)),
        }
    }
}

macro_rules! features {
    ($($feature:ident => ($name:literal, $since:ident, $arch:expr)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Feature { $($feature),+ }

        impl Feature {
            pub const ALL: &'static [Self] = &[$(Self::$feature),+];

            pub const fn name(self) -> &'static str {
                match self { $(Self::$feature => $name),+ }
            }

            pub const fn requirement(self) -> Requirement {
                match self {
                    $(Self::$feature => Requirement {
                        since: BytecodeVersion::$since,
                        architecture: $arch,
                    }),+
                }
            }

            pub const fn since(self) -> BytecodeVersion {
                self.requirement().since
            }

            pub fn check(self, target: &TargetCapabilities, location: &Location)
                -> Result<(), CapabilityError>
            {
                self.requirement().check(target, self.name(), location)
            }
        }
    };
}

use Architecture::{Any, AtLeast, Only};

features! {
    Baseline => ("baseline Tile IR", V13_2, AtLeast(80)),
    Allocation => ("alloca", V13_3, Any),
    Packing => ("pack/unpack", V13_3, Any),
    ScaledMma => ("block-scaled MMA", V13_3, AtLeast(100)),
    GatherScatterView => ("gather_scatter_view", V13_3, Any),
    StridedView => ("strided_view", V13_3, Any),
    AtomicRedView => ("atomic_red_view_tko", V13_3, Any),
    Insert => ("insert", V13_4, Any),
    IntegerPower => ("fpowi", V13_4, Any),
    GdcTokens => ("GDC tokens", V13_4, Any),
    AliasFence => ("memory_fence_alias_tko", V13_4, Any),
    Int4 => ("i4", V13_3, Any),
    Fp8 => ("f8E4M3FN/f8E5M2", V13_2, AtLeast(90)),
    E8Scale => ("f8E8M0FNU", V13_2, AtLeast(100)),
    Fp4 => ("f4E2M1FN", V13_3, AtLeast(100)),
    F8E5M3FNU => ("f8E5M3FNU", V13_4, AtLeast(107)),
    NewScaleMma => ("cuda_tile.mmaf_scaled with f8E5M3FNU scales", V13_4, Only(107)),
    PointerAttribute => ("ptr_attr", V13_4, Any),
    NearestAway => ("rounding_mode=nearest_away", V13_4, Any),
    ExtendedFloatRounding => ("ftof additional rounding modes (type-dependent)", V13_4, Any),
    SaturatingFToI => ("ftoi.saturating", V13_4, Any),
    Inbounds => ("inbounds=true", V13_4, Any),
    FastAccumulation => ("mmaf.fast_acc", V13_3, Any),
    ExpRounding => ("exp.rounding_mode (non-approximate)", V13_3, Any),
    Producer => ("module.producer", V13_3, Any),
    GlobalConstant => ("global.constant", V13_3, Any),
    GlobalVisibility => ("global.symbol_visibility", V13_3, Any),
    WorkerWarps => ("num_worker_warps_per_cta", V13_3, Any),
    ClusterCta => ("num_cta_in_cga greater than 1", V13_2, AtLeast(90)),
    WorkerWarpsRestricted => ("num_worker_warps_per_cta restricted to 4 or 8", V13_4, Any),
    Bf16AtomicAdd => ("bf16 atomic add", V13_3, AtLeast(90)),
    LoopReturn => ("return inside loop", V13_4, Any),
    HopperTarget => ("target sm_90", V13_3, Only(90)),
    RubinTarget => ("target sm_107", V13_4, Only(107)),
    ProgrammaticDependentLaunch => ("programmatic dependent launch", V13_4, AtLeast(90)),
}

/// Target names accepted by the dialect, paired with their earliest version
/// supported by this writer. This is not a numerically ordered feature ladder.
pub const TARGETS: &[(u32, Feature)] = &[
    (80, Feature::Baseline),
    (86, Feature::Baseline),
    (87, Feature::Baseline),
    (88, Feature::Baseline),
    (89, Feature::Baseline),
    (90, Feature::HopperTarget),
    (100, Feature::Baseline),
    (103, Feature::Baseline),
    (107, Feature::RubinTarget),
    (110, Feature::Baseline),
    (120, Feature::Baseline),
    (121, Feature::Baseline),
];

pub fn target_requirement(sm: u32) -> Option<Requirement> {
    TARGETS
        .iter()
        .find(|(target, _)| *target == sm)
        .map(|(_, feature)| feature.requirement())
}

pub const fn scalar_requirement(scalar: ScalarType) -> Requirement {
    use ScalarType::*;
    match scalar {
        I1 | I8 | I16 | I32 | I64 | F16 | BF16 | F32 | TF32 | F64 => Feature::Baseline,
        I4 => Feature::Int4,
        F8E4M3FN | F8E5M2 => Feature::Fp8,
        F8E8M0FNU => Feature::E8Scale,
        F4E2M1FN => Feature::Fp4,
        F8E5M3FNU => Feature::F8E5M3FNU,
    }
    .requirement()
}

pub fn attribute_feature(op: Opcode, name: &str, attr: &Attribute) -> Option<Feature> {
    match (name, attr) {
        ("rounding_mode", Attribute::Integer(7, _)) => Some(Feature::NearestAway),
        ("saturating", Attribute::Bool(true)) => Some(Feature::SaturatingFToI),
        ("inbounds", Attribute::Array(a))
            if a.iter().any(|v| matches!(v, Attribute::Bool(true))) =>
        {
            Some(Feature::Inbounds)
        }
        ("fast_acc", Attribute::Bool(true)) => Some(Feature::FastAccumulation),
        ("producer", _) => Some(Feature::Producer),
        ("rounding_mode", Attribute::Integer(n, _)) if op == Opcode::Exp && *n != 5 => {
            Some(Feature::ExpRounding)
        }
        _ => None,
    }
}

// The exhaustive match forces a requirements decision for each new opcode;
// the same declaration supplies enumeration for tests and documentation.
macro_rules! operations {
    ($($op:ident => $feature:ident),+ $(,)?) => {
        pub const fn opcode_requirement(op: Opcode) -> Requirement {
            match op { $(Opcode::$op => Feature::$feature),+ }.requirement()
        }
        pub const OPCODES: &[Opcode] = &[$(Opcode::$op),+];
    };
}

operations! {
    AbsF => Baseline,
    AbsI => Baseline,
    AddF => Baseline,
    AddI => Baseline,
    AndI => Baseline,
    Assert => Baseline,
    Assume => Baseline,
    AtomicCAS => Baseline,
    AtomicRMW => Baseline,
    Bitcast => Baseline,
    Break => Baseline,
    Broadcast => Baseline,
    Cat => Baseline,
    Ceil => Baseline,
    CmpF => Baseline,
    CmpI => Baseline,
    Constant => Baseline,
    Continue => Baseline,
    Cos => Baseline,
    CosH => Baseline,
    DivF => Baseline,
    DivI => Baseline,
    Entry => Baseline,
    Exp => Baseline,
    Exp2 => Baseline,
    ExtI => Baseline,
    Extract => Baseline,
    Floor => Baseline,
    Fma => Baseline,
    For => Baseline,
    FToF => Baseline,
    FToI => Baseline,
    GetGlobal => Baseline,
    GetIndexSpaceShape => Baseline,
    GetNumTileBlocks => Baseline,
    GetTensorShape => Baseline,
    GetTileBlockId => Baseline,
    Global => Baseline,
    If => Baseline,
    IntToPtr => Baseline,
    Iota => Baseline,
    IToF => Baseline,
    JoinTokens => Baseline,
    LoadPtrTko => Baseline,
    LoadViewTko => Baseline,
    Log => Baseline,
    Log2 => Baseline,
    Loop => Baseline,
    MakePartitionView => Baseline,
    MakeTensorView => Baseline,
    MakeToken => Baseline,
    MaxF => Baseline,
    MaxI => Baseline,
    MinF => Baseline,
    MinI => Baseline,
    MmaF => Baseline,
    MmaI => Baseline,
    Module => Baseline,
    MulF => Baseline,
    MulhiI => Baseline,
    MulI => Baseline,
    NegF => Baseline,
    NegI => Baseline,
    Offset => Baseline,
    OrI => Baseline,
    Permute => Baseline,
    Pow => Baseline,
    Print => Baseline,
    PtrToInt => Baseline,
    PtrToPtr => Baseline,
    Reduce => Baseline,
    RemF => Baseline,
    RemI => Baseline,
    Reshape => Baseline,
    Return => Baseline,
    Rsqrt => Baseline,
    Scan => Baseline,
    Select => Baseline,
    ShLI => Baseline,
    ShRI => Baseline,
    Sin => Baseline,
    SinH => Baseline,
    Sqrt => Baseline,
    StorePtrTko => Baseline,
    StoreViewTko => Baseline,
    SubF => Baseline,
    SubI => Baseline,
    Tan => Baseline,
    TanH => Baseline,
    TruncI => Baseline,
    XOrI => Baseline,
    Yield => Baseline,
    Atan2 => Baseline,
    Pack => Packing,
    Unpack => Packing,
    Alloca => Allocation,
    MmaFScaled => ScaledMma,
    MakeGatherScatterView => GatherScatterView,
    MakeStridedView => StridedView,
    AtomicRedViewTko => AtomicRedView,
    Insert => Insert,
    GdcLaunchDependentsTko => GdcTokens,
    GdcWaitTko => GdcTokens,
    FPowI => IntegerPower,
    MemoryFenceAliasTko => AliasFence,
}
