/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::catalog::TargetContract;
use super::contracts::{
    ActiveMask, Clc, ClcOperation, ClusterBarrier, ClusterBarrierMode, ClusterMemory,
    ClusterMemoryOperation, CpAsyncControl, CpAsyncCopy, CpAsyncMbarrier, DebugControl,
    DebugControlOperation, DotProduct, ExtendedMinMax, ExtendedMinMaxFormat, ExtendedMinMaxNan,
    ExtendedMinMaxOperation, ExtendedMinMaxSubnormal, IntegerMinMax, LdmatrixAdapter,
    LdmatrixSafety, LdmatrixVariant, MbarrierBasic, MbarrierExtended, MbarrierExtendedOperation,
    Movmatrix, PackedAlu, PackedAtomic, PackedConversion, PackedConversionDestinationFormat,
    PackedConversionSaturation, Prmt, PrmtMode, Redux, RegisterMma, RegisterMmaAccumulator,
    RegisterMmaElement, RegisterMmaOperation, RegisterMmaOverflow, RegisterMmaShape,
    ScalarArithmetic, ScalarArithmeticFormat, ScalarArithmeticOperation, ScalarArithmeticRounding,
    ScalarArithmeticSaturation, ScalarArithmeticSubnormal, ScalarConversion,
    ScalarConversionRounding, ScalarConversionSaturation, ScalarMath, ScalarMathFormat,
    ScalarMathOperation, ScalarMathPrecision, ScalarMathSubnormal, SparseMma, SparseMmaAccumulator,
    SparseMmaElement, SparseMmaMetadata, SparseMmaOverflow, SparseMmaShape, SpecialRegister,
    SpecialRegisterKind, StmatrixLayout, StmatrixMultiplicity, Tcgen05, Tcgen05CpGroup,
    Tcgen05CpMember, Tcgen05LdMultiplicity, Tcgen05LdShape, Tcgen05MmaAlias, Tcgen05MmaForm,
    Tcgen05Operation, Tma, TmaOperation, TmaReductionLoadMode, TmaReductionOperation, Vote,
    WarpBarrier, WarpMatch, WarpShuffle, WgmmaControl, WgmmaControlMode,
};
use super::core::{BackendLoweringMechanism, IntrinsicBackend, IntrinsicSource, RuntimeValidation};
use super::imported::ImportedAddressSpace;
use crate::ptx::InstructionPattern;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayFile {
    pub schema: u32,
    pub catalog_version: String,
    pub intrinsic_abi: u32,
    pub backend_profile: String,
    #[serde(default)]
    pub shards: Vec<String>,
    #[serde(rename = "intrinsic")]
    #[serde(default)]
    pub intrinsics: Vec<OverlayIntrinsic>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayShardFile {
    pub schema: u32,
    pub family: String,
    #[serde(rename = "intrinsic")]
    #[serde(default)]
    pub intrinsics: Vec<OverlayIntrinsic>,
    #[serde(default)]
    pub register_mma_int4: Option<RegisterMmaIntegerAdmission>,
    #[serde(default)]
    pub register_mma_int8: Option<RegisterMmaIntegerAdmission>,
    #[serde(default)]
    pub register_mma_b1: Option<RegisterMmaBinaryAdmission>,
    #[serde(default)]
    pub register_mma_f8f6f4_f32: Option<RegisterMmaF8F6F4Admission>,
    #[serde(default)]
    pub register_mma_f8f6f4_f16: Option<RegisterMmaF8F6F4Admission>,
    #[serde(default)]
    pub register_mma_mxf8f6f4_f32: Option<RegisterMmaF8F6F4Admission>,
    #[serde(default)]
    pub register_mma_fp8: Option<RegisterMmaFp8Admission>,
    #[serde(default)]
    pub register_mma_ampere_float: Option<RegisterMmaAmpereFloatAdmission>,
    #[serde(default, alias = "sparse_mma_int8")]
    pub sparse_mma_integer: Option<SparseMmaIntegerAdmission>,
    #[serde(default)]
    pub sparse_mma_f8f6f4_f32: Option<SparseMmaF8F6F4Admission>,
    #[serde(default)]
    pub sparse_mma_f8f6f4_f16: Option<SparseMmaF8F6F4F16Admission>,
    #[serde(default)]
    pub sparse_mma_fp8_f32: Option<SparseMmaFp8F32Admission>,
    #[serde(default)]
    pub sparse_mma_ordered_ampere_float: Option<SparseMmaOrderedAmpereFloatAdmission>,
    #[serde(default)]
    pub prmt: Option<PrmtAdmission>,
    #[serde(default)]
    pub packed_conversion_fp8: Option<PackedConversionFp8Admission>,
    #[serde(default)]
    pub packed_conversion_fp8_f16x2: Option<PackedConversionFp8F16x2Admission>,
    #[serde(default)]
    pub scalar_conversion: Option<ScalarConversionAdmission>,
    #[serde(default)]
    pub scalar_arithmetic: Option<ScalarArithmeticAdmission>,
    #[serde(default)]
    pub extended_minmax: Option<ExtendedMinMaxAdmission>,
    #[serde(default)]
    pub cluster_sreg: Option<ClusterSregAdmission>,
    #[serde(default)]
    pub cluster_barrier: Option<ClusterBarrierAdmission>,
    #[serde(default)]
    pub mbarrier_extended: Option<MbarrierExtendedAdmission>,
    #[serde(default)]
    pub special_registers: Option<SpecialRegisterAdmission>,
    #[serde(default)]
    pub debug_control: Option<DebugControlAdmission>,
    #[serde(default)]
    pub threadfence: Option<ThreadfenceAdmission>,
    #[serde(default)]
    pub cluster_memory: Option<ClusterMemoryAdmission>,
    #[serde(default)]
    pub stmatrix: Option<StmatrixAdmission>,
    #[serde(default)]
    pub clc: Option<ClcAdmission>,
    #[serde(default)]
    pub wgmma_controls: Option<WgmmaControlAdmission>,
    #[serde(default)]
    pub tma: Option<TmaAdmission>,
    #[serde(default)]
    pub tcgen05: Option<Tcgen05Admission>,
    #[serde(default)]
    pub scalar_math: Option<ScalarMathAdmission>,
}

/// Compact admission for unary scalar floating-point math operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarMathAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ScalarMathAdmissionVariant>,
}

/// One reviewed scalar math variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarMathAdmissionVariant {
    pub abi_id: String,
    #[serde(default)]
    pub libnvvm_evidence_profile: Option<String>,
    pub format: ScalarMathFormat,
    pub operation: ScalarMathOperation,
    pub precision: ScalarMathPrecision,
    pub subnormal: ScalarMathSubnormal,
}

/// Compact admission for the four existing `stmatrix` stores.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StmatrixAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<StmatrixAdmissionVariant>,
}

/// One reviewed `stmatrix` multiplicity and layout.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StmatrixAdmissionVariant {
    pub abi_id: String,
    pub multiplicity: StmatrixMultiplicity,
    pub layout: StmatrixLayout,
}

/// Compact admission for the remaining handwritten mbarrier operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtendedAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<MbarrierExtendedAdmissionVariant>,
}

/// One reviewed extended-mbarrier operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtendedAdmissionVariant {
    pub abi_id: String,
    pub operation: MbarrierExtendedOperation,
}

/// Compact admission for Hopper cluster special registers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterSregAdmission {
    pub axes: Vec<String>,
    pub xyz_product_count: usize,
    pub record_count: usize,
}

/// Compact admission for the six cluster-barrier instructions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrierAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClusterBarrierAdmissionVariant>,
}

/// One reviewed cluster-barrier spelling.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrierAdmissionVariant {
    pub abi_id: String,
    pub mode: ClusterBarrierMode,
}

/// Compact admission for cluster address mapping and remote shared reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemoryAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClusterMemoryAdmissionVariant>,
}

/// One reviewed cluster-memory operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemoryAdmissionVariant {
    pub abi_id: String,
    pub operation: ClusterMemoryOperation,
}

/// Compact admission for the three WGMMA control instructions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControlAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<WgmmaControlAdmissionVariant>,
}

/// One reviewed WGMMA control operation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControlAdmissionVariant {
    pub abi_id: String,
    pub mode: WgmmaControlMode,
}

/// Compact admission for the reviewed non-launch special-register reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegisterAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub registers: Vec<SpecialRegisterKind>,
    pub product_count: usize,
}

/// Compact admission for PTX debug controls.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebugControlAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub operations: Vec<DebugControlOperation>,
    /// Filled only when this pending shard is aggregated.
    #[serde(default)]
    pub abi_ids: Vec<String>,
}

/// Compact admission for the three CUDA thread fences.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadfenceAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ThreadfenceAdmissionVariant>,
}

/// One reviewed thread-fence scope.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadfenceAdmissionVariant {
    pub abi_id: String,
    pub scope: ThreadfenceScope,
}

/// Scope encoded by a PTX `membar` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadfenceScope {
    Cta,
    Device,
    System,
}

/// Compact admission for Cluster Launch Control.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClcAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClcAdmissionVariant>,
}

/// One reviewed Cluster Launch Control operation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClcAdmissionVariant {
    pub abi_id: String,
    pub operation: ClcOperation,
}

/// Compact admission for the existing TMA copy and group operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    #[serde(default)]
    pub reduce_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub reduce_libnvvm_evidence_profile: Option<String>,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<TmaAdmissionVariant>,
    #[serde(rename = "reduce_variant", default)]
    pub reduce_variants: Vec<TmaReductionAdmissionVariant>,
}

/// One reviewed TMA operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaAdmissionVariant {
    pub abi_id: String,
    pub operation: TmaOperation,
}

/// One reviewed tensor-reduction operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaReductionAdmissionVariant {
    pub abi_id: String,
    pub operation: TmaReductionOperation,
    pub load_mode: TmaReductionLoadMode,
    pub dimensions: u8,
}

/// Compact admission for the existing Tensor Core Generation 5 operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    #[serde(default)]
    pub cp_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub cp_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub ld_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub ld_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub st_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub st_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub offset_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub offset_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub control_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub control_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub mma_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub mma_libnvvm_evidence_profile: Option<String>,
    #[serde(rename = "mma_llvm_target_contract", default)]
    pub mma_llvm_target_contracts: Vec<TargetContract>,
    #[serde(rename = "mma_libnvvm_target_contract", default)]
    pub mma_libnvvm_target_contracts: Vec<TargetContract>,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<Tcgen05AdmissionVariant>,
    #[serde(rename = "cp_variant", default)]
    pub cp_variants: Vec<Tcgen05CpAdmissionVariant>,
    #[serde(rename = "ld_variant", default)]
    pub ld_variants: Vec<Tcgen05LdAdmissionVariant>,
    #[serde(rename = "st_variant", default)]
    pub st_variants: Vec<Tcgen05StAdmissionVariant>,
    #[serde(rename = "ld_offset_variant", default)]
    pub ld_offset_variants: Vec<Tcgen05LdAdmissionVariant>,
    #[serde(rename = "st_offset_variant", default)]
    pub st_offset_variants: Vec<Tcgen05StAdmissionVariant>,
    #[serde(rename = "mma_variant", default)]
    pub mma_variants: Vec<Tcgen05MmaAdmissionVariant>,
}

/// One reviewed tcgen05 operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05AdmissionVariant {
    pub abi_id: String,
    pub operation: Tcgen05Operation,
}

/// One reviewed tcgen05 copy member and CTA group.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05CpAdmissionVariant {
    pub abi_id: String,
    pub member: Tcgen05CpMember,
    pub group: Tcgen05CpGroup,
}

/// One reviewed tcgen05 load shape, repetition, and packing mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05LdAdmissionVariant {
    pub abi_id: String,
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub pack16: bool,
}

/// One reviewed tcgen05 store shape, repetition, and unpacking mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05StAdmissionVariant {
    pub abi_id: String,
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub unpack16: bool,
}

/// One reviewed tcgen05 MMA source form or compatibility alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05MmaAdmissionVariant {
    pub abi_id: String,
    pub form: Tcgen05MmaForm,
    #[serde(default)]
    pub alias: Option<Tcgen05MmaAlias>,
}

/// Compact admission for the closed `prmt` family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrmtAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<PrmtAdmissionVariant>,
}

/// One reviewed member of the `prmt` family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrmtAdmissionVariant {
    pub abi_id: String,
    pub mode: PrmtMode,
}

/// Compact admission for the closed scalar-f32 to packed-FP8 family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedConversionFp8Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub destination_formats: Vec<PackedConversionDestinationFormat>,
    pub saturations: Vec<PackedConversionSaturation>,
    pub product_count: usize,
}

/// Compact admission for packed FP8 conversions whose other side is `f16x2`.
///
/// Covers both directions: packing `f16x2` down to `e4m3x2`/`e5m2x2`, and
/// unpacking those back to `f16x2`. Both are single-operand conversions, unlike
/// the scalar-`f32` pair admitted by [`PackedConversionFp8Admission`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedConversionFp8F16x2Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub fp8_formats: Vec<PackedConversionFp8Format>,
    pub directions: Vec<PackedConversionFp8Direction>,
    pub relu_variants: bool,
    pub product_count: usize,
}

/// The FP8 side of an `f16x2` conversion pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionFp8Format {
    E4m3x2,
    E5m2x2,
}

/// Which way an FP8/`f16x2` conversion runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionFp8Direction {
    /// `f16x2` narrowed to packed FP8, always saturating to finite.
    Pack,
    /// Packed FP8 widened back to `f16x2`, which is always exact.
    Unpack,
}

/// Compact admission for scalar F32-to-TF32 conversions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversionAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ScalarConversionAdmissionVariant>,
}

/// One reviewed scalar conversion variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversionAdmissionVariant {
    pub abi_id: String,
    pub rounding: ScalarConversionRounding,
    pub saturation: ScalarConversionSaturation,
}

/// Compact admission for scalar floating-point arithmetic.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmeticAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ScalarArithmeticAdmissionVariant>,
}

/// One reviewed scalar arithmetic variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmeticAdmissionVariant {
    pub abi_id: String,
    pub format: ScalarArithmeticFormat,
    pub operation: ScalarArithmeticOperation,
    pub rounding: ScalarArithmeticRounding,
    pub subnormal: ScalarArithmeticSubnormal,
    pub saturation: ScalarArithmeticSaturation,
}

/// Compact admission for the exact extended floating-point min/max family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMaxAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ExtendedMinMaxAdmissionVariant>,
}

/// One reviewed extended min/max variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMaxAdmissionVariant {
    pub abi_id: String,
    pub format: ExtendedMinMaxFormat,
    pub operation: ExtendedMinMaxOperation,
    pub subnormal: ExtendedMinMaxSubnormal,
    pub nan: ExtendedMinMaxNan,
    pub xorsign_abs: bool,
}

/// Compact admission for a closed dense integer register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaIntegerAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaIntegerVariant>,
}

/// One reviewed member of a dense integer register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaIntegerVariant {
    pub shape: RegisterMmaShape,
    pub a_element: RegisterMmaElement,
    pub b_element: RegisterMmaElement,
    pub overflow: RegisterMmaOverflow,
}

/// Compact admission for the closed dense binary register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaBinaryAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaBinaryVariant>,
}

/// One reviewed member of the dense binary register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaBinaryVariant {
    pub shape: RegisterMmaShape,
    pub operation: RegisterMmaOperation,
}

/// Compact admission for one dense Blackwell `kind::f8f6f4` matrix.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaF8F6F4Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    /// Legacy shard metadata retained only so older overlays continue to parse.
    /// ABI identity is bound from the append-only ledger by catalog ID.
    #[serde(default, rename = "first_abi_id")]
    pub _legacy_first_abi_id: Option<String>,
    pub a_elements: Vec<RegisterMmaElement>,
    pub b_elements: Vec<RegisterMmaElement>,
    pub product_count: usize,
    pub targets: Vec<String>,
}

/// Compact admission for the standard FP8 register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaFp8Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    /// Legacy shard metadata retained only so older overlays continue to parse.
    /// ABI identity is bound from the append-only ledger by catalog ID.
    #[serde(default, rename = "first_abi_id")]
    pub _legacy_first_abi_id: Option<String>,
    pub shapes: Vec<RegisterMmaShape>,
    pub accumulators: Vec<RegisterMmaAccumulator>,
    pub a_elements: Vec<RegisterMmaElement>,
    pub b_elements: Vec<RegisterMmaElement>,
    pub product_count: usize,
}

/// Compact admission for the reviewed Ampere floating-point MMA forms.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaAmpereFloatAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    /// Legacy shard metadata retained only so older overlays continue to parse.
    /// ABI identity is bound from the append-only ledger by catalog ID.
    #[serde(default, rename = "first_abi_id")]
    pub _legacy_first_abi_id: Option<String>,
    pub product_count: usize,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaAmpereFloatVariant>,
}

/// One reviewed Ampere floating-point MMA form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaAmpereFloatVariant {
    pub shape: RegisterMmaShape,
    pub accumulator: RegisterMmaAccumulator,
    pub element: RegisterMmaElement,
}

/// Compact admission for a sparse integer register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaIntegerAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub metadata: SparseMmaMetadata,
    #[serde(rename = "variant")]
    pub variants: Vec<SparseMmaIntegerVariant>,
}

/// One reviewed member of a sparse integer register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaIntegerVariant {
    pub shape: SparseMmaShape,
    pub a_element: SparseMmaElement,
    pub b_element: SparseMmaElement,
    pub overflow: SparseMmaOverflow,
}

/// Compact admission for ordered sparse `kind::f8f6f4` F32 MMA.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaF8F6F4Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub a_elements: Vec<SparseMmaElement>,
    pub b_elements: Vec<SparseMmaElement>,
    pub product_count: usize,
}

/// Compact admission for ordered sparse `kind::f8f6f4` packed-F16 MMA.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaF8F6F4F16Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    /// Legacy shard metadata retained only so older overlays continue to parse.
    /// ABI identity is bound from the append-only ledger by catalog ID.
    #[serde(default, rename = "first_abi_id")]
    pub _legacy_first_abi_id: Option<String>,
    pub a_elements: Vec<SparseMmaElement>,
    pub b_elements: Vec<SparseMmaElement>,
    pub product_count: usize,
}

/// Compact admission for the reviewed plain (standard-metadata) SM89 sparse FP8
/// F32 MMA forms. Unlike the `kind::f8f6f4` families these carry no block-scale
/// qualifier: they are the ordinary `mma.sp.sync.aligned` forms that PTX ISA 8.4
/// introduced for `sm_89`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaFp8F32Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub a_elements: Vec<SparseMmaElement>,
    pub b_elements: Vec<SparseMmaElement>,
    pub product_count: usize,
}

/// Compact admission for the reviewed ordered-metadata Ampere floating sparse MMA forms.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaOrderedAmpereFloatAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<SparseMmaOrderedAmpereFloatVariant>,
}

/// One reviewed ordered-metadata Ampere floating sparse MMA form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaOrderedAmpereFloatVariant {
    pub shape: SparseMmaShape,
    pub accumulator: SparseMmaAccumulator,
    pub element: SparseMmaElement,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayIntrinsic {
    pub id: String,
    pub abi_id: String,
    pub operation_key: String,
    pub family: String,
    /// Imported LLVM records use the legacy `source_record` field below.
    /// PTX-native records must instead carry an explicit tagged source.
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    pub rust_module: String,
    pub rust_name: String,
    #[serde(default)]
    pub rust_arguments: Vec<String>,
    pub rust_result: String,
    pub safe: bool,
    #[serde(default)]
    pub must_use: bool,
    pub safe_allowlist_reason: Option<String>,
    pub public_rust_path: String,
    #[serde(default)]
    pub compatibility_rust_paths: Vec<String>,
    pub dialect_op_type: String,
    pub dialect_op_name: String,
    #[serde(default)]
    pub dialect_operands: Vec<String>,
    #[serde(default)]
    pub dialect_results: Vec<String>,
    #[serde(default)]
    pub llvm_symbol: Option<String>,
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Vec<String>,
    #[serde(default)]
    pub llvm_results: Vec<String>,
    pub pure: bool,
    pub memory: String,
    pub convergent: bool,
    pub execution_scope: String,
    pub minimum_ptx: String,
    #[serde(default)]
    pub minimum_sm: Option<String>,
    pub ptx_result: String,
    pub targets: String,
    pub ptx_isa_version: String,
    pub ptx_isa_section: String,
    pub ptx_isa_url: String,
    pub lowering: String,
    #[serde(default)]
    pub backend_lowerings: Vec<OverlayBackendLowering>,
    #[serde(default)]
    pub packed_atomic: Option<PackedAtomic>,
    #[serde(default)]
    pub redux: Option<Redux>,
    #[serde(default)]
    pub vote: Option<Vote>,
    #[serde(default)]
    pub active_mask: Option<ActiveMask>,
    #[serde(default)]
    pub warp_match: Option<WarpMatch>,
    #[serde(default)]
    pub warp_barrier: Option<WarpBarrier>,
    #[serde(default)]
    pub warp_shuffle: Option<WarpShuffle>,
    #[serde(default)]
    pub dot_product: Option<DotProduct>,
    #[serde(default)]
    pub packed_alu: Option<PackedAlu>,
    #[serde(default)]
    pub integer_minmax: Option<IntegerMinMax>,
    #[serde(default)]
    pub packed_conversion: Option<PackedConversion>,
    #[serde(default)]
    pub scalar_conversion: Option<ScalarConversion>,
    #[serde(default)]
    pub scalar_arithmetic: Option<ScalarArithmetic>,
    #[serde(default)]
    pub scalar_math: Option<ScalarMath>,
    #[serde(default)]
    pub extended_minmax: Option<ExtendedMinMax>,
    #[serde(default)]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default)]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default)]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default)]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default)]
    pub movmatrix: Option<Movmatrix>,
    #[serde(default)]
    pub mbarrier_extended: Option<MbarrierExtended>,
    #[serde(default)]
    pub register_mma: Option<RegisterMma>,
    #[serde(default)]
    pub sparse_mma: Option<SparseMma>,
    #[serde(default)]
    pub prmt: Option<Prmt>,
    #[serde(default)]
    pub cluster_barrier: Option<ClusterBarrier>,
    #[serde(default)]
    pub wgmma_control: Option<WgmmaControl>,
    #[serde(default)]
    pub special_register: Option<SpecialRegister>,
    #[serde(default)]
    pub debug_control: Option<DebugControl>,
    #[serde(default)]
    pub cluster_memory: Option<ClusterMemory>,
    #[serde(default)]
    pub clc: Option<Clc>,
    #[serde(default)]
    pub tma: Option<Tma>,
    #[serde(default)]
    pub tcgen05: Option<Tcgen05>,
    #[serde(default)]
    pub ldmatrix_variant: Option<LdmatrixVariant>,
    #[serde(default)]
    pub ldmatrix_safety: Option<LdmatrixSafety>,
    #[serde(default)]
    pub ldmatrix_adapter: Option<LdmatrixAdapter>,
    #[serde(default)]
    pub selected_address_space: Option<ImportedAddressSpace>,
    pub expected_ptx: InstructionPattern,
    pub summary: String,
}

/// Backend-specific lowering selected by reviewed evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayBackendLowering {
    pub backend: IntrinsicBackend,
    pub mechanism: BackendLoweringMechanism,
    pub evidence_profile: String,
    /// Optional exact target alternatives for this backend route.
    #[serde(default)]
    pub targets: Option<String>,
    /// Optional backend-profile floor. When absent, the intrinsic's native
    /// target requirement is used.
    #[serde(default)]
    pub minimum_ptx: Option<String>,
    #[serde(default)]
    pub minimum_sm: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SparseMmaElement;

    #[test]
    fn sparse_mma_admission_accepts_the_canonical_name_and_legacy_alias() {
        let canonical = r#"
schema = 25
family = "sparse_mma"

[sparse_mma_integer]
llvm_evidence_profile = "llvm"
libnvvm_evidence_profile = "libnvvm"
runtime_validation = "unexecuted"
metadata = "ordered"

[[sparse_mma_integer.variant]]
shape = "m16n8k64"
a_element = "s4"
b_element = "u4"
overflow = "wrapping"
"#;
        let parsed = toml::from_str::<OverlayShardFile>(canonical).unwrap();
        assert_eq!(
            parsed.sparse_mma_integer.unwrap().variants[0].b_element,
            SparseMmaElement::U4
        );

        let legacy = canonical.replace("sparse_mma_integer", "sparse_mma_int8");
        assert!(
            toml::from_str::<OverlayShardFile>(&legacy)
                .unwrap()
                .sparse_mma_integer
                .is_some()
        );
    }
}
