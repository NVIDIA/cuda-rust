/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::model::ImportedSelectionConstraints;

use crate::model::{
    BackendLoweringMechanism, CatalogSelection, ClcAdapter, DebugControlAdapter,
    ExtendedMinMaxAdapter, ImportedAddressSpace, PackedConversionSourceFormat, RuntimeValidation,
    ScalarArithmeticOperation,
};
use crate::render::collector_targets::generated_selection_alternatives;
use crate::render::common::llvm;
use crate::render::compat::render_compat_float;
use crate::render::families::{
    dot_products, packed_alu_register_constraint, packed_alus, packed_conversion_constraint,
    packed_conversion_ptx_mnemonic, packed_conversion_source, packed_conversion_source_width,
    packed_conversion_typed_llvm_name, packed_conversions, prmts, scalar_conversion_ptx_mnemonic,
    scalar_conversion_rounding_attr, scalar_conversion_saturation_attr,
};
use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn unsafe_safety_docs_never_reference_missing_addr() {
    let catalog = catalog_with_clc();
    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    let offenders = catalog
        .intrinsics
        .iter()
        .filter(|record| !record.rust.safe)
        .filter(|record| raw_abi_item(&raw, &record.id).contains("/// `addr`"))
        .filter(|record| {
            !record
                .rust
                .arguments
                .iter()
                .any(|argument| argument.starts_with('*'))
        })
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "unsafe safety docs reference `addr` without a pointer argument: {}",
        offenders.join(", ")
    );
}

#[test]
fn unsafe_safety_blocks_are_exclusive_to_one_family() {
    let catalog = catalog_with_clc();
    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    let mut families_by_block = BTreeMap::<&str, BTreeSet<&str>>::new();
    for record in catalog.intrinsics.iter().filter(|record| !record.rust.safe) {
        families_by_block
            .entry(raw_abi_safety_block(&raw, &record.id))
            .or_default()
            .insert(&record.family);
    }
    let collisions = families_by_block
        .values()
        .filter(|families| families.len() != 1)
        .collect::<Vec<_>>();
    assert!(
        collisions.is_empty(),
        "unsafe safety blocks shared across families: {collisions:?}; render a legitimate shared contract through a shared arm"
    );
}

#[test]
fn selection_alternatives_keep_predicates_and_constraints_grouped() {
    let selections = vec![
        CatalogSelection {
            source_record: "SELECT_A".into(),
            asm: "op.a $dst;".into(),
            predicates: vec!["HasA".into(), "HasCommon".into()],
            constraints: ImportedSelectionConstraints {
                address_space: Some(ImportedAddressSpace::Generic),
                immediate_bindings: vec![],
            },
        },
        CatalogSelection {
            source_record: "SELECT_B".into(),
            asm: "op.b $dst;".into(),
            predicates: vec!["HasB".into()],
            constraints: ImportedSelectionConstraints {
                address_space: Some(ImportedAddressSpace::Shared),
                immediate_bindings: vec![],
            },
        },
    ];

    let rendered = generated_selection_alternatives(&selections);
    assert_eq!(rendered.matches("GeneratedSelectionAlternative").count(), 2);
    assert!(rendered.contains(
            "source_record: \"SELECT_A\", asm: \"op.a $dst;\", predicates: &[\"HasA\", \"HasCommon\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Generic), immediate_bindings: &[] }"
        ));
    assert!(rendered.contains(
            "source_record: \"SELECT_B\", asm: \"op.b $dst;\", predicates: &[\"HasB\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Shared), immediate_bindings: &[] }"
        ));
}

#[test]
fn packed_alu_and_conversion_render_exact_pure_inline_ptx_adapters() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(packed_alus(&catalog).count(), 30);
    assert_eq!(packed_conversions(&catalog).count(), 18);

    let dialect = render_dialect_packed_alu(&catalog, "test-hash");
    for op in [
        "FmaBf16x2Op",
        "FmaReluBf16x2Op",
        "AddBf16x2Op",
        "SubBf16x2Op",
        "MulBf16x2Op",
        "MinBf16x2Op",
        "MaxBf16x2Op",
        "NegBf16x2Op",
        "AbsBf16x2Op",
        "FmaF16x2Op",
        "FmaReluF16x2Op",
        "AddF16x2Op",
        "SubF16x2Op",
        "MulF16x2Op",
        "MinF16x2Op",
        "MaxF16x2Op",
        "NegF16x2Op",
        "AbsF16x2Op",
        "AddF32x2Op",
        "AddFtzF32x2Op",
        "SubF32x2Op",
        "SubFtzF32x2Op",
        "MulF32x2Op",
        "MulFtzF32x2Op",
        "FmaF32x2Op",
        "FmaFtzF32x2Op",
    ] {
        assert!(dialect.contains(&format!("pub struct {op}")));
        assert!(dialect.contains(&format!("{op}::register(ctx)")));
    }
    let conversion_dialect = render_dialect_packed_conversion(&catalog, "test-hash");
    for op in [
        "CvtF32x2Bf16x2Op",
        "CvtF16x2F32Op",
        "CvtRzF16x2F32Op",
        "CvtRnReluF16x2F32Op",
        "CvtRnReluBf16x2F32Op",
        "CvtRzBf16x2F32Op",
        "CvtRnSatfiniteE4m3x2F32Op",
        "CvtRnSatfiniteReluE4m3x2F32Op",
        "CvtRnSatfiniteE5m2x2F32Op",
        "CvtRnSatfiniteReluE5m2x2F32Op",
    ] {
        assert!(conversion_dialect.contains(&format!("pub struct {op}")));
        assert!(conversion_dialect.contains(&format!("{op}::register(ctx)")));
    }
    assert!(conversion_dialect.contains("low f16 lane"));
    assert!(conversion_dialect.contains("low bf16 lane"));
    assert!(conversion_dialect.contains("low e4m3 lane"));
    assert!(conversion_dialect.contains("low e5m2 lane"));
    assert!(conversion_dialect.contains("vec![low, high]"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::bf16x2::fma_bf16x2"));
    assert!(importer.contains("cuda_device::f16x2::fma_f16x2"));
    assert!(importer.contains("cuda_device::convert::cvt_bf16x2_f32"));
    assert!(importer.contains("cuda_device::convert::cvt_f16x2_f32"));
    assert!(importer.contains("cuda_device::convert::cvt_rz_bf16x2_f32"));
    assert!(importer.contains("cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32"));
    assert!(importer.contains("cuda_device::convert::cvt_rn_satfinite_relu_e5m2x2_f32"));
    assert!(importer.contains("FmaBf16x2Op::build(ctx, arg0, arg1, arg2)"));
    assert!(importer.contains("CvtF32x2Bf16x2Op::build(ctx, arg0, arg1)"));

    let lowering = render_lowering(&catalog, "test-hash");
    for mnemonic in [
        "fma.rn.bf16x2",
        "fma.rn.relu.bf16x2",
        "add.rn.bf16x2",
        "sub.rn.bf16x2",
        "mul.rn.bf16x2",
        "min.bf16x2",
        "max.bf16x2",
        "neg.bf16x2",
        "abs.bf16x2",
        "fma.rn.f16x2",
        "fma.rn.relu.f16x2",
        "add.rn.f16x2",
        "sub.rn.f16x2",
        "mul.rn.f16x2",
        "min.f16x2",
        "max.f16x2",
        "neg.f16x2",
        "abs.f16x2",
    ] {
        assert!(lowering.contains(&format!(
            "convert_generated_packed_alu(ctx, rewriter, self.get_operation(), \"{mnemonic}\", 32)"
        )));
    }
    for mnemonic in [
        "add.rn.f32x2",
        "add.rn.ftz.f32x2",
        "sub.rn.f32x2",
        "sub.rn.ftz.f32x2",
        "mul.rn.f32x2",
        "mul.rn.ftz.f32x2",
        "fma.rn.f32x2",
        "fma.rn.ftz.f32x2",
    ] {
        assert!(lowering.contains(&format!(
            "convert_generated_packed_alu(ctx, rewriter, self.get_operation(), \"{mnemonic}\", 64)"
        )));
    }
    for mnemonic in [
        "cvt.rn.bf16x2.f32",
        "cvt.rn.f16x2.f32",
        "cvt.rz.f16x2.f32",
        "cvt.rn.relu.f16x2.f32",
        "cvt.rn.relu.bf16x2.f32",
        "cvt.rz.bf16x2.f32",
    ] {
        assert!(lowering.contains(&format!(
                "convert_generated_packed_f32x2(ctx, rewriter, self.get_operation(), None, \"{mnemonic}\", 32)"
            )));
    }
    for (intrinsic, mnemonic) in [
        ("llvm_nvvm_ff_to_e4m3x2_rn", "cvt.rn.satfinite.e4m3x2.f32"),
        (
            "llvm_nvvm_ff_to_e4m3x2_rn_relu",
            "cvt.rn.satfinite.relu.e4m3x2.f32",
        ),
        ("llvm_nvvm_ff_to_e5m2x2_rn", "cvt.rn.satfinite.e5m2x2.f32"),
        (
            "llvm_nvvm_ff_to_e5m2x2_rn_relu",
            "cvt.rn.satfinite.relu.e5m2x2.f32",
        ),
    ] {
        assert!(lowering.contains(&format!(
                "convert_generated_packed_f32x2(ctx, rewriter, self.get_operation(), Some(\"{intrinsic}\"), \"{mnemonic}\", 16)"
            )));
    }

    for record in packed_alus(&catalog) {
        let probe = render_probe(&catalog, record, "test-hash");
        let register = packed_alu_register_constraint(record);
        let constraints = std::iter::once(format!("={register}"))
            .chain(std::iter::repeat_n(
                register.to_owned(),
                record.rust.arguments.len(),
            ))
            .collect::<Vec<_>>()
            .join(",");
        assert!(probe.contains(&format!("\", \"{constraints}\"")));
        assert!(!probe.contains("asm sideeffect"));
        assert!(!probe.contains("~{memory}"));
    }
    let mut typed_probe_count = 0;
    for conversion in packed_conversions(&catalog) {
        let probe = render_probe(&catalog, conversion, "test-hash");
        if packed_conversion_typed_llvm_name(conversion).is_some() {
            typed_probe_count += 1;
            let llvm = llvm(conversion);
            let symbol = llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol);
            assert!(probe.contains(&format!("declare i16 @{symbol}(float, float)")));
            assert!(probe.contains(&format!("call i16 @{symbol}(float %high, float %low)")));
            assert!(!probe.contains(" asm "));
        } else if packed_conversion_source(conversion) == PackedConversionSourceFormat::F32x2 {
            assert!(probe.contains(&format!(
                "asm \"{} $0, $2, $1;\", \"{}\"(float %low, float %high)",
                packed_conversion_ptx_mnemonic(conversion),
                packed_conversion_constraint(conversion),
            )));
            assert!(!probe.contains("declare "));
        } else {
            // One packed source operand, so no high/low reordering.
            let source_ty = format!("i{}", packed_conversion_source_width(conversion));
            assert!(probe.contains(&format!(
                "asm \"{} $0, $1;\", \"{}\"({source_ty} %packed)",
                packed_conversion_ptx_mnemonic(conversion),
                packed_conversion_constraint(conversion),
            )));
            assert!(!probe.contains("declare "));
        }
        assert!(!probe.contains("asm sideeffect"));
    }
    assert_eq!(typed_probe_count, 4);

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub fn i0062(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
    assert!(raw.contains("pub fn i0071(_arg0: f32, _arg1: f32) -> u32"));
    assert!(raw.contains("pub fn i0072(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
    assert!(raw.contains("pub fn i0085(_arg0: f32, _arg1: f32) -> u32"));
    assert!(raw.contains("pub fn i0259(_arg0: f32, _arg1: f32) -> u16"));
    assert!(raw.contains("pub fn i0262(_arg0: f32, _arg1: f32) -> u16"));
    assert!(raw.contains("pub fn i0995(_arg0: u64, _arg1: u64) -> u64"));
    assert!(raw.contains("pub fn i1002(_arg0: u64, _arg1: u64, _arg2: u64) -> u64"));
    assert!(!raw.contains("#[must_use]\n#[inline(never)]\npub fn i0062"));
    let f16_raw = raw.find("pub fn i0072").unwrap();
    assert!(raw[..f16_raw].ends_with("#[must_use]\n#[inline(never)]\n"));

    let compatibility = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::Bf16x2);
    assert!(compatibility.contains("pub fn fma_bf16x2(arg0: u32, arg1: u32, arg2: u32)"));
    assert!(!compatibility.contains("fma_f16x2"));
    assert!(compatibility.contains("let _ = arg0;"));
    assert!(!compatibility.contains("let _ = (arg0);"));
    let compatibility = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::F16x2);
    let f16_compat = compatibility.find("pub fn fma_f16x2").unwrap();
    assert!(compatibility[..f16_compat].ends_with("#[must_use]\n#[inline(never)]\n"));
    assert!(!compatibility.contains("fma_bf16x2"));
    let compatibility = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::F32x2);
    assert!(compatibility.contains("pub fn add_f32x2(arg0: u64, arg1: u64) -> u64"));
    assert!(compatibility.contains("pub fn fma_ftz_f32x2(arg0: u64, arg1: u64, arg2: u64) -> u64"));
    assert!(!compatibility.contains("fma_f16x2"));
    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("`fma_f16x2` carries one packed `f16x2` value in a `u32`"));
    assert!(reference.contains(
        "native instruction starts at PTX 4.2 / `sm_53`; cuda-oxide admits it from sm_70+"
    ));
    assert!(reference.contains("LLVM-NVPTX PTX 6.0 on sm_70+"));
    assert!(reference.contains("libNVVM PTX 4.2 on sm_75+"));
    assert!(reference.contains(
            "`cvt_rn_relu_f16x2_f32` converts two `f32` inputs to packed `f16x2` using nearest-even rounding with ReLU"
        ));
    assert!(reference.contains("pure `cvt.rz.bf16x2.f32` inline PTX"));
    assert!(reference.contains(
            "LLVM-NVPTX uses typed `llvm.nvvm.ff.to.e4m3x2.rn` with `[high, low]` inputs; libNVVM uses pure `cvt.rn.satfinite.e4m3x2.f32` inline PTX"
        ));
    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/bf16x2.rs")));
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/f16x2.rs")));
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/f32x2.rs")));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/convert.rs"
    )));
    let compatibility = render_compat_packed_conversion(
        &catalog,
        "test-hash",
        "cuda_device::convert::",
        "convert",
        ("lo", "hi"),
    );
    assert!(compatibility.contains("pub fn cvt_bf16x2_f32(lo: f32, hi: f32) -> u32"));
    assert!(compatibility.contains("pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32"));
    assert!(compatibility.contains("pub fn cvt_rz_bf16x2_f32(lo: f32, hi: f32) -> u32"));
    assert!(compatibility.contains("pub fn cvt_rn_satfinite_e4m3x2_f32(lo: f32, hi: f32) -> u16"));
    assert!(
        compatibility.contains("pub fn cvt_rn_satfinite_relu_e5m2x2_f32(lo: f32, hi: f32) -> u16")
    );
    assert!(!compatibility.contains("pub fn cvt_f32x2_bf16x2("));
    assert!(!outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/tcgen05_conversion.rs"
    )));
}

#[test]
fn scalar_tf32_conversions_render_one_exact_attribute_carrier() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = scalar_conversions(&catalog).collect::<Vec<_>>();
    assert_eq!(records.len(), 10);

    for record in &records {
        assert_eq!(record.rust.arguments, ["f32"]);
        assert_eq!(record.rust.result, "u32");
        assert!(record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LlvmNvptx
                && lowering.mechanism == BackendLoweringMechanism::TypedNvvm
        }));
        assert!(record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LibNvvm
                && lowering.mechanism == BackendLoweringMechanism::InlinePtx
        }));
    }

    let dialect = render_dialect_scalar_conversion(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct ScalarConversionOp;").count(), 1);
    assert!(dialect.contains("pub enum ScalarConversionRoundingAttr"));
    assert!(dialect.contains("pub enum ScalarConversionSaturationAttr"));
    assert_eq!(
        dialect
            .matches("ScalarConversionOp::register(ctx);")
            .count(),
        1
    );
    for record in &records {
        assert!(dialect.contains(scalar_conversion_rounding_attr(record)));
        assert!(dialect.contains(scalar_conversion_saturation_attr(record)));
    }

    let importer = render_importer(&catalog, "test-hash");
    assert_eq!(
        importer
            .matches("let intrinsic = ScalarConversionOp::build")
            .count(),
        10
    );
    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for ScalarConversionOp")
            .count(),
        1
    );
    for record in &records {
        assert!(lowering.contains(&record.llvm_identifier()));
        assert!(lowering.contains(&scalar_conversion_ptx_mnemonic(record)));
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(&format!("declare i32 @{}(float)", llvm(record).symbol)));
    }

    let targets = render_targets(&catalog, "test-hash");
    let production_targets = render_targets_files(&catalog, "test-hash")
        .into_iter()
        .filter(|(path, _)| !path.to_string_lossy().contains("/tests/"))
        .map(|(_, contents)| contents)
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        production_targets
            .matches("GeneratedIntrinsicVariant::ScalarConversion {")
            .count(),
        11
    );
    assert!(targets.contains("Operation::get_op::<ScalarConversionOp>"));
    assert!(targets.contains("GeneratedScalarConversionRounding::NearestAway"));
    assert!(targets.contains("GeneratedScalarConversionSaturation::ReluSatfinite"));

    let compatibility = render_compat_packed_conversion(
        &catalog,
        "test-hash",
        "cuda_device::convert::",
        "convert",
        ("lo", "hi"),
    );
    assert!(compatibility.contains("pub fn cvt_rna_tf32_f32(value: f32) -> u32"));
    assert!(compatibility.contains("pub fn cvt_rz_relu_satfinite_tf32_f32(value: f32) -> u32"));
    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub fn i0368(_arg0: f32) -> u32"));
    assert!(raw.contains("pub fn i0377(_arg0: f32) -> u32"));
}

#[test]
fn packed_atomic_family_uses_one_attribute_dispatch_impl() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let rendered = render_lowering(&catalog, "test-hash");
    assert_eq!(
        rendered
            .matches("impl MirToLlvmConversion for PackedAtomicAddOp")
            .count(),
        1
    );
    assert_eq!(
        rendered.matches("PackedAtomicFormatAttr::F16x2)").count(),
        1
    );
    assert_eq!(
        rendered.matches("PackedAtomicFormatAttr::Bf16x2)").count(),
        1
    );
    for (op_type, ptx_type) in [
        ("NvvmAtomAddF16x2Op", "f16x2"),
        ("NvvmAtomAddBf16x2Op", "bf16x2"),
    ] {
        assert_eq!(
            rendered
                .matches(&format!("impl MirToLlvmConversion for {op_type}"))
                .count(),
            1
        );
        assert!(rendered.contains(&format!(
            "convert_packed_atom_add(ctx, rewriter, self.get_operation(), \"{ptx_type}\")"
        )));
    }

    let dialect = render_dialect_packed_atomic(&catalog, "test-hash");
    for (op_type, op_name) in [
        ("NvvmAtomAddF16x2Op", "nvvm.atom_add_f16x2"),
        ("NvvmAtomAddBf16x2Op", "nvvm.atom_add_bf16x2"),
    ] {
        assert!(dialect.contains(&format!("pub struct {op_type};")));
        assert!(dialect.contains(&format!("name = \"{op_name}\"")));
        assert!(dialect.contains(&format!("{op_type}::register(ctx);")));
    }
}

#[test]
fn packed_atomic_raw_abi_is_unsafe_and_must_use() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let rendered = render_raw_abi(&catalog, "test-hash").unwrap();
    for abi_id in ["i0014", "i0015"] {
        let signature = format!("pub unsafe fn {abi_id}(_arg0: *mut u32, _arg1: u32) -> u32");
        let index = rendered.find(&signature).unwrap();
        assert!(rendered[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
    }
}

#[test]
fn packed_atomic_compatibility_preserves_paths_signature_and_safety() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let rendered = render_compat_packed_atomic(&catalog, "test-hash");

    for name in ["atom_add_f16x2", "atom_add_bf16x2"] {
        let signature = format!("pub unsafe fn {name}(addr: *mut u32, val: u32) -> u32");
        let index = rendered.find(&signature).unwrap();
        assert!(rendered[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
        assert!(rendered.contains(&format!(
            "unreachable!(\"{name} called outside CUDA kernel context\")"
        )));
    }
    assert!(rendered.contains("relaxed GPU-scope operation"));
    assert!(rendered.contains("low lane first"));
    assert!(rendered.contains("may not form one old 32-bit snapshot"));
    assert!(rendered.contains("Requires PTX 6.2 and `sm_70+`"));
    assert!(rendered.contains("Requires PTX 7.8 and `sm_90+`"));
    assert!(rendered.contains("four writable, four-byte-aligned bytes in global memory"));
    assert!(rendered.contains("whole-word atomic or non-atomic lane access"));
    assert!(rendered.contains("Racing atomics must use mutually inclusive scopes"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert_eq!(
        outputs.get(&PathBuf::from("crates/cuda-device/src/generated/atomic.rs")),
        Some(&rendered)
    );
}

#[test]
fn dot_product_rendering_preserves_stable_paths_and_low_selector() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(dot_products(&catalog).count(), 4);

    let compatibility = render_compat_dotprod(&catalog, "test-hash");
    for name in ["dp4a_s32", "dp4a_u32", "dp2a_s32", "dp2a_u32"] {
        assert!(compatibility.contains(&format!("pub fn {name}(")));
    }

    let dialect = render_dialect_dotprod(&catalog, "test-hash");
    assert!(dialect.contains("pub struct Dp4aS32Op"));
    assert!(dialect.contains("pub struct Dp2aU32Op"));
    assert!(dialect.contains("NOpdsInterface<3>"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::dotprod::dp2a_s32"));
    assert!(importer.contains("let dot = Dp2aS32Op::build(ctx, a, b, c)"));
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, dot, \"v1:i0032\")"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for Dp4aS32Op"));
    assert!(lowering.contains("\"llvm_nvvm_idp2a_s_s\""));
    assert!(lowering.contains("\"dp2a.lo.s32.s32 $0, $1, $2, $3;\""));
    assert!(lowering.contains("\"dp2a.lo.s32.s32 $0, $1, $2, $3;\", true)"));

    let low = dot_products(&catalog)
        .find(|record| record.id == "dp2a_s32")
        .unwrap();
    let probe = render_probe(&catalog, low, "test-hash");
    assert!(probe.contains("call i32 @llvm.nvvm.idp2a.s.s(i32 %a, i32 %b, i1 false, i32 %c)"));
    assert!(!probe.contains("i1 true"));

    let target = render_targets(&catalog, "test-hash");
    assert!(target.contains("GeneratedImmediateBinding { argument_index: 2, value: 0 }"));
    assert!(!target.contains("GeneratedImmediateBinding { argument_index: 2, value: -1 }"));
}

#[test]
fn prmt_rendering_keeps_modes_and_zero_source_exact() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(prmts(&catalog).count(), 7);

    let dialect = render_dialect_prmt(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct PrmtOp").count(), 1);
    for mode in ["Generic", "F4e", "B4e", "Rc8", "Ecl", "Ecr", "Rc16"] {
        assert!(dialect.contains(mode));
    }

    let lowering = render_lowering(&catalog, "test-hash");
    for mode in ["rc8", "ecl", "ecr", "rc16"] {
        assert!(lowering.contains(&format!("prmt.b32.{mode} $0, $1, 0, $2;")));
    }
    assert!(lowering.contains("prmt.b32 $0, $1, $2, $3;"));
    assert!(lowering.contains("prmt.b32.f4e $0, $1, $2, $3;"));
    assert!(lowering.contains("prmt.b32.b4e $0, $1, $2, $3;"));
    assert!(lowering.contains("convert_generated_prmt"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("GeneratedIntrinsicVariant::Prmt"));
    assert!(targets.contains("Operation::get_op::<PrmtOp>"));
    assert!(targets.contains("GeneratedPrmtMode::Rc16"));

    let compatibility = render_compat_prmt(&catalog, "test-hash");
    assert_eq!(compatibility.matches("pub fn prmt").count(), 7);
}

#[test]
fn selected_special_register_probes_use_the_llvm_route() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();

    let clock = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == "clock")
        .unwrap();
    let typed = render_probe(&catalog, clock, "test-hash");
    assert!(typed.contains("declare i32 @llvm.nvvm.read.ptx.sreg.clock()"));
    assert!(typed.contains("call i32 @llvm.nvvm.read.ptx.sreg.clock()"));

    let gridid = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == "gridid")
        .unwrap();
    let inline = render_probe(&catalog, gridid, "test-hash");
    assert!(inline.contains("define i64 @probe_gridid_llvm_nvptx()"));
    assert!(inline.contains("call i64 asm \"mov.u64 $0, %gridid;\", \"=l\"()"));
}

#[test]
fn clc_rendering_preserves_api_and_uses_typed_llvm_routes() {
    let catalog = catalog_with_clc();
    validate_renderable(&catalog).unwrap();

    let compatibility = render_compat_clc(&catalog, "test-hash");
    assert!(
        compatibility
            .contains("pub unsafe fn clc_try_cancel(response: *mut u8, mbar: *mut Barrier)")
    );
    assert!(
        compatibility.contains(
            "pub unsafe fn clc_try_cancel_multicast(response: *mut u8, mbar: *mut Barrier)"
        )
    );
    assert!(
        compatibility
            .contains("pub unsafe fn clc_query_is_canceled(resp_lo: u64, resp_hi: u64) -> u32")
    );

    let dialect = render_dialect_clc(&catalog, "test-hash");
    for op in [
        "ClcTryCancelOp",
        "ClcTryCancelMulticastOp",
        "ClcQueryIsCanceledOp",
        "ClcQueryGetFirstCtaidXOp",
        "ClcQueryGetFirstCtaidYOp",
        "ClcQueryGetFirstCtaidZOp",
    ] {
        assert_eq!(dialect.matches(&format!("pub struct {op}")).count(), 1);
        assert!(dialect.contains(&format!("{op}::register(ctx)")));
    }
    assert_eq!(dialect.matches("NResultsInterface<0>").count(), 2);
    assert_eq!(dialect.matches("NResultsInterface<1>").count(), 4);
    assert_eq!(dialect.matches("impl Verify for ClcQuery").count(), 4);
    assert!(dialect.contains("is_integer_width(ctx, op.get_operand(0).get_type(ctx), 64)"));
    assert!(dialect.contains("is_integer_width(ctx, op.get_result(0).get_type(ctx), 32)"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::clc::clc_try_cancel"));
    assert!(importer.contains("ClcTryCancelOp::get_concrete_op_info()"));
    assert!(importer.contains("ClcQueryIsCanceledOp::get_concrete_op_info()"));
    // Destination is prepared before the intrinsic op is built, then the
    // result is written through it: two typed helper routes, in that order.
    assert!(importer.contains("helpers::prepare_destination_write("));
    assert!(importer.contains("helpers::emit_prepared_result_and_goto("));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("clc::convert_generated_clc_query"));
    assert!(lowering.contains("clc::convert_generated_clc_try_cancel"));
    assert!(lowering.contains(
            "convert_generated_clc_try_cancel(ctx, rewriter, self.get_operation(), operands_info, \"llvm__nvvm_dclusterlaunchcontrol_dtry_ucancel_dasync_dshared\")"
        ));
    assert!(lowering.contains(
            "convert_generated_clc_query(ctx, rewriter, self.get_operation(), operands_info, \"llvm__nvvm_dclusterlaunchcontrol_dquery_ucancel_dis_ucanceled\", true)"
        ));

    let request = clc_intrinsics(&catalog)
        .find(|record| record.id == "clc_try_cancel")
        .unwrap();
    let request_probe = render_probe(&catalog, request, "test-hash");
    assert_eq!(request_probe.matches("addrspacecast ptr").count(), 2);
    assert!(request_probe.contains(
            "call void @llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared(ptr addrspace(3) %response, ptr addrspace(3) %mbarrier)"
        ));

    let query = clc_intrinsics(&catalog)
        .find(|record| record.id == "clc_query_is_canceled")
        .unwrap();
    let query_probe = render_probe(&catalog, query, "test-hash");
    assert!(query_probe.contains("%response_high_shifted = shl i128 %response_high_i128, 64"));
    assert!(query_probe.contains("%response = or i128 %response_low_i128, %response_high_shifted"));
    assert!(query_probe.contains("%result = zext i1 %raw_result to i32"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains(
            "GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::ExactArchitecture(100), GeneratedHardwareAlternative::ExactArchitecture(101), GeneratedHardwareAlternative::ExactArchitecture(103), GeneratedHardwareAlternative::ExactArchitecture(110), GeneratedHardwareAlternative::ExactArchitecture(120), GeneratedHardwareAlternative::ExactArchitecture(121)])"
        ));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/clc.rs")));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/clc.rs"
    )));

    let mut wrong_adapter = catalog;
    wrong_adapter
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "clc_query_is_canceled")
        .unwrap()
        .clc
        .as_mut()
        .unwrap()
        .adapter = ClcAdapter::PairU64ToI128U32;
    assert!(validate_renderable(&wrong_adapter).is_err());
}

#[test]
fn debug_control_rendering_preserves_api_immediates_and_side_effects() {
    let catalog = catalog_with_debug_controls();
    validate_renderable(&catalog).unwrap();
    assert_eq!(debug_controls(&catalog).count(), 3);

    let compatibility = render_compat_debug_control(&catalog, "test-hash");
    assert!(compatibility.contains("pub fn trap() -> !"));
    assert!(compatibility.contains("pub fn breakpoint()"));
    assert!(compatibility.contains("pub fn prof_trigger<const N: u32>()"));
    assert!(compatibility.contains("pub(crate) fn __prof_trigger(_event_id: u32)"));
    assert!(compatibility.contains("__prof_trigger(N);"));

    let dialect = render_dialect_debug_control(&catalog, "test-hash");
    for op in ["TrapOp", "BreakpointOp", "PmEventOp"] {
        assert_eq!(dialect.matches(&format!("pub struct {op}")).count(), 1);
        assert!(dialect.contains(&format!("{op}::register(ctx)")));
    }
    assert!(dialect.contains("event_id: u32"));
    assert!(dialect.contains("pub fn new_with_event_id"));
    assert!(dialect.contains("pub fn get_event_id"));
    assert!(dialect.contains("filter(|value| *value <= 15)"));
    assert!(dialect.contains("requires a u32 event ID in 0..=15"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::debug::__prof_trigger"));
    assert!(importer.contains("mir::Operand::Constant"));
    assert!(importer.contains("u32::try_from(value)"));
    assert!(importer.contains("filter(|value| *value <= 15)"));
    assert!(importer.contains("PmEventOp::build(ctx, event_id)"));
    assert!(importer.contains("MirUnreachableOp::get_concrete_op_info()"));
    assert!(importer.contains("prof_trigger requires a compile-time constant event ID in 0..=15"));

    let lowering = render_lowering(&catalog, "test-hash");
    for op in ["TrapOp", "BreakpointOp", "PmEventOp"] {
        assert!(lowering.contains(&format!("impl MirToLlvmConversion for {op}")));
    }
    assert!(lowering.contains(
        "inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], \"trap;\", \"\")"
    ));
    assert!(lowering.contains(
        "inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], \"brkpt;\", \"\")"
    ));
    assert!(lowering.contains("let template = format!(\"pmevent {event_id};\")"));

    for record in debug_controls(&catalog) {
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("call void asm sideeffect"));
        assert!(!probe.contains("attributes #0 = { convergent }"));
    }

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/debug_control.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/debug_control.rs"
    )));

    let mut wrong_adapter = catalog.clone();
    wrong_adapter
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "pmevent")
        .unwrap()
        .debug_control
        .as_mut()
        .unwrap()
        .adapter = DebugControlAdapter::Direct;
    assert!(validate_renderable(&wrong_adapter).is_err());

    let mut wrong_runtime = catalog;
    wrong_runtime
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "trap")
        .unwrap()
        .debug_control
        .as_mut()
        .unwrap()
        .runtime_validation = RuntimeValidation::Executed;
    assert!(validate_renderable(&wrong_runtime).is_err());
}

#[test]
fn scalar_arithmetic_rendering_uses_one_closed_carrier() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = scalar_arithmetics(&catalog).collect::<Vec<_>>();
    assert_eq!(records.len(), 64);
    assert!(records.iter().all(|record| {
        record.semantics.pure
            == (record.scalar_arithmetic.as_ref().unwrap().operation
                != ScalarArithmeticOperation::Div)
    }));

    let compatibility = render_compat_float(&catalog, "test-hash");
    assert!(compatibility.contains("pub fn mul_rn_f64(arg0: f64, arg1: f64) -> f64"));
    assert!(
        compatibility.contains("pub fn fma_rp_ftz_sat_f32(arg0: f32, arg1: f32, arg2: f32) -> f32")
    );
    assert!(compatibility.contains("pub fn add_rp_ftz_sat_f32(arg0: f32, arg1: f32) -> f32"));
    assert_eq!(compatibility.matches("#[must_use]").count(), 113);

    let dialect = render_dialect_scalar_arithmetic(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct ScalarArithmeticOp").count(), 1);
    for attr in [
        "ScalarArithmeticFormatAttr",
        "ScalarArithmeticOperationAttr",
        "ScalarArithmeticRoundingAttr",
        "ScalarArithmeticSubnormalAttr",
        "ScalarArithmeticSaturationAttr",
    ] {
        assert!(dialect.contains(attr));
        assert!(dialect.contains(&format!("{attr}::register(ctx)")));
    }
    assert!(dialect.contains("variant is not admitted"));
    assert!(dialect.contains("ScalarArithmeticOperationAttr::Fma => 3"));

    let importer = render_importer(&catalog, "test-hash");
    assert_eq!(
        importer.matches("ScalarArithmeticOp::build(ctx").count(),
        64
    );
    assert!(importer.contains("cuda_device::float::fma_rp_ftz_sat_f32"));
    assert!(importer.contains("cuda_device::float::add_rp_ftz_sat_f32"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for ScalarArithmeticOp")
            .count(),
        1
    );
    assert!(lowering.contains("llvm_nvvm_mul_rn_d"));
    assert!(lowering.contains("fma.rp.ftz.sat.f32"));
    assert!(lowering.contains("llvm_nvvm_add_rn_d"));
    assert!(lowering.contains("add.rp.sat.ftz.f32"));
    assert!(lowering.contains("recipe.0, recipe.1, recipe.2, recipe.3"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("GeneratedIntrinsicVariant::ScalarArithmetic"));
    assert!(targets.contains("GeneratedScalarArithmeticFormat::F64"));
    assert!(targets.contains("GeneratedScalarArithmeticSaturation::Sat"));
    assert!(targets.contains("Operation::get_op::<ScalarArithmeticOp>"));

    let f64_mul = records
        .iter()
        .copied()
        .find(|record| record.id == "mul_rn_f64")
        .unwrap();
    let f64_probe = render_probe(&catalog, f64_mul, "test-hash");
    assert!(f64_probe.contains("declare double @llvm.nvvm.mul.rn.d(double, double)"));
    let f32_sat = records
        .iter()
        .copied()
        .find(|record| record.id == "fma_rp_ftz_sat_f32")
        .unwrap();
    let f32_probe = render_probe(&catalog, f32_sat, "test-hash");
    assert!(
        f32_probe.contains("call float asm \"fma.rp.ftz.sat.f32 $0, $1, $2, $3;\", \"=f,f,f,f\"")
    );
    assert!(!f32_probe.contains("@llvm.nvvm.fma.rp.ftz.sat.f"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/float.rs")));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/scalar_arithmetic.rs"
    )));
}

#[test]
fn extended_minmax_rendering_is_closed_pure_and_exact() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = extended_minmax(&catalog).collect::<Vec<_>>();
    assert_eq!(records.len(), 52);
    assert_eq!(
        records
            .iter()
            .filter(|record| record.target.minimum_ptx.encoded() == 70)
            .count(),
        20
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| record.target.minimum_ptx.encoded() == 72)
            .count(),
        32
    );
    for id in ["min_f16x2", "max_f16x2", "min_bf16x2", "max_bf16x2"] {
        assert_eq!(
            catalog
                .intrinsics
                .iter()
                .find(|record| record.id == id)
                .unwrap()
                .family,
            "packed_alu"
        );
        assert!(records.iter().all(|record| record.id != id));
    }

    let f16 = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::F16x2);
    assert!(f16.contains("pub fn min_ftz_nan_xorsign_abs_f16x2(a: u32, b: u32) -> u32"));
    let bf16 = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::Bf16x2);
    assert!(bf16.contains("pub fn max_nan_xorsign_abs_bf16x2(a: u32, b: u32) -> u32"));
    let float = render_compat_float(&catalog, "test-hash");
    assert!(float.contains("pub fn min_ftz_nan_xorsign_abs_f32(a: f32, b: f32) -> f32"));

    let dialect = render_dialect_extended_minmax(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct ExtendedMinMaxOp").count(), 1);
    assert_eq!(
        dialect
            .matches("            (ExtendedMinMaxFormatAttr::")
            .count(),
        52
    );
    for attr in [
        "ExtendedMinMaxFormatAttr",
        "ExtendedMinMaxOperationAttr",
        "ExtendedMinMaxSubnormalAttr",
        "ExtendedMinMaxNanAttr",
        "ExtendedMinMaxXorSignAbsAttr",
    ] {
        assert!(dialect.contains(attr));
        assert!(dialect.contains(&format!("{attr}::register(ctx)")));
    }
    assert!(dialect.contains("variant is not admitted"));

    let importer = render_importer(&catalog, "test-hash");
    assert_eq!(importer.matches("ExtendedMinMaxOp::build(ctx").count(), 52);
    assert!(importer.contains("cuda_device::float::min_xorsign_abs_f32"));
    assert!(importer.contains("cuda_device::bf16x2::max_nan_bf16x2"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for ExtendedMinMaxOp")
            .count(),
        1
    );
    assert!(lowering.contains("convert_generated_extended_minmax"));
    assert!(lowering.contains("min.ftz.NaN.xorsign.abs.f32"));
    assert!(lowering.contains("max.NaN.xorsign.abs.bf16x2"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("GeneratedIntrinsicVariant::ExtendedMinMax"));
    assert!(targets.contains("GeneratedPtxVersion::from_encoded(72)"));
    assert!(targets.contains("Operation::get_op::<ExtendedMinMaxOp>"));
    assert!(targets.contains("ExtendedMinMaxXorSignAbsAttr::Enabled"));

    let packed_record = records
        .iter()
        .copied()
        .find(|record| record.id == "min_ftz_nan_f16x2")
        .unwrap();
    let packed_probe = render_probe(&catalog, packed_record, "test-hash");
    assert!(packed_probe.contains("call i32 asm \"min.ftz.NaN.f16x2 $0, $1, $2;\", \"=r,r,r\""));
    assert!(!packed_probe.contains("sideeffect"));
    assert!(!packed_probe.contains("convergent"));
    let f32_record = records
        .iter()
        .copied()
        .find(|record| record.id == "max_ftz_nan_xorsign_abs_f32")
        .unwrap();
    let f32_probe = render_probe(&catalog, f32_record, "test-hash");
    assert!(
        f32_probe
            .contains("call float asm \"max.ftz.NaN.xorsign.abs.f32 $0, $1, $2;\", \"=f,f,f\"")
    );
    // Scalar 16-bit forms ride in `h` registers, not the packed `r` pair.
    let half_record = records
        .iter()
        .copied()
        .find(|record| record.id == "min_ftz_nan_f16")
        .unwrap();
    let half_probe = render_probe(&catalog, half_record, "test-hash");
    assert!(half_probe.contains("call i16 asm \"min.ftz.NaN.f16 $0, $1, $2;\", \"=h,h,h\""));
    let bf16_record = records
        .iter()
        .copied()
        .find(|record| record.id == "max_nan_xorsign_abs_bf16")
        .unwrap();
    let bf16_probe = render_probe(&catalog, bf16_record, "test-hash");
    assert!(
        bf16_probe.contains("call i16 asm \"max.NaN.xorsign.abs.bf16 $0, $1, $2;\", \"=h,h,h\"")
    );
    assert!(lowering.contains("MinMaxCarrier::Half16"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/extended_minmax.rs"
    )));
    for module in ["float", "f16", "bf16", "f16x2", "bf16x2"] {
        assert!(outputs.contains_key(&PathBuf::from(format!(
            "crates/cuda-device/src/generated/{module}.rs"
        ))));
    }

    let mut wrong_adapter = catalog;
    wrong_adapter
        .intrinsics
        .iter_mut()
        .find(|record| record.id == "min_xorsign_abs_f32")
        .unwrap()
        .extended_minmax
        .as_mut()
        .unwrap()
        .adapter = ExtendedMinMaxAdapter::DirectPackedU32;
    assert!(validate_renderable(&wrong_adapter).is_err());
}

#[test]
fn extended_minmax_alone_renders_float_wrappers() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut catalog = crate::resolve::resolve(&repo_root).unwrap();
    catalog
        .intrinsics
        .retain(|record| record.family == "extended_minmax");

    let (path, float) = render_compat_float_output(&catalog, "test-hash")
        .expect("extended min/max alone must render float wrappers");
    assert_eq!(
        path,
        PathBuf::from("crates/cuda-device/src/generated/float.rs")
    );
    assert!(float.contains("pub fn min_xorsign_abs_f32(a: f32, b: f32) -> f32"));
    assert!(!float.contains("pub fn add_rn_f32"));
}

/// Every reference row must carry exactly the six cells of its header.
///
/// A `|` in cell data splits the row before inline parsing, so it splits
/// the cell even inside a code span. Three catalog entries write a PTX
/// destination pair as `<register|predicate>`, and their rows rendered with
/// a seventh cell, which drops the backend-evidence column. Counting
/// unescaped pipes per row is the direct check.
#[test]
fn reference_rows_have_one_cell_per_header_column() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let reference = render_reference(&catalog, "test-hash");

    let mut rows = 0usize;
    let mut carried_a_pipe = 0usize;
    for line in reference
        .lines()
        .filter(|line| line.starts_with("| `cuda_intrinsics::"))
    {
        rows += 1;
        if line.contains("\\|") {
            carried_a_pipe += 1;
        }
        // Delimiters only: drop every escaped pipe first, then a six-column
        // row leaves seven.
        let delimiters = line.replace("\\|", "").matches('|').count();
        assert_eq!(
            delimiters, 7,
            "row has {delimiters} cell delimiters, expected 7: {line}"
        );
    }

    assert!(
        rows > 900,
        "expected the full reference, rendered {rows} rows"
    );
    assert!(
        carried_a_pipe >= 3,
        "expected the elect.sync and match.all.sync patterns to need escaping, \
             found {carried_a_pipe} rows with an escaped pipe"
    );
}
