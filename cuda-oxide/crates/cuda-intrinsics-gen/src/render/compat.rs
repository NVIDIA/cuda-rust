/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogFile, CatalogHardwareAlternative, CatalogHardwareTarget, CatalogIntrinsic, ClcOperation,
    ClusterBarrierOrdering, ClusterMemoryOperation, CpAsyncCachePolicy, CpAsyncControlOperation,
    CpAsyncMbarrierOperation, CpAsyncSourceSize, DebugControlOperation, MbarrierBasicOperation,
    MbarrierExtendedOperation, PackedAluFormat, PackedAtomicFormat, PackedConversionSourceFormat,
    RegisterMmaAdapter, RegisterMmaCompatibilitySource, RegisterMmaOverflow, ScalarMathFormat,
    SparseMmaAdapter, SparseMmaCompatibilitySource, SparseMmaOverflow, Tcgen05LdShape,
    Tcgen05MmaForm, Tcgen05MmaSelectorLayout, Tcgen05Operation, TmaAdapter, TmaOperation,
    WgmmaControlMode,
};
use crate::render::common::{hardware_target_label, llvm, rust_header, source_label};
use crate::render::families::{
    ClcSafetyArgNames, clc_intrinsics, cluster_barriers, cluster_memory, cp_async_controls,
    cp_async_copies, cp_async_mbarriers, debug_controls, dot_products, execution_control_family,
    expected_ptx_head, extended_minmax, extended_minmax_rust_type, integer_minmaxes,
    is_blackwell_ldmatrix, ldmatrix, mbarrier_basics, mbarrier_extended, movmatrix,
    packed_alu_format_shape, packed_alus, packed_atomics, packed_conversion_rust_arguments,
    packed_conversion_source, packed_conversions, prmts, register_mmas, render_clc_safety_lines,
    scalar_arithmetic_arity, scalar_arithmetic_rust_type, scalar_arithmetics, scalar_conversions,
    scalar_math_contract, scalar_maths, sparse_mma_fragment_counts, sparse_mma_metadata_rule,
    sparse_mma_ptx_head, sparse_mma_selector_description, sparse_mmas, sregs, stmatrices,
    stmatrix_compatibility_name, stmatrix_variant, sync_intrinsics, tcgen05_intrinsics,
    tcgen05_is_commit, tcgen05_is_multicast_commit, tcgen05_is_shift, tcgen05_ld_register_count,
    tcgen05_mma_runtime_parameters, tcgen05_mma_selector_parameters, tcgen05_participation_doc,
    tcgen05_st_register_count, threadfence_ptx_level, tma_intrinsics, wgmma_control,
    wgmma_controls,
};
use std::fmt::Write as _;
use std::path::PathBuf;

pub(super) fn render_compat_register_mma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::wmma` to keep public paths stable.\n\n");
    for record in register_mmas(catalog).filter(|record| {
        record.register_mma.as_ref().is_some_and(|mma| {
            mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
        })
    }) {
        let mma = record
            .register_mma
            .as_ref()
            .expect("register-MMA semantics");
        let [path] = record.rust.compatibility_paths.as_slice() else {
            panic!("generated register-MMA API requires one compatibility path");
        };
        assert_eq!(path, &format!("cuda_device::wmma::{}", record.rust.name));
        assert!(!record.rust.safe);
        assert!(record.rust.must_use);
        let argument_names: &[&str] =
            if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 {
                &[
                    "c",
                    "a",
                    "b",
                    "scale_a",
                    "byte_id_a",
                    "thread_id_a",
                    "scale_b",
                    "byte_id_b",
                    "thread_id_b",
                ]
            } else {
                &["c", "a", "b"]
            };
        assert_eq!(record.rust.arguments.len(), argument_names.len());

        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Lowers to `{}`. C, A, and B are this lane's fragments in PTX register order.",
            expected_ptx_head(record)
        )
        .unwrap();
        writeln!(
            output,
            "/// Requires PTX {} and `{}`.",
            record.target.minimum_ptx,
            hardware_target_label(&record.target.hardware)
        )
        .unwrap();
        match mma.overflow {
            RegisterMmaOverflow::Wrapping => {
                output.push_str("/// Signed accumulator overflow wraps.\n");
            }
            RegisterMmaOverflow::Satfinite => {
                output.push_str(
                    "/// Signed accumulator overflow clamps to the finite `i32` range.\n",
                );
            }
            RegisterMmaOverflow::NotApplicable => {}
        }
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// All 32 lanes must execute the same instruction with the same qualifiers, and no lane may have exited.\n",
        );
        output.push_str(
            "/// `c`, `a`, and `b` must contain this lane's fragments in the documented PTX layout.\n",
        );
        if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 {
            output.push_str(
                "/// `scale_a` and `scale_b` contain this lane's packed scale data.\n\
                 /// For `scale_vec::1X`, `byte_id_a` and `byte_id_b` must be in `0..=3`, `thread_id_a` in `0..=1`, and `thread_id_b` in `0..=3`; other values make the PTX operation undefined.\n",
            );
        }
        writeln!(
            output,
            "/// See the [PTX MMA fragment layouts]({}).",
            record.target.ptx_isa_url
        )
        .unwrap();
        output.push_str("/// This register-only operation is not a memory fence.\n");
        if record.rust.arguments.len() > 7 {
            output.push_str("#[allow(clippy::too_many_arguments)]\n");
        }
        output.push_str("#[must_use]\n#[inline(never)]\n");
        let arguments = argument_names
            .iter()
            .copied()
            .zip(&record.rust.arguments)
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "pub unsafe fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        writeln!(output, "    let _ = ({});", argument_names.join(", ")).unwrap();
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_ldmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::wmma` to keep public paths stable.\n\n");
    for record in ldmatrix(catalog).filter(|record| {
        is_blackwell_ldmatrix(record) && !record.rust.compatibility_paths.is_empty()
    }) {
        let [path] = record.rust.compatibility_paths.as_slice() else {
            panic!("generated Blackwell ldmatrix API requires one compatibility path");
        };
        assert_eq!(path, &format!("cuda_device::wmma::{}", record.rust.name));
        assert_eq!(record.rust.arguments, ["*const u8"]);
        assert!(!record.rust.safe);
        assert!(record.rust.must_use);

        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", expected_ptx_head(record)).unwrap();
        writeln!(
            output,
            "/// Requires PTX {} and `{}`.",
            record.target.minimum_ptx,
            hardware_target_label(&record.target.hardware)
        )
        .unwrap();
        output.push_str(
            "///\n/// # Safety\n\
             /// All 32 lanes must execute the same instruction and qualifiers, with no exited lanes.\n\
             /// `smem_ptr` must satisfy the lane-address mapping, alignment, and readable-byte contract documented by PTX.\n\
             #[must_use]\n#[inline(never)]\n",
        );
        writeln!(
            output,
            "pub unsafe fn {}(smem_ptr: *const u8) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        output.push_str("    let _ = smem_ptr;\n");
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_sparse_mma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::wmma` to keep public paths stable.\n\n");
    for record in sparse_mmas(catalog).filter(|record| {
        record.sparse_mma.as_ref().is_some_and(|mma| {
            mma.compatibility_source == SparseMmaCompatibilitySource::GeneratedStub
        })
    }) {
        let mma = record.sparse_mma.as_ref().expect("sparse-MMA semantics");
        let [path] = record.rust.compatibility_paths.as_slice() else {
            panic!("generated sparse-MMA API requires one compatibility path");
        };
        assert_eq!(path, &format!("cuda_device::wmma::{}", record.rust.name));
        assert!(matches!(
            mma.adapter,
            SparseMmaAdapter::C2U32A2U32B2U32MetadataU32SelectorU32ToD2U32
                | SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32
                | SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32
                | SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
                | SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32
                | SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
        ));

        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Lowers to `{}`. C, A, B, and metadata are this lane's PTX fragments.",
            sparse_mma_ptx_head(record)
        )
        .unwrap();
        writeln!(
            output,
            "/// Requires PTX {} and `{}`.",
            record.target.minimum_ptx,
            hardware_target_label(&record.target.hardware)
        )
        .unwrap();
        match mma.overflow {
            SparseMmaOverflow::NotApplicable => {}
            SparseMmaOverflow::Wrapping => {
                output.push_str("/// Signed accumulator overflow wraps.\n");
            }
            SparseMmaOverflow::Satfinite => output
                .push_str("/// Signed accumulator overflow clamps to the finite `i32` range.\n"),
        }
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// All 32 lanes must execute the same instruction with the same qualifiers, and no lane may have exited.\n",
        );
        let (c_count, a_count, b_count, _) = sparse_mma_fragment_counts(record);
        writeln!(
            output,
            "/// `c`, `a`, and `b` must contain this lane's {c_count}-, {a_count}-, and {b_count}-register fragments; `metadata` must contain its sparse metadata."
        )
        .unwrap();
        writeln!(output, "/// {}", sparse_mma_metadata_rule(mma)).unwrap();
        writeln!(
            output,
            "/// `selector` must be {}.",
            sparse_mma_selector_description(record)
        )
        .unwrap();
        writeln!(
            output,
            "/// See the [PTX sparse MMA fragment layouts]({}).",
            record.target.ptx_isa_url
        )
        .unwrap();
        output.push_str("/// This register-only operation is not a memory fence.\n");
        output.push_str("#[must_use]\n#[inline(never)]\n");
        let arguments = ["c", "a", "b", "metadata", "selector"]
            .into_iter()
            .zip(&record.rust.arguments)
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "pub unsafe fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        output.push_str("    let _ = (c, a, b, metadata, selector);\n");
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_sreg(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// This file is included lexically inside `cuda_device::thread` so existing\n// DefPaths remain stable during the generated-intrinsics migration.\n\n",
    );
    for record in sregs(catalog) {
        let Some(path) = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::thread::"))
        else {
            continue;
        };
        let name = path.rsplit("::").next().unwrap();
        writeln!(
            output,
            "/// Compatibility spelling for `{}`.",
            record.rust.public_path
        )
        .unwrap();
        writeln!(
            output,
            "/// The compiler replaces this call with {}.",
            source_label(record)
        )
        .unwrap();
        output.push_str("#[allow(non_snake_case)]\n#[inline(never)]\n");
        let safety = if record.rust.safe { "" } else { "unsafe " };
        writeln!(
            output,
            "pub {safety}fn {name}() -> {} {{",
            record.rust.result
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_cluster_sreg(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// This file is included inside `cuda_device::cluster`.\n// Its private leaves let public helpers compose generated reads.\n\n",
    );
    for record in sregs(catalog) {
        let Some(path) = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::cluster::__"))
        else {
            continue;
        };
        let name = path.rsplit("::").next().unwrap();
        writeln!(
            output,
            "/// Private compatibility spelling for `{}`.",
            record.rust.public_path
        )
        .unwrap();
        writeln!(
            output,
            "/// The compiler replaces this call with `{}`.",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("#[allow(non_snake_case)]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub(crate) fn {name}() -> {} {{",
            record.rust.result
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_special_register_module(
    catalog: &CatalogFile,
    hash: &str,
    module: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "// Included inside `cuda_device::{module}` to keep existing paths stable.\n"
    )
    .unwrap();
    let prefix = format!("cuda_device::{module}::");
    for record in sregs(catalog).filter(|record| record.special_register.is_some()) {
        let Some(path) = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(&prefix))
        else {
            continue;
        };
        let name = path.rsplit("::").next().unwrap();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Generated from {}.", source_label(record)).unwrap();
        output.push_str("#[inline(never)]\n");
        let safety = if record.rust.safe { "" } else { "unsafe " };
        writeln!(
            output,
            "pub {safety}fn {name}() -> {} {{",
            record.rust.result
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_fence(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// This file is included inside `cuda_device::fence` so existing paths stay stable.\n\n",
    );
    for record in sync_intrinsics(catalog).filter(|record| threadfence_ptx_level(record).is_some())
    {
        let Some(path) = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::fence::"))
        else {
            continue;
        };
        let name = path.rsplit("::").next().unwrap();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// The compiler replaces this call with `{}`.",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(output, "pub fn {name}() {{").unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_dotprod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::dotprod` to keep existing paths stable.\n\n");
    for record in dot_products(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::dotprod::"))
            .expect("dot-product compatibility path");
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (0..record.rust.arguments.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        if record.rust.arguments.len() == 1 {
            writeln!(output, "    let _ = {values};").unwrap();
        } else {
            writeln!(output, "    let _ = ({values});").unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_prmt(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::prmt` to keep the public paths stable.\n\n");
    for record in prmts(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::prmt::"))
            .expect("prmt compatibility path");
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (0..record.rust.arguments.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(output, "pub fn {}({arguments}) -> u32 {{", record.rust.name).unwrap();
        writeln!(output, "    let _ = ({values});").unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_cluster_barrier(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::cluster` to keep public paths stable.\n\n");
    for record in cluster_barriers(catalog) {
        let barrier = record
            .cluster_barrier
            .as_ref()
            .expect("cluster-barrier contract");
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::cluster::"))
            .expect("cluster-barrier compatibility path");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// Each non-exited cluster thread must arrive exactly once before completion, then execute the matching wait.\n",
        );
        if barrier.aligned {
            output.push_str(
                "/// Every non-exited thread in the warp must execute this aligned instruction with identical control flow.\n",
            );
        }
        if barrier.ordering == ClusterBarrierOrdering::Relaxed {
            output.push_str(
                "/// This relaxed arrival does not publish earlier memory accesses; use the required cluster-scope fence first.\n",
            );
        }
        output.push_str("#[inline(never)]\n");
        writeln!(output, "pub unsafe fn {}() {{", record.rust.name).unwrap();
        writeln!(output, "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")").unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_cluster_memory(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::cluster` to keep its public API stable.\n\n");
    for record in cluster_memory(catalog) {
        let cluster = record
            .cluster_memory
            .as_ref()
            .expect("cluster-memory contract");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        output.push_str("///\n/// # Safety\n");
        match cluster.operation {
            ClusterMemoryOperation::MapSharedRank => {
                output.push_str(
                    "/// `local_ptr` must point into CTA shared memory, and `target_rank` must name a rank in the same live cluster.\n\
                     /// The result is a cluster-shared pointer in address space 7. Dereferencing it performs a remote DSMEM access; the target CTA must remain live and synchronization must make the access race-free.\n",
                );
                output.push_str("#[must_use]\n#[inline(never)]\n");
                output.push_str(
                    "pub unsafe fn map_shared_rank<T>(local_ptr: *const T, target_rank: u32) -> *const T {\n\
                             let _ = (local_ptr, target_rank);\n\
                             unreachable!(\"map_shared_rank called outside CUDA kernel context\")\n\
                         }\n\n",
                );
                output.push_str(
                    "/// Maps a mutable CTA-shared address to another cluster rank.\n\
                     ///\n\
                     /// # Safety\n\
                     /// The mapping requirements above apply. The target CTA must remain live, and remote writes must be synchronized and race-free.\n\
                     #[must_use]\n#[inline(never)]\n\
                     pub unsafe fn map_shared_rank_mut<T>(local_ptr: *mut T, target_rank: u32) -> *mut T {\n\
                             let _ = (local_ptr, target_rank);\n\
                             unreachable!(\"map_shared_rank_mut called outside CUDA kernel context\")\n\
                     }\n\n",
                );
            }
            ClusterMemoryOperation::ReadU32 => {
                output.push_str(
                    "/// `local_ptr` must identify an aligned readable `u32` in CTA shared memory. `target_rank` must name a rank in the same live cluster.\n\
                     /// The target CTA must publish the value before this weak cluster load.\n",
                );
                output.push_str("#[must_use]\n#[inline(never)]\n");
                output.push_str(
                    "pub unsafe fn dsmem_read_u32(local_ptr: *const u32, target_rank: u32) -> u32 {\n\
                             let _ = (local_ptr, target_rank);\n\
                             unreachable!(\"dsmem_read_u32 called outside CUDA kernel context\")\n\
                         }\n\n",
                );
            }
        }
    }
    output
}

pub(super) fn render_compat_movmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::wmma` to keep its public API stable.\n\n");
    for record in movmatrix(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::wmma::"))
            .expect("movmatrix compatibility path");
        assert_eq!(path, &format!("cuda_device::wmma::{}", record.rust.name));
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n\
             /// Each lane supplies two packed b16 values. The warp collectively transposes the 8x8 tile.\n\
             /// No addressable memory is touched, but the result depends on every lane's\n\
             /// input, so the call is modelled as reading and writing inaccessible state:\n\
             /// two calls with equal operands are not interchangeable.\n\
             ///\n\
             /// # Safety\n\
             /// All 32 warp lanes must execute the same call, and no lane may have exited.\n",
        );
        writeln!(
            output,
            "/// Requires PTX {} and `sm_75+`.",
            record.target.minimum_ptx
        )
        .unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub unsafe fn {}(value: u32) -> u32 {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = value;\n");
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_debug_control(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::debug` to keep its public API stable.\n\n");
    for operation in [
        DebugControlOperation::Trap,
        DebugControlOperation::Breakpoint,
        DebugControlOperation::Pmevent,
    ] {
        let record = debug_controls(catalog)
            .find(|record| {
                record
                    .debug_control
                    .as_ref()
                    .is_some_and(|debug| debug.operation == operation)
            })
            .expect("complete debug-control family");
        match operation {
            DebugControlOperation::Trap => {
                writeln!(output, "/// {}", record.summary).unwrap();
                output.push_str("#[inline(never)]\npub fn trap() -> ! {\n");
                output.push_str(
                    "    unreachable!(\"trap called outside CUDA kernel context\")\n}\n\n",
                );
            }
            DebugControlOperation::Breakpoint => {
                writeln!(output, "/// {}", record.summary).unwrap();
                output.push_str("#[inline(never)]\npub fn breakpoint() {\n");
                output.push_str(
                    "    unreachable!(\"breakpoint called outside CUDA kernel context\")\n}\n\n",
                );
            }
            DebugControlOperation::Pmevent => {
                output.push_str(
                    r#"/// Triggers performance monitor event `N`, which must be in `0..=15`.
#[inline(always)]
pub fn prof_trigger<const N: u32>() {
    __prof_trigger(N);
}

#[doc(hidden)]
#[inline(never)]
pub(crate) fn __prof_trigger(_event_id: u32) {
    unreachable!("prof_trigger called outside CUDA kernel context")
}

"#,
                );
            }
        }
    }
    output
}

pub(super) fn render_compat_stmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::tcgen05` to keep public paths stable.\n\n");
    for record in stmatrices(catalog) {
        let (multiplicity, _) = stmatrix_variant(record).expect("stmatrix variant");
        let count = multiplicity.register_count();
        let name = stmatrix_compatibility_name(record);
        let path = record
            .rust
            .compatibility_paths
            .first()
            .expect("stmatrix compatibility path");
        let arguments = std::iter::once("smem_ptr: *mut u8".to_owned())
            .chain((0..count).map(|index| format!("r{index}: u32")))
            .collect::<Vec<_>>()
            .join(", ");
        let values = std::iter::once("smem_ptr".to_owned())
            .chain((0..count).map(|index| format!("r{index}")))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// All warp lanes must participate. The used row addresses must be valid, shared, and 16-byte aligned.\n",
        );
        output.push_str("#[inline(never)]\n");
        writeln!(output, "pub unsafe fn {name}({arguments}) {{").unwrap();
        writeln!(output, "    let _ = ({values});").unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_clc(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::clc` to keep its public API stable.\n\n");
    for record in clc_intrinsics(catalog) {
        let operation = record.clc.as_ref().expect("CLC contract").operation;
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// # Safety\n");
        render_clc_safety_lines(&mut output, operation, ClcSafetyArgNames::Compatibility);
        match operation {
            ClcOperation::TryCancel | ClcOperation::TryCancelMulticast => {
                output.push_str("#[inline(never)]\n");
                writeln!(
                    output,
                    "pub unsafe fn {}(response: *mut u8, mbar: *mut Barrier) {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (response, mbar);\n");
            }
            ClcOperation::QueryIsCanceled
            | ClcOperation::QueryGetFirstCtaidX
            | ClcOperation::QueryGetFirstCtaidY
            | ClcOperation::QueryGetFirstCtaidZ => {
                output.push_str("#[inline(never)]\n");
                writeln!(
                    output,
                    "pub unsafe fn {}(resp_lo: u64, resp_hi: u64) -> u32 {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (resp_lo, resp_hi);\n");
            }
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_counted_barrier(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(
        execution_control_family(catalog, "counted_barrier").count(),
        4
    );
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::barrier` to keep its public API stable.\n\n");
    for record in execution_control_family(catalog, "counted_barrier") {
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// # Safety\n/// Every participating thread must use a compatible barrier ID and expected thread count.\n#[inline(never)]\n",
        );
        writeln!(
            output,
            "pub unsafe fn {}(barrier_id: u32, thread_count: u32) {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = (barrier_id, thread_count);\n");
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_grid_dependency(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(
        execution_control_family(catalog, "grid_dependency").count(),
        2
    );
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// Included inside `cuda_device::grid` to keep its public API stable.\n\npub mod dependency {\n",
    );
    for (id, name) in [
        ("grid_dependency_launch_dependents", "trigger_dependents"),
        ("grid_dependency_wait", "wait"),
    ] {
        let record = execution_control_family(catalog, "grid_dependency")
            .find(|record| record.id == id)
            .expect("complete grid-dependency family");
        writeln!(output, "    /// {}", record.summary).unwrap();
        output.push_str(
            "    ///\n    /// # Safety\n    /// The kernel launch must participate in a valid programmatic dependent-launch protocol.\n    #[inline(never)]\n",
        );
        writeln!(output, "    pub unsafe fn {name}() {{").unwrap();
        writeln!(
            output,
            "        unreachable!(\"{name} called outside CUDA kernel context\")"
        )
        .unwrap();
        output.push_str("    }\n\n");
    }
    output.push_str("}\n");
    output
}

pub(super) fn render_compat_register_control(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(
        execution_control_family(catalog, "register_control").count(),
        2
    );
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::thread` to keep its public API stable.\n\n");
    for (id, public_name, hidden_name) in [
        ("setmaxnreg_inc", "setmaxnreg_inc", "__setmaxnreg_inc"),
        ("setmaxnreg_dec", "setmaxnreg_dec", "__setmaxnreg_dec"),
    ] {
        let record = execution_control_family(catalog, "register_control")
            .find(|record| record.id == id)
            .expect("complete register-control family");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// # Safety\n/// Every thread in the warpgroup must execute the same operation with the same count.\n#[inline(always)]\n",
        );
        writeln!(output, "pub unsafe fn {public_name}<const N: u32>() {{").unwrap();
        writeln!(output, "    unsafe {{ {hidden_name}(N) }}").unwrap();
        output.push_str("}\n\n#[doc(hidden)]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub(crate) unsafe fn {hidden_name}(_register_count: u32) {{"
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"{public_name} called outside CUDA kernel context\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_tma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::tma` to keep its public API stable.\n\n");
    for record in tma_intrinsics(catalog) {
        let tma = record.tma.as_ref().expect("TMA contract");
        let operation = tma.operation;
        writeln!(output, "/// {}", record.summary).unwrap();
        let dimensions = tma.dimensions();
        let is_g2s = matches!(
            operation,
            TmaOperation::G2sTile1d
                | TmaOperation::G2sTile2d
                | TmaOperation::G2sTile2dMulticast
                | TmaOperation::G2sTile2dMulticastCg2
                | TmaOperation::G2sTile3d
                | TmaOperation::G2sTile4d
                | TmaOperation::G2sTile5d
        );
        let is_s2g = matches!(
            operation,
            TmaOperation::S2gTile1d
                | TmaOperation::S2gTile2d
                | TmaOperation::S2gTile3d
                | TmaOperation::S2gTile4d
                | TmaOperation::S2gTile5d
        );
        let is_reduction = operation == TmaOperation::Reduce;
        if !record.rust.safe {
            output.push_str("///\n/// # Safety\n");
            if is_g2s {
                output.push_str(
                    "/// `dst`, `tensor_map`, and `barrier` must remain valid until the copy completes.\n",
                );
            } else if is_s2g {
                output.push_str(
                    "/// `src` and `tensor_map` must remain valid until the committed copy group completes.\n",
                );
            } else if is_reduction {
                output.push_str(
                    "/// `src` must name a live shared-memory tile and `tensor_map` must describe a compatible global destination until the committed reduction completes.\n",
                );
            } else if matches!(
                tma.adapter,
                TmaAdapter::DescriptorAndAddressPointers
                    | TmaAdapter::DescriptorOrdinalAndU32
                    | TmaAdapter::DescriptorOrdinalAndU64
                    | TmaAdapter::DescriptorAndImmediateU32
                    | TmaAdapter::DescriptorAndRuntimeU32
            ) {
                output.push_str(
                    "/// `tensor_map` must point to a writable, 128-byte tensor-map descriptor in global memory.\n",
                );
            } else {
                output.push_str("/// `tensor_map` must point to a live tensor-map descriptor.\n");
            }
        }
        output.push_str("#[inline(never)]\n");
        if is_g2s {
            let dimensions = dimensions.unwrap();
            let mut arguments = vec![
                "dst: *mut u8".to_owned(),
                "tensor_map: *const TmaDescriptor".to_owned(),
            ];
            let mut values = vec!["dst".to_owned(), "tensor_map".to_owned()];
            for index in 0..dimensions {
                arguments.push(format!("coord{index}: i32"));
                values.push(format!("coord{index}"));
            }
            arguments.push("barrier: *mut Barrier".into());
            values.push("barrier".into());
            if matches!(
                operation,
                TmaOperation::G2sTile2dMulticast | TmaOperation::G2sTile2dMulticastCg2
            ) {
                arguments.push("cta_mask: u16".into());
                values.push("cta_mask".into());
            }
            if arguments.len() > 7 {
                output.push_str("#[allow(clippy::too_many_arguments)]\n");
            }
            writeln!(
                output,
                "pub unsafe fn {}({}) {{",
                record.rust.name,
                arguments.join(", ")
            )
            .unwrap();
            writeln!(output, "    let _ = ({});", values.join(", ")).unwrap();
        } else if is_s2g || is_reduction {
            let dimensions = dimensions.unwrap();
            let mut arguments = vec![
                "src: *const u8".to_owned(),
                "tensor_map: *const TmaDescriptor".to_owned(),
            ];
            let mut values = vec!["src".to_owned(), "tensor_map".to_owned()];
            for index in 0..dimensions {
                arguments.push(format!("coord{index}: i32"));
                values.push(format!("coord{index}"));
            }
            writeln!(
                output,
                "pub unsafe fn {}({}) {{",
                record.rust.name,
                arguments.join(", ")
            )
            .unwrap();
            writeln!(output, "    let _ = ({});", values.join(", ")).unwrap();
        } else if operation == TmaOperation::CommitGroup {
            writeln!(output, "pub fn {}() {{", record.rust.name).unwrap();
        } else if matches!(
            operation,
            TmaOperation::WaitGroup | TmaOperation::WaitGroupRead
        ) {
            writeln!(output, "pub fn {}(n: u32) {{", record.rust.name).unwrap();
            output.push_str("    let _ = n;\n");
        } else if operation == TmaOperation::PrefetchTensorMap {
            writeln!(
                output,
                "pub unsafe fn {}(tensor_map: *const TmaDescriptor) {{",
                record.rust.name
            )
            .unwrap();
            output.push_str("    let _ = tensor_map;\n");
        } else if let Some(coordinate_count) = operation.prefetch_coordinate_count() {
            let mut arguments = vec!["tensor_map: *const TmaDescriptor".to_owned()];
            let mut values = vec!["tensor_map".to_owned()];
            for index in 0..coordinate_count {
                arguments.push(format!("coord{index}: i32"));
                values.push(format!("coord{index}"));
            }
            if operation.uses_prefetch_cache_hint() {
                arguments.push("cache_hint: u64".into());
                values.push("cache_hint".into());
            }
            writeln!(
                output,
                "pub unsafe fn {}({}) {{",
                record.rust.name,
                arguments.join(", ")
            )
            .unwrap();
            writeln!(output, "    let _ = ({});", values.join(", ")).unwrap();
        } else {
            match tma.adapter {
                TmaAdapter::DescriptorAndAddressPointers => {
                    writeln!(
                        output,
                        "pub unsafe fn {}(tensor_map: *mut TmaDescriptor, new_address: *const u8) {{",
                        record.rust.name
                    )
                    .unwrap();
                    output.push_str("    let _ = (tensor_map, new_address);\n");
                }
                TmaAdapter::DescriptorOrdinalAndU32 => {
                    writeln!(
                        output,
                        "pub unsafe fn {}(tensor_map: *mut TmaDescriptor, ordinal: u32, new_value: u32) {{",
                        record.rust.name
                    )
                    .unwrap();
                    output.push_str("    let _ = (tensor_map, ordinal, new_value);\n");
                }
                TmaAdapter::DescriptorOrdinalAndU64 => {
                    writeln!(
                        output,
                        "pub unsafe fn {}(tensor_map: *mut TmaDescriptor, ordinal: u32, new_value: u64) {{",
                        record.rust.name
                    )
                    .unwrap();
                    output.push_str("    let _ = (tensor_map, ordinal, new_value);\n");
                }
                TmaAdapter::DescriptorAndImmediateU32 | TmaAdapter::DescriptorAndRuntimeU32 => {
                    writeln!(
                        output,
                        "pub unsafe fn {}(tensor_map: *mut TmaDescriptor, new_value: u32) {{",
                        record.rust.name
                    )
                    .unwrap();
                    output.push_str("    let _ = (tensor_map, new_value);\n");
                }
                TmaAdapter::DescriptorPointerInjectBytes => {
                    writeln!(
                        output,
                        "pub unsafe fn {}(tensor_map: *const TmaDescriptor) {{",
                        record.rust.name
                    )
                    .unwrap();
                    output.push_str("    let _ = tensor_map;\n");
                }
                TmaAdapter::NoOperands => {
                    writeln!(output, "pub fn {}() {{", record.rust.name).unwrap();
                }
                _ => unreachable!("TMA compatibility operation category was matched"),
            }
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output.push_str(
        r#"/// Replace a global tensor-map address and publish it to the tensor-map proxy.
///
/// # Safety
/// `tensor_map` must point to a writable, 128-byte tensor-map descriptor in
/// global memory, and `new_address` must remain valid for every later TMA use.
#[inline(always)]
pub unsafe fn replace_tma_global_address(
    tensor_map: *mut TmaDescriptor,
    new_address: *const u8,
) {
    unsafe { tensormap_replace_global_address(tensor_map, new_address) };
    fence_proxy_tensormap_generic_release_gpu();
    unsafe { fence_proxy_tensormap_generic_acquire_gpu(tensor_map) };
}
"#,
    );
    output
}

pub(super) fn render_compat_wgmma_control(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(wgmma_controls(catalog).count(), 3);
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::wgmma` to keep its public API stable.\n\n");

    for mode in [WgmmaControlMode::Fence, WgmmaControlMode::CommitGroup] {
        let record = wgmma_control(catalog, mode);
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(output, "pub fn {}() {{", record.rust.name).unwrap();
        writeln!(
            output,
            "    unreachable!({:?})",
            format!("{} called outside CUDA kernel context", record.rust.name)
        )
        .unwrap();
        output.push_str("}\n\n");
    }

    let wait = wgmma_control(catalog, WgmmaControlMode::WaitGroup);
    writeln!(output, "/// {}", wait.summary).unwrap();
    output.push_str(
        r#"#[inline(always)]
pub fn wgmma_wait_group<const N: u32>() {
    __wgmma_wait_group(N as u64);
}

#[doc(hidden)]
#[inline(never)]
pub(crate) fn __wgmma_wait_group(_max_pending: u64) {
    unreachable!("wgmma_wait_group called outside CUDA kernel context")
}
"#,
    );
    output
}

fn render_compat_tcgen05_mma_record(output: &mut String, record: &CatalogIntrinsic) {
    let mma = record
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref())
        .expect("tcgen05 MMA record");
    let parameters = tcgen05_mma_runtime_parameters(mma);
    let signature = parameters
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let values = parameters
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ");

    writeln!(output, "/// {}", record.summary).unwrap();
    output.push_str("/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n");
    if mma.alias.is_some() && mma.form == Tcgen05MmaForm::WsTensor {
        output.push_str("/// This uses kind f8f6f4 and collector b0::discard.\n");
        output.push_str("/// `legacy_a_desc` is kept for compatibility; tensor A uses `a_tmem`.\n");
    } else if mma.alias.is_some() {
        output.push_str("/// This uses kind f8f6f4, CTA group 1, and collector a::discard.\n");
    } else {
        match mma.selector_layout {
            Tcgen05MmaSelectorLayout::Base { .. } => {
                output.push_str("/// `KIND` is 0=f16, 1=tf32, 2=f8f6f4, or 3=i8.\n");
                output.push_str(
                    "/// `CTA_GROUP` is 1 or 2. `COLLECTOR_A` is 0=discard, 1=lastuse, 2=fill, or 3=use.\n",
                );
            }
            Tcgen05MmaSelectorLayout::WarpSpecialized { .. } => {
                output.push_str("/// `KIND` is 0=f16, 1=tf32, 2=f8f6f4, or 3=i8.\n");
                output.push_str(
                    "/// `B_BUFFER` is 0 through 3. `B_USAGE` is 0=discard, 1=lastuse, 2=fill, or 3=use.\n",
                );
            }
        }
    }
    output.push_str("///\n/// # Safety\n");
    output
        .push_str("/// Tensor-memory addresses and descriptors must be valid for this MMA form.\n");
    if mma.alias.is_some() {
        output.push_str("#[inline(never)]\n");
        output.push_str("#[allow(clippy::too_many_arguments)]\n");
        writeln!(output, "pub unsafe fn {}({signature}) {{", record.rust.name).unwrap();
        writeln!(output, "    let _ = ({values});").unwrap();
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
        return;
    }

    let selectors = tcgen05_mma_selector_parameters(mma.selector_layout);
    let const_parameters = selectors
        .iter()
        .map(|(name, _)| format!("const {name}: u32"))
        .collect::<Vec<_>>()
        .join(", ");
    let selector_values = selectors
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ");
    output.push_str("#[inline(always)]\n");
    if parameters.len() > 5 {
        output.push_str("#[allow(clippy::too_many_arguments)]\n");
    }
    writeln!(
        output,
        "pub unsafe fn {}<{const_parameters}>({signature}) {{",
        record.rust.name
    )
    .unwrap();
    writeln!(
        output,
        "    unsafe {{ __{}({values}, {selector_values}) }}",
        record.rust.name
    )
    .unwrap();
    output.push_str("}\n\n#[doc(hidden)]\n#[inline(never)]\n");
    output.push_str("#[allow(clippy::too_many_arguments)]\n");
    let hidden_parameters = parameters
        .iter()
        .map(|(name, ty)| format!("_{name}: {ty}"))
        .chain(selectors.iter().map(|(_, name)| format!("_{name}: u32")))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "pub(crate) unsafe fn __{}({hidden_parameters}) {{",
        record.rust.name
    )
    .unwrap();
    writeln!(
        output,
        "    unreachable!(\"{} called outside CUDA kernel context\")",
        record.rust.name
    )
    .unwrap();
    output.push_str("}\n\n");
}

pub(super) fn render_compat_tcgen05(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(tcgen05_intrinsics(catalog).count(), 233);
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::tcgen05` to keep its public API stable.\n\n");
    for record in tcgen05_intrinsics(catalog) {
        let tcgen05 = record.tcgen05.as_ref().unwrap();
        if tcgen05.mma.is_some() {
            render_compat_tcgen05_mma_record(&mut output, record);
            continue;
        }
        let operation = tcgen05.operation;
        let has_half_split_offset = tcgen05
            .ld
            .is_some_and(|ld| ld.shape == Tcgen05LdShape::M16x32bx2)
            || tcgen05
                .st
                .is_some_and(|st| st.shape == Tcgen05LdShape::M16x32bx2);
        if has_half_split_offset {
            let (result, data) = if operation == Tcgen05Operation::Ld {
                let count = tcgen05_ld_register_count(record);
                (
                    if count == 1 {
                        "u32".into()
                    } else {
                        format!("CuSimd<u32, {count}>")
                    },
                    None,
                )
            } else {
                let count = tcgen05_st_register_count(record);
                (
                    "()".into(),
                    Some(if count == 1 {
                        "u32".into()
                    } else {
                        format!("CuSimd<u32, {count}>")
                    }),
                )
            };
            writeln!(output, "/// {}", record.summary).unwrap();
            output.push_str(
                "/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n",
            );
            output.push_str("///\n/// # Safety\n");
            output.push_str(
                "/// The tensor-memory address must be live, and the warp must execute this call uniformly.\n",
            );
            output.push_str("#[inline(always)]\n");
            if let Some(data) = &data {
                writeln!(
                    output,
                    "pub unsafe fn {}<const HALF_SPLIT_OFFSET: i32>(tmem_addr: u32, data: {data}) {{",
                    record.rust.name
                )
                .unwrap();
                writeln!(
                    output,
                    "    unsafe {{ __{}(tmem_addr, HALF_SPLIT_OFFSET as i64, data) }}",
                    record.rust.name
                )
                .unwrap();
            } else {
                writeln!(
                    output,
                    "pub unsafe fn {}<const HALF_SPLIT_OFFSET: i32>(tmem_addr: u32) -> {result} {{",
                    record.rust.name
                )
                .unwrap();
                writeln!(
                    output,
                    "    unsafe {{ __{}(tmem_addr, HALF_SPLIT_OFFSET as i64) }}",
                    record.rust.name
                )
                .unwrap();
            }
            output.push_str("}\n\n#[doc(hidden)]\n#[inline(never)]\n");
            if let Some(data) = data {
                writeln!(
                    output,
                    "pub(crate) unsafe fn __{}(_tmem_addr: u32, _half_split_offset: i64, _data: {data}) {{",
                    record.rust.name
                )
                .unwrap();
            } else {
                writeln!(
                    output,
                    "pub(crate) unsafe fn __{}(_tmem_addr: u32, _half_split_offset: i64) -> {result} {{",
                    record.rust.name
                )
                .unwrap();
            }
            writeln!(
                output,
                "    unreachable!(\"{} called outside CUDA kernel context\")",
                record.rust.name
            )
            .unwrap();
            output.push_str("}\n\n");
            continue;
        }
        let (arguments, values): (String, String) = if operation == Tcgen05Operation::St {
            let count = tcgen05_st_register_count(record);
            let data = if count == 1 {
                "u32".into()
            } else {
                format!("CuSimd<u32, {count}>")
            };
            (
                format!("tmem_addr: u32, data: {data}"),
                "tmem_addr, data".into(),
            )
        } else {
            let (arguments, values) = match operation {
                Tcgen05Operation::Alloc | Tcgen05Operation::AllocCg2 => {
                    ("dst_smem: *mut u32, n_cols: u32", "dst_smem, n_cols")
                }
                Tcgen05Operation::Dealloc | Tcgen05Operation::DeallocCg2 => {
                    ("tmem_addr: u32, n_cols: u32", "tmem_addr, n_cols")
                }
                Tcgen05Operation::RelinquishAllocPermit
                | Tcgen05Operation::FenceBeforeThreadSync
                | Tcgen05Operation::FenceAfterThreadSync
                | Tcgen05Operation::LoadWait
                | Tcgen05Operation::StoreWait
                | Tcgen05Operation::RelinquishAllocPermitCg2 => ("", ""),
                Tcgen05Operation::Commit
                | Tcgen05Operation::CommitSharedCluster
                | Tcgen05Operation::CommitCg2
                | Tcgen05Operation::CommitSharedClusterCg2 => ("mbar: *mut u64", "mbar"),
                Tcgen05Operation::MmaWsF16
                | Tcgen05Operation::MmaWsBf16
                | Tcgen05Operation::MmaWsTf32 => (
                    "d_tmem: u32, a_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool",
                    "d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d",
                ),
                Tcgen05Operation::MmaF16 | Tcgen05Operation::MmaF16Cg2 => (
                    "d_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool",
                    "d_tmem, a_desc, b_desc, idesc, enable_d",
                ),
                Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2 => {
                    ("tmem_addr: u32, smem_desc: u64", "tmem_addr, smem_desc")
                }
                Tcgen05Operation::Ld16x256bX8Pure
                | Tcgen05Operation::Ld16x256bPure
                | Tcgen05Operation::Ld => ("tmem_addr: u32", "tmem_addr"),
                Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2 => {
                    ("mbar: *mut u64, cta_mask: u16", "mbar, cta_mask")
                }
                Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2 => {
                    ("tmem_addr: u32", "tmem_addr")
                }
                Tcgen05Operation::St => unreachable!("store handled above"),
                Tcgen05Operation::Mma => unreachable!("generic MMA handled above"),
            };
            (arguments.into(), values.into())
        };
        let result: String = match operation {
            Tcgen05Operation::Ld16x256bX8Pure => "TmemF32x32".into(),
            Tcgen05Operation::Ld16x256bPure => "TmemF32x4".into(),
            Tcgen05Operation::Ld => {
                let count = tcgen05_ld_register_count(record);
                if count == 1 {
                    "u32".into()
                } else {
                    format!("CuSimd<u32, {count}>")
                }
            }
            Tcgen05Operation::St => "()".into(),
            _ => "()".into(),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        if let Some(participation) = tcgen05_participation_doc(operation) {
            writeln!(output, "/// {participation}").unwrap();
        }
        output.push_str(
            "/// All tcgen05 operations in the kernel must use the same CTA-group mode.\n",
        );
        if !record.rust.safe {
            output.push_str("///\n/// # Safety\n");
            if operation == Tcgen05Operation::St {
                output.push_str(
                    "/// The tensor-memory address must remain live and cover the selected tile.\n\
                     /// All active warp lanes must execute convergently with the same address. Complete the store wait before relying on completion.\n",
                );
            } else if operation == Tcgen05Operation::Ld {
                output.push_str(
                    "/// The tensor-memory address must remain live and cover the selected tile.\n\
                     /// All active warp lanes must execute convergently with the same address. Complete the load wait before using the result.\n",
                );
            } else if tcgen05_is_commit(operation) {
                if tcgen05_is_multicast_commit(operation) {
                    output.push_str(
                        "/// `mbar` must point to a live initialized cluster-shared mbarrier. `cta_mask` must select valid CTA ranks in its cluster.\n",
                    );
                } else {
                    output.push_str(
                        "/// `mbar` must point to a live initialized mbarrier valid for this CTA-group mode.\n",
                    );
                }
                output.push_str(
                    "/// The same thread that issued the tracked asynchronous tcgen05 operations must issue this commit.\n",
                );
            } else if tcgen05_is_shift(operation) {
                output.push_str(
                    "/// `tmem_addr` must name a live tensor-memory allocation, and its lane component must be a multiple of 32.\n\
                     /// Completion must be tracked by a matching commit from that same thread and observed through the selected mbarrier before relying on shifted data.\n",
                );
            } else {
                output.push_str("/// The caller must satisfy the tcgen05 address, lifetime, and participation rules.\n");
            }
        }
        output.push_str("#[inline(never)]\n");
        if record.rust.arguments.len() > 5 {
            output.push_str("#[allow(clippy::too_many_arguments)]\n");
        }
        let safety = if record.rust.safe { "" } else { "unsafe " };
        writeln!(
            output,
            "pub {safety}fn {}({arguments}) -> {result} {{",
            record.rust.name
        )
        .unwrap();
        if !values.is_empty() {
            if values.contains(',') {
                writeln!(output, "    let _ = ({values});").unwrap();
            } else {
                writeln!(output, "    let _ = {values};").unwrap();
            }
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_packed_atomic(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::atomic` to keep existing paths stable.\n\n");
    for record in packed_atomics(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::atomic::"))
            .expect("packed-atomic compatibility path");
        assert_eq!(path, &format!("cuda_device::atomic::{}", record.rust.name));
        let packed = record
            .packed_atomic
            .as_ref()
            .expect("packed-atomic semantics");
        let lane_type = match packed.format {
            PackedAtomicFormat::F16x2 => "f16",
            PackedAtomicFormat::Bf16x2 => "bf16",
        };
        let minimum_sm = match &record.target.hardware {
            CatalogHardwareTarget::AnyOf { alternatives } => match alternatives.as_slice() {
                [CatalogHardwareAlternative::MinimumSm { sm }] => *sm,
                _ => panic!("packed-atomic compatibility API requires one minimum SM"),
            },
            _ => panic!("packed-atomic compatibility API requires one minimum SM"),
        };
        assert!(!record.rust.safe);
        assert!(record.rust.must_use);
        assert_eq!(record.rust.arguments, ["*mut u32", "u32"]);
        assert_eq!(record.rust.result, "u32");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// `val` and the result pack two {lane_type} lanes into `u32`, low lane first."
        )
        .unwrap();
        output.push_str(
            "/// The lanes are atomic independently and may not form one old 32-bit snapshot.\n",
        );
        output.push_str("/// This is a relaxed GPU-scope operation. Each lane rounds to nearest-even and preserves subnormals.\n");
        writeln!(
            output,
            "/// Requires PTX {} and `sm_{minimum_sm}+`.",
            record.target.minimum_ptx
        )
        .unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// `addr` must point to four writable, four-byte-aligned bytes in global memory.\n",
        );
        output.push_str("/// Do not overlap this operation with a whole-word atomic or non-atomic lane access.\n");
        output.push_str("/// Racing atomics must use mutually inclusive scopes; host/system access is not included.\n");
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub unsafe fn {}(addr: *mut u32, val: u32) -> u32 {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = (addr, val);\n");
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_cp_async_copy(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// Included inside `cuda_device::async_copy` to keep existing paths stable.\n\n",
    );
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().expect("cp.async semantics");
        let bytes = copy.copy_size.bytes();
        let cache = match copy.cache_policy {
            CpAsyncCachePolicy::Ca => "cache-all",
            CpAsyncCachePolicy::Cg => "cache-global",
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Uses the {cache} policy and copies {bytes} bytes."
        )
        .unwrap();
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "/// Bytes after `src_size` are filled with zero; `src_size` must be at most {bytes}."
            )
            .unwrap();
        }
        output.push_str("///\n/// # Safety\n");
        writeln!(
            output,
            "/// `shared_dst` must point to {bytes} writable bytes in shared memory and be aligned to {bytes} bytes."
        )
        .unwrap();
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "/// `global_src` must point to `src_size` readable bytes in global memory and be aligned to {bytes} bytes."
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "/// `global_src` must point to {bytes} readable bytes in global memory and be aligned to {bytes} bytes."
            )
            .unwrap();
        }
        output.push_str(
            "/// Both ranges must remain valid, the source must remain unchanged, and the destination must not be accessed until this copy completes.\n\
             /// The issuing thread must complete the copy. Synchronize threads afterward before another thread accesses the destination.\n\
             /// User-authored completion assembly must include a compiler memory clobber.\n",
        );
        output.push_str("#[inline(never)]\n");
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "pub unsafe fn {}(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32) {{",
                record.rust.name
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "pub unsafe fn {}(_shared_dst: *mut u32, _global_src: *const u32) {{",
                record.rust.name
            )
            .unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    for record in cp_async_controls(catalog) {
        let control = record
            .cp_async_control
            .as_ref()
            .expect("cp.async control semantics");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// # Safety\n");
        match control.operation {
            CpAsyncControlOperation::CommitGroup => output.push_str(
                "/// This commits only copies issued by the executing thread and does not wait for completion.\n",
            ),
            CpAsyncControlOperation::WaitAll => output.push_str(
                "/// This waits only for copies issued by this thread. Synchronize threads before another thread accesses a completed destination.\n",
            ),
            CpAsyncControlOperation::WaitGroup => output.push_str(
                "/// `max_pending` must be a compile-time constant. Access only destinations whose copy groups this wait completes.\n",
            ),
        }
        output.push_str("#[inline(never)]\n");
        if control.operation == CpAsyncControlOperation::WaitGroup {
            writeln!(
                output,
                "pub unsafe fn {}(_max_pending: u32) {{",
                record.rust.name
            )
            .unwrap();
        } else {
            writeln!(output, "pub unsafe fn {}() {{", record.rust.name).unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    for record in cp_async_mbarriers(catalog) {
        let bridge = record
            .cp_async_mbarrier
            .as_ref()
            .expect("cp.async mbarrier semantics");
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::async_copy::"))
            .expect("cp.async mbarrier compatibility path");
        assert_eq!(
            path,
            &format!("cuda_device::async_copy::{}", record.rust.name)
        );
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// `barrier` must point to a live, initialized, eight-byte-aligned mbarrier object in shared memory.\n\
             /// This thread must have issued the `cp.async` operations being associated with it.\n\
             /// The object must remain valid until those operations complete.\n",
        );
        match bridge.operation {
            CpAsyncMbarrierOperation::Arrive => output.push_str(
                "/// This form increments the pending count before scheduling the asynchronous arrival.\n\
                 /// That increment must not exceed the barrier's pending-count limit.\n",
            ),
            CpAsyncMbarrierOperation::ArriveNoInc => output.push_str(
                "/// The initial pending count must already include the asynchronous arrival.\n",
            ),
        }
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub unsafe fn {}(_barrier: *mut crate::barrier::Barrier) {{",
            record.rust.name
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_mbarrier_basic(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::barrier` to keep existing paths stable.\n\n");
    for record in mbarrier_basics(catalog) {
        let mbarrier = record
            .mbarrier_basic
            .as_ref()
            .expect("basic mbarrier semantics");
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::barrier::"))
            .expect("basic mbarrier compatibility path");
        assert_eq!(path, &format!("cuda_device::barrier::{}", record.rust.name));
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// `bar` must point to a live, eight-byte-aligned `Barrier` in shared memory.\n",
        );
        match mbarrier.operation {
            MbarrierBasicOperation::Init => output.push_str(
                "/// Exactly one thread may initialize it. `expected_count` must be in `1..=0xFFFFF` and include every arrival in the phase.\n\
                 /// The barrier must be uninitialized or invalidated, with no concurrent barrier operation.\n",
            ),
            MbarrierBasicOperation::Arrive => output.push_str(
                "/// The barrier must be initialized, and this arrival must be included in the current phase's expected count.\n\
                 /// Use the returned token only with this barrier and phase.\n",
            ),
            MbarrierBasicOperation::ArriveNoComplete => output.push_str(
                "/// The barrier must be initialized. `count` must be a valid PTX arrival count and this operation must not complete the current phase.\n\
                 /// Use the returned opaque state only with this barrier and phase.\n",
            ),
            MbarrierBasicOperation::TestWait => output.push_str(
                "/// The barrier must be initialized. `token` must come from this barrier and phase.\n",
            ),
            MbarrierBasicOperation::Inval => output.push_str(
                "/// The barrier must be initialized and unused by all threads and asynchronous operations.\n\
                 /// Exactly one thread may invalidate it.\n",
            ),
        }
        output.push_str("#[inline(never)]\n");
        match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *mut Barrier, expected_count: u32) {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, expected_count);\n");
            }
            MbarrierBasicOperation::Arrive => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier) -> u64 {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = bar;\n");
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier, count: u32) -> u64 {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, count);\n");
            }
            MbarrierBasicOperation::TestWait => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier, token: u64) -> bool {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, token);\n");
            }
            MbarrierBasicOperation::Inval => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *mut Barrier) {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = bar;\n");
            }
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_mbarrier_extended(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::barrier` to keep existing paths stable.\n\n");
    for record in mbarrier_extended(catalog) {
        let contract = record
            .mbarrier_extended
            .as_ref()
            .expect("extended-mbarrier contract");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// The caller must satisfy the barrier, scope, and memory-ordering contract.\n",
        );
        output.push_str("#[inline(never)]\n");
        match contract.operation {
            MbarrierExtendedOperation::ArriveExpectTxCta
            | MbarrierExtendedOperation::ArriveExpectTxCluster => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier, _tx_count: u32, bytes: u32) -> u64 {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, bytes);\n");
            }
            MbarrierExtendedOperation::ArriveRemoteCluster => {
                writeln!(
                    output,
                    "pub unsafe fn {}(remote_bar_addr: u64) {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = remote_bar_addr;\n");
            }
            MbarrierExtendedOperation::TryWaitTokenCta => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier, token: u64) -> bool {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, token);\n");
            }
            MbarrierExtendedOperation::TryWaitParityCta
            | MbarrierExtendedOperation::TryWaitParityCluster => {
                writeln!(
                    output,
                    "pub unsafe fn {}(bar: *const Barrier, parity: u32) -> bool {{",
                    record.rust.name
                )
                .unwrap();
                output.push_str("    let _ = (bar, parity);\n");
            }
            MbarrierExtendedOperation::FenceProxyAsyncSharedCta
            | MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster
            | MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster
            | MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster => {
                writeln!(output, "pub unsafe fn {}() {{", record.rust.name).unwrap();
            }
            MbarrierExtendedOperation::Nanosleep => {
                writeln!(output, "pub unsafe fn {}(ns: u32) {{", record.rust.name).unwrap();
                output.push_str("    let _ = ns;\n");
            }
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_integer_minmax(
    catalog: &CatalogFile,
    hash: &str,
    module: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "// Included inside `cuda_device::{module}` to keep existing paths stable.\n"
    )
    .unwrap();
    for record in integer_minmaxes(catalog).filter(|record| record.rust.module == module) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(&format!("cuda_device::{module}::")))
            .expect("integer-min/max compatibility path");
        let scalar = &record.rust.result;
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}(arg0: {scalar}, arg1: {scalar}) -> {scalar} {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = (arg0, arg1);\n");
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_compat_packed_alu(
    catalog: &CatalogFile,
    hash: &str,
    format: PackedAluFormat,
) -> String {
    let mut output = rust_header(catalog, hash);
    let (module, _, _, _, _) = packed_alu_format_shape(format);
    writeln!(
        output,
        "// Included inside `cuda_device::{module}` to keep existing paths stable.\n"
    )
    .unwrap();
    for record in packed_alus(catalog).filter(|record| {
        record
            .packed_alu
            .as_ref()
            .is_some_and(|packed| packed.format == format)
    }) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(&format!("cuda_device::{module}::")))
            .expect("packed-ALU compatibility path");
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (0..record.rust.arguments.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        if record.rust.must_use {
            output.push_str("#[must_use]\n");
        }
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        if record.rust.arguments.len() == 1 {
            writeln!(output, "    let _ = {values};").unwrap();
        } else {
            writeln!(output, "    let _ = ({values});").unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    render_compat_extended_minmax_into(&mut output, catalog, module);
    output
}

pub(super) fn render_compat_scalar_minmax(
    catalog: &CatalogFile,
    hash: &str,
    module: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "// Included inside `cuda_device::{module}` to keep existing paths stable.\n"
    )
    .unwrap();
    render_compat_extended_minmax_into(&mut output, catalog, module);
    output
}

fn render_compat_extended_minmax_into(output: &mut String, catalog: &CatalogFile, module: &str) {
    for record in extended_minmax(catalog).filter(|record| record.rust.module == module) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(&format!("cuda_device::{module}::")))
            .expect("extended-minmax compatibility path");
        let ty = extended_minmax_rust_type(record);
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}(a: {ty}, b: {ty}) -> {ty} {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = (a, b);\n");
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
}

pub(super) fn render_compat_packed_conversion(
    catalog: &CatalogFile,
    hash: &str,
    path_prefix: &str,
    containing_module: &str,
    argument_names: (&str, &str),
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "// Included inside `cuda_device::{containing_module}`.\n"
    )
    .unwrap();
    for record in packed_conversions(catalog).filter(|record| {
        record
            .rust
            .compatibility_paths
            .iter()
            .any(|path| path.starts_with(path_prefix))
    }) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(path_prefix))
            .expect("packed-conversion compatibility path");
        // A packed source arrives in one register, so the parameter list
        // follows the record's own argument types rather than a fixed f32 pair.
        let parameter_names: Vec<&str> = match packed_conversion_source(record) {
            PackedConversionSourceFormat::F32x2 => vec![argument_names.0, argument_names.1],
            PackedConversionSourceFormat::E4m3x2
            | PackedConversionSourceFormat::E5m2x2
            | PackedConversionSourceFormat::F16x2 => vec!["packed"],
        };
        let parameter_types = packed_conversion_rust_arguments(record);
        let parameters = parameter_names
            .iter()
            .zip(&parameter_types)
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let discards = parameter_names.join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({parameters}) -> {} {{",
            path.rsplit("::")
                .next()
                .expect("packed-conversion compatibility function name"),
            record.rust.result,
        )
        .unwrap();
        if parameter_names.len() == 1 {
            writeln!(output, "    let _ = {discards};").unwrap();
        } else {
            writeln!(output, "    let _ = ({discards});").unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    if containing_module == "convert" {
        for record in scalar_conversions(catalog) {
            let path = record
                .rust
                .compatibility_paths
                .iter()
                .find(|path| path.starts_with("cuda_device::convert::"))
                .expect("scalar-conversion compatibility path");
            writeln!(output, "/// {}", record.summary).unwrap();
            output.push_str("#[must_use]\n#[inline(never)]\n");
            writeln!(output, "pub fn {}(value: f32) -> u32 {{", record.rust.name).unwrap();
            output.push_str("    let _ = value;\n");
            writeln!(
                output,
                "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
            )
            .unwrap();
            output.push_str("}\n\n");
        }
    }
    output
}

pub(super) fn render_compat_float(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::float`.\n\n");
    for record in scalar_arithmetics(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::float::"))
            .expect("scalar-arithmetic compatibility path");
        let ty = scalar_arithmetic_rust_type(record);
        let arity = scalar_arithmetic_arity(record);
        let parameters = (0..arity)
            .map(|index| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..arity)
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({parameters}) -> {ty} {{",
            record.rust.name
        )
        .unwrap();
        writeln!(output, "    let _ = ({arguments});").unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    for record in scalar_maths(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::float::"))
            .expect("scalar-math compatibility path");
        let ty = match scalar_math_contract(record).format {
            ScalarMathFormat::F16 => "u16",
            ScalarMathFormat::F32 => "f32",
            ScalarMathFormat::F64 => "f64",
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(output, "pub fn {}(arg0: {ty}) -> {ty} {{", record.rust.name).unwrap();
        writeln!(output, "    let _ = arg0;").unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    render_compat_extended_minmax_into(&mut output, catalog, "float");
    output
}

pub(super) fn render_compat_float_output(
    catalog: &CatalogFile,
    hash: &str,
) -> Option<(PathBuf, String)> {
    (scalar_arithmetics(catalog).next().is_some()
        || extended_minmax(catalog).next().is_some()
        || scalar_maths(catalog).next().is_some())
    .then(|| {
        (
            "crates/cuda-device/src/generated/float.rs".into(),
            render_compat_float(catalog, hash),
        )
    })
}
