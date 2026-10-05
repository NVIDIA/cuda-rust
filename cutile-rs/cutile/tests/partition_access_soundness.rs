/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adversarial soundness probes for the plain (`Partition`) access checker,
//! from the 2026-08-04 external review
//! (`.internal/tasks/in_progress/check_hoisting/REVIEW_FINDINGS_2026-08-04.md`):
//! the dynamic path must bound indices from BELOW as well as above, and a
//! foreign mapped index against a permuted target must discharge against the
//! REMAPPED root axis, not the logical coordinate.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod access_soundness_module {
    use cutile::core::*;

    /// A raw runtime scalar indexing a dynamic-extent partition: nothing
    /// bounds `idx` at compile time in either direction.
    #[cutile::entry]
    fn runtime_scalar_index<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        idx: i32,
    ) {
        let p = x.partition(shape![B]);
        let t = p.load([idx]);
        z.store(t);
    }

    /// A loop whose lower endpoint is provably negative: iteration `-2`
    /// produces a negative block index.
    #[cutile::entry]
    fn negative_loop_start<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        for j in -2i32..4i32 {
            let t = p.load([j]);
            z.store(t);
        }
    }

    /// A mapped component (root axis 0 of `z`) indexing logical axis 0 of a
    /// PERMUTED partition of `x` (`dim_map = [1, 0]`), so the real bound is
    /// `x`'s root axis 1. Only the WRONG equality — `dim(z,0) == dim(x,0)` —
    /// is declared: the check must NOT discharge.
    #[cutile::entry(
        preconditions = (
            dim(z, 0) == dim(x, 0),
        )
    )]
    fn permuted_wrong_axis<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let xp = x.partition_permuted(shape![BM, BN], const_array![1, 0]);
        for index in z.iter_indices() {
            let (bid_m, _bid_n) = index.components();
            let t = xp.load([bid_m, 0i32]);
            z.store(t, index);
        }
    }

    /// Positive control: the CORRECT equality — `dim(z,0) == dim(x,1)`, the
    /// remapped axis — is declared, so the axis-0 check discharges.
    #[cutile::entry(
        preconditions = (
            dim(z, 0) == dim(x, 1),
        )
    )]
    fn permuted_right_axis<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let xp = x.partition_permuted(shape![BM, BN], const_array![1, 0]);
        for index in z.iter_indices() {
            let (bid_m, _bid_n) = index.components();
            let t = xp.load([bid_m, 0i32]);
            z.store(t, index);
        }
    }
    /// An index that carries axis provenance, conditionally reassigned to a
    /// constant that is out of range. The join must not keep the old value's
    /// proof (issue #212): before the fix this compiled fully discharged and
    /// executed the out-of-range read.
    #[cutile::entry]
    fn reassigned_index_keeps_no_proof<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        flag: i32,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..num_tiles(&p, 0) {
            let mut j = i;
            if flag > 0i32 {
                j = 100i32;
            }
            acc = acc + p.load([j]);
        }
        z.store(acc);
    }
    /// A partition conditionally reassigned to a DIFFERENT tensor, accessed
    /// by an index walked from a third tensor tied to the first by a
    /// precondition. If the join keeps the pre-branch tensor origin, the
    /// cross-tensor rung bounds a `b`-run access against `a`'s extent
    /// (issue #212, structural residual).
    #[cutile::entry(
        preconditions = (dim(w, 0) == dim(a, 0),)
    )]
    fn reassigned_partition_keeps_no_origin<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        w: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        flag: i32,
    ) {
        let pw = w.partition(shape![B]);
        let mut p = a.partition(shape![B]);
        if flag > 0i32 {
            p = b.partition(shape![B]);
        }
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..num_tiles(&pw, 0) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// Mathematical range analysis says `index` is in `[0, 2]`, but the
    /// machine multiplies in wrapping `i32`: at `i = 2` the product wraps
    /// to 294,967,296 and `max` keeps it. No interval may survive the
    /// wrapping op, so this access must pay a real check
    /// (2026-08-12 review, S1).
    #[cutile::entry]
    fn wrapped_product_masked_by_max<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = max(i * -2_000_000_000i32, i);
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }

    /// The mathematical remainder is nonnegative, but the wrapped dividend
    /// makes the machine remainder negative (`i = 2` gives `-296`): the
    /// lower guard must stay (2026-08-12 review, S1).
    #[cutile::entry]
    fn wrapped_dividend_negative_remainder<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = (i * 2_000_000_000i32) % 1_000i32;
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }

    /// Static-extent twin of the wrapped product: the static fold must not
    /// discharge from a wrap-tainted interval either
    /// (2026-08-12 review, S1).
    #[cutile::entry]
    fn wrapped_product_static_extent<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [48] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = max(i * -2_000_000_000i32, i);
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }

    /// An exact condition is inlined, not joined. Reassigning the mapped
    /// index to the same proven value on the sole feasible path must preserve
    /// its store provenance (2026-08-12 review, P1).
    #[allow(unused_assignments)] // the deliberate overwrite is the regression shape
    #[cutile::entry]
    fn constant_if_preserves_mapped_index<const B: i32, const MAP_SHAPE: [i32; 1]>(
        mut z: MappedPartitionMut<f32, { [B] }, MAP_SHAPE>,
    ) {
        for index in z.iter_indices() {
            let mut same = index;
            if true {
                same = index;
            }
            if false {
                same = index;
            } else {
                same = index;
            }
            let tile: Tile<f32, { [B] }> = constant(0.0, shape![B]);
            z.store(tile, same);
        }
    }

    /// Publishing the sole taken path must preserve its RHS facts, not the
    /// pre-branch template's provenance. This exact branch replaces a proven
    /// iterand with an unproven out-of-range constant, so a check must remain.
    #[allow(unused_assignments)] // the deliberate overwrite is the regression shape
    #[cutile::entry]
    fn constant_if_reassignment_uses_rhs_facts<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..num_tiles(&p, 0) {
            let mut index = i;
            if true {
                index = 100i32;
            }
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }
}

use access_soundness_module::__module_ast_self;

fn compile(name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> Result<String, String> {
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    common::compile_to_ir(
        __module_ast_self,
        "access_soundness_module",
        name,
        &generics,
        strides,
        &[],
        &[],
        Some((16, 1, 1)),
        &CompileOptions::default(),
    )
    .map_err(|err| err.to_string())
}

// A signed `-1` passes `index < ceil(extent/tile)`, so the upper comparison
// alone is not a bounds check. The dynamic path must also prove or emit
// `0 <= index`.
#[test]
fn runtime_scalar_index_is_bounded_below() {
    common::with_test_stack(|| {
        let mlir = compile("runtime_scalar_index", &["64"], &[("z", &[1]), ("x", &[1])])
            .expect("compile runtime_scalar_index");
        assert!(
            mlir.contains("greater_than_or_equal"),
            "a runtime index needs a lower-bound guard, found none:\n{mlir}"
        );
    });
}

// Iteration -2 is a provably negative block index: reject at compile time,
// exactly as the static path rejects a provably out-of-range constant.
#[test]
fn provably_negative_loop_start_is_rejected() {
    common::with_test_stack(|| {
        let err = compile("negative_loop_start", &["64"], &[("z", &[1]), ("x", &[1])])
            .expect_err("a provably negative block index must not compile");
        assert!(
            err.contains("0 <=") || err.to_lowercase().contains("negative"),
            "expected a lower-bound diagnostic, got: {err}"
        );
    });
}

// The declared `dim(z,0) == dim(x,0)` talks about the WRONG root axis for a
// permuted target: the access is bounded by x's root axis 1 = dim_map[0]. The
// wrong-axis fact must not discharge it.
//
// Since cross-tensor obligations now relocate to launch, "not discharged" no
// longer means "a device assert appears" — it means the enforcement is on the
// axis the access actually needs. That is the stronger property, so it is what
// this asserts: a launch satisfying the DECLARED fact but violating the real
// one must still be rejected. If the wrong-axis equality were being believed,
// such a launch would sail through.
#[test]
fn wrong_axis_equality_must_not_discharge_a_permuted_access() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;
        let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "access_soundness_module",
            "permuted_wrong_axis",
        )
        .target("sm_120")
        .generics(vec![
            "32".to_string(),
            "32".to_string(),
            "4".to_string(),
            "4".to_string(),
        ])
        .strides(&[("z", &[256, 1]), ("x", &[256, 1])])
        .grid((16, 1, 1))
        .options(CompileOptions::default())
        .compile()
        .expect("compile permuted_wrong_axis");
        let checks = artifacts.launch_checks();
        assert!(
            !checks.is_empty(),
            "the axis-0 obligation must be enforced somewhere, and nothing \
             discharged it at compile time"
        );
        // Params in signature order: z, x. The access needs
        // `dim(z,0) == dim(x,1)`; the declared fact is `dim(z,0) == dim(x,0)`.
        // These shapes satisfy the declared fact and violate the real one.
        let declared_holds_real_fails = [vec![64i32, 64], vec![64, 32]];
        assert!(
            validate_launch_checks(
                checks,
                &declared_holds_real_fails,
                &declared_holds_real_fails,
                (1, 1, 1)
            )
            .is_err(),
            "a launch matching only the WRONG axis must be rejected: {checks:?}"
        );
        // The remapped axis agreeing is what actually makes the access safe.
        let real_holds = [vec![64i32, 64], vec![32, 64]];
        assert!(
            validate_launch_checks(checks, &real_holds, &real_holds, (1, 1, 1)).is_ok(),
            "a launch matching the remapped axis must be accepted: {checks:?}"
        );
    });
}

// Control: the remapped-axis equality is the right evidence and must
// discharge, proving the fix redirects the query rather than killing the rung.
#[test]
fn right_axis_equality_discharges_a_permuted_access() {
    common::with_test_stack(|| {
        let mlir = compile(
            "permuted_right_axis",
            &["32", "32", "4", "4"],
            &[("z", &[256, 1]), ("x", &[256, 1])],
        )
        .expect("compile permuted_right_axis");
        assert!(
            !(mlir.contains("out of bounds: dim 0") || mlir.contains("out of range: dim 0")),
            "the remapped-axis equality should discharge the axis-0 check:\n{mlir}"
        );
    });
}

// A conditionally reassigned index must not inherit the proof of the value
// it replaced. The join clears value-dependent facts, so the access keeps a
// real check somewhere; full discharge here means stale provenance leaked
// through the join again (issue #212).
#[test]
fn reassigned_index_keeps_its_check() {
    common::with_test_stack(|| {
        let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "access_soundness_module",
            "reassigned_index_keeps_no_proof",
        )
        .target("sm_120")
        .generics(vec!["16".to_string()])
        .strides(&[("z", &[1]), ("x", &[1])])
        .options(CompileOptions::default())
        .compile()
        .expect("compile reassigned_index_keeps_no_proof");
        let counts = artifacts.check_counts();
        let placed = counts.hoisted + counts.in_place + artifacts.launch_checks().len() as u32;
        assert!(
            placed >= 1,
            "the reassigned index must keep a real check; fully discharged \
             means the join leaked stale provenance: discharged={} hoisted={} \
             in_place={} launch={}",
            counts.discharged,
            counts.hoisted,
            counts.in_place,
            artifacts.launch_checks().len()
        );
    });
}

// A partition reassigned across a conditional must not keep the origin of the
// value it replaced; the cross-tensor rung would otherwise prove a b-access
// against a's extent (issue #212, structural residual).
#[test]
fn reassigned_partition_keeps_its_check() {
    common::with_test_stack(|| {
        let art = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "access_soundness_module",
            "reassigned_partition_keeps_no_origin",
        )
        .target("sm_120")
        .generics(vec!["16".to_string()])
        .strides(&[("z", &[1]), ("w", &[1]), ("a", &[1]), ("b", &[1])])
        .options(CompileOptions::default())
        .compile()
        .expect("compile reassigned_partition_keeps_no_origin");
        let c = art.check_counts();
        let placed = c.hoisted + c.in_place + art.launch_checks().len() as u32;
        assert!(
            placed >= 1,
            "reassigned partition kept a stale origin and discharged: \
             discharged={} hoisted={} in_place={} launch={}",
            c.discharged,
            c.hoisted,
            c.in_place,
            art.launch_checks().len()
        );
    });
}

// An interval fact is a claim about the machine value, and the machine wraps
// at i32 while interval composition is mathematical. An op whose mathematical
// range escapes the type's domain must yield NO interval — otherwise a later
// max/% narrows the mathematical range back into the domain while the wrapped
// machine value stays outside it, and the stale interval discharges or hoists
// away the only check standing between the access and memory it doesn't own
// (2026-08-12 review, S1: both scenarios, plus the static-fold twin).
#[test]
fn wrapping_intermediate_arithmetic_keeps_its_check() {
    common::with_test_stack(|| {
        for (name, is_static) in [
            ("wrapped_product_masked_by_max", false),
            ("wrapped_dividend_negative_remainder", false),
            ("wrapped_product_static_extent", true),
        ] {
            let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
                __module_ast_self,
                "access_soundness_module",
                name,
            )
            .target("sm_120")
            .generics(vec!["16".to_string()])
            .strides(&[("z", &[1]), ("x", &[1])])
            .options(CompileOptions::default())
            .compile()
            .unwrap_or_else(|err| panic!("compile {name}: {err}"));
            let c = artifacts.check_counts();
            assert_eq!(
                (c.discharged, c.hoisted, c.in_place),
                (0, 0, 1),
                "{name}: a wrap-tainted interval must not discharge or hoist \
                 the check (static extent: {is_static})"
            );
        }
        // Scenario 2's essence is the LOWER guard: the wrapped remainder is
        // negative, so the in-place check must be two-sided.
        let mlir = compile(
            "wrapped_dividend_negative_remainder",
            &["16"],
            &[("z", &[1]), ("x", &[1])],
        )
        .expect("compile wrapped_dividend_negative_remainder");
        assert!(
            mlir.contains("greater_than_or_equal"),
            "the wrapped remainder can be negative; the check must bound below:\n{mlir}"
        );
    });
}

#[test]
fn constant_if_preserves_facts_from_the_only_feasible_path() {
    common::with_test_stack(|| {
        let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "access_soundness_module",
            "constant_if_preserves_mapped_index",
        )
        .target("sm_120")
        .generics(vec!["16".to_string(), "1".to_string()])
        .strides(&[("z", &[1])])
        .grid((4, 1, 1))
        .options(CompileOptions::default())
        .compile()
        .expect("a constant if must preserve the taken path's mapped-index provenance");
        assert!(
            artifacts.ir_text().contains("store_view_tko"),
            "the proven mapped store should lower after an inlined exact condition"
        );

        let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "access_soundness_module",
            "constant_if_reassignment_uses_rhs_facts",
        )
        .target("sm_120")
        .generics(vec!["16".to_string()])
        .strides(&[("z", &[1]), ("x", &[1])])
        .options(CompileOptions::default())
        .compile()
        .expect("compile exact-branch reassignment");
        let counts = artifacts.check_counts();
        assert!(
            counts.hoisted + counts.in_place > 0,
            "the exact branch must publish the constant RHS without retaining the old iterand proof: {counts:?}"
        );
    });
}
