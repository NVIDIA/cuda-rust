/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core compiler struct for compiler2.
//!
//! Self-sufficient compiler that emits tile-ir ops directly, without wrapping
//! the old CUDATileFunctionCompiler.

use super::_module::CUDATileModules;
use crate::ast::{SourceLocation, SpanBase};
use crate::bounds::Bounds;
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::kernel_entry_generator::generate_entry_point;
use crate::kernel_naming::KernelNaming;
use crate::syn_utils::*;
use crate::types::get_cuda_tile_element_type_from_rust_primitive_str;
use crate::types::get_sig_param_mutability;
use cuda_async::device_context::Validator;

use super::_value::{BlockTerminator, CompilerContext, Mutability, TileRustValue};
use super::optimization_hints::{build_entry_optimization_hints, OptimizationHints};
use super::shared_types::EntryAttrs;
use super::tile_rust_type::TileRustType;
use crate::passes::proof_analysis::ProofResults;

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, DenseElements, FuncType, Module, ScalarType, TileElementType, TileType, Type,
};

use anyhow::Context as AnyhowContext;
use quote::ToTokens;
use std::any::type_name;
use std::cell::RefCell;
use std::collections::HashMap;
use syn::spanned::Spanned;

/// Compiles a single Rust function into Tile IR bytecode.
pub struct CUDATileFunctionCompiler<'m> {
    pub(crate) modules: &'m CUDATileModules,
    /// Bounds-check optimization policy for this compile, resolved once at
    /// construction. See [`crate::check_optimizations::CheckOptimizations`].
    pub(crate) check_opts: crate::check_optimizations::CheckOptimizations,
    pub(crate) module_name: String,
    pub(crate) _function_name: String,
    pub(crate) _function: &'m syn::ItemFn,
    pub(crate) entry: syn::ItemFn,
    pub(crate) entry_attrs: EntryAttrs,
    pub(crate) const_grid: Option<(u32, u32, u32)>,
    /// The believed-fact set, parsed once from declared preconditions. It is
    /// invariant for the whole compile, and every obligation resolution
    /// consults it — rebuilding it per obligation was O(accesses x facts) on
    /// the JIT hot path (issue #217).
    pub(crate) assumptions: crate::passes::obligation::Assumptions,
    /// Kernel parameter name → 0-based position in the signature (same order as
    /// `Validator.params` and the host arg-push order). Resolves the tensor
    /// names in obligations/preconditions to the `predicate::Atom::Dim` param
    /// index — the resolved-symbol identity the launch checks key on.
    pub(crate) param_index: HashMap<String, usize>,
    /// Parameter mutability by signature position. This decides the *frame* a
    /// parameter's extents live in: an immutable `&Tensor` is passed whole
    /// (view == root, which is what the host's shape array holds), while a
    /// `&mut Tensor` is slabbed and each work item sees one piece. Any atom
    /// naming an extent must respect it. See
    /// `.internal/tasks/in_progress/check_hoisting/BOUNDS_HOISTING_ANALYSIS.md` §SC1.
    pub(crate) param_is_mutable: Vec<bool>,
    /// Launch-time checks hoisted out of this kernel during compilation
    /// (`Resolution::Launch`). Interior-mutable because compile passes take
    /// `&self`; merged into the `Validator` in [`Self::get_validator`] and run
    /// once per launch by the host `validate_launch`.
    pub(crate) launch_checks: RefCell<Vec<cuda_async::predicate::LaunchCheck>>,
    pub(crate) gpu_name: String,
    pub(crate) target_capabilities: Option<cutile_ir::capabilities::TargetCapabilities>,
    pub(crate) optimization_hints: OptimizationHints,
    pub(crate) stride_args: HashMap<String, Vec<i32>>,
    pub(crate) generic_vars: GenericVars,
    pub(crate) validator: Validator,
    /// Module scopes, outermost first: the kernel's module at index 0 (the
    /// span base every diagnostic resolves against), then one entry per
    /// inline expansion in progress, pushed by [`Self::push_module_scope`].
    /// The innermost entry is the module whose source is being compiled,
    /// which is what module-level `const` names resolve against.
    pub(crate) module_name_stack: RefCell<Vec<String>>,
    pub(crate) typeck_results: RefCell<Option<crate::passes::type_inference::TypeckResults>>,
    /// Inline expansions currently in progress, innermost last. While
    /// non-empty, [`Self::ir_location`] scopes each op to the callee's own
    /// subprogram and wraps it in `Location::CallSite`, chaining "inlined
    /// at" info the bytecode debug section turns into inline debug frames.
    /// Interior mutability because inlining runs under `&self`.
    pub(crate) call_site_stack: RefCell<Vec<CallFrame>>,
    /// Per-compile partition-access check accounting: statically discharged /
    /// hoisted to a loop preheader / emitted in the access's own block.
    /// Reported on the `CUTILE_JIT_TIMING` line for coverage measurement.
    pub check_stats: CheckHoistStats,
}

/// One in-progress inline expansion.
pub(crate) enum CallFrame {
    /// User-function/method inlining: the callee body keeps its spans, so ops
    /// get the callee's own subprogram scope plus an "inlined at" chain —
    /// full debug frames, like the reference frontend gives user-authored
    /// helpers.
    Transparent {
        /// Where the call happened, in the caller's frame (itself chained
        /// when the caller was inlined too).
        caller: cutile_ir::ir::Location,
        callee_scope: Option<cutile_ir::ir::DISubprogram>,
        /// Span base of the module that owns the callee.
        span_base: SpanBase,
    },
    /// Core operations and callees without captured source are attributed
    /// to the caller's location, like builtins in the reference frontend.
    Opaque { caller: cutile_ir::ir::Location },
}

/// Pops the innermost call-site location on drop (see
/// [`CUDATileFunctionCompiler::push_call_site`]).
pub(crate) struct CallSiteGuard<'a>(&'a RefCell<Vec<CallFrame>>);

fn source_location(location: &cutile_ir::ir::Location) -> SourceLocation {
    use cutile_ir::ir::Location;
    match location {
        Location::Unknown => SourceLocation::unknown(),
        Location::FileLineCol {
            filename,
            line,
            column,
        } => SourceLocation::new(filename.clone(), *line as usize, *column as usize),
        Location::DebugInfo(info) => SourceLocation::new(
            info.filename.clone(),
            info.line as usize,
            info.column as usize,
        ),
        Location::CallSite { callee, .. } => source_location(callee),
    }
}

impl Drop for CallSiteGuard<'_> {
    fn drop(&mut self) {
        self.0.borrow_mut().pop();
    }
}

/// Pops the innermost module scope on drop (see
/// [`CUDATileFunctionCompiler::push_module_scope`]).
pub(crate) struct ModuleScopeGuard<'a>(&'a RefCell<Vec<String>>);

impl Drop for ModuleScopeGuard<'_> {
    fn drop(&mut self) {
        self.0.borrow_mut().pop();
    }
}

/// Counters for where partition-access bounds checks ended up.
#[derive(Debug, Default)]
pub struct CheckHoistStats {
    pub discharged: std::cell::Cell<u32>,
    pub hoisted: std::cell::Cell<u32>,
    pub in_place: std::cell::Cell<u32>,
}

struct FunctionParamTypes {
    names: Vec<String>,
    tile_types: Vec<TileRustType>,
}

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Route a `dim(a) == dim(b)` discharge through the obligation solver.
    ///
    /// Returns `true` when the equality is settled without an in-kernel check:
    /// at `Jit` if a declared `preconditions` fact entails it, otherwise at
    /// `Launch`, since both operands are extents the host holds before
    /// `cuLaunchKernel`. `false` only when it must fall to the device.
    ///
    /// Use this only where equality IS the requirement — today that is the
    /// shared mapped-grid check, whose zipped index streams need equal grids.
    /// An access bound wants [`Self::resolve_dim_le`] instead: emitting an
    /// equality there also rejects every target strictly larger than the
    /// source, which is safe.
    pub(crate) fn resolve_dim_eq(
        &self,
        a_tensor: &str,
        a_axis: usize,
        b_tensor: &str,
        b_axis: usize,
    ) -> bool {
        use cuda_async::predicate::{Atom, Predicate, Term};
        // Resolve both tensor names to param indices; if either is not a
        // parameter, the equality can't be canonicalized — conservatively not
        // discharged (falls to the existing in-kernel path).
        let (Some(&a_param), Some(&b_param)) = (
            self.param_index.get(a_tensor),
            self.param_index.get(b_tensor),
        ) else {
            return false;
        };
        let a = Term::atom(Atom::Dim {
            param: a_param,
            axis: a_axis,
        });
        let b = Term::atom(Atom::Dim {
            param: b_param,
            axis: b_axis,
        });
        let Some(predicate) = Predicate::eq(&a, &b) else {
            return false;
        };
        self.lower_obligation(
            predicate,
            format!("dim({a_tensor}, {a_axis}) == dim({b_tensor}, {b_axis})"),
        )
    }

    /// Route a `tiles(a) <= tiles(b)` discharge through the obligation
    /// solver — the exact form an access bound needs. An index drawn from
    /// `[0, ceil(dim(a)/t))` stays inside an axis of `ceil(dim(b)/t)` tiles
    /// (same tile `t`) exactly when the source has at most as many tiles;
    /// a target shorter in elements but equal in tiles is safe.
    ///
    /// Two steps, because the believed facts are equalities. A declared
    /// `dim(a) == dim(b)` entails the inequality but the solver has no
    /// entailment theory, so the equality is tried first purely as evidence
    /// (`Jit`, no check anywhere); only then is the inequality itself lowered,
    /// landing at `Launch` since extents are host-known. Emitting the
    /// inequality rather than the equality is what keeps the derived check
    /// faithful: with `<=`, the only over-rejection left is the sub-tile band
    /// where a shorter target happens to round up to the same tile count —
    /// exactness there needs an atom naming a tile count (see
    /// `HOISTING_COVERAGE.md`).
    pub(crate) fn resolve_dim_le(
        &self,
        a_tensor: &str,
        a_axis: usize,
        b_tensor: &str,
        b_axis: usize,
        tile: i32,
    ) -> bool {
        use crate::passes::obligation::{resolve, Obligation, Resolution};
        use cuda_async::predicate::{Atom, Predicate, Term};
        let (Some(&a_param), Some(&b_param)) = (
            self.param_index.get(a_tensor),
            self.param_index.get(b_tensor),
        ) else {
            return false;
        };
        let a = Term::atom(Atom::Dim {
            param: a_param,
            axis: a_axis,
        });
        let b = Term::atom(Atom::Dim {
            param: b_param,
            axis: b_axis,
        });
        // Step 1: a declared equality decides the goal at Jit. Evidence only —
        // when it is not entailed, no equality check is emitted anywhere.
        if let Some(eq) = Predicate::eq(&a, &b) {
            let obligation = Obligation::new(eq, "");
            if matches!(resolve(&obligation, &self.assumptions), Resolution::Jit) {
                return true;
            }
        }
        // Step 2: state the real goal — over TILE COUNTS, which is what the
        // walk is actually bounded by — and let it land where its operands
        // are known, a host compare at launch. Stating it over extents was
        // stricter than the device check it replaced (a target shorter in
        // elements but equal in tiles was rejected; issue #216); the
        // TileCount atom closes that band exactly. Without launch
        // relocation the goal stays with the access as a device check;
        // step 1 still ran, because a declared fact is a verified proof,
        // not a placement.
        if !self.check_opts.relocate_to_launch {
            return false;
        }
        if tile < 1 {
            return false;
        }
        let ta = Term::atom(Atom::TileCount {
            param: a_param,
            axis: a_axis,
            tile,
        });
        let tb = Term::atom(Atom::TileCount {
            param: b_param,
            axis: b_axis,
            tile,
        });
        let Some(le) = Predicate::le(&ta, &tb) else {
            return false;
        };
        self.lower_obligation(
            le,
            format!(
                "ceil(dim({a_tensor}, {a_axis})/{tile}) <= ceil(dim({b_tensor}, {b_axis})/{tile})"
            ),
        )
    }

    /// Lower a safety obligation: resolve `predicate` against the in-scope
    /// assumptions and route the result. Returns `true` when the obligation was
    /// placed without any in-kernel check — either statically discharged
    /// (`Jit`) or evacuated to a host launch check (`Launch`, collected here);
    /// `false` when it must fall to a device-stage (in-kernel) assert, which the
    /// caller emits.
    ///
    /// This is the shared "lower an assert to an obligation" path. Any client
    /// that can name a `Predicate` routes through it and the check lands at the
    /// earliest stage a dominating assumption allows — the bounded-access check
    /// today; a programmer `require`/`assert` op next. It is what lets *any*
    /// assert be hoisted out of the kernel when its operands are launch-known.
    pub(crate) fn lower_obligation(
        &self,
        predicate: cuda_async::predicate::Predicate,
        cause: impl Into<String>,
    ) -> bool {
        use crate::passes::obligation::{resolve, Obligation, Resolution};
        let obligation = Obligation::new(predicate, cause);
        match resolve(&obligation, &self.assumptions) {
            Resolution::Jit => true,
            Resolution::Launch(check) => {
                // Canonical predicates make duplicate detection exact. Two
                // obligations reducing to the same host compare (e.g. a shared
                // `Dim`'s divisibility, once per binding) cost one check.
                let mut checks = self.launch_checks.borrow_mut();
                if !checks.iter().any(|c| c.predicate == check.predicate) {
                    checks.push(check);
                }
                true
            }
            Resolution::Device => false,
        }
    }

    /// The kernel-visible extent of `partition`'s axis `axis`, taken from the
    /// *declared* parameter shape.
    ///
    /// The partition view type erases a `&mut Tensor` parameter's shape to `?`
    /// — the host passes a slab, so the view is built from runtime shape
    /// operands — which loses extents the signature states outright
    /// (`&mut Tensor<f32, {[1, N]}>` lowers to `tensor_view<?x?xf32>`). The
    /// validator still holds them, and the generated launcher rejects a
    /// mismatch (`valid_shape == given_shape`) before the kernel runs, so this
    /// is a *verified* assumption rather than a trusted one (SC2).
    ///
    /// This exists so that the binding check and the access check answer
    /// "what is this axis's extent?" from the *same* source. They previously
    /// disagreed — the binding read the declared shape and folded at compile
    /// time, the access read the erased view type and fell back to a launch
    /// check — which manufactured launch checks for extents the signature had
    /// already pinned.
    ///
    /// FRAME (SC1): the declared shape may stand in for the view extent only
    /// where the two coincide. That holds for `Partition` and `PartitionMut` —
    /// an immutable `&Tensor` has view == root, and a `&mut Tensor`'s declared
    /// shape *is* its per-CTA slab. It does **not** hold for a
    /// `MappedPartition*`, whose declared shape is the *tile* shape, so callers
    /// must be reachable only from plain partition paths. See
    /// `.internal/tasks/in_progress/check_hoisting/BOUNDS_HOISTING_ANALYSIS.md` §SC1.
    pub(crate) fn declared_view_extent(
        &self,
        partition: &TileRustValue,
        dim_map: &[i32],
        axis: usize,
    ) -> Option<i32> {
        let tensor_axis = *dim_map.get(axis).filter(|&&d| d >= 0)? as usize;
        let idx = *self.param_index.get(partition.tensor_origin.as_ref()?)?;
        let shape = match self.validator.params.get(idx)? {
            cuda_async::device_context::ValidParamType::Tensor(t) => &t.shape,
            _ => return None,
        };
        shape.get(tensor_axis).copied().filter(|&e| e >= 0)
    }

    /// Give `x.shape()`'s per-axis values their symbolic identity: axis `k` of
    /// a tensor *parameter*'s shape is exactly `extent(param, k)`, so label it
    /// with that atom.
    ///
    /// This is the bridge between the two fact systems. Facts carried on values
    /// (bounds, provenance) and predicates over [`cuda_async::predicate::Atom`]s
    /// have no shared vocabulary, so an extent read out of the signature was an
    /// opaque scalar: an obligation mentioning it could not be seen as
    /// launch-known and fell straight to a device check. One label makes every
    /// such extent speakable in the predicate language, for every obligation —
    /// nothing here is specific to any one check.
    ///
    /// FRAME (SC1): the atom picks the frame, via [`Self::extent_atom`] — the
    /// one place that decision lives. `x.shape()` inside the kernel returns
    /// whatever the kernel sees (the whole tensor for an immutable param, the
    /// per-CTA slab for a `&mut`), and the chosen atom resolves on the host
    /// against the array holding exactly that quantity.
    pub(crate) fn label_param_extents(&self, shape: &mut TileRustValue, tensor: &TileRustValue) {
        let Some(param) = tensor
            .tensor_origin
            .as_ref()
            .and_then(|name| self.param_index.get(name).copied())
        else {
            return;
        };
        let Some(dims) = shape
            .fields
            .as_mut()
            .and_then(|fields| fields.get_mut("dims"))
            .and_then(|dims| dims.values.as_mut())
        else {
            return;
        };
        for (axis, dim) in dims.iter_mut().enumerate() {
            if dim.term.is_none() {
                dim.term = Some(cuda_async::predicate::Term::atom(
                    self.extent_atom(param, axis),
                ));
            }
        }
    }

    /// The atom naming `param`'s extent on `axis`, in the frame the kernel
    /// actually sees — the single place the SC1 frame decision is encoded.
    ///
    /// An immutable `&Tensor` is passed whole, so its kernel-visible extent is
    /// the *root* extent: [`cuda_async::predicate::Atom::Dim`], the frame
    /// declared `preconditions` are stated in (so declared facts can entail
    /// obligations over it). A `&mut Tensor` is slabbed — each work item sees
    /// one piece — so its extent is
    /// [`cuda_async::predicate::Atom::ViewExtent`], resolved on the host
    /// against the partition shape. The two are distinct atom variants
    /// precisely so a root-frame fact can never entail a view-frame obligation
    /// by structural equality: the frame confusion is unrepresentable.
    pub(crate) fn extent_atom(&self, param: usize, axis: usize) -> cuda_async::predicate::Atom {
        if self.param_is_mutable.get(param).copied().unwrap_or(true) {
            cuda_async::predicate::Atom::ViewExtent { param, axis }
        } else {
            cuda_async::predicate::Atom::Dim { param, axis }
        }
    }

    /// `tensor`'s parameter index, but only when its kernel-visible extent IS
    /// its root extent — the frame declared `preconditions` speak in.
    ///
    /// Any reasoning that relates a partition axis to a *declared* fact must
    /// go through this: `dim(t, a)` names the whole tensor, so a fact about it
    /// says nothing about a `&mut` parameter's per-CTA slab, and a tile count
    /// taken over that slab is not the count the fact divides. Rather than
    /// re-deriving mutability, this asks [`Self::extent_atom`] — the one place
    /// the frame decision lives — and accepts only the root answer.
    pub(crate) fn root_framed_param(&self, tensor: &str) -> Option<usize> {
        let param = *self.param_index.get(tensor)?;
        matches!(
            self.extent_atom(param, 0),
            cuda_async::predicate::Atom::Dim { .. }
        )
        .then_some(param)
    }

    /// The `deny_in_kernel_checks` policy, applied at the one moment it means
    /// something: a *compiler-synthesized* safety check is about to become a
    /// device instruction.
    ///
    /// Every synthesized check must call this immediately before emitting its
    /// `Assert`. The attribute promises that no safety check remains in the
    /// kernel; a site that skips it makes the promise partial, which is worse
    /// than absent — an author who trusts the flag would ship exactly the
    /// register cost it claims to forbid. (A programmer-written `assert!` is
    /// not a synthesized check and must NOT route through here: the flag
    /// governs what the compiler adds, not what the author asked for.)
    ///
    /// `what` names the check; `remedy` says how to discharge it. The caller
    /// keeps emitting its own assert when the flag is off, so this is purely a
    /// policy gate — it never changes what is verified, only whether a residual
    /// in-kernel check is tolerated.
    pub(crate) fn deny_residual_check(
        &self,
        what: &str,
        remedy: &str,
        span: &proc_macro2::Span,
    ) -> Result<(), JITError> {
        if !self.entry_attrs.get_entry_arg_bool("deny_in_kernel_checks") {
            return Ok(());
        }
        self.jit_error_result(
            span,
            &format!(
                "`deny_in_kernel_checks`: {what} could not be discharged at compile time \
                 or hoisted to launch, so it would be emitted in the kernel and cost \
                 device registers. {remedy}. Or drop the flag."
            ),
        )
    }

    pub fn new(
        modules: &'m CUDATileModules,
        module_name: &str,
        function_name: &str,
        function_generic_args: &[String],
        stride_args: &[(&str, &[i32])],
        spec_args: &[(&str, &crate::specialization::SpecializationBits)],
        scalar_hints: &[(&str, &crate::specialization::DivHint)],
        const_grid: Option<(u32, u32, u32)>,
        gpu_name: String,
        compile_options: &crate::hints::CompileOptions,
    ) -> Result<Self, JITError> {
        // 1. Check module exists.
        if !modules.modules().contains_key(module_name) {
            return Err(JITError::Generic(format!(
                "Undefined module: {module_name}"
            )));
        }

        // 2. KernelNaming.
        let kernel_naming = KernelNaming::new(function_name);

        // 3. Look up function.
        let (_, function) = modules
            .functions()
            .get(kernel_naming.public_name())
            .with_context(|| format!("Undefined function: {function_name}"))?;

        // 4. Parse entry_attrs.
        let entry_attrs =
            get_meta_list_by_last_segment("entry", &function.attrs).ok_or_else(|| {
                modules
                    .resolve_span(module_name, &function.span())
                    .jit_error(&format!(
                    "function `{function_name}` is missing a required `#[entry(...)]` attribute"
                ))
            })?;
        let entry_attrs = EntryAttrs { entry_attrs };
        let proof_results = ProofResults::analyze_entry_attrs(&entry_attrs)?;

        // 5. Check unchecked_accesses.
        if entry_attrs.get_entry_arg_bool("unchecked_accesses") && function.sig.unsafety.is_none() {
            return modules
                .resolve_span(module_name, &function.span())
                .jit_error_result(
                    "kernel must be declared `unsafe` when `unchecked_accesses` is enabled",
                );
        }

        // 6. Parse optimization_hints.
        let mut optimization_hints = match entry_attrs.get_entry_arg_expr("optimization_hints") {
            Some(hints_expr) => OptimizationHints::parse(hints_expr, gpu_name.clone())?,
            None => {
                let mut hints = OptimizationHints::empty();
                hints.target_gpu_name = Some(gpu_name.clone());
                hints
            }
        };
        // Runtime compile options override entry-level hints.
        optimization_hints.apply_compile_options(compile_options);

        // 7. Build stride_args HashMap.
        let stride_args: HashMap<String, Vec<i32>> = stride_args
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect::<HashMap<_, _>>();

        // 8. Create GenericVars. `from_flat` has no span of its own; point
        // its diagnostics (too few / malformed host-supplied generics) at the
        // kernel's generic parameter list.
        let mut generic_vars =
            GenericVars::from_flat(&function.sig.generics, function_generic_args).map_err(
                |err| match err {
                    JITError::Located(message, location) if !location.is_known() => modules
                        .resolve_span(module_name, &function.sig.generics.span())
                        .jit_error(&message),
                    other => other,
                },
            )?;
        Self::add_module_const_vars_from_modules(modules, module_name, &mut generic_vars);

        // 9. generate_entry_point.
        let spec_args_map: HashMap<String, crate::specialization::SpecializationBits> = spec_args
            .iter()
            .map(|(k, v)| (k.to_string(), (*v).clone()))
            .collect();
        let scalar_max_divisibility = optimization_hints
            .target_gpu_name
            .as_ref()
            .and_then(|target| optimization_hints.tile_as_hints.get(target))
            .and_then(|hints| hints.max_divisibility);
        let scalar_hints_map: HashMap<String, crate::specialization::DivHint> = scalar_hints
            .iter()
            .map(|&(name, hint)| {
                let hint = scalar_max_divisibility.map_or(*hint, |max| hint.with_max(max));
                (name.to_string(), hint)
            })
            .collect();
        let (entry, validator) = generate_entry_point(
            modules,
            function,
            &generic_vars,
            &stride_args,
            &spec_args_map,
            &scalar_hints_map,
            modules.primitives(),
            &optimization_hints,
        )?;

        // 10. Check namespace collision.
        if modules
            .functions()
            .get(kernel_naming.entry_name().as_str())
            .is_some()
        {
            return modules
                .resolve_span(module_name, &function.span())
                .jit_error_result(&format!(
                    "Entry point namespace collision: {}",
                    kernel_naming.entry_name()
                ));
        }

        // 11. Optional print_ir.
        if entry_attrs.get_entry_arg_bool("print_ir") {
            println!("GENERATED ENTRY POINT: {module_name}::{function_name}");
            println!("{}", item_string_pretty(&entry.clone().into()));
            println!();
        }

        // Resolve parameter names to signature positions (same order as
        // `Validator.params` and the host arg-push order).
        let param_index: HashMap<String, usize> = function
            .sig
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| match arg {
                syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                    syn::Pat::Ident(pat_ident) => Some((pat_ident.ident.to_string(), i)),
                    _ => None,
                },
                syn::FnArg::Receiver(_) => None,
            })
            .collect();

        let param_is_mutable = get_sig_param_mutability(&function.sig);

        // 12. Build struct directly.
        let assumptions = crate::passes::obligation::Assumptions::from_preconditions(
            &proof_results,
            &param_index,
        );
        // The environment resolves first (the differential harness's
        // ablation switches), then --device-debug tightens placement: no
        // in-kernel code motion (see CheckOptimizations::device_debug for
        // why discharge and launch relocation are untouched).
        let mut check_opts = crate::check_optimizations::CheckOptimizations::from_env();
        if compile_options.device_debug {
            check_opts.hoist_to_preheaders = false;
        }
        Ok(CUDATileFunctionCompiler {
            modules,
            check_opts,
            module_name: module_name.to_string(),
            _function_name: function_name.to_string(),
            entry_attrs,
            const_grid,
            assumptions,
            param_index,
            param_is_mutable,
            launch_checks: RefCell::new(Vec::new()),
            gpu_name,
            target_capabilities: None,
            optimization_hints,
            _function: function,
            entry,
            validator,
            generic_vars,
            stride_args,
            module_name_stack: RefCell::new(vec![module_name.to_string()]),
            typeck_results: RefCell::new(None),
            call_site_stack: RefCell::new(Vec::new()),
            check_stats: CheckHoistStats::default(),
        })
    }

    // -----------------------------------------------------------------------
    // Error helper methods
    // -----------------------------------------------------------------------

    /// Adds the module-level `const`s visible from the module currently being
    /// compiled (see [`Self::current_module`]) as generic variables.
    pub(crate) fn add_module_const_vars(&self, generic_vars: &mut GenericVars) {
        Self::add_module_const_vars_from_modules(
            self.modules,
            &self.current_module(),
            generic_vars,
        );
    }

    /// Adds the `const`s visible from `module` — its own, its imports, core,
    /// then a unique global definition — as generic variables. Scoped per
    /// module: a helper module's `N` must not stand in for the kernel's `N`
    /// (or vice versa), which the flat, last-indexed-wins map allowed.
    fn add_module_const_vars_from_modules(
        modules: &CUDATileModules,
        module: &str,
        generic_vars: &mut GenericVars,
    ) {
        for (name, item) in modules.consts_visible_from(module).iter() {
            if generic_vars.var_type(name).is_some() {
                continue;
            }
            if let Some(value) = crate::type_aliases::const_item_i32_value(item) {
                generic_vars.inst_i32.insert(name.clone(), value);
            } else if let Some(value) = crate::type_aliases::const_item_bool_value(item) {
                generic_vars.inst_bool.insert(name.clone(), value);
            }
        }
    }

    /// The module whose source is being compiled right now: the kernel's
    /// module, or the callee's module inside an inline expansion.
    pub(crate) fn current_module(&self) -> String {
        self.module_name_stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| self.module_name.clone())
    }

    /// Enters `module`'s scope for the duration of the returned guard (an
    /// inline expansion of one of its functions or methods), so that its
    /// `const`s and the consts named in its types resolve in its own scope.
    pub(crate) fn push_module_scope(&self, module: &str) -> ModuleScopeGuard<'_> {
        self.module_name_stack.borrow_mut().push(module.to_string());
        ModuleScopeGuard(&self.module_name_stack)
    }

    pub(crate) fn span_base(&self) -> SpanBase {
        let stack = self.module_name_stack.borrow();
        self.modules
            .get_span_base(&stack[0])
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn resolve_span(&self, span: &proc_macro2::Span) -> SourceLocation {
        match self.call_site_stack.borrow().last() {
            Some(CallFrame::Transparent { span_base, .. }) => span_base.resolve_span(span),
            Some(CallFrame::Opaque { caller }) => source_location(caller),
            None => self.span_base().resolve_span(span),
        }
    }

    /// User modules keep their own locations across inlining. Compare span
    /// bases because the registry supports short and qualified core names.
    pub(crate) fn has_inline_source(&self, module_name: &str) -> bool {
        let Some(base) = self
            .modules
            .get_span_base(module_name)
            .filter(|b| b.is_known())
        else {
            return false;
        };
        let core_base = self
            .modules
            .name_resolver
            .core_module()
            .and_then(|core| self.modules.get_span_base(core));
        !core_base.is_some_and(|core| {
            core.file == base.file
                && core.base_line == base.base_line
                && core.base_col == base.base_col
        })
    }

    /// Convert a proc_macro2 span into a tile-ir Location for IR operations.
    ///
    /// Inside an inline expansion (see [`Self::push_call_site`]) the
    /// location is wrapped in `Location::CallSite`, recording where the
    /// containing function was inlined; nesting chains naturally because
    /// each pushed caller location was itself resolved under the outer
    /// stack.
    pub(crate) fn ir_location(&self, span: &proc_macro2::Span) -> cutile_ir::ir::Location {
        let stack = self.call_site_stack.borrow();
        let (caller, callee_scope, span_base) = match stack.last() {
            None => {
                let loc = self.resolve_span(span);
                return if loc.is_known() {
                    cutile_ir::ir::Location::FileLineCol {
                        filename: loc.file,
                        line: loc.line as u32,
                        column: loc.column as u32,
                    }
                } else {
                    cutile_ir::ir::Location::Unknown
                };
            }
            // Core operations are shown at the user's call site.
            Some(CallFrame::Opaque { caller }) => return caller.clone(),
            Some(CallFrame::Transparent {
                caller,
                callee_scope,
                span_base,
            }) => (caller, callee_scope, span_base),
        };
        // Transparent frame: real callee spans, resolved against the
        // callee's own module base; scope the op to the CALLEE's subprogram
        // (so inline debug frames name the inlined function, not the kernel
        // entry) and chain where it was inlined.
        let loc = span_base.resolve_span(span);
        if !loc.is_known() {
            return cutile_ir::ir::Location::Unknown;
        }
        let callee = match callee_scope {
            Some(sp) => cutile_ir::ir::Location::DebugInfo(cutile_ir::ir::DebugInfoLoc {
                filename: loc.file,
                line: loc.line as u32,
                column: loc.column as u32,
                scope: cutile_ir::ir::DebugScope::Subprogram(sp.clone()),
            }),
            None => cutile_ir::ir::Location::FileLineCol {
                filename: loc.file,
                line: loc.line as u32,
                column: loc.column as u32,
            },
        };
        cutile_ir::ir::Location::CallSite {
            callee: Box::new(callee),
            caller: Box::new(caller.clone()),
        }
    }

    /// Pushes an opaque inline expansion: every op inside is attributed to
    /// the call site, in the caller's frame.
    pub(crate) fn push_opaque_call_site(&self, call_span: &proc_macro2::Span) -> CallSiteGuard<'_> {
        let caller = self.ir_location(call_span);
        self.call_site_stack
            .borrow_mut()
            .push(CallFrame::Opaque { caller });
        CallSiteGuard(&self.call_site_stack)
    }

    /// Pushes an inline expansion; popped when the returned guard drops, so
    /// error paths unwind the stack correctly. `callee_name` and
    /// `callee_def_span` identify the function being inlined; its
    /// subprogram gets a synthetic linkage name (`name@file:line:col`,
    /// following the reference frontend) since inlined-away functions have
    /// no symbol.
    pub(crate) fn push_call_site(
        &self,
        call_span: &proc_macro2::Span,
        callee_name: &str,
        callee_def_span: &proc_macro2::Span,
        callee_module: &str,
    ) -> CallSiteGuard<'_> {
        // The call expression is caller-frame code: resolved under the
        // CURRENT stack (outer frame's base, or the entry module's).
        let caller = self.ir_location(call_span);
        let span_base = self
            .modules
            .get_span_base(callee_module)
            .cloned()
            .unwrap_or_default();
        let def = span_base.resolve_span(callee_def_span);
        let callee_scope = if def.is_known() {
            let (dir, base) = match def.file.rfind('/') {
                Some(i) => (&def.file[..i], &def.file[i + 1..]),
                None => ("", def.file.as_str()),
            };
            let file = cutile_ir::ir::DIFile {
                name: base.to_string(),
                directory: dir.to_string(),
            };
            // All inlined code belongs to the kernel's compilation unit,
            // even when its definition lives in another source file. Keep
            // that file on the subprogram, not on a second compile unit:
            // tileiras 13.3 rejects our live cross-CU regression with -g.
            let root = self.span_base();
            let (root_dir, root_file) = root.file.rsplit_once('/').unwrap_or(("", &root.file));
            let compile_unit = cutile_ir::ir::DICompileUnit {
                file: cutile_ir::ir::DIFile {
                    name: root_file.to_string(),
                    directory: root_dir.to_string(),
                },
            };
            let stem = base.split('.').next().unwrap_or("anonymous");
            Some(cutile_ir::ir::DISubprogram {
                file,
                line: def.line as u32,
                name: callee_name.to_string(),
                linkage_name: format!("{callee_name}@{stem}:{}:{}", def.line, def.column),
                compile_unit,
                scope_line: def.line as u32,
            })
        } else {
            None
        };
        self.call_site_stack
            .borrow_mut()
            .push(CallFrame::Transparent {
                caller,
                callee_scope,
                span_base,
            });
        CallSiteGuard(&self.call_site_stack)
    }

    pub(crate) fn jit_error(&self, span: &proc_macro2::Span, error_message: &str) -> JITError {
        self.resolve_span(span).jit_error(error_message)
    }

    pub(crate) fn jit_error_result<R>(
        &self,
        span: &proc_macro2::Span,
        error_message: &str,
    ) -> Result<R, JITError> {
        self.resolve_span(span).jit_error_result(error_message)
    }

    // -----------------------------------------------------------------------
    // Typeck query helper methods
    // -----------------------------------------------------------------------

    pub(crate) fn typeck_method_selection(
        &self,
        method_call_expr: &syn::ExprMethodCall,
    ) -> Option<crate::passes::type_inference::MethodSelection> {
        self.typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.method_selection(method_call_expr).cloned())
    }

    pub(crate) fn typeck_expr_syn_type(&self, expr: &syn::Expr) -> Option<syn::Type> {
        self.typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.syn_expr_type(expr))
    }

    pub(crate) fn typeck_expr_tile_type(
        &self,
        expr: &syn::Expr,
        generic_vars: &GenericVars,
        type_params: &HashMap<String, crate::types::TypeParam>,
    ) -> Result<Option<TileRustType>, JITError> {
        let cached_tile_type = self
            .typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.expr_type(expr).cloned());
        if cached_tile_type.is_some() {
            return Ok(cached_tile_type);
        }

        let Some(syn_type) = self.typeck_expr_syn_type(expr) else {
            return Ok(None);
        };
        self.compile_type(&syn_type, generic_vars, type_params)
    }

    // -----------------------------------------------------------------------
    // Compile
    // -----------------------------------------------------------------------

    /// Compile the kernel function into a `cutile_ir::Module`.
    pub fn compile(&self) -> Result<Module, JITError> {
        let mut module = Module::new(&self.module_name);
        if let Some(item) = self.modules.modules().get(&self.module_name) {
            for attr in &item.attrs {
                if attr.path().is_ident("noname") {
                    let options = crate::syn_utils::SingleMetaList::from_attribute(attr.clone());
                    if let Some(producer) = options.parse_string("producer") {
                        if let Some(caps) = &self.target_capabilities {
                            caps.require_version(
                                "module.producer",
                                cutile_ir::requirements::Feature::Producer.since(),
                                &self.ir_location(&item.span()),
                            )?;
                        }
                        module.producer = Some(producer);
                    }
                }
            }
        }
        self.emit_module_globals(&mut module)?;
        let entry_op = self.compile_entry_function(&mut module)?;
        module.functions.push(entry_op);
        if let Some(capabilities) = &self.target_capabilities {
            capabilities
                .validate_module(&module)
                .map_err(JITError::from)?;
        }
        Ok(module)
    }

    /// Attach the target already negotiated at JIT, or an explicit target in
    /// driver-free compile tests. The constructor remains driver-independent.
    pub fn with_target_capabilities(
        mut self,
        capabilities: cutile_ir::capabilities::TargetCapabilities,
    ) -> Result<Self, JITError> {
        if capabilities.architecture != self.gpu_name {
            return Err(JITError::Generic(format!(
                "compiler target {} differs from capability target {}",
                self.gpu_name, capabilities.architecture
            )));
        }
        self.target_capabilities = Some(capabilities);
        Ok(self)
    }

    fn compile_function_param_types(
        &self,
        fn_item: &syn::ItemFn,
        generic_vars: &GenericVars,
    ) -> Result<FunctionParamTypes, JITError> {
        let names = get_sig_param_names(&fn_item.sig);
        let (r_params, _r_result) = get_sig_types(&fn_item.sig, None);
        let mut tile_types = Vec::new();

        for (i, r_param_type) in r_params.iter().enumerate() {
            let mut type_params: HashMap<String, crate::types::TypeParam> = HashMap::new();
            if let Some(strides) = self.stride_args.get(names[i].as_str()) {
                type_params.insert(
                    "strides".to_string(),
                    crate::types::TypeParam::Strides(crate::types::TypeParamStrides::from(
                        syn::parse2::<syn::Type>(
                            format!(
                                "Array<{{[{}]}}>",
                                strides
                                    .iter()
                                    .map(|i| i.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                            .parse()
                            .unwrap(),
                        )
                        .unwrap(),
                    )),
                );
            }
            let Some(ty) = self.compile_type(r_param_type, generic_vars, &type_params)? else {
                return self.jit_error_result(
                    &r_param_type.span(),
                    &format!(
                        "unable to compile parameter type `{}`",
                        r_param_type.to_token_stream()
                    ),
                );
            };
            tile_types.push(ty);
        }

        Ok(FunctionParamTypes { names, tile_types })
    }

    fn initial_typeck_types(
        &self,
        param_types: &FunctionParamTypes,
        generic_vars: &GenericVars,
    ) -> Result<HashMap<String, TileRustType>, JITError> {
        let mut initial_types = param_types
            .names
            .iter()
            .cloned()
            .zip(param_types.tile_types.iter().cloned())
            .collect::<HashMap<_, _>>();

        let i32_ty: syn::Type = syn::parse_quote!(i32);
        for (key, _) in generic_vars.ordered_inst_i32() {
            let Some(ty) = self.compile_type(&i32_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic i32 type");
            };
            initial_types.insert(key.to_string(), ty);
        }

        let bool_ty: syn::Type = syn::parse_quote!(bool);
        for (key, _) in generic_vars.ordered_inst_bool() {
            let Some(ty) = self.compile_type(&bool_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic bool type");
            };
            initial_types.insert(key.to_string(), ty);
        }

        for (key, value) in generic_vars.ordered_inst_array() {
            let arr_ty =
                syn::parse2::<syn::Type>(format!("[i32;{}]", value.len()).parse().unwrap())
                    .unwrap();
            let Some(ty) = self.compile_type(&arr_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic array type");
            };
            initial_types.insert(key.to_string(), ty);
        }

        Ok(initial_types)
    }

    #[doc(hidden)]
    pub fn debug_typeck_dump(&self) -> Result<String, JITError> {
        let fn_item = self._function;
        let generic_vars = &self.generic_vars;
        let param_types = self.compile_function_param_types(fn_item, generic_vars)?;
        let initial_types = self.initial_typeck_types(&param_types, generic_vars)?;

        let mut typed_fn_item = fn_item.clone();
        crate::passes::node_ids::assign_expr_ids(&mut typed_fn_item);
        let typeck_results = crate::passes::type_inference::infer_function(
            self,
            &typed_fn_item,
            generic_vars,
            initial_types,
        )?;
        Ok(typeck_results.debug_dump())
    }

    /// Compile the entry function, returning its OpId.
    fn compile_entry_function(&self, module: &mut Module) -> Result<cutile_ir::ir::OpId, JITError> {
        let fn_item = &self.entry;
        let fn_name = fn_item.sig.ident.to_string();
        let generic_vars = &self.generic_vars;

        let param_types = self.compile_function_param_types(fn_item, generic_vars)?;
        let var_names = &param_types.names;
        let cuda_tile_argument_types = &param_types.tile_types;
        let mut arg_tile_ir_types = Vec::new();
        for ty in cuda_tile_argument_types {
            let tile_ir_ty = super::_type::convert_type(ty).ok_or_else(|| {
                JITError::Generic(format!(
                    "compiler2: failed to convert parameter type to tile-ir: {}",
                    ty.rust_ty.to_token_stream()
                ))
            })?;
            arg_tile_ir_types.push(tile_ir_ty);
        }

        let func_type = Type::Func(FuncType {
            inputs: arg_tile_ir_types.clone(),
            results: vec![],
        });

        let (region_id, block_id, block_args) =
            build_single_block_region(module, &arg_tile_ir_types);

        // Bind parameter names to block argument values using ported CompilerContext.
        let sig_param_mutability = get_sig_param_mutability(&fn_item.sig);
        let mut ctx = CompilerContext::empty();
        for (i, name) in var_names.iter().enumerate() {
            if i < block_args.len() {
                let ty = cuda_tile_argument_types[i].clone();
                let mut val = TileRustValue::new_value_kind_like(block_args[i], ty);
                val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                // Record which kernel parameter this value *is*. Names do not
                // survive inlining -- inside `Tensor::shape(&self)` the receiver
                // is `self` -- so provenance has to ride the value, not the
                // syntax, for anything downstream to resolve it to a parameter.
                val.tensor_origin = Some(name.clone());
                ctx.vars.insert(name.clone(), val);
                ctx.function_level_bindings.insert(name.clone());
            }
        }

        let initial_types = self.initial_typeck_types(&param_types, generic_vars)?;

        // Add const generics as variables.
        for (key, value) in generic_vars.ordered_inst_i32() {
            let tr_val = self.compile_constant(module, block_id, generic_vars, value)?;
            ctx.vars.insert(key.to_string(), tr_val);
        }

        for (key, value) in generic_vars.ordered_inst_bool() {
            let tr_val = self.compile_bool_constant(module, block_id, generic_vars, value)?;
            ctx.vars.insert(key.to_string(), tr_val);
        }

        // Add arrays as variables.
        for (key, value) in generic_vars.ordered_inst_array() {
            let arr_expr = syn::parse2::<syn::Expr>(format!("{value:?}").parse().unwrap()).unwrap();
            let arr_ty =
                syn::parse2::<syn::Type>(format!("[i32;{}]", value.len()).parse().unwrap())
                    .unwrap();
            let ty = self.compile_type(&arr_ty, generic_vars, &HashMap::new())?;
            let tr_val = self
                .compile_expression(module, block_id, &arr_expr, generic_vars, &mut ctx, ty)?
                .expect("Failed to compile CGA as var.");
            ctx.vars.insert(key.to_string(), tr_val);
        }

        ctx.default_terminator = Some(BlockTerminator::Return);
        ctx.fn_body = true;
        ctx.kernel_entry = true;

        let mut typed_fn_item = fn_item.clone();
        crate::passes::node_ids::assign_expr_ids(&mut typed_fn_item);
        let typeck_results = crate::passes::type_inference::infer_function(
            self,
            &typed_fn_item,
            generic_vars,
            initial_types,
        )?;
        let lowered_fn_item =
            crate::passes::typed_dispatch_lowering::lower_function(&typed_fn_item, &typeck_results);
        let previous_typeck_results = self.typeck_results.replace(Some(typeck_results));

        if std::env::var("CUTILE_DEBUG_COMPILER2").is_ok() {
            eprintln!(
                "compiler2: lowered entry function body:\n{}",
                quote::quote!(#lowered_fn_item)
            );
        }

        let return_value = self.compile_block(
            module,
            block_id,
            &lowered_fn_item.block,
            generic_vars,
            &mut ctx,
            None,
        );
        self.typeck_results.replace(previous_typeck_results);
        let return_value = return_value?;
        if return_value.is_some() {
            return self.jit_error_result(
                &fn_item.block.span(),
                "returning a value from this function is not supported",
            );
        }

        let entry_location = self.ir_location(&fn_item.sig.ident.span());
        // Debug info shows the name the user wrote; the `_entry` symbol
        // stays as the linkage name (the bytecode writer pairs the two).
        let di_name =
            KernelNaming::public_name_from_entry_name(&fn_name).unwrap_or_else(|| fn_name.clone());
        let mut entry_builder = OpBuilder::new(Opcode::Entry, entry_location)
            .attr("sym_name", Attribute::String(fn_name))
            .attr("di_name", Attribute::String(di_name))
            .attr("function_type", Attribute::Type(func_type))
            .region(region_id);

        // Forward optimization hints from the parsed hints.
        if let Some(hints_attr) = build_entry_optimization_hints(&self.optimization_hints) {
            entry_builder = entry_builder.attr("optimization_hints", hints_attr);
        }

        let (entry_id, _) = entry_builder.build(module);

        Ok(entry_id)
    }

    pub fn get_validator(&self) -> Validator {
        let mut validator = self.validator.clone();
        // Surface the launch checks hoisted during compilation so the host runs
        // them at launch.
        validator.launch_checks = self.launch_checks.borrow().clone();
        validator
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    // -----------------------------------------------------------------------
    // Helper methods ported from _function.rs
    // -----------------------------------------------------------------------

    pub fn compile_call_args(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Vec<TileRustValue>, JITError> {
        let mut result = vec![];
        for arg in args {
            let expected = if matches!(arg, syn::Expr::Lit(_) | syn::Expr::Unary(_)) {
                self.typeck_expr_tile_type(arg, generic_args, &HashMap::new())?
            } else {
                None
            };
            let value = self
                .compile_expression(module, block_id, arg, generic_args, ctx, expected)?
                .ok_or(self.jit_error(
                    &arg.span(),
                    &format!(
                        "Failed to compile argument: {:?}",
                        arg.to_token_stream().to_string()
                    ),
                ))?;
            result.push(value);
        }
        Ok(result)
    }

    pub(crate) fn compile_constant<T: Into<i64>>(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        generic_vars: &GenericVars,
        x: T,
    ) -> Result<TileRustValue, JITError> {
        let bounds = Bounds::exact(x.into());
        let rust_ty_str = type_name::<T>();
        let rust_ty = syn::parse2::<syn::Type>(rust_ty_str.parse()?).unwrap();
        let tr_ty = self
            .compile_type(&rust_ty, generic_vars, &HashMap::new())?
            .ok_or(self.jit_error(&rust_ty.span(), "failed to compile constant"))?;
        self.compile_constant_from_exact_bounds(module, block_id, bounds, tr_ty)
    }

    pub(crate) fn compile_bool_constant(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        generic_vars: &GenericVars,
        x: bool,
    ) -> Result<TileRustValue, JITError> {
        let rust_ty: syn::Type = syn::parse_quote!(bool);
        let tr_ty = self
            .compile_type(&rust_ty, generic_vars, &HashMap::new())?
            .ok_or(self.jit_error(&rust_ty.span(), "failed to compile bool constant"))?;
        self.compile_constant_from_exact_bounds(
            module,
            block_id,
            Bounds::exact(if x { 1 } else { 0 }),
            tr_ty,
        )
    }

    pub(crate) fn compile_constant_from_exact_bounds(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        bounds: Bounds<i64>,
        tr_ty: TileRustType,
    ) -> Result<TileRustValue, JITError> {
        if !bounds.is_exact() {
            return self.jit_error_result(
                &tr_ty.rust_ty.span(),
                &format!(
                    "expected a compile-time constant, but got a value with bounds [{}, {}]",
                    bounds.start, bounds.end
                ),
            );
        }
        let const_value = bounds.start;
        let TypeInstance::ElementType(type_inst) = &tr_ty.type_instance else {
            return self.jit_error_result(&tr_ty.rust_ty.span(), "expected a scalar element type");
        };
        let Some(const_ty_str) = get_cuda_tile_element_type_from_rust_primitive_str(
            &type_inst.rust_element_instance_ty,
            self.modules.primitives(),
        ) else {
            return self
                .jit_error_result(&tr_ty.rust_ty.span(), "failed to compile constant value");
        };

        // Build tile-ir Constant op directly (replaces operation_parse).
        let scalar = super::_type::scalar_from_name(&const_ty_str).ok_or_else(|| {
            JITError::Generic(format!(
                "unsupported scalar type for constant: {const_ty_str}"
            ))
        })?;
        let result_ty = Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(scalar),
        });
        let data = match scalar {
            ScalarType::I1 => vec![if const_value != 0 { 0xFF } else { 0x00 }],
            ScalarType::I4 => vec![(const_value as u8) & 0x0F],
            ScalarType::I8 => (const_value as i8).to_le_bytes().to_vec(),
            ScalarType::I16 => (const_value as i16).to_le_bytes().to_vec(),
            ScalarType::I32 => (const_value as i32).to_le_bytes().to_vec(),
            ScalarType::I64 => const_value.to_le_bytes().to_vec(),
            ScalarType::F16 => half::f16::from_f64(const_value as f64)
                .to_le_bytes()
                .to_vec(),
            ScalarType::BF16 => half::bf16::from_f64(const_value as f64)
                .to_le_bytes()
                .to_vec(),
            ScalarType::F32 => (const_value as f32).to_le_bytes().to_vec(),
            ScalarType::F64 => (const_value as f64).to_le_bytes().to_vec(),
            ScalarType::F8E4M3FN | ScalarType::F8E5M2 | ScalarType::F8E8M0FNU => {
                vec![const_value as u8]
            }
            ScalarType::F4E2M1FN => vec![(const_value as u8) & 0x0F],
            _ => (const_value as i32).to_le_bytes().to_vec(),
        };
        let (op_id, results) =
            OpBuilder::new(Opcode::Constant, self.ir_location(&tr_ty.rust_ty.span()))
                .result(result_ty.clone())
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: result_ty,
                        shape: vec![],
                        data,
                    }),
                )
                .build(module);
        append_op(module, block_id, op_id);
        let mut tr_val = TileRustValue::new_value_kind_like(results[0], tr_ty);
        tr_val.mutability = Mutability::Immutable;
        tr_val.bounds = Some(bounds);
        Ok(tr_val)
    }

    /// Return the Pass 2 side-table type for an expression.
    ///
    /// This path must not compile arguments or emit IR. A missing result means
    /// the type inference pass needs to learn the expression form.
    pub(crate) fn derive_type(
        &self,
        _module: &mut Module,
        _block_id: cutile_ir::ir::BlockId,
        expr: &syn::Expr,
        maybe_type_params: Option<Vec<crate::types::TypeParam>>,
        generic_vars: &GenericVars,
        _ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustType>, JITError> {
        let typeck_type_params = maybe_type_params
            .as_ref()
            .map(|type_params| {
                type_params
                    .iter()
                    .filter_map(|type_param| {
                        type_param
                            .name()
                            .map(|name| (name.to_string(), type_param.clone()))
                    })
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();

        self.typeck_expr_tile_type(expr, generic_vars, &typeck_type_params)
    }
}
