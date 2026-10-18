/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed API for Tile library descriptions (public preview `simt`, Tile IR
//! 13.5).
//!
//! A tile library pairs an *interface* (extern `preview$func` declarations
//! with tile-typed arguments) with an *implementation catalog*
//! (`preview$extern_impl` entries whose fragment arguments describe the
//! native SIMT ABI of a function in the code payload). This module builds
//! both as ordinary [`Module`]s, so [`write_bytecode`](crate::write_bytecode)
//! serializes them at 13.5 with the preview tag, and [`verify_library`]
//! checks the structural rules the upstream verifier enforces when the two
//! halves meet.
//!
//! ```text
//! ExternFunc { scale, [x: tile<128xf32> inout] }   ExternImpl { scale_impl implements scale,
//!   |                                                 [x: fragment<1xf32, local, (1):(1),
//!   v                                                               (128, 1, 1):(1, 0, 0)> inout] }
//! interface_module("lib", &[..]) --> .tilebc        implementation_module("lib", &[..]) --> .tilebc
//! ```

use crate::builder::{append_op, OpBuilder};
use crate::bytecode::Opcode;
use crate::ir::{
    Access, Attribute, BlockId, ConstParam, FragmentType, FuncType, Location, Module, OpId,
    ScratchRequest, SymExpr, Synchronization, TileElementType, TileType, Type, Value,
    ARG_NAMES_ATTR,
};
use crate::verify_target::{
    argument_accesses, injective_mode_size, mode_size, verify_const_arg_kind, verify_preview_op,
};
use crate::{Error, Result};

/// One argument of an extern declaration: a concrete tile and how the
/// callee uses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileArg {
    /// Argument name, printed as `%name` in the textual form.
    pub name: String,
    /// The concrete tile type at the call boundary.
    pub ty: TileType,
    /// How the callee uses the argument; `out` and `inout` arguments are
    /// also the declaration's results.
    pub access: Access,
}

impl TileArg {
    /// A declaration argument of the given tile type and access.
    pub fn new(name: impl Into<String>, ty: TileType, access: Access) -> Self {
        Self {
            name: name.into(),
            ty,
            access,
        }
    }
}

/// An extern `preview$func` declaration. Results are derived: one per
/// `out`/`inout` argument, in order (destination-passing style), which is
/// what the SIMT implementation path requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternFunc {
    /// The declaration symbol a `preview$call` names.
    pub name: String,
    /// Tile-typed arguments in declaration order; at least one is required.
    pub args: Vec<TileArg>,
    /// Constant parameters (`<[N: index]>`); empty for a concrete library.
    pub const_params: Vec<ConstParam>,
}

impl ExternFunc {
    /// A declaration without constant parameters.
    pub fn new(name: impl Into<String>, args: Vec<TileArg>) -> Self {
        Self {
            name: name.into(),
            args,
            const_params: Vec::new(),
        }
    }

    /// The tile types the declaration returns: its out/inout arguments.
    pub fn result_types(&self) -> Vec<Type> {
        self.args
            .iter()
            .filter(|a| a.access != Access::In)
            .map(|a| Type::Tile(a.ty.clone()))
            .collect()
    }
}

/// One argument of an implementation: the fragment the native function
/// receives for the matching declaration argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentArg {
    /// Argument name, printed as `%name` in the textual form.
    pub name: String,
    /// The fragment the native function receives: footprint, memory space,
    /// in-buffer order and the map onto the declaration's tile.
    pub fragment: FragmentType,
    /// Must equal the access of the matching declaration argument.
    pub access: Access,
}

impl FragmentArg {
    /// An implementation argument of the given fragment type and access.
    pub fn new(name: impl Into<String>, fragment: FragmentType, access: Access) -> Self {
        Self {
            name: name.into(),
            fragment,
            access,
        }
    }
}

/// A `preview$extern_impl` entry: the symbol the code payload exports and
/// how its arguments map onto the declaration's tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternImpl {
    /// The linked symbol (`.extern .func` in the consumer's PTX).
    pub symbol: String,
    /// The declaration this implements.
    pub implements: String,
    /// Fragment arguments, one per declaration argument, in order.
    pub args: Vec<FragmentArg>,
    /// The synchronization domain of one invocation. `None` means the
    /// implementation has no `synchronization` clause: nothing is attached
    /// to the op, written (payload bit 8 clear) or printed, as for an
    /// upstream module parsed without the clause.
    /// `Some(Synchronization::None)` is an explicit default clause: attached,
    /// written (bit 8 set, value 0) and printed as `synchronization<none>`,
    /// like the upstream scratch fixtures that write the clause out. Both
    /// mean no cooperative synchronization.
    pub synchronization: Option<Synchronization>,
    /// Shared-memory scratch the invocation needs; adds a trailing pointer
    /// to the native ABI.
    pub scratch: Option<ScratchRequest>,
    /// Target names such as `sm_120`; empty matches every target.
    pub arch: Vec<String>,
    /// `frontendABI` entries, printed-type key to target-type value.
    pub frontend_abi: Vec<(String, String)>,
    /// The implementation's own constant parameters, one per declaration
    /// parameter; a wildcard `?` leaves a slot unspecialized.
    pub const_params: Vec<ConstParam>,
    /// The bindings of the declaration's constant parameters; a named
    /// parameter must be forwarded as itself (`<[N]>`), see
    /// [`verify_library`].
    pub const_args: Vec<SymExpr>,
}

impl ExternImpl {
    /// A concrete implementation without a synchronization clause, scratch,
    /// arch restriction or ABI overrides.
    pub fn new(
        symbol: impl Into<String>,
        implements: impl Into<String>,
        args: Vec<FragmentArg>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            implements: implements.into(),
            args,
            synchronization: None,
            scratch: None,
            arch: Vec::new(),
            frontend_abi: Vec::new(),
            const_params: Vec::new(),
            const_args: Vec::new(),
        }
    }
}

/// An interface module: only extern declarations.
pub fn interface_module(name: &str, funcs: &[ExternFunc]) -> Module {
    let mut module = Module::new(name);
    for f in funcs {
        add_extern_func(&mut module, f);
    }
    module
}

/// An implementation catalog module: only implementations.
pub fn implementation_module(name: &str, impls: &[ExternImpl]) -> Module {
    let mut module = Module::new(name);
    for i in impls {
        add_extern_impl(&mut module, i);
    }
    module
}

fn access_dicts<'a>(accesses: impl Iterator<Item = &'a Access>) -> Attribute {
    Attribute::Array(accesses.map(|a| Attribute::access_dict(*a)).collect())
}

fn arg_names<'a>(names: impl Iterator<Item = &'a str>) -> Attribute {
    Attribute::Array(names.map(|n| Attribute::String(n.to_string())).collect())
}

/// Append an extern `preview$func` declaration to `module.functions`.
pub fn add_extern_func(module: &mut Module, f: &ExternFunc) -> OpId {
    let function_type = Type::Func(FuncType {
        inputs: f.args.iter().map(|a| Type::Tile(a.ty.clone())).collect(),
        results: f.result_types(),
    });
    let (op, _) = OpBuilder::new(Opcode::PreviewFunc, Location::Unknown)
        .attr("sym_name", Attribute::String(f.name.clone()))
        .attr("is_extern", Attribute::Bool(true))
        .attr("function_type", Attribute::Type(function_type))
        .attr("arg_attrs", access_dicts(f.args.iter().map(|a| &a.access)))
        .attr(
            "const_params",
            Attribute::Array(
                f.const_params
                    .iter()
                    .cloned()
                    .map(Attribute::ConstParam)
                    .collect(),
            ),
        )
        .attr(
            ARG_NAMES_ATTR,
            arg_names(f.args.iter().map(|a| a.name.as_str())),
        )
        .build(module);
    module.functions.push(op);
    op
}

/// Append a `preview$extern_impl` to `module.functions`. Optional clauses
/// are attached only when set, so the payload flags reflect what was asked
/// for: `synchronization: None` attaches no clause (bit 8 clear, nothing
/// printed) and `Some(Synchronization::None)` an explicit default clause
/// (bit 8 set, value 0, printed), as upstream does for a parsed clause.
pub fn add_extern_impl(module: &mut Module, i: &ExternImpl) -> OpId {
    let function_type = Type::Func(FuncType {
        inputs: i
            .args
            .iter()
            .map(|a| Type::Fragment(a.fragment.clone()))
            .collect(),
        results: Vec::new(),
    });
    let mut builder = OpBuilder::new(Opcode::PreviewExternImpl, Location::Unknown)
        .attr("sym_name", Attribute::String(i.symbol.clone()))
        .attr("implements", Attribute::String(i.implements.clone()));
    if !i.const_params.is_empty() {
        builder = builder.attr(
            "const_params",
            Attribute::Array(
                i.const_params
                    .iter()
                    .cloned()
                    .map(Attribute::ConstParam)
                    .collect(),
            ),
        );
    }
    if !i.const_args.is_empty() {
        builder = builder.attr(
            "const_args",
            Attribute::Array(
                i.const_args
                    .iter()
                    .cloned()
                    .map(Attribute::SymExpr)
                    .collect(),
            ),
        );
    }
    builder = builder
        .attr("function_type", Attribute::Type(function_type))
        .attr("arg_attrs", access_dicts(i.args.iter().map(|a| &a.access)));
    if let Some(scratch) = &i.scratch {
        builder = builder.attr("scratch_smem", Attribute::ScratchRequest(scratch.clone()));
    }
    if !i.arch.is_empty() {
        builder = builder.attr(
            "arch",
            Attribute::Array(i.arch.iter().cloned().map(Attribute::String).collect()),
        );
    }
    if !i.frontend_abi.is_empty() {
        builder = builder.attr(
            "frontendABI",
            Attribute::Dictionary(
                i.frontend_abi
                    .iter()
                    .map(|(k, v)| (k.clone(), Attribute::String(v.clone())))
                    .collect(),
            ),
        );
    }
    if let Some(synchronization) = i.synchronization {
        builder = builder.attr(
            "synchronization",
            Attribute::Synchronization(synchronization),
        );
    }
    let (op, _) = builder
        .attr(
            ARG_NAMES_ATTR,
            arg_names(i.args.iter().map(|a| a.name.as_str())),
        )
        .build(module);
    module.functions.push(op);
    op
}

/// Append a `preview$call` to `block`. `callee` is the declaration symbol;
/// the consumer resolves the implementation. Returns the op and its result
/// values.
pub fn preview_call(
    module: &mut Module,
    block: BlockId,
    callee: &str,
    const_args: Vec<SymExpr>,
    operands: &[Value],
    result_types: &[Type],
) -> (OpId, Vec<Value>) {
    let (op, results) = OpBuilder::new(Opcode::PreviewCall, Location::Unknown)
        .attr("callee", Attribute::String(callee.to_string()))
        .attr(
            "const_args",
            Attribute::Array(const_args.into_iter().map(Attribute::SymExpr).collect()),
        )
        .operands(operands.iter().copied())
        .results(result_types.iter().cloned())
        .build(module);
    append_op(module, block, op);
    (op, results)
}

/// Check a library's declarations and implementations before any bytecode
/// is written.
///
/// Each half is built as the module the writer would serialize and run
/// through the op-level verifier, which holds the rules of `PreviewOps.cpp`
/// and this crate's narrowings (a rank-0 `mem_layout` only for a by-value
/// `in` scalar in local memory, results equal to the out/inout tiles and
/// nothing else, and the contiguous out/inout run on declarations as well
/// as on implementations). This function adds only the rules that need
/// both halves:
///
/// - every declaration name is unique, and every implementation names a
///   declaration of the library;
/// - an implementation has the declaration's argument count, and each of
///   its arguments has the declaration argument's access and element type;
/// - an implementation's constant arguments bind the declaration's
///   parameters by kind, and a named implementation parameter is forwarded
///   as itself (SIMT implementations are monomorphized with the
///   declaration's bindings, so `<[N: index]> implements @f<[N]>` is the
///   only form);
/// - every concrete fragmap covers the declaration's tile with no gap and
///   no overlap: distinct invocations times distinct participants times the
///   footprint equals the tile's element count.
pub fn verify_library(funcs: &[ExternFunc], impls: &[ExternImpl]) -> Result<()> {
    let fail = |what: &str, why: String| Error::Verification(format!("{what}: {why}"));
    let decl_name = |f: &ExternFunc| format!("preview$func @{}", f.name);
    let impl_name = |i: &ExternImpl| format!("preview$extern_impl @{}", i.symbol);

    // The op-level rules, on the modules the writer would serialize.
    let interface = interface_module("library", funcs);
    for (f, &op) in funcs.iter().zip(&interface.functions) {
        verify_preview_op(&interface, interface.op(op)).map_err(|e| fail(&decl_name(f), e))?;
    }
    let catalog = implementation_module("library", impls);
    for (i, &op) in impls.iter().zip(&catalog.functions) {
        verify_preview_op(&catalog, catalog.op(op)).map_err(|e| fail(&impl_name(i), e))?;
    }

    let mut names = std::collections::HashSet::new();
    for f in funcs {
        if !names.insert(f.name.as_str()) {
            return Err(fail(&decl_name(f), "duplicate declaration".into()));
        }
    }
    for i in impls {
        let what = impl_name(i);
        let decl = funcs
            .iter()
            .find(|f| f.name == i.implements)
            .ok_or_else(|| {
                fail(
                    &what,
                    format!("implements unknown declaration @{}", i.implements),
                )
            })?;
        if i.args.len() != decl.args.len() {
            return Err(fail(
                &what,
                format!(
                    "input count ({}) must match extern function input count ({})",
                    i.args.len(),
                    decl.args.len()
                ),
            ));
        }
        if i.const_args.len() != decl.const_params.len() {
            return Err(fail(
                &what,
                format!(
                    "constant argument count ({}) must match extern function constant parameter \
                     count ({})",
                    i.const_args.len(),
                    decl.const_params.len()
                ),
            ));
        }
        // `verifyImplConstArgKinds`: with bindings given, the implementation
        // declares one parameter per declaration parameter, each argument
        // has the declared kind, and a named parameter is the identity
        // binding of the declaration parameter with the same name.
        if !i.const_args.is_empty() {
            if i.const_params.len() != decl.const_params.len() {
                return Err(fail(
                    &what,
                    format!(
                        "constant parameter count ({}) must match extern function constant \
                         parameter count ({}) when constant arguments are given",
                        i.const_params.len(),
                        decl.const_params.len()
                    ),
                ));
            }
            for (j, (arg, decl_param)) in i.const_args.iter().zip(&decl.const_params).enumerate() {
                verify_const_arg_kind(j, decl_param, arg).map_err(|e| fail(&what, e))?;
                let param = &i.const_params[j];
                if param.is_wildcard() {
                    continue;
                }
                if param.name != decl_param.name {
                    return Err(fail(
                        &what,
                        format!(
                            "const_params[{j}] name '{}' must match the implemented function's \
                             constant parameter name '{}'",
                            param.name, decl_param.name
                        ),
                    ));
                }
                if !matches!(arg, SymExpr::Var(name) if *name == param.name) {
                    return Err(fail(
                        &what,
                        format!(
                            "const_args[{j}] must be an identity reference to the \
                             implementation's own parameter '{}'; SIMT implementations are \
                             monomorphized with the declaration's bindings",
                            param.name
                        ),
                    ));
                }
            }
        }
        for (j, (arg, decl_arg)) in i.args.iter().zip(&decl.args).enumerate() {
            if arg.access != decl_arg.access {
                return Err(fail(
                    &what,
                    format!(
                        "argument {j} access ({}) must match extern function argument access ({})",
                        arg.access.name(),
                        decl_arg.access.name()
                    ),
                ));
            }
            let tile_element = match &decl_arg.ty.element_type {
                TileElementType::Scalar(s) => *s,
                TileElementType::Pointer(_) => {
                    return Err(fail(
                        &what,
                        format!("argument {j}: pointer tile elements are not supported"),
                    ))
                }
            };
            let fragment = &arg.fragment;
            if fragment.element_type != tile_element {
                return Err(fail(
                    &what,
                    format!(
                        "argument {j} fragment element type {:?} must match the declaration's {:?}",
                        fragment.element_type, tile_element
                    ),
                ));
            }
            // The map must tile the declaration's operand with nothing left
            // over. Symbolic fragmaps are checked by the consumer after
            // concretization; the op-level verifier has already required
            // three modes with representable products.
            if !fragment.fragmap.is_concrete() {
                continue;
            }
            let shapes = &fragment.fragmap.shapes;
            let strides = &fragment.fragmap.strides;
            let invocations = injective_mode_size(&shapes[0], &strides[0]);
            let participants = injective_mode_size(&shapes[1], &strides[1]);
            let footprint = mode_size(&shapes[2]);
            let covered = invocations
                .zip(participants)
                .zip(footprint)
                .and_then(|((inv, part), foot)| inv.checked_mul(part)?.checked_mul(foot));
            let tile_elements: i64 = decl_arg.ty.shape.iter().product();
            if covered != Some(tile_elements) {
                return Err(fail(
                    &what,
                    format!(
                        "argument {j} fragmap covers {} distinct invocation(s) * {} distinct \
                         participant(s) * {} footprint element(s), but the extern function \
                         tile has {tile_elements} element(s)",
                        invocations.unwrap_or(0),
                        participants.unwrap_or(0),
                        footprint.unwrap_or(0)
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The access of argument `index` recorded on a function-like op, for
/// callers inspecting a module built with this API.
pub fn argument_access(module: &Module, func: OpId, index: usize) -> Option<Access> {
    let op = module.op(func);
    let num_args = match op.attributes.iter().find(|(k, _)| k == "function_type") {
        Some((_, Attribute::Type(Type::Func(f)))) => f.inputs.len(),
        _ => return None,
    };
    let arg_attrs = op
        .attributes
        .iter()
        .find(|(k, _)| k == "arg_attrs")
        .map(|(_, v)| v);
    argument_accesses(arg_attrs, num_args)
        .ok()?
        .get(index)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Layout, MemorySpace, ScalarType};

    fn tile(n: i64) -> TileType {
        TileType {
            shape: vec![n],
            element_type: TileElementType::Scalar(ScalarType::F32),
        }
    }

    fn scale() -> ExternFunc {
        ExternFunc::new("scale", vec![TileArg::new("x", tile(128), Access::InOut)])
    }

    fn scale_impl(fragmap: Layout) -> ExternImpl {
        ExternImpl::new(
            "scale_impl",
            "scale",
            vec![FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![1],
                    ScalarType::F32,
                    MemorySpace::Local,
                    Layout::contiguous_mem(1),
                    fragmap,
                ),
                Access::InOut,
            )],
        )
    }

    #[test]
    fn elementwise_library_verifies() {
        verify_library(&[scale()], &[scale_impl(Layout::elementwise(128))]).unwrap();
        let mut warp = scale_impl(Layout::warp_lanes(128));
        warp.synchronization = Some(Synchronization::Warp);
        verify_library(&[scale()], &[warp]).unwrap();
    }

    #[test]
    fn coverage_access_and_synchronization_mismatches_are_reported() {
        let short = scale_impl(Layout::elementwise(64));
        assert!(verify_library(&[scale()], &[short])
            .unwrap_err()
            .to_string()
            .contains("covers"));
        let mut access = scale_impl(Layout::elementwise(128));
        access.args[0].access = Access::In;
        assert!(verify_library(&[scale()], &[access])
            .unwrap_err()
            .to_string()
            .contains("access"));
        let mut sync = scale_impl(Layout::elementwise(128));
        sync.synchronization = Some(Synchronization::Warp);
        assert!(verify_library(&[scale()], &[sync])
            .unwrap_err()
            .to_string()
            .contains("multiple of 32"));
        assert!(verify_library(&[], &[scale_impl(Layout::elementwise(128))])
            .unwrap_err()
            .to_string()
            .contains("unknown declaration"));
    }

    #[test]
    fn out_arguments_must_be_contiguous() {
        let f = ExternFunc::new(
            "f",
            vec![
                TileArg::new("a", tile(128), Access::InOut),
                TileArg::new("b", tile(128), Access::In),
                TileArg::new("c", tile(128), Access::Out),
            ],
        );
        let error = verify_library(&[f], &[]).unwrap_err().to_string();
        assert!(
            error.contains("contiguous") && error.contains("narrowing"),
            "{error}"
        );
    }

    #[test]
    fn declaration_results_follow_the_bound_outputs() {
        let f = ExternFunc::new(
            "f",
            vec![
                TileArg::new("a", tile(128), Access::In),
                TileArg::new("b", tile(64), Access::Out),
                TileArg::new("c", tile(32), Access::InOut),
            ],
        );
        assert_eq!(
            f.result_types(),
            vec![Type::Tile(tile(64)), Type::Tile(tile(32))]
        );
        let mut module = Module::new("lib");
        let op = add_extern_func(&mut module, &f);
        assert_eq!(argument_access(&module, op, 1), Some(Access::Out));
    }
}
