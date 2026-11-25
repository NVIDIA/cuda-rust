/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The structural rules the Tile compiler's verifier applies when a library's
//! two halves meet, checked here before any bytecode is written so a mistake
//! is reported in terms of the description rather than of a deserialized
//! module.
//!
//! A parametric library is checked twice over: the symbolic rules (every
//! parameter named is declared, no wildcard on a declaration, an
//! implementation forwards the declaration's parameters by name), and the
//! coverage rules at a probe instantiation, every integer parameter bound to
//! `64 * footprint`, which is a tile size every layout this crate builds
//! divides evenly.

use std::collections::{HashMap, HashSet};

use crate::Error;
use crate::model::{
    Access, ConstParam, Declaration, Expr, FragmentArg, Implementation, Library, MemorySpace,
};

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

impl Library {
    /// Check the library against the rules below; the first violation is
    /// the error.
    ///
    /// - names are identifiers, unique among declarations and among symbols;
    /// - a declaration has at least one argument, one dimension per tile,
    ///   and its written arguments form one contiguous run (the native
    ///   implementation path needs destination-passing results in order);
    /// - a declaration's extents are literals or expressions without
    ///   division over its own constant parameters, which cannot be `?`;
    /// - every implementation names a declaration, with one fragment per
    ///   argument, the same access and element type, one constant parameter
    ///   per declaration parameter (the same name, forwarded as itself, or
    ///   `?` bound by an expression), and layouts that name only its own
    ///   parameters;
    /// - a fragment covers the tile: invocations times participants times
    ///   footprint equals the tile size, the footprint equals the fragment's
    ///   shape, a rank-0 buffer layout only for a read-only local fragment;
    /// - the participants of every fragmap equal what the synchronization
    ///   clause promises (1 without a clause, 32 for a warp).
    pub fn verify(&self) -> Result<(), Error> {
        if !is_identifier(&self.name) {
            return Err(Error::Verify(format!(
                "module name `{}` is not an identifier",
                self.name
            )));
        }
        let mut names = HashSet::new();
        for decl in &self.declarations {
            verify_declaration(decl)?;
            if !names.insert(decl.name.as_str()) {
                return Err(Error::Verify(format!(
                    "declaration @{} appears twice",
                    decl.name
                )));
            }
        }
        let mut symbols = HashSet::new();
        for imp in &self.implementations {
            if !symbols.insert(imp.symbol.as_str()) {
                return Err(Error::Verify(format!(
                    "implementation @{} appears twice",
                    imp.symbol
                )));
            }
            let decl = self
                .declarations
                .iter()
                .find(|decl| decl.name == imp.implements)
                .ok_or_else(|| {
                    Error::Verify(format!(
                        "implementation @{} implements @{}, which the interface does not declare",
                        imp.symbol, imp.implements
                    ))
                })?;
            verify_implementation(imp, decl)?;
        }
        Ok(())
    }
}

fn verify_declaration(decl: &Declaration) -> Result<(), Error> {
    let at = |message: String| Error::Verify(format!("declaration @{}: {message}", decl.name));
    if !is_identifier(&decl.name) {
        return Err(Error::Verify(format!(
            "declaration name `{}` is not an identifier",
            decl.name
        )));
    }
    let mut params = HashSet::new();
    for param in &decl.const_params {
        match param {
            ConstParam::Wildcard => {
                return Err(at("a declaration cannot have a `?` parameter".into()));
            }
            ConstParam::Index(name) => {
                if !is_identifier(name) {
                    return Err(at(format!("parameter name `{name}` is not an identifier")));
                }
                if !params.insert(name.as_str()) {
                    return Err(at(format!("parameter `{name}` appears twice")));
                }
            }
        }
    }
    if decl.args.is_empty() {
        return Err(at("a declaration needs at least one argument".into()));
    }
    let mut seen = HashSet::new();
    for arg in &decl.args {
        if !is_identifier(&arg.name) {
            return Err(at(format!(
                "argument name `{}` is not an identifier",
                arg.name
            )));
        }
        if !seen.insert(arg.name.as_str()) {
            return Err(at(format!("argument `{}` appears twice", arg.name)));
        }
        if arg.ty.shape.len() != 1 {
            return Err(at(format!(
                "argument `{}` has {} dimensions; this writer emits one-dimensional tiles",
                arg.name,
                arg.ty.shape.len()
            )));
        }
        let extent = &arg.ty.shape[0];
        if let Some(value) = extent.as_int()
            && value <= 0
        {
            return Err(at(format!(
                "argument `{}` has a non-positive extent",
                arg.name
            )));
        }
        if !extent.is_dimension() {
            return Err(at(format!(
                "argument `{}` has the extent `{extent}`; a tile extent cannot divide",
                arg.name
            )));
        }
        let mut vars = Vec::new();
        extent.vars(&mut vars);
        if let Some(unknown) = vars.iter().find(|name| !params.contains(name.as_str())) {
            return Err(at(format!(
                "argument `{}` names `{unknown}`, which is not a parameter of the declaration",
                arg.name
            )));
        }
    }
    let written: Vec<usize> = decl
        .args
        .iter()
        .enumerate()
        .filter(|(_, arg)| arg.access.is_written())
        .map(|(i, _)| i)
        .collect();
    if written.is_empty() {
        return Err(at(
            "no argument is written; a declaration's results are its out and inout arguments"
                .into(),
        ));
    }
    if written.last().unwrap() - written[0] + 1 != written.len() {
        return Err(at(format!(
            "written arguments must form one contiguous run; `{}` sits between them",
            decl.args[written[0] + 1..*written.last().unwrap()]
                .iter()
                .find(|arg| !arg.access.is_written())
                .map(|arg| arg.name.as_str())
                .unwrap_or("?")
        )));
    }
    Ok(())
}

fn verify_implementation(imp: &Implementation, decl: &Declaration) -> Result<(), Error> {
    let at = |message: String| Error::Verify(format!("implementation @{}: {message}", imp.symbol));
    if !is_identifier(&imp.symbol) {
        return Err(Error::Verify(format!(
            "implementation symbol `{}` is not an identifier",
            imp.symbol
        )));
    }
    if imp.args.len() != decl.args.len() {
        return Err(at(format!(
            "{} fragment arguments for the {} arguments of @{}",
            imp.args.len(),
            decl.args.len(),
            decl.name
        )));
    }
    let bindings = verify_parameters(imp, decl).map_err(at)?;
    let participants = imp.synchronization.map_or(1, |sync| sync.participants());
    for (arg, decl_arg) in imp.args.iter().zip(&decl.args) {
        let tile_size = eval(&decl_arg.ty.shape[0], &bindings)
            .map_err(|message| at(format!("argument `{}`: {message}", arg.name)))?;
        verify_fragment(arg, tile_size, decl_arg.ty.element, participants, &bindings)
            .map_err(|message| at(format!("argument `{}`: {message}", arg.name)))?;
        if arg.access != decl_arg.access {
            return Err(at(format!(
                "argument `{}` is `{}` but the declaration says `{}`",
                arg.name,
                arg.access.name(),
                decl_arg.access.name()
            )));
        }
    }
    for name in &imp.arch {
        if name.is_empty() {
            return Err(at("an empty arch name".into()));
        }
    }
    Ok(())
}

/// The implementation's parameters against the declaration's, and the
/// probe binding every parameter gets for the coverage checks: a literal
/// argument's value for a wildcard slot, `64 * footprint` otherwise.
fn verify_parameters(
    imp: &Implementation,
    decl: &Declaration,
) -> Result<HashMap<String, i64>, String> {
    if imp.const_params.len() != decl.const_params.len()
        || imp.const_args.len() != decl.const_params.len()
    {
        return Err(format!(
            "{} constant parameters and {} arguments for the {} parameters of @{}",
            imp.const_params.len(),
            imp.const_args.len(),
            decl.const_params.len(),
            decl.name
        ));
    }
    let footprint = imp
        .args
        .iter()
        .map(|arg| arg.fragment.footprint())
        .max()
        .unwrap_or(1)
        .max(1);
    let probe = 64 * footprint;
    let mut bindings = HashMap::new();
    let mut named = HashSet::new();
    for ((param, arg), decl_param) in imp
        .const_params
        .iter()
        .zip(&imp.const_args)
        .zip(&decl.const_params)
    {
        let decl_name = decl_param.name();
        match param {
            ConstParam::Index(name) => {
                if !named.insert(name.as_str()) {
                    return Err(format!("parameter `{name}` appears twice"));
                }
                if name != decl_name {
                    return Err(format!(
                        "parameter `{name}` must be named `{decl_name}` like the declaration's"
                    ));
                }
                if *arg != Expr::Var(name.clone()) {
                    return Err(format!(
                        "parameter `{name}` must be forwarded as itself, found `{arg}`"
                    ));
                }
                bindings.insert(name.clone(), probe);
            }
            ConstParam::Wildcard => {
                let mut vars = Vec::new();
                arg.vars(&mut vars);
                if !vars.is_empty() {
                    return Err(format!(
                        "the argument `{arg}` for `?` names `{}`; a wildcard slot takes a literal",
                        vars[0]
                    ));
                }
                let value = eval(arg, &HashMap::new())
                    .map_err(|message| format!("the argument for `{decl_name}`: {message}"))?;
                bindings.insert(decl_name.to_string(), value);
            }
        }
    }
    for arg in &imp.args {
        let mut vars = Vec::new();
        arg.fragment.vars(&mut vars);
        if let Some(unknown) = vars.iter().find(|name| !named.contains(name.as_str())) {
            return Err(format!(
                "argument `{}` names `{unknown}`, which is not a parameter of the implementation",
                arg.name
            ));
        }
    }
    Ok(bindings)
}

fn eval(expr: &Expr, bindings: &HashMap<String, i64>) -> Result<i64, String> {
    expr.eval(&|name| bindings.get(name).copied())
}

fn verify_fragment(
    arg: &FragmentArg,
    tile_size: i64,
    element: crate::model::ScalarType,
    participants: i64,
    bindings: &HashMap<String, i64>,
) -> Result<(), String> {
    let fragment = &arg.fragment;
    if fragment.element != element {
        return Err(format!(
            "fragment element type {} differs from the tile element type {}",
            fragment.element.name(),
            element.name()
        ));
    }
    if fragment.shape.is_empty() || fragment.shape.iter().any(|d| *d <= 0) {
        return Err("fragment shape must have positive extents".into());
    }
    let footprint = fragment.footprint();
    let fragmap = &fragment.fragmap;
    if fragmap.rank() != 3 || fragmap.strides.len() != 3 {
        return Err(format!(
            "fragmap must have three modes (invocations, participants, footprint), found {}",
            fragmap.rank()
        ));
    }
    let probed = if bindings.is_empty() {
        String::new()
    } else {
        let mut pairs: Vec<String> = bindings.iter().map(|(k, v)| format!("{k} = {v}")).collect();
        pairs.sort();
        format!(" (at {})", pairs.join(", "))
    };
    let shapes: Vec<i64> = fragmap
        .shapes
        .iter()
        .map(|mode| eval(mode, bindings))
        .collect::<Result<_, _>>()?;
    if shapes[2] != footprint {
        return Err(format!(
            "fragmap footprint {} differs from the fragment's {} elements{probed}",
            shapes[2], footprint
        ));
    }
    let covered: i64 = shapes.iter().product();
    if covered != tile_size {
        return Err(format!(
            "fragmap covers {covered} elements of a {tile_size}-element tile{probed}"
        ));
    }
    if participants != 0 && shapes[1] != participants {
        return Err(format!(
            "fragmap has {} participants but the synchronization clause promises {participants}",
            shapes[1]
        ));
    }
    let mem = &fragment.mem_layout;
    if mem.shapes.len() != mem.strides.len() {
        return Err("mem_layout shapes and strides differ in length".into());
    }
    if mem.rank() == 0 {
        if footprint != 1 {
            return Err("a rank-0 mem_layout needs a one-element footprint".into());
        }
        if arg.access != Access::In || fragment.memory_space != MemorySpace::Local {
            return Err(
                "a rank-0 mem_layout (passed by value) is only valid for a read-only local fragment"
                    .into(),
            );
        }
    } else {
        let spanned: i64 = mem
            .shapes
            .iter()
            .map(|mode| eval(mode, bindings))
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .product();
        if spanned != footprint {
            return Err(format!(
                "mem_layout spans {spanned} elements but the footprint is {footprint}{probed}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model::*;

    fn tile() -> TileType {
        TileType::new(vec![128], ScalarType::F32)
    }

    fn scalar_fragment() -> FragmentType {
        FragmentType::new(
            vec![1],
            ScalarType::F32,
            MemorySpace::Local,
            Layout::scalar_mem(),
            Layout::elementwise(128),
        )
    }

    fn pointer_fragment() -> FragmentType {
        FragmentType::new(
            vec![1],
            ScalarType::F32,
            MemorySpace::Local,
            Layout::contiguous_mem(1),
            Layout::elementwise(128),
        )
    }

    fn relu() -> Library {
        Library::new(
            "lib",
            vec![Declaration::new(
                "relu",
                vec![
                    TileArg::new("x", tile(), Access::In),
                    TileArg::new("relu_out", tile(), Access::Out),
                ],
            )],
            vec![Implementation::new(
                "relu_impl",
                "relu",
                vec![
                    FragmentArg::new("x", scalar_fragment(), Access::In),
                    FragmentArg::new("relu_out", pointer_fragment(), Access::Out),
                ],
            )],
        )
    }

    /// `@relu<[N: index]>` with `@relu_impl<[N: index]> implements @relu<[N]>`.
    fn parametric_relu(fragmap: Layout, sync: Option<Synchronization>) -> Library {
        let n = Expr::var("N");
        let tile = TileType::symbolic(vec![n], ScalarType::F32);
        let fragment = |mem: Layout| {
            FragmentType::new(
                vec![1],
                ScalarType::F32,
                MemorySpace::Local,
                mem,
                fragmap.clone(),
            )
        };
        let mut imp = Implementation::parametric(
            "relu_impl",
            "relu",
            vec!["N".to_string()],
            vec![
                FragmentArg::new("x", fragment(Layout::scalar_mem()), Access::In),
                FragmentArg::new("relu_out", fragment(Layout::contiguous_mem(1)), Access::Out),
            ],
        );
        imp.synchronization = sync;
        Library::new(
            "lib",
            vec![Declaration::parametric(
                "relu",
                vec![ConstParam::index("N")],
                vec![
                    TileArg::new("x", tile.clone(), Access::In),
                    TileArg::new("relu_out", tile, Access::Out),
                ],
            )],
            vec![imp],
        )
    }

    #[test]
    fn the_relu_library_verifies() {
        relu().verify().unwrap();
    }

    #[test]
    fn written_arguments_must_be_contiguous() {
        let mut lib = relu();
        lib.declarations[0]
            .args
            .push(TileArg::new("y", tile(), Access::In));
        lib.declarations[0]
            .args
            .push(TileArg::new("z", tile(), Access::Out));
        let error = lib.verify().unwrap_err().to_string();
        assert!(
            error.contains("contiguous run") && error.contains("`y`"),
            "{error}"
        );
    }

    #[test]
    fn an_implementation_needs_its_declaration() {
        let mut lib = relu();
        lib.implementations[0].implements = "gelu".into();
        let error = lib.verify().unwrap_err().to_string();
        assert!(error.contains("does not declare"), "{error}");
    }

    #[test]
    fn access_and_coverage_are_checked() {
        let mut lib = relu();
        lib.implementations[0].args[1].access = Access::InOut;
        assert!(
            lib.verify()
                .unwrap_err()
                .to_string()
                .contains("`relu_out` is `inout`")
        );

        let mut lib = relu();
        lib.implementations[0].args[1].fragment.fragmap = Layout::elementwise(64);
        assert!(
            lib.verify()
                .unwrap_err()
                .to_string()
                .contains("covers 64 elements")
        );

        let mut lib = relu();
        lib.implementations[0].synchronization = Some(Synchronization::Warp);
        let error = lib.verify().unwrap_err().to_string();
        assert!(
            error.contains("1 participants") && error.contains("promises 32"),
            "{error}"
        );
        lib.implementations[0].args[0].fragment.fragmap = Layout::warp_lanes(128);
        lib.implementations[0].args[1].fragment.fragmap = Layout::warp_lanes(128);
        lib.verify().unwrap();
    }

    #[test]
    fn by_value_is_only_for_read_only_local_scalars() {
        let mut lib = relu();
        lib.implementations[0].args[1].fragment.mem_layout = Layout::scalar_mem();
        let error = lib.verify().unwrap_err().to_string();
        assert!(error.contains("rank-0 mem_layout"), "{error}");
    }

    #[test]
    fn parametric_libraries_verify_at_a_probe() {
        let n = Expr::var("N");
        parametric_relu(Layout::elementwise(n.clone()), None)
            .verify()
            .unwrap();
        parametric_relu(Layout::warp_lanes(n.clone()), Some(Synchronization::Warp))
            .verify()
            .unwrap();

        // A fragmap that does not cover the tile at the probe instantiation.
        let error = parametric_relu(Layout::elementwise(n.clone().floor_div(2)), None)
            .verify()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("covers 32 elements of a 64-element tile (at N = 64)"),
            "{error}"
        );

        // A parameter the implementation never declared.
        let mut lib = parametric_relu(Layout::elementwise(Expr::var("M")), None);
        lib.implementations[0].const_params = vec![ConstParam::index("N")];
        let error = lib.verify().unwrap_err().to_string();
        assert!(error.contains("names `M`"), "{error}");

        // The forwarding must be the identity.
        let mut lib = parametric_relu(Layout::elementwise(n.clone()), None);
        lib.implementations[0].const_args = vec![Expr::Int(128)];
        let error = lib.verify().unwrap_err().to_string();
        assert!(error.contains("forwarded as itself"), "{error}");

        // A wildcard slot bound to a literal is checked at that literal.
        let mut lib = parametric_relu(Layout::elementwise(256), None);
        lib.implementations[0].const_params = vec![ConstParam::Wildcard];
        lib.implementations[0].const_args = vec![Expr::Int(256)];
        lib.verify().unwrap();
        lib.implementations[0].const_args = vec![Expr::Int(128)];
        let error = lib.verify().unwrap_err().to_string();
        assert!(
            error.contains("covers 256 elements of a 128-element tile"),
            "{error}"
        );

        // Declarations reject wildcards and division.
        let mut lib = parametric_relu(Layout::elementwise(n.clone()), None);
        lib.declarations[0].const_params.push(ConstParam::Wildcard);
        assert!(
            lib.verify()
                .unwrap_err()
                .to_string()
                .contains("`?` parameter")
        );
        let mut lib = parametric_relu(Layout::elementwise(n.clone()), None);
        lib.declarations[0].args[0].ty = TileType::symbolic(vec![n.floor_div(2)], ScalarType::F32);
        assert!(
            lib.verify()
                .unwrap_err()
                .to_string()
                .contains("cannot divide")
        );
    }
}
