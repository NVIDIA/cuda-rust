/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The structural rules the Tile compiler's verifier applies when a library's
//! two halves meet, checked here before any bytecode is written so a mistake
//! is reported in terms of the description rather than of a deserialized
//! module.

use std::collections::HashSet;

use crate::Error;
use crate::model::{Access, Declaration, FragmentArg, Implementation, Library, MemorySpace};

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
    /// - every implementation names a declaration, with one fragment per
    ///   argument, the same access and element type;
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
        if arg.ty.shape[0] <= 0 {
            return Err(at(format!(
                "argument `{}` has a non-positive extent",
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
    let participants = imp.synchronization.map_or(1, |sync| sync.participants());
    for (arg, decl_arg) in imp.args.iter().zip(&decl.args) {
        verify_fragment(arg, decl_arg.ty.size(), decl_arg.ty.element, participants)
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

fn verify_fragment(
    arg: &FragmentArg,
    tile_size: i64,
    element: crate::model::ScalarType,
    participants: i64,
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
    if fragmap.shapes[2] != footprint {
        return Err(format!(
            "fragmap footprint {} differs from the fragment's {} elements",
            fragmap.shapes[2], footprint
        ));
    }
    if fragmap.size() != tile_size {
        return Err(format!(
            "fragmap covers {} elements of a {tile_size}-element tile",
            fragmap.size()
        ));
    }
    if participants != 0 && fragmap.shapes[1] != participants {
        return Err(format!(
            "fragmap has {} participants but the synchronization clause promises {participants}",
            fragmap.shapes[1]
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
    } else if mem.size() != footprint {
        return Err(format!(
            "mem_layout spans {} elements but the footprint is {footprint}",
            mem.size()
        ));
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
}
