/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The readable form of a library: the Tile IR textual syntax of both
//! modules, as the reference tools print it. It is written next to the
//! bytecode so a reader can see what the assembler will read.

use std::fmt::Write as _;

use crate::model::{Declaration, FragmentType, Implementation, Layout, Library, TileType};

fn layout(layout: &Layout) -> String {
    let list = |values: &[i64]| {
        values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "#cuda_tile.preview$layout<({}):({})>",
        list(&layout.shapes),
        list(&layout.strides)
    )
}

fn tile(ty: &TileType) -> String {
    let dims: Vec<String> = ty.shape.iter().map(|d| d.to_string()).collect();
    format!("!cuda_tile.tile<{}x{}>", dims.join("x"), ty.element.name())
}

fn fragment(ty: &FragmentType) -> String {
    let dims: Vec<String> = ty.shape.iter().map(|d| d.to_string()).collect();
    format!(
        "!cuda_tile.preview$fragment<{}x{}, {}, mem_layout = {}, fragmap = {}>",
        dims.join("x"),
        ty.element.name(),
        ty.memory_space.name(),
        layout(&ty.mem_layout),
        layout(&ty.fragmap)
    )
}

fn access(access: crate::model::Access) -> String {
    format!(
        "{{cuda_tile.preview$access = #cuda_tile.preview$access<{}>}}",
        access.name()
    )
}

fn declaration(decl: &Declaration) -> String {
    let args: Vec<String> = decl
        .args
        .iter()
        .map(|arg| format!("%{}: {} {}", arg.name, tile(&arg.ty), access(arg.access)))
        .collect();
    let results: Vec<String> = decl.results().map(tile).collect();
    let results = match results.len() {
        1 => results[0].clone(),
        _ => format!("({})", results.join(", ")),
    };
    format!(
        "  cuda_tile.preview$func extern @{}({}) -> {}\n",
        decl.name,
        args.join(", "),
        results
    )
}

fn implementation(imp: &Implementation) -> String {
    let args: Vec<String> = imp
        .args
        .iter()
        .map(|arg| {
            format!(
                "%{}: {} {}",
                arg.name,
                fragment(&arg.fragment),
                access(arg.access)
            )
        })
        .collect();
    let mut text = format!(
        "  cuda_tile.preview$extern_impl @{} implements @{}({})",
        imp.symbol,
        imp.implements,
        args.join(", ")
    );
    if let Some(sync) = imp.synchronization {
        let _ = write!(
            text,
            "\n    synchronization = #cuda_tile.preview$synchronization<{}>",
            sync.name()
        );
    }
    if !imp.arch.is_empty() {
        let names: Vec<String> = imp.arch.iter().map(|name| format!("\"{name}\"")).collect();
        let _ = write!(text, "\n    arch = [{}]", names.join(", "));
    }
    if !imp.frontend_abi.is_empty() {
        let entries: Vec<String> = imp
            .frontend_abi
            .iter()
            .map(|(key, value)| format!("{key} = \"{value}\""))
            .collect();
        let _ = write!(text, "\n    {{frontendABI = {{{}}}}}", entries.join(", "));
    }
    text.push('\n');
    text
}

impl Library {
    /// The interface module in textual Tile IR.
    pub fn interface_text(&self) -> String {
        let mut text = format!("cuda_tile.module @{} {{\n", self.name);
        for decl in &self.declarations {
            text.push_str(&declaration(decl));
        }
        text.push_str("}\n");
        text
    }

    /// The implementation module in textual Tile IR.
    pub fn implementation_text(&self) -> String {
        let mut text = format!("cuda_tile.module @{} {{\n", self.name);
        for imp in &self.implementations {
            text.push_str(&implementation(imp));
        }
        text.push_str("}\n");
        text
    }
}
