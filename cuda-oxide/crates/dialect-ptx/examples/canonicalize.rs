/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_ptx::{Projection, emit_canonical_module, raising::NativeCfgPlan};
use pliron::context::Context;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let first = arguments
        .next()
        .ok_or("usage: canonicalize [--native-cfg] <input.ptx> [output.ptx]")?;
    let native_cfg = first == "--native-cfg";
    let input = (if native_cfg {
        arguments.next()
    } else {
        Some(first)
    })
    .map(PathBuf::from)
    .ok_or("usage: canonicalize [--native-cfg] <input.ptx> [output.ptx]")?;
    let output = arguments.next().map(PathBuf::from);
    let source = std::fs::read_to_string(&input)?;
    let mut ctx = Context::new();
    dialect_ptx::register(&mut ctx);
    let emitted = if native_cfg {
        let projection = NativeCfgPlan::analyze(&source)?.materialize(&mut ctx);
        emit_canonical_module(&ctx, &projection.module())?
    } else {
        let projection = Projection::parse(&mut ctx, &source)?;
        emit_canonical_module(&ctx, &projection.module())?
    };
    let reparsed = ptx_parse::Document::parse(&emitted)?;
    if !reparsed.coverage().is_complete() {
        return Err(format!(
            "canonical PTX is structurally incomplete: {:?}",
            reparsed.coverage()
        )
        .into());
    }
    if let Some(output) = output {
        std::fs::write(output, emitted)?;
    } else {
        print!("{emitted}");
    }
    Ok(())
}
