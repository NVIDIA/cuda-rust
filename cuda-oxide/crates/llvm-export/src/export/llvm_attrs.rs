/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rendering for upstream LLVM function and call-site attributes.

use std::fmt::Write;

use crate::ops::{LlvmAttrValue, LlvmAttributesAttr};

use super::state::ModuleExportState;

fn write_quoted_llvm_string(value: &str, output: &mut String) {
    output.push('"');
    for byte in value.bytes() {
        match byte {
            b'"' | b'\\' | 0x00..=0x1f | 0x7f..=0xff => {
                write!(output, "\\{byte:02X}").unwrap();
            }
            _ => output.push(char::from(byte)),
        }
    }
    output.push('"');
}

impl ModuleExportState<'_> {
    /// Render LLVM attributes in the textual syntax accepted after function
    /// signatures and call instructions.
    ///
    /// `LlvmAttributesAttr` is the semantic source of truth. Callers must not
    /// special-case individual attribute names. Payload shapes whose textual
    /// spelling is not mechanically derivable are rejected here rather than
    /// silently emitting invalid LLVM IR.
    pub(super) fn export_llvm_attributes(
        &self,
        attrs: &LlvmAttributesAttr,
        output: &mut String,
    ) -> Result<(), String> {
        for (name, value) in attrs.iter() {
            write!(output, " ").unwrap();
            match value {
                LlvmAttrValue::Unit => write!(output, "{name}").unwrap(),
                LlvmAttrValue::Int(value) => {
                    return Err(format!(
                        "textual LLVM export does not yet support integer-valued function/call attribute `{name}` ({value}); integer payloads such as `memory` may require LLVM-specific decoding"
                    ));
                }
                LlvmAttrValue::Type(ty) => {
                    write!(output, "{name}(").unwrap();
                    self.export_type(*ty, output)?;
                    write!(output, ")").unwrap();
                }
                LlvmAttrValue::Str(value) => {
                    write_quoted_llvm_string(name, output);
                    write!(output, "=").unwrap();
                    write_quoted_llvm_string(value, output);
                }
            }
        }
        Ok(())
    }
}
