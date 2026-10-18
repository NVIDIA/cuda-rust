// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for LLVM dialect types.

use std::{any::Any, fmt};

use pliron::{
    builtin::type_interfaces::FunctionTypeInterface, context::Context, derive::type_interface_impl,
    dict_key, identifier::Identifier, printable::State, result::Result,
};
use pliron_llvm::types::{
    ArrayType, FuncType, PointerType, StructLayout, StructType, VectorType, VectorTypeKind,
    VoidType,
};

use crate::{
    ToMlirType,
    printers::{print_type, print_type_list},
};

/// `!llvm.ptr` (address space 0) or `!llvm.ptr<N>`.
#[type_interface_impl]
impl ToMlirType for PointerType {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        match self.address_space() {
            0 => write!(f, "!llvm.ptr")?,
            addr_space => write!(f, "!llvm.ptr<{addr_space}>")?,
        }
        Ok(())
    }
}

/// `!llvm.array<N x elem>`.
#[type_interface_impl]
impl ToMlirType for ArrayType {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "!llvm.array<{} x ", self.size())?;
        print_type(ctx, self.elem_type(), state, f)?;
        write!(f, ">")?;
        Ok(())
    }
}

/// `!llvm.void`.
#[type_interface_impl]
impl ToMlirType for VoidType {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "!llvm.void")?;
        Ok(())
    }
}

/// `!llvm.func<res (arg, ...)>`, with a trailing `...` argument when variadic.
#[type_interface_impl]
impl ToMlirType for FuncType {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "!llvm.func<")?;
        print_type(ctx, self.result_type(), state, f)?;
        write!(f, " (")?;
        let args = self.arg_types();
        print_type_list(ctx, args.iter().copied(), state, f)?;
        if self.is_var_arg() {
            write!(f, "{}...", if args.is_empty() { "" } else { ", " })?;
        }
        write!(f, ")>")?;
        Ok(())
    }
}

/// `vector<N x elem>`, or `vector<[N] x elem>` when scalable.
///
/// MLIR's LLVM dialect reuses the builtin vector type, so this is not
/// `!llvm.`-prefixed.
#[type_interface_impl]
impl ToMlirType for VectorType {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        match self.kind() {
            VectorTypeKind::Fixed => write!(f, "vector<{} x ", self.num_elements())?,
            VectorTypeKind::Scalable => write!(f, "vector<[{}] x ", self.num_elements())?,
        }
        print_type(ctx, self.elem_type(), state, f)?;
        write!(f, ">")?;
        Ok(())
    }
}

dict_key!(
    /// Names of the named structs currently being printed, innermost last.
    STRUCT_IN_PRINTING,
    "mlir_interop_struct_in_printing"
);

/// Record that we're now printing `name`, returning `true` if it's already
/// being printed higher up the stack (i.e., we've hit a recursive struct).
fn start_printing(state: &State, name: &Identifier) -> bool {
    let mut aux_data = state.aux_data_mut();
    let in_printing = aux_data
        .entry(STRUCT_IN_PRINTING.clone())
        .or_insert_with(|| Box::new(Vec::<Identifier>::new()) as Box<dyn Any>)
        .downcast_mut::<Vec<Identifier>>()
        .expect("failed to downcast struct-in-printing state");
    if in_printing.contains(name) {
        true
    } else {
        in_printing.push(name.clone());
        false
    }
}

/// We're done printing `name`, so pop it off the "under printing" stack.
fn done_printing(state: &State, name: &Identifier) {
    let mut aux_data = state.aux_data_mut();
    let in_printing = aux_data
        .get_mut(&STRUCT_IN_PRINTING)
        .expect("struct-in-printing state must have been created by now")
        .downcast_mut::<Vec<Identifier>>()
        .expect("failed to downcast struct-in-printing state");
    assert!(in_printing.last().unwrap() == name);
    in_printing.pop();
}

/// One of
///   - `!llvm.struct<(f0, f1)>` (anonymous),
///   - `!llvm.struct<packed (f0, f1)>` (anonymous, packed),
///   - `!llvm.struct<"name", (f0, f1)>` (named),
///   - `!llvm.struct<"name", opaque>` (named, no body), or
///   - `!llvm.struct<"name">` (a recursive reference to an enclosing struct).
#[type_interface_impl]
impl ToMlirType for StructType {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "!llvm.struct<")?;

        if let Some(name) = self.name() {
            if start_printing(state, &name) {
                // A recursive reference back to a struct we're already inside.
                write!(f, "\"{name}\">")?;
                return Ok(());
            }
            write!(f, "\"{name}\"")?;
            if self.is_opaque() {
                write!(f, ", opaque>")?;
                done_printing(state, &name);
                return Ok(());
            }
            write!(f, ", ")?;
        }

        if self.layout() == StructLayout::Packed {
            write!(f, "packed ")?;
        }
        write!(f, "(")?;
        print_type_list(ctx, self.fields(), state, f)?;
        write!(f, ")")?;

        if let Some(name) = self.name() {
            done_printing(state, &name);
        }
        write!(f, ">")?;
        Ok(())
    }
}
