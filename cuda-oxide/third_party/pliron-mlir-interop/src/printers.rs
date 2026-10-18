// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Pliron -> textual MLIR printers.

use std::{collections::HashMap, fmt};

use pliron::{
    attribute::{Attribute, attr_cast},
    basic_block::BasicBlock,
    builtin::attributes::ATTR_KEY_GIVEN_NAMES,
    common_traits::Named,
    context::{Context, Ptr},
    ident, input_err, input_err_noloc,
    linked_list::ContainsLinkedList,
    location::{Location, Source},
    op::op_cast,
    operation::Operation,
    printable::State,
    region::Region,
    result::Result,
    r#type::{TypeHandle, Typed, type_cast},
    uniqued_any,
    value::Value,
};

use crate::{Error, ToMlirAttr, ToMlirOp, ToMlirType};

/// Dispatch to `op`'s [ToMlirOp] implementation.
///
/// Fails if `op`'s concrete `Op` type does not implement [ToMlirOp].
pub fn print_op(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let op_obj = Operation::get_op_dyn(op, ctx);
    let Some(conv) = op_cast::<dyn ToMlirOp>(op_obj.as_ref()) else {
        return input_err!(
            op.deref(ctx).loc().clone(),
            Error::MissingOpTranslation(op_obj.get_opid().to_string())
        );
    };
    conv.to_mlir(ctx, state, f)
}

/// Dispatch to `ty`'s [ToMlirType] implementation.
///
/// Fails if `ty`'s concrete `Type` does not implement [ToMlirType].
pub fn print_type(
    ctx: &Context,
    ty: TypeHandle,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let ty_ref = ty.deref(ctx);
    let Some(conv) = type_cast::<dyn ToMlirType>(&*ty_ref) else {
        return input_err_noloc!(Error::MissingTypeTranslation(
            ty_ref.get_type_id().to_string()
        ));
    };
    conv.to_mlir(ctx, state, f)
}

/// Dispatch to `attr`'s [ToMlirAttr] implementation.
///
/// Fails if `attr`'s concrete `Attribute` does not implement [ToMlirAttr].
pub fn print_attr(
    ctx: &Context,
    attr: &dyn Attribute,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let Some(conv) = attr_cast::<dyn ToMlirAttr>(attr) else {
        return input_err_noloc!(Error::MissingAttrTranslation(
            attr.get_attr_id().to_string()
        ));
    };
    conv.to_mlir(ctx, state, f)
}

/// Print `types` as a comma separated list.
pub fn print_type_list(
    ctx: &Context,
    types: impl IntoIterator<Item = TypeHandle>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    for (i, ty) in types.into_iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        print_type(ctx, ty, state, f)?;
    }
    Ok(())
}

/// A value in one of MLIR's operation dictionaries.
enum MlirAttr<'a> {
    /// A pliron `Attribute`, translated through [ToMlirAttr].
    Attr(&'a dyn Attribute),
    /// Already-formatted MLIR attribute text, printed verbatim.
    Raw(String),
}

/// Print `entries` as `<open> key = value, ... <close>`,
/// or nothing when there are none.
fn print_dict(
    entries: &[(&str, MlirAttr<'_>)],
    open: &str,
    close: &str,
    ctx: &Context,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    write!(f, "{open}")?;
    for (i, (key, value)) in entries.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{key} = ")?;
        match value {
            MlirAttr::Attr(attr) => print_attr(ctx, *attr, state, f)?,
            MlirAttr::Raw(text) => write!(f, "{text}")?,
        }
    }
    write!(f, "{close}")?;
    Ok(())
}

/// Builder for MLIR's generic operation syntax:
///
/// ```text
/// operation         ::= op-result-list? generic-operation
/// generic-operation ::= string-literal `(` value-use-list? `)` successor-list?
///                       dictionary-properties? region-list? dictionary-attribute?
///                       `:` function-type
/// ```
///
/// e.g.,
///
/// ```text
/// %r0, %r1 = "mnemonic"(%o0, %o1) [^s0, ^s1] <{p = val}> ({ ... }) {attr = val}
///     : (t0, t1) -> (t2, t3)
/// ```
///
/// Operands, successors and regions come from the underlying `Operation`.
/// Results do too, unless overridden with [Self::results].
///
/// MLIR carries an operation's inherent data in the `<{...}>` (properties)
/// dictionary and everything else in the discardable `{...}` one. Both are
/// built up entry by entry.
pub struct GenericOp<'a> {
    op: Ptr<Operation>,
    mnemonic: &'a str,
    results: Option<Vec<Value>>,
    min_regions: usize,
    properties: Vec<(&'a str, MlirAttr<'a>)>,
    attributes: Vec<(&'a str, MlirAttr<'a>)>,
}

impl<'a> GenericOp<'a> {
    /// Print `op` with the given MLIR `mnemonic` and no dictionary entries.
    pub fn new(op: Ptr<Operation>, mnemonic: &'a str) -> Self {
        GenericOp {
            op,
            mnemonic,
            results: None,
            min_regions: 0,
            properties: vec![],
            attributes: vec![],
        }
    }

    /// Print `results` instead of the `Operation`'s own results.
    pub fn results(mut self, results: Vec<Value>) -> Self {
        self.results = Some(results);
        self
    }

    /// Print at least `n` regions, padding with empty ones.
    pub fn min_regions(mut self, n: usize) -> Self {
        self.min_regions = n;
        self
    }

    /// Add a pliron `Attribute` to the properties dictionary.
    pub fn prop_attr(mut self, name: &'a str, attr: &'a dyn Attribute) -> Self {
        self.properties.push((name, MlirAttr::Attr(attr)));
        self
    }

    /// Like [Self::prop_attr], but for an optional pliron attribute
    pub fn prop_attr_opt<A: Attribute>(self, name: &'a str, attr: Option<&'a A>) -> Self {
        match attr {
            Some(attr) => self.prop_attr(name, attr),
            None => self,
        }
    }

    /// Add pre-formatted MLIR text to the properties dictionary.
    pub fn prop_raw(mut self, name: &'a str, text: impl Into<String>) -> Self {
        self.properties.push((name, MlirAttr::Raw(text.into())));
        self
    }

    /// Add MLIR's `UnitAttr` to the properties dictionary when `set` is true.
    pub fn prop_unit(self, name: &'a str, set: bool) -> Self {
        if set {
            self.prop_raw(name, "unit")
        } else {
            self
        }
    }

    /// Like [Self::prop_raw], but for an optional pliron attribute
    pub fn prop_raw_opt(self, name: &'a str, text: Option<impl Into<String>>) -> Self {
        match text {
            Some(text) => self.prop_raw(name, text),
            None => self,
        }
    }

    /// Add pre-formatted MLIR text to the discardable attribute dictionary.
    pub fn attr_raw(mut self, name: &'a str, text: impl Into<String>) -> Self {
        self.attributes.push((name, MlirAttr::Raw(text.into())));
        self
    }

    /// Add a pliron `Attribute` to the discardable attribute dictionary.
    pub fn attr_attr(mut self, name: &'a str, attr: &'a dyn Attribute) -> Self {
        self.attributes.push((name, MlirAttr::Attr(attr)));
        self
    }

    /// Print the operation.
    pub fn print(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let operation = self.op.deref(ctx);

        // Results are `Value`s, whose types we look up, so that an overriding
        // `results()` list needs to carry only the values.
        let results: Vec<Value> = match &self.results {
            Some(results) => results.clone(),
            None => operation.results().collect(),
        };

        for (i, res) in results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}", res.unique_name(ctx))?;
        }
        if !results.is_empty() {
            write!(f, " = ")?;
        }

        write!(f, "\"{}\"(", self.mnemonic)?;
        for (i, opd) in operation.operands().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}", opd.unique_name(ctx))?;
        }
        write!(f, ")")?;

        if operation.get_num_successors() > 0 {
            write!(f, " [")?;
            for (i, succ) in operation.successors().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "^{}", succ.deref(ctx).unique_name(ctx))?;
            }
            write!(f, "]")?;
        }

        print_dict(&self.properties, " <{", "}>", ctx, state, f)?;

        let num_regions = operation.num_regions().max(self.min_regions);
        if num_regions > 0 {
            write!(f, " (")?;
            for (i, region) in operation.regions().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                print_region(ctx, region, state, f)?;
            }
            for i in operation.num_regions()..num_regions {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{{}}")?;
            }
            write!(f, ")")?;
        }

        print_dict(&self.attributes, " {", "}", ctx, state, f)?;

        write!(f, " : (")?;
        print_type_list(ctx, operation.operand_types(ctx), state, f)?;
        write!(f, ") -> (")?;
        print_type_list(ctx, results.iter().map(|res| res.get_type(ctx)), state, f)?;
        write!(f, ")")?;

        print_op_location(ctx, operation.loc(), state, f)
    }
}

/// Print ` loc(...)` for an operation at `loc`, or nothing for an unknown `loc`.
pub fn print_op_location(
    ctx: &Context,
    loc: &Location,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    if loc.is_unknown() {
        return Ok(());
    }
    write!(f, " loc(")?;
    print_location(ctx, loc, state, f)?;
    write!(f, ")")?;
    Ok(())
}

/// Print `loc` as the body of an MLIR `loc(...)`.
///
/// | pliron | MLIR |
/// |--------|------|
/// | [Location::SrcPos] | `"file":line:column` |
/// | [Location::Fused] | `fused<metadata>[loc, ...]` |
/// | [Location::Named] | `"name"(loc)` |
/// | [Location::CallSite] | `callsite(callee at caller)` |
/// | [Location::Unknown] | `unknown` |
pub fn print_location(
    ctx: &Context,
    loc: &Location,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    match loc {
        Location::SrcPos { src, pos } => {
            let file = match src {
                Source::File(path) => uniqued_any::get(ctx, *path).display().to_string(),
                Source::InMemory => "<in-memory>".to_string(),
            };
            // MLIR's line and column numbers have no sign.
            write!(
                f,
                "{}:{}:{}",
                mlir_string_literal(&file),
                pos.line.max(0),
                pos.column.max(0)
            )?;
        }
        Location::Fused {
            metadata,
            locations,
        } => {
            write!(f, "fused")?;
            if let Some(metadata) = metadata {
                write!(f, "<")?;
                print_attr(ctx, &**metadata, state, f)?;
                write!(f, ">")?;
            }
            write!(f, "[")?;
            for (i, loc) in locations.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                print_location(ctx, loc, state, f)?;
            }
            write!(f, "]")?;
        }
        Location::Named { name, child_loc } => {
            write!(f, "{}", mlir_string_literal(name))?;
            // MLIR leaves the child off when it is unknown.
            if !child_loc.is_unknown() {
                write!(f, "(")?;
                print_location(ctx, child_loc, state, f)?;
                write!(f, ")")?;
            }
        }
        Location::CallSite { callee, caller } => {
            write!(f, "callsite(")?;
            print_location(ctx, callee, state, f)?;
            write!(f, " at ")?;
            print_location(ctx, caller, state, f)?;
            write!(f, ")")?;
        }
        Location::Unknown => write!(f, "unknown")?,
    }
    Ok(())
}

/// Print `op` using MLIR's generic operation syntax, forwarding every pliron
/// attribute of `op` to MLIR's discardable attribute dictionary.
///
/// `GivenNamesAttr` are skipped
/// since names are printed as part of the SSA definitions.
///
/// This is a best-effort fallback for `Op`s with no dedicated
/// MLIR translation. Ops can customize their printing behaviour using [GenericOp],
/// or go completely wild and implement [ToMlirOp] fully manually.
pub fn print_generic_op(
    ctx: &Context,
    op: Ptr<Operation>,
    mnemonic: &str,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let operation = op.deref(ctx);
    let mut generic = GenericOp::new(op, mnemonic);
    // `ATTR_KEY_GIVEN_NAMES` holds pliron's names for results and block arguments.
    // Those become actual names when translating to MLIR. So we skip them as attributes.
    for (key, val) in operation
        .attributes
        .0
        .iter()
        .filter(|(key, _)| **key != ATTR_KEY_GIVEN_NAMES)
    {
        generic = generic.attr_attr(key.as_ref(), &**val);
    }
    generic.print(ctx, state, f)
}

/// Print `region` as `{ block* }`, per MLIR's `region ::= '{' block* '}'`.
///
/// Every block (including the entry block) is printed with an explicit label.
fn print_region(
    ctx: &Context,
    region: Ptr<Region>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    write!(f, "{{")?;
    {
        let _indent = state.indent();
        for block in region.deref(ctx).iter(ctx) {
            print_newline_indent(state, f)?;
            print_block(ctx, block, state, f)?;
        }
    }
    print_newline_indent(state, f)?;
    write!(f, "}}")?;
    Ok(())
}

/// Print `block` as `^label(args): operation*`.
fn print_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    {
        let block_ref = block.deref(ctx);
        write!(f, "^{}", block_ref.unique_name(ctx))?;
        if block_ref.get_num_arguments() > 0 {
            write!(f, "(")?;
            for (i, arg) in block_ref.arguments().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%{}: ", arg.unique_name(ctx))?;
                print_type(ctx, arg.get_type(ctx), state, f)?;
                let location = state
                    .aux_data_ref()
                    .get(&ident!("mlir_block_argument_locations"))
                    .and_then(|locations| locations.downcast_ref::<HashMap<Value, Location>>())
                    .and_then(|locations| locations.get(&arg).cloned())
                    .unwrap_or_else(|| arg.loc(ctx));
                print_op_location(ctx, &location, state, f)?;
            }
            write!(f, ")")?;
        }
        write!(f, ":")?;
    }

    let _indent = state.indent();
    for op in block.deref(ctx).iter(ctx) {
        print_newline_indent(state, f)?;
        print_op(ctx, op, state, f)?;
    }
    Ok(())
}

/// Override block argument locations for this print state.
///
/// Pliron normally gives each argument its parent block's location. Adapters
/// from IRs with individual argument locations can supply those locations
/// here, including an explicit unknown location to suppress the fallback.
/// Replaces any previously supplied overrides; values must belong to the
/// context being printed. The caller can set this inside [crate::MlirPrinterT].
pub fn set_block_argument_locations(
    state: &State,
    locations: impl IntoIterator<Item = (Value, Location)>,
) {
    state.aux_data_mut().insert(
        ident!("mlir_block_argument_locations"),
        Box::new(locations.into_iter().collect::<HashMap<_, _>>()),
    );
}

fn print_newline_indent(state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
    write!(
        f,
        "\n{:width$}",
        "",
        width = state.current_indent() as usize
    )?;
    Ok(())
}

/// Format `s` as an MLIR string literal, including the surrounding quotes.
///
/// MLIR's lexer accepts `\"`, `\\`, `\n`, `\t` and `\xx` (two hex digits) as
/// escapes; everything outside printable ASCII goes out as `\xx` bytes.
pub fn mlir_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for byte in s.bytes() {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(byte as char),
            _ => out.push_str(&format!("\\{byte:02X}")),
        }
    }
    out.push('"');
    out
}

/// Format an IEEE float of `bit_width` bits, given as its `bits` encoding, as
/// an MLIR hexadecimal float literal (e.g. `0x7FC00000`).
pub fn mlir_hex_float(bits: u128, bit_width: usize) -> String {
    format!("0x{:0width$X}", bits, width = bit_width / 4)
}

/// Format `elems` as an MLIR dense array attribute, e.g. `array<i32: 1, 2, 3>`.
pub fn mlir_dense_array<T: fmt::Display>(
    elem_type: &str,
    elems: impl IntoIterator<Item = T>,
) -> String {
    let elems = elems
        .into_iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if elems.is_empty() {
        format!("array<{elem_type}>")
    } else {
        format!("array<{elem_type}: {elems}>")
    }
}
