/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiled value representation for compiler2.
//!
//! Uses `cutile_ir::ir::Value` (Copy, index-based, no lifetimes).

use super::shared_types::Kind;
use super::tile_rust_type::TileRustType;
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::syn_utils::get_type_ident;
use cutile_ir::ir::Value;
use std::collections::{BTreeMap, BTreeSet};
use syn::Expr;

// Re-export shared types.
pub use super::shared_types::{BlockTerminator, LoopKind, Mutability};

/// Flattens all values in a `BTreeMap` of [`TileRustValue`]s into a linear list.
pub fn unpack_btree_to(
    btree: &BTreeMap<String, TileRustValue>,
    values: &mut Vec<Value>,
) -> Result<(), JITError> {
    for key in btree.keys() {
        let value = btree[key].clone();
        value.unpack_to(values)?;
    }
    Ok(())
}

/// Reconstructs a `BTreeMap` of [`TileRustValue`]s from a flat list of values.
pub fn repack_btree_from(
    old_btree: &BTreeMap<String, TileRustValue>,
    values: &Vec<Value>,
    mut pos: usize,
) -> Result<(BTreeMap<String, TileRustValue>, usize), JITError> {
    let mut new_btree = BTreeMap::new();
    for key in old_btree.keys() {
        let value = old_btree[key].clone();
        let res = value.repack_from(values, pos)?;
        new_btree.insert(key.to_string(), res.0);
        pos = res.1;
    }
    Ok((new_btree, pos))
}

/// Type-level metadata (named sub-fields) attached to structured values like views.
#[derive(Debug, Clone)]
pub struct TypeMeta {
    pub fields: BTreeMap<String, TileRustValue>,
}

impl TypeMeta {
    fn unpack_to(&self, values: &mut Vec<Value>) -> Result<(), JITError> {
        unpack_btree_to(&self.fields, values)
    }
    fn repack_from(&self, values: &Vec<Value>, pos: usize) -> Result<(TypeMeta, usize), JITError> {
        let res = repack_btree_from(&self.fields, values, pos)?;
        Ok((TypeMeta { fields: res.0 }, res.1))
    }
}

/// One enclosing loop, for hoisting bounds checks out of hot loop bodies.
///
/// `value_watermark` is the module's value count taken just before the loop
/// body block was built: a value with a smaller index was defined before the
/// loop and therefore dominates `preheader_block` (the block the `for` op is
/// appended to; ops emitted there during body compilation land before it).
#[derive(Debug, Clone)]
pub(crate) struct LoopFrame {
    pub(crate) preheader_block: cutile_ir::ir::BlockId,
    /// The loop body block. Hoisting is only sound for checks emitted
    /// directly in the body — never from nested conditional blocks, where
    /// the guarded access may not execute on every iteration.
    pub(crate) body_block: cutile_ir::ir::BlockId,
    pub(crate) value_watermark: u32,
    /// The raw induction block argument and any assumption-wrapped aliases
    /// the loop variable was bound to.
    pub(crate) induction_values: Vec<Value>,
    /// Loop bounds `[lower, upper)` as preheader values.
    pub(crate) lower: Value,
    pub(crate) upper: Value,
    /// True when the step is the constant 1, making `upper - 1` the exact
    /// maximum induction value for a non-empty loop.
    pub(crate) unit_step: bool,
    /// True when static bounds prove the loop executes at least once
    /// (`max(lower) < min(upper)`). Lets hoisted checks skip the vacuous-trip
    /// guard, and lets checks hoist past this loop entirely.
    pub(crate) known_non_empty: bool,
    /// The induction variable's inclusive value range `[lower, upper - 1]` when
    /// both loop bounds are compile-time constants (and the step is unit). Lets
    /// the hoister derive an affine index's static range from its `Term` (via
    /// `value_facts::term_range`) — discharging it as a compile-time constant
    /// instead of a runtime strongest-instance substitution. `None` when either
    /// bound is a runtime value.
    pub(crate) induction_range: Option<crate::bounds::Bounds<i64>>,
    /// True when the loop body contains an early exit (`continue`, `break`,
    /// `return`) anywhere, including inside nested conditionals. A check
    /// hoisted to the preheader assumes the guarded access executes on every
    /// iteration — at the loop extremes in particular — and an early exit
    /// breaks that: `if k >= limit { continue; }` before an access attains
    /// only `[0, limit)`, so testing the range's extreme traps spuriously
    /// (differential harness defect D2). No check hoists out of such a body,
    /// and no check from an inner loop hoists across it.
    pub(crate) has_early_exit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PartitionAxisOrigin {
    pub(crate) tensor: String,
    pub(crate) axis: usize,
    pub(crate) tile_dim: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DimOrigin {
    PartitionAxis {
        view: Value,
        axis: usize,
        tile_dim: i32,
    },
    Value(Value),
}

/// A compiled value: wraps a `cutile_ir::ir::Value` together with its Rust type, kind, bounds,
/// and metadata.
///
/// Port of old `TileRustValue<'c, 'a>` — all lifetime parameters are gone because
/// `cutile_ir::ir::Value` is `Copy` (a u32 arena index) and `TileRustType` is owned.
#[derive(Debug, Clone)]
pub struct TileRustValue {
    pub(crate) kind: Kind,
    pub(crate) fields: Option<BTreeMap<String, TileRustValue>>,
    pub(crate) values: Option<Vec<TileRustValue>>,
    pub(crate) value: Option<Value>,
    pub(crate) ty: TileRustType,
    pub(crate) type_meta: Option<TypeMeta>,
    pub(crate) mutability: Mutability,
    pub(crate) bounds: Option<Bounds<i64>>,
    pub(crate) string_literal: Option<syn::Expr>,
    pub(crate) enum_variant: Option<String>,
    pub(crate) enum_payload: Option<Box<syn::Expr>>,
    pub(crate) partition_origins: Option<Vec<Value>>,
    pub(crate) tensor_origin: Option<String>,
    pub(crate) partition_axis_origin: Option<PartitionAxisOrigin>,
    pub(crate) dim_origin: Option<DimOrigin>,
    pub(crate) index_origin: Option<DimOrigin>,
    pub(crate) bounded_axes: Option<Vec<DimOrigin>>,
    /// `floor(numerator / divisor)` provenance for a value produced by integer
    /// division by a constant. Analysis-side only: see [`crate::value_facts::FloorDiv`].
    pub(crate) floor_div: Option<crate::value_facts::FloorDiv>,
    /// Symbolic (canonical linear) form of this scalar value, when known:
    /// `sum(coeff*atom) + constant` over induction-variable / dim atoms. The
    /// affine fragment used by loop check-hoisting is [`Term::as_single_affine`].
    /// Consolidates the former `AffineForm { scale, var, offset }` (its single-
    /// `Iv`-atom special case).
    pub(crate) term: Option<cuda_async::predicate::Term>,
}

impl TileRustValue {
    pub fn new_struct(fields: BTreeMap<String, TileRustValue>, ty: TileRustType) -> TileRustValue {
        Self {
            fields: Some(fields),
            values: None,
            value: None,
            ty,
            kind: Kind::Struct,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_compound(values: Vec<TileRustValue>, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: Some(values),
            value: None,
            ty,
            kind: Kind::Compound,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_structured_type(
        value: Value,
        ty: TileRustType,
        type_meta: Option<TypeMeta>,
    ) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: Some(value),
            ty,
            kind: Kind::StructuredType,
            type_meta,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_primitive(
        value: Value,
        ty: TileRustType,
        bounds: Option<Bounds<i64>>,
    ) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: Some(value),
            ty,
            kind: Kind::PrimitiveType,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds,
            string_literal: None,
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_string(string_literal: Expr, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: None,
            ty,
            kind: Kind::String,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: Some(string_literal),
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_enum(
        variant: impl Into<String>,
        payload: Option<syn::Expr>,
        ty: TileRustType,
    ) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: None,
            ty,
            kind: Kind::Enum,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
            enum_variant: Some(variant.into()),
            enum_payload: payload.map(Box::new),
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn new_value_kind_like(value: Value, ty: TileRustType) -> TileRustValue {
        let kind = ty.kind.clone();
        match kind {
            Kind::StructuredType => Self::new_structured_type(
                value,
                ty,
                Some(TypeMeta {
                    fields: BTreeMap::new(),
                }),
            ),
            _ => Self {
                fields: None,
                values: None,
                value: Some(value),
                ty,
                kind,
                type_meta: None,
                mutability: Mutability::Unset,
                bounds: None,
                string_literal: None,
                enum_variant: None,
                enum_payload: None,
                partition_origins: None,
                tensor_origin: None,
                partition_axis_origin: None,
                dim_origin: None,
                index_origin: None,
                bounded_axes: None,
                floor_div: None,
                term: None,
            },
        }
    }

    pub fn new_literal(literal_expr: syn::Expr, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: None,
            ty,
            kind: Kind::PrimitiveType,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: Some(literal_expr),
            enum_variant: None,
            enum_payload: None,
            partition_origins: None,
            tensor_origin: None,
            partition_axis_origin: None,
            dim_origin: None,
            index_origin: None,
            bounded_axes: None,
            floor_div: None,
            term: None,
        }
    }

    pub fn verify(&self) -> Result<(), JITError> {
        match self.kind {
            Kind::String => {
                if !(self.string_literal.is_some()
                    && self.type_meta.is_none()
                    && self.value.is_none()
                    && self.values.is_none()
                    && self.fields.is_none())
                {
                    return JITError::generic("internal: string value has inconsistent fields set");
                }
            }
            Kind::PrimitiveType => {
                if !(self.value.is_some() && self.values.is_none() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: primitive value has inconsistent fields set",
                    );
                }
            }
            Kind::StructuredType => {
                if !(self.value.is_some() && self.values.is_none() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: structured type value has inconsistent fields set",
                    );
                }
            }
            Kind::Compound => {
                if !(self.value.is_none() && self.values.is_some() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: compound value has inconsistent fields set",
                    );
                }
            }
            Kind::Struct => {
                if !(self.value.is_none() && self.values.is_none() && self.fields.is_some()) {
                    return JITError::generic("internal: struct value has inconsistent fields set");
                }
            }
            Kind::Enum => {
                if !(self.value.is_none()
                    && self.values.is_none()
                    && self.fields.is_none()
                    && self.enum_variant.is_some())
                {
                    return JITError::generic("internal: enum value has inconsistent fields set");
                }
            }
        }
        Ok(())
    }

    pub fn get_type_meta_field(&self, name: &str) -> Option<&Self> {
        let Some(type_meta) = &self.type_meta else {
            return None;
        };
        type_meta.fields.get(name)
    }

    pub fn take_type_meta_field(self, name: &str) -> Option<Self> {
        let mut type_meta = self.type_meta?;
        type_meta.fields.remove(name)
    }

    pub fn insert_type_meta_field(
        &mut self,
        name: &str,
        val: TileRustValue,
    ) -> Result<(), JITError> {
        let Some(type_meta) = &mut self.type_meta else {
            return JITError::generic(&format!(
                "type metadata not supported for {:?} values",
                self.ty.kind
            ));
        };
        type_meta.fields.insert(name.to_string(), val.clone());
        Ok(())
    }

    pub fn get_token(&self) -> Option<&Self> {
        self.get_type_meta_field("token")
    }

    pub fn is_tile(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Tile")
    }

    pub fn is_partition(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Partition")
    }

    pub fn unpack_to(&self, values: &mut Vec<Value>) -> Result<(), JITError> {
        self.verify()?;
        match self.kind {
            Kind::String => {}
            Kind::PrimitiveType => {
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::StructuredType => {
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::Compound => {
                let Some(self_values) = &self.values else {
                    return JITError::generic("internal: compound value missing its element list");
                };
                for value in self_values {
                    value.unpack_to(values)?;
                }
            }
            Kind::Struct => {
                let Some(fields) = &self.fields else {
                    return JITError::generic("internal: struct value missing its fields");
                };
                unpack_btree_to(fields, values)?;
            }
            Kind::Enum => {}
        }
        Ok(())
    }

    /// Clears every fact a control-flow join can invalidate, recursively:
    /// interval bounds, the symbolic term, axis/index provenance, floor-div
    /// lineage, and the structural facts (`tensor_origin`, `bounded_axes`,
    /// `partition_origins`). Called when a value is reconstructed at a
    /// control-flow join or loop carry, where the value it describes may
    /// differ from the one the facts were established for — keeping them
    /// discharged bounds checks for conditionally reassigned indices and
    /// partitions (issue #212, both halves).
    ///
    /// Only type wiring survives: `kind`, `ty`, `mutability`, `type_meta`
    /// structure, and comptime payloads — reassignment cannot change what
    /// type the variable is. Clearing structural facts does NOT break
    /// partitions carried across their own store loops, because repack runs
    /// only on the mutated/captured set and storing through a partition
    /// does not reassign the binding — such partitions never enter this
    /// path and keep their brands.
    pub(crate) fn invalidate_join_facts(&mut self) {
        self.bounds = None;
        self.term = None;
        self.index_origin = None;
        self.partition_axis_origin = None;
        self.dim_origin = None;
        self.floor_div = None;
        self.partition_origins = None;
        // Structural facts too: a variable reaching this path was reassigned
        // (repack runs only on the mutated/captured set), so a branch may
        // have pointed it at a DIFFERENT partition. Keeping `tensor_origin`
        // or `bounded_axes` from the pre-branch template would let the
        // cross-tensor rung bound an access against the wrong tensor's extent
        // — an out-of-bounds access proven safe (issue #212, structural
        // residual). A partition that is merely READ across a branch is not
        // in the reassigned set and keeps its facts.
        self.tensor_origin = None;
        self.bounded_axes = None;
        if let Some(values) = &mut self.values {
            for v in values.iter_mut() {
                v.invalidate_join_facts();
            }
        }
        if let Some(fields) = &mut self.fields {
            for v in fields.values_mut() {
                v.invalidate_join_facts();
            }
        }
        if let Some(type_meta) = &mut self.type_meta {
            for v in type_meta.fields.values_mut() {
                v.invalidate_join_facts();
            }
        }
    }

    pub fn repack_from(
        &self,
        values: &Vec<Value>,
        mut pos: usize,
    ) -> Result<(Self, usize), JITError> {
        self.verify()?;
        let mut result = self.clone();
        match self.kind {
            Kind::String => {}
            Kind::PrimitiveType => {
                result.value = Some(values[pos]);
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::StructuredType => {
                result.value = Some(values[pos]);
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::Compound => {
                let Some(self_values) = &result.values else {
                    return JITError::generic("internal: compound value missing its element list");
                };
                let mut result_values = vec![];
                for value in self_values {
                    let res = value.repack_from(values, pos)?;
                    result_values.push(res.0);
                    pos = res.1;
                }
                result.values = Some(result_values);
            }
            Kind::Struct => {
                let Some(fields) = &result.fields else {
                    return JITError::generic("internal: struct value missing its fields");
                };
                let res = repack_btree_from(fields, values, pos)?;
                result.fields = Some(res.0);
                pos = res.1;
            }
            Kind::Enum => {}
        }
        result.verify()?;
        Ok((result, pos))
    }
}

/// Variable scope and control-flow state for a compilation block.
#[derive(Debug, Clone)]
pub struct CompilerContext {
    pub vars: BTreeMap<String, TileRustValue>,
    /// Bindings declared in the function body, not in a nested block. Inlined
    /// parameters inherit eligibility from their caller's argument binding.
    pub(crate) function_level_bindings: BTreeSet<String>,
    /// Explicit token installations, used to reject a later shadow that would
    /// otherwise discard the update when the block's metadata is copied back.
    pub(crate) explicit_token_updates: BTreeSet<String>,
    /// Explicit token installation cannot yet cross conditional/loop regions.
    /// Also inherited by inlined callees, whose own loop flags start afresh.
    pub(crate) token_update_in_region: bool,
    pub carry_vars: Option<Vec<String>>,
    pub default_terminator: Option<BlockTerminator>,
    pub module_scope: Vec<String>,
    /// Enclosing loops, innermost last. Lets check emission hoist
    /// loop-invariant and induction-variable bounds checks into the
    /// innermost loop's preheader instead of the hot loop body.
    pub(crate) loop_frames: Vec<LoopFrame>,
    /// True while this context compiles a function body block itself — the
    /// kernel entry body or an inlined callee's body. `compile_block` clears it
    /// on entry, so every nested block (an `if` branch, a loop body, a bare
    /// `{}`/`unsafe {}` block) compiled from a clone sees `false`: a
    /// Nested returns require the separate kernel-loop permission below;
    /// otherwise they would fall through and are rejected.
    pub(crate) fn_body: bool,
    /// A nested Tile IR return exits the kernel, never an inlined helper.
    pub(crate) kernel_entry: bool,
    /// Return is prohibited under any `for`, including an inner `loop`.
    pub(crate) inside_for: bool,
    /// The Tile IR loop op of the innermost enclosing source loop, if any.
    /// Inherited by nested blocks; decides whether a `break` is
    /// representable (only inside `cuda_tile.loop`).
    pub(crate) innermost_loop: Option<LoopKind>,
}

impl CompilerContext {
    pub fn empty() -> CompilerContext {
        Self {
            vars: BTreeMap::new(),
            function_level_bindings: BTreeSet::new(),
            explicit_token_updates: BTreeSet::new(),
            token_update_in_region: false,
            carry_vars: None,
            default_terminator: None,
            module_scope: vec![],
            loop_frames: vec![],
            fn_body: false,
            kernel_entry: false,
            inside_for: false,
            innermost_loop: None,
        }
    }

    pub fn var_keys(&self) -> Vec<String> {
        self.vars.keys().cloned().collect()
    }

    pub fn unpack_vars(&self) -> Result<Vec<Value>, JITError> {
        let mut result = vec![];
        unpack_btree_to(&self.vars, &mut result)?;
        Ok(result)
    }

    pub fn repack_vars(
        &self,
        vars: &Vec<Value>,
        module_scope: Vec<String>,
        carry_vars: Option<Vec<String>>,
        default_terminator: Option<BlockTerminator>,
    ) -> Result<CompilerContext, JITError> {
        let res = repack_btree_from(&self.vars, vars, 0)?;
        Ok(CompilerContext {
            vars: res.0,
            function_level_bindings: self.function_level_bindings.clone(),
            explicit_token_updates: self.explicit_token_updates.clone(),
            token_update_in_region: self.token_update_in_region,
            carry_vars,
            default_terminator,
            module_scope,
            loop_frames: self.loop_frames.clone(),
            fn_body: false,
            kernel_entry: self.kernel_entry,
            inside_for: self.inside_for,
            innermost_loop: self.innermost_loop,
        })
    }

    pub fn unpack_some_vars(&self, keys: &Vec<String>) -> Result<Vec<Value>, JITError> {
        let mut result = vec![];
        for key in keys {
            let Some(value) = self.vars.get(key) else {
                return JITError::generic(&format!("Variable not found {key}"));
            };
            value.unpack_to(&mut result)?;
        }
        Ok(result)
    }

    pub fn repack_some_vars(
        &mut self,
        keys: &Vec<String>,
        vars: &Vec<Value>,
        invalidate_bounds: bool,
    ) -> Result<(), JITError> {
        let mut pos = 0;
        for key in keys {
            let Some(value) = self.vars.get(key) else {
                return JITError::generic(&format!("Variable not found {key}"));
            };
            let (mut new_value, new_pos) = value.repack_from(vars, pos)?;
            if invalidate_bounds {
                new_value.invalidate_join_facts();
            }
            pos = new_pos;
            self.vars.insert(key.clone(), new_value);
        }
        Ok(())
    }

    /// Publishes complete values from a path compiled directly into the
    /// current block. Unlike [`Self::repack_some_vars`], this preserves the
    /// facts established by that path because there is no control-flow join:
    /// one compile-time-known branch is the only possible definition.
    pub fn replace_some_vars_from(
        &mut self,
        keys: &[String],
        source: &CompilerContext,
    ) -> Result<(), JITError> {
        for key in keys {
            let value = source
                .vars
                .get(key)
                .ok_or_else(|| JITError::Generic(format!("Variable not found {key}")))?;
            self.vars.insert(key.clone(), value.clone());
        }
        Ok(())
    }
}
