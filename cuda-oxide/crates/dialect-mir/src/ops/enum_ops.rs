/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR enum operations.
//!
//! This module defines enum construction and inspection operations for the MIR dialect.

use pliron::{
    builtin::op_interfaces::{
        NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::{FieldIndexAttr, VariantIndexAttr};
use crate::types::MirEnumType;

// ============================================================================
// MirConstructEnumOp
// ============================================================================

/// MIR construct enum operation.
///
/// Constructs an enum variant with optional payload fields.
///
/// # Operands
///
/// Takes N operands (the variant's fields, if any).
///
/// # Attributes
///
/// ```text
/// | Name                           | Type        | Description                    |
/// |--------------------------------|-------------|--------------------------------|
/// | `construct_enum_variant_index` | VariantIndexAttr | Index of variant to construct  |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type        |
/// |-------|-------------|
/// | `res` | MirEnumType |
/// ```
///
/// # Verification
///
/// - Number of operands must match the variant's field count.
/// - Each operand type must match corresponding variant field type.
/// - Result type must be an enum type.
#[pliron_op(
    name = "mir.construct_enum",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface],
    attributes = (construct_enum_variant_index: VariantIndexAttr)
)]
pub struct MirConstructEnumOp;

impl MirConstructEnumOp {
    /// Create a new MirConstructEnumOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstructEnumOp { op }
    }
}

impl Verify for MirConstructEnumOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Diagnose malformed parsed/hand-built IR before indexing result 0;
        // verifier interface ordering is not a safety guarantee.
        if op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "MirConstructEnumOp expects exactly one result, got {}",
                op.get_num_results()
            );
        }

        // Result must be an enum type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let enum_ty = match result_ty_obj.downcast_ref::<MirEnumType>() {
            Some(et) => et,
            None => {
                return verify_err!(op.loc(), "MirConstructEnumOp result must be an enum type");
            }
        };

        // Get variant index
        let variant_idx = match self.get_attr_construct_enum_variant_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirConstructEnumOp missing construct_enum_variant_index attribute"
                );
            }
        };

        // Get the variant
        let variant = match enum_ty.get_variant(variant_idx) {
            Some(v) => v,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirConstructEnumOp variant_index {} out of bounds for enum '{}' with {} variants",
                    variant_idx,
                    enum_ty.name(),
                    enum_ty.variant_count()
                );
            }
        };
        if enum_ty.variant_is_inhabited(variant_idx) == Some(false) {
            return verify_err!(
                op.loc(),
                "MirConstructEnumOp cannot construct uninhabited variant {}",
                variant_idx
            );
        }

        // Verify operand count matches variant field count
        let num_operands = op.get_num_operands();
        let num_fields = variant.field_types.len();
        if num_operands != num_fields {
            return verify_err!(
                op.loc(),
                "MirConstructEnumOp has {} operands but variant '{}' has {} fields",
                num_operands,
                variant.name,
                num_fields
            );
        }

        // Verify each operand type matches field type
        for i in 0..num_fields {
            let operand = op.get_operand(i);
            let operand_ty = operand.get_type(ctx);
            let expected_ty = variant.field_types[i];

            if operand_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirConstructEnumOp operand {} type mismatch for variant '{}'. Expected: {}, Actual: {}",
                    i,
                    variant.name,
                    expected_ty.disp(ctx),
                    operand_ty.disp(ctx)
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// MirGetDiscriminantOp
// ============================================================================

/// MIR get discriminant operation.
///
/// Extracts the discriminant (tag) from an enum value.
///
/// # Operands
///
/// ```text
/// | Name    | Type        |
/// |---------|-------------|
/// | `value` | MirEnumType |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                              |
/// |-------|-----------------------------------|
/// | `res` | IntegerType (enum's discriminant) |
/// ```
///
/// # Verification
///
/// - Operand must be an enum type.
/// - Result must be an integer type matching the enum's discriminant type.
#[pliron_op(
    name = "mir.get_discriminant",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirGetDiscriminantOp;

impl MirGetDiscriminantOp {
    /// Create a new MirGetDiscriminantOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirGetDiscriminantOp { op }
    }
}

impl Verify for MirGetDiscriminantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "MirGetDiscriminantOp expects exactly one operand and one result, got {} operand(s) and {} result(s)",
                op.get_num_operands(),
                op.get_num_results()
            );
        }

        // Operand must be an enum type
        let operand = op.get_operand(0);
        let operand_ty = operand.get_type(ctx);
        let operand_ty_obj = operand_ty.deref(ctx);

        let enum_ty = match operand_ty_obj.downcast_ref::<MirEnumType>() {
            Some(et) => et,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirGetDiscriminantOp operand must be an enum type"
                );
            }
        };
        if !enum_ty.variant_inhabited.iter().any(|value| *value != 0) {
            return verify_err!(
                op.loc(),
                "MirGetDiscriminantOp cannot inspect an uninhabited enum"
            );
        }

        // Result must match the enum's discriminant type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);

        if result_ty != enum_ty.discriminant_type() {
            return verify_err!(
                op.loc(),
                "MirGetDiscriminantOp result type must match enum discriminant type. Expected: {}, Actual: {}",
                enum_ty.discriminant_type().disp(ctx),
                result_ty.disp(ctx)
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirSetDiscriminantOp
// ============================================================================

/// MIR set discriminant operation.
///
/// Selects an enum variant at the memory location pointed to by its operand.
/// This is the device-side lowering of MIR's
/// `StatementKind::SetDiscriminant`.
///
/// # Operands
///
/// ```text
/// | Name            | Type                              |
/// |-----------------|-----------------------------------|
/// | `enum_ptr` | Pointer to MirEnumType |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name                             | Type               | Description             |
/// |----------------------------------|--------------------|-------------------------|
/// | `set_discriminant_variant_index` | `VariantIndexAttr` | Variant to make active. |
/// | `set_discriminant_enum_ty`       | `TypeAttr`         | The enum being written. |
/// ```
///
/// Why carry the enum type when the operand already points to it? Lowering
/// runs after the operand becomes an opaque `llvm.ptr`, and operand type
/// HISTORY breaks across value-forwarded kind-only casts. The attribute
/// rides the op, so the tag/niche layout always comes from the right enum:
///
/// ```text
/// build time:  ptr: mir.ptr<E, mut> ──► enum_ty attr = E   (stamped once)
/// lowering:    tag offset / carrier bits from enum_ty      (no history)
/// ```
///
/// # Results
///
/// None.
///
/// # Verification
///
/// - First operand must be a `MirPtrType` pointing to a `MirEnumType`.
/// - `set_discriminant_enum_ty` must be present and equal that pointee.
/// - The target variant must exist and be inhabited.
#[pliron_op(
    name = "mir.set_discriminant",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<0>],
    attributes = (
        set_discriminant_variant_index: VariantIndexAttr,
        set_discriminant_enum_ty: pliron::builtin::attributes::TypeAttr
    )
)]
pub struct MirSetDiscriminantOp;

impl MirSetDiscriminantOp {
    /// Create a new MirSetDiscriminantOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirSetDiscriminantOp { op }
    }
}

impl Verify for MirSetDiscriminantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Do not rely on interface-verifier ordering here: malformed parsed IR
        // must produce a diagnostic rather than letting get_operand(0) panic.
        if op.get_num_operands() != 1 {
            return verify_err!(
                op.loc(),
                "MirSetDiscriminantOp expects exactly one operand, got {}",
                op.get_num_operands()
            );
        }

        // The operand must be a pointer to an enum type. Keep all checks in
        // this borrow so returned references do not outlive the guard.
        let ptr_operand = op.get_operand(0);
        let ptr_ty = ptr_operand.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);

        match ptr_ty_obj.downcast_ref::<crate::types::MirPtrType>() {
            Some(ptr_type) => {
                if !ptr_type.is_mutable {
                    return verify_err!(
                        op.loc(),
                        "MirSetDiscriminantOp requires a mutable enum pointer"
                    );
                }
                let pointee = ptr_type.pointee.deref(ctx);
                match pointee.downcast_ref::<MirEnumType>() {
                    Some(enum_ty) => {
                        let Some(enum_ty_attr) = self.get_attr_set_discriminant_enum_ty(ctx) else {
                            return verify_err!(
                                op.loc(),
                                "MirSetDiscriminantOp missing set_discriminant_enum_ty"
                            );
                        };
                        if enum_ty_attr.get_type(ctx) != ptr_type.pointee {
                            return verify_err!(
                                op.loc(),
                                "MirSetDiscriminantOp enum_ty attribute must equal the \
                                 operand pointee"
                            );
                        }
                        let Some(target) = self.get_attr_set_discriminant_variant_index(ctx) else {
                            return verify_err!(
                                op.loc(),
                                "MirSetDiscriminantOp missing set_discriminant_variant_index"
                            );
                        };
                        let target = target.0 as usize;
                        if target >= enum_ty.variant_count() {
                            return verify_err!(
                                op.loc(),
                                "MirSetDiscriminantOp target variant {} is out of bounds",
                                target
                            );
                        }
                        if enum_ty.variant_is_inhabited(target) != Some(true) {
                            return verify_err!(
                                op.loc(),
                                "MirSetDiscriminantOp cannot select uninhabited variant {}",
                                target
                            );
                        }
                        Ok(())
                    }
                    None => verify_err!(
                        op.loc(),
                        "MirSetDiscriminantOp pointer must point to an enum type"
                    ),
                }
            }
            None => verify_err!(
                op.loc(),
                "MirSetDiscriminantOp first operand must be a pointer type"
            ),
        }
    }
}

// ============================================================================
// MirEnumPayloadOp
// ============================================================================

/// MIR enum payload extraction operation.
///
/// Extracts a payload field from an enum variant.
/// This operation is "unsafe" in the sense that the caller must ensure
/// the enum value actually has the specified variant (via discriminant check).
///
/// # Operands
///
/// ```text
/// | Name    | Type        |
/// |---------|-------------|
/// | `value` | MirEnumType |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name                   | Type        | Description                     |
/// |------------------------|-------------|---------------------------------|
/// | `payload_variant_index`| VariantIndexAttr | Variant to extract from         |
/// | `payload_field_index`  | FieldIndexAttr   | Field within variant to extract |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                    |
/// |-------|-------------------------|
/// | `res` | Type of extracted field |
/// ```
///
/// # Verification
///
/// - Operand must be an enum type.
/// - `variant_index` must be valid for the enum.
/// - `field_index` must be valid for the variant.
/// - Result type must match the field type.
#[pliron_op(
    name = "mir.enum_payload",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (payload_variant_index: VariantIndexAttr, payload_field_index: FieldIndexAttr)
)]
pub struct MirEnumPayloadOp;

impl MirEnumPayloadOp {
    /// Create a new MirEnumPayloadOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirEnumPayloadOp { op }
    }
}

impl Verify for MirEnumPayloadOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "MirEnumPayloadOp expects exactly one operand and one result, got {} operand(s) and {} result(s)",
                op.get_num_operands(),
                op.get_num_results()
            );
        }

        // Operand must be an enum type
        let operand = op.get_operand(0);
        let operand_ty = operand.get_type(ctx);
        let operand_ty_obj = operand_ty.deref(ctx);

        let enum_ty = match operand_ty_obj.downcast_ref::<MirEnumType>() {
            Some(et) => et,
            None => {
                return verify_err!(op.loc(), "MirEnumPayloadOp operand must be an enum type");
            }
        };

        // Get variant index
        let variant_idx = match self.get_attr_payload_variant_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirEnumPayloadOp missing payload_variant_index attribute"
                );
            }
        };

        // Get the variant
        let variant = match enum_ty.get_variant(variant_idx) {
            Some(v) => v,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirEnumPayloadOp variant_index {} out of bounds for enum '{}' with {} variants",
                    variant_idx,
                    enum_ty.name(),
                    enum_ty.variant_count()
                );
            }
        };
        if enum_ty.variant_is_inhabited(variant_idx) != Some(true) {
            return verify_err!(
                op.loc(),
                "MirEnumPayloadOp cannot extract a field from uninhabited variant {}",
                variant_idx
            );
        }

        // Get field index
        let field_idx = match self.get_attr_payload_field_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirEnumPayloadOp missing payload_field_index attribute"
                );
            }
        };

        // Validate field index
        if field_idx >= variant.field_types.len() {
            return verify_err!(
                op.loc(),
                "MirEnumPayloadOp field_index {} out of bounds for variant '{}' with {} fields",
                field_idx,
                variant.name,
                variant.field_types.len()
            );
        }

        // Result type must match field type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let expected_ty = variant.field_types[field_idx];

        if result_ty != expected_ty {
            return verify_err!(
                op.loc(),
                "MirEnumPayloadOp result type mismatch. Expected: {}, Actual: {}",
                expected_ty.disp(ctx),
                result_ty.disp(ctx)
            );
        }

        Ok(())
    }
}

/// Register enum operations into the given context.
pub fn register(ctx: &mut Context) {
    MirConstructEnumOp::register(ctx);
    MirGetDiscriminantOp::register(ctx);
    MirSetDiscriminantOp::register(ctx);
    MirEnumPayloadOp::register(ctx);
}
