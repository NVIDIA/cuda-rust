/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel-boundary detection, transparent-scalar ABI, and function-type
//! conversion (the boundary ABI: slice flattening, struct flattening,
//! packed shared-pointer rewrites).

use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirFP16Type, MirPtrType, MirSliceType, MirStructType,
    MirTupleType,
};
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, type_cast};

use super::{
    StructLayoutInfo, build_struct_slot_map, convert_type, is_zero_sized_type,
    llvm_packed_struct_contains_pointer_in_address_space, llvm_type_size_align,
};
use crate::convert::target_stable_storage::{StorageRewriteOptions, target_stable_storage_type};

// =============================================================================
// Kernel-Boundary Detection
// =============================================================================

/// Identifier of the attribute that marks a `MirFuncOp` / `llvm.func` as a
/// GPU kernel entry point.
///
/// Kept as a function (rather than a `const`) because pliron `Identifier`
/// construction needs the `try_into()` fallible path.
fn gpu_kernel_attr() -> pliron::identifier::Identifier {
    "gpu_kernel".try_into().expect("static identifier")
}

/// Returns `true` when `op` carries the `gpu_kernel` attribute.
///
/// The kernel-entry ABI differs from internal device-function ABI: at
/// kernel boundaries, aggregate parameters (structs, closures) are passed
/// as a single byval value to match what the host pushes via
/// `cuLaunchKernel`. Internal call sites still flatten aggregates the
/// same way they always did. This helper is the single source of truth
/// for that branch and is consumed by both [`convert_function_type`] and
/// the entry-block prologue in `lowering.rs`.
pub fn is_kernel_func(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&gpu_kernel_attr())
        .is_some()
}

/// Return the declaration index and type of the single non-ZST field of a
/// rustc-proven transparent scalar struct.
///
/// The importer marks the outer ABI from rustc rather than inferring it from
/// source field count. We still validate the MIR shape here so malformed or
/// hand-written dialect input cannot turn an arbitrary aggregate into a scalar
/// ABI value.
fn transparent_scalar_field_with_index(
    ctx: &mut Context,
    struct_ty: TypeHandle,
) -> Result<(usize, TypeHandle), anyhow::Error> {
    let (name, field_types, mem_to_decl, is_transparent_scalar) = {
        let ty_ref = struct_ty.deref(ctx);
        let s = ty_ref
            .downcast_ref::<MirStructType>()
            .ok_or_else(|| anyhow::anyhow!("transparent scalar ABI requires a MirStructType"))?;
        (
            s.name.clone(),
            s.field_types.clone(),
            s.memory_order(),
            s.is_transparent_scalar(),
        )
    };

    if !is_transparent_scalar {
        return Err(anyhow::anyhow!(
            "struct `{}` is not marked as a transparent scalar",
            name
        ));
    }

    let mut scalar_field = None;
    for decl_idx in mem_to_decl {
        let field_ty = field_types[decl_idx];
        let converted = convert_type(ctx, field_ty)?;
        if is_zero_sized_type(ctx, converted) {
            continue;
        }
        if scalar_field.replace((decl_idx, field_ty)).is_some() {
            return Err(anyhow::anyhow!(
                "transparent scalar struct `{}` has more than one non-ZST field",
                name
            ));
        }
    }

    scalar_field
        .ok_or_else(|| anyhow::anyhow!("transparent scalar struct `{}` has no non-ZST field", name))
}

/// Return the single non-ZST field of a rustc-proven transparent scalar struct.
pub(crate) fn transparent_scalar_field(
    ctx: &mut Context,
    struct_ty: TypeHandle,
) -> Result<TypeHandle, anyhow::Error> {
    Ok(transparent_scalar_field_with_index(ctx, struct_ty)?.1)
}

/// One aggregate layer traversed when a transparent scalar wrapper crosses an
/// ABI boundary. `field_slot` is the LLVM struct slot containing the next
/// nested wrapper or the final scalar.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransparentScalarLayer {
    pub llvm_struct_ty: TypeHandle,
    pub field_slot: u32,
}

/// Complete ABI projection for a rustc-proven transparent scalar wrapper.
///
/// `layers` are ordered outermost to innermost. A return lowers by extracting
/// those slots in order; a call result is rebuilt by inserting the scalar into
/// the same layers in reverse order.
#[derive(Clone, Debug)]
pub(crate) struct TransparentScalarAbiInfo {
    pub scalar_ty: TypeHandle,
    pub layers: Vec<TransparentScalarLayer>,
}

/// Build the scalar ABI projection, including the exact LLVM slot used at every
/// wrapper layer.
///
/// Slot indices come from [`build_struct_slot_map`], so ZST markers, explicit
/// padding, and rustc memory order cannot make the return/call paths disagree
/// with ordinary aggregate conversion.
pub(crate) fn transparent_scalar_abi_info(
    ctx: &mut Context,
    struct_ty: TypeHandle,
) -> Result<TransparentScalarAbiInfo, anyhow::Error> {
    let mut current = struct_ty;
    let mut layers = Vec::new();

    loop {
        let (decl_idx, field_ty) = transparent_scalar_field_with_index(ctx, current)?;
        let layout = {
            let ty_ref = current.deref(ctx);
            let s = ty_ref.downcast_ref::<MirStructType>().ok_or_else(|| {
                anyhow::anyhow!("transparent scalar ABI requires a MirStructType")
            })?;
            StructLayoutInfo::of_struct(s)
        };
        let map = build_struct_slot_map(ctx, &layout)?;
        let field_slot = map.decl_to_llvm[decl_idx].ok_or_else(|| {
            anyhow::anyhow!(
                "transparent scalar field {} unexpectedly lowered as a ZST",
                decl_idx
            )
        })?;
        layers.push(TransparentScalarLayer {
            llvm_struct_ty: map.llvm_struct_ty,
            field_slot,
        });

        let nested_transparent = {
            let field_ref = field_ty.deref(ctx);
            field_ref
                .downcast_ref::<MirStructType>()
                .is_some_and(MirStructType::is_transparent_scalar)
        };
        if nested_transparent {
            current = field_ty;
            continue;
        }

        let scalar_ty = convert_type(ctx, field_ty)?;
        if is_zero_sized_type(ctx, scalar_ty) {
            return Err(anyhow::anyhow!(
                "transparent scalar ABI resolved to a zero-sized field"
            ));
        }
        return Ok(TransparentScalarAbiInfo { scalar_ty, layers });
    }
}

/// LLVM ABI type for a rustc-proven transparent scalar struct.
///
/// Transparent wrappers can nest (`Outer(Inner(u32))`). rustc still reports
/// the outer ADT as one scalar, so recurse through transparent scalar fields
/// until reaching the actual scalar/pointer representation.
pub(crate) fn transparent_scalar_llvm_type(
    ctx: &mut Context,
    struct_ty: TypeHandle,
) -> Result<TypeHandle, anyhow::Error> {
    Ok(transparent_scalar_abi_info(ctx, struct_ty)?.scalar_ty)
}

/// Target-stable ABI projection for packed aggregates containing AS3 leaves
/// across the internal device return boundary.
///
/// The function body continues to use the semantic packed LLVM struct with
/// shared pointers. Only the physical return value recursively replaces AS3
/// leaves with generic pointers, whose width is stable across the legacy/PTX
/// and modern NVVM data layouts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedSharedInternalAbiInfo {
    pub semantic_ty: TypeHandle,
    pub storage_ty: TypeHandle,
}

/// Maximum number of shared-pointer leaves introduced by fixed-array expansion
/// in one packed-AS3 internal return carrier.
///
/// Struct/tuple nesting and direct pointer leaves remain proportional to source
/// structure. Arrays can encode an arbitrarily large number of per-element
/// extract/cast/insert sequences compactly, so only array-expanded AS3 leaves
/// count against this code-shape budget.
pub(crate) const MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES: u64 = 16;

/// Whether a MIR type converts to a zero-sized LLVM type.
///
/// MIR-side mirror of [`is_zero_sized_type`]: zero-length arrays, arrays of
/// zero-sized elements, and structs/tuples whose fields are all zero-sized
/// (including empty ones such as `PhantomData`) vanish at the LLVM level and
/// must not affect ABI-lane classification.
fn mir_type_is_zero_sized(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        return array_ty.size() == 0 || mir_type_is_zero_sized(ctx, array_ty.element_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return struct_ty
            .field_types
            .iter()
            .all(|field| mir_type_is_zero_sized(ctx, *field));
    }
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty
            .get_types()
            .iter()
            .all(|field| mir_type_is_zero_sized(ctx, *field));
    }
    false
}

/// Whether a MIR value shape belongs to the recursive packed-AS3 ABI lane.
///
/// Structs, tuples, and fixed arrays may nest recursively and may contain any
/// number of scalar pointer leaves. Zero-sized fields are skipped, matching the
/// post-conversion field scan this predicate replaced. The array-specific
/// expansion budget is enforced after conversion from the storage rewrite's
/// exact AS3 leaf count. Vectors and unrelated aggregate kinds remain
/// deliberately fail-closed.
fn packed_shared_internal_abi_mir_shape_is_supported(ctx: &Context, mir_ty: TypeHandle) -> bool {
    let children = {
        let ty_ref = mir_ty.deref(ctx);
        if ty_ref.is::<IntegerType>()
            || ty_ref.is::<MirFP16Type>()
            || ty_ref.is::<llvm_types::HalfType>()
            || ty_ref.is::<FP32Type>()
            || ty_ref.is::<FP64Type>()
            || ty_ref.is::<MirPtrType>()
            || ty_ref.is::<llvm_types::PointerType>()
        {
            return true;
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            Some(struct_ty.field_types.clone())
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            Some(tuple_ty.get_types().to_vec())
        } else {
            ty_ref
                .downcast_ref::<MirArrayType>()
                .map(|array_ty| vec![array_ty.element_type()])
        }
    };

    children.is_some_and(|children| {
        children.into_iter().all(|child| {
            mir_type_is_zero_sized(ctx, child)
                || packed_shared_internal_abi_mir_shape_is_supported(ctx, child)
        })
    })
}

/// Recognize a recursive packed-AS3 internal ABI shape.
///
/// The root must remain a byte-faithful packed struct. Nested structs/tuples,
/// multiple AS3 leaves, and bounded fixed arrays are admitted. Vectors and
/// unrelated aggregate kinds remain out of scope. The target-stable storage
/// utility owns the recursive AS3 -> generic rewrite and this classifier only
/// decides which semantic shapes may use it.
pub(crate) fn packed_shared_internal_abi_info(
    ctx: &mut Context,
    mir_ty: TypeHandle,
) -> Result<Option<PackedSharedInternalAbiInfo>, anyhow::Error> {
    let layout = {
        let ty_ref = mir_ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() else {
            return Ok(None);
        };
        StructLayoutInfo::of_struct(struct_ty)
    };

    if !packed_shared_internal_abi_mir_shape_is_supported(ctx, mir_ty) {
        return Ok(None);
    }

    let map = build_struct_slot_map(ctx, &layout)?;
    if !map.by_value_layout_faithful {
        return Ok(None);
    }
    let is_packed = map
        .llvm_struct_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::StructType>()
        .is_some_and(|struct_ty| struct_ty.layout() == llvm_types::StructLayout::Packed);
    if !is_packed {
        return Ok(None);
    }

    let rewrite = target_stable_storage_type(
        ctx,
        map.llvm_struct_ty,
        StorageRewriteOptions {
            canonicalize_bool: false,
        },
        "packed shared internal ABI",
    )?;
    if rewrite.shared_pointer_leaves == 0
        || rewrite.array_shared_pointer_leaves > MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES
    {
        return Ok(None);
    }
    let Some((storage_size, _)) = llvm_type_size_align(ctx, rewrite.ty) else {
        return Ok(None);
    };
    if layout.total_size > 0 && storage_size != layout.total_size {
        return Ok(None);
    }

    Ok(Some(PackedSharedInternalAbiInfo {
        semantic_ty: map.llvm_struct_ty,
        storage_ty: rewrite.ty,
    }))
}

/// Convert a type that crosses a function boundary as one LLVM value.
///
/// A struct with a natural-layout divergence is legal by value only when
/// [`build_struct_slot_map`] proved that a sequential packed LLVM struct
/// reproduces rustc's offsets and size. This keeps overlapping/union-like
/// legacy struct models fail-closed while allowing real `repr(packed)` values.
/// Packed values containing shared-memory pointers remain target-dependent
/// because AS3 pointer width differs between modern NVVM and PTX/legacy modes.
fn convert_by_value_abi_type(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    role: &str,
) -> Result<TypeHandle, anyhow::Error> {
    let layout = {
        let ty_ref = mir_ty.deref(ctx);
        ty_ref
            .downcast_ref::<MirStructType>()
            .map(StructLayoutInfo::of_struct)
    };
    let llvm_ty = if let Some(layout) = layout {
        let map = build_struct_slot_map(ctx, &layout)?;
        if !map.by_value_layout_faithful {
            return Err(anyhow::anyhow!(
                "{} has a rustc struct layout that cannot be represented by an LLVM struct value",
                role
            ));
        }
        map.llvm_struct_ty
    } else {
        convert_type(ctx, mir_ty)?
    };

    if llvm_packed_struct_contains_pointer_in_address_space(
        ctx,
        llvm_ty,
        llvm_types::address_space::SHARED,
    ) {
        return Err(anyhow::anyhow!(
            "{} contains a packed aggregate with a target-dependent shared-memory pointer",
            role
        ));
    }

    Ok(llvm_ty)
}

/// A grid-constant reference transports its pointee's bytes by value. Apply
/// the ordinary by-value layout proof, then require a target-stable storage
/// representation: AS3 pointers shrink from 64 to 32 bits in modern NVVM,
/// including inside arrays and naturally aligned structs. Keeping such a
/// pointee would change the launch packet while field addresses still use
/// rustc's offsets. A generic pointer retains the host's 64-bit storage.
pub(crate) fn convert_grid_constant_storage_type(
    ctx: &mut Context,
    mir_ty: TypeHandle,
) -> Result<TypeHandle, anyhow::Error> {
    let llvm_ty = convert_by_value_abi_type(ctx, mir_ty, "grid-constant parameter storage")?;
    if super::llvm_type_contains_pointer_in_address_space(
        ctx,
        llvm_ty,
        llvm_types::address_space::SHARED,
    ) {
        return Err(anyhow::anyhow!(
            "grid-constant parameter storage contains a target-dependent shared-memory pointer; use a payload with target-stable pointer storage"
        ));
    }
    Ok(llvm_ty)
}

/// Convert a MIR function type to an LLVM function type.
///
/// This handles the ABI-level transformations required for GPU kernels.
/// The transformations ensure that the generated LLVM IR matches the
/// C ABI expected by the CUDA runtime.
///
/// # ABI Transformations
///
/// ## Argument Flattening
///
/// Aggregate types are flattened to primitive types:
///
/// ```text
/// MIR:  fn kernel(slice: &[f32], point: Point)
/// LLVM: fn internal_fn(ptr: !ptr, len: i64, x: f32, y: f32)
/// ```
///
/// | MIR Argument            | Internal call ABI       | Kernel-entry ABI       |
/// |-------------------------|-------------------------|------------------------|
/// | `&[T]`                  | `(ptr, i64)`            | `(ptr, i64)`           |
/// | `DisjointSlice<T>`      | `(ptr, i64)`            | `(ptr, i64)`           |
/// | `struct { a: A, b: B }` | `(a: A', b: B')`        | one byval `{A', B'}`   |
/// | `repr(transparent)` scalar ADT | underlying scalar      | underlying scalar      |
/// | closure with N captures | N separate field args   | one byval struct       |
/// | Other                   | Converted type          | Converted type         |
///
/// Slices keep their `(ptr, len)` flattening on both sides because the
/// host-side launch helpers push the pointer and length as two driver
/// args. Structs and closures are unflattened only at kernel boundaries
/// because the host pushes them as a single scalar — see
/// `cuda_host::push_kernel_scalar`. The exception is a rustc-proven
/// `#[repr(transparent)]` `ValueAbi::Scalar` struct: its single non-ZST
/// field is the kernel parameter, matching the source type's transparent ABI.
/// Internal device-side call sites stay flattened: caller and callee are both
/// inside this backend, so the ABI is private and there is no host to disagree
/// with.
///
/// ## Return Type Handling
///
/// - Empty tuple `()` becomes `void`
/// - Empty struct `struct {}` becomes `void`
/// - Rustc-proven `repr(transparent)` scalar ADTs return the underlying scalar
/// - Other types are converted normally
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `func_type` - The MIR function type to convert
/// * `is_kernel_entry` - When `true`, treat aggregate (non-slice) params
///   as single byval values to match the host-side push ABI. When `false`,
///   keep the existing internal device-fn ABI that flattens struct fields
///   into individual scalars.
///
/// # Returns
///
/// The equivalent LLVM function type with ABI transformations applied.
///
/// # Example
///
/// ```text
/// MIR:  fn foo(a: &[f32], b: i32) -> f32
/// LLVM: fn foo(ptr, i64, i32) -> f32
///
/// MIR:  fn bar() -> ()
/// LLVM: fn bar() -> void
/// ```
///
/// # Note
///
/// At internal device-function boundaries the struct flattening must be
/// reversed in the entry block. At kernel-entry boundaries ordinary structs
/// arrive as one byval aggregate, while rustc-proven transparent scalar
/// wrappers arrive as their single non-ZST scalar field and must be rebuilt.
/// See `lowering.rs::build_entry_prologue` for the reconstruction paths.
pub fn convert_function_type(
    ctx: &mut Context,
    func_type: pliron::r#type::TypedHandle<FunctionType>,
    is_kernel_entry: bool,
) -> Result<pliron::r#type::TypedHandle<llvm_types::FuncType>, anyhow::Error> {
    // Extract input/output types before mutating context
    let (inputs_ptr, results_ptr) = {
        let func_ty_ref = func_type.deref(ctx);
        let interface = type_cast::<dyn FunctionTypeInterface>(&*func_ty_ref)
            .ok_or_else(|| anyhow::anyhow!("Type does not implement FunctionTypeInterface"))?;
        (interface.arg_types(), interface.res_types())
    };

    // Convert inputs, flattening slice/struct types for ABI compatibility.
    // Slices flatten on both ABIs; structs flatten only on the internal
    // device-fn ABI.
    let mut inputs = Vec::new();
    let inputs_vec: Vec<_> = inputs_ptr.to_vec();

    for t in inputs_vec {
        // Determine what kind of flattening this type needs
        // Extract all info first, then drop the borrow
        enum FlattenKind {
            /// `{ ptr, len }`, then one parameter per index-space layout field.
            Slice {
                space_tys: Vec<TypeHandle>,
            },
            Struct {
                field_types: Vec<TypeHandle>,
                mem_to_decl: Vec<usize>,
            },
            TransparentScalar(TypeHandle),
            None,
        }

        let flatten_kind = {
            let ty_ref = t.deref(ctx);
            if ty_ref.is::<MirSliceType>() {
                FlattenKind::Slice {
                    space_tys: Vec::new(),
                }
            } else if let Some(slice_ty) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
                FlattenKind::Slice {
                    space_tys: slice_ty.space_tys.clone(),
                }
            } else if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
                if is_kernel_entry && struct_ty.is_transparent_scalar() {
                    // rustc proves that this `repr(transparent)` ADT has a
                    // scalar ABI. Emit exactly its one non-ZST field as the
                    // kernel parameter instead of an aggregate `.b8[]` param.
                    FlattenKind::TransparentScalar(t)
                } else if is_kernel_entry {
                    // Ordinary kernel-boundary structs remain intact so the
                    // host's single `push_kernel_scalar(&closure)` push
                    // matches a single aggregate .param entry.
                    FlattenKind::None
                } else {
                    FlattenKind::Struct {
                        field_types: struct_ty.field_types.clone(),
                        mem_to_decl: struct_ty.memory_order(),
                    }
                }
            } else {
                FlattenKind::None
            }
        };

        match flatten_kind {
            FlattenKind::Slice { space_tys } => {
                let ptr_ty = llvm_types::PointerType::get_generic(ctx);
                let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
                inputs.push(ptr_ty.into());
                inputs.push(len_ty.into());
                for space_ty in space_tys {
                    let converted = convert_type(ctx, space_ty)?;
                    // A zero-sized index-space field contributes no parameter,
                    // as NVPTX has no empty `.param`.
                    if !is_zero_sized_type(ctx, converted) {
                        inputs.push(converted);
                    }
                }
            }
            FlattenKind::Struct {
                field_types,
                mem_to_decl,
            } => {
                // Flatten in MEMORY ORDER to match struct layout
                for mem_idx in 0..field_types.len() {
                    let decl_idx = mem_to_decl[mem_idx];
                    let converted = convert_by_value_abi_type(
                        ctx,
                        field_types[decl_idx],
                        "by-value function argument",
                    )?;
                    // Skip ZST fields - NVPTX can't handle empty params
                    if !is_zero_sized_type(ctx, converted) {
                        inputs.push(converted);
                    }
                }
            }
            FlattenKind::TransparentScalar(struct_ty) => {
                inputs.push(transparent_scalar_llvm_type(ctx, struct_ty)?);
            }
            FlattenKind::None => {
                let converted = convert_by_value_abi_type(ctx, t, "by-value function argument")?;
                // Skip ZST args - NVPTX can't handle empty params
                if !is_zero_sized_type(ctx, converted) {
                    inputs.push(converted);
                }
            }
        }
    }

    // Convert return type. A rustc-proven transparent scalar wrapper uses
    // the underlying scalar at the function ABI boundary, while its body
    // continues to use the ordinary converted aggregate representation.
    let ret_ty = if results_ptr.is_empty() {
        llvm_types::VoidType::get(ctx).into()
    } else {
        let mir_ret_ty = results_ptr[0];
        let is_transparent_scalar = {
            let ty_ref = mir_ret_ty.deref(ctx);
            ty_ref
                .downcast_ref::<MirStructType>()
                .is_some_and(MirStructType::is_transparent_scalar)
        };
        let ty = if is_transparent_scalar {
            transparent_scalar_llvm_type(ctx, mir_ret_ty)?
        } else if !is_kernel_entry {
            match packed_shared_internal_abi_info(ctx, mir_ret_ty)? {
                Some(abi) => abi.storage_ty,
                None => convert_by_value_abi_type(ctx, mir_ret_ty, "by-value function return")?,
            }
        } else {
            convert_by_value_abi_type(ctx, mir_ret_ty, "by-value function return")?
        };
        // Check if zero-sized (empty struct or struct with only ZST fields).
        if is_zero_sized_type(ctx, ty) {
            llvm_types::VoidType::get(ctx).into()
        } else {
            ty
        }
    };

    Ok(llvm_types::FuncType::get(ctx, ret_ty, inputs, false))
}

#[cfg(test)]
mod tests {
    use super::super::llvm_type_contains_pointer_in_address_space;
    use super::super::test_support::{make_ctx, mir_uint, struct_fields, transparent_u32};
    use super::*;
    use dialect_mir::types::StructAbiKind;

    #[test]
    fn transparent_scalar_return_type_uses_underlying_scalar() {
        let mut ctx = make_ctx();
        let wrapper = transparent_u32(&mut ctx, "Scalar");
        let func_ty = FunctionType::get(&ctx, vec![], vec![wrapper]);

        let lowered = convert_function_type(&mut ctx, func_ty, false).unwrap();
        let result_ty = lowered.deref(&ctx).result_type();
        let result_ty_ref = result_ty.deref(&ctx);
        let integer = result_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("transparent u32 return must lower to an integer");
        assert_eq!(integer.width(), 32);
    }

    #[test]
    fn nested_transparent_scalar_abi_records_each_rebuild_layer() {
        let mut ctx = make_ctx();
        let inner = transparent_u32(&mut ctx, "Inner");
        let outer: TypeHandle = MirStructType::get_with_full_layout_and_abi(
            &mut ctx,
            "Outer".into(),
            vec!["inner".into()],
            vec![inner],
            vec![0],
            vec![0],
            4,
            4,
            StructAbiKind::TransparentScalar,
        )
        .into();

        let info = transparent_scalar_abi_info(&mut ctx, outer).unwrap();
        let scalar_ty_ref = info.scalar_ty.deref(&ctx);
        let integer = scalar_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("nested transparent wrapper must resolve to u32");
        assert_eq!(integer.width(), 32);
        assert_eq!(info.layers.len(), 2);
        assert_eq!(info.layers[0].field_slot, 0);
        assert_eq!(info.layers[1].field_slot, 0);
    }

    #[test]
    fn ordinary_one_field_return_remains_aggregate() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let ordinary: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Ordinary".into(),
            vec!["value".into()],
            vec![u32_ty],
            vec![0],
            vec![0],
            4,
            4,
        )
        .into();
        let func_ty = FunctionType::get(&ctx, vec![], vec![ordinary]);

        let lowered = convert_function_type(&mut ctx, func_ty, false).unwrap();
        assert!(
            lowered
                .deref(&ctx)
                .result_type()
                .deref(&ctx)
                .is::<llvm_types::StructType>(),
            "ordinary one-field structs must not be scalarized"
        );
    }

    #[test]
    fn packed_shared_pointer_predicate_ignores_unpacked_direct_pointer() {
        let ctx = make_ctx();
        let shared: TypeHandle =
            llvm_types::PointerType::get(&ctx, llvm_types::address_space::SHARED).into();
        let unpacked: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (vec![shared], llvm_types::StructLayout::Unpacked),
        )
        .into();
        assert!(
            !llvm_packed_struct_contains_pointer_in_address_space(
                &ctx,
                unpacked,
                llvm_types::address_space::SHARED,
            ),
            "a direct AS3 pointer in an unpacked aggregate is not a packed physical-image hazard"
        );

        let packed: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (vec![shared], llvm_types::StructLayout::Packed),
        )
        .into();
        assert!(
            llvm_packed_struct_contains_pointer_in_address_space(
                &ctx,
                packed,
                llvm_types::address_space::SHARED,
            ),
            "an AS3 pointer under a packed struct must be rejected by by-value paths"
        );
    }

    #[test]
    fn packed_shared_internal_abi_genericizes_one_direct_shared_pointer() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedShared".into(),
            vec!["tag".into(), "ptr".into()],
            vec![tag, shared],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, packed)
            .expect("ABI classification must succeed")
            .expect("one direct packed AS3 pointer must use the internal carrier");
        assert_eq!(llvm_type_size_align(&ctx, abi.semantic_ty), Some((9, 1)));
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((9, 1)));
        assert!(llvm_packed_struct_contains_pointer_in_address_space(
            &ctx,
            abi.semantic_ty,
            llvm_types::address_space::SHARED,
        ));
        assert!(!llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::SHARED,
        ));
        assert!(llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::GENERIC,
        ));
    }

    #[test]
    fn packed_shared_internal_abi_genericizes_multiple_direct_shared_pointers() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedPair".into(),
            vec!["tag".into(), "left".into(), "right".into()],
            vec![tag, shared, shared],
            vec![0, 1, 2],
            vec![0, 1, 9],
            17,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, packed)
            .expect("ABI classification must succeed")
            .expect("multiple direct AS3 leaves must use the recursive internal carrier");
        assert_eq!(llvm_type_size_align(&ctx, abi.semantic_ty), Some((17, 1)));
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((17, 1)));
        assert!(llvm_packed_struct_contains_pointer_in_address_space(
            &ctx,
            abi.semantic_ty,
            llvm_types::address_space::SHARED,
        ));
        assert!(!llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::SHARED,
        ));

        let fields = struct_fields(&ctx, abi.storage_ty);
        assert_eq!(fields.len(), 3);
        for field in &fields[1..] {
            let field_ref = field.deref(&ctx);
            let pointer = field_ref
                .downcast_ref::<llvm_types::PointerType>()
                .expect("both pointer leaves must remain pointer-typed");
            assert_eq!(pointer.address_space(), llvm_types::address_space::GENERIC);
        }
    }

    #[test]
    fn packed_shared_internal_abi_genericizes_nested_struct_shared_pointers() {
        let mut ctx = make_ctx();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let inner: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "InnerSharedPair".into(),
            vec!["left".into(), "right".into()],
            vec![shared, shared],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let tag = mir_uint(&mut ctx, 8);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedNestedShared".into(),
            vec!["tag".into(), "inner".into()],
            vec![tag, inner],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, outer)
            .expect("classification must not error")
            .expect("nested AS3 leaves must use the recursive internal carrier");
        assert_eq!(llvm_type_size_align(&ctx, abi.semantic_ty), Some((17, 1)));
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((17, 1)));
        assert!(!llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::SHARED,
        ));

        let outer_fields = struct_fields(&ctx, abi.storage_ty);
        let inner_fields = struct_fields(&ctx, outer_fields[1]);
        assert_eq!(inner_fields.len(), 2);
        for field in inner_fields {
            let field_ref = field.deref(&ctx);
            let pointer = field_ref
                .downcast_ref::<llvm_types::PointerType>()
                .expect("nested shared leaves must remain pointer-typed");
            assert_eq!(pointer.address_space(), llvm_types::address_space::GENERIC);
        }
    }

    #[test]
    fn packed_shared_internal_abi_genericizes_nested_tuple_shared_pointer() {
        let mut ctx = make_ctx();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let word = mir_uint(&mut ctx, 32);
        let inner: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![shared, word],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let tag = mir_uint(&mut ctx, 8);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedNestedTupleShared".into(),
            vec!["tag".into(), "inner".into()],
            vec![tag, inner],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, outer)
            .expect("classification must not error")
            .expect("an AS3 leaf nested in a tuple must use the recursive carrier");
        let outer_fields = struct_fields(&ctx, abi.storage_ty);
        let inner_fields = struct_fields(&ctx, outer_fields[1]);
        let inner_field_ref = inner_fields[0].deref(&ctx);
        let pointer = inner_field_ref
            .downcast_ref::<llvm_types::PointerType>()
            .expect("nested tuple leaf must remain pointer-typed");
        assert_eq!(pointer.address_space(), llvm_types::address_space::GENERIC);
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((17, 1)));
    }

    #[test]
    fn packed_shared_internal_abi_skips_zero_sized_fields() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let byte = mir_uint(&mut ctx, 8);
        let marker: TypeHandle = MirArrayType::get(&mut ctx, byte, 0).into();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedZstMarker".into(),
            vec!["tag".into(), "marker".into(), "ptr".into()],
            vec![tag, marker, shared],
            vec![0, 1, 2],
            vec![0, 1, 1],
            9,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, packed)
            .expect("ABI classification must succeed")
            .expect("a zero-sized field must not knock the struct out of the ABI lane");
        assert_eq!(llvm_type_size_align(&ctx, abi.semantic_ty), Some((9, 1)));
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((9, 1)));
        assert!(!llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::SHARED,
        ));
    }

    #[test]
    fn packed_shared_internal_abi_genericizes_bounded_shared_pointer_array() {
        let mut ctx = make_ctx();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let shared_array: TypeHandle = MirArrayType::get(&mut ctx, shared, 2).into();
        let tag = mir_uint(&mut ctx, 8);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedArray".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, shared_array],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        let abi = packed_shared_internal_abi_info(&mut ctx, outer)
            .expect("classification must not error")
            .expect("a bounded AS3 array must use the target-stable internal carrier");
        assert_eq!(llvm_type_size_align(&ctx, abi.semantic_ty), Some((17, 1)));
        assert_eq!(llvm_type_size_align(&ctx, abi.storage_ty), Some((17, 1)));
        assert!(!llvm_type_contains_pointer_in_address_space(
            &ctx,
            abi.storage_ty,
            llvm_types::address_space::SHARED,
        ));

        let outer_fields = struct_fields(&ctx, abi.storage_ty);
        let array_ref = outer_fields[1].deref(&ctx);
        let array = array_ref
            .downcast_ref::<llvm_types::ArrayType>()
            .expect("bounded array field must remain an LLVM array");
        assert_eq!(array.size(), 2);
        let element = array.elem_type();
        let element_ref = element.deref(&ctx);
        let pointer = element_ref
            .downcast_ref::<llvm_types::PointerType>()
            .expect("array elements must remain pointer-typed");
        assert_eq!(pointer.address_space(), llvm_types::address_space::GENERIC);
    }

    #[test]
    fn packed_shared_internal_abi_accepts_array_rewrite_at_exact_bound() {
        let mut ctx = make_ctx();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let count = MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES;
        let shared_array: TypeHandle = MirArrayType::get(&mut ctx, shared, count).into();
        let tag = mir_uint(&mut ctx, 8);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedArrayAtBound".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, shared_array],
            vec![0, 1],
            vec![0, 1],
            1 + 8 * count,
            1,
        )
        .into();

        assert!(
            packed_shared_internal_abi_info(&mut ctx, outer)
                .expect("classification must not error")
                .is_some(),
            "the exact bounded-array rewrite limit must remain supported"
        );
    }

    #[test]
    fn packed_shared_internal_abi_rejects_array_rewrite_above_bound() {
        let mut ctx = make_ctx();
        let pointee = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let count = MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES + 1;
        let shared_array: TypeHandle = MirArrayType::get(&mut ctx, shared, count).into();
        let tag = mir_uint(&mut ctx, 8);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedArrayAboveBound".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, shared_array],
            vec![0, 1],
            vec![0, 1],
            1 + 8 * count,
            1,
        )
        .into();

        assert!(
            packed_shared_internal_abi_info(&mut ctx, outer)
                .expect("classification must not error")
                .is_none(),
            "array-expanded AS3 leaves above the explicit budget must fail closed"
        );
    }

    #[test]
    fn packed_shared_internal_abi_rejects_shared_pointer_vector() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let shared_pointer: TypeHandle =
            llvm_types::PointerType::get(&ctx, llvm_types::address_space::SHARED).into();
        let shared_vector: TypeHandle =
            llvm_types::VectorType::get(&ctx, shared_pointer, 2, llvm_types::VectorTypeKind::Fixed)
                .into();
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedVector".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, shared_vector],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        assert!(
            packed_shared_internal_abi_info(&mut ctx, outer)
                .expect("classification must not error")
                .is_none(),
            "shared-pointer vectors must remain outside the packed-AS3 internal ABI lane"
        );
    }
}
