/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM type printing.

use std::fmt::Write;

use pliron::{
    builtin::{
        type_interfaces::FunctionTypeInterface,
        types::{FP32Type, FP64Type, IntegerType},
    },
    r#type::TypeHandle,
};

use crate::types::{FuncType, HalfType, PointerType, StructLayout, StructType, VoidType};

use super::state::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    /// Recover the typed-pointer ABI carried by the name of a legacy NVVM
    /// floating-point atomic-add intrinsic. The internal IR intentionally uses
    /// opaque pointers, so declaration and call emission share this validation
    /// before restoring the scalar pointee required by LLVM 7.
    pub(super) fn legacy_nvvm_atomic_add_signature(
        &self,
        name: &str,
        function_type: &FuncType,
    ) -> Result<Option<(TypeHandle, u32)>, String> {
        if !self.legacy_typed_pointers() {
            return Ok(None);
        }
        let Some((width, address_space)) = super::names::legacy_nvvm_atomic_add_signature(name)
        else {
            return Ok(None);
        };

        let arguments = function_type.arg_types();
        let shape_matches = arguments.len() == 2
            && !function_type.is_var_arg()
            && function_type.result_type() == arguments[1]
            && arguments[0]
                .deref(self.ctx)
                .downcast_ref::<PointerType>()
                .is_some_and(|pointer| pointer.address_space() == address_space)
            && match width {
                32 => arguments[1].deref(self.ctx).is::<FP32Type>(),
                64 => arguments[1].deref(self.ctx).is::<FP64Type>(),
                _ => false,
            };
        if !shape_matches {
            return Err(format!(
                "legacy NVVM atomic-add intrinsic `@{name}` has an incompatible erased signature"
            ));
        }

        Ok(Some((arguments[1], address_space)))
    }

    pub(super) fn export_type(&self, ty: TypeHandle, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if self.legacy_typed_pointers() {
                self.export_canonical_pointer_type(addrspace, output);
            } else if addrspace != 0 {
                write!(output, "ptr addrspace({addrspace})").unwrap();
            } else {
                write!(output, "ptr").unwrap();
            }
        } else if ty_ref.is::<VoidType>() {
            write!(output, "void").unwrap();
        } else if ty_ref.is::<HalfType>() {
            write!(output, "half").unwrap();
        } else if ty_ref.is::<FP32Type>() {
            write!(output, "float").unwrap();
        } else if ty_ref.is::<FP64Type>() {
            write!(output, "double").unwrap();
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            let (open, close) = match struct_ty.layout() {
                StructLayout::Packed => ("<{ ", " }>"),
                StructLayout::Unpacked => ("{ ", " }"),
            };
            write!(output, "{open}").unwrap();
            for (i, elem_ty) in struct_ty.fields().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(elem_ty, output)?;
            }
            write!(output, "{close}").unwrap();
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            write!(output, "[{} x ", array_ty.size()).unwrap();
            self.export_type(array_ty.elem_type(), output)?;
            write!(output, "]").unwrap();
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            write!(output, "<{} x ", vec_ty.num_elements()).unwrap();
            self.export_type(vec_ty.elem_type(), output)?;
            write!(output, ">").unwrap();
        } else {
            return Err(format!(
                "cannot export unknown LLVM type `{}`",
                ty_ref.disp(self.ctx)
            ));
        }
        Ok(())
    }

    pub(super) fn is_pointer_type(&self, ty: TypeHandle) -> bool {
        ty.deref(self.ctx).is::<PointerType>()
    }

    pub(super) fn is_i8_type(&self, ty: TypeHandle) -> bool {
        ty.deref(self.ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 8)
    }

    /// Whether a fixed LLVM type has any storage. This deliberately does not
    /// reconstruct Rust layout: it only excludes unsized/opaque types and
    /// zero-byte LLVM carriers from an addressable by-value parameter.
    pub(super) fn fixed_type_has_storage(&self, ty: TypeHandle) -> Option<bool> {
        fn visit(
            state: &ModuleExportState<'_>,
            ty: TypeHandle,
            active: &mut rustc_hash::FxHashSet<TypeHandle>,
        ) -> Option<bool> {
            if !active.insert(ty) {
                return None;
            }
            let ty_ref = ty.deref(state.ctx);
            let result = if ty_ref.is::<IntegerType>()
                || ty_ref.is::<PointerType>()
                || ty_ref.is::<HalfType>()
                || ty_ref.is::<FP32Type>()
                || ty_ref.is::<FP64Type>()
            {
                Some(true)
            } else if let Some(array) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
                visit(state, array.elem_type(), active).map(|bytes| bytes && array.size() != 0)
            } else if let Some(vector) = ty_ref.downcast_ref::<crate::types::VectorType>() {
                visit(state, vector.elem_type(), active)
                    .map(|bytes| bytes && vector.num_elements() != 0)
            } else if let Some(structure) = ty_ref.downcast_ref::<StructType>() {
                if structure.is_opaque() {
                    None
                } else {
                    let mut has_storage = false;
                    for field in structure.fields() {
                        has_storage |= visit(state, field, active)?;
                    }
                    Some(has_storage)
                }
            } else {
                None
            };
            active.remove(&ty);
            result
        }
        visit(self, ty, &mut rustc_hash::FxHashSet::default())
    }

    /// Print the canonical legacy representation of an erased pointer.
    pub(super) fn export_canonical_pointer_type(&self, addrspace: u32, output: &mut String) {
        write!(output, "i8").unwrap();
        if addrspace != 0 {
            write!(output, " addrspace({addrspace})").unwrap();
        }
        write!(output, "*").unwrap();
    }

    /// Print a typed pointer to `pointee` in `addrspace` for LLVM 7 syntax.
    pub(super) fn export_pointer_to(
        &self,
        pointee: TypeHandle,
        addrspace: u32,
        output: &mut String,
    ) -> Result<(), String> {
        self.export_type(pointee, output)?;
        if addrspace != 0 {
            write!(output, " addrspace({addrspace})").unwrap();
        }
        write!(output, "*").unwrap();
        Ok(())
    }

    /// Print an LLVM 7 function-pointer type using the same recursively
    /// canonicalized argument and result types as function declarations.
    pub(super) fn export_function_pointer_type(
        &self,
        function_type: TypeHandle,
        output: &mut String,
    ) -> Result<(), String> {
        self.export_function_pointer_type_with_name(function_type, None, output)
    }

    /// A named function may retain by-value pointees which an anonymous
    /// opaque-pointer function type cannot represent.
    pub(super) fn export_named_function_pointer_type(
        &self,
        name: &str,
        output: &mut String,
    ) -> Result<(), String> {
        self.export_function_pointer_type_with_name(self.function_type(name)?, Some(name), output)
    }

    pub(super) fn export_function_parameter_type(
        &self,
        name: &str,
        index: usize,
        argument: TypeHandle,
        output: &mut String,
    ) -> Result<(), String> {
        if self.legacy_typed_pointers()
            && let Some(parameter) = self
                .function_grid_constants
                .get(name)
                .and_then(|parameters| parameters.iter().find(|parameter| parameter.index == index))
        {
            let argument_ref = argument.deref(self.ctx);
            let pointer = argument_ref.downcast_ref::<PointerType>().ok_or_else(|| {
                format!("grid-constant parameter {index} of `@{name}` is not a pointer")
            })?;
            self.export_pointer_to(parameter.pointee, pointer.address_space(), output)
        } else {
            self.export_type(argument, output)
        }
    }

    fn export_function_pointer_type_with_name(
        &self,
        function_type: TypeHandle,
        name: Option<&str>,
        output: &mut String,
    ) -> Result<(), String> {
        let function_ref = function_type.deref(self.ctx);
        let function_type = function_ref.downcast_ref::<FuncType>().ok_or_else(|| {
            format!(
                "expected function type, got `{}`",
                function_ref.disp(self.ctx)
            )
        })?;

        self.export_type(function_type.result_type(), output)?;
        write!(output, " (").unwrap();
        for (index, argument) in function_type.arg_types().iter().enumerate() {
            if index != 0 {
                write!(output, ", ").unwrap();
            }
            if let Some(name) = name {
                self.export_function_parameter_type(name, index, *argument, output)?;
            } else {
                self.export_type(*argument, output)?;
            }
        }
        if function_type.is_var_arg() {
            if !function_type.arg_types().is_empty() {
                write!(output, ", ").unwrap();
            }
            write!(output, "...").unwrap();
        }
        write!(output, ")*").unwrap();
        Ok(())
    }

    /// ABI alignment (bytes) of a type, when it can be stated exactly.
    ///
    /// Used as the fallback when no explicit alignment is stamped on a
    /// load/store/alloca op. Policy: exact or absent, never guessed. `None`
    /// (unknown type, or a computed value that is not a power of two and so
    /// not a legal `align`) makes the emitter omit the attribute, and LLVM
    /// falls back to the type's datalayout ABI alignment, which is always
    /// sound; a fabricated claim is not.
    pub(super) fn natural_alignment(&self, ty: TypeHandle) -> Option<u32> {
        let ty_ref = ty.deref(self.ctx);
        let align = if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            // ceil(width / 8); non-power-of-two widths are filtered below,
            // and widths past i128 decline because the datalayout caps
            // integer ABI alignment below the type's own size there.
            let bytes = int_ty.width().div_ceil(8).max(1);
            if bytes > 16 {
                return None;
            }
            bytes
        } else if ty_ref.is::<FP32Type>() {
            4
        } else if ty_ref.is::<FP64Type>() {
            8
        } else if ty_ref.is::<HalfType>() {
            2
        } else if ty_ref.is::<PointerType>() {
            8
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            // ABI alignment of `[N x T]` matches elem alignment.
            self.natural_alignment(array_ty.elem_type())?
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            // ABI alignment of an LLVM vector: power-of-2-rounded total width.
            let elem = self.natural_alignment(vec_ty.elem_type())?;
            let total = elem.saturating_mul(vec_ty.num_elements());
            let mut a = 1u32;
            while a.saturating_mul(2) <= total && a < 128 {
                a *= 2;
            }
            a
        } else {
            let struct_ty = ty_ref.downcast_ref::<StructType>()?;
            if struct_ty.layout() == StructLayout::Packed {
                1
            } else {
                // Max field alignment (1 if empty). May under-state a repr(align)
                // raise; the true alignment is carried on the op, not the type.
                let mut max = 1u32;
                for field in struct_ty.fields() {
                    max = max.max(self.natural_alignment(field)?);
                }
                max
            }
        };
        align.is_power_of_two().then_some(align)
    }
}
