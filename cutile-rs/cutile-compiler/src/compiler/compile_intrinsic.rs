/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Intrinsic compilation for compiler2.
//!
//! Handles macro execution, compiler_op calls, and check_partition_access
//! using tile-ir operations.

use syn::spanned::Spanned;

use super::_function::CUDATileFunctionCompiler;
use super::_type as types;
use super::_value::{CompilerContext, DimOrigin, Mutability, PartitionAxisOrigin, TileRustValue};
use super::shared_utils::{get_binary_op_from_op_str, get_const_hex, TileBinaryOp, OWNED_MAP_DIM};
use super::tile_rust_type::TileRustType;
use super::utils::{int_attr, rounding_mode_attr, signedness_attr};
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::generics::{
    get_cga_from_generic_argument, get_cga_from_type, GenericVars, TypeInstance,
};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::builder::{append_op, build_block, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, Module, Region, ScalarType, TileElementType, TileType, Type, Value,
};

use quote::ToTokens;
use std::collections::{BTreeMap, HashMap};
use syn::{Expr, ExprCall, ExprPath, GenericArgument, ItemFn, Lit, PathArguments};

const NESTED_MUTABLE_ACCESS_OFFSET_META: &str = "nested_mutable_access_offset";

/// Helper: determine signedness string from a Rust element type name.
fn get_signedness_str(element_type_str: &str) -> &'static str {
    super::utils::rust_int_signedness(element_type_str)
}

/// Convert a `TileRustType` that the old compiler built into a
/// `cutile_ir::ir::Type`.  Tries `types::convert_type` first (handles
/// primitives), then falls back to building a tile type from the
/// element-type name and shape when the type instance is structured.
fn tile_ir_type_from_trt(
    trt: &TileRustType,
    primitives: &HashMap<(String, String), syn::ItemImpl>,
) -> Option<cutile_ir::ir::Type> {
    // Fast path: primitives and simple cases handled by convert_type.
    if let Some(ty) = types::convert_type(trt) {
        return Some(ty);
    }
    // Structured types: extract element name + shape from the TypeInstance.
    if let TypeInstance::StructuredType(inst) = &trt.type_instance {
        // Use the same element-type resolution path the old compiler uses.
        let elem_name = trt.get_cuda_tile_element_type(primitives).ok()??;
        let shape: Vec<i64> = inst.shape.iter().map(|&d| d as i64).collect();
        return types::make_tile_type(&elem_name, &shape);
    }
    None
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub(super) fn scalar_i32_type(
        &self,
        span: &proc_macro2::Span,
    ) -> Result<TileRustType, JITError> {
        let rust_ty = syn::parse2::<syn::Type>("i32".parse()?).unwrap();
        self.compile_type(&rust_ty, &GenericVars::empty_unchecked(), &HashMap::new())?
            .ok_or_else(|| self.jit_error(span, "failed to compile i32 type"))
    }

    pub(crate) fn array_i32_type(
        &self,
        rank: usize,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<TileRustType, JITError> {
        let rust_ty = syn::parse_str::<syn::Type>(&format!("[i32; {rank}]")).unwrap();
        self.compile_type(&rust_ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(span, "failed to compile i32 array type"))
    }

    /// Static per-axis geometry of a partition-view value: tile extents,
    /// tensor extents (-1 where dynamic), and the dim-map remap.
    pub(crate) fn partition_static_geometry(
        &self,
        partition: &TileRustValue,
        span: &proc_macro2::Span,
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), JITError> {
        let Some(TypeParam::Tile(tile)) = partition.ty.params.first() else {
            return Err(self.jit_error(span, "partition type is missing its Tile parameter"));
        };
        let Some(TypeInstance::StructuredType(tile_inst)) = tile.type_instance.as_ref() else {
            return Err(self.jit_error(span, "the Tile parameter must be instantiated"));
        };
        let static_tile = tile_inst.shape.clone();
        let tensor = partition
            .ty
            .params
            .iter()
            .find_map(|p| match p {
                TypeParam::TensorView(tv) => Some(tv),
                _ => None,
            })
            .ok_or_else(|| self.jit_error(span, "partition type is missing a TensorView param"))?;
        let Some(TypeInstance::StructuredType(tensor_inst)) = tensor.type_instance.as_ref() else {
            return Err(self.jit_error(
                span,
                "expected a structured type instance for the tensor_view parameter",
            ));
        };
        let static_shape = tensor_inst.shape.clone();
        let dim_map = match partition.ty.params.iter().find_map(|p| match p {
            TypeParam::DimMap(dm) => Some(dm),
            _ => None,
        }) {
            Some(dim_map) => {
                let Some(TypeInstance::StructuredType(dim_map_inst)) =
                    dim_map.type_instance.as_ref()
                else {
                    return Err(self.jit_error(
                        span,
                        "expected a structured type instance for the dimension map",
                    ));
                };
                dim_map_inst.shape.clone()
            }
            None => (0..static_shape.len() as i32).collect(),
        };
        Ok((static_tile, static_shape, dim_map))
    }

    fn static_shape_from_value(
        &self,
        value: &TileRustValue,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<Vec<i32>, JITError> {
        self.static_shape_from_type(&value.ty, generic_vars, span)
    }

    fn static_shape_from_type(
        &self,
        ty: &TileRustType,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<Vec<i32>, JITError> {
        if let TypeInstance::StructuredType(instance) = &ty.type_instance {
            Ok(instance.shape.clone())
        } else if let Some(shape) = get_cga_from_type(&ty.rust_ty, generic_vars) {
            Ok(shape)
        } else {
            self.jit_error_result(span, "expected a statically shaped value")
        }
    }

    fn partition_tile_shape(
        &self,
        value: &TileRustValue,
        span: &proc_macro2::Span,
    ) -> Result<Vec<i32>, JITError> {
        let tile_shape = value.ty.params.iter().find_map(|param| match param {
            TypeParam::Tile(tile) => match tile.type_instance.as_ref() {
                Some(TypeInstance::StructuredType(instance)) => Some(instance.shape.clone()),
                _ => None,
            },
            _ => None,
        });
        tile_shape.ok_or_else(|| {
            self.jit_error(
                span,
                "nested mutable partition is missing tile-shape metadata",
            )
        })
    }

    fn compile_nested_mutable_access_offset_metadata(
        &self,
        module: &mut Module,
        block_id: BlockId,
        span: &proc_macro2::Span,
        partition_value: &TileRustValue,
        outer_tile: &TileRustValue,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let outer_shape = self.static_shape_from_value(outer_tile, generic_vars, span)?;
        let nested_shape = self.partition_tile_shape(partition_value, span)?;
        if outer_shape.len() != nested_shape.len() {
            return self.jit_error_result(
                span,
                &format!(
                    "nested mutable partition rank mismatch: outer tile rank {}, nested tile rank {}",
                    outer_shape.len(),
                    nested_shape.len()
                ),
            );
        }
        let rank = outer_shape.len();
        if rank > 3 {
            return self.jit_error_result(
                span,
                "nested mutable partition access offsets are only supported up to rank 3",
            );
        }
        for (i, nested_dim) in nested_shape.iter().enumerate() {
            if *nested_dim <= 0 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "nested mutable partition requires static positive nested tile dimensions, got nested dim {nested_dim} at axis {i}"
                    ),
                );
            }
        }
        if outer_shape.iter().any(|dim| *dim <= 0) {
            return Ok(None);
        }

        let i32_ty = self.scalar_i32_type(span)?;
        let scalar_i32_ir_ty = Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::I32),
        });
        let mut op_builder = OpBuilder::new(Opcode::GetTileBlockId, self.ir_location(span));
        for _ in 0..3 {
            op_builder = op_builder.result(scalar_i32_ir_ty.clone());
        }
        let (op_id, pid_results) = op_builder.build(module);
        append_op(module, block_id, op_id);

        let mut offsets = Vec::with_capacity(rank);
        for i in 0..rank {
            let outer_dim = outer_shape[i];
            let nested_dim = nested_shape[i];
            let pid = TileRustValue::new_primitive(pid_results[i], i32_ty.clone(), None);
            let ratio = ((outer_dim as i64 + nested_dim as i64 - 1) / nested_dim as i64) as i32;
            let offset = if ratio == 1 {
                pid
            } else {
                let ratio_value = self.compile_constant(module, block_id, generic_vars, ratio)?;
                self.compile_binary_op_from_values(
                    module,
                    block_id,
                    pid,
                    ratio_value,
                    &TileBinaryOp::Mul,
                    generic_vars,
                    ctx,
                    None,
                    span,
                )?
            };
            offsets.push(offset);
        }

        let array_ty = self.array_i32_type(rank, generic_vars, span)?;
        Ok(Some(TileRustValue::new_compound(offsets, array_ty)))
    }

    fn tile_type_for_element_shape(
        &self,
        element_name: &str,
        shape: &[i32],
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<TileRustType, JITError> {
        if let Some(tile_ty) = TileRustType::from_tile(element_name, shape) {
            return Ok(tile_ty);
        }

        let ty =
            syn::parse_str::<syn::Type>(&format!("Tile<{element_name}, {{ {shape:?} }}>")).unwrap();
        self.compile_type(&ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| {
                self.jit_error(
                    span,
                    &format!("failed to synthesize Tile<{element_name}, {{ {shape:?} }}>"),
                )
            })
    }

    fn static_element_count(shape: &[i32]) -> Option<i64> {
        let mut count = 1i64;
        for dim in shape {
            if *dim <= 0 {
                return None;
            }
            count = count.checked_mul(*dim as i64)?;
        }
        Some(count)
    }

    fn compile_fp4_pack_unpack(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        path_expr: &ExprPath,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 2 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`{}` expects 2 arguments, got {}",
                    path_expr.to_token_stream(),
                    call_expr.args.len()
                ),
            );
        }

        let ident = get_ident_from_path_expr(path_expr).to_string();
        let is_pack = match ident.as_str() {
            "__pack_f4e2m1fnx2_tile" => true,
            "__unpack_f4e2m1fnx2_tile" => false,
            other => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("unsupported FP4 pack/unpack helper `{other}`"),
                )
            }
        };

        let source = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[0],
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.args[0].span(),
                    "failed to compile FP4 pack/unpack source tile",
                )
            })?;
        let Some(source_value) = source.value else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "FP4 pack/unpack source must be a tile value",
            );
        };

        let shape_value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[1],
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.args[1].span(),
                    "failed to compile FP4 pack/unpack result shape",
                )
            })?;

        let source_shape =
            self.static_shape_from_value(&source, generic_vars, &call_expr.args[0].span())?;
        let result_shape =
            self.static_shape_from_value(&shape_value, generic_vars, &call_expr.args[1].span())?;

        let source_count = Self::static_element_count(&source_shape);
        let result_count = Self::static_element_count(&result_shape);

        if is_pack {
            if let Some(count) = source_count {
                if count % 2 != 0 {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("FP4 pack requires an even number of logical values, got {count}"),
                    );
                }
            }
            if let (Some(src), Some(dst)) = (source_count, result_count) {
                if src != dst * 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "FP4 pack shape mismatch: source has {src} logical values, result shape stores {}",
                            dst * 2
                        ),
                    );
                }
            }
        } else if let (Some(src), Some(dst)) = (source_count, result_count) {
            if dst != src * 2 {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "FP4 unpack shape mismatch: source stores {} logical values, result shape has {dst}",
                        src * 2
                    ),
                );
            }
        }

        let result_element = if is_pack { "f4e2m1fnx2" } else { "f4e2m1fn" };
        let source_element = if is_pack { "f4e2m1fn" } else { "f4e2m1fnx2" };

        let return_type = match return_type {
            Some(return_type) => {
                let return_shape =
                    self.static_shape_from_type(&return_type, generic_vars, &call_expr.span())?;
                if return_shape != result_shape {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "FP4 pack/unpack return shape {:?} does not match requested shape {:?}",
                            return_shape, result_shape
                        ),
                    );
                }
                return_type
            }
            None => self.tile_type_for_element_shape(
                result_element,
                &result_shape,
                generic_vars,
                &call_expr.span(),
            )?,
        };

        let source_flat_len = source_count.unwrap_or(-1);
        let result_flat_len = result_count.unwrap_or_else(|| {
            source_count
                .map(|count| if is_pack { count / 2 } else { count * 2 })
                .unwrap_or(-1)
        });

        let source_flat_ty =
            types::make_tile_type(source_element, &[source_flat_len]).ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    &format!("failed to build flat source tile type for `{source_element}`"),
                )
            })?;
        let result_flat_ty =
            types::make_tile_type(result_element, &[result_flat_len]).ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    &format!("failed to build flat result tile type for `{result_element}`"),
                )
            })?;
        let final_ty =
            tile_ir_type_from_trt(&return_type, self.modules.primitives()).ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    "failed to convert FP4 pack/unpack return type to Tile IR",
                )
            })?;

        let (flatten_op_id, flatten_results) =
            OpBuilder::new(Opcode::Reshape, self.ir_location(&call_expr.span()))
                .operand(source_value)
                .result(source_flat_ty)
                .build(module);
        append_op(module, block_id, flatten_op_id);

        let opcode = if is_pack {
            Opcode::Pack
        } else {
            Opcode::Unpack
        };
        let (pack_op_id, pack_results) =
            OpBuilder::new(opcode, self.ir_location(&call_expr.span()))
                .operand(flatten_results[0])
                .result(result_flat_ty)
                .build(module);
        append_op(module, block_id, pack_op_id);

        let (reshape_op_id, reshape_results) =
            OpBuilder::new(Opcode::Reshape, self.ir_location(&call_expr.span()))
                .operand(pack_results[0])
                .result(final_ty)
                .build(module);
        append_op(module, block_id, reshape_op_id);

        Ok(Some(TileRustValue::new_structured_type(
            reshape_results[0],
            return_type,
            None,
        )))
    }

    /// Compiles a `compiler_op` (intrinsic) function call.
    /// The compiler implements Rust-related functionality, such as polymorphism,
    /// for these functions.
    ///
    /// This handles the large dispatch table for calls to functions annotated
    /// with `#[cuda_tile::compiler_op(...)]`. These are internal operations
    /// like mma, tile ops, shape ops, reduce, arithmetic, cast, convert,
    /// return_type_meta_field, set_type_meta_field, check, and assume.
    pub fn compile_compiler_op_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        path_expr: &ExprPath,
        fn_item: &ItemFn,
        compiler_op_attrs: &SingleMetaList,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let call_expr_func_str = call_expr.func.to_token_stream().to_string();
        let ident = get_ident_from_path_expr(path_expr);
        let Some(compiler_op_name) = compiler_op_attrs.parse_string("name") else {
            return self.jit_error_result(
                &call_expr.span(),
                "compiler operation is missing a required `name` attribute",
            );
        };
        match compiler_op_name.as_str() {
            "mma" => {
                let mut operands =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let out = operands.remove(0);
                let out_type = out.ty.clone();
                let Some(out_rust_element_type) =
                    out_type.get_instantiated_rust_element_type(self.modules.primitives())
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine element type for `{}` output",
                            compiler_op_name
                        ),
                    );
                };
                let Some(_out_tile_ir_ty) = &out_type.tile_ir_ty else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to infer return type for `{}`; add a type annotation",
                            compiler_op_name
                        ),
                    );
                };
                let Some(out_cuda_tile_element_type) =
                    out_type.get_cuda_tile_element_type(self.modules.primitives())?
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine compiled element type for `{}`",
                            compiler_op_name
                        ),
                    );
                };
                let out_is_float = super::_type::scalar_from_name(&out_cuda_tile_element_type)
                    .is_some_and(|s| s.is_float());
                let (opcode, attrs) = if out_is_float {
                    (Opcode::MmaF, vec![])
                } else if !out_is_float {
                    let Some(lhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(self.modules.primitives())
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine left-hand operand element type for `mma`",
                        );
                    };
                    let Some(rhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(self.modules.primitives())
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine right-hand operand element type for `mma`",
                        );
                    };
                    (
                        Opcode::MmaI,
                        vec![
                            signedness_attr("signedness_lhs", get_signedness_str(&lhs_elem_ty)),
                            signedness_attr("signedness_rhs", get_signedness_str(&rhs_elem_ty)),
                        ],
                    )
                } else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`mma` does not support element type `{}`; expected a float or integer type",
                            out_rust_element_type
                        ),
                    );
                };
                // Get result type from the output value's type in the module.
                let result_type = module
                    .value_type(out.value.expect("Expected output to be a value."))
                    .clone();
                let (op_id, results) = OpBuilder::new(opcode, self.ir_location(&call_expr.span()))
                    .operands([
                        lhs.value.expect("Expected LHS to be a value."),
                        rhs.value.expect("Expected RHS to be a value."),
                        out.value.expect("Expected output to be a value."),
                    ])
                    .attrs(attrs)
                    .result(result_type)
                    .build(module);
                append_op(module, block_id, op_id);
                let value: Value = results[0];
                let tr_value = TileRustValue::new_value_kind_like(value, out_type);
                Ok(Some(tr_value))
            }
            "tile" => {
                let compiler_op_function = ident.to_string();
                if !compiler_op_function.ends_with("_tile") {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "tile operation function name must end with `_tile`, got `{}`",
                            compiler_op_function
                        ),
                    );
                }
                let op = compiler_op_function.split("_").collect::<Vec<&str>>()[0];
                let tile_binary_op = get_binary_op_from_op_str(op)?;
                let mut operands =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let res = self.compile_binary_op_from_values(
                    module,
                    block_id,
                    lhs,
                    rhs,
                    &tile_binary_op,
                    generic_vars,
                    ctx,
                    return_type,
                    &call_expr.span(),
                )?;
                Ok(Some(res))
            }
            "fp4_pack_unpack" => self.compile_fp4_pack_unpack(
                module,
                block_id,
                call_expr,
                path_expr,
                generic_vars,
                ctx,
                return_type,
            ),
            "shape" => {
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    "get_shape_dim" => {
                        let idx = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[1],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[1].span(),
                                    "failed to compile dimension index expression",
                                )
                            })?;
                        let Some(idx_bounds) = idx.bounds else {
                            return self.jit_error_result(
                                &call_expr.args[1].span(),
                                "dimension index must be a compile-time constant",
                            );
                        };
                        if !idx_bounds.is_exact() {
                            return self.jit_error_result(
                                &call_expr.args[1].span(),
                                "dimension index must have exact bounds (a single known value)",
                            );
                        }
                        let dim_index = idx_bounds.start;
                        let shape = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[0],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[0].span(),
                                    "failed to compile shape expression",
                                )
                            })?;
                        let Some(mut shape_fields) = shape.fields else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape value is missing its fields",
                            );
                        };
                        let Some(shape_dims) = shape_fields.remove("dims") else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape value is missing a `dims` field",
                            );
                        };
                        let Some(mut dims_values) = shape_dims.values else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape `dims` must be a compound (tuple) value",
                            );
                        };
                        let dim = dims_values.remove(dim_index as usize);
                        Ok(Some(dim))
                    }
                    "permute_array" => {
                        let src_slice = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[0],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[0].span(),
                                    "failed to compile source array for permutation",
                                )
                            })?;
                        let mut dst_slice = src_slice;
                        let Some(val_arr) = &mut dst_slice.values else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "expected a compound (tuple/array) value for permutation source",
                            );
                        };
                        *val_arr = {
                            let dim_map = self
                                .compile_expression(
                                    module,
                                    block_id,
                                    &call_expr.args[1],
                                    generic_vars,
                                    ctx,
                                    None,
                                )?
                                .ok_or_else(|| {
                                    self.jit_error(
                                        &call_expr.args[1].span(),
                                        "failed to compile dimension map for permutation",
                                    )
                                })?;
                            let TypeInstance::UserType(type_inst) = dim_map.ty.type_instance else {
                                return self.jit_error_result(
                                    &call_expr.args[1].span(),
                                    "expected a structured type for the dimension map argument",
                                );
                            };
                            let Some(dim_map) = type_inst.try_extract_cga(generic_vars) else {
                                return self.jit_error_result(
                                    &call_expr.args[1].span(),
                                    "dimension map must be a const generic array type",
                                );
                            };
                            if dim_map.len() != val_arr.len() {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "dimension map has {} entries but the array has {} elements",
                                        dim_map.len(),
                                        val_arr.len()
                                    ),
                                );
                            }
                            let mut result = vec![];
                            for i in 0..dim_map.len() {
                                // Permute by moving item from dim_map[i] -> i.
                                result.push(val_arr[dim_map[i] as usize].clone());
                            }
                            result
                        };
                        Ok(Some(dst_slice))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("unrecognized shape operation `{}`", compiler_op_function),
                    ),
                }
            }
            "dim_new" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`Dim::new` expects 1 argument, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let i32_tr_type = self
                    .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(&call_expr.span(), "failed to synthesize `i32` type")
                    })?;
                let mut value = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        Some(i32_tr_type),
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile dimension size",
                        )
                    })?;
                let dim_value = value.value.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        "dimension size must compile to a scalar value",
                    )
                })?;
                value.dim_origin = Some(DimOrigin::Value(dim_value));
                let return_type = match return_type {
                    Some(return_type) => return_type,
                    None => self
                        .compile_type(&syn::parse_quote!(Dim), generic_vars, &HashMap::new())?
                        .ok_or_else(|| {
                            self.jit_error(
                                &call_expr.span(),
                                "unable to infer return type for `Dim::new`",
                            )
                        })?,
                };
                let dim_origin = value.dim_origin.clone();
                let mut fields = BTreeMap::new();
                fields.insert("size".to_string(), value);
                let mut dim = TileRustValue::new_struct(fields, return_type);
                dim.dim_origin = dim_origin;
                Ok(Some(dim))
            }
            "dim_from_i32" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`IntoDim::into_dim` expects 1 argument, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let i32_tr_type = self
                    .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(&call_expr.span(), "failed to synthesize `i32` type")
                    })?;
                let mut value = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        Some(i32_tr_type),
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile dimension size",
                        )
                    })?;
                let dim_value = value.value.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        "dimension size must compile to a scalar value",
                    )
                })?;
                if value.dim_origin.is_none() {
                    value.dim_origin = Some(DimOrigin::Value(dim_value));
                }
                let return_type = match return_type {
                    Some(return_type) => return_type,
                    None => self
                        .compile_type(&syn::parse_quote!(Dim), generic_vars, &HashMap::new())?
                        .ok_or_else(|| {
                            self.jit_error(
                                &call_expr.span(),
                                "unable to infer return type for `IntoDim::into_dim`",
                            )
                        })?,
                };
                let dim_origin = value.dim_origin.clone();
                let bounds = value.bounds;
                let mut fields = BTreeMap::new();
                fields.insert("size".to_string(), value);
                let mut dim = TileRustValue::new_struct(fields, return_type);
                dim.dim_origin = dim_origin;
                dim.bounds = bounds;
                Ok(Some(dim))
            }
            "dim_value" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`Dim::value` expects 1 argument, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let dim = args.remove(0);
                let Some(fields) = dim.fields else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "`Dim::value` expects a compiler-created Dim",
                    );
                };
                let Some(mut size) = fields.get("size").cloned() else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "Dim is missing scalar size metadata",
                    );
                };
                if size.dim_origin.is_none() {
                    size.dim_origin = dim.dim_origin;
                }
                Ok(Some(size))
            }
            "coord" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("`coord` expects 1 argument, got {}", call_expr.args.len()),
                    );
                }
                // The tuple's arity fixes the coordinate rank; the matching
                // `Coord{rank}` shadow type is resolved below.
                let tuple_type = match &call_expr.args[0] {
                    syn::Expr::Tuple(tuple) if tuple.elems.len() >= 2 => {
                        let tuple_src = format!("({})", vec!["i32"; tuple.elems.len()].join(", "));
                        let tuple_ty = syn::parse_str::<syn::Type>(&tuple_src).map_err(|_| {
                            self.jit_error(
                                &call_expr.span(),
                                "failed to synthesize coordinate type",
                            )
                        })?;
                        Some(
                            self.compile_type(&tuple_ty, generic_vars, &HashMap::new())?
                                .ok_or_else(|| {
                                    self.jit_error(
                                        &call_expr.span(),
                                        "failed to synthesize coordinate type",
                                    )
                                })?,
                        )
                    }
                    _ => None,
                };
                let index = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        tuple_type,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(&call_expr.args[0].span(), "failed to compile coordinate")
                    })?;
                let Some(values) = index.values else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "`coord` expects a tuple coordinate",
                    );
                };
                let rank = values.len();
                let return_type = match return_type {
                    Some(return_type) => return_type,
                    None => {
                        let coord_ty = syn::parse_str::<syn::Type>(&format!("Coord{rank}"))
                            .ok()
                            .and_then(|ty| {
                                self.compile_type(&ty, generic_vars, &HashMap::new())
                                    .ok()
                                    .flatten()
                            });
                        coord_ty.ok_or_else(|| {
                            self.jit_error(
                                &call_expr.args[0].span(),
                                &format!(
                                    "`coord` does not support rank-{rank} coordinates (no `Coord{rank}` type)"
                                ),
                            )
                        })?
                    }
                };
                let array_ty = self.array_i32_type(rank, generic_vars, &call_expr.span())?;
                let coords = TileRustValue::new_compound(values, array_ty);
                let mut fields = BTreeMap::new();
                fields.insert("coords".to_string(), coords);
                Ok(Some(TileRustValue::new_struct(fields, return_type)))
            }
            "coord_as_array" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`coord_as_array` expects 1 argument, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let coord = args.remove(0);
                let Some(fields) = coord.fields else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "`coord_as_array` expects a coordinate created by `coord(...)`",
                    );
                };
                let Some(coords) = fields.get("coords").cloned() else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "coordinate is missing its metadata",
                    );
                };
                Ok(Some(coords))
            }
            "partition_with_bounds" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`Partition::with_bounds` expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let partition = args.remove(0);
                let bounds = args.remove(0);
                let Some(bound_values) = bounds.values else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "`Partition::with_bounds` expects a tuple of dimensions",
                    );
                };
                // Same operation as the `with_bounds` method call, so the same
                // body: see `apply_with_bounds`.
                self.apply_with_bounds(
                    module,
                    block_id,
                    partition,
                    bound_values,
                    return_type,
                    generic_vars,
                    ctx,
                    &call_expr.args[1].span(),
                )
                .map(Some)
            }
            "check_bounded_partition_access" => self.compile_check_bounded_partition_access(
                module,
                block_id,
                call_expr,
                generic_vars,
                ctx,
            ),
            "num_tiles" => {
                // Signature: fn num_tiles(view: &V, axis: i32) -> i32
                //
                // Lowers to `cuda_tile.get_index_space_shape` producing N scalar
                // i32 results (one per partition-view axis), then returns
                // result[axis]. The returned scalar carries DimOrigin metadata
                // so explicit `IntoDim::into_dim` can turn it into a Dim.
                // `axis` must be a compile-time constant.
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`num_tiles` expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let view = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        None,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile partition-view argument to `num_tiles`",
                        )
                    })?;
                let i32_tr_type = self
                    .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.span(),
                            "failed to synthesize `i32` type for `num_tiles`",
                        )
                    })?;
                let axis_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[1],
                        generic_vars,
                        ctx,
                        Some(i32_tr_type.clone()),
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[1].span(),
                            "failed to compile axis argument to `num_tiles`",
                        )
                    })?;

                let Some(axis_bounds) = axis_val.bounds else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "`num_tiles` axis must be a compile-time constant",
                    );
                };
                if !axis_bounds.is_exact() {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "`num_tiles` axis must have a single known value",
                    );
                }
                if axis_bounds.start < 0 {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!("`num_tiles` axis {} out of range", axis_bounds.start),
                    );
                }
                let axis = axis_bounds.start as usize;

                let view_value = view.value.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        "expected a direct value for the partition view",
                    )
                })?;
                let view_ty = module.value_type(view_value).clone();
                let Type::PartitionView(pv) = &view_ty else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("`num_tiles` expects a partition view, got `{:?}`", view_ty),
                    );
                };
                let rank = pv.tile_shape.len();
                if axis >= rank {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!("`num_tiles` axis {axis} out of range for rank-{rank} partition"),
                    );
                }

                let i32_scalar_ty = Type::Tile(TileType {
                    shape: vec![],
                    element_type: TileElementType::Scalar(ScalarType::I32),
                });
                let mut op_builder = OpBuilder::new(
                    Opcode::GetIndexSpaceShape,
                    self.ir_location(&call_expr.span()),
                )
                .operand(view_value);
                for _ in 0..rank {
                    op_builder = op_builder.result(i32_scalar_ty.clone());
                }
                let (op_id, results) = op_builder.build(module);
                append_op(module, block_id, op_id);

                let selected = results[axis];
                let return_type = match return_type {
                    Some(return_type) => return_type,
                    None => i32_tr_type,
                };
                let mut tr_value = TileRustValue::new_value_kind_like(selected, return_type);
                let parent_axis = pv.dim_map.get(axis).copied().ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} is missing from partition dim_map {:?}",
                            pv.dim_map
                        ),
                    )
                })?;
                if parent_axis < 0 {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} maps to invalid parent axis {parent_axis}"
                        ),
                    );
                }
                let parent_axis = parent_axis as usize;
                let Some(&parent_dim) = pv.tensor_view.shape.get(parent_axis) else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} maps to parent axis {parent_axis}, but parent tensor rank is {}",
                            pv.tensor_view.shape.len()
                        ),
                    );
                };
                let tile_dim = pv.tile_shape[axis] as i64;
                if tile_dim <= 0 {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("`num_tiles` axis {axis} has invalid tile dimension {tile_dim}"),
                    );
                }
                if parent_dim >= 0 {
                    let num_tiles = (parent_dim + tile_dim - 1) / tile_dim;
                    tr_value.bounds = Some(Bounds::exact(num_tiles));
                }
                tr_value.dim_origin = Some(DimOrigin::PartitionAxis {
                    view: view_value,
                    axis,
                    tile_dim: tile_dim as i32,
                });
                // Name the axis by TENSOR as well as by view value. The view
                // value identifies this exact partition, which proves accesses
                // back into it; the tensor name is what declared `dim(t, a)`
                // facts speak about, so it is what lets a count taken here
                // bound an access into a *different* partition whose extent a
                // precondition relates to this one. Both describe one axis;
                // they differ only in who can match them.
                //
                // The axis recorded is the ROOT axis `dim_map[axis]`, since
                // that is the axis declared facts name. Minted only for a
                // root-framed parameter: for a slabbed `&mut` the count above
                // is per-CTA, which no `dim(t, a)` fact describes.
                if let Some(tensor) = view
                    .tensor_origin
                    .as_ref()
                    .filter(|tensor| self.root_framed_param(tensor).is_some())
                {
                    tr_value.partition_axis_origin = Some(PartitionAxisOrigin {
                        tensor: tensor.clone(),
                        axis: parent_axis,
                        tile_dim: tile_dim as i32,
                    });
                }
                Ok(Some(tr_value))
            }
            "partition_index_coords" => {
                if call_expr.args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`partition_index_coords` expects 1 argument, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let index = args.remove(0);
                let Some(fields) = index.fields else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "`partition_index_coords` expects a compiler-created PartitionIndex",
                    );
                };
                let Some(coords) = fields.get("coords").cloned() else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "PartitionIndex is missing coordinate metadata",
                    );
                };
                Ok(Some(coords))
            }
            "validate_partition_store" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`validate_partition_store` expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let view = args.remove(0);
                let index = args.remove(0);
                let view_value = view.value.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        "expected a direct value for mapped partition index validation",
                    )
                })?;
                let coords = index
                    .fields
                    .as_ref()
                    .and_then(|fields| fields.get("coords"))
                    .cloned()
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[1].span(),
                            "MappedPartitionMut::store requires an index produced by this partition's iter_indices() iterator or built with coord(...)",
                        )
                    })?;

                // Degenerate case: a whole minted PartitionIndex. Every axis
                // is proven by stream provenance.
                if let Some(origin_values) = &index.partition_origins {
                    if !origin_values.contains(&view_value) {
                        return self.jit_error_result(
                            &call_expr.args[1].span(),
                            "MappedPartitionMut::store index was produced by a different mapped partition",
                        );
                    }
                    return Ok(Some(coords));
                }

                // Composite coordinate: per-axis proofs. Streamed axes need a
                // component minted by this partition's stream on that axis;
                // owned axes (OWNED map dim) need a Dim proof bound to this
                // partition's axis (or a shared-stream partner's, since
                // iter_indices_with establishes grid equality), or a constant
                // within the statically-known tile grid.
                let coord_values = coords.values.as_ref().ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[1].span(),
                        "coordinates must be a compound value",
                    )
                })?;
                let (tile_shape, map_shape) = self.mapped_partition_type_shapes(
                    &view,
                    generic_vars,
                    &call_expr.args[0].span(),
                )?;
                let rank = tile_shape.len();
                if coord_values.len() != rank {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!(
                            "store coordinate rank {} does not match mapped partition rank {rank}",
                            coord_values.len()
                        ),
                    );
                }
                // On a fully-streamed map, every axis needs stream provenance,
                // so only the minted index is acceptable.
                if map_shape.iter().all(|&dim| dim != OWNED_MAP_DIM) {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "MappedPartitionMut::store requires an index produced by this partition's iter_indices() iterator",
                    );
                }

                // First pass — streamed axes: provenance, and collect the
                // shared-stream partner views the minted components carry.
                let mut shared_views: Vec<cutile_ir::ir::Value> = vec![view_value];
                for axis in 0..rank {
                    if map_shape[axis] == OWNED_MAP_DIM {
                        continue;
                    }
                    let coord_value = &coord_values[axis];
                    let minted_here = match coord_value.index_origin.as_ref() {
                        Some(DimOrigin::PartitionAxis {
                            view: origin_view,
                            axis: origin_axis,
                            ..
                        }) if *origin_axis == axis => {
                            *origin_view == view_value
                                || coord_value
                                    .partition_origins
                                    .as_ref()
                                    .is_some_and(|origins| origins.contains(&view_value))
                        }
                        _ => false,
                    };
                    if !minted_here {
                        return self.jit_error_result(
                            &call_expr.args[1].span(),
                            &format!(
                                "streamed axis {axis} of a composite store index must be the component minted by this partition's iter_indices() (owned axes may use Dim indices instead)"
                            ),
                        );
                    }
                    if let Some(origins) = coord_value.partition_origins.as_ref() {
                        for origin in origins {
                            if !shared_views.contains(origin) {
                                shared_views.push(*origin);
                            }
                        }
                    }
                    self.check_stats
                        .discharged
                        .set(self.check_stats.discharged.get() + 1);
                }

                // Second pass — owned axes: Dim proof or constant rung. The
                // static geometry comes from the view's IR type (works for
                // kernel-parameter views, whose Rust type carries no
                // structured params).
                let view_ir_ty = module.value_type(view_value).clone();
                let Type::PartitionView(pv) = &view_ir_ty else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("expected a mapped partition view, got `{view_ir_ty:?}`"),
                    );
                };
                for axis in 0..rank {
                    if map_shape[axis] != OWNED_MAP_DIM {
                        continue;
                    }
                    let coord_value = &coord_values[axis];
                    if let Some(DimOrigin::PartitionAxis {
                        view: origin_view,
                        axis: origin_axis,
                        ..
                    }) = coord_value.index_origin.as_ref()
                    {
                        if *origin_axis == axis && shared_views.contains(origin_view) {
                            self.check_stats
                                .discharged
                                .set(self.check_stats.discharged.get() + 1);
                            continue;
                        }
                        return self.jit_error_result(
                            &call_expr.args[1].span(),
                            &format!(
                                "owned axis {axis} of a composite store index was produced by a different dimension"
                            ),
                        );
                    }
                    // Constant rung: a coordinate with known constant bounds
                    // checks statically against a statically-known axis grid.
                    let static_tile_dim = pv.tile_shape[axis];
                    let parent_axis = pv.dim_map.get(axis).copied().unwrap_or(axis as i32);
                    let static_shape_dim = if parent_axis >= 0 {
                        pv.tensor_view
                            .shape
                            .get(parent_axis as usize)
                            .copied()
                            .unwrap_or(-1)
                    } else {
                        -1
                    };
                    if let (Some(bounds), true) = (coord_value.bounds, static_shape_dim != -1) {
                        let num_partitions = (static_shape_dim + static_tile_dim as i64 - 1)
                            / static_tile_dim as i64;
                        if !(0 <= bounds.start && bounds.end < num_partitions) {
                            return self.jit_error_result(
                                &call_expr.args[1].span(),
                                &format!(
                                    "owned axis {axis}: constant range [{}, {}] is not within the {num_partitions}-tile grid",
                                    bounds.start, bounds.end
                                ),
                            );
                        }
                        self.check_stats
                            .discharged
                            .set(self.check_stats.discharged.get() + 1);
                        continue;
                    }
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!(
                            "owned axis {axis} of a composite store index must come from iterating the axis's Dim (num_tiles(&view, {axis})) or be a constant within the axis's static tile grid"
                        ),
                    );
                }
                Ok(Some(coords))
            }
            "swizzle_partition_index_2d" => {
                if call_expr.args.len() != 3 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`swizzle_partition_index_2d` expects 3 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let Expr::Path(path) = &*call_expr.func else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        "expected swizzle partition-index call to use a function path",
                    );
                };
                let Some(last_segment) = path.path.segments.last() else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        "expected swizzle partition-index function path",
                    );
                };
                let PathArguments::AngleBracketed(generic_args_for_call) = &last_segment.arguments
                else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        "swizzle_partition_index_2d requires explicit map-shape const generics",
                    );
                };
                let Some(map_shape_arg) = generic_args_for_call.args.iter().nth(1) else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        "swizzle_partition_index_2d missing map-shape generic argument",
                    );
                };
                let Some(map_shape) = get_cga_from_generic_argument(map_shape_arg, generic_vars)
                else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        "failed to resolve swizzle_partition_index_2d map-shape const generic",
                    );
                };
                if map_shape.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        &format!(
                            "swizzle_partition_index_2d expects a rank-2 map shape, got rank {}",
                            map_shape.len()
                        ),
                    );
                }
                let swizzle_m_value = map_shape[0];
                let swizzle_n_value = map_shape[1];
                if swizzle_m_value <= 0 || swizzle_n_value <= 0 {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        &format!(
                            "swizzle_partition_index_2d map shape dimensions must be positive, got [{swizzle_m_value}, {swizzle_n_value}]"
                        ),
                    );
                }
                let i32_tr_type = self
                    .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.span(),
                            "failed to synthesize `i32` type for swizzle partition index",
                        )
                    })?;
                let mut args = Vec::with_capacity(3);
                for arg in &call_expr.args {
                    args.push(
                        self.compile_expression(
                            module,
                            block_id,
                            arg,
                            generic_vars,
                            ctx,
                            Some(i32_tr_type.clone()),
                        )?
                        .ok_or_else(|| {
                            self.jit_error(
                                &arg.span(),
                                "failed to compile swizzle partition-index argument",
                            )
                        })?,
                    );
                }
                let tile_id = args.remove(0);
                let num_bid_m = args.remove(0);
                let num_bid_n = args.remove(0);
                let (bid_m, bid_n) = self.emit_swizzle_2d(
                    module,
                    block_id,
                    &tile_id,
                    &num_bid_m,
                    &num_bid_n,
                    swizzle_m_value,
                    swizzle_n_value,
                    generic_vars,
                    ctx,
                    &call_expr.span(),
                )?;

                let return_type = return_type.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.span(),
                        "unable to infer return type for swizzle partition index",
                    )
                })?;
                let array_ty = self.array_i32_type(2, generic_vars, &call_expr.span())?;
                let coords = TileRustValue::new_compound(vec![bid_m, bid_n], array_ty);
                let mut fields = BTreeMap::new();
                fields.insert("coords".to_string(), coords);
                Ok(Some(TileRustValue::new_struct(fields, return_type)))
            }
            "reduce" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("`reduce` expects 2 arguments, got {}", call_expr.args.len()),
                    );
                }
                let operand = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        None,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile reduce operand",
                        )
                    })?;
                let Expr::Lit(lit_expr) = &call_expr.args[1] else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "the dimension argument must be an integer literal",
                    );
                };
                let Lit::Int(int_lit) = &lit_expr.lit else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "Dim arg must be an integer.",
                    );
                };
                let dim = int_lit.base10_parse::<i32>().map_err(|e| {
                    self.jit_error(
                        &call_expr.args[1].span(),
                        &format!("Failed to parse lit int: {e}"),
                    )
                })?;
                let TypeInstance::StructuredType(structured_type) = operand.ty.type_instance else {
                    return self
                        .jit_error_result(&call_expr.args[0].span(), "expected a struct value");
                };
                let Some(primitive_type) = structured_type.primitive_type else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "Expected primitive type to be defined.",
                    );
                };
                let Some(element_type) = primitive_type.get_rust_element_instance_ty() else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "Failed to obtain rust element instance type.",
                    );
                };

                let reduce_op_string = ident.to_string();
                let (identity, closure_block_op): (String, syn::Block) =
                    match reduce_op_string.as_str() {
                        "reduce_min" => (
                            get_const_hex(&element_type, "max")?,
                            syn::parse_quote! { { min(curr, prev) } },
                        ),
                        "reduce_max" => (
                            get_const_hex(&element_type, "min")?,
                            syn::parse_quote! { { max(curr, prev) } },
                        ),
                        "reduce_sum" => (
                            get_const_hex(&element_type, "zero")?,
                            syn::parse_quote! { { curr + prev } },
                        ),
                        "reduce_prod" => (
                            get_const_hex(&element_type, "one")?,
                            syn::parse_quote! { { curr * prev } },
                        ),
                        _ => {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Unsupported reduce operation: {reduce_op_string}"),
                            );
                        }
                    };

                let mut shape = structured_type.shape.clone();
                shape.remove(dim as usize);
                // Build tile-ir types directly (no format→parse chain for the IR types).
                let ir_result_type = types::make_tile_type(
                    &element_type,
                    &shape.iter().map(|&s| s as i64).collect::<Vec<_>>(),
                )
                .ok_or_else(|| {
                    self.jit_error(
                        &call_expr.span(),
                        "Failed to build tile-ir type for reduce result.",
                    )
                })?;
                let ir_operand_type =
                    types::make_scalar_tile_type(&element_type).ok_or_else(|| {
                        self.jit_error(
                            &call_expr.span(),
                            "Failed to build tile-ir type for reduce operand.",
                        )
                    })?;
                // TileRustTypes needed for closure body variables and result wrapping.
                let tile_rust_result_type = match TileRustType::from_tile(&element_type, &shape) {
                    Some(t) => t,
                    None => {
                        let ty = syn::parse_str::<syn::Type>(&format!(
                            "Tile<{element_type}, {{ {shape:#?} }}>"
                        ))
                        .unwrap();
                        self.compile_type(&ty, generic_vars, &HashMap::new())?
                            .unwrap()
                    }
                };
                let tile_rust_iter_operand_type =
                    match TileRustType::from_scalar_tile(&element_type) {
                        Some(t) => t,
                        None => {
                            let ty = syn::parse_str::<syn::Type>(&format!(
                                "Tile<{element_type}, {{ [] }}>"
                            ))
                            .unwrap();
                            self.compile_type(&ty, generic_vars, &HashMap::new())?
                                .unwrap()
                        }
                    };
                // Build the reduce body region.
                let region_id = {
                    let local_var_types = &[
                        ir_operand_type.clone(), // operand_i_current_iter
                        ir_operand_type.clone(), // operand_i_prev_iter
                    ];
                    let (local_block_id, local_block_args) = build_block(module, local_var_types);
                    let local_var_names = ["curr", "prev"];
                    let mut local_vars = CompilerContext::empty();
                    for i in 0..local_block_args.len() {
                        let value: Value = local_block_args[i];
                        let name = local_var_names[i];
                        let ty = tile_rust_iter_operand_type.clone();
                        let tile_rust_val = TileRustValue::new_value_kind_like(value, ty);
                        local_vars.vars.insert(name.to_string(), tile_rust_val);
                    }
                    // This is a binary op on the Tile type.
                    let op = self
                        .compile_block(
                            module,
                            local_block_id,
                            &closure_block_op,
                            generic_vars,
                            &mut local_vars,
                            return_type,
                        )?
                        .ok_or_else(|| {
                            self.jit_error(&call_expr.span(), "failed to compile reduce operation")
                        })?;
                    let Some(op_value) = op.value else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "Failed to obtain value from reduce compilation.",
                        );
                    };
                    let (yield_op_id, _) =
                        OpBuilder::new(Opcode::Yield, self.ir_location(&call_expr.span()))
                            .operand(op_value)
                            .build(module);
                    append_op(module, local_block_id, yield_op_id);
                    module.alloc_region(Region {
                        blocks: vec![local_block_id],
                    })
                };

                // Build the reduce op itself.
                // Build a properly typed identities attribute from the hex identity.
                let identity_attr = {
                    let scalar_ty = super::_type::scalar_from_name(&element_type)
                        .unwrap_or(cutile_ir::ir::ScalarType::I32);
                    let ir_ty = cutile_ir::ir::Type::Scalar(scalar_ty);
                    let hex_str = identity.trim_start_matches("0x").trim_start_matches("0X");
                    let bits = u64::from_str_radix(hex_str, 16).unwrap_or(0);
                    if scalar_ty.is_float() {
                        let float_val = match element_type.as_str() {
                            "f32" => f32::from_bits(bits as u32) as f64,
                            "f64" => f64::from_bits(bits),
                            "f16" => half::f16::from_bits(bits as u16).to_f64(),
                            "bf16" => half::bf16::from_bits(bits as u16).to_f64(),
                            _ => bits as f64,
                        };
                        Attribute::Float(float_val, ir_ty)
                    } else {
                        Attribute::Integer(bits as i64, ir_ty)
                    }
                };
                let (op_id, results) =
                    OpBuilder::new(Opcode::Reduce, self.ir_location(&call_expr.span()))
                        .attrs([
                            int_attr("dim", dim as i64),
                            (
                                "identities".to_string(),
                                Attribute::Array(vec![identity_attr]),
                            ),
                        ])
                        .operand(operand.value.ok_or_else(|| {
                            self.jit_error(
                                &call_expr.args[0].span(),
                                "Expect value for reduce op operand.",
                            )
                        })?)
                        .result(ir_result_type)
                        .region(region_id)
                        .build(module);
                append_op(module, block_id, op_id);
                let value: Value = results[0];
                let tr_value = TileRustValue::new_value_kind_like(value, tile_rust_result_type);
                Ok(Some(tr_value))
            }
            "arithmetic" => {
                let num_operands = call_expr.args.len();
                match num_operands {
                    2 => {
                        let binary_op = get_binary_op_from_op_str(&ident.to_string())?;
                        // Binary arithmetic operation.
                        let mut args = self.compile_call_args(
                            module,
                            block_id,
                            &call_expr.args,
                            generic_vars,
                            ctx,
                        )?;
                        let lhs = args.remove(0);
                        let rhs = args.remove(0);
                        Ok(Some(self.compile_binary_op_from_values(
                            module,
                            block_id,
                            lhs,
                            rhs,
                            &binary_op,
                            generic_vars,
                            ctx,
                            return_type,
                            &call_expr.span(),
                        )?))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("arithmetic ops with {num_operands} operands not supported"),
                    ),
                }
            }
            "cast" => {
                let compiler_op_function = ident.to_string();
                // For casts, we require the rust types compiles to the same value.
                // We therefore only need to update the rust type.
                let args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                if args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("cast expects 1 argument, got {}", args.len()),
                    );
                }
                let mut new_value = args[0].clone();
                let old_type = new_value.ty.rust_ty.clone();
                match compiler_op_function.as_str() {
                    "scalar_to_tile" => {
                        let element_type = get_rust_element_type_primitive(&old_type);
                        if let Some(t) = TileRustType::from_scalar_tile(&element_type) {
                            new_value.ty = t;
                        } else {
                            new_value.ty.rust_ty = syn::parse_str::<syn::Type>(&format!(
                                "Tile<{element_type}, {{[]}}>"
                            ))
                            .unwrap();
                        }
                    }
                    "tile_to_scalar" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, self.modules.primitives())
                        else {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!(
                                    "Failed to cast from {} to {}",
                                    old_type.to_token_stream(),
                                    get_sig_output_type(&fn_item.sig).to_token_stream()
                                ),
                            );
                        };
                        new_value.ty.rust_ty =
                            syn::parse2::<syn::Type>(element_type.to_string().parse().unwrap())
                                .unwrap();
                    }
                    "pointer_to_tile" => {
                        // Wrapping preserves the pointer's constness: a
                        // `*const E` wraps into `PointerTile<*const E, {[]}>`.
                        let old_ty_str = old_type.to_token_stream().to_string();
                        let (is_mutable, element_type) =
                            match crate::types::get_ptr_type(&old_ty_str) {
                                Some((is_mutable, element)) => (is_mutable, element),
                                None => (true, get_rust_element_type_primitive(&old_type)),
                            };
                        if let Some(t) = TileRustType::from_scalar_ptr(&element_type, is_mutable) {
                            new_value.ty = t;
                        } else {
                            new_value.ty.rust_ty = syn::parse_str::<syn::Type>(&format!(
                                "PointerTile<{} {element_type}, {{[]}}>",
                                crate::types::ptr_prefix(is_mutable)
                            ))
                            .unwrap();
                        }
                    }
                    "tile_to_pointer" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, self.modules.primitives())
                        else {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!(
                                    "Failed to cast from {} to {}",
                                    old_type.to_token_stream(),
                                    get_sig_output_type(&fn_item.sig).to_token_stream()
                                ),
                            );
                        };
                        // Unwrapping preserves the tile's constness.
                        let is_mutable =
                            !old_type.to_token_stream().to_string().contains("* const");
                        new_value.ty.rust_ty = syn::parse2::<syn::Type>(
                            format!("{} {element_type}", crate::types::ptr_prefix(is_mutable))
                                .parse()
                                .unwrap(),
                        )
                        .unwrap();
                    }
                    "cast_const" | "cast_tile_const" => {
                        new_value.ty.set_pointer_constness(false);
                    }
                    "cast_mut" | "cast_tile_mut" => {
                        new_value.ty.set_pointer_constness(true);
                    }
                    _ => {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!("Unsupported cast compiler_op: {}", compiler_op_function),
                        );
                    }
                }
                Ok(Some(new_value))
            }
            "convert" => {
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    "convert_scalar" | "convert_tile" => {
                        let mut args = self.compile_call_args(
                            module,
                            block_id,
                            &call_expr.args,
                            generic_vars,
                            ctx,
                        )?;
                        if args.len() != 1 {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("convert expects 1 argument, got {}", args.len()),
                            );
                        }
                        let mut arg = args.pop().unwrap();
                        let new_type_compiled = if let Some(return_type) = return_type {
                            return_type
                        } else {
                            let PathArguments::AngleBracketed(generic_args) =
                                &path_expr.path.segments.last().unwrap().arguments
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Failed to get type parameters for {}",
                                        path_expr.to_token_stream()
                                    ),
                                );
                            };
                            if generic_args.args.len() != 1 {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Expected 1 generic argument for convert, got {}",
                                        generic_args.args.len()
                                    ),
                                );
                            }
                            let GenericArgument::Type(new_type) = &generic_args.args[0] else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Failed to get type parameters for {}",
                                        path_expr.to_token_stream()
                                    ),
                                );
                            };
                            let Some(new_type_compiled) =
                                self.compile_type(new_type, generic_vars, &HashMap::new())?
                            else {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "{compiler_op_function} failed to compile new type: {}",
                                        new_type.to_token_stream()
                                    ),
                                );
                            };
                            new_type_compiled
                        };
                        let old_element_type_str = arg
                            .ty
                            .type_instance
                            .get_rust_element_instance_ty()
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    "Type resolution failed for old element type.",
                                )
                            })?;
                        let new_element_type_str = new_type_compiled
                            .type_instance
                            .get_rust_element_instance_ty()
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    "Type resolution failed for new element type.",
                                )
                            })?;
                        if old_element_type_str == new_element_type_str {
                            // Identity conversion — update to the resolved type.
                            arg.ty = new_type_compiled;
                            return Ok(Some(arg));
                        }
                        let output_type =
                            tile_ir_type_from_trt(&new_type_compiled, self.modules.primitives())
                                .ok_or_else(|| {
                                    self.jit_error(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to obtain tile-ir type for convert {}",
                                            call_expr.to_token_stream()
                                        ),
                                    )
                                })?;
                        // These aren't required for all ops.
                        let (op_id, results) = match (
                            old_element_type_str.as_str(),
                            new_element_type_str.as_str(),
                        ) {
                            // TODO (hme): There are some more like this that make sense, but no time to implement.
                            ("i64", "i32") => {
                                // cuda_tile.trunci %from %overflow
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            }
                            ("i32", "i64") => {
                                // cuda_tile.exti %from %signedness
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            }
                            // Integer → float: IToF with signedness from source type.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .is_some_and(|s| s.is_integer())
                                    && super::_type::scalar_from_name(to)
                                        .is_some_and(|s| s.is_float()) =>
                            {
                                let signedness = signedness_attr(
                                    "signedness",
                                    get_signedness_str(&old_element_type_str),
                                );
                                let rounding = rounding_mode_attr("nearest_even");
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                OpBuilder::new(Opcode::IToF, self.ir_location(&call_expr.span()))
                                    .attrs([signedness, rounding])
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            // Float → integer: FToI with signedness from target type.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .is_some_and(|s| s.is_float())
                                    && super::_type::scalar_from_name(to)
                                        .is_some_and(|s| s.is_integer()) =>
                            {
                                let signedness = signedness_attr(
                                    "signedness",
                                    get_signedness_str(&new_element_type_str),
                                );
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                let rounding =
                                    rounding_mode_attr("nearest_int_to_zero");
                                OpBuilder::new(Opcode::FToI, self.ir_location(&call_expr.span()))
                                    .attrs([signedness, rounding])
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            // Float → float: all float type pairs use FToF
                            // with NEAREST_EVEN rounding. This covers f16, bf16,
                            // f32, f64, tf32, f8e4m3fn, f8e5m2 — matching
                            // cutile-python's _get_type_conversion_encoder.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .is_some_and(|s| s.is_float())
                                    && super::_type::scalar_from_name(to)
                                        .is_some_and(|s| s.is_float()) =>
                            {
                                let rounding = rounding_mode_attr("nearest_even");
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                OpBuilder::new(Opcode::FToF, self.ir_location(&call_expr.span()))
                                    .attr("rounding_mode", rounding.1)
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            _ => {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Unsupported conversion {old_element_type_str:#?} -> {new_element_type_str:#?}"
                                    ),
                                )
                            }
                        };
                        append_op(module, block_id, op_id);
                        let value: Value = results[0];
                        Ok(Some(TileRustValue::new_value_kind_like(
                            value,
                            new_type_compiled,
                        )))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unsupported convert compiler_op: {}", compiler_op_function),
                    ),
                }
            }
            "return_type_meta_field" => {
                let Some(type_meta_field) = compiler_op_attrs.parse_string("type_meta_field")
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected return_type_meta_field {compiler_op_attrs:#?}"),
                    );
                };
                let args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                if args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "return_type_meta_field expects 1 argument, got {}",
                            args.len()
                        ),
                    );
                }
                let value = args[0].clone();
                let Some(ref type_meta) = value.type_meta else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "Undefined type_meta for value {value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"
                        ),
                    );
                };
                let Some(return_value) = type_meta.fields.get(&type_meta_field) else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("undefined type metadata field `{type_meta_field}` on this value"),
                    );
                };
                let mut return_value = return_value.clone();
                if type_meta_field == "shape" {
                    self.label_param_extents(&mut return_value, &value);
                }
                Ok(Some(return_value))
            }
            "set_nested_mutable_partition_access_offset" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "set_nested_mutable_partition_access_offset expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }

                let var_path = match &call_expr.args[0] {
                    Expr::Path(var_path) => var_path,
                    Expr::Reference(reference) => match &*reference.expr {
                        Expr::Path(var_path) => var_path,
                        _ => {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                &format!(
                                    "first argument to `set_nested_mutable_partition_access_offset` must be a mutable partition variable, got `{}`",
                                    call_expr.args[0].to_token_stream()
                                ),
                            )
                        }
                    },
                    _ => {
                        return self.jit_error_result(
                            &call_expr.args[0].span(),
                            &format!(
                                "first argument to `set_nested_mutable_partition_access_offset` must be a mutable partition variable, got `{}`",
                                call_expr.args[0].to_token_stream()
                            ),
                        )
                    }
                };
                let var_name = get_ident_from_path_expr(var_path)
                    .to_token_stream()
                    .to_string();

                let outer_tile = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[1],
                        generic_vars,
                        ctx,
                        None,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[1].span(),
                            "failed to compile nested mutable outer tile shape",
                        )
                    })?;
                let mut partition_value = ctx.vars.get(var_name.as_str()).cloned().ok_or_else(
                    || {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            &format!(
                                "first argument to `set_nested_mutable_partition_access_offset` must be a known variable, got `{}`",
                                call_expr.args[0].to_token_stream()
                            ),
                        )
                    },
                )?;
                if partition_value.mutability != Mutability::Mutable {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`set_nested_mutable_partition_access_offset` requires a mutable variable, but got {:?}",
                            partition_value.mutability
                        ),
                    );
                }
                let access_offset = self.compile_nested_mutable_access_offset_metadata(
                    module,
                    block_id,
                    &call_expr.span(),
                    &partition_value,
                    &outer_tile,
                    generic_vars,
                    ctx,
                )?;
                if let Some(access_offset) = access_offset {
                    partition_value
                        .insert_type_meta_field(NESTED_MUTABLE_ACCESS_OFFSET_META, access_offset)?;
                    ctx.vars.insert(var_name, partition_value);
                }
                Ok(None)
            }
            "set_type_meta_field" | "set_access_token" => {
                let Some(type_meta_field) = compiler_op_attrs.parse_string("type_meta_field")
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected set_type_meta_field {compiler_op_attrs:#?}"),
                    );
                };
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "set_type_meta_field expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let Expr::Path(var_arg) = &call_expr.args[0] else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "first argument to `set_type_meta_field` must be a simple variable path, got `{}`",
                            call_expr.to_token_stream()
                        ),
                    );
                };
                let var_name = get_ident_from_path_expr(var_arg)
                    .to_token_stream()
                    .to_string();
                if !ctx.vars.contains_key(var_name.as_str()) {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "first argument to `set_type_meta_field` must be a known variable, got `{}`",
                            call_expr.to_token_stream()
                        ),
                    );
                }
                let explicit_token_update =
                    type_meta_field == "token" && compiler_op_name == "set_type_meta_field";
                if explicit_token_update {
                    // TODO (hme): Treat token installation as a mutation in
                    // collect_mutated_variables_from_block, carrying token SSA
                    // values through blocks, branches and loops uniformly. Then
                    // remove the immutable-token metadata copy-back workaround.
                    if ctx.inside_for || ctx.innermost_loop.is_some() || ctx.token_update_in_region
                    {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "set_token/set_tensor_token is not supported inside conditional or loop regions; install the token before entering the region",
                        );
                    }
                    if !ctx.function_level_bindings.contains(&var_name) {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "set_token/set_tensor_token requires a function-level tensor binding; bindings declared inside nested blocks are not supported",
                        );
                    }
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let type_meta_value = args[1].clone();
                let type_value = &mut args[0];
                let Some(ref mut type_meta) = type_value.type_meta else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "Undefined type_meta for value {type_value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"
                        ),
                    );
                };
                let old_val = type_meta
                    .fields
                    .insert(type_meta_field.clone(), type_meta_value);
                if old_val.is_none() {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "undefined type metadata field `{type_meta_field}` on this value; cannot set a field that does not exist"
                        ),
                    );
                }
                let result_value = type_value.clone();
                // The ordering token is metadata, not data: installing one on
                // a read-only tensor is how a kernel chains its loads after an
                // external ordering point (a PDL wait). Every other field
                // still requires a mutable binding.
                let sets_ordering_token = type_meta_field == "token";
                if result_value.mutability != Mutability::Mutable && !sets_ordering_token {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`set_type_meta_field` requires a mutable variable, but got {:?}",
                            result_value.mutability
                        ),
                    );
                }
                ctx.vars.insert(var_name.clone(), result_value);
                if explicit_token_update {
                    ctx.explicit_token_updates.insert(var_name);
                }
                Ok(None)
            }
            "check" => {
                if self.entry_attrs.get_entry_arg_bool("unchecked_accesses") {
                    // Skip checks if unchecked_accesses is set.
                    return Ok(None);
                }
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    // Both view types walk the same ladder over the same
                    // goals; they differ only in the Rust type of the
                    // receiver, which the checker never inspects.
                    "check_partition_access" | "check_partition_access_mut" => self
                        .compile_check_partition_access(
                            module,
                            block_id,
                            call_expr,
                            generic_vars,
                            ctx,
                        ),
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected compiler_op call {}", call_expr_func_str),
                    ),
                }
            }
            "assume" => {
                let tr_value =
                    self.compile_assumption_call(call_expr, module, block_id, generic_vars, ctx)?;
                Ok(Some(tr_value))
            }
            _ => self.jit_error_result(
                &call_expr.span(),
                &format!("Unexpected compiler_op {compiler_op_attrs:#?}"),
            ),
        }
    }

    /// Emits the flat-tile-id → `(bid_m, bid_n)` schedule math for one 2-D
    /// (sub-)grid: linear traversal for a `[1, 1]` map, grouped-M bands for
    /// `[GM, 1]`, and grouped-MN for `[GM, GN]`. Shared by the
    /// `swizzle_partition_index_2d` intrinsic and the rank-N `iter_indices()`
    /// loop lowering (which applies it to the trailing two axes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_swizzle_2d(
        &self,
        module: &mut Module,
        block_id: BlockId,
        tile_id: &TileRustValue,
        num_bid_m: &TileRustValue,
        num_bid_n: &TileRustValue,
        swizzle_m_value: i32,
        swizzle_n_value: i32,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &proc_macro2::Span,
    ) -> Result<(TileRustValue, TileRustValue), JITError> {
        let tile_id = tile_id.clone();
        let num_bid_m = num_bid_m.clone();
        let num_bid_n = num_bid_n.clone();
        let (mut bid_m, mut bid_n) = if swizzle_m_value == 1 && swizzle_n_value == 1 {
            let bid_m = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                num_bid_n.clone(),
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let bid_n = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                num_bid_n.clone(),
                &TileBinaryOp::Rem,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            (bid_m, bid_n)
        } else if swizzle_n_value == 1 {
            let swizzle_m =
                self.compile_constant(module, block_id, generic_vars, swizzle_m_value)?;
            let num_bid_in_m_band = self.compile_binary_op_from_values(
                module,
                block_id,
                swizzle_m.clone(),
                num_bid_n.clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let group_m_id = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                num_bid_in_m_band.clone(),
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let first_bid_m = self.compile_binary_op_from_values(
                module,
                block_id,
                group_m_id.clone(),
                swizzle_m.clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let remaining_m = self.compile_binary_op_from_values(
                module,
                block_id,
                num_bid_m.clone(),
                first_bid_m.clone(),
                &TileBinaryOp::Sub,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let actual_group_size_m = self.compile_binary_op_from_values(
                module,
                block_id,
                remaining_m,
                swizzle_m,
                &TileBinaryOp::Min,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let m_band_start = self.compile_binary_op_from_values(
                module,
                block_id,
                group_m_id,
                num_bid_in_m_band,
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_m_band = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                m_band_start,
                &TileBinaryOp::Sub,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_group_m = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_m_band.clone(),
                actual_group_size_m.clone(),
                &TileBinaryOp::Rem,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let bid_m = self.compile_binary_op_from_values(
                module,
                block_id,
                first_bid_m,
                tile_in_group_m,
                &TileBinaryOp::Add,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let bid_n = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_m_band,
                actual_group_size_m,
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            (bid_m, bid_n)
        } else {
            let swizzle_m =
                self.compile_constant(module, block_id, generic_vars, swizzle_m_value)?;
            let swizzle_n =
                self.compile_constant(module, block_id, generic_vars, swizzle_n_value)?;
            let num_bid_in_m_band = self.compile_binary_op_from_values(
                module,
                block_id,
                swizzle_m.clone(),
                num_bid_n.clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let group_m_id = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                num_bid_in_m_band.clone(),
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let first_bid_m = self.compile_binary_op_from_values(
                module,
                block_id,
                group_m_id.clone(),
                swizzle_m.clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let remaining_m = self.compile_binary_op_from_values(
                module,
                block_id,
                num_bid_m.clone(),
                first_bid_m.clone(),
                &TileBinaryOp::Sub,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let actual_group_size_m = self.compile_binary_op_from_values(
                module,
                block_id,
                remaining_m,
                swizzle_m,
                &TileBinaryOp::Min,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let m_band_start = self.compile_binary_op_from_values(
                module,
                block_id,
                group_m_id,
                num_bid_in_m_band,
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_m_band = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                m_band_start,
                &TileBinaryOp::Sub,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let n_group_capacity = self.compile_binary_op_from_values(
                module,
                block_id,
                actual_group_size_m.clone(),
                swizzle_n.clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let group_n_id = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_m_band.clone(),
                n_group_capacity.clone(),
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let first_bid_n = self.compile_binary_op_from_values(
                module,
                block_id,
                group_n_id,
                swizzle_n,
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_n_group = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_m_band,
                n_group_capacity,
                &TileBinaryOp::Rem,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_group_m = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_n_group.clone(),
                actual_group_size_m.clone(),
                &TileBinaryOp::Rem,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let bid_m = self.compile_binary_op_from_values(
                module,
                block_id,
                first_bid_m,
                tile_in_group_m,
                &TileBinaryOp::Add,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let tile_in_group_n = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_in_n_group,
                actual_group_size_m,
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let bid_n = self.compile_binary_op_from_values(
                module,
                block_id,
                first_bid_n,
                tile_in_group_n,
                &TileBinaryOp::Add,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            (bid_m, bid_n)
        };

        if let Some(bounds) = num_bid_m.bounds {
            if bounds.is_exact() && bounds.start > 0 {
                bid_m.bounds = Some(Bounds::new(0, bounds.start - 1));
            }
        }
        if let Some(bounds) = num_bid_n.bounds {
            if bounds.is_exact() && bounds.start > 0 {
                bid_n.bounds = Some(Bounds::new(0, bounds.start - 1));
            }
        }
        Ok((bid_m, bid_n))
    }

    /// Emits the rank-N flat-tile-id → per-axis block-id schedule for
    /// `iter_indices()`. Leading axes traverse linearly (their map dims must
    /// be 1) via row-major decomposition; the trailing two axes reuse
    /// [`Self::emit_swizzle_2d`]. Rank-1 is a direct linear traversal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_mapped_partition_schedule(
        &self,
        module: &mut Module,
        block_id: BlockId,
        tile_id: &TileRustValue,
        num_bids: &[TileRustValue],
        map_shape: &[i32],
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &proc_macro2::Span,
    ) -> Result<Vec<TileRustValue>, JITError> {
        let rank = map_shape.len();
        debug_assert_eq!(rank, num_bids.len());
        for (axis, &dim) in map_shape.iter().enumerate().take(rank.saturating_sub(2)) {
            if dim != 1 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "mapped partition schedules group only the trailing two axes: leading map axis {axis} must be 1, got {dim}"
                    ),
                );
            }
        }
        let exact_axis_bounds = |num_bid: &TileRustValue| {
            num_bid.bounds.and_then(|bounds| {
                if bounds.is_exact() && bounds.start > 0 {
                    Some(Bounds::new(0, bounds.start - 1))
                } else {
                    None
                }
            })
        };
        if rank == 1 {
            if map_shape[0] != 1 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "rank-1 mapped partition maps must be [1], got [{}]",
                        map_shape[0]
                    ),
                );
            }
            let mut bid = tile_id.clone();
            bid.bounds = exact_axis_bounds(&num_bids[0]);
            return Ok(vec![bid]);
        }

        let m_axis = rank - 2;
        let n_axis = rank - 1;
        let mut bids: Vec<TileRustValue> = Vec::with_capacity(rank);
        let mut inner_tile_id = tile_id.clone();
        if rank > 2 {
            let inner_total = self.compile_binary_op_from_values(
                module,
                block_id,
                num_bids[m_axis].clone(),
                num_bids[n_axis].clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            let mut lead = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                inner_total.clone(),
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            inner_tile_id = self.compile_binary_op_from_values(
                module,
                block_id,
                tile_id.clone(),
                inner_total,
                &TileBinaryOp::Rem,
                generic_vars,
                ctx,
                None,
                span,
            )?;
            // Row-major decomposition of the leading flat id: the last leading
            // axis varies fastest.
            for axis in 0..rank - 2 {
                let mut bid = if axis + 1 < rank - 2 {
                    let mut stride = num_bids[axis + 1].clone();
                    for num_bid in &num_bids[axis + 2..rank - 2] {
                        stride = self.compile_binary_op_from_values(
                            module,
                            block_id,
                            stride,
                            num_bid.clone(),
                            &TileBinaryOp::Mul,
                            generic_vars,
                            ctx,
                            None,
                            span,
                        )?;
                    }
                    let bid = self.compile_binary_op_from_values(
                        module,
                        block_id,
                        lead.clone(),
                        stride.clone(),
                        &TileBinaryOp::Div,
                        generic_vars,
                        ctx,
                        None,
                        span,
                    )?;
                    lead = self.compile_binary_op_from_values(
                        module,
                        block_id,
                        lead,
                        stride,
                        &TileBinaryOp::Rem,
                        generic_vars,
                        ctx,
                        None,
                        span,
                    )?;
                    bid
                } else {
                    lead.clone()
                };
                bid.bounds = exact_axis_bounds(&num_bids[axis]);
                bids.push(bid);
            }
        }
        let (bid_m, bid_n) = self.emit_swizzle_2d(
            module,
            block_id,
            &inner_tile_id,
            &num_bids[m_axis],
            &num_bids[n_axis],
            map_shape[m_axis],
            map_shape[n_axis],
            generic_vars,
            ctx,
            span,
        )?;
        bids.push(bid_m);
        bids.push(bid_n);
        Ok(bids)
    }
}
