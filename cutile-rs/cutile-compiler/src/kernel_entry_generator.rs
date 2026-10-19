/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generates CUDA Tile IR entry-point functions from Rust kernel signatures.
//! Handles tensor argument unpacking, validation, and shape/stride boilerplate.

use crate::ast::SourceLocation;
use crate::compiler::_module::CUDATileModules;
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::hints::OptimizationHints;
use crate::kernel_naming::KernelNaming;
use crate::syn_utils::{get_fn_arg_var_name, get_ident_from_path_expr, get_ident_generic_args};
use crate::types::{get_primitives_attrs, get_type_mutability};
use cuda_async::device_context::{
    PointerParamType, ScalarParamType, TensorParamType, ValidParamType, Validator,
};
use proc_macro2::Ident;
use proc_macro2::Span;
use quote::ToTokens;
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::{Expr, FnArg, GenericArgument, ItemFn, ItemImpl, Lit, Stmt, Token};

#[derive(Clone, Debug, PartialEq, Eq)]
enum TensorParamKind {
    Tensor,
    MappedPartitionMut { map_shape: String },
}

struct TensorInput {
    pub fn_name: String,
    pub var_name: String,
    pub element_type: String,
    pub dim_type: String,
    pub rank: i32,
    input_tensor_shape: InputTensorShape,
    param_kind: TensorParamKind,
    static_strides: Vec<String>,
    pub mutable: bool,
    pub max_divisibility: Option<i32>,
    pub spec: Option<crate::specialization::SpecializationBits>,
}

impl TensorInput {
    pub fn new(
        fn_name: String,
        src_fn_arg: &FnArg,
        generic_vars: &GenericVars,
        stride_args: &HashMap<String, Vec<i32>>,
        spec_args: &HashMap<String, crate::specialization::SpecializationBits>,
        primitives: &HashMap<(String, String), ItemImpl>,
        opt_hints: &OptimizationHints,
    ) -> Result<Self, JITError> {
        let FnArg::Typed(typed_arg) = src_fn_arg else {
            return SourceLocation::unknown().jit_error_result("Failed to get arg type.");
        };
        let ty = &*typed_arg.ty;
        let Some(element_type) = get_tensor_element_type(ty, primitives, generic_vars)? else {
            return SourceLocation::unknown().jit_error_result("Failed to get element type.");
        };
        let param_kind = get_tensor_param_kind(ty)?;
        let var_name = get_fn_arg_var_name(src_fn_arg);
        let input_tensor_shape = get_tensor_shape(ty, generic_vars)?;
        let static_strides = stride_args
            .get(var_name.as_str())
            .unwrap()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        let dim_type = "i32".to_string();
        let rank = input_tensor_shape.shape.len() as i32;
        let mutable = get_type_mutability(ty)
            || matches!(param_kind, TensorParamKind::MappedPartitionMut { .. });
        let max_divisibility = opt_hints
            .target_gpu_name
            .as_ref()
            .and_then(|gpu| opt_hints.get_sm_hints(gpu))
            .and_then(|hints| hints.max_divisibility);
        let spec = spec_args.get(var_name.as_str()).cloned();
        let res = Self {
            fn_name,
            var_name,
            element_type,
            dim_type,
            rank,
            input_tensor_shape,
            param_kind,
            static_strides,
            mutable,
            max_divisibility,
            spec,
        };
        res.validate()?;
        Ok(res)
    }
    pub fn ptr_arg(&self) -> FnArg {
        let var_name = self.var_name.clone();
        let element_type = self.element_type.clone();
        syn::parse2::<syn::FnArg>(
            format!("{var_name}_ptr: PointerTile<*mut {element_type}, {{[]}}>")
                .parse()
                .unwrap(),
        )
        .unwrap()
    }
    pub fn i_arg(&self, i_arg_name: &str, i: i32) -> FnArg {
        let var_name = self.var_name.clone();
        let dim_type = self.dim_type.clone();
        syn::parse2::<syn::FnArg>(
            format!("{var_name}_{i_arg_name}_{i}: {dim_type}")
                .parse()
                .unwrap(),
        )
        .unwrap()
    }

    pub fn validate(&self) -> Result<(), JITError> {
        if let TensorParamKind::MappedPartitionMut { .. } = &self.param_kind {
            // Mutable partitions are capped at rank 3 below; rank-1 and rank-2
            // mapped partitions traverse linearly or with a grouped 2-D
            // swizzle, rank-3 adds a linear leading axis.
            if self.rank < 1 || self.rank > 3 {
                return SourceLocation::unknown()
                    .jit_error_result("MappedPartitionMut parameters must have rank 1 through 3.");
            }
        }
        if self.mutable {
            // If it's mutable, it must be at most rank 3.
            if self.rank > 3 {
                return SourceLocation::unknown().jit_error_result(
                    "Unable to partition tensors with rank > 3. \
                     Try collapsing dimensions to obtain rank 3 partitioned tensors.",
                );
            }
        }
        let var_name = self.var_name.clone();
        let fn_name = self.fn_name.clone();
        if self.static_strides.len() != self.rank as usize {
            return SourceLocation::unknown().jit_error_result(&format!(
                "The tensor provided for parameter {var_name} \
                     may not have the rank expected by kernel {fn_name}: \
                     Tensor parameter {var_name} \
                     expects a tensor of rank {}, but {} strides were provided.",
                self.rank,
                self.static_strides.len()
            ));
        }
        Ok(())
    }

    pub fn dim_arg(&self, i: i32) -> FnArg {
        self.i_arg("dim", i)
    }
    pub fn stride_arg(&self, i: i32) -> FnArg {
        self.i_arg("stride", i)
    }
    pub fn partition_dim_arg(&self, i: i32) -> FnArg {
        self.i_arg("partition_dim", i)
    }
    pub fn partition_stride_arg(&self, i: i32) -> FnArg {
        self.i_arg("partition_stride", i)
    }

    #[expect(dead_code, reason = "Reserved for future device-side offset path")]
    pub fn offset_arg(&self) -> FnArg {
        let var_name = self.var_name.clone();
        let dim_type = self.dim_type.clone();
        syn::parse2::<syn::FnArg>(format!("{var_name}_offset: {dim_type}").parse().unwrap())
            .unwrap()
    }

    fn generate_args(&self) -> Punctuated<FnArg, Token![,]> {
        let mut fn_args = Punctuated::<FnArg, Token![,]>::new();
        // ptr
        fn_args.push(self.ptr_arg());
        // No offset param for immutable tensors — offset is applied
        // host-side by pushing an already-offset device pointer.
        // dims
        for i in 0..self.rank {
            fn_args.push(self.dim_arg(i));
        }
        // strides
        for i in 0..self.rank {
            fn_args.push(self.stride_arg(i));
        }
        if self.mutable {
            // dims
            for i in 0..self.rank {
                fn_args.push(self.partition_dim_arg(i));
            }
            // strides
            for i in 0..self.rank {
                fn_args.push(self.partition_stride_arg(i));
            }
        }
        fn_args
    }

    #[allow(clippy::ptr_arg)] // public signature stays as-is
    fn get_dynamic_elements(
        &self,
        static_elements: &Vec<String>,
        i_arg_name: String,
    ) -> Vec<String> {
        let var_name = self.var_name.clone();
        let mut dynamic_elements = vec![];
        for (i, dim) in static_elements.iter().enumerate() {
            if dim == "- 1" || dim == "-1" {
                dynamic_elements.push(format!("{var_name}_{i_arg_name}_{i}"));
            }
        }
        dynamic_elements
    }

    fn get_assume_non_negative_stmt(var_name: String) -> Stmt {
        let stmt = syn::parse2::<syn::Stmt>(
            format!("let {var_name} = unsafe {{ assume_bounds_lower::<_, 0>({var_name}) }};")
                .parse()
                .unwrap(),
        )
        .unwrap();
        stmt
    }

    pub(crate) fn get_assume_div_by(var_name: String, div_by: i32) -> Stmt {
        let stmt = syn::parse2::<syn::Stmt>(
            format!("let {var_name} = unsafe {{ assume_div_by::<_, {div_by}>({var_name}) }};")
                .parse()
                .unwrap(),
        )
        .unwrap();
        stmt
    }

    /// Computes effective divisor: auto-inferred from spec, capped by max_divisibility.
    /// When max_divisibility is set, it acts as a ceiling (max_divisibility).
    /// When no spec is available, max_divisibility is used directly as before.
    fn effective_div(&self, inferred: Option<i32>) -> i32 {
        match (inferred, self.max_divisibility) {
            (Some(auto_val), Some(max_div)) => auto_val.min(max_div),
            (Some(auto_val), None) => auto_val,
            (None, Some(fallback)) => fallback,
            (None, None) => 1,
        }
    }

    fn generate_statements(&self) -> Vec<Stmt> {
        let mut statements = Vec::new();
        let var_name = &self.var_name;
        let i_type = &self.dim_type;
        let element_type = &self.element_type;

        // TODO (hme): Would be great to generate comments for debugging.
        // if self.mutable {
        //     let item = syn::parse2::<syn::Item>(
        //         format!("// Construct immutable Tensor {var_name}.").parse().unwrap()
        //     ).unwrap();
        //     statements.push(syn::Stmt::Item(item));
        // } else {
        //     let item = syn::parse2::<syn::Item>(
        //         format!("// Construct mutable Tensor {var_name} from partition.").parse().unwrap()
        //     ).unwrap();
        //     statements.push(syn::Stmt::Item(item));
        // }

        // Shape. Mutable tensor args are exposed to the user kernel as tile-shaped
        // tensors, but the entry wrapper constructs a view of the full tensor so
        // Tensor::store can index the output partition by the current tile-block id.
        let dim_i_arg_name = "dim";
        let dims_arg_name = format!("{}s", dim_i_arg_name);
        let tensor_view_shape = if self.mutable {
            vec!["-1".to_string(); self.rank as usize]
        } else {
            self.input_tensor_shape.shape.clone()
        };
        // At this point, we have concrete shape data for everything.
        let dynamic_dims: Vec<String> =
            self.get_dynamic_elements(&tensor_view_shape, dim_i_arg_name.to_string());
        let dims_var = format!("{var_name}_{dims_arg_name}");
        for (i, dynamic_dim_var) in dynamic_dims.iter().enumerate() {
            statements.push(Self::get_assume_non_negative_stmt(dynamic_dim_var.clone()));
            let inferred = self
                .spec
                .as_ref()
                .and_then(|s| s.shape_div.get(i))
                .map(|h| h.divisor);
            let div = self.effective_div(inferred);
            if div > 1 {
                statements.push(Self::get_assume_div_by(dynamic_dim_var.clone(), div));
            }
        }
        let dims = syn::parse2::<syn::Stmt>(
            format!(
                "let {dims_var}: &[{i_type}] = &[{}];",
                dynamic_dims.join(",")
            )
            .parse()
            .unwrap(),
        )
        .unwrap();
        statements.push(dims);

        let shape_param = &self.input_tensor_shape.shape_param;
        let tensor_view_shape_param = if self.mutable {
            format!("{{[{}]}}", vec!["-1"; self.rank as usize].join(","))
        } else {
            shape_param.clone()
        };
        let shape_var = format!("{dims_var}_shape");
        let dims_shape_stmnt = syn::parse2::<syn::Stmt>(
                format!("let {shape_var}: Shape<{tensor_view_shape_param}> = Shape::<{tensor_view_shape_param}>{{ dims: {var_name}_{dims_arg_name} }};").parse().unwrap()
        ).unwrap();
        statements.push(dims_shape_stmnt);

        // Strides
        let stride_i_arg_name = "stride";
        let strides_arg_name = format!("{}s", stride_i_arg_name);
        let dynamic_strides: Vec<String> =
            self.get_dynamic_elements(&self.static_strides, stride_i_arg_name.to_string());
        for (i, dynamic_stride_var) in dynamic_strides.iter().enumerate() {
            statements.push(Self::get_assume_non_negative_stmt(
                dynamic_stride_var.clone(),
            ));
            let inferred = self
                .spec
                .as_ref()
                .and_then(|s| s.stride_div.get(i))
                .map(|h| h.divisor);
            let div = self.effective_div(inferred);
            if div > 1 {
                statements.push(Self::get_assume_div_by(dynamic_stride_var.clone(), div));
            }
        }
        let strides_var = format!("{var_name}_{strides_arg_name}");
        let strides = syn::parse2::<syn::Stmt>(
            format!(
                "let {strides_var}: &[{i_type}] = &[{}];",
                dynamic_strides.join(",")
            )
            .parse()
            .unwrap(),
        )
        .unwrap();
        statements.push(strides);

        let strides_param = format!("{{[{}]}}", self.static_strides.join(","));
        let strides_array_var = format!("{strides_var}_array");
        let strides_array_stmnt = syn::parse2::<syn::Stmt>(
            format!("let {strides_array_var}: Array<{strides_param}> = Array::<{strides_param}>{{ dims: {var_name}_{strides_arg_name} }};").parse().unwrap()
        ).unwrap();
        statements.push(strides_array_stmnt);

        // Create fresh token.
        let token_var = format!("{var_name}_token");
        let strides_array_stmnt = syn::parse2::<syn::Stmt>(
            format!("let {token_var}: Token = new_token_unordered();")
                .parse()
                .unwrap(),
        )
        .unwrap();
        statements.push(strides_array_stmnt);

        // Use the full tensor pointer. Mutable output partition indexing is
        // handled by Tensor::load/store inside the kernel body.
        let ptr_var = format!("{var_name}_ptr");
        let final_ptr_var = ptr_var;
        let inferred_ptr_div = self.spec.as_ref().map(|s| s.base_ptr_div.divisor);
        let ptr_div = self.effective_div(inferred_ptr_div);
        if ptr_div > 1 {
            statements.push(Self::get_assume_div_by(final_ptr_var.clone(), ptr_div));
        }

        let final_shape_var = shape_var.clone();

        match &self.param_kind {
            TensorParamKind::Tensor => {
                let tensor_stmnt = syn::parse2::<syn::Stmt>(
                    format!("let {var_name}: Tensor<{element_type}, {shape_param}> = unsafe {{ make_tensor_view({final_ptr_var}, {final_shape_var}, {strides_array_var}, {token_var}) }};").parse().unwrap()
                ).unwrap();
                statements.push(tensor_stmnt);
            }
            TensorParamKind::MappedPartitionMut { map_shape } => {
                let tensor_var = format!("{var_name}_tensor");
                let tensor_stmnt = syn::parse2::<syn::Stmt>(
                    format!("let {tensor_var}: Tensor<{element_type}, {shape_param}> = unsafe {{ make_tensor_view({final_ptr_var}, {final_shape_var}, {strides_array_var}, {token_var}) }};").parse().unwrap()
                ).unwrap();
                statements.push(tensor_stmnt);

                let partition_shape_var = format!("{var_name}_partition_shape");
                let partition_shape_stmnt = syn::parse2::<syn::Stmt>(
                    format!("let {partition_shape_var}: Shape<{shape_param}> = Shape::<{shape_param}>{{ dims: &[] }};").parse().unwrap()
                ).unwrap();
                statements.push(partition_shape_stmnt);

                let scheduled_stmnt = syn::parse2::<syn::Stmt>(
                    format!("let mut {var_name}: MappedPartitionMut<{element_type}, {shape_param}, {map_shape}> = unsafe {{ make_mapped_partition_view::<{element_type}, {shape_param}, {shape_param}, {map_shape}, padding::None>(&{tensor_var}, {partition_shape_var}, padding::None, {token_var}) }};").parse().unwrap()
                ).unwrap();
                statements.push(scheduled_stmnt);
            }
        }
        statements
    }
    fn generate_arg(&self) -> Expr {
        let var_name = self.var_name.clone();
        if matches!(self.param_kind, TensorParamKind::MappedPartitionMut { .. }) {
            return syn::parse2::<syn::Expr>(var_name.parse().unwrap()).unwrap();
        }
        let ref_type = if self.mutable { "&mut" } else { "&" };
        syn::parse2::<syn::Expr>(format!("{ref_type} {var_name}").parse().unwrap()).unwrap()
    }
}

fn normalized_fn_arg_type(param: &FnArg, ty: syn::Type) -> Result<FnArg, JITError> {
    let FnArg::Typed(typed_param) = param else {
        return SourceLocation::unknown().jit_error_result("Unexpected receiver argument.");
    };
    let mut typed_param = typed_param.clone();
    *typed_param.ty = ty;
    Ok(FnArg::Typed(typed_param))
}

fn get_tensor_param_kind(ty: &syn::Type) -> Result<TensorParamKind, JITError> {
    let (type_ident, type_generic_args) = get_ident_generic_args(ty);
    let Some(type_ident) = type_ident else {
        return SourceLocation::unknown().jit_error_result("Expected tensor parameter type.");
    };
    match type_ident.to_string().as_str() {
        "Tensor" => Ok(TensorParamKind::Tensor),
        "MappedPartitionMut" => {
            let Some(map_shape) = type_generic_args.args.iter().nth(2) else {
                return SourceLocation::unknown().jit_error_result(
                    "MappedPartitionMut expects map shape as its third generic argument.",
                );
            };
            Ok(TensorParamKind::MappedPartitionMut {
                map_shape: generic_arg_to_const_array_string(map_shape)?,
            })
        }
        other => SourceLocation::unknown().jit_error_result(&format!(
            "Unsupported tensor-like parameter type `{other}`."
        )),
    }
}

fn generic_arg_to_const_array_string(arg: &GenericArgument) -> Result<String, JITError> {
    match arg {
        GenericArgument::Const(expr) => Ok(expr.to_token_stream().to_string()),
        GenericArgument::Type(syn::Type::Path(path)) => Ok(path.to_token_stream().to_string()),
        _ => SourceLocation::unknown()
            .jit_error_result("MappedPartitionMut map shape must be an i32 const generic array."),
    }
}

/// Generates a Tile IR entry-point wrapper for a kernel function, including tensor argument unpacking.
pub fn generate_entry_point(
    modules: &CUDATileModules,
    fn_item: &ItemFn,
    generic_vars: &GenericVars,
    stride_args: &HashMap<String, Vec<i32>>,
    spec_args: &HashMap<String, crate::specialization::SpecializationBits>,
    scalar_hints: &HashMap<String, crate::specialization::DivHint>,
    primitives: &HashMap<(String, String), ItemImpl>,
    opt_hints: &OptimizationHints,
) -> Result<(ItemFn, Validator), JITError> {
    // Construct an entry point which takes tile and pointer parameters, constructs tensors views, and calls the original function.
    let mut fn_entry = fn_item.clone();
    let kernel_naming = KernelNaming::new(fn_item.sig.ident.to_string().as_str());
    let fn_name = kernel_naming.public_name().to_string();
    let fn_impl_name = kernel_naming.user_impl_name();
    let fn_entry_name = kernel_naming.entry_name();
    fn_entry.sig.ident = Ident::new(fn_entry_name.as_str(), fn_item.sig.ident.span());
    // Generate entry point parameters, entry point body statements, and call arguments for function call.
    fn_entry.sig.inputs.clear();
    let mut statements = vec![];
    let mut final_stmnt_args = vec![];
    let mut fn_params_concrete_types: Vec<ValidParamType> = vec![];

    for param in fn_item.sig.inputs.iter() {
        match param {
            FnArg::Receiver(_) => {
                return SourceLocation::unknown().jit_error_result("Unexpected receiver argument.");
            }
            FnArg::Typed(typed_param) => {
                let normalized_ty = modules.normalize_type_aliases(&typed_param.ty)?;
                let normalized_param = normalized_fn_arg_type(param, normalized_ty.clone())?;
                let ty = &normalized_ty;
                match ty {
                    syn::Type::Reference(_type_ref) => {
                        let tensor_input = TensorInput::new(
                            fn_name.clone(),
                            &normalized_param,
                            generic_vars,
                            stride_args,
                            spec_args,
                            primitives,
                            opt_hints,
                        )?;
                        statements.extend(tensor_input.generate_statements());
                        final_stmnt_args.push(tensor_input.generate_arg());
                        fn_entry.sig.inputs.extend(tensor_input.generate_args());
                        fn_params_concrete_types.push(ValidParamType::Tensor(TensorParamType {
                            element_type: tensor_input.element_type,
                            shape: tensor_input
                                .input_tensor_shape
                                .shape
                                .iter()
                                .map(|s| {
                                    if s == "- 1" {
                                        -1
                                    } else {
                                        s.parse::<i32>()
                                            .unwrap_or_else(|_| panic!("{}", s.to_string()))
                                    }
                                })
                                .collect::<Vec<i32>>(),
                        }))
                    }
                    syn::Type::Path(type_path) => {
                        let type_name = get_ident_from_path_expr(&syn::ExprPath {
                            attrs: vec![],
                            qself: None,
                            path: type_path.path.clone(),
                        })
                        .to_string();
                        if type_name == "MappedPartitionMut" {
                            let tensor_input = TensorInput::new(
                                fn_name.clone(),
                                &normalized_param,
                                generic_vars,
                                stride_args,
                                spec_args,
                                primitives,
                                opt_hints,
                            )?;
                            statements.extend(tensor_input.generate_statements());
                            final_stmnt_args.push(tensor_input.generate_arg());
                            fn_entry.sig.inputs.extend(tensor_input.generate_args());
                            fn_params_concrete_types.push(ValidParamType::Tensor(
                                TensorParamType {
                                    element_type: tensor_input.element_type,
                                    shape: tensor_input
                                        .input_tensor_shape
                                        .shape
                                        .iter()
                                        .map(|s| {
                                            if s == "- 1" {
                                                -1
                                            } else {
                                                s.parse::<i32>()
                                                    .unwrap_or_else(|_| panic!("{}", s.to_string()))
                                            }
                                        })
                                        .collect::<Vec<i32>>(),
                                },
                            ));
                            continue;
                        }
                        let var_name = get_fn_arg_var_name(param);
                        final_stmnt_args
                            .push(syn::parse2::<Expr>(var_name.parse().unwrap()).unwrap());

                        // Besides Tile and PointerTile, we only really know what to do with primitive types T.
                        // But since Tile<T, {[]}> <=> T and PointerTile<* mut T, {[]}> <=> * mut T,
                        // we ought to support it.
                        let TypeInstance::ElementType(element_type_inst) =
                            generic_vars.instantiate_type(ty, primitives)?
                        else {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "Unsupported type for entry point parameter: {type_path:#?}"
                            ));
                        };
                        let var_type = element_type_inst.rust_element_instance_ty;
                        let var_arg = syn::parse2::<syn::FnArg>(
                            format!("{var_name}: {var_type}").parse().unwrap(),
                        )
                        .unwrap();
                        fn_entry.sig.inputs.push(var_arg);
                        // Emit assume_div_by for scalar integer params with DivHints.
                        if let Some(hint) = scalar_hints.get(&var_name) {
                            if hint.divisor > 1 {
                                statements.push(TensorInput::get_assume_div_by(
                                    var_name.clone(),
                                    hint.divisor,
                                ));
                            }
                        }
                        fn_params_concrete_types.push(ValidParamType::Scalar(ScalarParamType {
                            element_type: var_type,
                        }));
                    }
                    syn::Type::Ptr(type_ptr) => {
                        let var_name = get_fn_arg_var_name(param);
                        final_stmnt_args
                            .push(syn::parse2::<Expr>(var_name.parse().unwrap()).unwrap());
                        let TypeInstance::PtrType(ptr_type_inst) =
                            generic_vars.instantiate_type(ty, primitives)?
                        else {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "Unsupported pointer type for entry point parameter: {type_ptr:#?}"
                            ));
                        };
                        let var_type = ptr_type_inst.instance_ty.to_token_stream().to_string();
                        let var_arg = syn::parse2::<syn::FnArg>(
                            format!("{var_name}: {var_type}").parse().unwrap(),
                        )
                        .unwrap();
                        fn_entry.sig.inputs.push(var_arg);
                        // Emit assume_div_by for raw pointer params with DivHints.
                        if let Some(hint) = scalar_hints.get(&var_name) {
                            if hint.divisor > 1 {
                                statements.push(TensorInput::get_assume_div_by(
                                    var_name.clone(),
                                    hint.divisor,
                                ));
                            }
                        }
                        fn_params_concrete_types.push(ValidParamType::Pointer(PointerParamType {
                            mutable: ptr_type_inst.is_mutable,
                            element_type: var_type,
                        }));
                    }
                    _ => {
                        let var_name = get_fn_arg_var_name(param);
                        final_stmnt_args
                            .push(syn::parse2::<Expr>(var_name.parse().unwrap()).unwrap());
                        fn_entry.sig.inputs.push(param.clone());
                    }
                }
            }
        }
    }

    // Final statement is to call the function for which we generated an entry point.
    let generic_args = generic_vars.ordered_param_vars.join(", ");
    let final_stmnt_args_str = final_stmnt_args
        .iter()
        .map(|x| x.to_token_stream().to_string())
        .collect::<Vec<_>>();
    let unsafety_str = if fn_item.sig.unsafety.is_some() {
        "unsafe"
    } else {
        ""
    };
    let final_stmnt = syn::parse2::<syn::Stmt>(
        format!(
            "{unsafety_str} {{ {fn_impl_name}::<{generic_args}>({}) }};",
            final_stmnt_args_str.join(",")
        )
        .parse()
        .unwrap(),
    )
    .unwrap();
    statements.push(final_stmnt);
    fn_entry.block.stmts = statements;

    // This source code is not visible.
    let mut visitor = SpanSetter::new(Span::call_site());
    visitor.visit_item_fn_mut(&mut fn_entry);

    Ok((
        fn_entry,
        Validator {
            params: fn_params_concrete_types,
            launch_checks: Vec::new(),
        },
    ))
}

/// Parsed tensor shape information from generic parameters or const expressions.
pub(crate) struct InputTensorShape {
    #[expect(
        dead_code,
        reason = "Generic const generic array variable name, used in template expansion"
    )]
    generic_cga_var: Option<String>,
    shape_param: String,
    shape: Vec<String>,
}

/// Extracts the tensor shape from a type's generic arguments, resolving const generics.
pub fn get_tensor_shape(
    ty: &syn::Type,
    generic_vars: &GenericVars,
) -> Result<InputTensorShape, JITError> {
    if let syn::Type::Reference(type_ref) = ty {
        return get_tensor_shape(&type_ref.elem, generic_vars);
    }
    // We assume this is a variadic type.
    let (_type_ident, type_generic_args) = get_ident_generic_args(ty);
    let mut shape: Option<InputTensorShape> = None;
    for generic_arg in &type_generic_args.args {
        match generic_arg {
            GenericArgument::Type(syn::Type::Path(type_path)) => {
                // Currently, this is either shape or element_type
                let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                if shape.is_none() && generic_vars.inst_array.contains_key(&last_ident) {
                    // This is something like Shape<D> for const generic array D: [i32; N].
                    let array_instance = generic_vars.inst_array.get(&last_ident).unwrap();
                    shape = Some(InputTensorShape {
                        generic_cga_var: Some(last_ident.clone()),
                        shape_param: last_ident,
                        shape: array_instance.iter().map(|elem| elem.to_string()).collect(),
                    });
                }
            }
            GenericArgument::Const(Expr::Block(block_expr)) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                // This is something like Tensor<E, {[...]}>
                if block_expr.block.stmts.len() != 1 {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Expected exactly 1 statement in block expression, got {}",
                        block_expr.block.stmts.len()
                    ));
                }
                let statement = &block_expr.block.stmts[0];
                let Stmt::Expr(statement_expr, _) = statement else {
                    return SourceLocation::unknown()
                        .jit_error_result("Unexpected block expression.");
                };
                match statement_expr {
                    Expr::Array(array_expr) => {
                        // This is something like Tensor<E, {[1, 2, -1]}>
                        let mut _shape = vec![];
                        for elem in &array_expr.elems {
                            match elem {
                                Expr::Lit(lit) => {
                                    let val = match &lit.lit {
                                        Lit::Int(int_lit) => int_lit.to_string(),
                                        _ => {
                                            return SourceLocation::unknown().jit_error_result(
                                                &format!(
                                            "Unexpected array element {elem:#?} in {array_expr:#?}"
                                        ),
                                            )
                                        }
                                    };
                                    _shape.push(val);
                                }
                                Expr::Unary(unary_expr) => {
                                    _shape.push(unary_expr.to_token_stream().to_string());
                                }
                                Expr::Path(path) => {
                                    let ident = get_ident_from_path_expr(path);
                                    match generic_vars.inst_i32.get(ident.to_string().as_str()) {
                                        Some(val) => _shape.push(val.to_string()),
                                        None => {
                                            return SourceLocation::unknown().jit_error_result(
                                                &format!("Undefined generic parameter {ident}"),
                                            );
                                        }
                                    }
                                }
                                Expr::Index(index) => {
                                    let Expr::Path(path) = index.expr.as_ref() else {
                                        return SourceLocation::unknown().jit_error_result(
                                            &format!(
                                                "Unexpected const generic array base {elem:#?}"
                                            ),
                                        );
                                    };
                                    let ident = get_ident_from_path_expr(path);
                                    let Some(shape) =
                                        generic_vars.inst_array.get(ident.to_string().as_str())
                                    else {
                                        return SourceLocation::unknown().jit_error_result(
                                            &format!(
                                                "Undefined const generic array parameter {ident}"
                                            ),
                                        );
                                    };
                                    let i = crate::types::parse_signed_literal_as_i32(&index.index);
                                    let Some(dim) = shape.get(i as usize) else {
                                        return SourceLocation::unknown()
                                                .jit_error_result(&format!(
                                                    "Index {i} out of bounds for const generic array `{ident}` of length {}",
                                                    shape.len()
                                                ));
                                    };
                                    _shape.push(dim.to_string());
                                }
                                _ => {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "Unexpected array element {elem:#?} in {array_expr:#?}"
                                    ))
                                }
                            }
                        }
                        if shape.is_none() {
                            shape = Some(InputTensorShape {
                                generic_cga_var: None,
                                shape_param: block_expr.block.to_token_stream().to_string(),
                                shape: _shape,
                            });
                        }
                    }
                    Expr::Repeat(repeat_expr) => {
                        // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                        let thing_to_repeat = repeat_expr.expr.to_token_stream().to_string();
                        match &*repeat_expr.len {
                            Expr::Path(len_path) => {
                                // This is something like Tensor<E, {[-1; N]}>
                                let num_rep_var = len_path.to_token_stream().to_string();
                                if generic_vars.get_i32(&num_rep_var).is_none() {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "Expected instance for generic argument {}",
                                        num_rep_var
                                    ));
                                }
                                let num_rep = generic_vars.get_i32(&num_rep_var).unwrap();
                                if shape.is_none() {
                                    shape = Some(InputTensorShape {
                                        generic_cga_var: None,
                                        shape_param: block_expr.block.to_token_stream().to_string(),
                                        shape: vec![thing_to_repeat; num_rep as usize],
                                    });
                                }
                            }
                            Expr::Lit(len_lit) => {
                                // This is something like Tensor<E, {[-1; 3]}>
                                let num_rep: u32 = len_lit
                                    .to_token_stream()
                                    .to_string()
                                    .parse::<u32>()
                                    .unwrap();
                                if shape.is_none() {
                                    shape = Some(InputTensorShape {
                                        generic_cga_var: None,
                                        shape_param: block_expr.block.to_token_stream().to_string(),
                                        shape: vec![thing_to_repeat; num_rep as usize],
                                    });
                                }
                            }
                            _ => {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "Unexpected repeat expression: {repeat_expr:#?}"
                                ))
                            }
                        }
                    }
                    _ => {
                        return SourceLocation::unknown()
                            .jit_error_result("Unexpected block expression.")
                    }
                }
            }
            _ => {}
        }
    }
    let Some(res) = shape else {
        return SourceLocation::unknown()
            .jit_error_result(&format!("Unable to get shape for {ty:#?}"));
    };
    Ok(res)
}

/// Returns the CUDA Tile element type string for a tensor type's first generic argument.
pub fn get_tensor_element_type(
    ty: &syn::Type,
    primitives: &HashMap<(String, String), ItemImpl>,
    generic_vars: &GenericVars,
) -> Result<Option<String>, JITError> {
    let (_type_ident, type_generic_args) = get_ident_generic_args(ty);
    let Some(GenericArgument::Type(syn::Type::Path(type_path))) = type_generic_args.args.first()
    else {
        return SourceLocation::unknown().jit_error_result("Expected generic argument type path.");
    };
    let ident = &type_path.path.segments.last().unwrap().ident;
    get_element_type(ident, primitives, generic_vars)
}

/// Resolves an element type ident to its CUDA Tile type string via primitives or generic vars.
pub fn get_element_type(
    ident: &Ident,
    primitives: &HashMap<(String, String), ItemImpl>,
    generic_vars: &GenericVars,
) -> Result<Option<String>, JITError> {
    let ident_str = ident.to_string();
    #[allow(unused_assignments)]
    let mut element_type: Option<String> = None;
    if get_primitives_attrs("ElementType", &ident_str, primitives).is_some() {
        element_type = Some(ident_str);
    } else {
        let Some(element_type_inst) = generic_vars.inst_types.get(&ident_str) else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "Unable to instantiate element type from ident {ident:#?}"
            ));
        };
        element_type = Some(element_type_inst.clone());
    }
    Ok(element_type)
}

/// A visitor that walks the AST and overrides every span it encounters.
pub struct SpanSetter {
    pub target_span: Span,
}

impl SpanSetter {
    pub fn new(span: Span) -> Self {
        Self { target_span: span }
    }
}

impl VisitMut for SpanSetter {
    fn visit_span_mut(&mut self, span: &mut Span) {
        *span = self.target_span;
    }
}
