/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Kernel launcher generation.
//!
//! This module generates launcher functions for GPU kernel entry points
//! with support for direct invocation via `IntoDeviceOp` arguments.
//! These launchers provide a type-safe interface for invoking
//! CUDA Tile kernels from Rust code.
//!
//! ## Overview
//!
//! For each function marked with `#[cutile::entry]`, this module (with
//! `_module.rs`, which emits the struct and its `TileKernel` / `IntoFuture`
//! impls) generates:
//!
//! - a launcher struct `MyKernel<.., DI>` implementing `DeviceOp` (the launch),
//!   `TileKernel` (`.generics()`, `.grid()`, `.compile_options()`), and
//!   `GraphNode` when its argument op `DI` is one
//! - the public launcher fn `my_kernel(..)`, whose every argument is an
//!   `IntoDeviceOp`: a materialized value (`Tensor`, `Arc<Tensor>`, `&Tensor`,
//!   `&TensorView`, a `Partition`, a scalar) or a lazy `DeviceOp` producing one
//! - `my_kernel_apply(..)`, taking the already-materialized argument tuple
//!   (used by `cutile::api`)
//!
//! Executing the launcher:
//!
//! 1. **Executes** the argument op and **infers** generic parameters from the inputs
//! 2. **Compiles** the specialization (with caching)
//! 3. **Validates** element types, partition shapes, declared preconditions,
//!    hoisted launch checks, and the inferred grid against the compiled kernel
//! 4. **Launches** the kernel as a device operation
//! 5. **Returns** the arguments, recovered to the types the caller passed
//!
//! ## Generated Launcher Structure
//!
//! ```text
//! // For this kernel:
//! #[cutile::entry]
//! fn my_kernel<T: ElementType, const N: i32>(
//!     output: &mut Tensor<T, {[N]}>,
//!     input: &Tensor<T, {[-1]}>,
//! ) { }
//!
//! // Generates this launcher (the struct takes the KernelOutput / KernelInput
//! // intermediates and the argument op as type parameters):
//! pub fn my_kernel<T: Send + DType, _A0, _Q0, _A1, _S1>(
//!     output: _A0,
//!     input: _A1,
//! ) -> MyKernel<T, _S1, _Q0, KernelArgs<(
//!         PrepareOutput<<_A0 as IntoDeviceOp<_Q0>>::Op, T>,
//!         PrepareInput<<_A1 as IntoDeviceOp<_S1>>::Op, T>,
//!     )>>
//! where
//!     _A0: IntoDeviceOp<_Q0>, _Q0: KernelOutput<T>,
//!     _A1: IntoDeviceOp<_S1>, _S1: KernelInput<T>,
//! { /* wraps each argument in its op and hands the tuple op to MyKernel::launch */ }
//!
//! // and the materialized-tuple form used internally by cutile::api:
//! pub fn my_kernel_apply<T: Send + DType>(
//!     input: (Partition<Tensor<T>>, Arc<Tensor<T>>),
//! ) -> MyKernel<T, Arc<Tensor<T>>, Partition<Tensor<T>>, Value<(Partition<Tensor<T>>, Arc<Tensor<T>>)>>
//! ```
//!
//! `MyKernel<..>` executes to `(<_Q0 as KernelOutput<T>>::Returned, <_S1 as
//! KernelInput<T>>::Returned)` — the argument types the caller passed in.
//!
//! ## Generic Parameter Inference
//!
//! The launcher automatically infers generic parameters from input tensors:
//!
//! - **Type parameters** (`T`) - Inferred from tensor element types
//! - **Const scalars** (`const N: i32`) - Inferred from tensor shapes
//! - **Const arrays** (`const S: [i32; N]`) - Inferred from partition shapes
//!
//! ## Parameter Transformation
//!
//! Kernel parameters are transformed for the launcher:
//!
//! - `&mut Tensor<T, S>` → `impl KernelOutput<T>` (accepts `Partition<Tensor<T>>` or `Partition<&mut Tensor<T>>`)
//! - `&Tensor<T, S>` → `impl KernelInput<T>` (accepts `Tensor<T>`, `Arc<Tensor<T>>`, or `&Tensor<T>`)
//! - `*mut T` → `DevicePointer<T>` (unsafe only)
//! - Scalars remain unchanged
//!
//! Both `&Tensor` and `&mut Tensor` params return the same type that was passed in
//! via `KernelInput::recover` / `KernelOutput::recover`.
//!
//! ## Grid Inference
//!
//! Launch grid dimensions are automatically inferred from partitioned tensors.
//! If multiple partitions exist, their grids must match.

use cutile_compiler::kernel_naming::KernelNaming;
use cutile_compiler::syn_utils::*;
use cutile_compiler::types::get_ptr_type;
use proc_macro2::Ident;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use std::collections::HashMap;
use syn::{
    parse_quote, AngleBracketedGenericArguments, Expr, ExprBlock, ExprCall, FnArg, GenericArgument,
    GenericParam, Generics, ImplItemFn, ItemFn, Lit, PatType, Stmt, Type, TypeParam, TypeReference,
    WherePredicate,
};

use crate::error::{Error, SpannedError};

/// Classification of generic parameter types.
///
/// Used to determine how to handle and infer different kinds of generic parameters.
#[derive(Debug, PartialOrd, PartialEq)]
pub(crate) enum SupportedGenericType {
    /// Type parameter (e.g., `T: ElementType`)
    TypeParam,
    /// Const scalar parameter (e.g., `const N: i32`)
    ConstScalar,
    /// Const array parameter (e.g., `const S: [i32; 2]`)
    ConstArray,
    /// Unknown or unsupported parameter type
    Unknown,
}

/// Tracks required generic parameters and their inference state.
///
/// Manages the set of generic parameters that need to be inferred from
/// kernel arguments. Tracks which parameters have been successfully inferred
/// and generates the final generic argument list.
#[derive(Debug)]
pub(crate) struct RequiredGenerics {
    /// Names of generic parameters in order
    names: Vec<String>,
    /// Names of compile-time generic type parameters required by launcher functions.
    /// This lets generated launcher functions support type param inference via impls of the DType trait.
    launcher_type_params: Vec<String>,
    /// Type annotations for const generic parameters (None for type params)
    types: Vec<Option<Type>>,
    /// Inferred expressions for each generic parameter
    expressions: HashMap<String, Option<String>>,
}

impl RequiredGenerics {
    fn new(generics: &Generics) -> Self {
        let req_generics = get_supported_generic_params(generics);
        let (names, types): (Vec<String>, _) = req_generics.into_iter().unzip();
        let mut expressions: HashMap<String, Option<String>> = HashMap::new();
        for name in &names {
            expressions.insert(name.clone(), None);
        }
        Self {
            names,
            launcher_type_params: vec![],
            types,
            expressions,
        }
    }
    /// Returns `true` if the given generic parameter has not yet been inferred.
    pub(crate) fn is_required(&self, s: &str) -> bool {
        matches!(self.expressions.get(s), None | Some(None))
    }
    /// Builds a runtime expression string that concatenates all inferred generic values.
    pub(crate) fn to_expr_str(&self) -> String {
        let mut res = vec![];
        for name in &self.names {
            let Some(Some(expr_str)) = self.expressions.get(name) else {
                // TODO (hme): Some generics not assigned in function signature can't be inferred.
                return format!("panic!(\"Failed to infer value for generic parameter {name}\")");
            };
            res.push(expr_str.clone());
        }
        format!("vec![{}].concat()", res.join(","))
    }

    /// Classifies the generic parameter with the given name.
    pub(crate) fn get_ty(&self, name: &str) -> SupportedGenericType {
        let Some(index) = self.names.iter().position(|n| n == name) else {
            return SupportedGenericType::Unknown;
        };
        let Some(ty) = &self.types[index] else {
            // If there is no type, it's not a const VAR_NAME: Type.
            // It is a type param.
            return SupportedGenericType::TypeParam;
        };
        match ty {
            Type::Array(_) => SupportedGenericType::ConstArray,
            _ => SupportedGenericType::ConstScalar,
        }
    }
    /// Builds a `Generics` containing only the type parameters needed by the launcher.
    pub(crate) fn get_required_generics(&self) -> Generics {
        let mut type_params = vec![];
        for name in &self.names {
            let is_launcher_type_param = self.launcher_type_params.contains(name);
            if is_launcher_type_param && self.get_ty(name) == SupportedGenericType::TypeParam {
                type_params.push(format!("{}: Send + DType", name.clone()));
            }
        }
        syn::parse2::<Generics>(format!("<{}>", type_params.join(", ")).parse().unwrap()).unwrap()
    }
    /// Builds angle-bracketed generic arguments for the launcher type parameters.
    pub(crate) fn get_generic_args(&self) -> AngleBracketedGenericArguments {
        let mut type_params = vec![];
        for name in &self.names {
            let is_launcher_type_param = self.launcher_type_params.contains(name);
            if is_launcher_type_param && self.get_ty(name) == SupportedGenericType::TypeParam {
                type_params.push(name.to_string());
            }
        }
        syn::parse2::<AngleBracketedGenericArguments>(
            format!("<{}>", type_params.join(", ")).parse().unwrap(),
        )
        .unwrap()
    }
}

/// Joins a vector of strings into a nested cons-cell tuple structure.
///
/// Converts a list of values into a right-associated nested tuple structure
/// used by async combinators. For example, `["a", "b", "c"]` becomes `"(a, (b, c))"`.
///
/// ## Parameters
///
/// - `vals`: Vector of string representations of values to join
///
/// ## Returns
///
/// A string representing the nested tuple structure
///
/// ## Examples
///
/// - `[]` → `"()"`
/// - `["a"]` → `"a"`
/// - `["a", "b"]` → `"(a, b)"`
/// - `["a", "b", "c"]` → `"(a, (b, c))"`
pub fn join_as_cons_tuple(vals: &[String]) -> String {
    if vals.is_empty() {
        return "()".to_string();
    }
    if vals.len() == 1 {
        return vals[0].clone();
    };
    let mut cons = vals.last().expect("Impossible").clone();
    for i in (0..vals.len() - 1).rev() {
        cons = format!("({}, {})", vals[i], cons);
    }
    cons
}

/// Wraps an expression in a `value()` call if needed for device-operation combinators.
///
/// Helper function used when building launcher code. If `wrap_as_val` is true,
/// wraps the expression in `value()` to create a device-operation value.
///
/// ## Parameters
///
/// - `expr`: The expression string to potentially wrap
/// - `wrap_as_val`: Whether to wrap the expression in `value()`
///
/// ## Returns
///
/// Either the original expression or `value(expr)`
fn zippable(expr: &str, wrap_as_val: bool) -> String {
    if !wrap_as_val {
        return expr.to_string();
    }
    format!("value({})", expr)
}

/// Generates async code to zip inputs into a cons-cell tuple structure.
///
/// Creates a block of statements that uses the `zip!` macro to combine async
/// values into a nested tuple. Each input is zipped with the accumulator in
/// reverse order to create a right-associated structure.
///
/// ## Parameters
///
/// - `inputs`: Vector of input expression strings to zip
/// - `var_name`: Name of the variable to bind the zipped result to
/// - `wrap_as_val`: Whether to wrap inputs in `value()` calls
///
/// ## Returns
///
/// An expression block containing the zip statements
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`). See `zip_and_then_flatten`
/// for the actively used version.
#[allow(dead_code)]
pub fn zip_cons(inputs: &[String], var_name: &str, wrap_as_val: bool) -> ExprBlock {
    let mut zip_block = syn::parse2::<ExprBlock>(quote! {{
    }})
    .unwrap();
    if inputs.is_empty() {
        return zip_block;
    }
    let mut i = inputs.len() - 1;
    zip_block.block.stmts.push(parse_stmt(format!(
        "let {var_name} = {};",
        zippable(&inputs[i], wrap_as_val)
    )));
    while i != 0 {
        i -= 1;
        zip_block.block.stmts.push(parse_stmt(format!(
            "let {var_name} = zip!({}, {var_name});",
            zippable(&inputs[i], wrap_as_val)
        )));
    }
    zip_block
}

/// Generates launcher code to zip inputs, then flatten them into a flat tuple.
///
/// Similar to `zip_cons` but adds a final `and_then` step that flattens the
/// nested cons-cell structure into a regular flat tuple. This is the standard
/// approach used for op-based launcher helpers.
///
/// ## Parameters
///
/// - `inputs`: Vector of input expression strings to zip
/// - `var_name`: Name of the variable to bind the result to
/// - `wrap_as_val`: Whether to wrap inputs in `value()` calls
///
/// ## Returns
///
/// An expression block containing:
/// 1. Zip statements building a nested tuple
/// 2. An `and_then` that flattens it to a flat tuple
///
/// ## Example
///
/// For inputs `["a", "b", "c"]`, generates code equivalent to:
/// ```rust,ignore
/// let result = zip!(a, zip!(b, c));
/// let result = result.then(|(a, (b, c))| value((a, b, c)));
/// ```
pub fn zip_and_then_flatten(inputs: &[String], var_name: &str, wrap_as_val: bool) -> ExprBlock {
    let mut zip_block = syn::parse2::<ExprBlock>(quote! {{
    }})
    .unwrap();
    if inputs.is_empty() {
        zip_block
            .block
            .stmts
            .push(parse_stmt(format!("let {var_name} = value(());")));
        return zip_block;
    }
    let mut i = inputs.len() - 1;
    zip_block.block.stmts.push(parse_stmt(format!(
        "let {var_name} = {};",
        zippable(&inputs[i], wrap_as_val)
    )));
    while i != 0 {
        i -= 1;
        zip_block.block.stmts.push(parse_stmt(format!(
            "let {var_name} = zip!({}, {var_name});",
            zippable(&inputs[i], wrap_as_val)
        )));
    }
    zip_block.block.stmts.push(parse_stmt(format!(
        r#"
            let {var_name} = {var_name}.then(|{}| {{
                value({})
            }});
        "#,
        join_as_cons_tuple(inputs),
        to_tuple_string(inputs)
    )));
    zip_block
}

/// Generates a type alias name for a kernel argument.
///
/// Helper function for creating unique type alias names for kernel launcher arguments.
/// For example, generates `"FooKernelArg0"`, `"FooKernelArg1"`, etc.
///
/// ## Parameters
///
/// - `launcher_name`: The name of the kernel launcher
/// - `i`: The argument index
///
/// ## Returns
///
/// A string like `"LauncherNameArgN"`
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`).
#[allow(dead_code)]
fn kernel_arg_alias(launcher_name: &str, i: usize) -> String {
    format!("{launcher_name}Arg{i}")
}

/// Generates a type alias for kernel launcher arguments.
///
/// Creates a type alias that represents the tuple of all kernel arguments,
/// including any generic parameters from the kernel function signature.
///
/// ## Parameters
///
/// - `generic_args`: Generic parameters from the kernel (e.g., `<T, const N: i32>`)
/// - `arg_tys`: Types of all kernel arguments
/// - `_launcher_name`: Name of the launcher (currently unused)
/// - `launcher_args_name`: Name for the type alias (e.g., `"FooKernelArgs"`)
///
/// ## Returns
///
/// A tuple of:
/// 1. The constructed type (e.g., `FooKernelArgs<T, N>`)
/// 2. The type alias definition as tokens (e.g., `type FooKernelArgs<T, N> = (Tensor<T, [N]>, i32);`)
pub fn generate_launcher_arg_types(
    generic_args: &AngleBracketedGenericArguments,
    arg_tys: &Vec<Type>,
    _launcher_name: &str,
    launcher_args_name: &str,
) -> (Type, TokenStream2) {
    let launcher_args_ident = Ident::new(launcher_args_name, Span::call_site());
    let launcher_args_type: Type = if !generic_args.args.is_empty() {
        parse_quote! { #launcher_args_ident #generic_args }
    } else {
        parse_quote! { #launcher_args_ident }
    };
    (
        launcher_args_type.clone(),
        quote! { type #launcher_args_type = ( #(#arg_tys,)* ); },
    )
}

/// Converts a vector of strings into a flat tuple string representation.
///
/// Helper function that formats argument names into a comma-separated tuple string.
/// Used when generating launcher code.
///
/// ## Parameters
///
/// - `args`: Vector of argument names/expressions
///
/// ## Returns
///
/// A string like `"(arg1, arg2, arg3,)"` with trailing comma
///
/// ## Example
///
/// `["x", "y", "z"]` → `"(x, y, z,)"`
pub fn to_tuple_string(args: &[String]) -> String {
    format!(
        "({})",
        args.iter()
            .map(|s| format!("{s},"))
            .collect::<Vec<String>>()
            .join("")
    )
}

/// Generates the complete launcher code for a GPU kernel.
///
/// This is the main function that transforms a kernel function (marked with `#[entry]`)
/// into launcher helpers that can be called from Rust code. It generates:
/// 1. Type aliases for kernel arguments
/// 2. A launcher struct implementing `DeviceOp`
/// 3. A direct launcher for materialized arguments
/// 4. Auto-wraps Tensor→Arc for &Tensor params
///
/// ## Parameters
///
/// - `item`: The kernel function AST
/// - `module_name`: Name of the module containing the kernel
/// - `function_name`: Name of the kernel function
/// - `function_entry_name`: MLIR entry point name for the kernel
/// - `launcher_name`: Name for the generated launcher struct
/// - `launcher_args_name`: Name for the generated argument type alias
///
/// ## Returns
///
/// A tuple of:
/// 1. `RequiredGenerics` - Information about generic parameters and constraints
/// 2. `(Type, TokenStream2)` - The launcher argument type and its definition
/// 3. `TokenStream2` - The complete launcher implementation
///
/// ## Generated Code Structure
///
/// For a kernel `fn my_kernel<T>(x: &mut Tensor<T, [128]>, y: &Tensor<T, [-1]>)`, generates:
/// ```rust,ignore
/// type MyKernelArgs<T> = (Arc<Tensor<T>>, Arc<Tensor<T>>);
/// pub struct MyKernel<T> {
///     args: MyKernelArgs<T>,
/// }
/// impl<T: DType> DeviceOp for MyKernel<T> {
///     type Output = ();
///     unsafe fn execute(mut self, ctx: &ExecutionContext) -> Self::Output {
///         // Kernel launch logic here
///     }
/// }
/// ```
pub fn generate_kernel_launcher(
    item: &ItemFn,
    module_name: &str,
    function_name: &str,
    function_entry_name: &str,
    launcher_name: &str,
    _launcher_args_name: &str,
    tile_rust_crate_root: &syn::Ident,
) -> Result<
    (
        RequiredGenerics,
        (Type, Type),
        TokenStream2,
        KernelInputInfo,
    ),
    Error,
> {
    let unsafety = item.sig.unsafety;
    let is_unsafe = unsafety.is_some();
    let launcher_ident = Ident::new(launcher_name, Span::call_site());
    let mut launcher_method = syn::parse2::<ImplItemFn>(quote! {
        unsafe fn execute(mut self, ctx: &ExecutionContext) -> Result<<Self as DeviceOp>::Output, DeviceError> {}
    })
    .unwrap();
    let mut specialization_method = syn::parse2::<ImplItemFn>(quote! {
        unsafe fn _resolve_specialization(
            mut self,
            ctx: &ExecutionContext,
        ) -> Result<Specialization<ModuleAstFn>, DeviceError> {}
    })
    .unwrap();

    // Generate launcher signature.
    let param_names = get_sig_param_names(&item.sig);
    let param_names_tuple_str = to_tuple_string(&param_names);
    let (input_types, _output_type) = get_sig_types(&item.sig, None);
    let mut stride_args = vec![];
    let mut spec_args: Vec<String> = vec![];
    let mut spec_ref_exprs: Vec<String> = vec![];
    let mut scalar_hint_exprs: Vec<String> = vec![];
    let mut scalar_hint_value_exprs: Vec<String> = vec![];
    let mut builder_statements = vec![];
    let mut launch_grid_expr_strs = vec![];
    let mut validator_statements = vec![];
    let mut arg_types: Vec<Type> = vec![];
    // Track element type per param (Some for &Tensor params, None for others).
    let mut param_element_types: Vec<Option<String>> = vec![];
    let mut tensor_param_kinds: HashMap<String, TensorParamKind> = HashMap::new();

    let mut required_generics: RequiredGenerics = RequiredGenerics::new(&item.sig.generics);
    for (i, ty) in input_types.iter().enumerate() {
        let var_name = &param_names[i];
        match ty {
            Type::Reference(ref_ty) => {
                let res = get_tensor_code(i, var_name, ref_ty, &mut required_generics)?;
                let tensor_kind = if ref_ty.mutability.is_some() {
                    TensorParamKind::Output
                } else {
                    TensorParamKind::Input
                };
                tensor_param_kinds.insert(var_name.clone(), tensor_kind);
                arg_types.push(res.fn_arg.ty.as_ref().clone());
                param_element_types.push(res.element_type_name);
                stride_args.push(res.stride_expr_str);
                spec_args.push(res.spec_expr_str);
                spec_ref_exprs.push(res.spec_ref_expr_str);
                builder_statements.extend(res.builder_statements);
                launch_grid_expr_strs.extend(res.launch_grid_expr_strs);
                validator_statements.extend(res.validator_statements.block.stmts);
            }
            Type::Path(path_ty) => {
                let ident = get_ident_from_path(&path_ty.path);
                let type_name = ident.to_string();
                if type_name == "MappedPartitionMut" {
                    let ref_ty: TypeReference = parse_quote!(&mut #path_ty);
                    let res = get_tensor_code(i, var_name, &ref_ty, &mut required_generics)?;
                    tensor_param_kinds.insert(var_name.clone(), TensorParamKind::Output);
                    arg_types.push(res.fn_arg.ty.as_ref().clone());
                    param_element_types.push(res.element_type_name);
                    stride_args.push(res.stride_expr_str);
                    spec_args.push(res.spec_expr_str);
                    spec_ref_exprs.push(res.spec_ref_expr_str);
                    builder_statements.extend(res.builder_statements);
                    launch_grid_expr_strs.extend(res.launch_grid_expr_strs);
                    validator_statements.extend(res.validator_statements.block.stmts);
                    continue;
                }
                arg_types.push(syn::parse2::<Type>(type_name.parse().unwrap()).unwrap());
                if required_generics.is_required(&type_name) {
                    required_generics
                        .launcher_type_params
                        .push(type_name.clone());
                    // T is DType, so we can use T::DTYPE.as_str();
                    required_generics.expressions.insert(
                        type_name.clone(),
                        Some(format!("vec![{type_name}::DTYPE.as_str().to_string()]")),
                    );
                }
                builder_statements.push(parse_stmt(format!("kernel_launch.push_arg({var_name});")));
                // For integer scalar params, auto-compute DivHint at launch time.
                if cutile_compiler::specialization::is_integer_scalar(&type_name) {
                    scalar_hint_exprs.push(format!(
                        r#"("{var_name}".to_string(), {tile_rust_crate_root}::cutile_compiler::specialization::DivHint::from_value({var_name} as i32))"#
                    ));
                    scalar_hint_value_exprs.push(format!(
                        "{tile_rust_crate_root}::cutile_compiler::specialization::DivHint::from_value({var_name} as i32)"
                    ));
                }
                param_element_types.push(None);
            }
            Type::Ptr(ptr_type) => {
                // Let's require this to be unsafe, even though all unsafe operations on pointers
                // are marked as such.
                if !is_unsafe {
                    return ptr_type
                        .err("Pointers can only be used in unsafe kernel entry points.");
                }
                let ptr_str = ptr_type.to_token_stream().to_string();
                let Some((_is_mutable, type_name)) = get_ptr_type(&ptr_str) else {
                    return ptr_type.err(&format!("Unexpected pointer type: {}", ptr_str));
                };
                // `*const` and `*mut` params launch identically: the host
                // passes a device address either way; constness is a
                // device-side type discipline, not a launch property.
                arg_types.push(
                    syn::parse2::<Type>(format!("DevicePointer<{}>", type_name).parse().unwrap())
                        .unwrap(),
                );
                if required_generics.is_required(&type_name) {
                    required_generics
                        .launcher_type_params
                        .push(type_name.clone());
                    // T is DType, so we can use T::DTYPE.as_str();
                    required_generics.expressions.insert(
                        type_name.clone(),
                        Some(format!("vec![{type_name}::DTYPE.as_str().to_string()]")),
                    );
                }
                builder_statements.push(parse_stmt(format!(
                    "unsafe {{ kernel_launch.push_device_ptr({var_name}.cu_deviceptr()); }}"
                )));
                scalar_hint_exprs.push(format!(
                    r#"("{var_name}".to_string(), {tile_rust_crate_root}::cutile_compiler::specialization::DivHint::from_ptr({var_name}.cu_deviceptr()))"#
                ));
                scalar_hint_value_exprs.push(format!(
                    "{tile_rust_crate_root}::cutile_compiler::specialization::DivHint::from_ptr({var_name}.cu_deviceptr())"
                ));
                // The specialization's pointee type must be the `DevicePointer<T>`
                // element type: a user `.generics(..)` list can widen `T`, and the
                // kernel would then index the buffer with the wider element size.
                let pointee_ty = syn::parse_str::<Type>(&type_name).unwrap();
                validator_statements.extend(
                    syn::parse2::<ExprBlock>(quote! {{
                        {
                            let ValidParamType::Pointer(pointer_validator) = &validator.params[#i] else {
                                panic!("Unexpected validator type {:#?}", &validator.params[#i]);
                            };
                            let given_dtype = <#pointee_ty as DType>::DTYPE.as_str();
                            let compiled_dtype = #tile_rust_crate_root::cutile_compiler::types::get_ptr_type(&pointer_validator.element_type)
                                .map(|(_, pointee)| pointee)
                                .unwrap_or_else(|| pointer_validator.element_type.clone());
                            kernel_launch_assert_with(compiled_dtype == given_dtype, || format!("{} element type mismatch: the kernel was specialized for a `*{}` pointer but the launch passes a `DevicePointer<{}>` (check the `.generics(..)` list)", #var_name, compiled_dtype, given_dtype))?;
                        }
                    }})
                    .unwrap()
                    .block
                    .stmts,
                );
                param_element_types.push(None);
            }
            _ => {
                return ty.err("Unable to generate launcher: unsupported parameter type.");
            }
        }
    }
    let preconditions = parse_metadata_preconditions(item)?;
    validator_statements.extend(metadata_precondition_validation_stmts(
        &preconditions,
        &tensor_param_kinds,
    )?);

    // Build KernelInput metadata: which params are &Tensor (Arc) and need
    // KernelInput type params on the launcher struct.
    let mut ki_type_param_names: Vec<String> = vec![];
    let mut ki_element_type_names: Vec<String> = vec![];
    let mut ki_param_idx: Vec<Option<usize>> = vec![];
    // Build KernelOutput metadata: which params are &mut Tensor (Partition).
    let mut ko_type_param_names: Vec<String> = vec![];
    let mut ko_element_type_names: Vec<String> = vec![];
    let mut ko_param_idx: Vec<Option<usize>> = vec![];
    for (i, ty) in arg_types.iter().enumerate() {
        let ty_str = ty.to_token_stream().to_string();
        if ty_str.starts_with("Arc <") {
            let idx = ki_type_param_names.len();
            ki_type_param_names.push(format!("_K{}", idx));
            ki_element_type_names.push(
                param_element_types[i]
                    .clone()
                    .expect("&Tensor param must have element type"),
            );
            ki_param_idx.push(Some(idx));
            ko_param_idx.push(None);
        } else if ty_str.contains("Partition") {
            let idx = ko_type_param_names.len();
            ko_type_param_names.push(format!("_P{}", idx));
            // Extract element type from "tensor :: Partition < tensor :: Tensor < T > >"
            let elem = ty_str
                .split("Tensor <")
                .nth(1)
                .and_then(|s| s.split('>').next())
                .map(|s| s.trim().to_string())
                .expect("Partition param must have element type");
            ko_element_type_names.push(elem);
            ko_param_idx.push(Some(idx));
            ki_param_idx.push(None);
        } else {
            ki_param_idx.push(None);
            ko_param_idx.push(None);
        }
    }
    // Build the recovered return tuple expression:
    // KernelInput params call recover, KernelOutput params call recover, others pass through.
    let recovered_fields: Vec<String> = param_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            if let Some(ki_idx) = ki_param_idx[i] {
                let ki_name = &ki_type_param_names[ki_idx];
                let elem = &ki_element_type_names[ki_idx];
                format!("<{ki_name} as KernelInput<{elem}>>::recover({name})")
            } else if let Some(ko_idx) = ko_param_idx[i] {
                let ko_name = &ko_type_param_names[ko_idx];
                let elem = &ko_element_type_names[ko_idx];
                format!("<{ko_name} as KernelOutput<{elem}>>::recover({name})")
            } else {
                name.clone()
            }
        })
        .collect();
    let recovered_tuple_str = to_tuple_string(&recovered_fields);
    let kernel_input_info = KernelInputInfo {
        type_param_names: ki_type_param_names,
        element_type_names: ki_element_type_names,
        param_kernel_input_idx: ki_param_idx.clone(),
        ko_type_param_names: ko_type_param_names.clone(),
        ko_element_type_names: ko_element_type_names.clone(),
        param_kernel_output_idx: ko_param_idx.clone(),
        recovered_tuple_str: recovered_tuple_str.clone(),
    };

    // Build stored and returned arg type lists for KernelInput parameterization.
    // Stored types are what DI produces and execute() receives.
    // Returned types are what execute() returns after calling recover.
    let ki_info = &kernel_input_info;
    let stored_arg_types: Vec<Type> = arg_types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            if let Some(ki_idx) = ki_info.param_kernel_input_idx[i] {
                let ki_name = &ki_info.type_param_names[ki_idx];
                let elem = &ki_info.element_type_names[ki_idx];
                syn::parse_str::<Type>(&format!("<{ki_name} as KernelInput<{elem}>>::Stored"))
                    .unwrap()
            } else if let Some(ko_idx) = ki_info.param_kernel_output_idx[i] {
                let ko_name = &ki_info.ko_type_param_names[ko_idx];
                let elem = &ki_info.ko_element_type_names[ko_idx];
                syn::parse_str::<Type>(&format!("<{ko_name} as KernelOutput<{elem}>>::Stored"))
                    .unwrap()
            } else {
                ty.clone()
            }
        })
        .collect();
    let returned_arg_types: Vec<Type> = arg_types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            if let Some(ki_idx) = ki_info.param_kernel_input_idx[i] {
                let ki_name = &ki_info.type_param_names[ki_idx];
                let elem = &ki_info.element_type_names[ki_idx];
                syn::parse_str::<Type>(&format!("<{ki_name} as KernelInput<{elem}>>::Returned"))
                    .unwrap()
            } else if let Some(ko_idx) = ki_info.param_kernel_output_idx[i] {
                let ko_name = &ki_info.ko_type_param_names[ko_idx];
                let elem = &ki_info.ko_element_type_names[ko_idx];
                syn::parse_str::<Type>(&format!("<{ko_name} as KernelOutput<{elem}>>::Returned"))
                    .unwrap()
            } else {
                ty.clone()
            }
        })
        .collect();

    // Build inline tuple types (no type alias — simpler with associated types).
    let stored_args_type: Type = parse_quote! { ( #(#stored_arg_types,)* ) };
    let returned_args_type: Type = parse_quote! { ( #(#returned_arg_types,)* ) };
    let stored_args_type_str = stored_args_type.to_token_stream().to_string();

    // Prepare generics.
    let generic_params = required_generics.get_required_generics();
    let generic_args = required_generics.get_generic_args();

    // Add KernelInput (_K) and KernelOutput (_P) type params to struct generics.
    let mut struct_generics = generic_params.clone();
    for (ki_idx, ki_name) in ki_info.type_param_names.iter().enumerate() {
        let elem = &ki_info.element_type_names[ki_idx];
        struct_generics.params.push(
            syn::parse_str::<GenericParam>(&format!("{ki_name}: KernelInput<{elem}>")).unwrap(),
        );
    }
    for (ko_idx, ko_name) in ki_info.ko_type_param_names.iter().enumerate() {
        let elem = &ki_info.ko_element_type_names[ko_idx];
        struct_generics.params.push(
            syn::parse_str::<GenericParam>(&format!("{ko_name}: KernelOutput<{elem}>")).unwrap(),
        );
    }
    let device_op_param: GenericParam = parse_quote! { DI: DeviceOp<Output=#stored_args_type> };
    struct_generics.params.push(device_op_param.clone());

    let mut struct_args = generic_args.clone();
    for ki_name in &ki_info.type_param_names {
        struct_args
            .args
            .push(syn::parse_str::<GenericArgument>(ki_name).unwrap());
    }
    for ko_name in &ki_info.ko_type_param_names {
        struct_args
            .args
            .push(syn::parse_str::<GenericArgument>(ko_name).unwrap());
    }
    let device_op_arg: GenericArgument = parse_quote! { DI };
    struct_args.args.push(device_op_arg.clone());

    // launch_output_type is used for the unified launcher's return type.
    let mut launch_output_type = generic_args.clone();
    for (i, is_arc) in ki_param_idx.iter().enumerate() {
        if is_arc.is_some() {
            launch_output_type
                .args
                .push(syn::parse_str::<GenericArgument>(&format!("_S{}", i)).unwrap());
        }
    }
    for (i, is_part) in ko_param_idx.iter().enumerate() {
        if is_part.is_some() {
            launch_output_type
                .args
                .push(syn::parse_str::<GenericArgument>(&format!("_Q{}", i)).unwrap());
        }
    }
    // The argument op is spelled with nameable types (`KernelArgs` of
    // `PrepareOutput` / `PrepareInput` / bare `IntoDeviceOp::Op`s) rather than
    // an opaque `impl DeviceOp`, so `impl GraphNode for Launcher<.., DI> where
    // DI: GraphNode` is visible at the call site: a launcher over
    // pre-allocated inputs records into a CudaGraph scope, one over an
    // allocating op such as `api::zeros(..).partition(..)` does not compile
    // there. Built below alongside the direct launcher's where clauses.
    let mut launcher_arg_op_types: Vec<String> = vec![];

    // ── execute() method body ───────────────────────────────────────────────

    let init_stmts = syn::parse2::<ExprBlock>(quote! {{
        let module_name = #module_name;
        let function_name = #function_name;
        let function_entry = #function_entry_name;
        let input = self.input.take().unwrap();
    }})
    .unwrap()
    .block
    .stmts;
    launcher_method.block.stmts.extend(init_stmts);
    let specialization_init_stmts = syn::parse2::<ExprBlock>(quote! {{
        let module_name = #module_name;
        let function_name = #function_name;
        let input = self.input.take().unwrap();
    }})
    .unwrap()
    .block
    .stmts;
    specialization_method
        .block
        .stmts
        .extend(specialization_init_stmts);
    let execute_input_stmt = parse_stmt(format!(
        r#"let {param_names_tuple_str}: {stored_args_type_str} = input.execute(ctx)?;"#
    ));
    launcher_method.block.stmts.push(execute_input_stmt.clone());
    specialization_method.block.stmts.push(execute_input_stmt);

    if !required_generics.names.is_empty() {
        let generics_stmt = parse_stmt(format!(
            r#"
            let function_generics: Vec<String> = if self.function_generics.is_some() {{
                self.function_generics.take().unwrap()
            }} else {{
                {}
            }};
            "#,
            required_generics.to_expr_str()
        ));
        launcher_method.block.stmts.push(generics_stmt.clone());
        specialization_method.block.stmts.push(generics_stmt);
    } else {
        let generics_stmt = parse_stmt("let function_generics: Vec<String> = vec![];".to_string());
        launcher_method.block.stmts.push(generics_stmt.clone());
        specialization_method.block.stmts.push(generics_stmt);
    }

    let stride_args_stmt = parse_stmt(format!(
        "let stride_args: Vec<(String, Vec<i32>)> =  vec![{}];",
        stride_args.join(",")
    ));
    specialization_method
        .block
        .stmts
        .push(stride_args_stmt.clone());
    let spec_args_stmt = parse_stmt(format!(
        "let spec_args: Vec<(String, {tile_rust_crate_root}::cutile_compiler::specialization::SpecializationBits)> = vec![{}];",
        spec_args.join(",")
    ));
    specialization_method
        .block
        .stmts
        .push(spec_args_stmt.clone());

    // Emit scalar_hints (populated for integer scalar and raw pointer params).
    let scalar_hints_stmt = parse_stmt(format!(
        "let scalar_hints: Vec<(String, {tile_rust_crate_root}::cutile_compiler::specialization::DivHint)> = vec![{}];",
        scalar_hint_exprs.join(",")
    ));
    specialization_method
        .block
        .stmts
        .push(scalar_hints_stmt.clone());

    // Launch-site probe: on the steady state the launcher skips key
    // construction, toolchain fingerprinting, hashing, and the global cache
    // entirely — the volatile specialization inputs are compared against the
    // site's last resolution by value, borrowed from the arguments.
    // Fixed-size arrays: the probe borrows them, so a steady-state launch
    // allocates nothing here.
    launcher_method.block.stmts.push(parse_stmt(format!(
        "let __specs: [&{tile_rust_crate_root}::cutile_compiler::specialization::SpecializationBits; {}] = [{}];",
        spec_ref_exprs.len(),
        spec_ref_exprs.join(",")
    )));
    launcher_method.block.stmts.push(parse_stmt(format!(
        "let __hint_vals: [{tile_rust_crate_root}::cutile_compiler::specialization::DivHint; {}] = [{}];",
        scalar_hint_value_exprs.len(),
        scalar_hint_value_exprs.join(",")
    )));
    launcher_method.block.stmts.push(parse_stmt(format!(
        "static __SITE: {tile_rust_crate_root}::tile_kernel::LaunchSite = {tile_rust_crate_root}::tile_kernel::LaunchSite::new();"
    )));
    launcher_method.block.stmts.push(parse_stmt(
        "let __const_grid = if self._const_grid { Some(self._grid) } else { None };".to_string(),
    ));
    launcher_method.block.stmts.push(parse_stmt(
        "let __compile_options = std::mem::take(&mut self._compile_options);".to_string(),
    ));
    launcher_method.block.stmts.push(parse_stmt(
        "let __site_hit = __SITE.get(ctx.get_device_id(), &function_generics, &__specs,          &__hint_vals, __const_grid, &__compile_options);"
            .to_string(),
    ));
    launcher_method.block.stmts.push(parse_stmt(format!(
        "let (function, validator) = if let Some(__hit) = __site_hit {{
            __hit
        }} else {{
            let stride_args: Vec<(String, Vec<i32>)> = vec![{strides}];
            let spec_args: Vec<(String, {root}::cutile_compiler::specialization::SpecializationBits)> = vec![{specs}];
            let scalar_hints: Vec<(String, {root}::cutile_compiler::specialization::DivHint)> = vec![{hints}];
            let __generics_snapshot = function_generics.clone();
            let __specs_snapshot: Vec<{root}::cutile_compiler::specialization::SpecializationBits> =
                spec_args.iter().map(|(_, s)| s.clone()).collect();
            let __hints_snapshot = __hint_vals.to_vec();
            let __options_snapshot = __compile_options.clone();
            let (function, validator) = self.jit_compile(
                ctx, __module_ast_self,
                module_name, function_name, function_entry,
                function_generics, stride_args, spec_args, scalar_hints,
                __const_grid,
                __compile_options,
                _SOURCE_HASH,
            )?;
            __SITE.store({root}::tile_kernel::SiteResolution::new(
                ctx.get_device_id(),
                __generics_snapshot,
                __specs_snapshot,
                __hints_snapshot,
                __const_grid,
                __options_snapshot,
                function.clone(),
                validator.clone(),
            ));
            (function, validator)
        }};",
        strides = stride_args.join(","),
        specs = spec_args.join(","),
        hints = scalar_hint_exprs.join(","),
        root = tile_rust_crate_root,
    )));

    let specialization_stmts = syn::parse2::<ExprBlock>(quote! {{
        let const_grid = if self._const_grid { Some(self._grid) } else { None };
        let compile_options = std::mem::take(&mut self._compile_options);
        return Ok(_specialization_from_context(
            ctx,
            __module_ast_self as ModuleAstFn,
            module_name,
            function_name,
            function_generics,
            stride_args,
            spec_args,
            scalar_hints,
            const_grid,
            compile_options,
            _SOURCE_HASH,
        ));
    }})
    .unwrap()
    .block
    .stmts;
    specialization_method
        .block
        .stmts
        .extend(specialization_stmts);

    // (The jit_compile call lives inside the launch-site miss branch above;
    // the LINKING Phase B AST-builder note applies there.)
    // Per-parameter runtime extents for launch-time check hoisting; the
    // per-arg validator statements fill in each tensor param's shape by
    // signature index (non-tensor params stay empty).
    launcher_method.block.stmts.push(parse_stmt(
        "let mut param_shapes: Vec<Vec<i32>> = if validator.launch_checks.is_empty() { Vec::new() } else { vec![Vec::new(); validator.params.len()] };"
            .to_string(),
    ));
    // The kernel-visible view per parameter: the partition slab for a
    // `&mut Tensor` output, the whole tensor otherwise. Resolves the
    // view-frame atoms in hoisted checks, while `param_shapes` (the root
    // frame) resolves `dim(...)` atoms and declared preconditions.
    launcher_method.block.stmts.push(parse_stmt(
        "let mut view_shapes: Vec<Vec<i32>> = if validator.launch_checks.is_empty() { Vec::new() } else { vec![Vec::new(); validator.params.len()] };"
            .to_string(),
    ));
    launcher_method.block.stmts.extend(validator_statements);

    // Above the `!_compile_only` gate so warmup validates the grid too.
    launcher_method.block.stmts.push(parse_stmt(
        "if self._programmatic_dependent_launch { validate_programmatic_dependent_launch(ctx.get_device_id())?; }".to_string(),
    ));
    launcher_method.block.stmts.push(parse_stmt(format!(
        "let launch_grid: (u32, u32, u32) = self.infer_launch_grid(&[{}])?;",
        launch_grid_expr_strs.join(",")
    )));
    // Run the compiler-emitted launch checks (safety checks hoisted out of the
    // kernel) against the runtime extents, before the kernel is launched.
    launcher_method.block.stmts.push(parse_stmt(
        "validate_launch_checks(&validator.launch_checks, &param_shapes, &view_shapes, launch_grid)?;"
            .to_string(),
    ));

    let mut launch_only_stmts: Vec<Stmt> = vec![parse_stmt(
        "let mut kernel_launch = AsyncKernelLaunch::new(function.clone());".to_string(),
    )];
    launch_only_stmts.extend(builder_statements);
    launch_only_stmts.extend(
        syn::parse2::<ExprBlock>(quote! {{
            kernel_launch
                .set_launch_config(LaunchConfig {
                    grid_dim: launch_grid,
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0
                });
            if self._programmatic_dependent_launch {
                // SAFETY: the generated builder's unsafe opt-in transferred
                // the dependency/lifetime obligations to its caller.
                unsafe { kernel_launch.programmatic_dependent_launch(); }
            }
            kernel_launch.execute(ctx)?;
        }})
        .unwrap()
        .block
        .stmts,
    );
    let gated_launch = syn::parse2::<Stmt>(quote! {
        if !self._compile_only {
            #(#launch_only_stmts)*
        }
    })
    .unwrap();
    launcher_method.block.stmts.push(gated_launch);
    // Return with KernelInput::recover applied to each &Tensor param.
    launcher_method
        .block
        .stmts
        .push(parse_stmt(format!(r#"return Ok({recovered_tuple_str});"#)));

    // ── _apply launcher (used internally by api.rs) ───────────────────────
    // Takes a concrete tuple with Arc<Tensor<T>> for &Tensor params.
    // Specialized: _K = Arc<Tensor<T>> for all KernelInput params.

    let kernel_naming = KernelNaming::new(function_name);
    let concrete_args_type: Type = parse_quote! { ( #(#arg_types,)* ) };

    // Build launch_output_type specialized with Arc for _apply.
    let mut apply_launch_output_type = generic_args.clone();
    for ki_name in &ki_info.type_param_names {
        // Find the corresponding Arc<Tensor<T>> type for this _K param.
        let ki_idx_in_types = ki_info
            .param_kernel_input_idx
            .iter()
            .enumerate()
            .find(|(_, idx)| {
                idx.map(|k| ki_info.type_param_names[k] == *ki_name)
                    .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .unwrap();
        let arc_type = &arg_types[ki_idx_in_types];
        apply_launch_output_type.args.push(
            syn::parse_str::<GenericArgument>(&arc_type.to_token_stream().to_string()).unwrap(),
        );
    }
    // Also specialize _P = Partition<Tensor<T>> for _apply.
    for ko_name in &ki_info.ko_type_param_names {
        let ko_idx_in_types = ki_info
            .param_kernel_output_idx
            .iter()
            .enumerate()
            .find(|(_, idx)| {
                idx.map(|k| ki_info.ko_type_param_names[k] == *ko_name)
                    .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .unwrap();
        let part_type = &arg_types[ko_idx_in_types];
        apply_launch_output_type.args.push(
            syn::parse_str::<GenericArgument>(&part_type.to_token_stream().to_string()).unwrap(),
        );
    }
    // `_apply` wraps an already-materialized tuple, so its argument op is a
    // plain `Value` — nameable and always graph-recordable.
    let apply_value_op: GenericArgument = parse_quote! { Value<#concrete_args_type> };
    apply_launch_output_type.args.push(apply_value_op);

    let apply_return_type = quote! { #launcher_ident #apply_launch_output_type };
    let apply_name = kernel_naming.apply_name();
    let launcher_apply_ident = Ident::new(apply_name.as_str(), Span::call_site());
    let launcher_apply = syn::parse2::<ItemFn>(quote! {
        pub #unsafety fn #launcher_apply_ident #generic_params (input: #concrete_args_type) -> #apply_return_type {
            return #launcher_ident::launch(value(input));
        }
    })
    .unwrap();

    // ── Unified launcher (the primary public entry point) ───────────────────

    let arg_aliases = arg_types
        .iter()
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>();

    let launcher_direct_ident = Ident::new(kernel_naming.public_name(), Span::call_site());
    // The return type is filled in once every argument's op type is known.
    let mut launcher_direct = syn::parse2::<ItemFn>(quote! {
        pub #unsafety fn #launcher_direct_ident #generic_params() -> () {}
    })
    .unwrap();
    launcher_direct.sig.generics.make_where_clause();
    let mut function_params = vec![];
    let mut is_arc_param = vec![];
    for (i, _arg_ty) in arg_types.iter().enumerate() {
        let function_param = format!("arg{}", i);
        let type_param_name = format!("_A{}", i);
        let arg_type_str = &arg_aliases[i];
        let is_arc = arg_type_str.starts_with("Arc <");
        is_arc_param.push(is_arc);

        launcher_direct.sig.inputs.push(FnArg::Typed(
            syn::parse2::<PatType>(
                format!("{}: {}", function_param, type_param_name)
                    .parse()
                    .unwrap(),
            )
            .unwrap(),
        ));

        let where_clause = launcher_direct
            .sig
            .generics
            .where_clause
            .as_mut()
            .expect("Impossible.");

        if is_arc {
            // KernelInput bound: accepts Tensor<T>, Arc<Tensor<T>>, or &Tensor<T>.
            let intermediate_type = format!("_S{}", i);
            let elem = &ki_info.element_type_names[ki_param_idx[i].unwrap()];
            launcher_arg_op_types.push(format!(
                "PrepareInput<<{type_param_name} as IntoDeviceOp<{intermediate_type}>>::Op, {elem}>"
            ));
            launcher_direct.sig.generics.params.push(GenericParam::Type(
                syn::parse2::<TypeParam>(type_param_name.parse().unwrap()).unwrap(),
            ));
            launcher_direct.sig.generics.params.push(GenericParam::Type(
                syn::parse2::<TypeParam>(intermediate_type.parse().unwrap()).unwrap(),
            ));
            where_clause.predicates.push(
                syn::parse2::<WherePredicate>(
                    format!("{}: IntoDeviceOp<{}>", type_param_name, intermediate_type)
                        .parse()
                        .unwrap(),
                )
                .unwrap(),
            );
            let ki_idx = ki_param_idx[i].unwrap();
            let elem = &ki_info.element_type_names[ki_idx];
            where_clause.predicates.push(
                syn::parse2::<WherePredicate>(
                    format!("{}: KernelInput<{}>", intermediate_type, elem)
                        .parse()
                        .unwrap(),
                )
                .unwrap(),
            );
        } else if ko_param_idx[i].is_some() {
            // KernelOutput bound: accepts Partition<Tensor<T>> or Partition<&mut Tensor<T>>.
            let intermediate_type = format!("_Q{}", i);
            let elem = &ki_info.ko_element_type_names[ko_param_idx[i].unwrap()];
            launcher_arg_op_types.push(format!(
                "PrepareOutput<<{type_param_name} as IntoDeviceOp<{intermediate_type}>>::Op, {elem}>"
            ));
            launcher_direct.sig.generics.params.push(GenericParam::Type(
                syn::parse2::<TypeParam>(type_param_name.parse().unwrap()).unwrap(),
            ));
            launcher_direct.sig.generics.params.push(GenericParam::Type(
                syn::parse2::<TypeParam>(intermediate_type.parse().unwrap()).unwrap(),
            ));
            where_clause.predicates.push(
                syn::parse2::<WherePredicate>(
                    format!("{}: IntoDeviceOp<{}>", type_param_name, intermediate_type)
                        .parse()
                        .unwrap(),
                )
                .unwrap(),
            );
            let ko_idx = ko_param_idx[i].unwrap();
            let elem = &ki_info.ko_element_type_names[ko_idx];
            where_clause.predicates.push(
                syn::parse2::<WherePredicate>(
                    format!("{}: KernelOutput<{}>", intermediate_type, elem)
                        .parse()
                        .unwrap(),
                )
                .unwrap(),
            );
        } else {
            // Scalars: direct IntoDeviceOp bound.
            launcher_arg_op_types.push(format!(
                "<{type_param_name} as IntoDeviceOp<{arg_type_str}>>::Op"
            ));
            launcher_direct.sig.generics.params.push(GenericParam::Type(
                syn::parse2::<TypeParam>(type_param_name.parse().unwrap()).unwrap(),
            ));
            where_clause.predicates.push(
                syn::parse2::<WherePredicate>(
                    format!("{}: IntoDeviceOp<{}>", type_param_name, arg_type_str)
                        .parse()
                        .unwrap(),
                )
                .unwrap(),
            );
        }
        function_params.push(function_param);
    }
    // The argument op: `KernelArgs<(op0, op1, ..)>`, nameable so the launcher
    // type in the return position carries it (see `launcher_arg_op_types`).
    let kernel_args_type: GenericArgument = syn::parse_str::<GenericArgument>(&format!(
        "KernelArgs<{}>",
        to_tuple_string(&launcher_arg_op_types)
    ))
    .unwrap();
    launch_output_type.args.push(kernel_args_type);
    launcher_direct.sig.output = syn::parse2::<syn::ReturnType>(quote! {
        -> #launcher_ident #launch_output_type
    })
    .unwrap();
    // Convert each arg into a DeviceOp, applying KernelInput::prepare for
    // &Tensor params and KernelOutput::prepare for &mut Tensor params.
    let mut di_var_names: Vec<String> = vec![];
    for (i, var) in function_params.iter().enumerate() {
        let di_var = format!("_di{}", i);
        if is_arc_param[i] {
            launcher_direct.block.stmts.push(parse_stmt(format!(
                "let {di_var} = PrepareInput::new({var}.into_op());"
            )));
        } else if ko_param_idx[i].is_some() {
            launcher_direct.block.stmts.push(parse_stmt(format!(
                "let {di_var} = PrepareOutput::new({var}.into_op());"
            )));
        } else {
            launcher_direct
                .block
                .stmts
                .push(parse_stmt(format!("let {di_var} = {var}.into_op();")));
        }
        di_var_names.push(di_var);
    }
    launcher_direct.block.stmts.push(parse_stmt(format!(
        "let input = KernelArgs({});",
        to_tuple_string(&di_var_names)
    )));
    launcher_direct.block.stmts.push(parse_stmt(format!(
        "return {}::launch(input);",
        launcher_ident
    )));

    let returned_args_type_2 = returned_args_type.clone();
    Ok((
        required_generics,
        (stored_args_type, returned_args_type),
        quote! {
            impl #struct_generics #launcher_ident #struct_args {
                #specialization_method
            }
            impl #struct_generics DeviceOp for #launcher_ident #struct_args {
                type Output = #returned_args_type_2;
                #launcher_method
            }
            // A launch allocates nothing itself, so it is graph-recordable
            // exactly when its argument op is: `Value`s over pre-allocated
            // tensors qualify, an allocating input op does not (its allocation
            // node would return a different address on replay).
            impl #struct_generics GraphNode for #launcher_ident #struct_args where DI: GraphNode {}
            #launcher_apply
            #launcher_direct
        },
        kernel_input_info,
    ))
}

/// Parses a string into a Rust statement AST node.
///
/// Helper function that converts a string representation of a statement
/// into a `syn::Stmt` for insertion into generated code blocks.
///
/// ## Parameters
///
/// - `s`: String containing valid Rust statement code
///
/// ## Returns
///
/// A parsed `Stmt` AST node
fn parse_stmt(s: String) -> Stmt {
    syn::parse::<Stmt>(s.parse().unwrap()).unwrap()
}

/// Parses a string into a Rust expression AST node.
///
/// Helper function that converts a string representation of an expression
/// into a `syn::Expr` for use in generated code.
///
/// ## Parameters
///
/// - `s`: String containing valid Rust expression code
///
/// ## Returns
///
/// A parsed `Expr` AST node
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`).
#[allow(dead_code)]
fn parse_expr(s: String) -> Expr {
    syn::parse::<Expr>(s.parse().unwrap()).unwrap()
}

/// Metadata about KernelInput params for the struct definition in _module.rs.
pub struct KernelInputInfo {
    /// Names of the KernelInput type params, e.g., ["_K0", "_K1"].
    pub type_param_names: Vec<String>,
    /// Element type name for each KernelInput param, e.g., ["T", "SrcType"].
    pub element_type_names: Vec<String>,
    /// For each param in the args tuple, the index into type_param_names (if it's
    /// a KernelInput param), or None (if it's a partition/scalar).
    pub param_kernel_input_idx: Vec<Option<usize>>,
    /// Names of the KernelOutput type params, e.g., ["_P0", "_P1"].
    pub ko_type_param_names: Vec<String>,
    /// Element type name for each KernelOutput param.
    pub ko_element_type_names: Vec<String>,
    /// For each param, the index into ko_type_param_names (if partition).
    pub param_kernel_output_idx: Vec<Option<usize>>,
    /// The "recovered" return expression with both KernelInput and KernelOutput recover calls.
    pub recovered_tuple_str: String,
}

/// Code generation result for a tensor kernel parameter.
///
/// Contains all the generated code components needed for a single tensor
/// parameter in the generated launcher.
///
/// ## Fields
///
/// - `fn_arg`: The launcher function parameter (e.g., `x: Arc<Tensor<T>>`)
/// - `stride_expr_str`: Expression string for computing tensor strides
/// - `builder_statements`: Statements to add to the kernel builder
/// - `launch_grid_expr_strs`: Expressions for computing launch grid dimensions
struct TensorLaunchCode {
    fn_arg: PatType, // FnArg::Typed(PatType)
    stride_expr_str: String,
    spec_expr_str: String,
    /// Borrowed form of the same spec, for the launch-site probe: no clone.
    spec_ref_expr_str: String,
    builder_statements: Vec<Stmt>,
    launch_grid_expr_strs: Vec<String>,
    validator_statements: ExprBlock,
    /// Element type name (e.g., "T", "SrcType", "f32") for &Tensor params.
    element_type_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorParamKind {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DimEqualityPrecondition {
    lhs: String,
    lhs_axis: usize,
    rhs: String,
    rhs_axis: usize,
}

enum Precondition {
    DimEq(DimEqualityPrecondition),
    /// `dim(tensor, axis) % divisor == 0`, divisor a positive integer literal.
    DimDivisible {
        tensor: String,
        axis: usize,
        divisor: i64,
    },
}

fn parse_metadata_preconditions(item: &ItemFn) -> Result<Vec<Precondition>, Error> {
    let Some(entry_attrs) = get_meta_list_by_last_segment("entry", &item.attrs) else {
        return Ok(Vec::new());
    };
    let Some(expr) = entry_attrs.parse_custom_expr("preconditions") else {
        return Ok(Vec::new());
    };
    let calls = match expr {
        Expr::Tuple(tuple) => tuple.elems.iter().collect::<Vec<_>>(),
        Expr::Paren(paren) => vec![paren.expr.as_ref()],
        Expr::Binary(_) => vec![expr],
        Expr::Call(call)
            if precondition_call_name(call).as_deref() == Some("same_partition_axis") =>
        {
            return call.err(
                "`same_partition_axis` preconditions have been replaced by `dim(lhs, axis) == dim(rhs, axis)`",
            );
        }
        _ => {
            return item
                .sig
                .ident
                .err("`preconditions` must be an equality or tuple of equalities")
        }
    };
    let mut preconditions = Vec::new();
    for expr in calls {
        let Expr::Binary(binary) = expr else {
            return item
                .sig
                .ident
                .err("each `preconditions` entry must be a metadata equality");
        };
        preconditions.push(parse_precondition(binary)?);
    }
    Ok(preconditions)
}

fn parse_precondition(binary: &syn::ExprBinary) -> Result<Precondition, Error> {
    if !matches!(binary.op, syn::BinOp::Eq(_)) {
        return binary.err("precondition predicates must use `==`");
    }
    // Divisibility spelling: `dim(t, k) % divisor == 0`.
    if let Expr::Binary(rem) = binary.left.as_ref() {
        if matches!(rem.op, syn::BinOp::Rem(_)) {
            let is_zero = match binary.right.as_ref() {
                Expr::Lit(lit) => match &lit.lit {
                    Lit::Int(i) => i.base10_parse::<i64>().map(|v| v == 0).unwrap_or(false),
                    _ => false,
                },
                _ => false,
            };
            if !is_zero {
                return binary.err(
                    "a `%` precondition must compare against literal `0`: `dim(t, k) % d == 0`",
                );
            }
            let (tensor, axis) = parse_dim_precondition_expr(&rem.left)?;
            let Expr::Lit(lit) = rem.right.as_ref() else {
                return rem.right.err(
                    "a `%` precondition divisor must be a positive integer literal (a const \
                     generic cannot be verified by the launcher, which runs before \
                     monomorphization)",
                );
            };
            let Lit::Int(int_lit) = &lit.lit else {
                return rem
                    .right
                    .err("a `%` precondition divisor must be a positive integer literal");
            };
            let divisor = int_lit
                .base10_parse::<i64>()
                .map_err(|err| crate::error::syn_err(int_lit.span(), &err.to_string()))?;
            if divisor < 1 {
                return rem.right.err(&format!(
                    "a `%` precondition divisor must be >= 1, got {divisor}"
                ));
            }
            return Ok(Precondition::DimDivisible {
                tensor,
                axis,
                divisor,
            });
        }
    }
    parse_dim_equality_precondition(binary).map(Precondition::DimEq)
}

fn parse_dim_equality_precondition(
    binary: &syn::ExprBinary,
) -> Result<DimEqualityPrecondition, Error> {
    if !matches!(binary.op, syn::BinOp::Eq(_)) {
        return binary.err("precondition predicates must use `==`");
    }
    let (lhs, lhs_axis) = parse_dim_precondition_expr(&binary.left)?;
    let (rhs, rhs_axis) = parse_dim_precondition_expr(&binary.right)?;
    Ok(DimEqualityPrecondition {
        lhs,
        lhs_axis,
        rhs,
        rhs_axis,
    })
}

fn parse_dim_precondition_expr(expr: &Expr) -> Result<(String, usize), Error> {
    let Expr::Call(call) = expr else {
        return expr
            .err("precondition metadata expressions must be calls like `dim(tensor, axis)`");
    };
    let Some(func_name) = precondition_call_name(call) else {
        return call
            .func
            .err("precondition metadata expressions must use a function path");
    };
    if func_name != "dim" {
        return call.func.err(&format!(
            "unsupported precondition metadata expression `{func_name}`; expected `dim`"
        ));
    }
    if call.args.len() != 2 {
        return call.err(&format!(
            "`dim` expects 2 arguments, got {}",
            call.args.len()
        ));
    }
    Ok((
        parse_precondition_tensor_arg(&call.args[0])?,
        parse_precondition_axis_arg(&call.args[1])?,
    ))
}

fn precondition_call_name(call: &ExprCall) -> Option<String> {
    let Expr::Path(func_path) = call.func.as_ref() else {
        return None;
    };
    func_path
        .path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn parse_precondition_tensor_arg(expr: &Expr) -> Result<String, Error> {
    let Expr::Path(path) = expr else {
        return expr.err("precondition tensor arguments must be parameter names");
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return expr.err("precondition tensor arguments must be simple parameter names");
    }
    Ok(path.path.segments[0].ident.to_string())
}

fn parse_precondition_axis_arg(expr: &Expr) -> Result<usize, Error> {
    let Expr::Lit(lit_expr) = expr else {
        return expr.err("precondition axes must be integer literals");
    };
    let Lit::Int(lit_int) = &lit_expr.lit else {
        return expr.err("precondition axes must be integer literals");
    };
    lit_int
        .base10_parse::<usize>()
        .map_err(|err| crate::error::syn_err(lit_int.span(), &err.to_string()))
}

fn precondition_shape_expr(name: &str, kind: TensorParamKind) -> TokenStream2 {
    let ident = Ident::new(name, Span::call_site());
    match kind {
        TensorParamKind::Input => quote! { #ident.shape().to_vec() },
        TensorParamKind::Output => quote! { KernelOutputStored::shape_as_i32(&#ident) },
    }
}

fn metadata_precondition_validation_stmts(
    preconditions: &[Precondition],
    tensor_param_kinds: &HashMap<String, TensorParamKind>,
) -> Result<Vec<Stmt>, Error> {
    let mut statements = Vec::new();
    for precondition in preconditions {
        let precondition = match precondition {
            Precondition::DimEq(eq) => eq,
            Precondition::DimDivisible {
                tensor,
                axis,
                divisor,
            } => {
                let Some(&kind) = tensor_param_kinds.get(tensor) else {
                    return Err(crate::error::syn_err(
                        Span::call_site(),
                        &format!("precondition references unknown tensor parameter `{tensor}`"),
                    ));
                };
                let shape_expr = precondition_shape_expr(tensor, kind);
                let axis = *axis;
                let divisor = *divisor;
                let validation = syn::parse2::<ExprBlock>(quote! {{
                    {
                        let shape: Vec<i32> = #shape_expr;
                        kernel_launch_assert_with(
                            #axis < shape.len(), || format!(
                                "dim({}, {}) % {} == 0 failed: axis is out of range for shape {:?}",
                                #tensor, #axis, #divisor, shape
                            ))?;
                        kernel_launch_assert_with(
                            (shape[#axis] as i64).rem_euclid(#divisor) == 0, || format!(
                                "dim({}, {}) % {} == 0 failed: extent is {}",
                                #tensor, #axis, #divisor, shape[#axis]
                            ))?;
                    }
                }})
                .unwrap();
                statements.extend(validation.block.stmts);
                continue;
            }
        };
        let Some(&lhs_kind) = tensor_param_kinds.get(&precondition.lhs) else {
            return Err(crate::error::syn_err(
                Span::call_site(),
                &format!(
                    "precondition references unknown tensor parameter `{}`",
                    precondition.lhs
                ),
            ));
        };
        let Some(&rhs_kind) = tensor_param_kinds.get(&precondition.rhs) else {
            return Err(crate::error::syn_err(
                Span::call_site(),
                &format!(
                    "precondition references unknown tensor parameter `{}`",
                    precondition.rhs
                ),
            ));
        };
        let lhs_name = &precondition.lhs;
        let rhs_name = &precondition.rhs;
        let lhs_axis = precondition.lhs_axis;
        let rhs_axis = precondition.rhs_axis;
        let lhs_shape_expr = precondition_shape_expr(lhs_name, lhs_kind);
        let rhs_shape_expr = precondition_shape_expr(rhs_name, rhs_kind);
        let validation = syn::parse2::<ExprBlock>(quote! {{
            {
                let lhs_shape: Vec<i32> = #lhs_shape_expr;
                let rhs_shape: Vec<i32> = #rhs_shape_expr;
                kernel_launch_assert_with(
                    #lhs_axis < lhs_shape.len(), || format!(
                        "dim({}, {}) == dim({}, {}) failed: left axis is out of range for shape {:?}",
                        #lhs_name,
                        #lhs_axis,
                        #rhs_name,
                        #rhs_axis,
                        lhs_shape
                    ))?;
                kernel_launch_assert_with(
                    #rhs_axis < rhs_shape.len(), || format!(
                        "dim({}, {}) == dim({}, {}) failed: right axis is out of range for shape {:?}",
                        #lhs_name,
                        #lhs_axis,
                        #rhs_name,
                        #rhs_axis,
                        rhs_shape
                    ))?;
                kernel_launch_assert_with(
                    lhs_shape[#lhs_axis] == rhs_shape[#rhs_axis], || format!(
                        "dim({}, {}) == dim({}, {}) failed: axis extents differ ({} vs {})",
                        #lhs_name,
                        #lhs_axis,
                        #rhs_name,
                        #rhs_axis,
                        lhs_shape[#lhs_axis],
                        rhs_shape[#rhs_axis],
                    ))?;
            }
        }})
        .unwrap();
        statements.extend(validation.block.stmts);
    }
    Ok(statements)
}

/// Generates launcher code for a tensor kernel parameter.
///
/// Analyzes a tensor parameter and generates all the necessary code for:
/// 1. The launcher function parameter type
/// 2. Stride calculations
/// 3. Kernel builder statements (pushing arguments, metadata)
/// 4. Launch grid dimension calculations
///
/// ## Parameters
///
/// - `var_name`: Name of the parameter variable
/// - `ty`: The tensor reference type (e.g., `&mut Tensor<T, [128]>`)
/// - `required_generics`: Accumulator for tracking required generic constraints
///
/// ## Returns
///
/// A `TensorLaunchCode` struct containing all generated code components
fn get_tensor_code(
    var_idx: usize,
    var_name: &str,
    ty: &TypeReference,
    required_generics: &mut RequiredGenerics,
) -> Result<TensorLaunchCode, Error> {
    // FnArg
    let (type_ident, type_generic_args) = get_ident_generic_args(&Type::Reference(ty.clone()));
    let Some(type_ident) = type_ident else {
        return ty.err("Expected a named type identifier for tensor parameter.");
    };
    let type_ident = type_ident.to_string();
    if type_ident != "Tensor" && type_ident != "MappedPartitionMut" {
        return ty.err(&format!("Expected Tensor type, got {}.", type_ident));
    }
    if type_ident == "MappedPartitionMut" && ty.mutability.is_none() {
        return ty.err("internal error: MappedPartitionMut launcher lowering must use mutable output metadata.");
    }
    let is_mapped_partition = type_ident == "MappedPartitionMut";
    if is_mapped_partition {
        let Some(map_shape) = type_generic_args.args.iter().nth(2) else {
            return ty.err("MappedPartitionMut expects map shape as its third generic argument.");
        };
        infer_mapped_partition_map_generics(var_name, map_shape, required_generics)?;
    }
    let Some(GenericArgument::Type(syn::Type::Path(element_type_path))) =
        type_generic_args.args.first()
    else {
        return ty.err("Expected generic argument type path for tensor element type.");
    };

    // Infer generics from type data available in this type.
    let mut shape_generic_args = type_generic_args.clone();
    if is_mapped_partition {
        while shape_generic_args.args.len() > 2 {
            shape_generic_args.args.pop();
        }
    }
    infer_shape_params_from_tensor_type(
        var_name,
        &shape_generic_args,
        required_generics,
        ty.mutability.is_some(),
    )?;

    let dtype = element_type_path
        .path
        .segments
        .last()
        .unwrap()
        .ident
        .to_string();
    let tensor_type = if ty.mutability.is_some() {
        format!("tensor::Partition<tensor::Tensor<{dtype}>>")
    } else {
        format!("Arc<tensor::Tensor<{dtype}>>")
    };
    let fn_arg =
        syn::parse::<PatType>(format!("{var_name}: {tensor_type}").parse().unwrap()).unwrap();
    // Stride expr.
    let stride_expr_str = if ty.mutability.is_some() {
        format!(
            r#"(
            "{var_name}".to_string(),
            KernelOutputStored::strides_hint(&{var_name})
        )"#
        )
    } else {
        format!(
            r#"(
        "{var_name}".to_string(),
        {var_name}.spec().stride_one.iter()
            .map(|&is_one| if is_one {{ 1 }} else {{ -1 }})
            .collect::<Vec<i32>>()
        )"#
        )
    };
    // Spec expr.
    let spec_expr_str = if ty.mutability.is_some() {
        format!(
            r#"(
            "{var_name}".to_string(),
            KernelOutputStored::spec(&{var_name}).clone()
        )"#
        )
    } else {
        format!(
            r#"(
        "{var_name}".to_string(),
        {var_name}.spec().clone()
        )"#
        )
    };
    let spec_ref_expr_str = if ty.mutability.is_some() {
        format!("KernelOutputStored::spec(&{var_name})")
    } else {
        format!("{var_name}.spec()")
    };
    // Builder and validator statements.
    let var_ident = Ident::new(var_name, Span::call_site());
    let mut builder_statements = vec![];
    let mut launch_grid_expr_strs = vec![];
    let validator_statements = if ty.mutability.is_some() {
        builder_statements.push(parse_stmt(format!(
            "KernelOutputStored::retain(&{var_name}, ctx)?;"
        )));
        builder_statements.push(parse_stmt(format!(
            "KernelOutputStored::push_kernel_args(&{var_name}, &mut kernel_launch);"
        )));
        launch_grid_expr_strs.push(format!("KernelOutputStored::grid_bound(&{var_name})?"));
        syn::parse2::<ExprBlock>(quote! {{
            {
                let ValidParamType::Tensor(tensor_validator) = &validator.params[#var_idx] else {
                    panic!("Unexpected validator type {:#?}", &validator.params[#var_idx]);
                };
                // The compiled specialization's element type must be the tensor's
                // dtype. A user `.generics(..)` list overrides the inferred type, and
                // a wider-typed kernel reads and writes past a narrower buffer.
                let given_dtype = KernelOutputStored::dtype_str(&#var_ident);
                kernel_launch_assert_with(tensor_validator.element_type == given_dtype, || format!("{} element type mismatch: the kernel was specialized for `{}` but the tensor holds `{}` (check the `.generics(..)` list)", #var_name, tensor_validator.element_type, given_dtype))?;
                let valid_shape = &tensor_validator.shape;
                let given_shape: &[usize] = KernelOutputStored::partition_shape(&#var_ident);
                kernel_launch_assert_with(valid_shape.len() == given_shape.len(), || format!("{} rank mismatch: Expected {}, got {}", #var_name, valid_shape.len(), given_shape.len()))?;
                kernel_launch_assert_with(zip(valid_shape, given_shape).all(|(&expected, &given)| expected as i64 == given as i64), || format!("{} partition shape mismatch. Expected {:?}, got {:?}", #var_name, valid_shape, given_shape))?;
                // Runtime extents for launch-time check hoisting (indexed by
                // signature position): root shape resolves `Dim` atoms, and the
                // partition slab -- this param's kernel-visible view -- resolves
                // `ViewExtent` atoms.
                if !validator.launch_checks.is_empty() {
                    param_shapes[#var_idx] = KernelOutputStored::shape_as_i32(&#var_ident);
                    view_shapes[#var_idx] = given_shape.iter().map(|&x| x as i32).collect();
                }
            }
        }})
        .unwrap()
    } else {
        builder_statements.push(parse_stmt(format!(
            "KernelInputStored::retain(&{var_name}, ctx)?;"
        )));
        builder_statements.push(parse_stmt(format!(
            "KernelInputStored::push_kernel_args(&{var_name}, &mut kernel_launch);"
        )));

        syn::parse2::<ExprBlock>(quote! {{
            {
                let ValidParamType::Tensor(tensor_validator) = &validator.params[#var_idx] else {
                    panic!("Unexpected validator type {:#?}", &validator.params[#var_idx]);
                };
                // See the output branch: the specialization's element type must be
                // the tensor's dtype, or a wider-typed kernel reads past the buffer.
                let given_dtype = KernelInputStored::dtype_str(&#var_ident);
                kernel_launch_assert_with(tensor_validator.element_type == given_dtype, || format!("{} element type mismatch: the kernel was specialized for `{}` but the tensor holds `{}` (check the `.generics(..)` list)", #var_name, tensor_validator.element_type, given_dtype))?;
                let valid_shape = &tensor_validator.shape;
                let given_shape = #var_ident.shape();
                kernel_launch_assert_with(valid_shape.len() == given_shape.len(), || format!("{} rank mismatch: Expected {}, got {}", #var_name, valid_shape.len(), given_shape.len()))?;
                let pred = zip(valid_shape, given_shape).all(|(&expected, &given)| expected == -1 || expected == given);
                kernel_launch_assert_with(pred, || {
                    // Only a failing launch pays for the message.
                    let valid_shape_mixed = zip(valid_shape, given_shape).map(|(&expected, &given)|{
                        if expected == -1 { given } else { expected }
                    }).collect::<Vec<_>>();
                    format!("{} partition shape mismatch. Expected {:?}, got {:?}", #var_name, valid_shape_mixed, given_shape)
                })?;
                // TODO (hme): add validation for strides here too.
                // Runtime extents for launch-time check hoisting (indexed by
                // signature position). An immutable tensor is passed whole, so
                // its view is its root shape.
                if !validator.launch_checks.is_empty() {
                    param_shapes[#var_idx] = #var_ident.shape().to_vec();
                    view_shapes[#var_idx] = param_shapes[#var_idx].clone();
                }
            }
        }})
        .unwrap()
    };

    let element_type_name = if ty.mutability.is_none() {
        Some(dtype.clone())
    } else {
        None
    };
    Ok(TensorLaunchCode {
        fn_arg,
        stride_expr_str,
        spec_expr_str,
        spec_ref_expr_str,
        builder_statements,
        launch_grid_expr_strs,
        validator_statements,
        element_type_name,
    })
}

fn infer_mapped_partition_map_generics(
    var_name: &str,
    map_shape: &GenericArgument,
    required_generics: &mut RequiredGenerics,
) -> Result<(), Error> {
    let mapped_shape_expr = |dim: usize| {
        format!(
            "vec![KernelOutputStored::map_shape_as_i32(&{var_name}).expect(\"MappedPartitionMut missing map shape\")[{dim}].to_string()]"
        )
    };
    match map_shape {
        GenericArgument::Type(Type::Path(path)) => {
            let Some(segment) = path.path.segments.last() else {
                return map_shape
                    .err("MappedPartitionMut map shape must be a const generic array.");
            };
            let ident = segment.ident.to_string();
            match required_generics.get_ty(&ident) {
                SupportedGenericType::ConstArray => {
                    required_generics.expressions.insert(
                        ident,
                        Some(format!(
                            "KernelOutputStored::map_shape_as_i32(&{var_name}).expect(\"MappedPartitionMut missing map shape\").iter().map(|x| x.to_string()).collect::<Vec<String>>()"
                        )),
                    );
                    Ok(())
                }
                SupportedGenericType::Unknown => Ok(()),
                _ => map_shape.err(
                    "MappedPartitionMut map shape generic must be a const generic array parameter.",
                ),
            }
        }
        GenericArgument::Const(Expr::Block(block)) => {
            if block.block.stmts.len() != 1 {
                return block
                    .err("MappedPartitionMut map shape block must contain one expression.");
            }
            let Stmt::Expr(Expr::Array(array), _) = &block.block.stmts[0] else {
                return block.err("MappedPartitionMut map shape must be an array expression.");
            };
            for (dim, elem) in array.elems.iter().enumerate() {
                if let Expr::Path(path) = elem {
                    let ident = get_ident_from_path_expr(path).to_string();
                    match required_generics.get_ty(&ident) {
                        SupportedGenericType::ConstScalar => {
                            required_generics
                                .expressions
                                .insert(ident, Some(mapped_shape_expr(dim)));
                        }
                        SupportedGenericType::Unknown => {}
                        _ => {
                            return elem.err(
                                "MappedPartitionMut map shape elements must be i32 const scalar parameters.",
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        _ => map_shape.err("MappedPartitionMut map shape must be a const generic array."),
    }
}

/// Infers and registers runtime expressions for tensor shape parameters.
///
/// Analyzes a tensor type's generic arguments and generates runtime expressions
/// to extract shape information from the tensor at runtime. This enables the
/// launcher to provide shape information to the MLIR compiler for kernel
/// instantiation.
///
/// ## Parameters
///
/// - `var_name`: Name of the tensor variable
/// - `type_generic_args`: Generic arguments from the tensor type (e.g., `<T, {[128, N]}>`)
/// - `required_generics`: Accumulator for tracking required generic constraints
/// - `is_mutable`: Whether the tensor parameter is mutable (affects which shape to use)
///
/// ## Behavior
///
/// For each generic argument:
/// - **Type parameters** (e.g., `T`): Generates `var_name.dtype().as_str().to_string()`
/// - **Const array parameters** (e.g., `N`, `S`):
///   - Mutable: Generates `var_name.partition_shape` extraction
///   - Immutable: Generates `var_name.shape` extraction
///
/// ## Example
///
/// For `x: &mut Tensor<T, {[128, N]}>`, generates:
/// - `T` → `x.dtype().as_str().to_string()`
/// - `N` → Extract from `x.partition_shape`
pub fn infer_shape_params_from_tensor_type(
    var_name: &str,
    type_generic_args: &AngleBracketedGenericArguments,
    required_generics: &mut RequiredGenerics,
    is_mutable: bool,
) -> Result<(), Error> {
    // We assume this is a variadic type.
    for generic_arg in &type_generic_args.args {
        match generic_arg {
            GenericArgument::Type(syn::Type::Path(type_path)) => {
                // Currently, this is either shape or element_type.
                let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                match required_generics.get_ty(&last_ident) {
                    SupportedGenericType::TypeParam => {
                        // This is an element type.
                        required_generics
                            .launcher_type_params
                            .push(last_ident.clone());
                        required_generics.expressions.insert(
                            last_ident.clone(),
                            Some(format!("vec![{var_name}.dtype_str().to_string()]")),
                        );
                    }
                    SupportedGenericType::ConstArray => {
                        // This is a CGA type.
                        if is_mutable {
                            required_generics.expressions.insert(last_ident.clone(), Some(format!("KernelOutputStored::partition_shape_as_i32(&{var_name}).iter().map(|x| x.to_string()).collect::<Vec<String>>()")));
                        } else {
                            // This might make sense for a small tensor.
                            required_generics.expressions.insert(last_ident.clone(), Some(format!("{var_name}.shape().iter().map(|x| x.to_string()).collect::<Vec<String>>()")));
                        }
                    }
                    SupportedGenericType::ConstScalar => {
                        return type_path
                            .err("Unexpected constant scalar type in tensor generic argument.");
                    }
                    SupportedGenericType::Unknown => {}
                }
            }
            GenericArgument::Const(Expr::Block(block_expr)) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                // This is something like Tensor<E, {[...]}>
                if block_expr.block.stmts.len() != 1 {
                    return block_expr.err(&format!(
                        "Expected exactly 1 statement in block expression, got {}.",
                        block_expr.block.stmts.len()
                    ));
                }
                let statement = &block_expr.block.stmts[0];
                let Stmt::Expr(statement_expr, _) = statement else {
                    return block_expr
                        .err("Unexpected block expression: expected an expression statement.");
                };
                match statement_expr {
                    Expr::Array(array_expr) => {
                        // This is something like Tensor<E, {[1, 2, -1]}>
                        for (i, elem) in array_expr.elems.iter().enumerate() {
                            match elem {
                                Expr::Lit(_lit) => {
                                    // Nothing to do to build generic arg expressions.
                                    continue;
                                }
                                Expr::Unary(_unary_expr) => {
                                    // Nothing to do to build generic arg expressions.
                                    continue;
                                }
                                Expr::Path(path) => {
                                    let ident = get_ident_from_path_expr(path).to_string();
                                    match required_generics.get_ty(&ident) {
                                        SupportedGenericType::TypeParam => {
                                            // This is an element type.
                                            return path.err(
                                                "Unexpected type param in array type expression.",
                                            );
                                        }
                                        SupportedGenericType::ConstArray => {
                                            // This is a CGA type.
                                            return path.err("Unexpected const generic array param in array type expression.");
                                        }
                                        SupportedGenericType::ConstScalar => {
                                            if is_mutable {
                                                required_generics.expressions.insert(ident.clone(), Some(format!("vec![KernelOutputStored::partition_shape_as_i32(&{var_name})[{i}].to_string()]")));
                                            } else {
                                                required_generics.expressions.insert(
                                                    ident.clone(),
                                                    Some(format!(
                                                        "vec![{var_name}.shape()[{i}].to_string()]"
                                                    )),
                                                );
                                            }
                                        }
                                        SupportedGenericType::Unknown => {}
                                    }
                                }
                                Expr::Index(index) => {
                                    let Expr::Path(path) = index.expr.as_ref() else {
                                        return elem.err(
                                            "Unsupported array element in tensor shape expression.",
                                        );
                                    };
                                    let ident = get_ident_from_path_expr(path).to_string();
                                    match required_generics.get_ty(&ident) {
                                        SupportedGenericType::ConstArray => {
                                            // A projection like `S[0]` does not determine the
                                            // whole `S`; keep it explicit instead of rejecting the
                                            // entry signature.
                                            continue;
                                        }
                                        SupportedGenericType::Unknown => {}
                                        _ => {
                                            return elem.err(
                                                "Only const generic array projections like `S[0]` are supported in tensor shape index expressions.",
                                            );
                                        }
                                    }
                                }
                                _ => {
                                    return elem.err(
                                        "Unsupported array element in tensor shape expression.",
                                    );
                                }
                            }
                        }
                    }
                    Expr::Repeat(repeat_expr) => {
                        // TODO (hme): Unclear under what circumstance it would be beneficial to support this.
                        return repeat_expr
                            .err("Repeat expressions in tensor shape are not yet supported.");
                    }
                    _ => {
                        return block_expr
                            .err("Unexpected block expression in tensor const generic argument.");
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
