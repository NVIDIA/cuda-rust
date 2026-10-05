/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic parameter resolution: maps Rust generic type and const parameters to
//! concrete values, and infers generic arguments from call-site context.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use crate::syn_utils::{
    get_call_expression_generics, get_ident_from_path_expr, get_ident_generic_args, get_sig_types,
    get_supported_generic_params, get_type_ident, maybe_generic_args, strip_generic_args_lifetimes,
    CGAParameter, VarCGAParameter,
};
use crate::types::{
    get_ptr_type, get_ptr_type_instance, is_element_type, is_element_type_ptr,
    parse_signed_literal_as_i32, try_extract_cga,
};
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use syn::{
    AngleBracketedGenericArguments, Expr, ExprCall, ExprMethodCall, GenericArgument, GenericParam,
    Generics, ImplItemFn, ItemImpl, Lit, PathArguments, Signature, Stmt, Type, TypePath,
};

/// Classification of a generic parameter variable.
#[derive(Debug)]
pub enum GenericVarType {
    // This is a generic type parameter. The T in <T, ...>
    /// A generic type parameter (the `T` in `<T, ...>`).
    TypeVariable,
    // This is a const generic parameter. The const X: T in <const X: T, ...>
    /// A const generic parameter (the `const X: T` in `<const X: T, ...>`).
    ConstVariable,
    // This is a length variable. The N in <const X: [i32; N], ...>
    /// An array length variable (the `N` in `<const X: [i32; N], ...>`).
    LengthVariable,
}

/// `GenericVars` = symbol table that maps generic parameter names (`E`, `BM`) to their concrete instantiated values (`f32`, `64`) for this specific compilation.
///
/// Rust generics need concrete values at compile-time.
///
/// **Example kernel**:
/// ```rust,ignore
/// fn gemm<E: ElementType, const BM: i32, const BN: i32, const K: i32>(
///     z: &mut Tensor<E, {[BM, BN]}>,
///     x: &Tensor<E, {[BM, K]}>,
///     y: &Tensor<E, {[K, BN]}>
/// )
/// ```
/// **When called with**: `gemm::<f32, 64, 64, 32>(...)`
/// **GenericVars stores the mapping**:
/// ```rust,ignore
/// GenericVars {
///     inst_types: {
///         "E" => f32
///     },
///     inst_i32: {
///         "BM" => 64,
///         "BN" => 64,
///         "K" => 32
///     }
/// }
/// ```
/// **Usage**: When compiler sees `Tensor<E, {[BM, BN]}>` in the code:
///  - Lookup `"E"` → `f32`
///  - Lookup `"BM"` → `64`, `"BN"` → `64`
///  - Substitute: `Tensor<f32, {[64, 64]}>`
#[derive(Debug, Clone)]
pub struct GenericVars {
    // Generic type param (the T in scalar: T or x: Tensor<T, ...>) to concrete impl ElementType.
    // TODO (hme): Ensure the type mapped here is _always_ concrete. Check inlining specifically.
    pub inst_types: HashMap<String, String>,
    // Generic i32 param name to i32.
    pub inst_i32: HashMap<String, i32>,
    // Generic bool param name to bool.
    pub inst_bool: HashMap<String, bool>,
    // Generic array param name to generic array instance.
    pub inst_array: HashMap<String, Vec<i32>>,
    // A map from the length variable of a const generic array to the corresponding key in inst_array.
    pub len2array: HashMap<String, String>,
    pub ordered_param_vars: Vec<String>,
}

impl GenericVars {
    fn ordered_map_keys<'a, T>(&'a self, map: &'a HashMap<String, T>) -> Vec<&'a String> {
        let mut keys = Vec::new();
        let mut seen = HashSet::new();

        for param in &self.ordered_param_vars {
            if map.contains_key(param) && seen.insert(param.as_str()) {
                keys.push(param);
            }
        }

        let mut remaining = map
            .keys()
            .filter(|key| !seen.contains(key.as_str()))
            .collect::<Vec<_>>();
        remaining.sort();
        keys.extend(remaining);
        keys
    }

    pub fn ordered_inst_i32(&self) -> Vec<(&str, i32)> {
        self.ordered_map_keys(&self.inst_i32)
            .into_iter()
            .map(|key| (key.as_str(), self.inst_i32[key]))
            .collect()
    }

    pub fn ordered_inst_bool(&self) -> Vec<(&str, bool)> {
        self.ordered_map_keys(&self.inst_bool)
            .into_iter()
            .map(|key| (key.as_str(), self.inst_bool[key]))
            .collect()
    }

    pub fn ordered_inst_array(&self) -> Vec<(&str, &[i32])> {
        self.ordered_map_keys(&self.inst_array)
            .into_iter()
            .map(|key| (key.as_str(), self.inst_array[key].as_slice()))
            .collect()
    }

    /// Returns the kind of generic variable for the given name, if it exists.
    pub fn var_type(&self, var: &str) -> Option<GenericVarType> {
        if self.inst_types.contains_key(var) {
            Some(GenericVarType::TypeVariable)
        } else if self.len2array.contains_key(var) {
            Some(GenericVarType::LengthVariable)
        } else if self.inst_i32.contains_key(var)
            || self.inst_bool.contains_key(var)
            || self.inst_array.contains_key(var)
        {
            Some(GenericVarType::ConstVariable)
        } else {
            None
        }
    }

    pub fn is_empty(generics: &Generics) -> bool {
        for generic_param in &generics.params {
            match generic_param {
                GenericParam::Type(_) => return false,
                GenericParam::Const(_) => return false,
                GenericParam::Lifetime(_) => continue,
            }
        }
        true
    }
    /// Creates an empty instance without validation. Use [`empty`](Self::empty) when possible.
    pub fn empty_unchecked() -> Self {
        let inst_types: HashMap<String, String> = HashMap::new();
        let inst_i32: HashMap<String, i32> = HashMap::new();
        let inst_bool: HashMap<String, bool> = HashMap::new();
        let inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let len2array: HashMap<String, String> = HashMap::new();
        let ordered_param_vars: Vec<String> = Vec::new();
        GenericVars {
            inst_types,
            inst_i32,
            inst_bool,
            inst_array,
            len2array,
            ordered_param_vars,
        }
    }
    /// Creates an empty instance, returning an error if the signature has unresolved generics.
    pub fn empty(generics: &Generics) -> Result<Self, JITError> {
        if !Self::is_empty(generics) {
            return SourceLocation::unknown().jit_error_result(
                "expected no generic parameters, but found type or const parameters",
            );
        }
        Ok(Self::empty_unchecked())
    }
    pub fn from_flat(generics: &Generics, args: &[String]) -> Result<Self, JITError> {
        // This is used to initialize generics for kernel entry points.
        let mut inst_types: HashMap<String, String> = HashMap::new();
        // These are kernel entry points and require known length.
        // There are no variable length consts.
        let mut inst_i32: HashMap<String, i32> = HashMap::new();
        let mut inst_bool: HashMap<String, bool> = HashMap::new();
        let mut inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let len2array: HashMap<String, String> = HashMap::new();
        let mut ordered_param_vars: Vec<String> = Vec::new();
        let mut pos: usize = 0;
        for generic_param in &generics.params {
            match generic_param {
                GenericParam::Const(const_param) => {
                    // const X: i32 or const Y: [i32; 3]
                    match &const_param.ty {
                        syn::Type::Array(_ty_arr) => {
                            let cga_param = CGAParameter::from_const_param(const_param);
                            let mut inst: Vec<i32> = vec![];
                            if pos + cga_param.length as usize > args.len() {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "not enough generic arguments to instantiate const array parameter `{}`; need {} more value(s)",
                                    cga_param.name, cga_param.length as usize - (args.len() - pos)
                                ));
                            }
                            for _ in 0..cga_param.length {
                                let arg = args[pos].parse::<i32>().map_err(|err| {
                                    SourceLocation::unknown().jit_error(&format!(
                                        "generic argument `{}` for const array parameter `{}` is not an i32: {err}",
                                        args[pos], cga_param.name
                                    ))
                                })?;
                                inst.push(arg);
                                pos += 1;
                            }
                            ordered_param_vars.push(cga_param.name.clone());
                            inst_array.insert(cga_param.name.clone(), inst);
                        }
                        syn::Type::Path(type_path) => {
                            if pos + 1 > args.len() {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "not enough generic arguments to instantiate const parameter `{}`",
                                    const_param.ident
                                ));
                            }
                            let name = const_param.ident.to_string();
                            let ty = type_path.to_token_stream().to_string();
                            if ty == "i32" {
                                let arg = args[pos].parse::<i32>().map_err(|err| {
                                    SourceLocation::unknown().jit_error(&format!(
                                        "generic argument `{}` for const parameter `{}` is not an i32: {err}",
                                        args[pos], const_param.ident
                                    ))
                                })?;
                                ordered_param_vars.push(name.clone());
                                inst_i32.insert(name, arg);
                            } else if ty == "bool" {
                                let arg = args[pos].parse::<bool>().map_err(|err| {
                                    SourceLocation::unknown().jit_error(&format!(
                                        "failed to parse bool generic argument `{}` for `{}`: {err}",
                                        args[pos], const_param.ident
                                    ))
                                })?;
                                ordered_param_vars.push(name.clone());
                                inst_bool.insert(name, arg);
                            } else {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "const generic `{}` must be `i32` or `bool`, got `{ty}`",
                                    const_param.ident
                                ));
                            }
                            pos += 1;
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unsupported type for const generic parameter `{}`; only `i32`, `bool`, and `[i32; N]` are supported",
                                const_param.ident
                            ));
                        }
                    }
                }
                GenericParam::Type(type_param) => {
                    // Host-supplied generics can run short (a launcher built
                    // with too few `.generics(...)`); indexing past them was
                    // a panic, not a diagnostic.
                    if pos >= args.len() {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "not enough generic arguments to instantiate type parameter `{}`: \
                             expected at least {} but got {}",
                            type_param.ident,
                            pos + 1,
                            args.len()
                        ));
                    }
                    let var_str = type_param.ident.to_string();
                    ordered_param_vars.push(var_str.clone());
                    inst_types.insert(var_str, args[pos].clone());
                    pos += 1;
                }
                GenericParam::Lifetime(_) => {}
            }
        }
        if pos != args.len() {
            return SourceLocation::unknown().jit_error_result(&format!(
                "too many generic arguments: expected {pos} but got {}",
                args.len()
            ));
        }
        Ok(GenericVars {
            inst_types,
            inst_i32,
            inst_bool,
            inst_array,
            len2array,
            ordered_param_vars,
        })
    }
    /// Constructs a `GenericVars` by matching explicit generic arguments to parameter declarations.
    pub fn from_expr_generic_args(
        &self,
        generics: &Generics,
        expr_generic_args: &Option<AngleBracketedGenericArguments>,
    ) -> Result<Self, JITError> {
        // This is used to initialize generics for inlined functions and methods.
        // self are the generics from the caller scope.
        // generics are generics from the callee scope.
        // The returned GenericVars instance are instantiated generic vars for the callee.
        // println!("inlining function with \n generics={generics:#?} \n expr_generic_args={expr_generic_args:#?}");
        let mut inst_types: HashMap<String, String> = HashMap::new();
        let mut inst_i32: HashMap<String, i32> = HashMap::new();
        let mut inst_bool: HashMap<String, bool> = HashMap::new();
        let mut inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let mut len2array: HashMap<String, String> = HashMap::new();
        let mut ordered_param_vars: Vec<String> = vec![];
        if expr_generic_args.is_none() {
            return Ok(GenericVars {
                inst_types,
                inst_i32,
                inst_bool,
                inst_array,
                len2array,
                ordered_param_vars,
            });
        }
        let expr_generic_args = expr_generic_args.as_ref().unwrap();
        if generics.params.len() != expr_generic_args.args.len() {
            return SourceLocation::unknown().jit_error_result(&format!(
                "generic parameter count ({}) does not match argument count ({})",
                generics.params.len(),
                expr_generic_args.args.len()
            ));
        }

        let num_args = expr_generic_args.args.len();
        for i in 0..num_args {
            let generic_arg = &expr_generic_args.args[i];
            let generic_param = &generics.params[i];
            match (generic_arg, generic_param) {
                (GenericArgument::Const(const_arg), GenericParam::Const(const_param)) => {
                    // Instantiate a const generic param from a const arg expr.
                    match (&const_arg, &const_param.ty) {
                        (syn::Expr::Path(arg_path_expr), syn::Type::Array(_param_ty_array)) => {
                            let ident_str = get_ident_from_path_expr(arg_path_expr).to_string();
                            if VarCGAParameter::is_var_cga(const_param) {
                                let var_cga_param = VarCGAParameter::from_const_param(const_param);
                                let name = var_cga_param.name.to_string();
                                let Some(inst) = self.inst_array.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known const generic array"
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name.clone(), inst.clone());
                                len2array.insert(var_cga_param.length_var, name);
                            } else {
                                let cga_param = CGAParameter::from_const_param(const_param);
                                let name = cga_param.name.to_string();
                                let Some(inst) = self.inst_array.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known const generic array"
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name, inst.clone());
                                if cga_param.length as usize != inst.len() {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "const array parameter `{}` expects {} elements but got {}",
                                        cga_param.name,
                                        cga_param.length,
                                        inst.len()
                                    ));
                                }
                            }
                        }
                        (syn::Expr::Path(arg_path_expr), syn::Type::Path(param_ty_path)) => {
                            let name = const_param.ident.to_string();
                            let ident_str = get_ident_from_path_expr(arg_path_expr).to_string();
                            let param_ty = param_ty_path.to_token_stream().to_string();
                            ordered_param_vars.push(name.clone());
                            if param_ty == "bool" {
                                let Some(inst) = self.inst_bool.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known bool const generic"
                                    ));
                                };
                                inst_bool.insert(name, *inst);
                            } else {
                                let Some(inst) = self.inst_i32.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known const generic scalar"
                                    ));
                                };
                                inst_i32.insert(name, *inst);
                            }
                        }
                        (syn::Expr::Lit(arg_lit_expr), syn::Type::Path(param_ty_path)) => {
                            let name = const_param.ident.to_string();
                            ordered_param_vars.push(name.clone());
                            if param_ty_path.to_token_stream().to_string() == "bool" {
                                let Lit::Bool(bool_lit) = &arg_lit_expr.lit else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "expected bool literal for const generic `{name}`"
                                    ));
                                };
                                inst_bool.insert(name, bool_lit.value);
                            } else {
                                let literal_i32 = parse_signed_literal_as_i32(const_arg);
                                inst_i32.insert(name, literal_i32);
                            }
                        }
                        (syn::Expr::Block(_), syn::Type::Array(ty_arr)) => {
                            let name = const_param.ident.to_string();
                            let from_generic_args = self;
                            if let Some(res) = try_get_const_generic_from_generic_argument(
                                generic_arg,
                                from_generic_args,
                            ) {
                                // This is something like
                                // CONST_ARG -> N
                                inst_i32.insert(name.clone(), res);
                            } else {
                                let Some(res) =
                                    get_cga_from_generic_argument(generic_arg, from_generic_args)
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "unable to resolve generic argument `{}` for parameter `{}`",
                                        generic_arg.to_token_stream(),
                                        generic_param.to_token_stream()
                                    ));
                                };
                                // This is something like
                                // {[...]} -> CONST_ARRAY_PARAM
                                inst_array.insert(name.clone(), res);
                                if let Expr::Path(length_expr) = &ty_arr.len {
                                    let length_var = length_expr
                                        .path
                                        .get_ident()
                                        .unwrap()
                                        .to_string()
                                        .to_string();
                                    len2array.insert(length_var, name.clone());
                                }
                            }
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unable to resolve generic argument `{}` for parameter `{}`",
                                generic_arg.to_token_stream(),
                                generic_param.to_token_stream()
                            ));
                        }
                    }
                }
                (GenericArgument::Type(ty_arg), GenericParam::Const(const_param)) => {
                    // Instantiate a const generic param from a type arg.
                    // println!("instantiating generic={const_param:#?} \n expr={generic_arg:#?}");
                    let Some(arg_ident_str) = get_type_ident(ty_arg) else {
                        return SourceLocation::unknown().jit_error_result(
                            "unable to extract type identifier from const generic argument",
                        );
                    };
                    match &const_param.ty {
                        syn::Type::Array(_param_ty_array) => {
                            if VarCGAParameter::is_var_cga(const_param) {
                                let var_cga_param = VarCGAParameter::from_const_param(const_param);
                                let name = var_cga_param.name.to_string();
                                let Some(inst) =
                                    self.inst_array.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known const generic array",
                                        arg_ident_str
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name.clone(), inst.clone());
                                len2array.insert(var_cga_param.length_var, name);
                            } else {
                                let cga_param = CGAParameter::from_const_param(const_param);
                                let name = cga_param.name.to_string();
                                let Some(inst) =
                                    self.inst_array.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known const generic array",
                                        arg_ident_str
                                    ));
                                };
                                if cga_param.length as usize != inst.len() {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "const array parameter `{}` expects {} elements but got {}",
                                        cga_param.name,
                                        cga_param.length,
                                        inst.len()
                                    ));
                                }
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name, inst.clone());
                            }
                        }
                        syn::Type::Path(param_ty_path) => {
                            let name = const_param.ident.to_string();
                            ordered_param_vars.push(name.clone());
                            if param_ty_path.to_token_stream().to_string() == "bool" {
                                let Some(inst) =
                                    self.inst_bool.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known bool const generic",
                                        arg_ident_str
                                    ));
                                };
                                inst_bool.insert(name.clone(), *inst);
                            } else {
                                let Some(inst) =
                                    self.inst_i32.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known const generic scalar",
                                        arg_ident_str
                                    ));
                                };
                                inst_i32.insert(name.clone(), *inst);
                            }
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unable to resolve generic argument `{}` for parameter `{}`",
                                generic_arg.to_token_stream(),
                                generic_param.to_token_stream()
                            ));
                        }
                    }
                }
                (GenericArgument::Type(ty_arg), GenericParam::Type(ty_param)) => {
                    // Instantiate a type parameter from a type arg.
                    let Some(arg_ident_str) = get_type_ident(ty_arg) else {
                        return SourceLocation::unknown().jit_error_result(
                            "unable to extract type identifier from type argument",
                        );
                    };
                    let name = ty_param.ident.to_string();
                    let Some(inst) = self.inst_types.get(arg_ident_str.to_string().as_str()) else {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "type `{}` is not a known generic type parameter",
                            arg_ident_str
                        ));
                    };
                    ordered_param_vars.push(name.clone());
                    inst_types.insert(name.clone(), inst.clone());
                }
                (_, _) => {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Generic arg / param not supported: {generic_arg:#?} \n {generic_param:#?}"
                    ));
                }
            }
        }
        Ok(GenericVars {
            inst_types,
            inst_i32,
            inst_bool,
            inst_array,
            len2array,
            ordered_param_vars,
        })
    }
    /// Looks up a const-generic `i32` value by parameter name.
    pub fn get_i32(&self, name: &str) -> Option<i32> {
        if let Some(inst) = self.inst_i32.get(name) {
            return Some(*inst);
        }
        if let Some(arr_name) = self.len2array.get(name) {
            // Internal invariant: the length var exists but the array may not; `?` yields None.
            return Some(self.inst_array.get(arr_name)?.len() as i32);
        }
        None
    }

    /// Looks up a const-generic `bool` value by parameter name.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.inst_bool.get(name).copied()
    }

    /// Merges another `GenericVars` into this one, erroring on conflicting entries.
    pub fn merge(mut self, other: GenericVars) -> Result<GenericVars, JITError> {
        self.inst_types = self.inst_types.merge_if_eq(other.inst_types);
        self.inst_i32 = self.inst_i32.merge_if_eq(other.inst_i32);
        self.inst_bool = self.inst_bool.merge_if_eq(other.inst_bool);
        self.inst_array = self.inst_array.merge_if_eq(other.inst_array);
        self.len2array = self.len2array.merge_if_eq(other.len2array);
        if !self.ordered_param_vars.is_empty() && !other.ordered_param_vars.is_empty() {
            if self.ordered_param_vars != other.ordered_param_vars {
                return SourceLocation::unknown().jit_error_result(&format!(
                    "Ordered param vars mismatch: {:?} != {:?}",
                    self.ordered_param_vars, other.ordered_param_vars
                ));
            }
        } else {
            // At least one is empty. Take the value of the other.
            self.ordered_param_vars.extend(other.ordered_param_vars);
        }
        Ok(self)
    }

    /// Resolves a possibly-generic `syn::Type` into a concrete [`TypeInstance`].
    pub fn instantiate_type(
        &self,
        ty: &syn::Type,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<TypeInstance, JITError> {
        // Identify all generic args, and replace them with concrete (instantiated) values.
        // The Instantiable trait returns an instance even if the given type is concrete.
        let maybe_generic_ty = ty.clone();
        // Is this an element type? This is just a T.
        if let Some(instance) =
            TypeInstanceTokenType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::TokenType(instance));
        }
        if let Some(instance) =
            TypeInstanceElementType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::ElementType(instance));
        }
        // Is this a pointer? Something like * mut T.
        if let Some(instance) =
            TypeInstancePtrType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::PtrType(instance));
        }
        // Is this a string?
        if let Some(instance) =
            TypeInstanceStringType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::StringType(instance));
        }
        // Assume it's a structured type, something like Tile/Tensor/Partition/PartitionMut <T, Shape>
        // This also handles PointerTile<* mut T, Shape>.
        if let Some(instance) =
            TypeInstanceStructuredType::instantiate(&maybe_generic_ty, self, primitives)
        {
            Ok(TypeInstance::StructuredType(instance))
        } else {
            SourceLocation::unknown().jit_error_result(&format!(
                "unable to resolve generic type `{}`",
                maybe_generic_ty.to_token_stream()
            ))
        }
    }
}

/// Trait for types that can be instantiated from a generic `syn::Type`.
pub trait Instantiable {
    fn instantiate(
        generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self>
    where
        Self: Sized;
}

/// A fully-instantiated type, classifying how a Rust type maps to the CUDA Tile type system.
#[derive(Debug, Clone)]
pub enum TypeInstance {
    /// A plain user-defined or unresolved type.
    UserType(TypeInstanceUserType),
    /// The `str` string type.
    StringType(TypeInstanceStringType),
    /// The `Token` ordering-token type.
    TokenType(TypeInstanceTokenType),
    /// A scalar element type (e.g. `f32`, `i32`).
    ElementType(TypeInstanceElementType),
    /// A pointer type (`*mut E` / `*const E`).
    PtrType(TypeInstancePtrType),
    /// A shaped type with element type and dimensions (e.g. `Tile<f32, {[128]}>`).
    StructuredType(TypeInstanceStructuredType),
}

impl TypeInstance {
    /// Returns the concrete Rust element type name, if applicable.
    pub fn get_rust_element_instance_ty(&self) -> Option<String> {
        match self {
            Self::UserType(_inst) => None,
            Self::StringType(_inst) => None,
            Self::TokenType(_inst) => None,
            Self::ElementType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::PtrType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::StructuredType(inst) => {
                if let Some(primitive_type) = &inst.primitive_type {
                    primitive_type.get_rust_element_instance_ty()
                } else {
                    None
                }
            }
        }
    }
    /// Returns the concrete (instantiated) `syn::Type`.
    pub fn get_instantiated_type(&self) -> &syn::Type {
        match self {
            Self::UserType(inst) => &inst.maybe_generic_ty,
            Self::StringType(inst) => &inst.instance_ty,
            Self::TokenType(inst) => &inst.instance_ty,
            Self::ElementType(inst) => &inst.instance_ty,
            Self::PtrType(inst) => &inst.instance_ty,
            Self::StructuredType(inst) => &inst.instance_ty,
        }
    }
    /// Returns the original (possibly generic) `syn::Type`.
    pub fn get_source_type(&self) -> &syn::Type {
        match self {
            Self::UserType(inst) => &inst.maybe_generic_ty,
            Self::StringType(inst) => &inst.generic_ty,
            Self::TokenType(inst) => &inst.generic_ty,
            Self::ElementType(inst) => &inst.generic_ty,
            Self::PtrType(inst) => &inst.generic_ty,
            Self::StructuredType(inst) => &inst.generic_ty,
        }
    }
}

#[derive(Debug, Clone)]
/// Primitive type instance: either a scalar element type or a pointer type.
pub enum TypInstancePrimitiveType {
    /// Scalar element type (e.g. `f32`).
    ElementType(TypeInstanceElementType),
    /// Pointer type (e.g. `*mut f32`).
    PtrType(TypeInstancePtrType),
}

impl TypInstancePrimitiveType {
    pub fn get_rust_element_instance_ty(&self) -> Option<String> {
        match self {
            Self::ElementType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::PtrType(inst) => Some(inst.rust_element_instance_ty.clone()),
        }
    }
    pub fn get_instantiated_type(&self) -> &syn::Type {
        match self {
            Self::ElementType(inst) => &inst.instance_ty,
            Self::PtrType(inst) => &inst.instance_ty,
        }
    }
}

#[derive(Debug, Clone)]
/// A user-defined or unresolved type that may still contain generics.
pub struct TypeInstanceUserType {
    pub(crate) maybe_generic_ty: syn::Type,
}

impl Instantiable for TypeInstanceUserType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // TODO (np): Add check for unresolved generics - return None if the type contains
        // generic parameters that haven't been instantiated yet. For now, we accept all types.
        Some(Self {
            maybe_generic_ty: maybe_generic_ty.clone(),
        })
    }
}

impl TypeInstanceUserType {
    /// Attempts to extract a const generic array from this type's generic arguments.
    pub fn try_extract_cga(&self, generic_vars: &GenericVars) -> Option<Vec<i32>> {
        try_extract_cga(&self.maybe_generic_ty, generic_vars)
    }
}

#[derive(Debug, Clone)]
/// A resolved `str` string type instance.
pub struct TypeInstanceStringType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
}

impl Instantiable for TypeInstanceStringType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // let string_lit = syn::Type::Path(maybe_generic_ty) else {
        //     panic!()
        // };
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if maybe_generic_type_str == "str" {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved `Token` ordering-token type instance.
pub struct TypeInstanceTokenType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
}

impl Instantiable for TypeInstanceTokenType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if maybe_generic_type_str == "Token" {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved scalar element type (e.g. `f32`) with its concrete Rust name.
pub struct TypeInstanceElementType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) rust_element_instance_ty: String,
}

impl Instantiable for TypeInstanceElementType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if is_element_type(&maybe_generic_type_str, primitives) {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
                rust_element_instance_ty: maybe_generic_type_str,
            })
        } else if let Some(rust_element_instance_ty) = generic_vars
            .inst_types
            .get(&maybe_generic_type_str)
            .cloned()
        {
            let instance_ty =
                syn::parse2::<syn::Type>(rust_element_instance_ty.parse().unwrap()).unwrap();
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty,
                rust_element_instance_ty,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved pointer type (`*mut E` / `*const E`) with mutability and element info.
pub struct TypeInstancePtrType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) is_mutable: bool,
    pub(crate) rust_element_instance_ty: String,
}

impl Instantiable for TypeInstancePtrType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if is_element_type_ptr(&maybe_generic_type_str, primitives) {
            let (is_mutable, ptr_ty) =
                get_ptr_type(&maybe_generic_type_str).expect("Unexpected pointer type.");
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
                is_mutable,
                rust_element_instance_ty: ptr_ty,
            })
        } else if let Some((is_mutable, ptr_ty)) = get_ptr_type(&maybe_generic_type_str) {
            let ptr_prefix = if is_mutable { "* mut " } else { "* const " };
            if let Some(concrete_ptr_ty) = generic_vars.inst_types.get(&ptr_ty) {
                let instance_ty = syn::parse2::<syn::Type>(
                    format!("{ptr_prefix} {concrete_ptr_ty}").parse().unwrap(),
                )
                .unwrap();
                Some(Self {
                    generic_ty: maybe_generic_ty.clone(),
                    instance_ty,
                    is_mutable,
                    rust_element_instance_ty: concrete_ptr_ty.to_string(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved shaped type (e.g. `Tile<f32, {[128, 64]}>`) with element type and shape.
pub struct TypeInstanceStructuredType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) primitive_type: Option<TypInstancePrimitiveType>,
    pub(crate) shape: Vec<i32>,
}

impl Instantiable for TypeInstanceStructuredType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // This is Tile/Tensor/Partition/PartitionMut <T, Shape>.
        // This also handles PointerTile <* mut T, Shape>.
        let mut type_generic_args = maybe_generic_args(maybe_generic_ty)?;
        strip_generic_args_lifetimes(&mut type_generic_args);
        let generic_ty = maybe_generic_ty.clone();
        let mut instance_ty = maybe_generic_ty.clone();
        let mut primitive_type: Option<TypInstancePrimitiveType> = None;
        let mut shape: Option<Vec<i32>> = None;
        let structured_type_name = get_type_ident(maybe_generic_ty).map(|ident| ident.to_string());
        let allows_extra_cga = matches!(
            structured_type_name.as_deref(),
            Some("MappedPartitionMut" | "PartitionIndices" | "GatherScatterView" | "StridedView")
        );

        let inst_mut_ref = if let Type::Reference(inner_elem) = &mut instance_ty {
            &mut *inner_elem.elem
        } else {
            &mut instance_ty
        };
        let instance_generics = if let Type::Path(type_path) = inst_mut_ref {
            let last_seg = type_path
                .path
                .segments
                .last_mut()
                .unwrap_or_else(|| panic!("Unexpected structured type {maybe_generic_ty:#?}."));
            let PathArguments::AngleBracketed(type_params) = &mut last_seg.arguments else {
                panic!(
                    "Unexpected structured type generic arguments {:#?} for {maybe_generic_ty:#?}",
                    last_seg.arguments
                );
            };
            // This is a type of the form StructuredType<...>
            type_params
        } else {
            panic!("Unexpected structured type {maybe_generic_ty:#?}.");
        };

        for generic_arg in instance_generics.args.iter_mut() {
            match generic_arg {
                GenericArgument::Lifetime(_) => continue,
                GenericArgument::Type(type_param) => {
                    // Currently, this is either shape or element_type
                    match type_param {
                        syn::Type::Path(type_path) => {
                            let last_ident =
                                type_path.path.segments.last().unwrap().ident.to_string();
                            // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                            if generic_vars.inst_array.contains_key(&last_ident) {
                                // This is something like Shape<D> for const generic array D: [i32; N].
                                let array_instance =
                                    generic_vars.inst_array.get(&last_ident).unwrap();
                                if shape.is_some() && !allows_extra_cga {
                                    panic!("Unexpected array arg: {last_ident:#?}")
                                }
                                if shape.is_none() {
                                    shape = Some(array_instance.clone());
                                }
                                let shape_str = array_instance
                                    .iter()
                                    .map(|x| x.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                *generic_arg = GenericArgument::Const(
                                    syn::parse2::<Expr>(
                                        format!("{{[{}]}}", shape_str).parse().unwrap(),
                                    )
                                    .unwrap(),
                                );
                            } else if let Some(local_rust_element_instance_ty) =
                                generic_vars.inst_types.get(&last_ident)
                            {
                                // This is something like T.
                                let instance_ty = syn::parse2::<Type>(
                                    local_rust_element_instance_ty.parse().unwrap(),
                                )
                                .unwrap();
                                primitive_type = Some(TypInstancePrimitiveType::ElementType(
                                    TypeInstanceElementType {
                                        generic_ty: type_param.clone(),
                                        instance_ty: instance_ty.clone(),
                                        rust_element_instance_ty: local_rust_element_instance_ty
                                            .clone(),
                                    },
                                ));
                                *generic_arg = GenericArgument::Type(instance_ty);
                            } else if is_element_type(&last_ident, primitives) {
                                // This is something like f32.
                                primitive_type = Some(TypInstancePrimitiveType::ElementType(
                                    TypeInstanceElementType {
                                        generic_ty: type_param.clone(),
                                        instance_ty: type_param.clone(),
                                        rust_element_instance_ty: last_ident.clone(),
                                    },
                                ));
                                *generic_arg = GenericArgument::Type(
                                    syn::parse2::<Type>(last_ident.parse().unwrap()).unwrap(),
                                );
                            } else if let Some(value) = generic_vars.inst_i32.get(&last_ident) {
                                // This is something like N for const generic N: i32.
                                if structured_type_name.as_deref() == Some("MappedPartitionMut") {
                                    *generic_arg = GenericArgument::Const(
                                        syn::parse2::<Expr>(value.to_string().parse().unwrap())
                                            .unwrap(),
                                    );
                                } else {
                                    panic!("Unexpected const arg {last_ident} for variadic type {maybe_generic_ty:#?}");
                                }
                            } else if allows_extra_cga {
                                // Map-shape metadata beyond the tile shape does not affect
                                // TileRustType's element or shape instantiation.
                            } else {
                                // Can't resolve this type here — e.g. an element
                                // var left unbound when a `P: PointerTo<E>` generic
                                // is composed two levels deep. Give up gracefully so
                                // the caller tries other instantiation strategies and
                                // ultimately reports a spanned error, instead of
                                // panicking the whole compile.
                                return None;
                            }
                        }
                        syn::Type::Ptr(_) => {
                            let Some(ptr_inst) = TypeInstancePtrType::instantiate(
                                type_param,
                                generic_vars,
                                primitives,
                            ) else {
                                panic!("Unexpected primitives {primitives:#?}.")
                            };
                            *generic_arg = GenericArgument::Type(ptr_inst.instance_ty.clone());
                            primitive_type = Some(TypInstancePrimitiveType::PtrType(ptr_inst));
                        }
                        syn::Type::Reference(type_ref) => {
                            unimplemented!("TypeInstanceStructuredType::instantiate: Type::Reference not supported: {:#?}", type_ref);
                        }
                        _ => {}
                    }
                }
                GenericArgument::Const(const_expr) => {
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match const_expr {
                        Expr::Lit(_) | Expr::Unary(_) if allows_extra_cga => {
                            // Map-shape metadata beyond the tile shape does not affect
                            // TileRustType's element or shape instantiation.
                        }
                        Expr::Path(path) if allows_extra_cga => {
                            let ident = get_ident_from_path_expr(path);
                            if let Some(value) =
                                generic_vars.inst_i32.get(ident.to_string().as_str())
                            {
                                *generic_arg = GenericArgument::Const(
                                    syn::parse2::<Expr>(value.to_string().parse().unwrap())
                                        .unwrap(),
                                );
                            }
                        }
                        Expr::Block(block_expr) => {
                            // This is something like Tensor<E, {[...]}>
                            assert_eq!(block_expr.block.stmts.len(), 1);
                            let statement = &block_expr.block.stmts[0];
                            let Stmt::Expr(statement_expr, _) = statement else {
                                panic!("Unexpected block expression.")
                            };
                            match statement_expr {
                                Expr::Array(array_expr) => {
                                    // This is something like Tensor<E, {[1, 2, -1]}>
                                    let mut _shape = vec![];
                                    for elem in &array_expr.elems {
                                        match elem {
                                            Expr::Lit(lit) => {
                                                let val = match &lit.lit {
                                                    Lit::Int(int_lit) => int_lit.base10_parse::<i32>().unwrap(),
                                                    _ => unimplemented!("Unexpected array element {elem:#?} in {array_expr:#?}"),
                                                };
                                                _shape.push(val);
                                            },
                                            Expr::Unary(unary_expr ) => {
                                                let unary_expr_str = unary_expr.to_token_stream().to_string();
                                                if unary_expr_str == "- 1" {
                                                    _shape.push(-1);
                                                } else {
                                                    panic!("Unexpected unary expression {unary_expr_str:#?} in {array_expr:#?}")
                                                }
                                            },
                                            Expr::Path(path) => {
                                                let ident = get_ident_from_path_expr(path);
                                                match generic_vars.inst_i32.get(ident.to_string().as_str()) {
                                                    Some(val) => _shape.push(*val),
                                                    None => panic!("Undefined generic parameter {ident}")
                                                }
                                            },
                                            Expr::Index(_) => {
                                                _shape.push(parse_expr_as_i32(elem, generic_vars));
                                            }
                                            _ => unimplemented!("Unexpected array element {elem:#?} in {array_expr:#?}"),
                                        }
                                    }
                                    let shape_str = _shape
                                        .iter()
                                        .map(|x| x.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    *generic_arg = GenericArgument::Const(
                                        syn::parse2::<Expr>(
                                            format!("{{[{}]}}", shape_str).parse().unwrap(),
                                        )
                                        .unwrap(),
                                    );
                                    if shape.is_some() && !allows_extra_cga {
                                        panic!(
                                            "Unexpected array arg in structured type {maybe_generic_ty:#?}"
                                        )
                                    }
                                    if shape.is_none() {
                                        shape = Some(_shape);
                                    }
                                }
                                Expr::Repeat(repeat_expr) => {
                                    // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                                    let repeat_expr_expr = &*repeat_expr.expr;
                                    let thing_to_repeat = match repeat_expr_expr {
                                        Expr::Lit(lit) => {
                                            match &lit.lit {
                                                Lit::Int(int_lit) => int_lit.base10_parse::<i32>().unwrap(),
                                                _ => unimplemented!("Unexpected repeat expr {repeat_expr_expr:#?} in {repeat_expr:#?}"),
                                            }
                                        },
                                        Expr::Unary(unary_expr ) => {
                                            let unary_expr_str = unary_expr.to_token_stream().to_string();
                                            if unary_expr_str == "- 1" {
                                                -1
                                            } else {
                                                unimplemented!("Unexpected unary expression {repeat_expr_expr:#?} in {repeat_expr:#?}")
                                            }
                                        },
                                        Expr::Path(path) => {
                                            let ident = get_ident_from_path_expr(path);
                                            match generic_vars.inst_i32.get(ident.to_string().as_str()) {
                                                Some(val) => *val,
                                                None => panic!("Undefined generic parameter {ident}")
                                            }
                                        },
                                        _ => unimplemented!("Unexpected unary expression {repeat_expr_expr:#?} in {repeat_expr:#?}"),
                                    };
                                    let num_rep = match &*repeat_expr.len {
                                        Expr::Path(len_path) => {
                                            // This is something like Tensor<E, {[-1; N]}>
                                            let num_rep_var =
                                                len_path.to_token_stream().to_string();
                                            if generic_vars.get_i32(&num_rep_var).is_none() {
                                                panic!(
                                                    "Expected instance for generic argument {}",
                                                    num_rep_var
                                                );
                                            }
                                            generic_vars.get_i32(&num_rep_var).unwrap()
                                        }
                                        Expr::Lit(len_lit) => {
                                            // This is something like Tensor<E, {[-1; 3]}>
                                            len_lit
                                                .to_token_stream()
                                                .to_string()
                                                .parse::<i32>()
                                                .unwrap()
                                        }
                                        _ => unimplemented!(
                                            "Unexpected repeat expression: {repeat_expr:#?}"
                                        ),
                                    };
                                    let repeat_str = format!("{{[{thing_to_repeat}; {num_rep}]}}");
                                    *generic_arg = GenericArgument::Const(
                                        syn::parse2::<Expr>(repeat_str.parse().unwrap()).unwrap(),
                                    );
                                    if shape.is_some() && !allows_extra_cga {
                                        panic!(
                                            "Unexpected array arg in structured type {maybe_generic_ty:#?}"
                                        )
                                    }
                                    if shape.is_none() {
                                        shape = Some(vec![thing_to_repeat; num_rep as usize]);
                                    }
                                }
                                _ => panic!("Unexpected block expression."),
                            }
                        }
                        _ => panic!("Unexpected const expression {const_expr:#?}"),
                    }
                }
                _ => panic!("Unexpected generic argument {generic_arg:#?}"),
            }
        }
        match shape {
            Some(shape) => Some(Self {
                generic_ty,
                instance_ty,
                primitive_type,
                shape,
            }),
            _ => {
                panic!("Unable to parse {maybe_generic_ty:#?} \n primitive_type = {primitive_type:#?} \n shape = {shape:#?}");
            }
        }
    }
}

trait MergeIfEqual {
    fn merge_if_eq(self, other: Self) -> Self;
}

impl<K: Hash + Eq, V: Clone + PartialEq> MergeIfEqual for HashMap<K, V> {
    fn merge_if_eq(mut self, other: Self) -> Self {
        for (key, value) in other {
            if let Some(self_value) = self.insert(key, value.clone()) {
                assert!(self_value == value);
            }
        }
        self
    }
}

#[derive(Debug)]
/// Classification of a generic argument for inference purposes.
pub enum GenericArgType {
    // Any type. The T in <T, ...> or the element type E of a pointer *mut E.
    /// A type argument (e.g. the `T` in `<T>`).
    Type,
    // Any const generic expression. The D_i in <{[..., D_i, ...]}> or an array expression {[...]} in <..., {[...]}, ...>
    /// A const generic expression (e.g. `{[BM, BN]}`).
    GenericConstExpr,
}

#[derive(Debug)]
/// Infers generic arguments for a function or method call from the call-site context.
pub struct GenericArgInference {
    // Attempt to infer generics for function or impl/method pair from context of call site.
    // This is used to:
    // 1. Infer return type generic args.
    // 2. Construct GenericVars instance for inlined function and method calls.

    // TODO (hme): This is basically two structs at this point. Separate fn from method call.
    // pub impl_params: Option<Vec<String>>,
    pub method_params: Option<Vec<String>>,
    // pub impl_sig: Signature,
    // pub method_sig: Signature,
    pub sig: Signature,
    pub params: Vec<String>,
    pub param2arg: HashMap<String, Option<(GenericArgType, String)>>,
    pub param2cga: HashMap<String, Type>,
    /// Caller-side const-generic instances (`D` → `128`), used to normalize
    /// symbolic shape elements before unification: argument types reaching
    /// inline inference mix entry-substituted literals with unsubstituted
    /// caller symbols, and both forms of the same dimension must unify.
    /// Populated by [`Self::map_args_to_params`]; empty otherwise.
    caller_scalars: HashMap<String, i32>,
    /// Caller-side const generic array instances (`S` → `[128, 64]`), so a
    /// shape element written as a projection (`S[0]`) unifies with a
    /// per-dimension callee parameter (`fn f<const M: i32>(x: Tile<E, {[M]}>)`).
    /// Populated alongside `caller_scalars`.
    caller_arrays: HashMap<String, Vec<i32>>,
}

// TODO (hme): Separate generic parameter inference from type inference procedure.
//  Rewrite type inference procedure to instantiate types from an instance of GenericVars.
//  The basic procedure is carried out the same way, but eliminate any assumption about
//  the structure of types when inferring generic parameters from input types.
//  We still make an assumption about the structure of types on the instantiation side.
impl GenericArgInference {
    /// Creates an inference context for a method call, merging impl and method generics.
    pub fn new_method(impl_item: &ItemImpl, impl_method: &ImplItemFn) -> Self {
        let mut merged_generics = vec![];

        let impl_generics = get_supported_generic_params(&impl_item.generics);
        let mut impl_params = vec![];
        for item in &impl_generics {
            impl_params.push(item.0.clone());
            merged_generics.push(item.clone());
        }
        let method_generics = get_supported_generic_params(&impl_method.sig.generics);
        let mut method_params = vec![];
        for item in method_generics {
            method_params.push(item.0.clone());
            merged_generics.push(item)
        }

        let mut param2cga = HashMap::new();
        let mut param2arg = HashMap::new();
        let mut params = vec![];
        for (name, maybe_ty) in merged_generics {
            params.push(name.to_string());
            if let Some(ty) = maybe_ty {
                param2cga.insert(name.clone(), ty.clone());
            }
            param2arg.insert(name, None);
        }
        // TODO (hme): Change params: method_params
        //  when refactored to separate structs for method vs. fn calls.
        Self {
            sig: impl_method.sig.clone(),
            param2cga,
            param2arg,
            params,
            method_params: Some(method_params),
            caller_scalars: HashMap::new(),
            caller_arrays: HashMap::new(),
        }
    }

    pub fn new_function(sig: Signature) -> Self {
        let fn_generics = get_supported_generic_params(&sig.generics);
        let mut param2cga = HashMap::new();
        let mut param2arg = HashMap::new();
        let mut params = vec![];
        for (name, maybe_ty) in fn_generics {
            params.push(name.to_string());
            if let Some(ty) = maybe_ty {
                param2cga.insert(name.clone(), ty.clone());
            }
            param2arg.insert(name, None);
        }
        Self {
            sig,
            param2cga,
            param2arg,
            params,
            method_params: None,
            caller_scalars: HashMap::new(),
            caller_arrays: HashMap::new(),
        }
    }

    /// Maps positional call arguments to their corresponding parameter names.
    ///
    /// `caller_generic_vars` supplies the caller's const-generic instances so
    /// symbolic shape elements normalize before unification: argument types
    /// mix entry-substituted literals (`128`) with unsubstituted caller
    /// symbols (`D`) for the same dimension.
    #[allow(clippy::ptr_arg)] // public signature stays as-is
    pub fn map_args_to_params(
        &mut self,
        call_arg_rust_tys: &Vec<syn::Type>,
        self_ty: Option<&Type>,
        caller_generic_vars: &GenericVars,
    ) -> Result<(), JITError> {
        self.caller_scalars = caller_generic_vars.inst_i32.clone();
        self.caller_arrays = caller_generic_vars.inst_array.clone();
        let (fn_arg_types, _return_type) = get_sig_types(&self.sig, self_ty);
        // Get the generic parameters in this function signature.
        for i in 0..call_arg_rust_tys.len() {
            let call_arg_rust_ty = &call_arg_rust_tys[i];
            let fn_arg_types = &fn_arg_types[i];
            self.add_generic_args(fn_arg_types, call_arg_rust_ty)?;
        }
        self.propagate_pointer_to_bounds();
        Ok(())
    }

    /// Mirrors rustc's trait-driven inference for pointer ops: a generic
    /// param declared `P: PointerTo<E>` that positional binding resolved
    /// to a concrete raw pointer also determines `E` — its pointee —
    /// because the only `PointerTo` impls are `*mut E` and `*const E`.
    /// Without this, an element var that appears only in bounds and the
    /// return type (the shape of every generic pointer memory op) would
    /// stay unbound on the JIT track while rustc accepts the same call.
    fn propagate_pointer_to_bounds(&mut self) {
        for param in self.sig.generics.params.clone() {
            let syn::GenericParam::Type(type_param) = param else {
                continue;
            };
            let pointee = match self.param2arg.get(&type_param.ident.to_string()) {
                Some(Some((_, bound_arg))) => match crate::types::get_ptr_type(bound_arg) {
                    Some((_, pointee)) => pointee,
                    None => continue,
                },
                _ => continue,
            };
            for bound in &type_param.bounds {
                let syn::TypeParamBound::Trait(trait_bound) = bound else {
                    continue;
                };
                let Some(segment) = trait_bound.path.segments.last() else {
                    continue;
                };
                if segment.ident != "PointerTo" {
                    continue;
                }
                let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
                    continue;
                };
                let Some(GenericArgument::Type(element_ty)) = args.args.first() else {
                    continue;
                };
                let Some(element_ident) = get_type_ident(element_ty) else {
                    continue;
                };
                if let Some(slot @ None) = self.param2arg.get_mut(&element_ident.to_string()) {
                    *slot = Some((GenericArgType::Type, pointee.clone()));
                }
            }
        }
    }
    /// Applies explicitly provided generic arguments from a function call expression.
    pub fn apply_provided_generics_fn_call(
        &mut self,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
    ) {
        let Some(expr_generic_args) = get_call_expression_generics(call_expr) else {
            return;
        };
        assert_eq!(expr_generic_args.args.len(), self.params.len());
        for i in 0..expr_generic_args.args.len() {
            let param = &self.params[i];
            let arg = &expr_generic_args.args[i];
            if let Some(Some(_)) = self.param2arg.get(param) {
                // The type for this has already been inferred.
                // Our compiler doesn't need to check anything. Rust already has.
                continue;
            }
            match arg {
                // Get generic args from call f::<...>()
                // and infer generic parameters in fn f<...>.
                // Since these are generic arguments as part of a function call expression,
                // they are either "type" or "const."
                GenericArgument::Type(syn::Type::Infer(_)) => {
                    continue;
                }
                GenericArgument::Type(arg_ty) => {
                    let Some(arg_type_ident) = get_type_ident(arg_ty) else {
                        panic!("apply_provided_generics_fn_call: Failed to get ident for type {arg_ty:#?}");
                    };
                    let var_string = arg_type_ident.to_string();
                    let var_str = var_string.as_str();
                    if let Some(arg_type_var_type) = generic_vars.var_type(var_str) {
                        // This is a generic type variable.
                        match arg_type_var_type {
                            GenericVarType::TypeVariable => {
                                let Some(inst_type) = generic_vars.inst_types.get(var_str) else {
                                    panic!("Undefined instance type {var_str}")
                                };
                                self.param2arg.insert(
                                    param.to_string(),
                                    Some((GenericArgType::Type, inst_type.to_string())),
                                );
                            }
                            GenericVarType::ConstVariable => {
                                if let Some(inst_i32) = generic_vars.inst_i32.get(var_str) {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            (*inst_i32).to_string(),
                                        )),
                                    );
                                } else if let Some(inst_bool) = generic_vars.inst_bool.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            (*inst_bool).to_string(),
                                        )),
                                    );
                                } else if let Some(inst_arr) = generic_vars.inst_array.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            format!("{{[{:?}]}}", inst_arr),
                                        )),
                                    );
                                } else {
                                    panic!("Undefined instance type {var_str}")
                                }
                            }
                            GenericVarType::LengthVariable => {
                                panic!(
                                    "Unexpected GenericVarType::LengthVariable {var_str}. \
                                Length variables can only be inferred from const generic arrays."
                                )
                            }
                        }
                    } else {
                        // This is a type (not a type variable).
                        self.param2arg.insert(
                            param.to_string(),
                            Some((GenericArgType::Type, var_str.to_string())),
                        );
                    };
                }
                GenericArgument::Const(c) => {
                    self.param2arg.insert(
                        param.to_string(),
                        Some((
                            GenericArgType::GenericConstExpr,
                            c.to_token_stream().to_string(),
                        )),
                    );
                }
                _ => {}
            }
        }
    }

    /// Applies explicitly provided generic arguments from a method call expression.
    pub fn apply_provided_generics_method_call(
        &mut self,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
    ) {
        let Some(expr_generic_args) = &method_call_expr.turbofish else {
            return;
        };
        // These won't be the same length.
        let Some(method_params) = &self.method_params else {
            panic!(
                "Method params undefined for {}",
                method_call_expr.to_token_stream()
            )
        };
        assert_eq!(expr_generic_args.args.len(), method_params.len());
        // A method turbofish names the METHOD generics only (Rust semantics);
        // impl generics come from the receiver. `self.params` is impl params
        // followed by method params, so pair against the latter.
        let method_params = method_params.clone();
        for (param, arg) in method_params.iter().zip(expr_generic_args.args.iter()) {
            if let Some(Some(_)) = self.param2arg.get(param) {
                // The type for this has already been inferred.
                // Our compiler doesn't need to check anything. Rust already has.
                continue;
            }
            match arg {
                // Get generic args from call f::<...>()
                // and infer generic parameters in fn f<...>.
                // Since these are generic arguments as part of a function call expression,
                // they are either "type" or "const."
                GenericArgument::Type(syn::Type::Infer(_)) => {
                    continue;
                }
                GenericArgument::Type(arg_ty) => {
                    let Some(arg_type_ident) = get_type_ident(arg_ty) else {
                        panic!("apply_provided_generics_fn_call: Failed to get ident for type {arg_ty:#?}");
                    };
                    let var_string = arg_type_ident.to_string();
                    let var_str = var_string.as_str();
                    if let Some(arg_type_var_type) = generic_vars.var_type(var_str) {
                        // This is a generic type variable.
                        match arg_type_var_type {
                            GenericVarType::TypeVariable => {
                                let Some(inst_type) = generic_vars.inst_types.get(var_str) else {
                                    panic!("Undefined instance type {var_str}")
                                };
                                self.param2arg.insert(
                                    param.to_string(),
                                    Some((GenericArgType::Type, inst_type.to_string())),
                                );
                            }
                            GenericVarType::ConstVariable => {
                                if let Some(inst_i32) = generic_vars.inst_i32.get(var_str) {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            (*inst_i32).to_string(),
                                        )),
                                    );
                                } else if let Some(inst_bool) = generic_vars.inst_bool.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            (*inst_bool).to_string(),
                                        )),
                                    );
                                } else if let Some(inst_arr) = generic_vars.inst_array.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            format!("{{[{:?}]}}", inst_arr),
                                        )),
                                    );
                                } else {
                                    panic!("Undefined instance type {var_str}")
                                }
                            }
                            GenericVarType::LengthVariable => {
                                panic!(
                                    "Unexpected GenericVarType::LengthVariable {var_str}. \
                                Length variables can only be inferred from const generic arrays."
                                )
                            }
                        }
                    } else {
                        // This is a type (not a type variable).
                        self.param2arg.insert(
                            param.to_string(),
                            Some((GenericArgType::Type, var_str.to_string())),
                        );
                    };
                }
                GenericArgument::Const(c) => {
                    self.param2arg.insert(
                        param.to_string(),
                        Some((
                            GenericArgType::GenericConstExpr,
                            c.to_token_stream().to_string(),
                        )),
                    );
                }
                _ => {}
            }
        }
    }

    /// Returns `true` if all generic parameters have been resolved.
    pub fn verify(&self) -> bool {
        // Check if computed and succeeded.
        for val in self.param2arg.values() {
            if val.is_none() {
                return false;
            }
        }
        true
    }

    /// Builds a [`GenericVars`] from the inferred parameter-to-argument mapping.
    pub fn get_generic_vars_instance(
        &self,
        from_generic_args: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> GenericVars {
        // This function constructs an instance of GenericVars from a filled instance of self.param2arg.
        // If a value of self.param2arg is a key in from_generic_args, then it's a type variable.
        let mut to_generic_vars = GenericVars::empty_unchecked();

        // arg_map keys are target param names, and values are their values.
        for (name, v) in &self.param2arg {
            let Some((ast_name, ast_string)) = v else {
                // Not inferable from the argument types — e.g. a method const
                // generic supplied only via turbofish (`load_pipelined::<4>`).
                // Leave it unresolved; callers merge explicit turbofish args
                // afterward, and a genuinely missing param still fails with a
                // spanned "no resolved value" error at its use site.
                continue;
            };
            match *ast_name {
                GenericArgType::Type => {
                    // Check if ast_string is a variable.
                    if let Some(inst_i32_val) = from_generic_args.inst_i32.get(ast_string) {
                        // This is a const generic i32
                        to_generic_vars.inst_i32.insert(name.clone(), *inst_i32_val);
                    } else if let Some(inst_bool_val) = from_generic_args.inst_bool.get(ast_string)
                    {
                        // This is a const generic bool
                        to_generic_vars
                            .inst_bool
                            .insert(name.clone(), *inst_bool_val);
                    } else if let Some(inst_arr_val) = from_generic_args.inst_array.get(ast_string)
                    {
                        // This is a const generic array
                        to_generic_vars
                            .inst_array
                            .insert(name.clone(), inst_arr_val.clone());
                        if let Some(generic_cga) = self.param2cga.get(name) {
                            let Type::Array(ty_arr) = generic_cga else {
                                panic!("Expected array type.")
                            };
                            if let Expr::Path(length_expr) = &ty_arr.len {
                                let length_var = length_expr
                                    .path
                                    .get_ident()
                                    .unwrap()
                                    .to_string()
                                    .to_string();
                                to_generic_vars.len2array.insert(length_var, name.clone());
                            }
                        }
                    // Check if it's a type parameter.
                    } else if let Some(inst_type_val) = from_generic_args.inst_types.get(ast_string)
                    {
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), inst_type_val.to_string());
                    // Check if it's a ptr of generic type param.
                    } else if let Some((is_mutable, element_type)) =
                        get_ptr_type_instance(ast_string, from_generic_args, primitives)
                    {
                        let instantiated_ptr = if is_mutable {
                            format!("* mut {element_type}")
                        } else {
                            format!("* const {element_type}")
                        };
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), instantiated_ptr);
                    // Check if it's an element type.
                    } else if is_element_type(ast_string.as_str(), primitives) {
                        // This is a concrete element type.
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), ast_string.to_string());
                    } else if is_element_type_ptr(ast_string.as_str(), primitives) {
                        // This is a ptr with concrete element type.
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), ast_string.to_string());
                    } else {
                        // ZST marker types (for example `ordering::Acquire`
                        // and `scope::Device`) are ordinary concrete type
                        // arguments. They are not ElementType instances, but
                        // rustc has already accepted them against the method
                        // bounds, and the JIT only needs to preserve the
                        // selected type while inferring the return type.
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), ast_string.to_string());
                    }
                }
                GenericArgType::GenericConstExpr => {
                    let generic_arg = match syn::parse_str::<GenericArgument>(ast_string) {
                        Ok(generic_arg) => generic_arg,
                        Err(_) => {
                            let expr = syn::parse_str::<Expr>(ast_string).unwrap_or_else(|err| {
                                panic!(
                                    "failed to parse inferred const generic `{ast_string}` for `{name}` as a generic argument or expression: {err}"
                                )
                            });
                            GenericArgument::Const(expr)
                        }
                    };
                    if let Some(res) = try_get_bool_const_generic_from_generic_argument(
                        &generic_arg,
                        from_generic_args,
                    ) {
                        to_generic_vars.inst_bool.insert(name.clone(), res);
                    } else if let Some(res) =
                        try_get_const_generic_from_generic_argument(&generic_arg, from_generic_args)
                    {
                        // These are args -> param pairs like:
                        // {[..., 128, ...]} -> {[..., CONST_PARAM, ...]}
                        // {[..., CONST_ARG, ...]} -> {[..., CONST_PARAM, ...]}
                        to_generic_vars.inst_i32.insert(name.clone(), res);
                    } else {
                        let Some(res) =
                            get_cga_from_generic_argument(&generic_arg, from_generic_args)
                        else {
                            unimplemented!("Unexpected param2arg pair ({name}, {v:?}).")
                        };
                        // These are args -> param pairs like:
                        // {[...]} -> CONST_ARRAY_PARAM
                        to_generic_vars.inst_array.insert(name.clone(), res);
                        if let Some(generic_cga) = self.param2cga.get(name) {
                            let Type::Array(ty_arr) = generic_cga else {
                                panic!("Expected array type.")
                            };
                            if let Expr::Path(length_expr) = &ty_arr.len {
                                let length_var = length_expr
                                    .path
                                    .get_ident()
                                    .unwrap()
                                    .to_string()
                                    .to_string();
                                to_generic_vars.len2array.insert(length_var, name.clone());
                            }
                        }
                    }
                }
            }
        }
        to_generic_vars
    }

    pub(crate) fn add_type_constraints(
        &mut self,
        type_param: &syn::Type,
        type_arg: &syn::Type,
    ) -> Result<(), JITError> {
        self.add_generic_args(type_param, type_arg)
    }

    /// Records `arg_val` as the inferred value of the shape parameter
    /// `param_var`. Symbolic elements are normalized through the caller's
    /// generics: argument types mix entry-substituted literals with caller
    /// symbols for the same dimension (`128` vs `D`), and both must unify. A
    /// dynamic `-1` carries no information about the parameter — it never
    /// binds and never overrides — while two different concrete values are a
    /// genuine conflict.
    fn bind_shape_param(&mut self, param_var: &str, arg_val: String) -> Result<(), JITError> {
        let normalize = |scalars: &HashMap<String, i32>, value: String| match scalars.get(&value) {
            Some(instance) => instance.to_string(),
            None => value,
        };
        let arg_val = normalize(&self.caller_scalars, arg_val);
        if arg_val == "- 1" || arg_val == "-1" {
            return Ok(());
        }
        let replaced = self.param2arg.insert(
            param_var.to_string(),
            Some((GenericArgType::GenericConstExpr, arg_val.clone())),
        );
        if let Some(Some((_, previous))) = replaced {
            let previous = normalize(&self.caller_scalars, previous);
            if previous != "- 1" && previous != "-1" && previous != arg_val {
                return Err(JITError::Generic(format!(
                    "conflicting const-generic inference for `{param_var}` in call to `{callee}`: one argument implies `{previous}`, another implies `{arg_val}`",
                    callee = self.sig.ident,
                )));
            }
        }
        Ok(())
    }

    /// The value of a caller-side `PARAM[i]` shape element: `PARAM` must be
    /// one of the caller's const generic arrays and `i` an integer literal in
    /// range. `None` for any other index expression.
    fn caller_array_element(&self, index: &syn::ExprIndex) -> Option<i32> {
        let Expr::Path(base) = index.expr.as_ref() else {
            return None;
        };
        let array = self
            .caller_arrays
            .get(&base.path.get_ident()?.to_string())?;
        let Expr::Lit(lit) = index.index.as_ref() else {
            return None;
        };
        let Lit::Int(position) = &lit.lit else {
            return None;
        };
        array.get(position.base10_parse::<usize>().ok()?).copied()
    }

    fn add_generic_args(
        &mut self,
        type_param: &syn::Type,
        type_arg: &syn::Type,
    ) -> Result<(), JITError> {
        // Adds generic arguments to arg_map.
        // arg_map maps generic parameters (present in arg_map upon initialization) to various GenericArgument patterns (see below).
        // Each key in arg_map specifies the set of generic parameters in a function / method signature.
        // add_generic_args is called by map_args_to_params, which enumerates over all input types in a function / method signature.
        // The inferred ast fragment for each key (generic parameter) must be identical to an already inferred generic parameter.
        // The values mapped to by arg_map are ast fragments, which may be one of "Type", "GenericArgument", or "Expr".
        // When an occurrence of a generic parameter is found in type_param, the corresponding pattern is recorded in arg_map
        // as a pair of strings "(ast_fragment, string_repr)", where ast fragment is one of "Type", "GenericArgument", or "Expr",
        // and string_repr is a string-based representation of the AST fragment corresponding to the pattern.

        // The procedure succeeds if all keys in the resulting arg_map are populated by non-None values.
        // arg_map can then  be used by the infer_type function to infer the type or const vars
        // occurring in the function signature.

        // Once generic arguments have been computed, this method can also be used to infer generic arguments to a function call.

        if let (syn::Type::Tuple(param_tuple), syn::Type::Tuple(arg_tuple)) = (type_param, type_arg)
        {
            if param_tuple.elems.len() != arg_tuple.elems.len() {
                return Ok(());
            }
            for (param_elem, arg_elem) in param_tuple.elems.iter().zip(arg_tuple.elems.iter()) {
                self.add_generic_args(param_elem, arg_elem)?;
            }
            return Ok(());
        }

        let (Some(mut param_generic_args), Some(mut arg_generic_args)) =
            (maybe_generic_args(type_param), maybe_generic_args(type_arg))
        else {
            // Check if type itself is a generic param.
            self.add_generic_type(type_param, type_arg);
            return Ok(());
        };
        strip_generic_args_lifetimes(&mut param_generic_args);
        strip_generic_args_lifetimes(&mut arg_generic_args);

        // println!("remap ident: {param_ident:#?}, {arg_ident:#?}");
        // println!("remap: {:?} to {:?}", arg_generic_args.to_token_stream().to_string(), param_generic_args.to_token_stream().to_string());

        // Make sure there are the same number of generic arguments.
        assert_eq!(
            arg_generic_args.args.len(),
            param_generic_args.args.len(),
            "{arg_generic_args:#?}\n!=\n{param_generic_args:#?}"
        );

        for i in 0..arg_generic_args.args.len() {
            let arg_arg = &arg_generic_args.args[i];
            let param_arg = &param_generic_args.args[i];
            // Supports:
            // E -> f32
            // *mut E -> *mut f32
            // {[..., CONST_PARAM, ...]} -> {[..., 128, ...]}
            // {[..., CONST_PARAM, ...]} -> {[..., CONST_ARG, ...]}
            // CONST_ARRAY_PARAM -> CONST_ARRAY_ARG
            // CONST_ARRAY_PARAM -> {[...]}
            // CONST_ARRAY_PARAM -> {[-1; 2]}
            // TODO (HME): Unclear if we need any of this.
            // CONST_PARAM -> CONST_ARG
            // CONST_PARAM -> 128
            match (arg_arg, param_arg) {
                (GenericArgument::Type(arg_type), GenericArgument::Type(param_type)) => {
                    self.add_generic_type(param_type, arg_type);
                    match (arg_type, param_type) {
                        (syn::Type::Path(_arg_type_path), syn::Type::Path(param_type_path)) => {
                            if maybe_generic_args(param_type).is_some()
                                && maybe_generic_args(arg_type).is_some()
                            {
                                self.add_generic_args(param_type, arg_type)?;
                            }
                            // Something like (Tensor<f32, ...>, Tensor<E, ...>)
                            let param_ident = &param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            let arg_type_str = arg_type.to_token_stream().to_string();
                            if self.param2arg.contains_key(param_ident) {
                                let replaced_arg = self.param2arg.insert(
                                    param_ident.to_string(),
                                    Some((GenericArgType::Type, arg_type_str.to_string())),
                                );
                                if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                                    assert_eq!(arg, arg_type_str.to_string());
                                }
                            }
                        }
                        (syn::Type::Ptr(arg_type_ptr), syn::Type::Ptr(param_type_ptr)) => {
                            // Something like (PointerTile<*mut f32, ...>, PointerTile<*mut E, ...>)
                            let param_elem_ty = match get_type_ident(&param_type_ptr.elem) {
                                Some(ident) => ident.to_string(),
                                None => panic!(
                                    "Unable to extract ident from pointer {param_type_ptr:#?}"
                                ),
                            };
                            let arg_type_str = arg_type_ptr.elem.to_token_stream().to_string();
                            if self.param2arg.contains_key(&param_elem_ty) {
                                let replaced_arg = self.param2arg.insert(
                                    param_elem_ty.to_string(),
                                    Some((GenericArgType::Type, arg_type_str.to_string())),
                                );
                                if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                                    assert_eq!(arg, arg_type_str.to_string());
                                }
                            }
                        }
                        (syn::Type::Reference(arg_ref), syn::Type::Reference(param_ref)) => {
                            self.add_generic_args(&param_ref.elem, &arg_ref.elem)?;
                        }
                        _ => {}
                    }
                }
                (GenericArgument::Const(arg_const), GenericArgument::Type(param_type)) => {
                    match param_type {
                        syn::Type::Path(param_type_path) => {
                            // Something like (Tensor<E, {[...]}>, Tensor<E, CONST_ARRAY_PARAM>)
                            let param_ident = &param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            if self.param2arg.contains_key(param_ident) {
                                let arg_const_str = &arg_const.to_token_stream().to_string();
                                let _replaced_arg = self.param2arg.insert(
                                    param_ident.to_string(),
                                    Some((
                                        GenericArgType::GenericConstExpr,
                                        arg_const_str.to_string(),
                                    )),
                                );
                                // TODO (hme): Confirm this was too strict.
                                // if let Some(Some((_arg_type, arg))) = _replaced_arg {
                                //     assert_eq!(arg, arg_const_str.to_string());
                                // }
                            }
                        }
                        _ => panic!("Unexpected generics {param_type:#?} {arg_const:#?}"),
                    }
                }
                // `Tile<E, S>` against `Tile<E, {[M, N]}>`: the caller passes its
                // whole const generic array where the callee takes one parameter
                // per dimension. Each `M`, `N` binds to the caller's instance of
                // that element (the projected spelling `{[S[0], S[1]]}` takes the
                // array/array path below). Anything else here is not a shape
                // relation and binds nothing, as before.
                (
                    GenericArgument::Type(arg_type),
                    GenericArgument::Const(Expr::Block(param_block)),
                ) => {
                    let caller_array = get_type_ident(arg_type)
                        .and_then(|ident| self.caller_arrays.get(&ident.to_string()).cloned());
                    let param_elems = match param_block.block.stmts.first() {
                        Some(Stmt::Expr(Expr::Array(param_array), _)) => Some(&param_array.elems),
                        _ => None,
                    };
                    if let (Some(caller_array), Some(param_elems)) = (caller_array, param_elems) {
                        if caller_array.len() != param_elems.len() {
                            return Err(JITError::Generic(format!(
                                "rank mismatch in call to `{}`: the argument's shape has {} dimensions but the parameter's has {}",
                                self.sig.ident,
                                caller_array.len(),
                                param_elems.len()
                            )));
                        }
                        for (param_elem, value) in param_elems.iter().zip(caller_array) {
                            let param_var = param_elem.to_token_stream().to_string();
                            if self.param2arg.contains_key(&param_var) {
                                self.bind_shape_param(&param_var, value.to_string())?;
                            }
                        }
                    }
                }
                (GenericArgument::Const(arg_const), GenericArgument::Const(param_const)) => {
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match (arg_const, param_const) {
                        (Expr::Block(arg_expr), Expr::Block(param_expr)) => {
                            assert_eq!(arg_expr.block.stmts.len(), 1);
                            let Stmt::Expr(arg_stmt_expr, _) = &arg_expr.block.stmts[0] else {
                                panic!("Unexpected block expression.")
                            };
                            let Stmt::Expr(param_stmt_expr, _) = &param_expr.block.stmts[0] else {
                                panic!("Unexpected block expression.")
                            };
                            match (arg_stmt_expr, param_stmt_expr) {
                                (Expr::Array(arg_array_expr), Expr::Array(param_array_expr)) => {
                                    // Something like (Tensor<f32, {[...]}>, Tensor<E, {[...]}>)
                                    for i in 0..arg_array_expr.elems.iter().len() {
                                        let param_elem = &param_array_expr.elems[i];
                                        let param_var = param_elem.to_token_stream().to_string();
                                        if self.param2arg.contains_key(&param_var) {
                                            let arg_elem = &arg_array_expr.elems[i];
                                            let unsupported_element = |what: &str| {
                                                JITError::Generic(format!(
                                                    "cannot infer the generics of `{}` from the shape element `{}` ({what}); \
                                                     shape elements must be integer literals, `-1`, const generic parameters, \
                                                     or `PARAM[i]` projections of a const generic array",
                                                    self.sig.ident,
                                                    arg_elem.to_token_stream(),
                                                ))
                                            };
                                            let arg_val = match arg_elem {
                                                Expr::Lit(lit) => match &lit.lit {
                                                    Lit::Int(_) => arg_elem.to_token_stream().to_string(),
                                                    _ => return Err(unsupported_element("a non-integer literal")),
                                                },
                                                Expr::Unary(_) | Expr::Path(_) => {
                                                    arg_elem.to_token_stream().to_string()
                                                }
                                                // `S[0]`: the caller projects its const generic
                                                // array onto a per-dimension callee parameter.
                                                // Resolve the element now, so it unifies like a
                                                // literal. This was an `unimplemented!` panic.
                                                Expr::Index(index) => match self.caller_array_element(index) {
                                                    Some(value) => value.to_string(),
                                                    None => {
                                                        return Err(unsupported_element(
                                                            "an index that is not `PARAM[i]` over a caller const generic array with a literal index",
                                                        ))
                                                    }
                                                },
                                                _ => return Err(unsupported_element("an unsupported expression")),
                                            };
                                            self.bind_shape_param(&param_var, arg_val)?;
                                        }
                                    }
                                },
                                (_, Expr::Repeat(_param_expr)) => {
                                    // TODO (hme): Check that this is okay.
                                    // If param is comprised of variadic literals, then skip it.
                                }
                                _ => panic!("Unexpected block expression:\nparam=\n{param_stmt_expr:#?}\narg=\n{arg_stmt_expr:#?}")
                            }
                        }
                        (Expr::Lit(arg_lit), Expr::Lit(param_lit)) => {
                            let arg_lit_str = arg_lit.to_token_stream().to_string();
                            let param_lit_str = param_lit.to_token_stream().to_string();
                            if arg_lit_str != param_lit_str {
                                return Err(JITError::Generic(format!(
                                    "const-generic mismatch in call to `{callee}`: the signature fixes `{param_lit_str}` but the argument provides `{arg_lit_str}`",
                                    callee = self.sig.ident,
                                )));
                            }
                        }
                        _ => unimplemented!(
                            "Unsupported Const inference {param_const:#?} {arg_const:#?}"
                        ),
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn add_generic_type(&mut self, param_type: &Type, arg_type: &Type) {
        let arg_map = &mut self.param2arg;
        match (arg_type, param_type) {
            (syn::Type::Path(_arg_type_path), syn::Type::Path(param_type_path)) => {
                // Something like (Tensor<f32, ...>, Tensor<E, ...>)
                let param_ident = &param_type_path
                    .path
                    .segments
                    .last()
                    .unwrap()
                    .ident
                    .to_string();
                let arg_type_str = arg_type.to_token_stream().to_string();
                if arg_map.contains_key(param_ident) {
                    let _replaced_arg = arg_map.insert(
                        param_ident.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    // TODO (hme): Check that this is okay.
                    // if let Some(Some((_generic_arg_type, arg))) = _replaced_arg {
                    //     assert_eq!(arg, arg_type_str.to_string(), "param_type={param_type:#?},\narg_type={arg_type:#?},\narg_map={arg_map:#?}");
                    // }
                }
            }
            (syn::Type::Ptr(_arg_type_path), syn::Type::Path(param_type_path)) => {
                // Something like (PointerTile<*mut f32, ...>, PointerTile<P, ...>)
                let param_ident = &param_type_path
                    .path
                    .segments
                    .last()
                    .unwrap()
                    .ident
                    .to_string();
                let arg_type_str = arg_type.to_token_stream().to_string();
                if arg_map.contains_key(param_ident) {
                    let replaced_arg = arg_map.insert(
                        param_ident.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                        // A shared pointer-typed generic param can bind to both
                        // `*const T` and `*mut T`: rustc coerces at the call site
                        // and Tile IR erases constness, so tolerate a const/mut-only
                        // difference — only a different pointee is a real mismatch.
                        debug_assert_eq!(
                            arg.replace("* const ", "* mut "),
                            arg_type_str.replace("* const ", "* mut ")
                        );
                    }
                }
            }
            (syn::Type::Ptr(arg_type_ptr), syn::Type::Ptr(param_type_ptr)) => {
                // Something like (PointerTile<*mut f32, ...>, PointerTile<*mut E, ...>)
                let param_elem_ty = match get_type_ident(&param_type_ptr.elem) {
                    Some(ident) => ident.to_string(),
                    None => panic!("Unable to extract ident from pointer {param_type_ptr:#?}"),
                };
                let arg_type_str = arg_type_ptr.elem.to_token_stream().to_string();
                if arg_map.contains_key(&param_elem_ty) {
                    let replaced_arg = arg_map.insert(
                        param_elem_ty.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                        // A shared pointer-typed generic param can bind to both
                        // `*const T` and `*mut T`: rustc coerces at the call site
                        // and Tile IR erases constness, so tolerate a const/mut-only
                        // difference — only a different pointee is a real mismatch.
                        debug_assert_eq!(
                            arg.replace("* const ", "* mut "),
                            arg_type_str.replace("* const ", "* mut ")
                        );
                    }
                }
            }
            (syn::Type::Reference(arg_ref), syn::Type::Reference(param_ref)) => {
                self.add_generic_type(&param_ref.elem, &arg_ref.elem);
            }
            (syn::Type::Reference(arg_ref), _) => {
                self.add_generic_type(param_type, &arg_ref.elem);
            }
            (_, syn::Type::Reference(param_ref)) => {
                self.add_generic_type(&param_ref.elem, arg_type);
            }
            _ => {}
        }
    }

    pub fn infer_type(&self, ty: &syn::Type, _generic_vars: &GenericVars) -> syn::Type {
        let arg_map = &self.param2arg;
        // println!("Infer generic args for {} using \n {arg_map:#?}", ty.to_token_stream().to_string());
        let Some(mut result_args) = maybe_generic_args(ty) else {
            // Is it a generic arg itself?
            // TODO (hme): *Really* need to make this recursive and just call with the following types.
            let mut result = ty.clone();
            match &mut result {
                syn::Type::Path(param_type_path) => {
                    // This is a type var or concrete type.
                    let param_ident_str = param_type_path
                        .path
                        .segments
                        .last()
                        .unwrap()
                        .ident
                        .to_string();
                    match arg_map.get(param_ident_str.as_str()) {
                        None => {
                            // This is not a generic parameter.
                        }
                        Some(None) => {
                            panic!("Failed to infer generic parameter {param_ident_str} \n{arg_map:#?}")
                        }
                        Some(Some((GenericArgType::Type, target_ty))) => {
                            result = syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                        }
                        Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                            result = syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                        }
                    }
                }
                syn::Type::Ptr(param_type_ptr) => {
                    // This is a pointer with type var or concrete type for element type.
                    match *param_type_ptr.elem.clone() {
                        Type::Path(type_path) => {
                            let param_ident =
                                type_path.path.segments.last().unwrap().ident.to_string();
                            match arg_map.get(param_ident.as_str()) {
                                None => {
                                    // This is not a generic parameter.
                                }
                                Some(None) => {
                                    panic!("Failed to infer generic parameter {param_ident}")
                                }
                                Some(Some((GenericArgType::Type, target_ty))) => {
                                    *param_type_ptr.elem =
                                        syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                                }
                                Some(Some((arg_type, _arg))) => {
                                    panic!("Unexpected arg type {arg_type:#?}")
                                }
                            };
                        }
                        _ => panic!("Unable to extract ident from pointer {param_type_ptr:#?}"),
                    }
                }
                syn::Type::Array(array_ty) => {
                    // Something like [T; N]
                    let syn::Type::Path(elem) = &mut *array_ty.elem else {
                        panic!("Unexpected element type for array {array_ty:#?}")
                    };
                    // This is a type var or concrete type.
                    let elem_ident_str = elem.path.segments.last().unwrap().ident.to_string();
                    match arg_map.get(elem_ident_str.as_str()) {
                        None => {} // This is not a generic parameter.
                        Some(None) => panic!(
                            "Failed to infer generic parameter {elem_ident_str} \n{arg_map:#?}"
                        ),
                        Some(Some((GenericArgType::Type, target_ty))) => {
                            *elem = syn::parse2::<TypePath>(target_ty.parse().unwrap()).unwrap()
                        }
                        Some(Some((GenericArgType::GenericConstExpr, _target_ty))) => {
                            panic!("Unexpected element type for array {array_ty:#?}")
                        }
                    }
                    if let Expr::Path(len_path_expr) = array_ty.len.clone() {
                        // // This is a type var or concrete type.
                        let len_ident_str = len_path_expr
                            .path
                            .segments
                            .last()
                            .unwrap()
                            .ident
                            .to_string();
                        match arg_map.get(len_ident_str.as_str()) {
                            None => {} // This is not a generic parameter.
                            Some(None) => panic!(
                                "Failed to infer generic parameter {len_ident_str} \n{arg_map:#?}"
                            ),
                            Some(Some((GenericArgType::Type, _target_ty))) => {
                                panic!("Unexpected length type for array {array_ty:#?}")
                            }
                            Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                                array_ty.len =
                                    syn::parse2::<Expr>(target_ty.parse().unwrap()).unwrap()
                            }
                        }
                    } else {
                        // Nothing to do.
                    }
                }
                _ => {}
            }
            return result;
        };

        // for arg in param_generic_args.args.iter_mut() {
        //     println!("Infer generic args {:?}", arg);
        // }
        for arg in result_args.args.iter_mut() {
            match arg {
                GenericArgument::Type(param_type) => {
                    match param_type {
                        syn::Type::Path(param_type_path) => {
                            // This is a type var or concrete type.
                            let param_ident_str = param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            match arg_map.get(param_ident_str.as_str()) {
                                None => {
                                    // This is not a generic parameter.
                                }
                                Some(None) => {
                                    panic!("Failed to infer generic parameter {param_ident_str} \n{arg_map:#?}")
                                }
                                Some(Some((GenericArgType::Type, target_ty))) => {
                                    *arg =
                                        syn::parse2::<GenericArgument>(target_ty.parse().unwrap())
                                            .unwrap();
                                }
                                Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                                    let target_expr =
                                        syn::parse2::<Expr>(target_ty.parse().unwrap()).unwrap();
                                    *arg = GenericArgument::Const(target_expr);
                                }
                            }
                        }
                        syn::Type::Ptr(param_type_ptr) => {
                            // This is a pointer with type var or concrete type for element type.
                            match *param_type_ptr.elem.clone() {
                                Type::Path(type_path) => {
                                    let param_ident =
                                        type_path.path.segments.last().unwrap().ident.to_string();
                                    match arg_map.get(param_ident.as_str()) {
                                        None => {
                                            // This is not a generic parameter.
                                        }
                                        Some(None) => {
                                            panic!(
                                                "Failed to infer generic parameter {param_ident}"
                                            )
                                        }
                                        Some(Some((GenericArgType::Type, target_ty))) => {
                                            *param_type_ptr.elem =
                                                syn::parse2::<Type>(target_ty.parse().unwrap())
                                                    .unwrap();
                                        }
                                        Some(Some((arg_type, _arg))) => {
                                            panic!("Unexpected arg type {arg_type:#?}")
                                        }
                                    };
                                }
                                _ => panic!(
                                    "Unable to extract ident from pointer {param_type_ptr:#?}"
                                ),
                            }
                        }
                        syn::Type::Reference(_param_ref) => {}
                        _ => {}
                    }
                }
                GenericArgument::Const(param_const) => {
                    // This is a literal or array expression.
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match param_const {
                        Expr::Block(param_expr) => {
                            assert_eq!(param_expr.block.stmts.len(), 1);
                            let Stmt::Expr(param_stmt_expr, _) = &mut param_expr.block.stmts[0]
                            else {
                                panic!("Unexpected block expression.")
                            };
                            if let Expr::Array(param_array_expr) = param_stmt_expr {
                                for i in 0..param_array_expr.elems.iter().len() {
                                    let param_elem = &mut param_array_expr.elems[i];
                                    let param_var = param_elem.to_token_stream().to_string();
                                    match arg_map.get(param_var.as_str()) {
                                        None => {
                                            // This is not a generic parameter.
                                        }
                                        Some(None) => {
                                            panic!("Failed to infer generic parameter {param_var}")
                                        }
                                        Some(Some((
                                            GenericArgType::GenericConstExpr,
                                            target_expr,
                                        ))) => {
                                            *param_elem =
                                                syn::parse2::<Expr>(target_expr.parse().unwrap())
                                                    .unwrap();
                                        }
                                        Some(Some((arg_type, _arg))) => {
                                            panic!("Unexpected arg type {arg_type:#?}")
                                        }
                                    }
                                }
                            }
                        }
                        Expr::Path(param_path) => {
                            unimplemented!("Expr::Path not supported {:#?}", param_path);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        let mut return_ty = ty.clone();
        match &mut return_ty {
            Type::Path(type_path) => {
                let last_seg = type_path.path.segments.last_mut().unwrap();
                last_seg.arguments = PathArguments::AngleBracketed(result_args);
            }
            Type::Reference(ref_type) => match &mut *ref_type.elem {
                Type::Path(type_path) => {
                    let last_seg = type_path.path.segments.last_mut().unwrap();
                    last_seg.arguments = PathArguments::AngleBracketed(result_args);
                }
                _ => panic!("Unexpected ref type {:#?}", ref_type),
            },
            _ => panic!("get_ident_generic_args: Unexpected type {:#?}", return_ty),
        }
        return_ty
    }
}

pub fn get_cga_from_type(ty: &syn::Type, generic_args: &GenericVars) -> Option<Vec<i32>> {
    // We assume this is a variadic type.
    let (_type_ident, type_generic_args) = get_ident_generic_args(ty);
    let mut shape: Option<Vec<i32>> = None;
    for type_generic_arg in &type_generic_args.args {
        let res = get_cga_from_generic_argument(type_generic_arg, generic_args);
        if let Some(value) = res {
            shape = Some(value);
        }
    }
    shape
}

/// Attempts to extract a const generic expression from a single generic argument.
pub fn try_get_const_generic_from_generic_argument(
    generic_arg: &GenericArgument,
    generic_args: &GenericVars,
) -> Option<i32> {
    let mut result: Option<i32> = None;
    match generic_arg {
        GenericArgument::Type(syn::Type::Path(type_path)) => {
            let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
            // println!("get_variadic_type_args: Type::Path: {}", last_ident);
            if generic_args.inst_i32.contains_key(&last_ident) {
                // This is something like N for const generic N: i32.
                result = Some(*generic_args.inst_i32.get(&last_ident).unwrap());
            }
            // If it's anything else, then return None.
        }
        GenericArgument::Const(Expr::Lit(lit)) => {
            // println!("expand GenericArgument::Const? {const_param:#?}");
            let Lit::Int(int_lit) = &lit.lit else {
                panic!("Expected int literal, got {:#?}", lit)
            };
            // This is something like 32 in Tile<E, {[32]}>
            // TODO (hme): Add a test for this.
            result = Some(int_lit.base10_parse().unwrap());
        }
        _ => {}
    }
    result
}

/// Attempts to extract a bool const generic expression from a single generic argument.
pub fn try_get_bool_const_generic_from_generic_argument(
    generic_arg: &GenericArgument,
    generic_args: &GenericVars,
) -> Option<bool> {
    match generic_arg {
        GenericArgument::Type(Type::Path(type_path)) => {
            let last_ident = type_path.path.segments.last()?.ident.to_string();
            generic_args.inst_bool.get(&last_ident).copied()
        }
        GenericArgument::Const(Expr::Path(path)) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_args.inst_bool.get(&ident).copied()
        }
        GenericArgument::Const(Expr::Lit(lit)) => match &lit.lit {
            Lit::Bool(bool_lit) => Some(bool_lit.value),
            _ => None,
        },
        _ => None,
    }
}

/// Extracts a const generic array value from a generic argument, resolving variables.
pub fn get_cga_from_generic_argument(
    generic_arg: &GenericArgument,
    generic_args: &GenericVars,
) -> Option<Vec<i32>> {
    let mut shape: Option<Vec<i32>> = None;
    match generic_arg {
        GenericArgument::Type(type_param) => {
            if let syn::Type::Path(type_path) = type_param {
                // This must be a CGA, or it will fail.
                let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                if generic_args.inst_array.contains_key(&last_ident) {
                    // This is something like Shape<D> for const generic array D: [i32; N].
                    let array_instance = generic_args.inst_array.get(&last_ident).unwrap();
                    if shape.is_some() {
                        panic!("Unexpected array arg: {last_ident:#?}")
                    }
                    shape = Some(array_instance.clone());
                } else if generic_args.inst_i32.contains_key(&last_ident) {
                    // This is something like N for const generic N: i32.
                    // This should have been handled by
                    // try_get_const_generic_from_generic_argument.
                    unimplemented!("Unexpected const arg {last_ident} for type {type_param:#?}");
                } else {
                    unimplemented!("Failed to get cga for {type_param:#?}");
                }
            }
        }
        GenericArgument::Const(Expr::Block(block_expr)) => {
            // println!("expand GenericArgument::Const? {const_param:#?}");
            // This is something like Tensor<E, {[...]}>
            assert_eq!(block_expr.block.stmts.len(), 1);
            let statement = &block_expr.block.stmts[0];
            let Stmt::Expr(statement_expr, _) = statement else {
                panic!("Unexpected block expression.")
            };
            match statement_expr {
                Expr::Array(array_expr) => {
                    // This is something like Tensor<E, {[1, 2, -1]}>
                    let mut _shape: Vec<i32> = vec![];
                    for elem in &array_expr.elems {
                        _shape.push(parse_expr_as_i32(elem, generic_args));
                    }
                    shape = Some(_shape);
                }
                Expr::Repeat(repeat_expr) => {
                    // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                    let thing_to_repeat = parse_expr_as_i32(&repeat_expr.expr, generic_args);
                    match &*repeat_expr.len {
                        Expr::Path(len_path) => {
                            // This is something like Tensor<E, {[-1; N]}>
                            let num_rep_var = len_path.to_token_stream().to_string();
                            if generic_args.get_i32(&num_rep_var).is_none() {
                                panic!("Expected instance for generic argument {}", num_rep_var);
                            }
                            let num_rep = generic_args.get_i32(&num_rep_var).unwrap();
                            shape = Some(vec![thing_to_repeat; num_rep as usize]);
                        }
                        Expr::Lit(len_lit) => {
                            // This is something like Tensor<E, {[-1; 3]}>
                            let num_rep: u32 = len_lit
                                .to_token_stream()
                                .to_string()
                                .parse::<u32>()
                                .unwrap();
                            shape = Some(vec![thing_to_repeat; num_rep as usize]);
                        }
                        _ => {
                            unimplemented!("Unexpected repeat expression: {repeat_expr:#?}")
                        }
                    }
                }
                _ => panic!("Unexpected block expression."),
            }
        }
        _ => {}
    }
    shape
}

pub fn parse_expr_as_i32(expr: &Expr, generic_args: &GenericVars) -> i32 {
    match expr {
        Expr::Lit(_lit) => parse_signed_literal_as_i32(expr),
        Expr::Unary(_unary_expr) => parse_signed_literal_as_i32(expr),
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path);
            match generic_args.inst_i32.get(ident.to_string().as_str()) {
                Some(val) => *val,
                None => panic!("Undefined generic parameter {ident}"),
            }
        }
        Expr::Index(index) => {
            let Expr::Path(path) = index.expr.as_ref() else {
                unimplemented!("Unexpected const generic array base {expr:#?}");
            };
            let ident = get_ident_from_path_expr(path);
            let Some(shape) = generic_args.inst_array.get(ident.to_string().as_str()) else {
                panic!("Undefined const generic array parameter {ident}");
            };
            let i = parse_signed_literal_as_i32(&index.index);
            let Some(dim) = shape.get(i as usize) else {
                panic!(
                    "Index {i} out of bounds for const generic array `{ident}` of length {}",
                    shape.len()
                );
            };
            *dim
        }
        _ => unimplemented!("Unexpected expression {expr:#?}"),
    }
}

#[cfg(test)]
mod inference_tests {
    use super::*;

    fn mma_like_inference() -> GenericArgInference {
        let sig: Signature = syn::parse_quote! {
            fn mma_like<const M: i32, const N: i32, const K: i32>(
                lhs: Tile<f32, { [M, K] }>,
                rhs: Tile<f32, { [K, N] }>,
            ) -> Tile<f32, { [M, N] }>
        };
        GenericArgInference::new_function(sig)
    }

    /// Two arguments implying different concrete values for the same shape
    /// param is a genuine conflict: it must produce the named JIT error,
    /// never silently unify.
    #[test]
    fn conflicting_concrete_dims_error_instead_of_unifying() {
        let mut inference = mma_like_inference();
        let args: Vec<Type> = vec![
            syn::parse_quote!(Tile<f32, { [16, 128] }>),
            syn::parse_quote!(Tile<f32, { [64, 32] }>),
        ];
        let err = inference
            .map_args_to_params(&args, None, &GenericVars::empty_unchecked())
            .expect_err("K = 128 vs K = 64 must conflict");
        let message = err.to_string();
        assert!(
            message.contains("conflicting const-generic inference")
                && message.contains("`K`")
                && message.contains("mma_like")
                && message.contains("128")
                && message.contains("64"),
            "conflict error must name the param, callee, and both values: {message}"
        );
    }

    /// Normalization is evaluation under the caller's instantiation: a
    /// symbol and its instance are the same dimension and must unify.
    #[test]
    fn symbolic_and_substituted_forms_of_one_dim_unify() {
        let mut inference = mma_like_inference();
        let mut caller = GenericVars::empty_unchecked();
        caller.inst_i32.insert("D".to_string(), 128);
        let args: Vec<Type> = vec![
            syn::parse_quote!(Tile<f32, { [16, D] }>),
            syn::parse_quote!(Tile<f32, { [128, 32] }>),
        ];
        inference
            .map_args_to_params(&args, None, &caller)
            .expect("D (=128) and 128 are the same dimension");
        let vars = inference.get_generic_vars_instance(&caller, &HashMap::new());
        assert_eq!(vars.inst_i32.get("K"), Some(&128));
        assert_eq!(vars.inst_i32.get("M"), Some(&16));
        assert_eq!(vars.inst_i32.get("N"), Some(&32));
    }

    /// A caller shape written as projections of its const generic array
    /// (`S[0]`, `S[1]`) unifies with a per-dimension callee's parameters.
    /// This was an `unimplemented!` panic (2026-08 audit).
    #[test]
    fn projected_array_elements_unify_with_per_dimension_params() {
        let sig: Signature = syn::parse_quote! {
            fn scale_rows<const M: i32, const N: i32>(t: Tile<f32, { [M, N] }>) -> Tile<f32, { [M, N] }>
        };
        let mut inference = GenericArgInference::new_function(sig);
        let mut caller = GenericVars::empty_unchecked();
        caller.inst_array.insert("S".to_string(), vec![4, 8]);
        let args: Vec<Type> = vec![syn::parse_quote!(Tile<f32, { [S[0], S[1]] }>)];
        inference
            .map_args_to_params(&args, None, &caller)
            .expect("S[0] and S[1] resolve through the caller's array");
        let vars = inference.get_generic_vars_instance(&caller, &HashMap::new());
        assert_eq!(vars.inst_i32.get("M"), Some(&4));
        assert_eq!(vars.inst_i32.get("N"), Some(&8));

        // An index the caller cannot resolve is a diagnostic, not a panic.
        let mut inference = GenericArgInference::new_function(syn::parse_quote! {
            fn scale_rows<const M: i32, const N: i32>(t: Tile<f32, { [M, N] }>) -> Tile<f32, { [M, N] }>
        });
        let args: Vec<Type> = vec![syn::parse_quote!(Tile<f32, { [Q[0], 8] }>)];
        let err = inference
            .map_args_to_params(&args, None, &caller)
            .expect_err("Q is not a caller array");
        let message = err.to_string();
        assert!(
            message.contains("Q") && message.contains("scale_rows") && message.contains("PARAM[i]"),
            "unexpected error: {message}"
        );
    }

    /// The caller may also pass its whole const generic array (`Tile<f32, S>`)
    /// where the callee takes one parameter per dimension; each element binds
    /// to the caller's instance.
    #[test]
    fn whole_array_argument_binds_per_dimension_params() {
        let sig: Signature = syn::parse_quote! {
            fn scale_rows<const M: i32, const N: i32>(t: Tile<f32, { [M, N] }>) -> Tile<f32, { [M, N] }>
        };
        let mut inference = GenericArgInference::new_function(sig);
        let mut caller = GenericVars::empty_unchecked();
        caller.inst_array.insert("S".to_string(), vec![4, 8]);
        let args: Vec<Type> = vec![syn::parse_quote!(Tile<f32, S>)];
        inference
            .map_args_to_params(&args, None, &caller)
            .expect("S resolves through the caller's array");
        let vars = inference.get_generic_vars_instance(&caller, &HashMap::new());
        assert_eq!(vars.inst_i32.get("M"), Some(&4));
        assert_eq!(vars.inst_i32.get("N"), Some(&8));
    }

    /// Normalization cannot over-normalize: two different symbols with
    /// different instances still conflict after resolution.
    #[test]
    fn distinct_symbols_with_distinct_instances_still_conflict() {
        let mut inference = mma_like_inference();
        let mut caller = GenericVars::empty_unchecked();
        caller.inst_i32.insert("D".to_string(), 128);
        caller.inst_i32.insert("E".to_string(), 64);
        let args: Vec<Type> = vec![
            syn::parse_quote!(Tile<f32, { [16, D] }>),
            syn::parse_quote!(Tile<f32, { [E, 32] }>),
        ];
        let err = inference
            .map_args_to_params(&args, None, &caller)
            .expect_err("D (=128) vs E (=64) must conflict after normalization");
        assert!(
            err.to_string()
                .contains("conflicting const-generic inference"),
            "unexpected error: {err}"
        );
    }
}
