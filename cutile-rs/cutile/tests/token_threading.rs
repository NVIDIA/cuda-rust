/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Token threading through the tensor API compiles to the intended Tile IR.
//!
//! `Tensor::store` returns its completion token, `Tensor::token` reads the
//! tensor's current token, and `Tensor::set_token` installs one. These tests
//! compile kernels to Tile IR text (no GPU, no assembler) and check the SSA
//! dataflow: which token each memory op consumes. Source order is not
//! evidence of a dependency; the operand is.

use cutile_compiler::compile_api::KernelCompiler;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[cutile::module]
mod token_module {
    use cutile::core::*;

    /// Two stores through one tensor: the second must be ordered after the
    /// first by consuming the token the first store returned.
    #[cutile::entry()]
    fn store_then_store(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = x.load_like(z);
        z.store(t);
        z.store(t + t);
    }

    /// The token `store` returns is the tensor's token afterwards: both are
    /// handed to a token-consuming op, which must see the same SSA value.
    #[cutile::entry()]
    fn store_token_is_tensor_token(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = x.load_like(z);
        let stored: Token = z.store(t);
        let current: Token = z.token();
        let _a: Token = unsafe { gdc_launch_dependents_tko(Some(stored)) };
        let _b: Token = unsafe { gdc_launch_dependents_tko(Some(current)) };
    }

    /// PDL producer: the launch-dependents signal consumes the store's token.
    #[cutile::entry()]
    fn producer(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [4] }> = x.load_like(z);
        let stored: Token = z.store(t);
        let _signal: Token = unsafe { gdc_launch_dependents_tko(Some(stored)) };
    }

    /// PDL consumer: after `input.set_token(wait)`, the load of `input`
    /// consumes the wait's token.
    #[cutile::entry()]
    fn consumer(out: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        unsafe { input.set_token(gdc_wait_tko(None)) };
        let t: Tile<f32, { [4] }> = input.load_like(out);
        out.store(t);
    }

    /// Two loads through one read-only partition must not be chained: a
    /// load's completion token stays inside `Partition::load`, so both
    /// loads start from the tensor's entry token and can be issued together.
    #[cutile::entry()]
    fn two_loads_one_partition(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![4]);
        let a: Tile<f32, { [4] }> = p.load([program_id(0)]);
        let b: Tile<f32, { [4] }> = p.load([program_id(0) + 1]);
        z.store(a + b);
    }

    /// Control: a wait whose token is never installed orders nothing.
    #[cutile::entry()]
    fn consumer_without_set(out: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let _ready: Token = unsafe { gdc_wait_tko(None) };
        let t: Tile<f32, { [4] }> = input.load_like(out);
        out.store(t);
    }

    /// Explicit installation on a block-local shadow must be rejected.
    #[cutile::entry()]
    fn shadow_in_block(
        out: &mut Tensor<f32, { [4] }>,
        input: &Tensor<f32, { [-1] }>,
        other: &Tensor<f32, { [-1] }>,
    ) {
        unsafe {
            let input = other;
            input.set_token(gdc_wait_tko(None));
        }
        let t: Tile<f32, { [4] }> = input.load_like(out);
        out.store(t);
    }

    #[cutile::entry()]
    fn shadow_without_install(
        out: &mut Tensor<f32, { [4] }>,
        input: &Tensor<f32, { [-1] }>,
        other: &Tensor<f32, { [-1] }>,
    ) {
        unsafe { other.set_token(gdc_wait_tko(None)) };
        {
            let input = other;
            let _token = input.token();
        }
        let t: Tile<f32, { [4] }> = input.load_like(out);
        out.store(t);
    }

    #[cutile::entry()]
    #[allow(unused_mut)] // The binding's mutability is the regression.
    fn mutable_alias(out: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let mut alias = input;
        unsafe { alias.set_token(gdc_wait_tko(None)) };
        let t: Tile<f32, { [4] }> = alias.load_like(out);
        out.store(t);
    }

    fn helper(other: &Tensor<f32, { [-1] }>, input: &Tensor<f32, { [-1] }>) {
        unsafe { other.set_token(gdc_wait_tko(None)) };
        let _token = input.token();
    }

    #[cutile::entry()]
    fn swapped_params(
        out: &mut Tensor<f32, { [4] }>,
        input: &Tensor<f32, { [-1] }>,
        other: &Tensor<f32, { [-1] }>,
    ) {
        unsafe { other.set_token(gdc_wait_tko(None)) };
        helper(input, other);
        let x: Tile<f32, { [4] }> = input.load_like(out);
        let y: Tile<f32, { [4] }> = other.load_like(out);
        out.store(x + y);
    }

    #[cutile::entry()]
    fn if_else(input: &Tensor<f32, { [-1] }>) {
        if program_id(0) == 0 {
            unsafe { input.set_token(gdc_wait_tko(None)) };
        } else {
            unsafe { input.set_token(gdc_wait_tko(None)) };
        }
    }

    #[cutile::entry()]
    fn else_only(input: &Tensor<f32, { [-1] }>) {
        if program_id(0) == 0 {
            let _token = input.token();
        } else {
            unsafe { set_tensor_token(input, gdc_wait_tko(None)) };
        }
    }

    #[cutile::entry()]
    fn constant_if(input: &Tensor<f32, { [-1] }>) {
        if true {
            unsafe { input.set_token(gdc_wait_tko(None)) };
        }
    }

    #[cutile::entry()]
    fn for_loop(input: &Tensor<f32, { [-1] }>) {
        for _i in 0..2 {
            unsafe { input.set_token(gdc_wait_tko(None)) };
        }
    }

    #[cutile::entry()]
    fn while_loop(input: &Tensor<f32, { [-1] }>) {
        let mut i = 0;
        while i < 2 {
            unsafe { set_tensor_token(input, gdc_wait_tko(None)) };
            i = i + 1;
        }
    }

    #[cutile::entry()]
    fn loop_break(input: &Tensor<f32, { [-1] }>) {
        loop {
            unsafe { input.set_token(gdc_wait_tko(None)) };
            break;
        }
    }

    #[cutile::entry()]
    fn short_circuit(input: &Tensor<f32, { [-1] }>) {
        let _condition = program_id(0) == 0 && {
            unsafe { input.set_token(gdc_wait_tko(None)) };
            true
        };
    }

    #[cutile::entry()]
    fn constant_short_circuit(input: &Tensor<f32, { [-1] }>) {
        let _condition = true && {
            unsafe { input.set_token(gdc_wait_tko(None)) };
            true
        };
    }

    fn wait_in_helper(input: &Tensor<f32, { [-1] }>) {
        unsafe { input.set_token(gdc_wait_tko(None)) };
    }

    #[cutile::entry()]
    fn helper_in_if(input: &Tensor<f32, { [-1] }>) {
        if program_id(0) == 0 {
            wait_in_helper(input);
        }
    }

    #[cutile::entry()]
    fn helper_in_loop(input: &Tensor<f32, { [-1] }>) {
        for _i in 0..2 {
            wait_in_helper(input);
        }
    }

    #[cutile::entry()]
    fn block_local_direct(input: &Tensor<f32, { [-1] }>) {
        unsafe {
            let alias = input;
            set_tensor_token(alias, gdc_wait_tko(None));
        }
    }

    #[cutile::entry()]
    fn block_local_helper(input: &Tensor<f32, { [-1] }>) {
        {
            let alias = input;
            wait_in_helper(alias);
        }
    }

    #[cutile::entry()]
    fn update_then_shadow(input: &Tensor<f32, { [-1] }>, other: &Tensor<f32, { [-1] }>) {
        unsafe {
            input.set_token(gdc_wait_tko(None));
            let input = other;
            let _token = input.token();
        }
    }

    #[cutile::entry()]
    fn helper_then_shadow(input: &Tensor<f32, { [-1] }>, other: &Tensor<f32, { [-1] }>) {
        {
            wait_in_helper(input);
            let input = other;
            let _token = input.token();
        }
    }

    #[cutile::entry()]
    fn initializer_then_shadow(input: &Tensor<f32, { [-1] }>, other: &Tensor<f32, { [-1] }>) {
        {
            let input = {
                unsafe { input.set_token(gdc_wait_tko(None)) };
                other
            };
            let _token = input.token();
        }
    }
}

use token_module::__module_ast_self as token_module_ast;

fn compile(entry: &str, args: &[(&str, &[i32])]) -> String {
    KernelCompiler::new(token_module_ast, "token_module", entry)
        .target("sm_120")
        .bytecode_version(BytecodeVersion::V13_4)
        .strides(args)
        .options(CompileOptions::default())
        .compile()
        .unwrap_or_else(|e| panic!("compile {entry}: {e:?}"))
        .ir_text()
}

/// Lines of `ir` containing `op`, in order.
fn lines_with<'a>(ir: &'a str, op: &str) -> Vec<&'a str> {
    ir.lines().filter(|l| l.contains(op)).collect()
}

/// The last SSA value an IR line defines: for memory ops that is the
/// completion token (`%28, %29 = load_view_tko ...` -> `%29`).
fn result_of(line: &str) -> &str {
    let (lhs, _) = line
        .trim()
        .split_once(" = ")
        .unwrap_or_else(|| panic!("no result on line: {line}"));
    lhs.rsplit(',').next().unwrap().trim()
}

/// The `token = %N` operand of an IR line.
fn token_operand(line: &str) -> &str {
    let start = line
        .find("token = ")
        .unwrap_or_else(|| panic!("no token operand: {line}"));
    let rest = &line[start + "token = ".len()..];
    let end = rest
        .find(|c: char| !(c == '%' || c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn second_store_consumes_the_first_stores_token() {
    common::with_test_stack(|| {
        let ir = compile("store_then_store", &[("z", &[1]), ("x", &[1])]);
        let stores = lines_with(&ir, "store_view_tko");
        assert_eq!(stores.len(), 2, "IR:\n{ir}");
        let first = result_of(stores[0]);
        assert_eq!(
            token_operand(stores[1]),
            first,
            "second store must consume the first store's token.\nIR:\n{ir}"
        );
    });
}

#[test]
fn token_method_reads_the_stores_token() {
    common::with_test_stack(|| {
        let ir = compile("store_token_is_tensor_token", &[("z", &[1]), ("x", &[1])]);
        let store = result_of(lines_with(&ir, "store_view_tko")[0]);
        let signals = lines_with(&ir, "gdc_launch_dependents_tko");
        assert_eq!(signals.len(), 2, "IR:\n{ir}");
        for signal in signals {
            assert_eq!(
                token_operand(signal),
                store,
                "both spellings must yield the store's token.\nIR:\n{ir}"
            );
        }
    });
}

#[test]
fn producer_signal_consumes_the_stores_token() {
    common::with_test_stack(|| {
        let ir = compile("producer", &[("z", &[1]), ("x", &[1])]);
        let store = result_of(lines_with(&ir, "store_view_tko")[0]);
        let signal = lines_with(&ir, "gdc_launch_dependents_tko")[0];
        assert_eq!(token_operand(signal), store, "IR:\n{ir}");
    });
}

#[test]
fn consumer_load_consumes_the_waits_token() {
    common::with_test_stack(|| {
        let ir = compile("consumer", &[("out", &[1]), ("input", &[1])]);
        let wait = result_of(lines_with(&ir, "gdc_wait_tko")[0]);
        let load = lines_with(&ir, "load_view_tko")[0];
        assert_eq!(
            token_operand(load),
            wait,
            "the load of producer data must consume the wait's token.\nIR:\n{ir}"
        );
    });
}

#[test]
fn consumer_without_set_does_not_depend_on_the_wait() {
    common::with_test_stack(|| {
        let ir = compile("consumer_without_set", &[("out", &[1]), ("input", &[1])]);
        let wait = result_of(lines_with(&ir, "gdc_wait_tko")[0]);
        let load = lines_with(&ir, "load_view_tko")[0];
        assert_ne!(
            token_operand(load),
            wait,
            "without set_token the load must not be chained to the wait.\nIR:\n{ir}"
        );
    });
}

#[test]
fn shadowed_binding_does_not_leak_its_token_out_of_the_block() {
    common::with_test_stack(|| {
        let ir = compile(
            "shadow_without_install",
            &[("out", &[1]), ("input", &[1]), ("other", &[1])],
        );
        let wait = result_of(lines_with(&ir, "gdc_wait_tko")[0]);
        let load = lines_with(&ir, "load_view_tko")[0];
        assert_ne!(
            token_operand(load),
            wait,
            "the outer `input` must keep its own token; the shadow's wait token leaked.\nIR:\n{ir}"
        );
    });
}

#[test]
fn mutable_shared_reference_keeps_its_installed_token() {
    common::with_test_stack(|| {
        let ir = compile("mutable_alias", &[("out", &[1]), ("input", &[1])]);
        let wait = result_of(lines_with(&ir, "gdc_wait_tko")[0]);
        assert_eq!(
            token_operand(lines_with(&ir, "load_view_tko")[0]),
            wait,
            "IR:\n{ir}"
        );
    });
}

#[test]
fn renamed_helper_parameters_do_not_overwrite_another_tensors_token() {
    common::with_test_stack(|| {
        let ir = compile(
            "swapped_params",
            &[("out", &[1]), ("input", &[1]), ("other", &[1])],
        );
        let waits = lines_with(&ir, "gdc_wait_tko");
        let loads = lines_with(&ir, "load_view_tko");
        assert_eq!(waits.len(), 2, "IR:\n{ir}");
        assert_eq!(loads.len(), 2, "IR:\n{ir}");
        assert_eq!(token_operand(loads[0]), result_of(waits[1]), "IR:\n{ir}");
        assert_eq!(token_operand(loads[1]), result_of(waits[0]), "IR:\n{ir}");
    });
}

fn assert_rejected(entry: &str, args: &[(&str, &[i32])], expected: &str) {
    let result = KernelCompiler::new(token_module_ast, "token_module", entry)
        .target("sm_120")
        .bytecode_version(BytecodeVersion::V13_4)
        .strides(args)
        .compile();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("{entry} unexpectedly compiled"),
    };
    match error {
        cutile_compiler::error::JITError::Located(message, location) => {
            assert_eq!(message, expected, "{entry}: {location:?}");
            assert!(
                location.file.ends_with("token_threading.rs"),
                "{entry}: {location:?}"
            );
            assert!(location.line > 0, "{entry}: {location:?}");
        }
        error => panic!("{entry}: expected a source-located error, got {error}"),
    }
}

#[test]
fn explicit_token_updates_in_control_flow_are_rejected() {
    common::with_test_stack(|| {
        for entry in [
            "if_else",
            "else_only",
            "constant_if",
            "for_loop",
            "while_loop",
            "loop_break",
            "short_circuit",
            "constant_short_circuit",
            "helper_in_if",
            "helper_in_loop",
        ] {
            assert_rejected(entry, &[("input", &[1])], "set_token/set_tensor_token is not supported inside conditional or loop regions; install the token before entering the region");
        }
    });
}

#[test]
fn explicit_token_updates_on_block_local_bindings_are_rejected() {
    common::with_test_stack(|| {
        let message = "set_token/set_tensor_token requires a function-level tensor binding; bindings declared inside nested blocks are not supported";
        for entry in ["block_local_direct", "block_local_helper"] {
            assert_rejected(entry, &[("input", &[1])], message);
        }
        assert_rejected(
            "shadow_in_block",
            &[("out", &[1]), ("input", &[1]), ("other", &[1])],
            message,
        );
    });
}

#[test]
fn rebinding_after_explicit_token_installation_is_rejected() {
    common::with_test_stack(|| {
        for entry in [
            "update_then_shadow",
            "helper_then_shadow",
            "initializer_then_shadow",
        ] {
            assert_rejected(entry, &[("input", &[1]), ("other", &[1])], "cannot rebind a tensor after set_token/set_tensor_token; install the token on the final function-level binding before creating views");
        }
    });
}

/// Regression for the #298 slowdown of load-bound kernels: the second load
/// from a read-only partition consumed the first load's completion token,
/// serializing the two loads. Both must consume the same (entry) token.
#[test]
fn loads_through_a_read_only_partition_are_not_chained() {
    common::with_test_stack(|| {
        let ir = compile("two_loads_one_partition", &[("z", &[1]), ("x", &[1])]);
        let loads = lines_with(&ir, "load_view_tko");
        assert_eq!(loads.len(), 2, "IR:\n{ir}");
        assert_eq!(
            token_operand(loads[0]),
            token_operand(loads[1]),
            "the second load must not depend on the first load's completion.\nIR:\n{ir}"
        );
        let first_result = result_of(loads[0]);
        assert_ne!(token_operand(loads[1]), first_result, "IR:\n{ir}");
    });
}
