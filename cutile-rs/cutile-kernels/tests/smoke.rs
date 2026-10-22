/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::compile_api::KernelCompiler;
use cutile_compiler::ast::Module;

const TEST_STACK_SIZE: usize = 32_000_000;
const TARGET: &str = "sm_120";

type ModuleAst = fn() -> Module;

struct SmokeCase {
    name: &'static str,
    module_ast: ModuleAst,
    module_name: &'static str,
    function_name: &'static str,
    generics: Vec<String>,
    strides: Vec<(&'static str, Vec<i32>)>,
    grid: Option<(u32, u32, u32)>,
}

fn g(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn compile_cases(cases: Vec<SmokeCase>) {
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(move || {
            for case in cases {
                eprintln!("smoke compiling {}", case.name);
                let stride_refs: Vec<(&str, &[i32])> = case
                    .strides
                    .iter()
                    .map(|(name, strides)| (*name, strides.as_slice()))
                    .collect();
                let mut compiler =
                    KernelCompiler::new(case.module_ast, case.module_name, case.function_name)
                        .target(TARGET)
                        .generics(case.generics)
                        .strides(&stride_refs);
                if let Some(grid) = case.grid {
                    compiler = compiler.grid(grid);
                }
                let artifacts = compiler
                    .compile()
                    .unwrap_or_else(|err| panic!("{} failed to compile: {err}", case.name));
                let ir = artifacts.ir_text();
                assert!(!ir.trim().is_empty(), "{} emitted empty IR", case.name);
                let bytecode = artifacts.bytecode().unwrap_or_else(|err| {
                    panic!("{} bytecode serialization failed: {err}", case.name)
                });
                assert!(!bytecode.is_empty(), "{} emitted empty bytecode", case.name);
                assert_eq!(
                    &bytecode[..8],
                    &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00],
                    "{} emitted invalid TileIR bytecode magic",
                    case.name
                );
            }
        })
        .expect("spawn smoke test thread")
        .join()
        .expect("smoke test thread panicked");
}

#[test]
fn smoke_compile_argmax_embedding_pointwise_and_norms() {
    compile_cases(vec![
        SmokeCase {
            name: "argmax_blocks_f16",
            module_ast: cutile_kernels::argmax::argmax_blocks_f16_module::__module_ast_self,
            module_name: "argmax_blocks_f16_module",
            function_name: "argmax_blocks_f16",
            generics: g(&["64"]),
            strides: vec![
                ("logits", vec![1]),
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "lm_head_argmax_blocks_f16",
            module_ast: cutile_kernels::argmax::lm_head_argmax_blocks_f16_module::__module_ast_self,
            module_name: "lm_head_argmax_blocks_f16_module",
            function_name: "lm_head_argmax_blocks_f16",
            generics: g(&["64"]),
            strides: vec![
                ("weights", vec![64, 1]),
                ("hidden", vec![64, 1]),
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "argmax_reduce_blocks_to_u32",
            module_ast:
                cutile_kernels::argmax::argmax_reduce_blocks_to_u32_module::__module_ast_self,
            module_name: "argmax_reduce_blocks_to_u32_module",
            function_name: "argmax_reduce_blocks_to_u32",
            generics: g(&["64"]),
            strides: vec![
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
                ("out", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "embedding_batch_f16",
            module_ast: cutile_kernels::embeddings::embedding_batch_f16_module::__module_ast_self,
            module_name: "embedding_batch_f16_module",
            function_name: "embedding_batch_f16",
            generics: g(&["64", "32"]),
            strides: vec![
                ("token_ids", vec![1]),
                ("table", vec![64, 1]),
                ("out", vec![32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_2d_f16",
            module_ast: cutile_kernels::pointwise::add_2d_f16_module::__module_ast_self,
            module_name: "add_2d_f16_module",
            function_name: "add_2d_f16",
            generics: g(&["32"]),
            strides: vec![
                ("out", vec![32, 1]),
                ("lhs", vec![64, 1]),
                ("rhs", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "silu_mul_2d_f16",
            module_ast: cutile_kernels::pointwise::silu_mul_2d_f16_module::__module_ast_self,
            module_name: "silu_mul_2d_f16_module",
            function_name: "silu_mul_2d_f16",
            generics: g(&["32"]),
            strides: vec![
                ("out", vec![32, 1]),
                ("gate", vec![64, 1]),
                ("up", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "gather_row_f16",
            module_ast: cutile_kernels::pointwise::gather_row_f16_module::__module_ast_self,
            module_name: "gather_row_f16_module",
            function_name: "gather_row_f16",
            generics: g(&["32"]),
            strides: vec![("src", vec![64, 1]), ("out", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rms_norm_f16",
            module_ast: cutile_kernels::norms::rms_norm_f16_module::__module_ast_self,
            module_name: "rms_norm_f16_module",
            function_name: "rms_norm_f16",
            generics: g(&["64", "32"]),
            strides: vec![("x", vec![64, 1]), ("w", vec![1]), ("out", vec![64, 1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_rms_norm_f16",
            module_ast: cutile_kernels::norms::add_rms_norm_f16_module::__module_ast_self,
            module_name: "add_rms_norm_f16_module",
            function_name: "add_rms_norm_f16",
            generics: g(&["64", "32"]),
            strides: vec![
                ("residual", vec![64, 1]),
                ("x", vec![64, 1]),
                ("w", vec![1]),
                ("out", vec![64, 1]),
                ("residual_out", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_norm_f16",
            module_ast: cutile_kernels::norms::qk_norm_f16_module::__module_ast_self,
            module_name: "qk_norm_f16_module",
            function_name: "qk_norm_f16",
            generics: g(&["64", "32"]),
            strides: vec![
                ("q", vec![64, 1]),
                ("k", vec![64, 1]),
                ("q_weight", vec![1]),
                ("k_weight", vec![1]),
                ("out", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
    ]);
}

#[test]
fn smoke_compile_positional_and_kv_cache() {
    compile_cases(vec![
        SmokeCase {
            name: "rope_seq_f16",
            module_ast: cutile_kernels::positional::rope_seq_f16_module::__module_ast_self,
            module_name: "rope_seq_f16_module",
            function_name: "rope_seq_f16",
            generics: g(&["64", "32"]),
            strides: vec![
                ("x", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rope_seq_dynpos_f16",
            module_ast: cutile_kernels::positional::rope_seq_dynpos_f16_module::__module_ast_self,
            module_name: "rope_seq_dynpos_f16_module",
            function_name: "rope_seq_dynpos_f16",
            generics: g(&["64", "32"]),
            strides: vec![
                ("x", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_rope_dynpos_f16",
            module_ast: cutile_kernels::positional::qk_rope_dynpos_f16_module::__module_ast_self,
            module_name: "qk_rope_dynpos_f16_module",
            function_name: "qk_rope_dynpos_f16",
            generics: g(&["64", "32", "1"]),
            strides: vec![
                ("q", vec![512, 64, 1]),
                ("k", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "kv_cache_update_seq_f16",
            module_ast: cutile_kernels::kv_cache::kv_cache_update_seq_f16_module::__module_ast_self,
            module_name: "kv_cache_update_seq_f16_module",
            function_name: "kv_cache_update_seq_f16",
            generics: g(&["64", "32", "16"]),
            strides: vec![
                ("new_k", vec![512, 64, 1]),
                ("new_v", vec![512, 64, 1]),
                ("k_cache", vec![512, 32, 1]),
                ("v_cache", vec![512, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "kv_cache_update_seq_dynpos_f16",
            module_ast:
                cutile_kernels::kv_cache::kv_cache_update_seq_dynpos_f16_module::__module_ast_self,
            module_name: "kv_cache_update_seq_dynpos_f16_module",
            function_name: "kv_cache_update_seq_dynpos_f16",
            generics: g(&["64", "32", "128"]),
            strides: vec![
                ("new_k", vec![512, 64, 1]),
                ("new_v", vec![512, 64, 1]),
                ("k_cache", vec![8192, 32, 1]),
                ("v_cache", vec![8192, 32, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
    ]);
}

#[test]
fn smoke_compile_attention() {
    compile_cases(vec![
        SmokeCase {
            name: "flash_attn_causal_seq_f16",
            module_ast: cutile_kernels::attention::flash_attn_causal_seq_f16_module::__module_ast_self,
            module_name: "flash_attn_causal_seq_f16_module",
            function_name: "flash_attn_causal_seq_f16",
            generics: g(&["16", "32", "64"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "flash_attn_causal_seq_dynpos_f16",
            module_ast:
                cutile_kernels::attention::flash_attn_causal_seq_dynpos_f16_module::__module_ast_self,
            module_name: "flash_attn_causal_seq_dynpos_f16_module",
            function_name: "flash_attn_causal_seq_dynpos_f16",
            generics: g(&["16", "32", "64"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_prefill_causal",
            module_ast: cutile_kernels::attention::fmha_prefill_causal_module::__module_ast_self,
            module_name: "fmha_prefill_causal_module",
            function_name: "fmha_prefill_causal",
            generics: g(&["f16", "16", "32", "64", "1", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_prefill_gqa",
            module_ast: cutile_kernels::attention::fmha_prefill_gqa_module::__module_ast_self,
            module_name: "fmha_prefill_gqa_module",
            function_name: "fmha_prefill_gqa",
            generics: g(&["f16", "16", "32", "64", "4", "64", "1", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![256, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_causal",
            module_ast: cutile_kernels::attention::fmha_causal_module::__module_ast_self,
            module_name: "fmha_causal_module",
            function_name: "fmha_causal",
            generics: g(&["f16", "16", "32", "64", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_decode_gqa_split",
            module_ast: cutile_kernels::attention::fmha_decode_gqa_split_module::__module_ast_self,
            module_name: "fmha_decode_gqa_split_module",
            function_name: "fmha_decode_gqa_split",
            generics: g(&["f16", "4", "32", "64", "4", "1"]),
            strides: vec![
                ("q", vec![256, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("att_out", vec![256, 64, 1]),
                ("lse_out", vec![4, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "splitk_reduce_merge",
            module_ast: cutile_kernels::attention::splitk_reduce_merge_module::__module_ast_self,
            module_name: "splitk_reduce_merge_module",
            function_name: "splitk_reduce_merge",
            generics: g(&["f16", "4", "64", "32", "4", "16", "1"]),
            strides: vec![
                ("att_partial", vec![1024, 64, 1]),
                ("lse_partial", vec![16, 1]),
                ("out", vec![128, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
    ]);
}

#[test]
fn smoke_compile_experimental() {
    use cutile_kernels::experimental::kvbm::{
        contiguous_to_stacked_f16_generics, stacked_to_contiguous_f16_generics, ContiguousLayout,
        StackedLayout,
    };

    compile_cases(vec![
        SmokeCase {
            name: "fmha_prefill_gqa_lpt",
            module_ast:
                cutile_kernels::experimental::attention::fmha_prefill_gqa_lpt_module::__module_ast_self,
            module_name: "fmha_prefill_gqa_lpt_module",
            function_name: "fmha_prefill_gqa_lpt",
            generics: g(&["f16", "16", "32", "64", "4", "64", "1", "1", "1", "1", "1"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "attention_decode_kernel_grouped",
            module_ast:
                cutile_kernels::experimental::attention::attention_decode_kernel_grouped_module::__module_ast_self,
            module_name: "attention_decode_kernel_grouped_module",
            function_name: "attention_decode_kernel_grouped",
            generics: g(&["f16", "64", "32", "64", "4", "4", "4"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_norm_rope_kv_prefill_raw_f16",
            module_ast:
                cutile_kernels::experimental::fused_transformer::qk_norm_rope_kv_prefill_raw_f16_module::__module_ast_self,
            module_name: "qk_norm_rope_kv_prefill_raw_f16_module",
            function_name: "qk_norm_rope_kv_prefill_raw_f16",
            generics: g(&["64", "32", "128"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_norm_rope_kv_decode_raw_f16",
            module_ast:
                cutile_kernels::experimental::fused_transformer::qk_norm_rope_kv_decode_raw_f16_module::__module_ast_self,
            module_name: "qk_norm_rope_kv_decode_raw_f16_module",
            function_name: "qk_norm_rope_kv_decode_raw_f16",
            generics: g(&["64", "32", "128"]),
            strides: vec![("position_start", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "copy_stacked_to_contiguous_f16",
            module_ast:
                cutile_kernels::experimental::kvbm::copy_stacked_to_contiguous_f16_module::__module_ast_self,
            module_name: "copy_stacked_to_contiguous_f16_module",
            function_name: "copy_stacked_to_contiguous_f16",
            generics: stacked_to_contiguous_f16_generics(
                StackedLayout::Nhd,
                ContiguousLayout::Universal,
                16,
                32,
            ),
            strides: vec![("stacked_tensors", vec![1]), ("contiguous_tensors", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "copy_contiguous_to_stacked_f16",
            module_ast:
                cutile_kernels::experimental::kvbm::copy_contiguous_to_stacked_f16_module::__module_ast_self,
            module_name: "copy_contiguous_to_stacked_f16_module",
            function_name: "copy_contiguous_to_stacked_f16",
            generics: contiguous_to_stacked_f16_generics(
                StackedLayout::Nhd,
                ContiguousLayout::Universal,
                16,
                32,
            ),
            strides: vec![("stacked_tensors", vec![1]), ("contiguous_tensors", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "group_gemm_f16_nt_desc",
            module_ast:
                cutile_kernels::experimental::moe::group_gemm_f16_nt_desc_module::__module_ast_self,
            module_name: "group_gemm_f16_nt_desc_module",
            function_name: "group_gemm_f16_nt_desc",
            generics: g(&["16", "16", "32", "120"]),
            strides: vec![
                ("a_ptrs", vec![1]),
                ("b_ptrs", vec![1]),
                ("c_ptrs", vec![1]),
                ("a_metas", vec![8, 1]),
                ("b_metas", vec![8, 1]),
                ("c_metas", vec![8, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_rms_norm_decode_raw_f16",
            module_ast:
                cutile_kernels::experimental::norms::add_rms_norm_decode_raw_f16_module::__module_ast_self,
            module_name: "add_rms_norm_decode_raw_f16_module",
            function_name: "add_rms_norm_decode_raw_f16",
            generics: g(&["64", "32"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
        // Element-generic experimental siblings (bf16). For the fmha_prefill_gqa_lpt
        // kernel (generalized in place) the bf16 case lives above; these cover the
        // additive raw-pointer / descriptor kernels over a non-f16 element.
        SmokeCase {
            name: "qk_norm_rope_kv_prefill_raw<bf16>",
            module_ast:
                cutile_kernels::experimental::fused_transformer::qk_norm_rope_kv_prefill_raw_module::__module_ast_self,
            module_name: "qk_norm_rope_kv_prefill_raw_module",
            function_name: "qk_norm_rope_kv_prefill_raw",
            generics: g(&["bf16", "64", "32", "128"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_norm_rope_kv_decode_raw<bf16>",
            module_ast:
                cutile_kernels::experimental::fused_transformer::qk_norm_rope_kv_decode_raw_module::__module_ast_self,
            module_name: "qk_norm_rope_kv_decode_raw_module",
            function_name: "qk_norm_rope_kv_decode_raw",
            generics: g(&["bf16", "64", "32", "128"]),
            strides: vec![("position_start", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "copy_stacked_to_contiguous<bf16>",
            module_ast:
                cutile_kernels::experimental::kvbm::copy_stacked_to_contiguous_module::__module_ast_self,
            module_name: "copy_stacked_to_contiguous_module",
            function_name: "copy_stacked_to_contiguous",
            generics: [
                vec!["bf16".to_string()],
                stacked_to_contiguous_f16_generics(
                    StackedLayout::Nhd,
                    ContiguousLayout::Universal,
                    16,
                    32,
                ),
            ]
            .concat(),
            strides: vec![("stacked_tensors", vec![1]), ("contiguous_tensors", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "copy_contiguous_to_stacked<bf16>",
            module_ast:
                cutile_kernels::experimental::kvbm::copy_contiguous_to_stacked_module::__module_ast_self,
            module_name: "copy_contiguous_to_stacked_module",
            function_name: "copy_contiguous_to_stacked",
            generics: [
                vec!["bf16".to_string()],
                contiguous_to_stacked_f16_generics(
                    StackedLayout::Nhd,
                    ContiguousLayout::Universal,
                    16,
                    32,
                ),
            ]
            .concat(),
            strides: vec![("stacked_tensors", vec![1]), ("contiguous_tensors", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "group_gemm_nt_desc<bf16>",
            module_ast:
                cutile_kernels::experimental::moe::group_gemm_nt_desc_module::__module_ast_self,
            module_name: "group_gemm_nt_desc_module",
            function_name: "group_gemm_nt_desc",
            generics: g(&["bf16", "16", "16", "32", "120"]),
            strides: vec![
                ("a_ptrs", vec![1]),
                ("b_ptrs", vec![1]),
                ("c_ptrs", vec![1]),
                ("a_metas", vec![8, 1]),
                ("b_metas", vec![8, 1]),
                ("c_metas", vec![8, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_rms_norm_decode_raw<bf16>",
            module_ast:
                cutile_kernels::experimental::norms::add_rms_norm_decode_raw_module::__module_ast_self,
            module_name: "add_rms_norm_decode_raw_module",
            function_name: "add_rms_norm_decode_raw",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![],
            grid: Some((1, 1, 1)),
        },
    ]);
}

#[test]
fn smoke_compile_generic_kernels() {
    compile_cases(vec![
        // Data-movement kernels over a non-float element (i32): pure copy,
        // proving these are not float-locked.
        SmokeCase {
            name: "embedding_batch<i32>",
            module_ast: cutile_kernels::embeddings::embedding_batch_module::__module_ast_self,
            module_name: "embedding_batch_module",
            function_name: "embedding_batch",
            generics: g(&["i32", "64", "32"]),
            strides: vec![
                ("token_ids", vec![1]),
                ("table", vec![64, 1]),
                ("out", vec![32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "gather_row<i32>",
            module_ast: cutile_kernels::pointwise::gather_row_module::__module_ast_self,
            module_name: "gather_row_module",
            function_name: "gather_row",
            generics: g(&["i32", "64"]),
            strides: vec![("src", vec![64, 1]), ("out", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "kv_cache_update_seq<i32>",
            module_ast: cutile_kernels::kv_cache::kv_cache_update_seq_module::__module_ast_self,
            module_name: "kv_cache_update_seq_module",
            function_name: "kv_cache_update_seq",
            generics: g(&["i32", "64", "32", "16"]),
            strides: vec![
                ("new_k", vec![2048, 64, 1]),
                ("new_v", vec![2048, 64, 1]),
                ("k_cache", vec![1024, 32, 1]),
                ("v_cache", vec![1024, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "kv_cache_update_seq_dynpos<i32>",
            module_ast:
                cutile_kernels::kv_cache::kv_cache_update_seq_dynpos_module::__module_ast_self,
            module_name: "kv_cache_update_seq_dynpos_module",
            function_name: "kv_cache_update_seq_dynpos",
            generics: g(&["i32", "64", "32", "128"]),
            strides: vec![
                ("new_k", vec![2048, 64, 1]),
                ("new_v", vec![2048, 64, 1]),
                ("k_cache", vec![4096, 32, 1]),
                ("v_cache", vec![4096, 32, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_2d<i32>",
            module_ast: cutile_kernels::pointwise::add_2d_module::__module_ast_self,
            module_name: "add_2d_module",
            function_name: "add_2d",
            generics: g(&["i32", "32"]),
            strides: vec![
                ("out", vec![32, 1]),
                ("lhs", vec![32, 1]),
                ("rhs", vec![32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        // Float-math kernels over bf16: compute stays f32, IO is the element.
        SmokeCase {
            name: "silu_mul_2d<bf16>",
            module_ast: cutile_kernels::pointwise::silu_mul_2d_module::__module_ast_self,
            module_name: "silu_mul_2d_module",
            function_name: "silu_mul_2d",
            generics: g(&["bf16", "32"]),
            strides: vec![
                ("out", vec![32, 1]),
                ("gate", vec![32, 1]),
                ("up", vec![32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rms_norm<bf16>",
            module_ast: cutile_kernels::norms::rms_norm_module::__module_ast_self,
            module_name: "rms_norm_module",
            function_name: "rms_norm",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![("x", vec![64, 1]), ("w", vec![1]), ("out", vec![64, 1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "silu_mul_2d<f8e4m3fn>",
            module_ast: cutile_kernels::pointwise::silu_mul_2d_module::__module_ast_self,
            module_name: "silu_mul_2d_module",
            function_name: "silu_mul_2d",
            generics: g(&["f8e4m3fn", "32"]),
            strides: vec![
                ("out", vec![32, 1]),
                ("gate", vec![32, 1]),
                ("up", vec![32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rms_norm<f8e4m3fn>",
            module_ast: cutile_kernels::norms::rms_norm_module::__module_ast_self,
            module_name: "rms_norm_module",
            function_name: "rms_norm",
            generics: g(&["f8e4m3fn", "64", "32"]),
            strides: vec![("x", vec![64, 1]), ("w", vec![1]), ("out", vec![64, 1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "flash_attn_causal_seq<f8e5m2>",
            module_ast: cutile_kernels::attention::flash_attn_causal_seq_module::__module_ast_self,
            module_name: "flash_attn_causal_seq_module",
            function_name: "flash_attn_causal_seq",
            generics: g(&["f8e5m2", "16", "32", "64"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "add_rms_norm<bf16>",
            module_ast: cutile_kernels::norms::add_rms_norm_module::__module_ast_self,
            module_name: "add_rms_norm_module",
            function_name: "add_rms_norm",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![
                ("residual", vec![64, 1]),
                ("x", vec![64, 1]),
                ("w", vec![1]),
                ("out", vec![64, 1]),
                ("residual_out", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_norm<bf16>",
            module_ast: cutile_kernels::norms::qk_norm_module::__module_ast_self,
            module_name: "qk_norm_module",
            function_name: "qk_norm",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![
                ("q", vec![64, 1]),
                ("k", vec![64, 1]),
                ("q_weight", vec![1]),
                ("k_weight", vec![1]),
                ("out", vec![64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rope_seq<bf16>",
            module_ast: cutile_kernels::positional::rope_seq_module::__module_ast_self,
            module_name: "rope_seq_module",
            function_name: "rope_seq",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![
                ("x", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "rope_seq_dynpos<bf16>",
            module_ast: cutile_kernels::positional::rope_seq_dynpos_module::__module_ast_self,
            module_name: "rope_seq_dynpos_module",
            function_name: "rope_seq_dynpos",
            generics: g(&["bf16", "64", "32"]),
            strides: vec![
                ("x", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_rope_dynpos<f16>",
            module_ast: cutile_kernels::positional::qk_rope_dynpos_module::__module_ast_self,
            module_name: "qk_rope_dynpos_module",
            function_name: "qk_rope_dynpos",
            generics: g(&["f16", "64", "32", "1"]),
            strides: vec![
                ("q", vec![512, 64, 1]),
                ("k", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_rope_dynpos<bf16>",
            module_ast: cutile_kernels::positional::qk_rope_dynpos_module::__module_ast_self,
            module_name: "qk_rope_dynpos_module",
            function_name: "qk_rope_dynpos",
            generics: g(&["bf16", "64", "32", "1"]),
            strides: vec![
                ("q", vec![512, 64, 1]),
                ("k", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "qk_rope_dynpos<f32>",
            module_ast: cutile_kernels::positional::qk_rope_dynpos_module::__module_ast_self,
            module_name: "qk_rope_dynpos_module",
            function_name: "qk_rope_dynpos",
            generics: g(&["f32", "64", "32", "1"]),
            strides: vec![
                ("q", vec![512, 64, 1]),
                ("k", vec![512, 64, 1]),
                ("inv_freq", vec![1]),
                ("position_start", vec![1]),
                ("out", vec![32, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "argmax_blocks<bf16>",
            module_ast: cutile_kernels::argmax::argmax_blocks_module::__module_ast_self,
            module_name: "argmax_blocks_module",
            function_name: "argmax_blocks",
            generics: g(&["bf16", "64"]),
            strides: vec![
                ("logits", vec![1]),
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "argmax_blocks<f32>",
            module_ast: cutile_kernels::argmax::argmax_blocks_module::__module_ast_self,
            module_name: "argmax_blocks_module",
            function_name: "argmax_blocks",
            generics: g(&["f32", "64"]),
            strides: vec![
                ("logits", vec![1]),
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "lm_head_argmax_blocks<bf16>",
            module_ast: cutile_kernels::argmax::lm_head_argmax_blocks_module::__module_ast_self,
            module_name: "lm_head_argmax_blocks_module",
            function_name: "lm_head_argmax_blocks",
            generics: g(&["bf16", "64"]),
            strides: vec![
                ("weights", vec![64, 1]),
                ("hidden", vec![64, 1]),
                ("block_max", vec![1]),
                ("block_idx", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "sample<f32>",
            module_ast: cutile_kernels::sampling::sample_module::__module_ast_self,
            module_name: "sample_module",
            function_name: "sample",
            generics: g(&["f32", "128"]),
            strides: vec![("logits", vec![128, 1]), ("u", vec![1]), ("out", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "sample<bf16>",
            module_ast: cutile_kernels::sampling::sample_module::__module_ast_self,
            module_name: "sample_module",
            function_name: "sample",
            generics: g(&["bf16", "128"]),
            strides: vec![("logits", vec![128, 1]), ("u", vec![1]), ("out", vec![1])],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "flash_attn_causal_seq<bf16>",
            module_ast: cutile_kernels::attention::flash_attn_causal_seq_module::__module_ast_self,
            module_name: "flash_attn_causal_seq_module",
            function_name: "flash_attn_causal_seq",
            generics: g(&["bf16", "16", "32", "64"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "flash_attn_causal_seq_dynpos<bf16>",
            module_ast:
                cutile_kernels::attention::flash_attn_causal_seq_dynpos_module::__module_ast_self,
            module_name: "flash_attn_causal_seq_dynpos_module",
            function_name: "flash_attn_causal_seq_dynpos",
            generics: g(&["bf16", "16", "32", "64"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        // fmha_* / splitk are generic in place; bf16 proves genericity and (for
        // the load_view_tko kernels) exercises the const-arg resolver fix over a
        // non-f16 element.
        SmokeCase {
            name: "fmha_prefill_causal<bf16>",
            module_ast: cutile_kernels::attention::fmha_prefill_causal_module::__module_ast_self,
            module_name: "fmha_prefill_causal_module",
            function_name: "fmha_prefill_causal",
            generics: g(&["bf16", "16", "32", "64", "1", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_prefill_gqa<bf16>",
            module_ast: cutile_kernels::attention::fmha_prefill_gqa_module::__module_ast_self,
            module_name: "fmha_prefill_gqa_module",
            function_name: "fmha_prefill_gqa",
            generics: g(&["bf16", "16", "32", "64", "4", "64", "1", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![256, 64, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_causal<bf16>",
            module_ast: cutile_kernels::attention::fmha_causal_module::__module_ast_self,
            module_name: "fmha_causal_module",
            function_name: "fmha_causal",
            generics: g(&["bf16", "16", "32", "64", "1", "1"]),
            strides: vec![
                ("q", vec![2048, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("out", vec![64, 64, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "fmha_decode_gqa_split<bf16>",
            module_ast: cutile_kernels::attention::fmha_decode_gqa_split_module::__module_ast_self,
            module_name: "fmha_decode_gqa_split_module",
            function_name: "fmha_decode_gqa_split",
            generics: g(&["bf16", "4", "32", "64", "4", "1"]),
            strides: vec![
                ("q", vec![256, 64, 1]),
                ("k", vec![2048, 64, 1]),
                ("v", vec![2048, 64, 1]),
                ("att_out", vec![256, 64, 1]),
                ("lse_out", vec![4, 1]),
                ("position_start", vec![1]),
            ],
            grid: Some((1, 1, 1)),
        },
        SmokeCase {
            name: "splitk_reduce_merge<bf16>",
            module_ast: cutile_kernels::attention::splitk_reduce_merge_module::__module_ast_self,
            module_name: "splitk_reduce_merge_module",
            function_name: "splitk_reduce_merge",
            generics: g(&["bf16", "4", "64", "32", "4", "16", "1"]),
            strides: vec![
                ("att_partial", vec![1024, 64, 1]),
                ("lse_partial", vec![16, 1]),
                ("out", vec![128, 32, 1]),
            ],
            grid: Some((1, 1, 1)),
        },
    ]);
}
