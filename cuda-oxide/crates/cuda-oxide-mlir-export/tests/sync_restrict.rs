//! Restricted proxy-fence direction, ordering, identity, and optional native-parser checks.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::ops::{MirFuncOp, MirReturnOp};
use dialect_nvvm::ops::{
    FenceProxyAsyncGenericAcquireSharedClusterClusterOp,
    FenceProxyAsyncGenericReleaseSharedCtaClusterOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::FunctionType,
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};

/// Source module containing release/acquire proxy fences whose contracts can be mutated independently.
struct Fixture {
    /// Source IR context owning the fixture's registered types and operations.
    ctx: Context,
    /// Source module containing the operations under test.
    module: ModuleOp,
    /// Source release-fence operation used for identity and attribute rejection cases.
    release: Ptr<Operation>,
    /// Source acquire-fence operation used for identity and attribute rejection cases.
    acquire: Ptr<Operation>,
}

impl Fixture {
    /// Build canonical restricted release/acquire fences with their native catalog identities.
    fn new() -> Self {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "sync_restrict".try_into().unwrap());
        let ty = FunctionType::get(&ctx, vec![], vec![]);
        let raw = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, raw, TypeAttr::new(ty.into()))
            .set_symbol_name(&mut ctx, "proxy_fences".try_into().unwrap());
        module.append_operation(&mut ctx, raw, 0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(raw.deref(&ctx).get_region(0), &ctx);
        let release = FenceProxyAsyncGenericReleaseSharedCtaClusterOp::build(&mut ctx);
        let acquire = FenceProxyAsyncGenericAcquireSharedClusterClusterOp::build(&mut ctx);
        for (op, marker) in [(release, "v1:i0314"), (acquire, "v1:i0315")] {
            op.deref_mut(&ctx).attributes.set(
                "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                StringAttr::new(marker.into()),
            );
            op.insert_at_back(block, &ctx);
        }
        let ret = Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
        Self {
            ctx,
            module,
            release,
            acquire,
        }
    }

    /// Export the fixture through the selected full-CuTe or primitives profile.
    fn translate(&self, primitives: bool) -> Result<String, String> {
        let module = if primitives {
            CutlassPrimitivesMlir22::new("sm_100a")
                .unwrap()
                .translate_module(&self.ctx, &self.module)
        } else {
            CutlassFullCuteMlir22::new("sm_100a")
                .unwrap()
                .translate_module(&self.ctx, &self.module)
        };
        module
            .and_then(|m| render_module(&m).map_err(Into::into))
            .map_err(|e| e.to_string())
    }
}

/// Check both profiles keep acquire/release order and the corresponding shared-memory restriction.
#[test]
fn selected_cutlass_profiles_preserve_proxy_direction_and_each_memory_order() {
    let fixture = Fixture::new();
    for primitives in [false, true] {
        let text = fixture.translate(primitives).unwrap();
        let fences = text
            .lines()
            .filter(|line| line.contains("\"nvvm.fence.proxy.sync_restrict\""))
            .collect::<Vec<_>>();
        assert_eq!(fences.len(), 2, "{text}");
        for (line, order) in fences.iter().zip(["release", "acquire"]) {
            assert!(
                line.contains(&format!("order = #nvvm.mem_order<{order}>")),
                "{text}"
            );
            assert!(
                line.contains("fromProxy = #nvvm.proxy_kind<generic>"),
                "{text}"
            );
            assert!(line.contains("toProxy = #nvvm.proxy_kind<async>"), "{text}");
            assert!(line.contains(": () -> ()"), "{text}");
        }
        assert!(!text.contains("inline_asm"), "{text}");
        assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
    }
}

/// Reject fences carrying the other direction's catalog identity.
#[test]
fn proxy_fences_reject_swapped_intrinsic_markers() {
    for acquire in [false, true] {
        let fixture = Fixture::new();
        let (op, wrong, expected) = if acquire {
            (fixture.acquire, "v1:i0314", "v1:i0315")
        } else {
            (fixture.release, "v1:i0315", "v1:i0314")
        };
        op.deref_mut(&fixture.ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new(wrong.into()),
        );
        let error = fixture.translate(false).unwrap_err();
        assert!(
            error.contains(&format!("expected intrinsic marker {expected}")),
            "{error}"
        );
    }
}

/// Reject source attributes the reviewed fence recipe has not explicitly consumed.
#[test]
fn proxy_fences_reject_unexpected_semantic_attributes() {
    let fixture = Fixture::new();
    fixture.release.deref_mut(&fixture.ctx).attributes.set(
        "order".try_into().unwrap(),
        StringAttr::new("acquire".into()),
    );
    let error = fixture.translate(false).unwrap_err();
    assert!(error.contains("residual attributes"), "{error}");
}

/// When the pinned parser is available, verify it accepts both exported fence forms.
#[test]
#[ignore = "requires CUDA_OXIDE_CUTLASS_PYTHON pointing to the selected CUTLASS Python environment"]
fn selected_cutlass_native_parser_verifies_both_proxy_fences() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let python = std::env::var_os("CUDA_OXIDE_CUTLASS_PYTHON")
        .expect("set CUDA_OXIDE_CUTLASS_PYTHON to the selected CUTLASS Python executable");
    let text = Fixture::new().translate(false).unwrap();
    let mut child = Command::new(python)
        .args(["-c", "import sys\nfrom cutlass._mlir import ir\nfrom cutlass._mlir.dialects import nvvm\nwith ir.Context():\n    module = ir.Module.parse(sys.stdin.read())\n    assert module.operation.verify()\n"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
