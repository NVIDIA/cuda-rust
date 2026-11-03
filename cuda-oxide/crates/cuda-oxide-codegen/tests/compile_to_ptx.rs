/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused contract tests for the standalone compiler.

#![cfg(unix)]

use cuda_oxide_codegen::experimental::{
    CodegenModule, CompilationStage, CompileError, CompileOptions, Compiler, DiagnosticLevel,
    Optimization, Target, Toolchain,
};
use dialect_mir::ops::{MirCallOp, MirFuncOp, MirReturnOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        types::{FP32Type, FunctionType},
    },
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    printable::Printable,
};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
static FAKE_TOOLS_LOCK: Mutex<()> = Mutex::new(());

struct FakeTools {
    _guard: MutexGuard<'static, ()>,
    root: PathBuf,
    llc: PathBuf,
    opt: Option<PathBuf>,
}

impl FakeTools {
    fn successful(with_opt: bool) -> Self {
        let guard = FAKE_TOOLS_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = unique_test_dir("success");
        let llc = write_tool(
            &root,
            "llc",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
out=""
target="sm_80"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -mcpu=*) target="${1#-mcpu=}" ;;
    -o) shift; out="$1" ;;
  esac
  shift
done
printf '.version 8.0\n.target %s\n.address_size 64\n.visible .entry fake() { ret; }\n' "$target" > "$out"
"#,
        );
        let opt = with_opt.then(|| {
            write_tool(
                &root,
                "opt",
                r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
input=""
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; out="$1" ;;
    -*) ;;
    *) input="$1" ;;
  esac
  shift
done
cp "$input" "$out"
"#,
            )
        });
        Self {
            _guard: guard,
            root,
            llc,
            opt,
        }
    }

    fn post_opt_dynamic_stack() -> Self {
        let guard = FAKE_TOOLS_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = unique_test_dir("post_opt_dynamic_stack");
        let llc = write_tool(
            &root,
            "llc",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
out=""
input=""
target="sm_80"
ptx73=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -mcpu=*) target="${1#-mcpu=}" ;;
    -mattr=+ptx73) ptx73=1 ;;
    -o) shift; out="$1" ;;
    -*) ;;
    *) input="$1" ;;
  esac
  shift
done
grep -F -q 'call ptr addrspace(3) @llvm.stacksave.p3()' "$input" || {
  echo "fake opt output did not reach llc" >&2
  exit 11
}
grep -F -q 'call void @llvm.stackrestore.p3(ptr addrspace(3) %post_opt_saved_stack)' "$input" || {
  echo "fake opt output did not contain the reachable stack restore" >&2
  exit 14
}
[ "$ptx73" -eq 1 ] || {
  echo "final llc input did not request +ptx73" >&2
  exit 12
}
printf '.version 7.3\n.target %s\n.address_size 64\n.visible .entry fake() { ret; }\n' "$target" > "$out"
"#,
        );
        let opt = Some(write_tool(
            &root,
            "opt",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
input=""
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; out="$1" ;;
    -*) ;;
    *) input="$1" ;;
  esac
  shift
done
if grep -F -q '@llvm.stacksave' "$input"; then
  echo "source unexpectedly required a dynamic stack" >&2
  exit 13
fi
awk '
  /^define .*@empty\(/ {
    in_empty = 1
  }
  in_empty && /^[[:space:]]*ret void/ {
    print "  %post_opt_saved_stack = call ptr addrspace(3) @llvm.stacksave.p3()"
    print "  call void @llvm.stackrestore.p3(ptr addrspace(3) %post_opt_saved_stack)"
    introduced = 1
  }
  {
    print
  }
  in_empty && /^}/ {
    in_empty = 0
  }
  END {
    if (!introduced) {
      exit 1
    }
  }
' "$input" > "$out" || {
  echo "fake opt could not add calls to the exported kernel" >&2
  exit 15
}
printf '\ndeclare ptr addrspace(3) @llvm.stacksave.p3()\ndeclare void @llvm.stackrestore.p3(ptr addrspace(3))\n' >> "$out"
"#,
        ));
        Self {
            _guard: guard,
            root,
            llc,
            opt,
        }
    }

    fn post_opt_local_memory() -> Self {
        let guard = FAKE_TOOLS_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = unique_test_dir("post_opt_local_memory");
        let llc = write_tool(
            &root,
            "llc",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
out=""
input=""
target="sm_80"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -mcpu=*) target="${1#-mcpu=}" ;;
    -o) shift; out="$1" ;;
    -*) ;;
    *) input="$1" ;;
  esac
  shift
done
grep -F -q '%__cuda_oxide_local_x330931360973637261746368095b7533323b20345d_ = alloca [4 x i32]' "$input" || {
  echo "tagged post-opt alloca did not reach llc" >&2
  exit 21
}
printf '.version 8.0\n.target %s\n.address_size 64\n.visible .entry fake() { ret; }\n' "$target" > "$out"
"#,
        );
        let opt = Some(write_tool(
            &root,
            "opt",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
input=""
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; out="$1" ;;
    -*) ;;
    *) input="$1" ;;
  esac
  shift
done
awk '
  /^define .*@empty\(/ {
    in_empty = 1
  }
  in_empty && /^[[:space:]]*ret void/ {
    print "  %local_diag_index = add i64 0, 1"
    print "  %__cuda_oxide_local_x330931360973637261746368095b7533323b20345d_ = alloca [4 x i32], align 4"
    print "  %local_diag_element = getelementptr [4 x i32], ptr %__cuda_oxide_local_x330931360973637261746368095b7533323b20345d_, i64 0, i64 %local_diag_index"
    introduced = 1
  }
  {
    print
  }
  in_empty && /^}/ {
    in_empty = 0
  }
  END {
    if (!introduced) {
      exit 1
    }
  }
' "$input" > "$out" || {
  echo "fake opt could not add tagged local-memory alloca" >&2
  exit 22
}
"#,
        ));
        Self {
            _guard: guard,
            root,
            llc,
            opt,
        }
    }

    fn failing_llc() -> Self {
        let guard = FAKE_TOOLS_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = unique_test_dir("failing_llc");
        let llc = write_tool(
            &root,
            "llc",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
echo "deliberate llc failure" >&2
exit 7
"#,
        );
        Self {
            _guard: guard,
            root,
            llc,
            opt: None,
        }
    }

    fn failing_opt() -> Self {
        let mut tools = Self::successful(false);
        tools.opt = Some(write_tool(
            &tools.root,
            "opt",
            r#"#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "LLVM version 21.0.0"
  exit 0
fi
echo "deliberate opt failure" >&2
exit 9
"#,
        ));
        tools
    }

    fn compiler(&self) -> Compiler {
        let toolchain = Toolchain::from_paths(self.llc.clone(), self.opt.clone()).unwrap();
        Compiler::new(toolchain)
    }
}

impl Drop for FakeTools {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn unique_test_dir(label: &str) -> PathBuf {
    let counter = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "cuda_oxide_codegen_test_{}_{}_{}",
        std::process::id(),
        label,
        counter
    ));
    std::fs::create_dir(&root).unwrap();
    root
}

fn write_tool(root: &Path, name: &str, contents: &str) -> PathBuf {
    let path = root.join(name);
    std::fs::write(&path, contents).unwrap();
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&path, permissions).unwrap();
    // Absorb Linux's fork/exec ETXTBSY window before any test relies on the
    // tool. A concurrently spawning test can fork while this script's write
    // fd is briefly open; until that child execs (closing its inherited
    // CLOEXEC fd), executing the fresh script fails with "text file busy".
    // Probing here keeps the race out of every toolchain-resolution assert.
    for attempt in 0..50 {
        match std::process::Command::new(&path).arg("--version").output() {
            Ok(_) => break,
            Err(_) if attempt < 49 => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(error) => panic!(
                "fake tool {} never became executable: {error}",
                path.display()
            ),
        }
    }
    path
}

fn module_block(
    ctx: &mut pliron::context::Context,
    module: &pliron::builtin::ops::ModuleOp,
) -> pliron::context::Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    let existing = {
        let region = module_region.deref(ctx);
        region.iter(ctx).next()
    };
    existing.unwrap_or_else(|| {
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(module_region, ctx);
        block
    })
}

fn add_empty_function(module: &mut CodegenModule, kernel: bool) {
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let func_type = FunctionType::get(ctx, vec![], vec![]);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, op, TypeAttr::new(func_type.into()));
        function.set_symbol_name(ctx, "empty".try_into().unwrap());
        let entry = BasicBlock::new(ctx, None, vec![]);
        let function_region = function.get_operation().deref(ctx).get_region(0);
        entry.insert_at_back(function_region, ctx);
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret.insert_at_back(entry, ctx);
        function.get_operation().insert_at_back(module_block, ctx);
    });
    if kernel {
        module.mark_kernel_entry("empty").unwrap();
    }
}

fn module_text(module: &CodegenModule) -> String {
    module.inspect(|ctx, module| module.get_operation().deref(ctx).disp(ctx).to_string())
}

#[test]
fn compilation_preserves_input_and_is_repeatable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Compiler>();

    let tools = FakeTools::successful(true);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("repeatable").unwrap();
    add_empty_function(&mut module, true);
    let before = module_text(&module);
    assert!(
        before.contains("gpu_kernel"),
        "typed helper marks entry:\n{before}"
    );

    let options = CompileOptions::new(Target::parse("sm_80").unwrap());
    let first = compiler.compile(&mut module, &options).unwrap();
    let after_first = module_text(&module);
    let second = compiler.compile(&mut module, &options).unwrap();
    let after_second = module_text(&module);

    assert_eq!(before, after_first, "lowering must affect only the clone");
    assert_eq!(before, after_second, "a second compile sees the same input");
    assert_eq!(first.ptx(), second.ptx());
    assert_eq!(first.target(), &Target::parse("sm_80").unwrap());
    assert!(first.ptx().starts_with(b".version"));
    assert!(first.diagnostics().iter().any(|diagnostic| {
        diagnostic.level == DiagnosticLevel::Note && diagnostic.stage == CompilationStage::Codegen
    }));
}

#[test]
fn verbose_option_surfaces_pipeline_diagnostics() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("verbose").unwrap();
    add_empty_function(&mut module, true);
    let options = CompileOptions::new(Target::parse("sm_80").unwrap())
        .with_optimization(Optimization::None)
        .with_verbose(true);

    let compilation = compiler.compile(&mut module, &options).unwrap();
    assert!(
        compilation
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("Target:")),
        "expected a verbose target-selection diagnostic, got {:?}",
        compilation.diagnostics()
    );
}

#[test]
fn the_default_verbose_setting_suppresses_pipeline_diagnostics() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("quiet").unwrap();
    add_empty_function(&mut module, true);
    let options =
        CompileOptions::new(Target::parse("sm_80").unwrap()).with_optimization(Optimization::None);
    assert!(!options.verbose(), "verbose defaults to false");

    let compilation = compiler.compile(&mut module, &options).unwrap();
    assert!(
        !compilation
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("Target:")),
        "the verbose target-selection diagnostic must be absent by default, got {:?}",
        compilation.diagnostics()
    );
}

#[test]
fn from_paths_records_a_toolchain_diagnostic_on_success() {
    let tools = FakeTools::successful(true);
    let toolchain = Toolchain::from_paths(tools.llc.clone(), tools.opt.clone()).unwrap();
    let selection = toolchain
        .diagnostics()
        .iter()
        .find(|diagnostic| diagnostic.message.starts_with("explicit toolchain:"))
        .expect("explicit toolchain construction should report what it selected");
    assert_eq!(selection.level, DiagnosticLevel::Note);
    assert_eq!(selection.stage, CompilationStage::Toolchain);
    assert!(
        selection.message.contains(&tools.llc.display().to_string()),
        "{}",
        selection.message
    );
}

#[test]
fn discover_records_a_toolchain_diagnostic_on_success() {
    // Unlike the rest of this suite, discovery cannot be faked: it reads the
    // Rust sysroot and PATH. Skip where no LLVM 21+ llc exists.
    let Ok(toolchain) = Toolchain::discover() else {
        return;
    };
    let selection = toolchain
        .diagnostics()
        .iter()
        .find(|diagnostic| diagnostic.message.starts_with("discovered toolchain:"))
        .expect("discovery should report what it selected");
    assert_eq!(selection.level, DiagnosticLevel::Note);
    assert_eq!(selection.stage, CompilationStage::Toolchain);
    assert!(
        selection
            .message
            .contains(&toolchain.llc_path().display().to_string()),
        "{}",
        selection.message
    );
}

#[test]
fn compile_owned_matches_compile_and_leaves_no_reusable_module() {
    let tools = FakeTools::successful(true);
    let compiler = tools.compiler();
    let options = CompileOptions::new(Target::parse("sm_80").unwrap());

    let mut borrowed_module = CodegenModule::new("borrowed").unwrap();
    add_empty_function(&mut borrowed_module, true);
    let borrowed = compiler.compile(&mut borrowed_module, &options).unwrap();

    let mut owned_module = CodegenModule::new("owned").unwrap();
    add_empty_function(&mut owned_module, true);
    let owned = compiler.compile_owned(owned_module, &options).unwrap();

    assert_eq!(borrowed.ptx(), owned.ptx());
    assert_eq!(borrowed.target(), owned.target());

    // What the clone in `compile` buys, and what `compile_owned` gives up:
    // the borrowed module survives its compilation and produces the same PTX
    // again. `owned_module` was consumed above and cannot be recompiled, which
    // the type system enforces at every call site.
    let again = compiler.compile(&mut borrowed_module, &options).unwrap();
    assert_eq!(again.ptx(), borrowed.ptx());
}

#[test]
fn erased_owned_root_is_a_structured_error_not_a_panic() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("erased").unwrap();
    module.edit(|ctx, module| Operation::erase(module.get_operation(), ctx));

    let marker_error = module.mark_kernel_entry("missing").unwrap_err();
    assert!(matches!(marker_error, CompileError::InvalidModule { .. }));

    let error = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap())
                .with_optimization(Optimization::None),
        )
        .unwrap_err();
    assert!(matches!(error, CompileError::InvalidModule { .. }));
    assert_eq!(error.stage(), CompilationStage::Input);
}

#[test]
fn invalid_target_is_rejected_before_compilation() {
    for target in ["", "80", "sm_9", "sm_90x"] {
        let error = Target::parse(target).unwrap_err();
        assert!(matches!(error, CompileError::InvalidTarget { .. }));
        assert_eq!(error.stage(), CompilationStage::Input);
    }
}

#[test]
fn kernel_marker_resolves_only_owned_top_level_functions() {
    let mut module = CodegenModule::new("marker").unwrap();
    add_empty_function(&mut module, false);
    module.mark_kernel_entry("empty").unwrap();

    let text = module_text(&module);
    assert!(text.contains("gpu_kernel"));
    let error = module.mark_kernel_entry("missing").unwrap_err();
    assert!(matches!(error, CompileError::InvalidModule { .. }));
    assert_eq!(error.stage(), CompilationStage::Input);
}

#[test]
fn kernel_marker_rejects_a_root_whose_region_was_moved() {
    let mut module = CodegenModule::new("missing_region").unwrap();
    module.edit(|ctx, module| {
        let region = module.get_operation().deref(ctx).get_region(0);
        let replacement =
            pliron::builtin::ops::ModuleOp::new(ctx, "replacement_owner".try_into().unwrap());
        pliron::region::Region::move_to_op(region, replacement.get_operation(), ctx);
    });

    let error = module.mark_kernel_entry("missing").unwrap_err();
    assert!(matches!(error, CompileError::InvalidModule { .. }));
    assert_eq!(error.stage(), CompilationStage::Input);
}

#[test]
fn malformed_mir_is_rejected_without_mutating_the_source() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("malformed").unwrap();
    module.edit(|ctx, module| {
        let block = module_block(ctx, module);
        let function_type = FunctionType::get(ctx, vec![], vec![]);
        let operation = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, operation, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, "missing_return".try_into().unwrap());
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(operation.deref(ctx).get_region(0), ctx);
        operation.insert_at_back(block, ctx);
    });
    let before = module_text(&module);
    let options =
        CompileOptions::new(Target::parse("sm_80").unwrap()).with_optimization(Optimization::None);

    let error = compiler.compile(&mut module, &options).unwrap_err();
    assert!(matches!(error, CompileError::Verification { .. }));
    assert_eq!(error.stage(), CompilationStage::MirPreparation);
    assert_eq!(module_text(&module), before);
}

fn add_llvm_declaration(module: &mut CodegenModule, name: &str) {
    module.edit(|ctx, module| {
        use llvm_export::{ops::FuncOp, types::FuncType};
        use pliron::builtin::types::{IntegerType, Signedness};

        let block = module_block(ctx, module);
        let i32_type = IntegerType::get(ctx, 32, Signedness::Signless);
        let function_type = FuncType::get(ctx, i32_type.into(), vec![i32_type.into()], false);
        let function = FuncOp::new(ctx, name.try_into().unwrap(), function_type);
        function.get_operation().insert_at_back(block, ctx);
    });
}

#[test]
fn standalone_v1_rejects_libdevice_and_other_externs() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let options =
        CompileOptions::new(Target::parse("sm_80").unwrap()).with_optimization(Optimization::None);

    for symbol in ["__nv_sinf", "user_device_external"] {
        let mut module = CodegenModule::new("extern_test").unwrap();
        add_llvm_declaration(&mut module, symbol);
        let error = compiler.compile(&mut module, &options).unwrap_err();
        match error {
            CompileError::UnsupportedLinking { symbols } => {
                assert_eq!(symbols, [symbol]);
            }
            other => panic!("expected UnsupportedLinking, got {other:?}"),
        }
    }
}

fn add_exp_call(module: &mut CodegenModule) {
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let f32_type = FP32Type::get(ctx);
        let function_type = FunctionType::get(ctx, vec![f32_type.into()], vec![f32_type.into()]);
        let function_op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, function_op, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, "uses_exp".try_into().unwrap());

        let function_region = function_op.deref(ctx).get_region(0);
        let body = BasicBlock::new(ctx, None, vec![f32_type.into()]);
        body.insert_at_back(function_region, ctx);
        let argument = body.deref(ctx).get_argument(0);

        let call = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![f32_type.into()],
            vec![argument],
            vec![],
            0,
        );
        MirCallOp::new(call).set_attr_callee(
            ctx,
            StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_EXP_F32.into()),
        );
        call.insert_at_back(body, ctx);

        let result = call.deref(ctx).get_result(0);
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        ret.insert_at_back(body, ctx);
        function_op.insert_at_back(module_block, ctx);
    });
}

#[test]
fn lowered_libdevice_call_is_rejected_before_ptx_generation() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("libdevice_call").unwrap();
    add_exp_call(&mut module);
    let before = module_text(&module);
    let options =
        CompileOptions::new(Target::parse("sm_80").unwrap()).with_optimization(Optimization::None);

    let error = compiler.compile(&mut module, &options).unwrap_err();
    assert!(matches!(
        error,
        CompileError::UnsupportedLinking { ref symbols }
            if symbols == &["__nv_expf".to_string()]
    ));
    assert_eq!(error.stage(), CompilationStage::Linking);
    assert_eq!(
        module_text(&module),
        before,
        "the failed compile used a clone"
    );
}

#[test]
fn requested_optimization_requires_opt() {
    let tools = FakeTools::successful(false);
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("no_opt").unwrap();
    add_empty_function(&mut module, false);

    let error = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CompileError::OptimizationUnavailable { .. }
    ));
    assert_eq!(error.stage(), CompilationStage::Optimization);
}

#[test]
fn requirements_introduced_by_opt_reach_llc() {
    let tools = FakeTools::post_opt_dynamic_stack();
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("post_opt_dynamic_stack").unwrap();
    add_empty_function(&mut module, true);

    let compilation = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap()),
        )
        .unwrap();

    assert_eq!(compilation.target(), &Target::parse("sm_80").unwrap());
    assert!(compilation.ptx().starts_with(b".version 7.3\n"));
}

#[test]
fn verbose_post_opt_local_memory_is_reported() {
    let tools = FakeTools::post_opt_local_memory();
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("post_opt_local_memory").unwrap();
    add_empty_function(&mut module, true);

    let compilation = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap()).with_verbose(true),
        )
        .unwrap();

    let diagnostic = compilation
        .diagnostics()
        .iter()
        .find(|diagnostic| diagnostic.message.contains("local `scratch`"))
        .expect("verbose optimized build should report the surviving Rust local");
    assert!(diagnostic.message.contains("`[u32; 4]`, 16 bytes"));
    assert!(
        diagnostic
            .message
            .contains("indexed by a non-constant value")
    );
}

#[test]
fn quiet_post_opt_local_memory_is_not_reported() {
    let tools = FakeTools::post_opt_local_memory();
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("quiet_post_opt_local_memory").unwrap();
    add_empty_function(&mut module, true);

    let compilation = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap()),
        )
        .unwrap();

    assert!(
        compilation
            .diagnostics()
            .iter()
            .all(|diagnostic| !diagnostic.message.contains("local `scratch`")),
        "local-memory promotion diagnostics are opt-in through verbose mode"
    );
}

#[test]
fn tool_failures_capture_stderr_as_structured_errors() {
    let tools = FakeTools::failing_llc();
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("llc_failure").unwrap();
    add_empty_function(&mut module, false);
    let before = module_text(&module);
    let options =
        CompileOptions::new(Target::parse("sm_80").unwrap()).with_optimization(Optimization::None);
    let error = compiler.compile(&mut module, &options).unwrap_err();
    assert!(
        matches!(&error, CompileError::Codegen { message } if message.contains("deliberate llc failure"))
    );
    assert_eq!(error.diagnostic().level, DiagnosticLevel::Error);
    assert_eq!(error.diagnostic().stage, CompilationStage::Codegen);
    assert_eq!(module_text(&module), before);
    drop(compiler);
    drop(tools);

    let tools = FakeTools::failing_opt();
    let compiler = tools.compiler();
    let mut module = CodegenModule::new("opt_failure").unwrap();
    add_empty_function(&mut module, false);
    let before = module_text(&module);
    let error = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_80").unwrap()),
        )
        .unwrap_err();
    assert!(matches!(
        &error,
        CompileError::OptimizationFailed { message }
            if message.contains("deliberate opt failure")
    ));
    assert_eq!(error.stage(), CompilationStage::Optimization);
    assert_eq!(module_text(&module), before);
}
