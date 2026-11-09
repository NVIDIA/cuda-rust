/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory ordering of foreign tile-library calls through branches, loops and
//! wrappers. Compile-only: each test checks def-use edges between ordering
//! tokens in the Tile IR, not just that it verifies.
//!
//! The invariants, checked for every pair of operations in each kernel by
//! [`assert_token_invariants`]:
//!
//! 1. A foreign call is ordered after every memory access (load, store,
//!    atomic) that precedes it in program order.
//! 2. A memory access is ordered after every foreign call that precedes it.
//! 3. A foreign call is ordered after every foreign call that precedes it.
//! 4. Inside a loop, 1 to 3 also hold from one iteration to the next: an
//!    operation is ordered after every call and, for a call, every access of
//!    the previous iteration, whatever their order in the body.
//!
//! Inside a loop, 1 to 3 must hold within one iteration: the checker does
//! not follow the loop's back edges for them, so an access that waits only
//! for the previous iteration's call fails. The exception is an operation
//! in a branch that ends with `continue` or `break`: the operations after
//! that branch only follow it from a later iteration or after the loop.
//!
//! Operations in different branches of one `if` are not related. Two more
//! properties keep the ordering from costing more than it should:
//!
//! 5. Memory accesses are not ordered among themselves by the foreign-call
//!    tokens: disjoint stores of a `Dim` loop stay unordered.
//! 6. A kernel whose modules declare no foreign function carries no extra
//!    token.
//!
//! Every compiled kernel also passes `verify_dominance`, so no token defined
//! in a region is used outside it.

use std::collections::{HashMap, HashSet};

use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[cutile::module]
mod ordering_kernels {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches the library's `bump`.
    #[cutile::foreign]
    pub unsafe fn bump(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;

    /// # Safety
    ///
    /// Matches the library's `bump_row`.
    #[cutile::foreign]
    pub unsafe fn bump_row(#[inout] x: Tile<f32, { [1, 32] }>) -> Tile<f32, { [1, 32] }>;

    /// A wrapper that takes and returns only tiles: no view or tensor, so no
    /// token, reaches it through its parameters.
    pub fn twice(x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }> {
        let once = unsafe { bump(x) };
        unsafe { bump(once) }
    }

    pub trait Bump {
        fn bumped(self) -> Self;
    }

    impl Bump for Tile<f32, { [32] }> {
        fn bumped(self) -> Tile<f32, { [32] }> {
            unsafe { bump(self) }
        }
    }

    #[cutile::entry()]
    fn branch(y: &mut Tensor<f32, { [32] }>, flag: i32) {
        let t = y.load();
        y.store(t);
        let mut u = t;
        if flag > 0 {
            u = unsafe { bump(t) };
        }
        y.store(u);
    }

    #[cutile::entry()]
    fn range_loop(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        for _ in 0..n {
            y.store(u);
            u = unsafe { bump(u) };
        }
        y.store(u);
    }

    #[cutile::entry()]
    fn while_loop(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        let mut i = 0i32;
        while i < n {
            y.store(u);
            u = unsafe { bump(u) };
            i = i + 1i32;
        }
        y.store(u);
    }

    #[cutile::entry()]
    fn loop_with_break(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        let mut i = 0i32;
        loop {
            if i >= n {
                break;
            }
            y.store(u);
            u = unsafe { bump(u) };
            i = i + 1i32;
        }
        y.store(u);
    }

    /// `Dim::new` is deprecated, but a `Dim` loop is the one that threads its
    /// stores through a token accumulator.
    #[allow(deprecated)]
    #[cutile::entry()]
    fn dim_loop<const N: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / 32);
        let tile_shape = shape![1, 32];
        let mut v = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let t: Tile<f32, { [1, 32] }> = constant(0.0, tile_shape);
            let u = unsafe { bump_row(t) };
            v.store(u, coord((0i32, j)));
        }
    }

    #[cutile::entry()]
    fn wrappers(y: &mut Tensor<f32, { [32] }>) {
        let t = y.load();
        y.store(t);
        let a = twice(t);
        y.store(a);
        let b = a.bumped();
        y.store(b);
    }

    static COUNTER: Global<AtomicI32, { [] }> = Global::new(0);

    /// A helper that stores through a tensor parameter.
    pub fn put(y: &mut Tensor<f32, { [32] }>, t: Tile<f32, { [32] }>) {
        y.store(t);
    }

    /// The store after the call in iteration k must come before the call in
    /// iteration k+1.
    #[cutile::entry()]
    fn store_after_call(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        for _ in 0..n {
            u = unsafe { bump(u) };
            y.store(u);
        }
        y.store(u);
    }

    /// The load after the call in iteration k must come before the call in
    /// iteration k+1 (write after read, once calls can write memory).
    #[cutile::entry()]
    fn load_after_call(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        for _ in 0..n {
            u = unsafe { bump(u) };
            let w = y.load();
            u = u + w;
        }
        y.store(u);
    }

    #[cutile::entry()]
    fn store_in_branch_then_call(y: &mut Tensor<f32, { [32] }>, flag: i32) {
        let t = y.load();
        if flag > 0 {
            y.store(t);
        }
        let u = unsafe { bump(t) };
        y.store(u);
    }

    #[cutile::entry()]
    fn store_in_loop_then_call(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let t = y.load();
        for _ in 0..n {
            y.store(t);
        }
        let u = unsafe { bump(t) };
        y.store(u);
    }

    #[cutile::entry()]
    fn store_in_helper_then_call(y: &mut Tensor<f32, { [32] }>) {
        let t = y.load();
        put(y, t);
        let u = unsafe { bump(t) };
        y.store(u);
    }

    /// Accesses to two tensors: the call is ordered after both, and both
    /// tensors' later accesses after the call.
    #[cutile::entry()]
    fn two_tensors(x: &mut Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
        let a = x.load();
        let b = y.load();
        x.store(b);
        let u = unsafe { bump(a) };
        y.store(u);
        x.store(u);
    }

    /// A call in one branch and a store in the other; the call after the
    /// `if` is ordered after both.
    #[cutile::entry()]
    fn exclusive_branches(y: &mut Tensor<f32, { [32] }>, flag: i32) {
        let t = y.load();
        let mut u = t;
        if flag > 0 {
            u = unsafe { bump(t) };
        } else {
            y.store(t);
        }
        let v = unsafe { bump(u) };
        y.store(v);
    }

    /// An atomic on a `Global` is a memory access too. It consumes no token, so
    /// only invariant 1 applies to it.
    #[cutile::entry()]
    fn atomic_then_call(y: &mut Tensor<f32, { [32] }>) {
        let one = constant(1i32, shape![]);
        let (_old, _token) = COUNTER.atomic_add(one, ordering::AcqRel, scope::Device);
        let t = y.load();
        let u = unsafe { bump(t) };
        let (_old, _token) = COUNTER.atomic_add(one, ordering::AcqRel, scope::Device);
        y.store(u);
    }

    /// Disjoint stores in a `Dim` loop, in a kernel that carries the
    /// foreign-call tokens.
    #[allow(deprecated)]
    #[cutile::entry()]
    fn dim_loop_without_calls<const N: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / 32);
        let tile_shape = shape![1, 32];
        let mut v = out
            .partition_mut(tile_shape)
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let t: Tile<f32, { [1, 32] }> = constant(0.0, tile_shape);
            v.store(t, coord((0i32, j)));
        }
    }

    #[cutile::entry()]
    fn nested(y: &mut Tensor<f32, { [32] }>, flag: i32, n: i32) {
        let mut u = y.load();
        y.store(u);
        if flag > 0 {
            for _ in 0..n {
                u = twice(u);
            }
        }
        y.store(u);
    }

    /// A read through a `&Tensor` after a call in an `unsafe {}` block. The
    /// block does not write the read-only tensor's token back.
    #[cutile::entry()]
    fn read_after_unsafe_block(x: &Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
        let a = x.load_tile(const_shape![32], [0]);
        let u = unsafe { bump(a) };
        let b = x.load_tile(const_shape![32], [0]);
        y.store(u + b);
    }

    /// The same read after a plain `{}` block that holds the call.
    #[cutile::entry()]
    fn read_after_block(x: &Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
        let a = x.load_tile(const_shape![32], [0]);
        let u = {
            let b = a + a;
            unsafe { bump(b) }
        };
        let c = x.load_tile(const_shape![32], [0]);
        y.store(u + c);
    }

    /// The read in iteration k must wait for the call in iteration k, not
    /// only for the one in iteration k-1.
    #[cutile::entry()]
    fn read_after_call_in_loop(x: &Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>, n: i32) {
        let p = x.partition(const_shape![32]);
        let mut u = p.load([0]);
        for _ in 0..n {
            u = unsafe { bump(u) };
            u = u + p.load([0]);
        }
        y.store(u);
    }

    /// A call in a branch that ends with `continue`: the accesses after the
    /// branch follow it from the next iteration.
    #[cutile::entry()]
    fn call_then_continue(x: &Tensor<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>, n: i32) {
        let p = x.partition(const_shape![32]);
        let mut u = y.load();
        let mut i = 0i32;
        loop {
            if i >= n {
                break;
            }
            i = i + 1i32;
            if i == 2i32 {
                u = unsafe { bump(u) };
                continue;
            }
            u = u + p.load([0]);
            y.store(u);
        }
        y.store(u);
    }
}

/// Kernels whose modules declare no foreign function.
#[cutile::module]
mod plain_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn plain_loop(y: &mut Tensor<f32, { [32] }>, n: i32) {
        let mut u = y.load();
        for _ in 0..n {
            y.store(u);
            u = u + u;
        }
        y.store(u);
    }
}

fn compile(entry: &str) -> String {
    compile_with(entry, vec![], &[("y", &[1])])
}

fn compile_with(entry: &str, generics: Vec<String>, strides: &[(&str, &[i32])]) -> String {
    compile_in(
        ordering_kernels::__module_ast_self,
        "ordering_kernels",
        entry,
        generics,
        strides,
    )
}

fn compile_in(
    module_ast: fn() -> cutile_compiler::ast::Module,
    module: &str,
    entry: &str,
    generics: Vec<String>,
    strides: &[(&str, &[i32])],
) -> String {
    let artifacts = KernelCompiler::new(module_ast, module, entry)
        .generics(generics)
        .strides(strides)
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    artifacts
        .module()
        .verify_dominance()
        .unwrap_or_else(|e| panic!("{entry}: {e:?}"));
    artifacts
        .bytecode()
        .unwrap_or_else(|e| panic!("{entry}: bytecode: {e}"));
    artifacts.ir_text()
}

/// Every `%name` in `s`.
fn ssa_refs(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut refs = vec![];
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            refs.push(s[start..i].to_string());
        } else {
            i += 1;
        }
    }
    refs
}

/// The operands of a terminator line (`yield %a, %b : ...`), in order.
fn terminator_operands(line: &str) -> Vec<String> {
    ssa_refs(line.split(':').next().unwrap_or(""))
}

fn is_loop_header(line: &str) -> bool {
    let rhs = line
        .split_once(" = ")
        .map_or(line, |(_, rhs)| rhs)
        .trim_start();
    rhs.starts_with("for ") || rhs.starts_with("cuda_tile.loop") || rhs.starts_with("loop ")
}

/// Which edges [`def_operands`] records around a loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edges {
    /// Also the back edges, from a loop's block arguments to what `continue`
    /// passes back: a walk reaches earlier iterations.
    All,
    /// No back edges: a walk stays within one iteration.
    SameIteration,
}

/// Maps each SSA value to the values it is computed from. Region-aware: an
/// `if`'s results depend positionally on the operands of its `yield`s, a
/// loop's results on its `break`s (and, for `for`, its `continue`s), and a
/// loop's block arguments on its initial values (`iter_values(%a = %init)`)
/// and, with [`Edges::All`], on what `continue` passes back. `break` and
/// `continue` belong to the innermost loop, even inside an `if`. A walk thus
/// follows tokens through branches and, with [`Edges::All`], around loops.
fn def_operands(mlir: &str, edges: Edges) -> HashMap<String, Vec<String>> {
    struct Region {
        results: Vec<String>,
        block_args: Vec<String>,
        is_loop: bool,
        depth: i32,
    }
    let mut defs: HashMap<String, Vec<String>> = HashMap::new();
    let mut regions: Vec<Region> = Vec::new();
    let mut depth = 0i32;
    for line in mlir.lines() {
        let trimmed = line.trim_start();
        let terminator = ["yield", "continue", "break"]
            .into_iter()
            .find(|t| trimmed.starts_with(&format!("{t} ")) || trimmed == *t);
        if let Some(terminator) = terminator {
            let operands = terminator_operands(trimmed);
            let target = if terminator == "yield" {
                regions.last()
            } else {
                regions.iter().rev().find(|r| r.is_loop)
            };
            if let Some(region) = target {
                for (i, operand) in operands.iter().enumerate() {
                    // A `for` loop's results are its last `continue`'s values;
                    // a `loop`'s are its `break`'s.
                    if let Some(result) = region.results.get(i) {
                        defs.entry(result.clone())
                            .or_default()
                            .push(operand.clone());
                    }
                    if terminator == "continue" && edges == Edges::All {
                        if let Some(arg) = region.block_args.get(i) {
                            defs.entry(arg.clone()).or_default().push(operand.clone());
                        }
                    }
                }
            }
        } else {
            let opens = line.trim_end().ends_with('{') && !trimmed.starts_with('}');
            let (results, rhs) = match line.split_once(" = ") {
                Some((lhs, rhs)) => (
                    lhs.split(',')
                        .map(str::trim)
                        .filter(|s| s.starts_with('%'))
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    rhs,
                ),
                None => (vec![], trimmed),
            };
            let mut block_args = Vec::new();
            if let Some(iter) = rhs.split("iter_values(").nth(1) {
                let iter = iter.split(')').next().unwrap_or("");
                for pair in iter.split(',') {
                    let refs = ssa_refs(pair);
                    if let [arg, init] = refs.as_slice() {
                        defs.entry(arg.clone()).or_default().push(init.clone());
                        block_args.push(arg.clone());
                    }
                }
            }
            let operands = ssa_refs(rhs.split(" iter_values(").next().unwrap_or(rhs));
            for r in &results {
                defs.entry(r.clone()).or_default().extend(operands.clone());
            }
            if opens && (is_loop_header(line) || rhs.trim_start().starts_with("if ")) {
                regions.push(Region {
                    results,
                    block_args,
                    is_loop: is_loop_header(line),
                    depth,
                });
            }
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        while regions.last().is_some_and(|r| depth <= r.depth) {
            regions.pop();
        }
    }
    defs
}

/// Whether `source` transitively depends on `target`.
fn depends_on(defs: &HashMap<String, Vec<String>>, source: &str, target: &str) -> bool {
    let mut stack = vec![source.to_string()];
    let mut seen = HashSet::new();
    while let Some(v) = stack.pop() {
        if v == target {
            return true;
        }
        if seen.insert(v.clone()) {
            if let Some(ops) = defs.get(&v) {
                stack.extend(ops.iter().cloned());
            }
        }
    }
    false
}

/// An operation's ordering tokens: the token it consumes, if any, and the
/// one it produces. `Global` loads, stores and atomics consume none: their
/// acquire/release semantics order them.
#[derive(Debug, Clone)]
struct Tokens {
    input: Option<String>,
    output: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Access,
    Call,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Frame {
    /// The line of the region's header.
    id: usize,
    is_loop: bool,
    /// For an `if`: 0 in the `then` branch, 1 in the `else` branch.
    branch: usize,
}

/// A memory access or foreign call, with its tokens and the branches and
/// loops around it, outermost first.
#[derive(Debug, Clone)]
struct Event {
    kind: Kind,
    tokens: Tokens,
    text: String,
    frames: Vec<Frame>,
    /// Whether a branch around it ends with `continue` or `break`, so the
    /// operations after that branch can only follow it from a later
    /// iteration or after the loop.
    leaves_iteration: bool,
}

/// Ops that access memory and produce a token.
const ACCESS_OPS: &[&str] = &[
    "load_view_tko",
    "store_view_tko",
    "load_ptr_tko",
    "store_ptr_tko",
    "atomic_",
    "multimem_",
];

fn events(mlir: &str) -> Vec<Event> {
    let mut events = Vec::new();
    let mut frames: Vec<(Frame, i32)> = Vec::new();
    // The `if` branches, as (header line, branch), that end with `continue`
    // or `break`.
    let mut leaving: HashSet<(usize, usize)> = HashSet::new();
    let mut depth = 0i32;
    for (n, line) in mlir.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("} else") {
            if let Some((frame, _)) = frames.last_mut() {
                frame.branch += 1;
            }
        }
        let exits = ["continue", "break"]
            .into_iter()
            .any(|t| trimmed.starts_with(&format!("{t} ")) || trimmed == t);
        if let Some((frame, _)) = frames.last().filter(|(f, _)| exits && !f.is_loop) {
            leaving.insert((frame.id, frame.branch));
        }
        let rhs = line.split_once(" = ").map(|(_, rhs)| rhs.trim_start());
        if let Some(rhs) = rhs {
            let lhs = line.split_once(" = ").unwrap().0;
            let output = lhs.split(',').next_back().unwrap().trim().to_string();
            let current: Vec<Frame> = frames.iter().map(|(f, _)| f.clone()).collect();
            if rhs.starts_with("preview$call") {
                let args = rhs.split_once('(').unwrap().1.split(')').next().unwrap();
                events.push(Event {
                    kind: Kind::Call,
                    tokens: Tokens {
                        input: ssa_refs(args).pop(),
                        output,
                    },
                    text: trimmed.to_string(),
                    frames: current,
                    leaves_iteration: false,
                });
            } else if ACCESS_OPS.iter().any(|op| rhs.starts_with(op)) {
                // View ops print `token = %t`; pointer ops (`Global`) print
                // the token as their last operand, typed `token`.
                let input = rhs
                    .split("token = ")
                    .nth(1)
                    .and_then(|r| ssa_refs(r).into_iter().next())
                    .or_else(|| {
                        let (operands, types) = rhs.split_once(" : ")?;
                        let inputs = types.split(" -> ").next()?;
                        inputs
                            .trim()
                            .ends_with("token")
                            .then(|| ssa_refs(operands).pop())
                            .flatten()
                    });
                events.push(Event {
                    kind: Kind::Access,
                    tokens: Tokens { input, output },
                    text: trimmed.to_string(),
                    frames: current,
                    leaves_iteration: false,
                });
            }
        }
        let opens = line.trim_end().ends_with('{') && !trimmed.starts_with('}');
        let header = rhs.unwrap_or(trimmed);
        if opens && (is_loop_header(line) || header.starts_with("if ")) {
            frames.push((
                Frame {
                    id: n,
                    is_loop: is_loop_header(line),
                    branch: 0,
                },
                depth,
            ));
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        while frames.last().is_some_and(|(_, d)| depth <= *d) {
            frames.pop();
        }
    }
    for event in &mut events {
        event.leaves_iteration = event
            .frames
            .iter()
            .any(|f| leaving.contains(&(f.id, f.branch)));
    }
    events
}

/// The frames `a` and `b` share, and whether they are in different branches
/// of one `if`.
fn relation(a: &Event, b: &Event) -> (Vec<Frame>, bool) {
    let mut common = Vec::new();
    for (fa, fb) in a.frames.iter().zip(&b.frames) {
        if fa.id != fb.id {
            break;
        }
        if fa.branch != fb.branch {
            return (common, true);
        }
        common.push(fa.clone());
    }
    (common, false)
}

/// Checks invariants 1 to 4 (see the module docs) for every pair of
/// operations in `mlir`. Returns the number of pairs checked.
#[track_caller]
fn assert_token_invariants(mlir: &str) -> usize {
    let all_edges = def_operands(mlir, Edges::All);
    let same_iteration = def_operands(mlir, Edges::SameIteration);
    let events = events(mlir);
    let mut checked = 0;
    let mut require = |later: &Event, earlier: &Event, edges: Edges, why: &str| {
        // Every access ordered after an earlier one waits on an input token.
        let Some(input) = &later.tokens.input else {
            panic!(
                "missing input token: {why}: {} after {}\n{mlir}",
                later.text, earlier.text
            );
        };
        let defs = match edges {
            Edges::All => &all_edges,
            Edges::SameIteration => &same_iteration,
        };
        checked += 1;
        assert!(
            depends_on(defs, input, &earlier.tokens.output),
            "{why}: `{}` must be ordered after `{}`:\n{mlir}",
            later.text,
            earlier.text
        );
    };
    for (i, a) in events.iter().enumerate() {
        for (j, b) in events.iter().enumerate() {
            if a.kind == Kind::Access && b.kind == Kind::Access {
                continue;
            }
            let (common, exclusive) = relation(a, b);
            if exclusive {
                continue;
            }
            // `b` precedes `a` in program order, in the same iteration of
            // the loops around both, unless `b` leaves that iteration first.
            if j < i {
                let why = match (b.kind, a.kind) {
                    (Kind::Access, Kind::Call) => "1: call after an earlier access",
                    (Kind::Call, Kind::Access) => "2: access after an earlier call",
                    _ => "3: call after an earlier call",
                };
                let edges = if b.leaves_iteration {
                    Edges::All
                } else {
                    Edges::SameIteration
                };
                require(a, b, edges, why);
            }
            // `b` in the previous iteration of a loop around both.
            if j >= i && common.iter().any(|f| f.is_loop) {
                require(a, b, Edges::All, "4: after the previous iteration");
            }
        }
    }
    checked
}

fn count(mlir: &str, kind: Kind) -> usize {
    events(mlir).iter().filter(|e| e.kind == kind).count()
}

/// Compiles `entry`, checks invariants 1 to 4, and checks it has `accesses`
/// memory accesses and `calls` foreign calls (so a test cannot pass by
/// compiling the wrong thing).
#[track_caller]
fn check(ir: &str, accesses: usize, calls: usize) {
    assert_eq!(
        (count(ir, Kind::Access), count(ir, Kind::Call)),
        (accesses, calls),
        "{ir}"
    );
    assert!(assert_token_invariants(ir) > 0, "{ir}");
}

#[test]
fn a_call_in_a_branch_is_ordered_both_ways() {
    common::with_test_stack(|| {
        let ir = compile("branch");
        check(&ir, 3, 1);
        // The branch without a call passes the incoming tokens through: it
        // holds only its `yield`.
        let lines: Vec<&str> = ir.lines().map(str::trim).collect();
        let else_at = lines.iter().position(|l| *l == "} else {").unwrap();
        assert!(lines[else_at + 1].starts_with("yield "), "{ir}");
        assert_eq!(lines[else_at + 2], "}", "{ir}");
    });
}

#[test]
fn calls_in_a_range_loop_are_carried_between_iterations() {
    common::with_test_stack(|| check(&compile("range_loop"), 3, 1));
}

#[test]
fn calls_in_a_while_loop_are_carried_between_iterations() {
    common::with_test_stack(|| check(&compile("while_loop"), 3, 1));
}

#[test]
fn calls_in_a_loop_with_break_are_carried_between_iterations() {
    common::with_test_stack(|| check(&compile("loop_with_break"), 3, 1));
}

/// A `Dim` loop threads its stores through a token accumulator; the call
/// tokens are carried next to it.
#[test]
fn calls_in_a_dim_loop_are_carried_between_iterations() {
    common::with_test_stack(|| {
        let ir = compile_with("dim_loop", vec!["256".into()], &[("out", &[256, 1])]);
        check(&ir, 1, 1);
    });
}

/// The case the first version missed: a store after the call in a loop body
/// comes before the next iteration's call.
#[test]
fn a_store_after_a_call_is_ordered_before_the_next_iterations_call() {
    common::with_test_stack(|| check(&compile("store_after_call"), 3, 1));
}

#[test]
fn a_load_after_a_call_is_ordered_before_the_next_iterations_call() {
    common::with_test_stack(|| check(&compile("load_after_call"), 3, 1));
}

#[test]
fn a_store_in_a_branch_is_ordered_before_a_later_call() {
    common::with_test_stack(|| check(&compile("store_in_branch_then_call"), 3, 1));
}

#[test]
fn a_store_in_a_loop_is_ordered_before_a_later_call() {
    common::with_test_stack(|| check(&compile("store_in_loop_then_call"), 3, 1));
}

#[test]
fn a_store_in_a_helper_is_ordered_before_a_later_call() {
    common::with_test_stack(|| check(&compile("store_in_helper_then_call"), 3, 1));
}

#[test]
fn a_call_is_ordered_with_the_accesses_to_every_tensor() {
    common::with_test_stack(|| {
        let ir = compile_with("two_tensors", vec![], &[("x", &[1]), ("y", &[1])]);
        check(&ir, 5, 1);
    });
}

#[test]
fn a_call_after_an_if_is_ordered_after_both_branches() {
    common::with_test_stack(|| check(&compile("exclusive_branches"), 3, 2));
}

#[test]
fn atomics_are_ordered_with_calls() {
    common::with_test_stack(|| check(&compile("atomic_then_call"), 4, 1));
}

/// Calls inside wrappers that take only tiles, a function and a trait
/// method, are ordered with the caller's accesses in both directions.
#[test]
fn calls_in_tile_only_wrappers_are_ordered_with_the_caller() {
    common::with_test_stack(|| check(&compile("wrappers"), 4, 3));
}

/// A wrapper called in a loop in a branch: the order holds through all
/// three.
#[test]
fn calls_in_a_wrapper_in_a_loop_in_a_branch_are_ordered() {
    common::with_test_stack(|| check(&compile("nested"), 3, 2));
}

/// Reads through a `&Tensor` after a call in an `unsafe {}` block, after a
/// `{}` block holding the call, and after a call in the same loop iteration.
#[test]
fn a_read_after_an_unsafe_block_is_ordered_after_its_call() {
    common::with_test_stack(|| {
        let ir = compile_with(
            "read_after_unsafe_block",
            vec![],
            &[("x", &[1]), ("y", &[1])],
        );
        check(&ir, 3, 1);
    });
}

#[test]
fn a_read_after_a_block_is_ordered_after_its_call() {
    common::with_test_stack(|| {
        let ir = compile_with("read_after_block", vec![], &[("x", &[1]), ("y", &[1])]);
        check(&ir, 3, 1);
    });
}

#[test]
fn a_read_after_a_call_in_a_loop_is_ordered_after_that_call() {
    common::with_test_stack(|| {
        let ir = compile_with(
            "read_after_call_in_loop",
            vec![],
            &[("x", &[1]), ("y", &[1])],
        );
        check(&ir, 3, 1);
    });
}

/// The accesses after a branch that ends with `continue` cannot follow its
/// call in the same iteration; they follow it from the next one.
#[test]
fn a_call_before_continue_is_ordered_with_the_next_iteration() {
    common::with_test_stack(|| {
        let ir = compile_with("call_then_continue", vec![], &[("x", &[1]), ("y", &[1])]);
        check(&ir, 4, 1);
    });
}

/// Invariant 5: the access token only feeds foreign calls, so a `Dim` loop's
/// disjoint stores stay unordered with each other.
#[test]
fn accesses_are_not_serialized_by_the_call_tokens() {
    common::with_test_stack(|| {
        let ir = compile_with(
            "dim_loop_without_calls",
            vec!["256".into()],
            &[("out", &[256, 1])],
        );
        let defs = def_operands(&ir, Edges::All);
        let stores: Vec<Event> = events(&ir)
            .into_iter()
            .filter(|e| e.text.contains("store_view_tko"))
            .collect();
        assert_eq!(stores.len(), 1, "{ir}");
        let store = &stores[0];
        assert!(
            !depends_on(
                &defs,
                store.tokens.input.as_ref().unwrap(),
                &store.tokens.output
            ),
            "a store must not wait for the previous iteration's store:\n{ir}"
        );
    });
}

/// Invariant 6: without foreign declarations, loops carry no extra token.
#[test]
fn kernels_without_foreign_declarations_carry_no_extra_token() {
    common::with_test_stack(|| {
        let ir = compile_in(
            plain_kernels::__module_ast_self,
            "plain_kernels",
            "plain_loop",
            vec![],
            &[("y", &[1])],
        );
        let header = ir.lines().find(|l| is_loop_header(l)).unwrap();
        assert!(!header.contains("token"), "{ir}");
        assert_eq!(ir.matches("make_token").count(), 1, "{ir}");
    });
}

/// The checker itself: the `store_after_call` IR with the call's input
/// token replaced by the kernel's first token, which depends on nothing,
/// must fail it.
#[test]
fn the_checker_rejects_a_missing_edge() {
    common::with_test_stack(|| {
        let ir = compile("store_after_call");
        let call = events(&ir)
            .into_iter()
            .find(|e| e.kind == Kind::Call)
            .unwrap();
        let input = call.tokens.input.unwrap();
        let first_token = ir
            .lines()
            .find(|l| l.contains("= make_token"))
            .and_then(|l| ssa_refs(l).into_iter().next())
            .unwrap();
        let call_line = ir.lines().find(|l| l.contains("preview$call")).unwrap();
        let broken = ir.replace(
            call_line,
            &call_line.replace(&format!("{input})"), &format!("{first_token})")),
        );
        assert_ne!(broken, ir);
        let result = std::panic::catch_unwind(|| assert_token_invariants(&broken));
        assert!(result.is_err(), "the checker accepted:\n{broken}");
    });
}
