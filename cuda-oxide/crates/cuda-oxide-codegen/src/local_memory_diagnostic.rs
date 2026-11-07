/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Post-optimization diagnostics for Rust locals that remain stack-backed.
//!
//! Rust MIR locals are tagged before lowering and exported with an encoded SSA
//! name. The ordinary LLVM `opt` pipeline removes locals that mem2reg/SROA
//! can promote. This module inspects the exact LLVM IR subsequently passed to
//! `llc` and reports only tagged allocas that still exist there.
//!
//! The analysis is intentionally conservative. Dynamic GEP indices and obvious
//! address escapes get specific reasons; every other surviving tagged alloca
//! receives a generic "survived scalar replacement" explanation rather than a
//! guessed cause.
//!
//! Most reports are opt-in (verbose builds). One is on by default: a local
//! indexed by a loop counter in a loop LLVM left rolled. That shape has a
//! one-line fix (`#[unroll]`), it is easy to write without noticing (a
//! `for e in 0..4 { r[e] = .. }` fill), and it costs a stack frame and
//! local-memory traffic in every thread.

use std::collections::{HashMap, HashSet};
use std::path::Path;

const LOCAL_MEMORY_VALUE_PREFIX: &str = "%__cuda_oxide_local_x";
const LOCAL_MEMORY_ALLOCA_PREFIX: &str = "%__cuda_oxide_local_alloca_x";

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalProvenance {
    local_index: usize,
    size_bytes: u64,
    name: String,
    type_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromotionBlocker {
    /// A dynamic index derived from a loop induction variable.
    LoopCounterIndex,
    DynamicIndex,
    AddressEscapes,
    SurvivedScalarReplacement,
}

#[derive(Clone, Debug)]
struct FunctionBody<'a> {
    name: String,
    lines: Vec<&'a str>,
}

#[derive(Clone, Debug)]
struct SurvivingAlloca<'a> {
    value: &'a str,
    provenance: LocalProvenance,
}

/// Which surviving locals to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reporting {
    /// Only locals indexed by the counter of a rolled loop (default builds).
    LoopCounterIndexOnly,
    /// Every surviving tagged local (verbose builds).
    All,
}

/// Inspect optimized LLVM IR and return user-facing warning diagnostics.
pub(crate) fn diagnose_file(path: &Path, reporting: Reporting) -> std::io::Result<Vec<String>> {
    let llvm = std::fs::read_to_string(path)?;
    Ok(diagnose_text(&llvm, reporting))
}

fn diagnose_text(llvm: &str, reporting: Reporting) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for function in function_bodies(llvm) {
        let function_name = demangle_function_name(&function.name);
        let allocas = surviving_allocas(&function);
        if allocas.is_empty() {
            continue;
        }
        let roots: Vec<&str> = allocas.iter().map(|alloca| alloca.value).collect();
        // one pass over the function for every local, so a kernel with many
        // surviving locals costs no more than one with a single local
        let uses = LocalUses::new(&function.lines, &roots);
        // built on first use and shared by every local of the function
        let mut index_analysis = None;
        for (local, alloca) in allocas.iter().enumerate() {
            let blocker = uses.blocker(local, &function.lines, &mut index_analysis);
            if reporting == Reporting::LoopCounterIndexOnly
                && blocker != PromotionBlocker::LoopCounterIndex
            {
                continue;
            }
            diagnostics.push(format_diagnostic(
                &function_name,
                &alloca.provenance,
                blocker,
            ));
        }
    }
    diagnostics
}

/// Render the containing function readably in the warning.
///
/// Kernel entry points already carry their exported source name, but a
/// non-inlined device function survives optimization under its mangled Rust
/// symbol. Demangle those (dropping the disambiguating hash) so the warning
/// points at the source function; anything unmangled passes through untouched.
fn demangle_function_name(name: &str) -> String {
    match rustc_demangle::try_demangle(name) {
        Ok(demangled) => format!("{demangled:#}"),
        Err(_) => name.to_string(),
    }
}

fn function_bodies(llvm: &str) -> Vec<FunctionBody<'_>> {
    let mut functions = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_lines = Vec::new();

    for line in llvm.lines() {
        if current_name.is_none() {
            if let Some(name) = defined_function_name(line) {
                current_name = Some(name);
                current_lines.push(line);
            }
            continue;
        }

        current_lines.push(line);
        if line.trim() == "}" {
            functions.push(FunctionBody {
                name: current_name.take().unwrap(),
                lines: std::mem::take(&mut current_lines),
            });
        }
    }

    functions
}

fn defined_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("define ") {
        return None;
    }
    let at = trimmed.find('@')? + 1;
    let rest = &trimmed[at..];
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some(quoted[..end].to_string());
    }
    let end = rest.find('(')?;
    Some(rest[..end].trim().to_string())
}

fn surviving_allocas<'a>(function: &FunctionBody<'a>) -> Vec<SurvivingAlloca<'a>> {
    function
        .lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if !trimmed.contains(" = alloca ") {
                return None;
            }
            let value = trimmed.split_once(" = ")?.0.trim();
            let provenance = decode_provenance_name(value)?;
            Some(SurvivingAlloca { value, provenance })
        })
        .collect()
}

fn decode_provenance_name(value: &str) -> Option<LocalProvenance> {
    let encoded = value
        .strip_prefix(LOCAL_MEMORY_VALUE_PREFIX)
        .or_else(|| value.strip_prefix(LOCAL_MEMORY_ALLOCA_PREFIX))?;
    // The exporter terminates the hex payload with a `_` sentinel. LLVM
    // uniques colliding value names by appending bare digits (a second inlined
    // copy of the same local becomes `<name>1`), and digits are valid hex, so
    // without the sentinel a uniquing suffix would silently extend the payload
    // and garble the report. Key the decode on the sentinel: hex digits up to
    // the first `_`, and reject names that lack it.
    let hex_len = encoded
        .bytes()
        .take_while(|byte| byte.is_ascii_hexdigit())
        .count();
    if encoded.as_bytes().get(hex_len) != Some(&b'_') {
        return None;
    }
    let encoded = &encoded[..hex_len];
    if encoded.is_empty() || encoded.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for index in (0..encoded.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&encoded[index..index + 2], 16).ok()?);
    }
    let decoded = String::from_utf8(bytes).ok()?;
    let mut fields = decoded.splitn(4, '\t');
    Some(LocalProvenance {
        local_index: fields.next()?.parse().ok()?,
        size_bytes: fields.next()?.parse().ok()?,
        name: fields.next()?.to_string(),
        type_name: fields.next()?.to_string(),
    })
}

/// How the function uses each surviving local: the dynamic indices of GEPs
/// into it and whether its address escapes.
struct LocalUses<'a> {
    dynamic_indices: Vec<Vec<&'a str>>,
    escapes: Vec<bool>,
}

impl<'a> LocalUses<'a> {
    fn new(lines: &[&'a str], roots: &[&'a str]) -> Self {
        let mut users: HashMap<&str, Vec<&str>> = HashMap::new();
        for line in lines {
            let Some((lhs, rhs)) = assignment(line) else {
                continue;
            };
            if !is_pointer_identity_or_derivation(rhs) {
                continue;
            }
            for operand in value_tokens(rhs) {
                users.entry(operand).or_default().push(lhs);
            }
        }

        // the locals each pointer is derived from, through GEPs, casts, phis,
        // and selects; a pointer usually has one, and a phi or select of two
        // locals' pointers has both
        let mut owners: HashMap<&'a str, Vec<usize>> = HashMap::new();
        let mut pending = Vec::new();
        for (local, &root) in roots.iter().enumerate() {
            owners.entry(root).or_default().push(local);
            pending.push((root, local));
        }
        while let Some((value, local)) = pending.pop() {
            for &user in users.get(value).into_iter().flatten() {
                let user_owners = owners.entry(user).or_default();
                if !user_owners.contains(&local) {
                    user_owners.push(local);
                    pending.push((user, local));
                }
            }
        }

        let mut dynamic_indices = vec![Vec::new(); roots.len()];
        let mut escapes = vec![false; roots.len()];
        for line in lines {
            if let Some((base, indices)) = gep_dynamic_indices(line, &owners) {
                for &local in &owners[base] {
                    dynamic_indices[local].extend(&indices);
                }
            }
            for token in escaping_values(line) {
                for &local in owners.get(token).into_iter().flatten() {
                    escapes[local] = true;
                }
            }
        }

        Self {
            dynamic_indices,
            escapes,
        }
    }

    fn blocker(
        &self,
        local: usize,
        lines: &[&'a str],
        index_analysis: &mut Option<IndexAnalysis<'a>>,
    ) -> PromotionBlocker {
        let dynamic_indices = &self.dynamic_indices[local];
        if !dynamic_indices.is_empty() {
            // unrolling helps only if it makes every index constant and
            // nothing else pins the local: one index that also mixes in a
            // runtime value, or an escaping address, keeps it in memory
            if !self.escapes[local] {
                let analysis = index_analysis.get_or_insert_with(|| IndexAnalysis::new(lines));
                if dynamic_indices
                    .iter()
                    .all(|index| analysis.counter_derived(index))
                {
                    return PromotionBlocker::LoopCounterIndex;
                }
            }
            return PromotionBlocker::DynamicIndex;
        }

        if self.escapes[local] {
            return PromotionBlocker::AddressEscapes;
        }

        PromotionBlocker::SurvivedScalarReplacement
    }
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let (lhs, rhs) = trimmed.split_once(" = ")?;
    lhs.starts_with('%').then_some((lhs.trim(), rhs.trim()))
}

fn is_pointer_identity_or_derivation(rhs: &str) -> bool {
    rhs.starts_with("getelementptr ")
        || rhs.starts_with("bitcast ")
        || rhs.starts_with("addrspacecast ")
        || rhs.starts_with("freeze ")
        || rhs.starts_with("phi ")
        || rhs.starts_with("select ")
}

/// The base pointer and non-constant index operands of a GEP into one of
/// `pointers`.
fn gep_dynamic_indices<'a, V>(
    line: &'a str,
    pointers: &HashMap<&str, V>,
) -> Option<(&'a str, Vec<&'a str>)> {
    let (_, rhs) = assignment(line)?;
    if !rhs.starts_with("getelementptr ") {
        return None;
    }

    let base = value_tokens(rhs).find(|token| pointers.contains_key(token))?;
    let fields: Vec<&str> = rhs.split(',').collect();
    let base_field = fields
        .iter()
        .position(|field| value_tokens(field).any(|token| token == base))?;
    let indices: Vec<&str> = fields[base_field + 1..]
        .iter()
        .flat_map(|field| value_tokens(field))
        .collect();
    (!indices.is_empty()).then_some((base, indices))
}

/// One SSA definition through which an index can be computed.
struct IntegerDefinition<'a> {
    opcode: &'a str,
    /// The value operands (`%` names). A phi's are its incoming values,
    /// without the incoming block labels.
    operands: Vec<&'a str>,
    /// Whether an operand is an integer literal (an `add %i, 1` step).
    has_literal_operand: bool,
    /// Whether this widens an `i1`: Rust's `Range::next` advances its counter
    /// by `zext i1 %has_next`, not by a literal 1.
    is_bool_zext: bool,
    /// The block defining this value, without its `%`.
    block: &'a str,
    /// A phi's `(incoming value, incoming block)` pairs, `%` stripped from the
    /// block; empty for every other opcode.
    incoming: Vec<(&'a str, &'a str)>,
}

/// Opcodes an index can be computed through on its way from a loop counter.
/// Anything else (a load, a call, a compare) ends the search: the value is
/// data, not a counter.
const INDEX_ARITHMETIC: &[&str] = &[
    "phi", "add", "sub", "mul", "shl", "lshr", "ashr", "or", "and", "xor", "zext", "sext", "trunc",
    "select",
];

/// A conditional branch's condition and its `[true, false]` targets.
struct ConditionalBranch<'a> {
    condition: &'a str,
    targets: [&'a str; 2],
}

/// Which values of one function are computed only from the counters of
/// constant-trip loops LLVM left rolled. Built once per function and shared by
/// every local in it; each step is linear in the function's size.
struct IndexAnalysis<'a> {
    definitions: HashMap<&'a str, IntegerDefinition<'a>>,
    /// Members of recurrences that are constant-trip loop counters.
    counters: HashSet<&'a str>,
    /// Members of every other recurrence (a hash, a data-stepped position).
    other_recurrences: HashSet<&'a str>,
    /// Per counter loop header, the `[preheader, latch]` blocks (sorted) of a
    /// counter phi there: a phi with the same two edges picks its value by
    /// iteration (first or later), which unrolling resolves.
    counter_headers: HashMap<&'a str, Vec<[&'a str; 2]>>,
    /// Memoized [`Self::counter_derived`] results.
    derived: HashMap<&'a str, bool>,
}

impl<'a> IndexAnalysis<'a> {
    fn new(lines: &[&'a str]) -> Self {
        let mut definitions = HashMap::new();
        let mut compares = HashMap::new();
        // a block ends in at most one conditional branch
        let mut branches = HashMap::new();
        // the entry block may be unlabeled; nothing branches back to it
        let mut block = "";
        for line in lines {
            if let Some(label) = block_label(line) {
                block = label;
                continue;
            }
            if let Some(branch) = conditional_branch(line) {
                branches.insert(block, branch);
                continue;
            }
            let Some((lhs, rhs)) = assignment(line) else {
                continue;
            };
            let Some(opcode) = rhs.split_whitespace().next() else {
                continue;
            };
            if opcode == "icmp" {
                compares.insert(lhs, (value_tokens(rhs).collect(), has_integer_literal(rhs)));
                continue;
            }
            if !INDEX_ARITHMETIC.contains(&opcode) {
                continue;
            }

            let incoming = if opcode == "phi" {
                phi_incoming(rhs)
            } else {
                Vec::new()
            };
            let operands = if opcode == "phi" {
                incoming
                    .iter()
                    .map(|(value, _)| *value)
                    .filter(|value| value.starts_with('%'))
                    .collect()
            } else {
                value_tokens(rhs).collect()
            };
            definitions.insert(
                lhs,
                IntegerDefinition {
                    opcode,
                    operands,
                    has_literal_operand: has_integer_literal(rhs),
                    is_bool_zext: rhs.starts_with("zext i1 "),
                    block,
                    incoming,
                },
            );
        }

        let mut counters = HashSet::new();
        let mut other_recurrences = HashSet::new();
        let mut counter_headers: HashMap<&str, Vec<[&str; 2]>> = HashMap::new();
        for recurrence in recurrences(&definitions) {
            if !is_loop_counter(&recurrence, &definitions, &compares, &branches) {
                other_recurrences.extend(recurrence);
                continue;
            }
            for member in &recurrence {
                let definition = &definitions[member];
                if let [(_, first), (_, second)] = definition.incoming[..] {
                    let mut edges = [first, second];
                    edges.sort_unstable();
                    counter_headers
                        .entry(definition.block)
                        .or_default()
                        .push(edges);
                }
            }
            counters.extend(recurrence);
        }

        Self {
            definitions,
            counters,
            other_recurrences,
            counter_headers,
            derived: HashMap::new(),
        }
    }

    /// Whether `root` is computed only from constant-trip loop counters and
    /// literals, so fully unrolling those loops turns it into a constant.
    fn counter_derived(&mut self, root: &'a str) -> bool {
        if let Some(known) = self.known(root) {
            return known;
        }

        // recurrences are classified up front, so what remains is acyclic;
        // an explicit stack, because an unrolled body can chain tens of
        // thousands of integer ops and recursion would overflow rustc's stack
        let mut stack: Vec<(&'a str, usize)> = vec![(root, 0)];
        while let Some(top) = stack.last_mut() {
            let value = top.0;
            let definition = &self.definitions[value];
            if top.1 == 0 && definition.opcode == "phi" && !self.iteration_phi(definition) {
                // the phi picks between its values by a runtime branch, so its
                // result stays dynamic even when every incoming value is not
                self.derived.insert(value, false);
                stack.pop();
                continue;
            }
            let operands = &definition.operands;
            if let Some(&operand) = operands.get(top.1) {
                top.1 += 1;
                if self.known(operand).is_none() {
                    stack.push((operand, 0));
                }
                continue;
            }

            // a phi or select of literals alone is chosen by control flow or
            // data, which unrolling does not fix
            let derived = !operands.is_empty()
                && operands
                    .iter()
                    .all(|operand| self.known(operand) == Some(true));
            self.derived.insert(value, derived);
            stack.pop();
        }
        self.derived[root]
    }

    /// Whether a phi outside any recurrence picks its value only by loop
    /// iteration: it has a single incoming edge, or it sits in a counter loop's
    /// header with exactly that counter's preheader and latch edges.
    fn iteration_phi(&self, phi: &IntegerDefinition<'a>) -> bool {
        match phi.incoming[..] {
            [_] => true,
            [(_, first), (_, second)] => {
                let mut edges = [first, second];
                edges.sort_unstable();
                self.counter_headers
                    .get(phi.block)
                    .is_some_and(|headers| headers.contains(&edges))
            }
            _ => false,
        }
    }

    fn known(&self, value: &str) -> Option<bool> {
        if self.counters.contains(value) {
            Some(true)
        } else if self.other_recurrences.contains(value) || !self.definitions.contains_key(value) {
            // an argument, a load, a call, a compare, or another recurrence
            Some(false)
        } else {
            self.derived.get(value).copied()
        }
    }
}

/// Every recurrence among `definitions`: the strongly connected components of
/// the operand graph that contain a cycle. Tarjan's algorithm with an explicit
/// stack, so it is linear and cannot overflow on a long chain.
fn recurrences<'a>(definitions: &HashMap<&'a str, IntegerDefinition<'a>>) -> Vec<Vec<&'a str>> {
    // (discovery index, lowest index reachable) per visited value
    let mut order: HashMap<&'a str, (usize, usize)> = HashMap::new();
    let mut on_stack = HashSet::new();
    let mut component_stack = Vec::new();
    let mut found = Vec::new();

    for &root in definitions.keys() {
        if order.contains_key(root) {
            continue;
        }
        let mut call_stack: Vec<(&'a str, usize)> = vec![(root, 0)];
        order.insert(root, (order.len(), order.len()));
        component_stack.push(root);
        on_stack.insert(root);

        while let Some(top) = call_stack.last_mut() {
            let value = top.0;
            if let Some(&operand) = definitions[value].operands.get(top.1) {
                top.1 += 1;
                if !definitions.contains_key(operand) {
                    continue;
                }
                match order.get(operand) {
                    None => {
                        order.insert(operand, (order.len(), order.len()));
                        component_stack.push(operand);
                        on_stack.insert(operand);
                        call_stack.push((operand, 0));
                    }
                    Some(&(operand_index, _)) if on_stack.contains(operand) => {
                        let entry = order.get_mut(value).unwrap();
                        entry.1 = entry.1.min(operand_index);
                    }
                    Some(_) => {}
                }
                continue;
            }

            call_stack.pop();
            let (index, low) = order[value];
            if let Some(&(parent, _)) = call_stack.last() {
                let entry = order.get_mut(parent).unwrap();
                entry.1 = entry.1.min(low);
            }
            if low == index {
                let mut component = Vec::new();
                while let Some(member) = component_stack.pop() {
                    on_stack.remove(member);
                    component.push(member);
                    if member == value {
                        break;
                    }
                }
                if component.len() > 1 || definitions[value].operands.contains(&value) {
                    found.push(component);
                }
            }
        }
    }
    found
}

/// Whether a recurrence is the counter of a loop with a compile-time trip
/// count, the only kind `#[unroll]` can turn into constants:
///
/// - each of its phis is a loop-header phi with two edges: a literal start,
///   not a runtime value, and the next value carried back by a back-edge
///   branch (one that ends the block feeding the phi and can jump back to it).
///   A phi that merges paths inside the loop lets runtime data pick the step;
/// - a literal compare of it alone ends the loop: it is the condition of each
///   such back-edge branch, or it is joined with an early exit by an `or` the
///   loop exits on, or by an `and` the loop continues on. An `if i == 32`
///   inside the body, or an exit that also needs a runtime condition, does not
///   bound the trip count;
/// - it is made only of phis and counter-like steps: a literal `add`, or the
///   widened-`i1` step of Rust's range desugaring, where the `i1` is a literal
///   compare of the counter that must hold for the back edge to be taken, so
///   the counter advances on every iteration. Any other opcode, such as the
///   `mul` of a `h * a + c` hash, or a step by a runtime bool or by a compare
///   that can stop it short of the bound, makes it data unrolling would not
///   fold.
fn is_loop_counter(
    recurrence: &[&str],
    definitions: &HashMap<&str, IntegerDefinition<'_>>,
    compares: &HashMap<&str, (Vec<&str>, bool)>,
    branches: &HashMap<&str, ConditionalBranch<'_>>,
) -> bool {
    let members: HashSet<&str> = recurrence.iter().copied().collect();
    // a compare of the recurrence, and nothing else, against a literal
    let counter_test = |condition: &str| {
        compares
            .get(condition)
            .is_some_and(|(operands, has_literal)| {
                *has_literal
                    && !operands.is_empty()
                    && operands.iter().all(|operand| members.contains(operand))
            })
    };
    let phis: Vec<&IntegerDefinition<'_>> = recurrence
        .iter()
        .map(|member| &definitions[member])
        .filter(|definition| definition.opcode == "phi")
        .collect();
    if phis.is_empty() {
        return false;
    }

    // every phi of the recurrence must be a loop-header phi: a literal start
    // and one value carried back by a back-edge branch. A phi that merges
    // in-loop paths (`if ready() { i += 1 } else { i += 2 }`) or resets the
    // counter on one (`if ready() { i = 0 }`) lets runtime data decide the
    // trip count, even though every step and start is a literal
    let mut back_edges = Vec::with_capacity(phis.len());
    for phi in &phis {
        let [first, second] = phi.incoming[..] else {
            return false;
        };
        let ((_, from), (start, _)) = match (members.contains(first.0), members.contains(second.0))
        {
            (true, false) => (first, second),
            (false, true) => (second, first),
            _ => return false,
        };
        let Some(branch) = branches.get(from) else {
            return false;
        };
        let jumps_back = branch
            .targets
            .iter()
            .filter(|target| **target == phi.block)
            .count();
        if start.parse::<i64>().is_err() || jumps_back != 1 {
            return false;
        }
        back_edges.push((branch, phi.block));
    }

    // whether `branch`, which can jump back to `header`, leaves the loop once
    // the literal compare says so, whatever any other condition says
    let bounds = |branch: &ConditionalBranch<'_>, header: &str| {
        let continues_on_true = branch.targets[0] == header;
        if counter_test(branch.condition) {
            return true;
        }
        definitions.get(branch.condition).is_some_and(|condition| {
            let joins_test = condition
                .operands
                .iter()
                .any(|operand| counter_test(operand));
            // `br (done | early), exit, header` exits once `done` holds;
            // `br (more & ready), header, exit` stops once `more` fails
            match condition.opcode {
                "or" => joins_test && !continues_on_true,
                "and" => joins_test && continues_on_true,
                _ => false,
            }
        })
    };
    if !back_edges
        .iter()
        .all(|(branch, header)| bounds(branch, header))
    {
        return false;
    }

    // the conditions every back edge needs: `br %more, header, exit` and
    // `br (%more & ready), header, exit` both iterate only while `%more` holds
    let mut continue_conditions = HashSet::new();
    for (branch, header) in &back_edges {
        if branch.targets[0] != *header {
            continue;
        }
        continue_conditions.insert(branch.condition);
        if let Some(condition) = definitions.get(branch.condition)
            && condition.opcode == "and"
        {
            continue_conditions.extend(condition.operands.iter().copied());
        }
    }
    let is_step = |definition: &IntegerDefinition<'_>| {
        matches!(definition.opcode, "add" | "sub" | "or")
            && (definition.has_literal_operand
                || definition.operands.iter().any(|operand| {
                    definitions.get(operand).is_some_and(|widened| {
                        widened.is_bool_zext
                            && widened.operands.first().is_some_and(|condition| {
                                continue_conditions.contains(condition) && counter_test(condition)
                            })
                    })
                }))
    };
    let only_counter_ops = recurrence.iter().all(|member| {
        let definition = &definitions[member];
        definition.opcode == "phi" || is_step(definition)
    });
    let steps = recurrence
        .iter()
        .any(|member| is_step(&definitions[member]));
    only_counter_ops && steps
}

/// The label a line opens a basic block with (`body:`, `5:  ; preds = ..`).
fn block_label(line: &str) -> Option<&str> {
    if line.starts_with(char::is_whitespace)
        || line.starts_with(';')
        || line.starts_with("define ")
        || line.starts_with('}')
    {
        return None;
    }
    let (label, _) = line.split_once(':')?;
    let label = label.trim_matches('"');
    (!label.is_empty() && !label.contains(char::is_whitespace)).then_some(label)
}

/// `br i1 %cond, label %a, label %b`.
fn conditional_branch(line: &str) -> Option<ConditionalBranch<'_>> {
    let rest = line.trim_start().strip_prefix("br i1 ")?;
    let mut fields = rest.split(',').map(str::trim);
    let condition = fields.next()?;
    let mut target = || {
        fields
            .next()?
            .strip_prefix("label %")
            .map(|label| label.trim_matches('"'))
    };
    Some(ConditionalBranch {
        condition,
        targets: [target()?, target()?],
    })
}

/// A phi's `[ value, %block ]` pairs.
fn phi_incoming(rhs: &str) -> Vec<(&str, &str)> {
    rhs.split('[')
        .skip(1)
        .filter_map(|pair| {
            let pair = pair.split(']').next()?;
            let (value, block) = pair.split_once(',')?;
            let block = block.trim().trim_start_matches('%').trim_matches('"');
            Some((value.trim(), block))
        })
        .collect()
}

fn has_integer_literal(rhs: &str) -> bool {
    rhs.split(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '[' | ']'))
        .any(|token| !token.is_empty() && token.parse::<i64>().is_ok())
}

/// The values a line lets escape: passed to a non-intrinsic call, returned,
/// converted to an integer, or stored as data.
fn escaping_values(line: &str) -> Vec<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with(';') || trimmed.contains(" = getelementptr ") {
        return Vec::new();
    }

    if ((trimmed.starts_with("call ") || trimmed.contains(" = call "))
        && !trimmed.contains("@llvm."))
        || trimmed.starts_with("ret ")
        || trimmed.contains(" = ptrtoint ")
    {
        return value_tokens(trimmed).collect();
    }

    if let Some(store) = trimmed.strip_prefix("store ")
        && let Some(stored_value) = store.split(',').next()
    {
        return value_tokens(stored_value).collect();
    }

    Vec::new()
}

fn value_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '(' | ')' | '[' | ']'))
        .filter(|token| token.starts_with('%'))
}

fn format_diagnostic(
    function: &str,
    provenance: &LocalProvenance,
    blocker: PromotionBlocker,
) -> String {
    let binding = if provenance.name.is_empty() {
        format!("_{}", provenance.local_index)
    } else {
        provenance.name.clone()
    };
    let bytes = if provenance.size_bytes == 1 {
        "1 byte".to_string()
    } else {
        format!("{} bytes", provenance.size_bytes)
    };
    let mut message = format!(
        "warning: local `{binding}` (`{}`, {bytes}) in `{function}` could not be promoted to registers",
        provenance.type_name
    );

    match blocker {
        PromotionBlocker::LoopCounterIndex => {
            message.push_str(
                "\n  = note: indexed by the counter of a loop LLVM did not unroll, so the array stays in local memory",
            );
            message.push_str(
                "\n  = help: add `#[unroll]` to that loop so every index becomes a constant",
            );
        }
        PromotionBlocker::DynamicIndex => {
            message.push_str(
                "\n  = note: indexed by a non-constant value; the allocation survives LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: use constant indexing or a small match/select when the index range is bounded",
            );
        }
        PromotionBlocker::AddressEscapes => {
            message.push_str(
                "\n  = note: the allocation's address escapes as a first-class pointer, preventing scalar replacement",
            );
            message.push_str(
                "\n  = help: keep accesses local to the value when possible so LLVM can promote the storage",
            );
        }
        PromotionBlocker::SurvivedScalarReplacement => {
            message.push_str(
                "\n  = note: the allocation survives mem2reg and LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: inspect dynamic indexing, pointer escapes, volatile accesses, or aliasing around this local; large aggregates are also harder to promote because scalar replacement must split every access into independent scalars",
            );
        }
    }

    message
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(provenance: &str) -> String {
        let mut output = String::new();
        for byte in provenance.as_bytes() {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output.push('_');
        output
    }

    fn alloca_name(provenance: &str) -> String {
        format!("{LOCAL_MEMORY_VALUE_PREFIX}{}", encoded(provenance))
    }

    #[test]
    fn dynamic_index_reports_the_rust_local() {
        let name = alloca_name("3\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %idx) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  %value = load i32, ptr %elt, align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `scratch` (`[u32; 4]`, 16 bytes)"));
        assert!(diagnostics[0].contains("indexed by a non-constant value"));
    }

    /// The rolled `for e in 0..4 { r[e] = .. }` fill, as `opt` leaves it
    fn rolled_counter_loop(name: &str) -> String {
        format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %next = phi i64 [ 1, %entry ], [ %step, %body ]\n  %e = phi i64 [ 0, %entry ], [ %next, %body ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %e\n  store float 1.0, ptr %slot, align 4\n  %step = add nuw nsw i64 %next, 1\n  %done = icmp eq i64 %step, 5\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        )
    }

    #[test]
    fn rolled_loop_counter_index_is_reported_by_default() {
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let diagnostics =
            diagnose_text(&rolled_counter_loop(&name), Reporting::LoopCounterIndexOnly);
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert!(diagnostics[0].contains("local `r` (`[f32; 4]`, 16 bytes)"));
        assert!(diagnostics[0].contains("counter of a loop"));
        // full unrolling is the bare attribute; `cuda-macros` takes only a
        // partial-unroll factor, and has no unroll budget the help could raise
        let attributes: Vec<&str> = diagnostics[0]
            .match_indices("#[")
            .map(|(start, _)| {
                let rest = &diagnostics[0][start..];
                &rest[..=rest.find(']').unwrap()]
            })
            .collect();
        assert_eq!(attributes, ["#[unroll]"], "{}", diagnostics[0]);
    }

    #[test]
    fn range_next_counter_steps_by_a_widened_bool() {
        // `for k in 0..8` as opt leaves it: `Range::next` adds `zext i1`
        let name = alloca_name("30\t128\tv\t[F32x4; 8]");
        let llvm = format!(
            "define void @softmax() {{\nentry:\n  {name} = alloca [8 x {{ [4 x float] }}], align 16\n  br label %head\n\nhead:\n  %next = phi i64 [ 1, %entry ], [ %step, %latch ]\n  %k = phi i64 [ 0, %entry ], [ %next, %latch ]\n  %slot = getelementptr inbounds nuw [16 x i8], ptr {name}, i64 %k\n  store float 1.0, ptr %slot, align 16\n  br label %latch\n\nlatch:\n  %more = icmp ult i64 %next, 8\n  %inc = zext i1 %more to i64\n  %step = add nuw nsw i64 %next, %inc\n  br i1 %more, label %head, label %exit\n\nexit:\n  ret void\n}}\n"
        );
        assert_eq!(
            diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).len(),
            1
        );
    }

    #[test]
    fn data_dependent_recurrence_is_not_a_loop_counter() {
        // an index that steps by loaded data stays dynamic after unrolling
        let name = alloca_name("3\t16\tring\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(ptr %deltas) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  br label %body\n\nbody:\n  %pos = phi i64 [ 0, %entry ], [ %pos.next, %body ]\n  %slot.index = and i64 %pos, 3\n  %slot = getelementptr [4 x i32], ptr {name}, i64 0, i64 %slot.index\n  store i32 1, ptr %slot, align 4\n  %delta = load i64, ptr %deltas, align 8\n  %pos.next = add i64 %pos, %delta\n  %done = icmp eq i64 %pos.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
    }

    #[test]
    fn runtime_trip_count_is_not_reported_by_default() {
        // `for i in 0..n` with `n` a kernel argument: `#[unroll]` cannot make
        // `arr[i]` constant, so the default build stays quiet
        let name = alloca_name("3\t16\tarr\t[u32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i64 %n)")
            .replace("icmp eq i64 %step, 5", "icmp eq i64 %step, %n");
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn literal_compare_off_the_exit_branch_is_not_a_bound() {
        // `if i == 0 { .. }` inside `for i in 0..n`: the literal compare feeds
        // data, not the loop's exit test, so the trip count is still runtime
        let name = alloca_name("3\t16\tarr\t[u32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i64 %n)")
            .replace("icmp eq i64 %step, 5", "icmp eq i64 %step, %n")
            .replace(
                "  store float 1.0, ptr %slot, align 4\n",
                "  %first = icmp eq i64 %next, 1\n  %value = select i1 %first, float 1.0, float 2.0\n  store float %value, ptr %slot, align 4\n",
            );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn long_index_chains_do_not_overflow_the_stack() {
        // a fully unrolled hash can feed an index through a very long chain of
        // integer ops; the search must not recurse once per link
        let name = alloca_name("3\t16\tt\t[u32; 4]");
        let mut body = String::from("  %v0 = add i64 %seed, 1\n");
        for link in 1..100_000 {
            body.push_str(&format!("  %v{link} = xor i64 %v{}, 7\n", link - 1));
        }
        let llvm = format!(
            "define void @kernel(i64 %seed) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n{body}  %slot = getelementptr [4 x i32], ptr {name}, i64 0, i64 %v99999\n  store i32 1, ptr %slot, align 4\n  ret void\n}}\n"
        );
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn hash_recurrence_is_not_a_loop_counter() {
        // `h = h * a + c` steps by a literal but is data: its `mul` keeps every
        // unrolled index dynamic
        let name = alloca_name("3\t16\tt\t[f32; 256]");
        let llvm = format!(
            "define void @kernel(i32 %seed) {{\nentry:\n  {name} = alloca [256 x float], align 4\n  br label %body\n\nbody:\n  %h = phi i32 [ %seed, %entry ], [ %h.next, %body ]\n  %n = phi i32 [ 0, %entry ], [ %n.next, %body ]\n  %scaled = mul i32 %h, 1664525\n  %h.next = add i32 %scaled, 1013904223\n  %top = lshr i32 %h.next, 24\n  %slot.index = zext i32 %top to i64\n  %slot = getelementptr [256 x float], ptr {name}, i64 0, i64 %slot.index\n  %value = load float, ptr %slot, align 4\n  %n.next = add i32 %n, 1\n  %done = icmp eq i32 %n.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
    }

    #[test]
    fn counter_plus_runtime_offset_is_not_a_loop_counter() {
        // `t[(i + offset) & 3]`: unrolling fixes `i` but not `offset`
        let name = alloca_name("3\t16\tarr\t[u32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i64 %offset)")
            .replace(
                "ptr {name}, i64 %e\n".replace("{name}", &name).as_str(),
                format!("ptr {name}, i64 %wrapped\n").as_str(),
            )
            .replace(
                "  %slot = ",
                "  %shifted = add i64 %e, %offset\n  %wrapped = and i64 %shifted, 3\n  %slot = ",
            );
        assert!(llvm.contains("%wrapped\n"), "{llvm}");
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        let verbose = diagnose_text(&llvm, Reporting::All);
        assert!(
            verbose[0].contains("indexed by a non-constant value"),
            "{verbose:?}"
        );
    }

    #[test]
    fn literal_branch_inside_a_runtime_loop_is_not_a_bound() {
        // `for i in 0..n { if i == 32 { .. } }`: the literal compare decides a
        // branch, but not the loop's back edge, so the trip count is runtime
        let name = alloca_name("3\t16\tarr\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %n) {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %i\n  store float 1.0, ptr %slot, align 4\n  %is32 = icmp eq i64 %i, 32\n  br i1 %is32, label %then, label %latch\n\nthen:\n  call void @llvm.nvvm.barrier0()\n  br label %latch\n\nlatch:\n  %i.next = add nuw i64 %i, 1\n  %done = icmp eq i64 %i.next, %n\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn runtime_start_is_not_a_constant_trip_count() {
        // `for e in start..4`: a literal end alone leaves the trip count runtime
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i64 %start)")
            .replace("[ 1, %entry ]", "[ %start, %entry ]");
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
    }

    #[test]
    fn long_recurrences_are_checked_in_linear_time() {
        // an unrolled hash over many state words: each link reads one word of
        // the previous iteration's state, so every phi closes its own long
        // cycle, and a search that re-walks each cycle it closes is quadratic
        let name = alloca_name("3\t16\tt\t[u32; 4]");
        let links = 20_000;
        let last = links - 1;
        let mut phis = String::new();
        let mut body = String::from("  %v0 = xor i64 %h0, 7\n");
        for link in 0..links {
            phis.push_str(&format!(
                "  %h{link} = phi i64 [ 1, %entry ], [ %v{last}, %body ]\n"
            ));
            if link > 0 {
                body.push_str(&format!("  %v{link} = xor i64 %v{}, %h{link}\n", link - 1));
            }
        }
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  br label %body\n\nbody:\n{phis}{body}  %idx = and i64 %v{last}, 3\n  %slot = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  store i32 1, ptr %slot, align 4\n  %done = icmp eq i64 %v{last}, 9\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        let start = std::time::Instant::now();
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        // generous for slow CI and debug builds; the quadratic search takes
        // far longer than this on the same input
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn constant_bound_loop_with_an_early_exit_is_a_loop_counter() {
        // `for e in 0..4 { if x[e] < 0.0 { break } r[e] = .. }`: the back edge
        // tests `or i1` of the literal bound and the break condition
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i1 %early)")
            .replace(
                "br i1 %done, label %exit, label %body",
                "%quit = or i1 %done, %early\n  br i1 %quit, label %exit, label %body",
            );
        assert!(llvm.contains("%quit = or i1"), "{llvm}");
        assert_eq!(
            diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).len(),
            1
        );
    }

    #[test]
    fn exit_that_also_needs_a_runtime_condition_is_not_a_bound() {
        // `while !(e >= 4 && ready()) { .. }`: the loop exits on an `and`, so
        // it can run past the literal bound while `ready` is false; the same
        // holds for a back edge taken on `more || keep`
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let exits_on_and = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i1 %ready)")
            .replace(
                "br i1 %done, label %exit, label %body",
                "%quit = and i1 %done, %ready\n  br i1 %quit, label %exit, label %body",
            );
        let continues_on_or = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i1 %keep)")
            .replace(
                "%done = icmp eq i64 %step, 5\n  br i1 %done, label %exit, label %body",
                "%more = icmp ult i64 %step, 5\n  %stay = or i1 %more, %keep\n  br i1 %stay, label %body, label %exit",
            );
        for llvm in [exits_on_and, continues_on_or] {
            assert!(!llvm.contains("br i1 %done"), "{llvm}");
            assert!(
                diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty(),
                "{llvm}"
            );
            assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
        }
    }

    #[test]
    fn counter_stepped_by_a_runtime_bool_is_not_a_loop_counter() {
        // `i += usize::from(ready())`: the step is a widened `i1`, like the
        // range desugaring's, but runtime data decides it, so the trip count
        // is not a compile-time constant
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %i\n  store float 1.0, ptr %slot, align 4\n  %flag = call i1 @ready()\n  %inc = zext i1 %flag to i64\n  %i.next = add nuw nsw i64 %i, %inc\n  %done = icmp eq i64 %i.next, 4\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn counter_that_can_stall_short_of_its_bound_is_not_a_loop_counter() {
        // `i += usize::from(i < 4)` with `if i == 8 || ready() { break }`: the
        // step is a widened literal compare, but it stops `i` at 4, so only
        // `ready()` ends the loop
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n  %idx = and i64 %i, 3\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %idx\n  store float 1.0, ptr %slot, align 4\n  %more = icmp ult i64 %i, 4\n  %inc = zext i1 %more to i64\n  %i.next = add nuw nsw i64 %i, %inc\n  %hit = icmp eq i64 %i.next, 8\n  %flag = call i1 @ready()\n  %quit = or i1 %hit, %flag\n  br i1 %quit, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn escaping_local_is_not_blamed_on_its_loop() {
        // `for i in 0..4 { r[i] = .. }; consume(r.as_ptr())`: unrolling makes
        // every index constant, but the escaping address still pins `r`
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = rolled_counter_loop(&name).replace(
            "exit:\n  ret void",
            &format!("exit:\n  call void @consume(ptr {name})\n  ret void"),
        );
        assert!(llvm.contains("@consume"), "{llvm}");
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn index_picked_by_a_runtime_branch_is_not_a_loop_counter() {
        // `let idx = if ready() { i } else { 0 }`: both incoming values are
        // constant after unrolling, but the phi picks one at run time
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %join ]\n  %flag = call i1 @ready()\n  br i1 %flag, label %pick, label %join\n\npick:\n  br label %join\n\njoin:\n  %idx = phi i64 [ %i, %pick ], [ 0, %body ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %idx\n  store float 1.0, ptr %slot, align 4\n  %i.next = add nuw nsw i64 %i, 1\n  %done = icmp eq i64 %i.next, 4\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn counter_stepped_or_reset_by_a_runtime_branch_is_not_a_loop_counter() {
        // every step and start is a literal, but a phi inside the loop lets
        // `ready()` pick the next value, so the trip count is runtime
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let picks_the_step = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %i\n  store float 1.0, ptr %slot, align 4\n  %flag = call i1 @ready()\n  br i1 %flag, label %one, label %two\n\none:\n  %a = add nuw nsw i64 %i, 1\n  br label %latch\n\ntwo:\n  %b = add nuw nsw i64 %i, 2\n  br label %latch\n\nlatch:\n  %i.next = phi i64 [ %a, %one ], [ %b, %two ]\n  %done = icmp ugt i64 %i.next, 3\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        let resets = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %join ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %i\n  store float 1.0, ptr %slot, align 4\n  %stepped = add nuw nsw i64 %i, 1\n  %flag = call i1 @ready()\n  br i1 %flag, label %reset, label %join\n\nreset:\n  br label %join\n\njoin:\n  %i.next = phi i64 [ %stepped, %body ], [ 0, %reset ]\n  %done = icmp eq i64 %i.next, 4\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        for llvm in [picks_the_step, resets] {
            assert!(
                diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty(),
                "{llvm}"
            );
            assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
        }
    }

    #[test]
    fn many_counted_loops_are_checked_in_linear_time() {
        // a long run of separate counted loops: matching each counter against
        // every branch of the function is quadratic in the number of loops
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let loops = 20_000;
        let last = loops - 1;
        let mut body = String::new();
        for k in 0..loops {
            let preheader = if k == 0 {
                "entry".to_string()
            } else {
                format!("l{}", k - 1)
            };
            body.push_str(&format!(
                "l{k}:\n  %i{k} = phi i64 [ 0, %{preheader} ], [ %n{k}, %l{k} ]\n  %n{k} = add nuw nsw i64 %i{k}, 1\n  %d{k} = icmp eq i64 %n{k}, 4\n  br i1 %d{k}, label %l{}, label %l{k}\n\n",
                k + 1
            ));
        }
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %l0\n\n{body}l{loops}:\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %i{last}\n  store float 1.0, ptr %slot, align 4\n  ret void\n}}\n"
        );
        let start = std::time::Instant::now();
        assert_eq!(
            diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).len(),
            1
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn many_surviving_locals_are_checked_in_linear_time() {
        // a local inlined many times into one kernel, each copy indexed at run
        // time: rescanning the whole function per surviving local is quadratic
        let base = alloca_name("3\t16\tscratch\t[u32; 4]");
        let locals = 10_000;
        let mut allocas = String::new();
        let mut body = String::new();
        for k in 0..locals {
            allocas.push_str(&format!("  {base}{k} = alloca [4 x i32], align 4\n"));
            body.push_str(&format!(
                "  %p{k} = getelementptr [4 x i32], ptr {base}{k}, i64 0, i64 %idx\n  store i32 {k}, ptr %p{k}, align 4\n"
            ));
        }
        let llvm =
            format!("define void @kernel(i64 %idx) {{\nentry:\n{allocas}{body}  ret void\n}}\n");
        let start = std::time::Instant::now();
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), locals);
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.contains("indexed by a non-constant value")),
            "{:?}",
            diagnostics.first()
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn many_element_accesses_are_checked_in_linear_time() {
        // a fully unrolled body that stores to every element of an escaping
        // array: each GEP derives one more pointer from the alloca, and a scan
        // that rechecks every derived pointer on every line is quadratic
        let name = alloca_name("6\t80000\tbuf\t[u32; 20000]");
        let elements = 20_000;
        let mut body = String::new();
        for element in 0..elements {
            body.push_str(&format!(
                "  %p{element} = getelementptr inbounds [20000 x i32], ptr {name}, i64 0, i64 {element}\n  store i32 {element}, ptr %p{element}, align 4\n"
            ));
        }
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [20000 x i32], align 4\n{body}  call void @consume(ptr {name})\n  ret void\n}}\n"
        );
        let start = std::time::Instant::now();
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert!(
            diagnostics[0].contains("address escapes"),
            "{diagnostics:?}"
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn runtime_index_is_reported_only_in_verbose_builds() {
        let name = alloca_name("3\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %idx) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  %value = load i32, ptr %elt, align 4\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn index_loaded_inside_a_loop_is_not_a_loop_counter() {
        // a data-dependent index in a loop: unrolling would not make it constant
        let name = alloca_name("3\t16\thist\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(ptr %keys) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n  %key.ptr = getelementptr i64, ptr %keys, i64 %i\n  %key = load i64, ptr %key.ptr, align 8\n  %bin = getelementptr [4 x i32], ptr {name}, i64 0, i64 %key\n  store i32 1, ptr %bin, align 4\n  %i.next = add i64 %i, 1\n  %done = icmp eq i64 %i.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        let verbose = diagnose_text(&llvm, Reporting::All);
        assert!(
            verbose[0].contains("indexed by a non-constant value"),
            "{verbose:?}"
        );
    }

    #[test]
    fn optimizer_suffixes_do_not_hide_surviving_allocas() {
        let provenance = "8\t64\tvalues\t[u64; 8]";
        let name = format!("{}.sroa.0", alloca_name(provenance));
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [8 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `values`"));
    }

    #[test]
    fn name_uniquing_suffixes_do_not_garble_the_report() {
        // LLVM uniques colliding value names by appending bare digits: the
        // same local inlined twice into one kernel yields `<name>` and
        // `<name>1`. The `_` sentinel ends the hex payload, so the digit must
        // not leak into the decoded fields.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let base = alloca_name(provenance);
        let uniqued = format!("{base}1");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {base} = alloca [4 x i32], align 4\n  {uniqued} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 2);
        for diagnostic in &diagnostics {
            assert!(
                diagnostic.contains("local `scratch` (`[u32; 4]`, 16 bytes)"),
                "uniquing digit corrupted the decoded provenance: {diagnostic}"
            );
        }
    }

    #[test]
    fn capped_payloads_still_attribute_the_local() {
        // The exporter caps the payload length before hex-encoding; a
        // truncated trailing type spelling must not lose the report.
        let full = format!("7\t4096\tstate\t{}", "VeryLongTypeName".repeat(8));
        let truncated = &full[..40];
        let name = alloca_name(truncated);
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [512 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `state`"));
        assert!(diagnostics[0].contains("4096 bytes"));
    }

    #[test]
    fn payload_without_the_sentinel_is_not_trusted() {
        // Without the terminating `_` a trailing uniquing digit is
        // indistinguishable from payload, so such names must be ignored
        // rather than half-decoded.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let name = alloca_name(provenance);
        let unterminated = name.trim_end_matches('_');
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {unterminated} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::All).is_empty());
    }

    #[test]
    fn address_escape_gets_a_specific_reason() {
        let name = alloca_name("4\t32\tstate\tState");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca %State, align 8\n  call void @consume(ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn llvm_lifetime_intrinsics_are_not_reported_as_address_escapes() {
        let name = alloca_name("5\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  call void @llvm.lifetime.start.p0(i64 16, ptr {name})\n  call void @llvm.lifetime.end.p0(i64 16, ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("survives mem2reg and LLVM scalar replacement"));
        assert!(!diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn mangled_containing_functions_are_demangled() {
        let name = alloca_name("2\t8\tacc\t[f32; 2]");
        let llvm = format!(
            "define void @_ZN6kernel5inner17h0123456789abcdefE() {{\nentry:\n  {name} = alloca [2 x float], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("in `kernel::inner`"),
            "expected demangled function name without hash: {}",
            diagnostics[0]
        );
    }

    #[test]
    fn untagged_codegen_allocas_are_ignored() {
        let llvm = "define void @kernel(i64 %idx) {\nentry:\n  %tmp = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr %tmp, i64 0, i64 %idx\n  ret void\n}\n";
        assert!(diagnose_text(llvm, Reporting::All).is_empty());
    }

    #[test]
    fn promoted_local_is_silent_when_the_alloca_is_absent() {
        let llvm = "define i32 @kernel(i32 %value) {\nentry:\n  ret i32 %value\n}\n";
        assert!(diagnose_text(llvm, Reporting::All).is_empty());
    }
}
