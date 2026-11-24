/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Control-flow sources of divergence for the uniformity analyses in `program.rs`.
//!
//! A value can differ across threads without any operand that differs across
//! threads. Two shapes produce such values, and both are invisible to an analysis
//! that only follows data flow:
//!
//! ```text
//!   sync dependence (divergent join)        temporal divergence (divergent loop exit)
//!   ────────────────────────────────        ─────────────────────────────────────────
//!   if tid < 16 { n = 1 } else { n = 3 }    let mut i = 0;
//!   // `n` is a block argument fed by        while i < tid { i += 1 }
//!   // two constants, yet it depends on      // `i` is a block argument fed by `0` and
//!   // which arm the thread took.            // `i + 1`, yet threads leave the loop
//!                                            // with different values.
//! ```
//!
//! [`DivergenceStructure`] records, once per kernel, the join blocks of every
//! two-way branch and the shape of every natural loop. The uniformity fixpoints
//! then call [`DivergenceStructure::taint_block_args`] to mark the block arguments
//! that depend on a divergent branch.

use std::collections::{HashMap, HashSet};

use pliron::{context::Context, linked_list::ContainsLinkedList, value::Value};

use super::ir_utils::{BlockGraph, op_name};

/// One natural loop: its header, the blocks whose `mir.cond_br` can leave it, and
/// the blocks those exiting edges reach.
#[derive(Debug)]
struct LoopShape {
    /// Loop header (target of a back edge in reverse post-order).
    header: usize,
    /// Blocks inside the loop that end in a `mir.cond_br` with a successor outside it.
    exiting: Vec<usize>,
    /// Successors of `exiting` blocks that lie outside the loop.
    exits: Vec<usize>,
}

/// Join blocks of every two-way branch and the natural loops of one kernel CFG.
#[derive(Debug)]
pub(super) struct DivergenceStructure {
    /// Condition operand of each block that ends in a two-way `mir.cond_br`.
    branch_cond: HashMap<usize, Value>,
    /// For each such block, the blocks where paths starting at its two
    /// successors first meet (excluding paths that cross a back edge).
    join_blocks: HashMap<usize, Vec<usize>>,
    /// Natural loops, one entry per header.
    loops: Vec<LoopShape>,
}

impl DivergenceStructure {
    /// Compute join blocks and loop shapes over the reachable blocks `rpo` (old
    /// block indices in reverse post-order).
    pub(super) fn build(ctx: &Context, graph: &BlockGraph, rpo: &[usize]) -> Self {
        let pos: HashMap<usize, usize> = rpo.iter().enumerate().map(|(p, &b)| (b, p)).collect();
        let mut preds: HashMap<usize, Vec<usize>> = HashMap::new();
        for &b in rpo {
            for &s in &graph.succs[b] {
                if pos.contains_key(&s) {
                    preds.entry(s).or_default().push(b);
                }
            }
        }

        let mut branch_cond = HashMap::new();
        for &b in rpo {
            let blk = graph.blocks[b].deref(ctx);
            for op_ptr in blk.iter(ctx) {
                if op_name(ctx, op_ptr) == "mir.cond_br" {
                    let op = op_ptr.deref(ctx);
                    if op.get_num_operands() >= 1 && graph.succs[b].len() == 2 {
                        branch_cond.insert(b, op.get_operand(0));
                    }
                }
            }
        }

        let mut join_blocks = HashMap::new();
        for &b in branch_cond.keys() {
            join_blocks.insert(b, Self::joins_of(graph, rpo, &pos, &preds, b));
        }

        let loops = Self::natural_loops(graph, rpo, &pos, &preds, &branch_cond);
        Self {
            branch_cond,
            join_blocks,
            loops,
        }
    }

    /// Join blocks of the two-way branch ending block `b`.
    ///
    /// Each successor reached through a forward edge starts with its own label;
    /// labels flow along forward edges in reverse post-order, and a block reached
    /// by two different labels is a join (it then starts a label of its own, as a
    /// phi would). Back edges are not followed: values carried around a loop are
    /// handled by the loop rule in [`Self::taint_block_args`].
    fn joins_of(
        graph: &BlockGraph,
        rpo: &[usize],
        pos: &HashMap<usize, usize>,
        preds: &HashMap<usize, Vec<usize>>,
        b: usize,
    ) -> Vec<usize> {
        let b_pos = pos[&b];
        let seeds: HashSet<usize> = graph.succs[b]
            .iter()
            .copied()
            .filter(|s| pos.get(s).is_some_and(|&p| p > b_pos))
            .collect();
        let mut label: HashMap<usize, usize> = HashMap::new();
        let mut joins = Vec::new();
        for &x in &rpo[b_pos + 1..] {
            let mut incoming: HashSet<usize> = HashSet::new();
            if seeds.contains(&x) {
                incoming.insert(x);
            }
            for p in preds.get(&x).into_iter().flatten() {
                if *p != b
                    && pos[p] < pos[&x]
                    && let Some(&l) = label.get(p)
                {
                    incoming.insert(l);
                }
            }
            match incoming.len() {
                0 => {}
                1 => {
                    label.insert(x, *incoming.iter().next().unwrap_or(&x));
                }
                _ => {
                    joins.push(x);
                    label.insert(x, x);
                }
            }
        }
        joins
    }

    /// Natural loops of the CFG: for each back edge `u -> h`, the loop body is `h`
    /// plus every block that reaches `u` without passing through `h`.
    fn natural_loops(
        graph: &BlockGraph,
        rpo: &[usize],
        pos: &HashMap<usize, usize>,
        preds: &HashMap<usize, Vec<usize>>,
        branch_cond: &HashMap<usize, Value>,
    ) -> Vec<LoopShape> {
        let mut bodies: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &u in rpo {
            for &h in &graph.succs[u] {
                if pos.get(&h).is_some_and(|&ph| ph <= pos[&u]) {
                    let body = bodies.entry(h).or_insert_with(|| HashSet::from([h]));
                    let mut stack = vec![u];
                    while let Some(x) = stack.pop() {
                        if body.insert(x) {
                            stack.extend(preds.get(&x).into_iter().flatten().copied());
                        }
                    }
                }
            }
        }
        let mut loops: Vec<LoopShape> = bodies
            .into_iter()
            .map(|(header, body)| {
                let mut exiting = Vec::new();
                let mut exits = Vec::new();
                for &x in &body {
                    if !branch_cond.contains_key(&x) {
                        continue;
                    }
                    let outside: Vec<usize> = graph.succs[x]
                        .iter()
                        .copied()
                        .filter(|s| pos.contains_key(s) && !body.contains(s))
                        .collect();
                    if !outside.is_empty() {
                        exiting.push(x);
                        exits.extend(outside);
                    }
                }
                exiting.sort_unstable();
                exits.sort_unstable();
                exits.dedup();
                LoopShape {
                    header,
                    exiting,
                    exits,
                }
            })
            .collect();
        loops.sort_by_key(|l| pos[&l.header]);
        loops
    }

    /// Add to `divergent` the block arguments that depend on a divergent branch,
    /// returning `true` when the set grew.
    ///
    /// - Sync dependence: when the condition of a two-way branch is in
    ///   `divergent`, every argument of its join blocks is divergent.
    /// - Temporal divergence: when the condition of any branch that leaves a loop
    ///   is in `divergent`, threads leave that loop in different iterations, so
    ///   every argument of the loop header and of its exit blocks is divergent.
    pub(super) fn taint_block_args(
        &self,
        ctx: &Context,
        graph: &BlockGraph,
        divergent: &mut HashSet<Value>,
    ) -> bool {
        let mut changed = false;
        let mut taint_args = |blk: usize, divergent: &mut HashSet<Value>| {
            for arg in graph.blocks[blk].deref(ctx).arguments() {
                changed |= divergent.insert(arg);
            }
        };
        for (b, joins) in &self.join_blocks {
            if divergent.contains(&self.branch_cond[b]) {
                for &j in joins {
                    taint_args(j, divergent);
                }
            }
        }
        for lp in &self.loops {
            if lp
                .exiting
                .iter()
                .any(|e| divergent.contains(&self.branch_cond[e]))
            {
                taint_args(lp.header, divergent);
                for &x in &lp.exits {
                    taint_args(x, divergent);
                }
            }
        }
        changed
    }
}
