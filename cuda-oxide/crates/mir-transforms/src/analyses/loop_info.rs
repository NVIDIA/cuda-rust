/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Finds the loops in a function.
//!
//! A function's code is a graph of basic blocks (straight-line chunks of code)
//! connected by branches; this graph is the "control-flow graph" (CFG). This
//! analysis looks at that graph and reports where the loops are. It is the same
//! idea as LLVM's `LoopInfo`, rebuilt on pliron's CFG and dominator tools.
//!
//! It is a **reusable analysis** (it only reads the IR and reports facts, it
//! never changes the code), so it is not tied to the unroller. For each loop it
//! reports the header, latch(es), the set of body blocks, the preheader, the
//! exiting blocks, and the exit blocks (all defined below). Other passes that
//! work on loops can reuse it.
//!
//! Vocabulary used throughout, with the standard definitions:
//!
//!   * **Dominates.** Block `A` *dominates* block `B` when every path from the
//!     function's entry to `B` must pass through `A` first. (A block always
//!     dominates itself.)
//!   * **Back-edge.** A branch `latch -> header` where `header` dominates
//!     `latch`: control flows "backwards" to a block it already went through,
//!     which is what makes a loop loop.
//!   * **Natural loop.** Given a back-edge, the loop is the `header` plus every
//!     block that can reach the `latch` without first going through the
//!     `header`. In a `while i < n { body }` loop, the header is the block
//!     holding the `i < n` test and the body blocks are everything reachable
//!     from it that branches back.
//!   * Several back-edges that land on the **same** header are treated as one
//!     loop with several latches (e.g. a loop body with two `continue` sites).
//!
//! A CFG is **reducible** when every loop has exactly one entry block (its
//! header); equivalently, the only back-edges are the loop ones. Rust MIR plus
//! mem2reg always produce reducible CFGs. In a reducible CFG any two loops are
//! either fully nested or completely separate (they never partly overlap), so
//! the loops form a clean forest of "loop inside loop inside ..." trees, called
//! the **loop-nesting forest**.

use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::ControlFlowGraph;
use pliron::graph::dominance::DomTree;
use pliron::region::Region;
use rustc_hash::{FxHashMap, FxHashSet};

/// A loop's position in [LoopInfo]'s list (just an index into it).
pub type LoopId = usize;

/// One loop.
#[derive(Debug, Clone)]
pub struct Loop {
    /// The **header**: the loop's single entry block (every iteration starts
    /// here) and the block that dominates all the others in the loop. For a
    /// `while i < n` loop it is the block holding the `i < n` test.
    pub header: Ptr<BasicBlock>,
    /// The **latch(es)**: blocks that branch back to `header` to start the next
    /// iteration. A simple counted loop has exactly one; there can be more when
    /// the body has several places that loop back (e.g. multiple `continue`s).
    pub latches: Vec<Ptr<BasicBlock>>,
    /// Every block in the loop, the `header` and the `latches` included.
    pub blocks: FxHashSet<Ptr<BasicBlock>>,
    /// The loop that immediately encloses this one, if this loop is nested.
    pub parent: Option<LoopId>,
    /// The loops nested directly inside this one.
    pub children: Vec<LoopId>,
}

/// Every loop found in a region, plus how they nest (the loop-nesting forest)
/// and, for each block, which loop most tightly encloses it.
#[derive(Debug, Default)]
pub struct LoopInfo {
    loops: Vec<Loop>,
    top_level: Vec<LoopId>,
    innermost: FxHashMap<Ptr<BasicBlock>, LoopId>,
}

impl LoopInfo {
    /// Find all the loops in `region`. Needs the region's dominator tree (which
    /// records, for every block, the blocks guaranteed to run before it), since
    /// that is how we tell a back-edge from an ordinary forward branch.
    pub fn compute(
        ctx: &Context,
        region: Ptr<Region>,
        dom: &DomTree<Ptr<Region>, Context>,
    ) -> Self {
        // 1. Find the back-edges. A branch `block -> succ` is a back-edge when
        //    `succ` dominates `block` (control loops back to a block that always
        //    ran before it). Group the back-edges by the header (`succ`) they
        //    land on, so each distinct header becomes one loop.
        let mut latches_by_header: FxHashMap<Ptr<BasicBlock>, Vec<Ptr<BasicBlock>>> =
            FxHashMap::default();
        let all_blocks: Vec<Ptr<BasicBlock>> = region.nodes(ctx).collect();
        for &block in &all_blocks {
            for succ in region.successors(ctx, &block) {
                if dom.dominates(&succ, &block) {
                    latches_by_header.entry(succ).or_default().push(block);
                }
            }
        }

        // 2. Work out each loop's body. The body is the header plus every block
        //    that can reach a latch without going through the header. We find it
        //    by walking the CFG *backwards* from the latches (block -> its
        //    predecessors), stopping at the header, which we never step past.
        let mut loops: Vec<Loop> = Vec::with_capacity(latches_by_header.len());
        for (header, latches) in latches_by_header {
            let mut blocks: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
            blocks.insert(header);
            let mut worklist: Vec<Ptr<BasicBlock>> = Vec::new();
            for &latch in &latches {
                if blocks.insert(latch) {
                    worklist.push(latch);
                }
            }
            while let Some(n) = worklist.pop() {
                for pred in region.predecessors(ctx, &n) {
                    if blocks.insert(pred) {
                        worklist.push(pred);
                    }
                }
            }
            loops.push(Loop {
                header,
                latches,
                blocks,
                parent: None,
                children: Vec::new(),
            });
        }

        // 3. Work out the nesting. Sort the loops by body size, smallest first;
        //    a nested loop is always smaller than the one around it.
        let mut by_size: Vec<LoopId> = (0..loops.len()).collect();
        by_size.sort_by_key(|&i| loops[i].blocks.len());

        // A loop L's parent is the smallest loop that is strictly bigger than L
        // and whose body contains L's header. Two loops of equal size can't nest
        // one inside the other, so we only look at strictly-larger loops.
        for (pos, &li) in by_size.iter().enumerate() {
            let header = loops[li].header;
            let li_size = loops[li].blocks.len();
            for &mi in by_size.iter().skip(pos + 1) {
                if loops[mi].blocks.len() > li_size && loops[mi].blocks.contains(&header) {
                    loops[li].parent = Some(mi);
                    break;
                }
            }
        }
        let mut top_level = Vec::new();
        for li in 0..loops.len() {
            match loops[li].parent {
                Some(p) => loops[p].children.push(li),
                None => top_level.push(li),
            }
        }

        // 4. For each block, record the innermost (smallest) loop it belongs to.
        //    Visiting smaller loops first means the first one we record for a
        //    block is the tightest fit, so later (bigger) loops don't overwrite.
        let mut innermost: FxHashMap<Ptr<BasicBlock>, LoopId> = FxHashMap::default();
        for &li in &by_size {
            for &b in &loops[li].blocks {
                innermost.entry(b).or_insert(li);
            }
        }

        LoopInfo {
            loops,
            top_level,
            innermost,
        }
    }

    /// Every loop that was found, at any nesting depth.
    pub fn loops(&self) -> &[Loop] {
        &self.loops
    }

    /// The outermost loops (the ones not nested inside any other loop).
    pub fn top_level(&self) -> &[LoopId] {
        &self.top_level
    }

    /// The smallest loop that `block` sits inside, or `None` if `block` is not
    /// in any loop.
    pub fn innermost_loop(&self, block: Ptr<BasicBlock>) -> Option<LoopId> {
        self.innermost.get(&block).copied()
    }

    /// `true` when the region has no loops at all.
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// The loop's **preheader**: the one block outside the loop that branches
    /// into the header. It is where the loop is entered from, and it is the
    /// natural spot to put setup code that runs once before the loop. Returns
    /// `None` when the header is entered from outside in zero or more than one
    /// place (no single preheader); a caller that needs one can create it.
    ///
    /// ```text
    ///   preheader  ->  header  <-+   (header is entered only from preheader,
    ///                    |       |    plus the latch's back-edge)
    ///                  body -----+
    /// ```
    pub fn preheader(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Option<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        let mut outside = region
            .predecessors(ctx, &l.header)
            .into_iter()
            .filter(|p| !l.blocks.contains(p));
        let first = outside.next()?;
        if outside.next().is_none() {
            Some(first)
        } else {
            None
        }
    }

    /// The **exiting blocks**: blocks inside the loop that can branch to a block
    /// outside it. These are the places the loop can leave from (in a `while`
    /// loop, the header, because its test can branch out).
    pub fn exiting_blocks(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Vec<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        l.blocks
            .iter()
            .copied()
            .filter(|&b| {
                region
                    .successors(ctx, &b)
                    .iter()
                    .any(|s| !l.blocks.contains(s))
            })
            .collect()
    }

    /// The **exit blocks**: blocks outside the loop that the loop branches to.
    /// These are where control lands after leaving the loop. (Exiting blocks are
    /// the inside ends of those branches; exit blocks are the outside ends.)
    pub fn exit_blocks(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Vec<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        let mut seen: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
        let mut out = Vec::new();
        for &b in &l.blocks {
            for s in region.successors(ctx, &b) {
                if !l.blocks.contains(&s) && seen.insert(s) {
                    out.push(s);
                }
            }
        }
        out
    }
}
