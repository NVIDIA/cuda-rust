/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural PTX schedule analysis and deterministic perturbation.
//!
//! The analyzer owns neither CUDA execution nor a fuzzer input generator. It
//! turns a PTX module into stable schedule-sensitive sites and can then apply
//! a seeded perturbation to those sites. Static site discovery, mutation, and
//! triage therefore use the same source model.

use dialect_ptx::cfg::ControlFlow;
use ptx_parse::{Document, EditScript, Instruction, ParseError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;

pub mod campaign;

pub const DEFAULT_MAX_SLEEP_NS: u32 = 64_000;

#[derive(Debug, Error)]
pub enum ScheduleError {
    #[error("could not parse PTX: {0}")]
    Parse(#[from] ParseError),
    #[error("could not recover PTX control flow: {0}")]
    ControlFlow(#[from] dialect_ptx::cfg::CfgError),
    #[error("PTX edit failed: {0}")]
    Edit(#[from] ptx_parse::EditError),
    #[error("intensity must be finite and non-negative, got {0}")]
    InvalidIntensity(f64),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum SiteKind {
    Atomic,
    Reduction,
    Barrier,
    Fence,
    OrderedMemory,
    WarpCollective,
    AsyncProxy,
    GridDependency,
    TensorMapMutation,
    Backedge,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScheduleSite {
    pub ordinal: usize,
    pub callable: String,
    pub kind: SiteKind,
    pub span: Range<usize>,
    pub block: Option<usize>,
    pub head: String,
    pub text: String,
    pub predicate: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScheduleAnalysis {
    sites: Vec<ScheduleSite>,
}

impl ScheduleAnalysis {
    pub fn sites(&self) -> &[ScheduleSite] {
        &self.sites
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InjectionDecision {
    pub site: ScheduleSite,
    pub before_ns: u32,
    pub after_ns: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RewriteReport {
    pub seed: u64,
    pub intensity: f64,
    pub sites_total: usize,
    pub sites_injected: usize,
    pub injected_ns_per_visit: u64,
    pub decisions: Vec<InjectionDecision>,
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    pub ptx: String,
    pub report: RewriteReport,
}

#[derive(Clone, Debug)]
pub struct InjectionOptions {
    pub seed: u64,
    pub intensity: f64,
    pub max_sleep_ns: u32,
    pub focus: Option<String>,
}

impl Default for InjectionOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            intensity: 1.0,
            max_sleep_ns: DEFAULT_MAX_SLEEP_NS,
            focus: None,
        }
    }
}

/// Analyze all executable callables and return schedule-sensitive sites in
/// source order. Control-flow recovery is fail-closed: malformed branches
/// are reported instead of being silently treated as non-loops.
pub fn analyze_ptx(source: &str) -> Result<ScheduleAnalysis, ScheduleError> {
    let document = Document::parse(source)?;
    let control_flow = ControlFlow::analyze(&document)?;
    let mut sites = Vec::new();
    let mut seen = HashMap::<usize, usize>::new();

    for instruction in document.instructions() {
        let Some(kind) = classify_instruction(instruction) else {
            continue;
        };
        let callable = callable_name(&document, instruction.span().start);
        add_site(&mut sites, &mut seen, instruction, callable, kind, None);
    }

    // The CFG is source ordered, so an edge to the same or an earlier block
    // is a conservative intraprocedural back-edge. This catches spin loops
    // even when their labels, predicates, or branch spelling are unusual.
    for callable_cfg in control_flow.callables() {
        for block in callable_cfg.blocks() {
            let Some(&statement) = block.instructions().last() else {
                continue;
            };
            let has_backedge = block
                .successors()
                .iter()
                .any(|edge| edge.block().index() <= block.id().index());
            if !has_backedge {
                continue;
            }
            let Some(instruction) = document.instruction_for_statement(statement) else {
                continue;
            };
            if !matches!(instruction.base_opcode(), "bra" | "brx") {
                continue;
            }
            add_site(
                &mut sites,
                &mut seen,
                instruction,
                callable_cfg.name().to_string(),
                SiteKind::Backedge,
                Some(block.id().index()),
            );
        }
    }

    sites.sort_by_key(|site| site.span.start);
    for (ordinal, site) in sites.iter_mut().enumerate() {
        site.ordinal = ordinal;
    }
    Ok(ScheduleAnalysis { sites })
}

/// Analyze and rewrite a PTX module with deterministic per-site sleeps.
pub fn perturb_ptx(source: &str, options: &InjectionOptions) -> Result<Rewrite, ScheduleError> {
    if !options.intensity.is_finite() || options.intensity < 0.0 {
        return Err(ScheduleError::InvalidIntensity(options.intensity));
    }
    let analysis = analyze_ptx(source)?;
    let intensity = options.intensity;
    let max_sleep_ns = options.max_sleep_ns.max(1);
    let mut rng = SplitMix64::new(options.seed);
    let mut decisions = Vec::with_capacity(analysis.sites.len());
    let mut edits = EditScript::new();
    let mut injected_sites = 0;
    let mut total_ns = 0u64;

    for site in &analysis.sites {
        let (before_point, after_point) = injection_points(source, site);
        let placement = rng.unit();
        let hit = options
            .focus
            .as_deref()
            .is_some_and(|focus| site.head.contains(focus) || site.text.contains(focus));
        let selected = if intensity == 0.0 {
            false
        } else if options.focus.is_some() {
            rng.unit()
                < if hit {
                    (0.95 * intensity).min(1.0)
                } else {
                    (0.15 * intensity).min(1.0)
                }
        } else {
            rng.unit() < (0.75 * intensity).min(1.0)
        };

        let (before_ns, after_ns) = if !selected {
            (0, 0)
        } else if site.kind == SiteKind::Backedge {
            // A delay after a branch is unreachable. Bias loop perturbations
            // before the back-edge so the scheduler observes the delay.
            (draw_delay(&mut rng, intensity, max_sleep_ns), 0)
        } else if hit {
            (0, draw_long(&mut rng, intensity, max_sleep_ns))
        } else if placement < 0.4 {
            (draw_delay(&mut rng, intensity, max_sleep_ns), 0)
        } else if placement < 0.8 {
            (0, draw_delay(&mut rng, intensity, max_sleep_ns))
        } else {
            (
                draw_delay(&mut rng, intensity, max_sleep_ns),
                draw_delay(&mut rng, intensity, max_sleep_ns),
            )
        };

        if before_ns > 0 {
            edits.insert(
                before_point,
                format!(
                    "{}nanosleep.u32 {before_ns}; // ptx_schedule before\n",
                    line_indent(source, before_point)
                ),
            )?;
        }
        if after_ns > 0 {
            edits.insert(
                after_point,
                format!("\nnanosleep.u32 {after_ns}; // ptx_schedule after"),
            )?;
        }
        if before_ns > 0 || after_ns > 0 {
            injected_sites += 1;
            total_ns += u64::from(before_ns) + u64::from(after_ns);
        }
        decisions.push(InjectionDecision {
            site: site.clone(),
            before_ns,
            after_ns,
        });
    }

    let body = edits.apply(source)?;
    let header = format!(
        "// ptx_schedule: seed={} intensity={} sites_total={} sites_injected={} injected_ns_per_visit={}\n",
        options.seed,
        intensity,
        analysis.sites.len(),
        injected_sites,
        total_ns
    );
    Ok(Rewrite {
        ptx: format!("{header}{body}"),
        report: RewriteReport {
            seed: options.seed,
            intensity,
            sites_total: analysis.sites.len(),
            sites_injected: injected_sites,
            injected_ns_per_visit: total_ns,
            decisions,
        },
    })
}

/// Return edit points that remain outside PTX inline-assembly blocks.
///
/// The parser intentionally exposes the instruction inside `{ ... }` inline
/// assembly because it is useful for analysis. Inserting text at that
/// instruction's span would nevertheless corrupt the assembly statement. The
/// codegen backend emits stable begin/end comments around these blocks, so use
/// the surrounding line boundaries as safe insertion points.
fn injection_points(source: &str, site: &ScheduleSite) -> (usize, usize) {
    let before_start = source[..site.span.start].rfind("// begin inline asm");
    let before_end = source[..site.span.start].rfind("// end inline asm");
    if before_start.is_none_or(|start| before_end.is_some_and(|end| end > start)) {
        return (site.span.start, site.span.end);
    }

    let begin = before_start.expect("checked above");
    let begin_line = source[..begin].rfind('\n').map_or(0, |newline| newline + 1);
    let end_marker = source[site.span.end..]
        .find("// end inline asm")
        .map_or(site.span.end, |offset| {
            site.span.end + offset + "// end inline asm".len()
        });
    (begin_line, end_marker)
}

fn classify_instruction(instruction: &Instruction<'_>) -> Option<SiteKind> {
    let head = instruction.head();
    if head.starts_with("atom.") {
        return Some(SiteKind::Atomic);
    }
    if head.starts_with("red.") {
        return Some(SiteKind::Reduction);
    }
    if [
        "bar.sync",
        "bar.arrive",
        "bar.red",
        "bar.warp",
        "barrier.",
        "mbarrier.",
    ]
    .iter()
    .any(|prefix| head.starts_with(prefix))
    {
        return Some(SiteKind::Barrier);
    }
    if head.starts_with("membar.") || head.starts_with("fence.") {
        return Some(SiteKind::Fence);
    }
    // `redux.` needs its own entry: it is a warp-wide register reduction with
    // the same participation contract as the rest of this list, and the `red.`
    // arm above does not reach it -- that prefix carries the dot, so
    // `redux.sync.add.s32` matches neither. Without an entry a `redux.sync`
    // instruction is not a site of any kind, so the analyzer walks past the
    // warp collective it exists to perturb.
    if [
        "activemask",
        "match.any",
        "match.all",
        "vote.",
        "elect.",
        "shfl.",
        "redux.",
    ]
    .iter()
    .any(|prefix| head.starts_with(prefix))
    {
        return Some(SiteKind::WarpCollective);
    }

    // The asynchronous proxy pipeline: the bulk-copy and matrix issues, and
    // the commit/wait pairs that order them against the synchronous proxy.
    // None of the arms above reach these. `cp.async.bulk.tensor...` carries
    // `mbarrier::complete_tx::bytes` in its qualifier list but does not start
    // with `mbarrier.`, so the barrier arm walks past it; `wgmma.fence` is a
    // fence that does not start with `fence.`; and `tcgen05.ld` splits to a
    // base of `tcgen05` rather than `ld`, so the ordered-memory arm below
    // returns before seeing it. `clusterlaunchcontrol.try_cancel.async`
    // completes through an mbarrier exactly as the bulk-tensor copy does, and
    // its `query_cancel` reads that result, so the family is taken whole --
    // the same shape as the three above.
    //
    // An opcode that classifies as nothing is not a site of any kind, so it
    // receives no injection and never appears in a report. That left the
    // pipeline a Hopper or Blackwell kernel is built around as the part a
    // campaign could not reach.
    //
    // * `cp.reduce.async` needs its own prefix: the TMA reduce path emits
    //   `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group`,
    //   and `cp.async` does not match it (reduce sits before async), yet its
    //   commit/wait pair does match `cp.async` -- half-covered without this.
    if [
        "cp.async",
        "cp.reduce.async",
        "wgmma.",
        "tcgen05.",
        "clusterlaunchcontrol.",
    ]
    .iter()
    .any(|prefix| head.starts_with(prefix))
    {
        return Some(SiteKind::AsyncProxy);
    }

    // Programmatic dependent launch orders one grid against another. It is a
    // schedule boundary like a barrier or fence, but belongs to neither the
    // synchronous thread/CTA primitives nor the asynchronous proxy pipeline.
    if head.starts_with("griddepcontrol.") {
        return Some(SiteKind::GridDependency);
    }

    // Tensor-map replacement mutates a descriptor through the generic proxy.
    // Its ordering edge is `fence.proxy.tensormap::generic.*`, so keep it
    // separate from `AsyncProxy`: a campaign must be able to delay the
    // mutation independently from the fence that publishes it.
    if head.starts_with("tensormap.replace.") {
        return Some(SiteKind::TensorMapMutation);
    }

    let mut parts = head.split('.');
    let base = parts.next()?;
    if !matches!(base, "ld" | "st") {
        return None;
    }
    let ordered = parts.any(|part| {
        matches!(
            part,
            "volatile" | "acquire" | "release" | "relaxed" | "acq_rel" | "mmio"
        )
    });
    if ordered && !head.starts_with("ld.global.nc") {
        Some(SiteKind::OrderedMemory)
    } else {
        None
    }
}

fn add_site(
    sites: &mut Vec<ScheduleSite>,
    seen: &mut HashMap<usize, usize>,
    instruction: &Instruction<'_>,
    callable: String,
    kind: SiteKind,
    block: Option<usize>,
) {
    let span = instruction.span();
    if let Some(index) = seen.get(&span.start).copied() {
        if kind == SiteKind::Backedge {
            sites[index].kind = kind;
            sites[index].block = block;
        }
        return;
    }
    let index = sites.len();
    seen.insert(span.start, index);
    sites.push(ScheduleSite {
        ordinal: index,
        callable,
        kind,
        span,
        block,
        head: instruction.head().to_string(),
        text: instruction.text().to_string(),
        predicate: instruction
            .predicate()
            .map(|predicate| predicate.text().to_string()),
    });
}

fn callable_name(document: &Document<'_>, offset: usize) -> String {
    document
        .definitions()
        .find(|definition| {
            definition
                .callable()
                .body_span()
                .is_some_and(|body| body.start <= offset && offset < body.end)
        })
        .map_or_else(
            || "<module>".to_string(),
            |definition| definition.callable().name().to_string(),
        )
}

fn line_indent(source: &str, offset: usize) -> &str {
    let line_start = source[..offset].rfind('\n').map_or(0, |index| index + 1);
    let indent_end = source[line_start..]
        .char_indices()
        .find(|(_, character)| *character != ' ' && *character != '\t')
        .map_or(offset, |(index, _)| line_start + index)
        .min(offset);
    &source[line_start..indent_end]
}

fn draw_delay(rng: &mut SplitMix64, intensity: f64, max_sleep_ns: u32) -> u32 {
    if rng.unit() < 0.25 {
        return 0;
    }
    let high = scaled_max(intensity, max_sleep_ns);
    if rng.unit() < 0.5 {
        rng.range(1, high.min(2_000))
    } else {
        rng.range(2_000.min(high), high)
    }
}

fn draw_long(rng: &mut SplitMix64, intensity: f64, max_sleep_ns: u32) -> u32 {
    let high = scaled_max(intensity, max_sleep_ns);
    rng.range(2_000.min(high), high)
}

fn scaled_max(intensity: f64, max_sleep_ns: u32) -> u32 {
    ((f64::from(max_sleep_ns) * intensity.min(1.0)).round() as u32).max(1)
}

#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, low: u32, high: u32) -> u32 {
        if low >= high {
            return low;
        }
        low + (self.next() % (u64::from(high) - u64::from(low) + 1)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTX: &str = r#".version 8.9
.target sm_80
.address_size 64

.visible .entry race(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
L_loop:
    atom.global.add.u32 %r0, [%rd1], 1;
    red.global.add.u32 [%rd1], %r0;
    ld.global.acquire.u32 %r0, [%rd1];
    st.shared.release.u32 [%r2], %r0;
    bar.sync 0;
    membar.gl;
    shfl.sync.idx.b32 %r0, %r0, 0, 31;
    @%p0 bra L_loop;
    ret;
}
"#;

    #[test]
    fn discovers_structural_sites_and_extra_ordered_memory_forms() {
        let analysis = analyze_ptx(PTX).unwrap();
        let kinds: Vec<_> = analysis.sites().iter().map(|site| site.kind).collect();
        // The recovered Python classifier reports six of these sites: it
        // misses ld.global.acquire and st.shared.release because its regex
        // only accepts an ordering qualifier immediately after ld./st.
        assert_eq!(kinds.len(), 8);
        assert!(kinds.len() > 6);
        assert!(kinds.contains(&SiteKind::Atomic));
        assert!(kinds.contains(&SiteKind::Reduction));
        assert_eq!(
            kinds
                .iter()
                .filter(|kind| **kind == SiteKind::OrderedMemory)
                .count(),
            2
        );
        assert!(kinds.contains(&SiteKind::Backedge));
        assert!(analysis.sites().iter().all(|site| site.callable == "race"));
    }

    #[test]
    fn rewrite_is_deterministic_and_preserves_site_report() {
        let options = InjectionOptions {
            seed: 42,
            intensity: 1.0,
            ..InjectionOptions::default()
        };
        let first = perturb_ptx(PTX, &options).unwrap();
        let second = perturb_ptx(PTX, &options).unwrap();
        assert_eq!(first.ptx, second.ptx);
        assert_eq!(first.report.decisions.len(), first.report.sites_total);
        assert_eq!(
            first.report.sites_injected,
            first
                .report
                .decisions
                .iter()
                .filter(|decision| decision.before_ns > 0 || decision.after_ns > 0)
                .count()
        );
        assert!(first.ptx.contains("nanosleep.u32"));
        assert!(first.ptx.starts_with("// ptx_schedule: seed=42"));
    }

    /// Every warp-level collective PTX has, one instruction each, plus the
    /// `red.*` memory reduction whose prefix looks like `redux`'s but is not.
    ///
    /// `bar.warp.sync` is deliberately absent: the barrier arm above claims it,
    /// and that is the right kind for it.
    const WARP_COLLECTIVES: &str = r#".version 8.9
.target sm_100a
.address_size 64

.visible .entry collectives(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .f32 %f0;
    red.global.add.u32 [%rd1], %r0;
    activemask.b32 %r0;
    match.any.sync.b32 %r0, %r0, 31;
    match.all.sync.b32 %r0|%p0, %r0, 31;
    vote.sync.ballot.b32 %r0, %p0, 31;
    elect.sync %r0|%p0, 31;
    shfl.sync.idx.b32 %r0, %r0, 0, 31;
    redux.sync.add.s32 %r0, %r0, 31;
    redux.sync.min.u32 %r0, %r0, 31;
    redux.sync.and.b32 %r0, %r0, 31;
    redux.sync.min.f32 %f0, %f0, 31;
    redux.sync.max.abs.NaN.f32 %f0, %f0, 31;
    ret;
}
"#;

    #[test]
    fn every_warp_collective_is_a_site() {
        let analysis = analyze_ptx(WARP_COLLECTIVES).unwrap();
        let by_head: Vec<(&str, SiteKind)> = analysis
            .sites()
            .iter()
            .map(|site| (site.head.as_str(), site.kind))
            .collect();

        // One site per instruction in the fixture: an unclassified opcode is
        // not a site at all, which is how the eight `redux.sync` forms used to
        // vanish from a schedule campaign without a word.
        assert_eq!(by_head.len(), 12, "{by_head:?}");

        for (head, kind) in &by_head {
            let expected = if head.starts_with("red.") {
                SiteKind::Reduction
            } else {
                SiteKind::WarpCollective
            };
            assert_eq!(*kind, expected, "{head} classified as {kind:?}");
        }
    }

    /// `red.` is tested before the warp-collective list and `redux` starts with
    /// those three letters, so the two must not be confused in either
    /// direction: the memory reduction stays `Reduction`, and every register
    /// reduction is a `WarpCollective`.
    #[test]
    fn redux_is_a_warp_collective_and_red_is_still_a_reduction() {
        let analysis = analyze_ptx(WARP_COLLECTIVES).unwrap();
        let kinds: Vec<_> = analysis.sites().iter().map(|site| site.kind).collect();
        assert_eq!(
            kinds
                .iter()
                .filter(|kind| **kind == SiteKind::Reduction)
                .count(),
            1
        );
        assert_eq!(
            analysis
                .sites()
                .iter()
                .filter(|site| site.head.starts_with("redux."))
                .count(),
            5
        );
        assert!(
            analysis
                .sites()
                .iter()
                .filter(|site| site.head.starts_with("redux."))
                .all(|site| site.kind == SiteKind::WarpCollective)
        );
    }

    /// A perturbation campaign has to be able to reach a `redux.sync`, which is
    /// the whole point of classifying it.
    #[test]
    fn a_redux_site_can_be_perturbed() {
        let rewrite = perturb_ptx(
            WARP_COLLECTIVES,
            &InjectionOptions {
                seed: 7,
                intensity: 1.0,
                focus: Some("redux.sync".to_string()),
                ..InjectionOptions::default()
            },
        )
        .unwrap();
        let redux_injected = rewrite
            .report
            .decisions
            .iter()
            .filter(|decision| decision.site.head.starts_with("redux."))
            .filter(|decision| decision.before_ns > 0 || decision.after_ns > 0)
            .count();
        assert!(redux_injected > 0, "{:?}", rewrite.report.decisions);
        assert!(rewrite.ptx.contains("nanosleep.u32"));
    }

    #[test]
    fn zero_intensity_is_a_valid_noop() {
        let rewrite = perturb_ptx(
            PTX,
            &InjectionOptions {
                intensity: 0.0,
                ..InjectionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(rewrite.report.sites_injected, 0);
        assert!(!rewrite.ptx.contains("nanosleep.u32"));
    }

    /// Every spelling here is one the backend actually emits: they were taken
    /// verbatim from the `.ptx` the example corpus builds, not hand-written.
    /// The one exception is `cp.reduce.async.bulk.tensor...`, which no example
    /// emits today; its spelling is the mir-lower TMA-reduce inline-asm
    /// template, verbatim.
    /// `mbarrier.arrive` and `fence.proxy.async` sit alongside them on purpose,
    /// as the controls for the two arms this classification could have stolen
    /// from.
    const ASYNC_PROXY: &str = r#".version 8.7
.target sm_100a
.address_size 64

.visible .entry async_proxy(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .b64 %rd1;
    cp.async.ca.shared.global [%r0], [%rd1], 4;
    cp.async.commit_group;
    cp.async.wait_all;
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r0], [%rd1], [%r0];
    cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%rd1, {%r0, %r0}], [%r0];
    wgmma.fence.sync.aligned;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r0];
    tcgen05.ld.sync.aligned.16x256b.x1.b32 %r0, [%r0];
    tcgen05.wait::ld.sync.aligned;
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r0, 32;
    tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
    clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%r0], [%r0];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p0, %rd1;
    mbarrier.arrive.shared.b64 %rd1, [%r0];
    fence.proxy.async.shared::cta;
    ret;
}
"#;

    #[test]
    fn async_proxy_pipeline_instructions_are_sites() {
        let analysis = analyze_ptx(ASYNC_PROXY).unwrap();
        let async_sites: Vec<_> = analysis
            .sites()
            .iter()
            .filter(|site| site.kind == SiteKind::AsyncProxy)
            .collect();
        // The fifteen async-proxy instructions in the fixture: three
        // cp.async, one bulk-tensor issue, one bulk-tensor reduce, three
        // wgmma, five tcgen05 and two clusterlaunchcontrol.
        assert_eq!(async_sites.len(), 15, "{:?}", async_sites);
        for prefix in [
            "cp.async",
            "cp.reduce.async",
            "wgmma.",
            "tcgen05.",
            "clusterlaunchcontrol.",
        ] {
            assert!(
                async_sites.iter().any(|site| site.head.starts_with(prefix)),
                "no AsyncProxy site for {prefix}"
            );
        }
    }

    /// The two arms this sits between, and which of them could actually take
    /// one of these instructions away.
    ///
    /// `wgmma.fence.sync.aligned` is the live collision: it contains
    /// `fence.`, so relaxing the fence arm from a prefix test to a substring
    /// test silently reclassifies it, and this test fails when that is done.
    ///
    /// The barrier arm is not a collision, and the reason is worth recording:
    /// `cp.async.bulk.tensor...mbarrier::complete_tx::bytes` spells that
    /// qualifier with `::`, so it never contains the `mbarrier.` the barrier
    /// arm looks for. The assertions on `mbarrier.` and `fence.` below are
    /// there to pin that the two real instructions keep their own kinds.
    #[test]
    fn async_proxy_classification_leaves_mbarrier_and_fence_alone() {
        let analysis = analyze_ptx(ASYNC_PROXY).unwrap();
        let kind_of = |prefix: &str| {
            analysis
                .sites()
                .iter()
                .find(|site| site.head.starts_with(prefix))
                .map(|site| site.kind)
        };
        assert_eq!(kind_of("mbarrier."), Some(SiteKind::Barrier));
        assert_eq!(kind_of("fence."), Some(SiteKind::Fence));
        assert_eq!(kind_of("wgmma.fence"), Some(SiteKind::AsyncProxy));
    }

    /// A campaign has to be able to reach the pipeline, which is the whole
    /// point of classifying it.
    #[test]
    fn an_async_proxy_site_can_be_perturbed() {
        let rewrite = perturb_ptx(
            ASYNC_PROXY,
            &InjectionOptions {
                seed: 7,
                intensity: 1.0,
                ..InjectionOptions::default()
            },
        )
        .unwrap();
        let injected = rewrite
            .report
            .decisions
            .iter()
            .filter(|decision| decision.site.kind == SiteKind::AsyncProxy)
            .filter(|decision| decision.before_ns > 0 || decision.after_ns > 0)
            .count();
        assert!(injected > 0, "{:?}", rewrite.report.decisions);
        assert!(rewrite.ptx.contains("nanosleep.u32"));
        // The rewrite must still be analyzable, and must not have dropped or
        // corrupted the instructions it wrapped.
        let reanalyzed = analyze_ptx(&rewrite.ptx).unwrap();
        assert!(
            reanalyzed
                .sites()
                .iter()
                .filter(|site| site.kind == SiteKind::AsyncProxy)
                .count()
                >= 15
        );
    }

    /// Both spellings are emitted by mir-lower for programmatic dependent
    /// launch. They order one grid against another and must remain distinct
    /// from the asynchronous proxy pipeline. The trailing `fence.acq_rel.gpu`
    /// is an unaffected control: a neighbouring kind that must keep its kind.
    /// No in-tree example emits `griddepcontrol` yet, so the spellings come
    /// from the mir-lower lowering (inline asm on the libNVVM path, the
    /// `llvm.nvvm.griddepcontrol.*` intrinsics otherwise), as the AsyncProxy
    /// fixture does for `cp.reduce.async`.
    const GRID_DEPENDENCY: &str = r#".version 8.7
.target sm_90
.address_size 64

.visible .entry grid_dependency()
{
    griddepcontrol.launch_dependents;
    griddepcontrol.wait;
    fence.acq_rel.gpu;
    ret;
}
"#;

    #[test]
    fn grid_dependency_instructions_are_sites() {
        let analysis = analyze_ptx(GRID_DEPENDENCY).unwrap();
        let grid_dependency_sites: Vec<_> = analysis
            .sites()
            .iter()
            .filter(|site| site.kind == SiteKind::GridDependency)
            .collect();

        assert_eq!(analysis.sites().len(), 3, "{:?}", analysis.sites());
        assert_eq!(
            grid_dependency_sites.len(),
            2,
            "{:?}",
            grid_dependency_sites
        );
        assert_eq!(
            grid_dependency_sites[0].head,
            "griddepcontrol.launch_dependents"
        );
        assert_eq!(grid_dependency_sites[1].head, "griddepcontrol.wait");

        let fence = analysis
            .sites()
            .iter()
            .find(|site| site.head == "fence.acq_rel.gpu")
            .expect("fence control should remain a schedule site");

        assert_eq!(fence.kind, SiteKind::Fence);
    }

    #[test]
    fn a_grid_dependency_site_can_be_perturbed() {
        let rewrite = perturb_ptx(
            GRID_DEPENDENCY,
            &InjectionOptions {
                seed: 7,
                intensity: 1.0,
                focus: Some("griddepcontrol".to_string()),
                ..InjectionOptions::default()
            },
        )
        .unwrap();
        let injected = rewrite
            .report
            .decisions
            .iter()
            .filter(|decision| decision.site.kind == SiteKind::GridDependency)
            .filter(|decision| decision.before_ns > 0 || decision.after_ns > 0)
            .count();
        assert!(injected > 0, "{:?}", rewrite.report.decisions);
        assert!(rewrite.ptx.contains("nanosleep.u32"));
    }

    /// The three operand forms the TMA lowering emits: a register value
    /// (`global_address`), an ordinal plus register (`global_dim`), and an
    /// immediate (`swizzle_mode`). The release fence is the publication edge
    /// that orders the generic-proxy descriptor mutation before a tensor-map
    /// consumer. No in-tree example emits `tensormap.replace` yet, so the
    /// spellings come from the mir-lower template
    /// (`convert/intrinsics/tma.rs`), as the AsyncProxy fixture does for
    /// `cp.reduce.async`.
    const TENSORMAP_MUTATIONS: &str = r#".version 8.7
.target sm_90a
.address_size 64

.visible .entry tensormap_mutations()
{
    .reg .b32 %r0;
    .reg .b64 %rd0;
    tensormap.replace.tile.global_address.global.b1024.b64 [%rd0], %rd0;
    tensormap.replace.tile.global_dim.global.b1024.b32 [%rd0], 0, %r0;
    tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%rd0], 3;
    fence.proxy.tensormap::generic.release.gpu;
    ret;
}
"#;

    #[test]
    fn tensormap_replace_instructions_are_sites() {
        let analysis = analyze_ptx(TENSORMAP_MUTATIONS).unwrap();
        let mutations: Vec<_> = analysis
            .sites()
            .iter()
            .filter(|site| site.kind == SiteKind::TensorMapMutation)
            .collect();

        assert_eq!(mutations.len(), 3, "{mutations:?}");
        assert!(
            mutations
                .iter()
                .any(|site| site.head.contains(".global_address."))
        );
        assert!(
            mutations
                .iter()
                .any(|site| site.head.contains(".global_dim."))
        );
        assert!(
            mutations
                .iter()
                .any(|site| site.head.contains(".swizzle_mode."))
        );
    }

    #[test]
    fn tensormap_mutation_classification_leaves_proxy_fence_alone() {
        let analysis = analyze_ptx(TENSORMAP_MUTATIONS).unwrap();
        let kind_of = |prefix: &str| {
            analysis
                .sites()
                .iter()
                .find(|site| site.head.starts_with(prefix))
                .map(|site| site.kind)
        };

        assert_eq!(
            kind_of("tensormap.replace.tile.global_address"),
            Some(SiteKind::TensorMapMutation)
        );
        assert_eq!(
            kind_of("tensormap.replace.tile.global_dim"),
            Some(SiteKind::TensorMapMutation)
        );
        assert_eq!(
            kind_of("tensormap.replace.tile.swizzle_mode"),
            Some(SiteKind::TensorMapMutation)
        );
        assert_eq!(kind_of("fence.proxy.tensormap"), Some(SiteKind::Fence));
    }

    /// A focused campaign must be able to insert a delay after the descriptor
    /// mutation and before the release fence that publishes it.
    #[test]
    fn a_tensormap_mutation_can_be_delayed_before_its_publish_fence() {
        let rewrite = perturb_ptx(
            TENSORMAP_MUTATIONS,
            &InjectionOptions {
                seed: 7,
                intensity: 1.0,
                focus: Some("global_dim".to_string()),
                ..InjectionOptions::default()
            },
        )
        .unwrap();

        let decision = rewrite
            .report
            .decisions
            .iter()
            .find(|decision| decision.site.head.contains(".global_dim."))
            .expect("global_dim tensor-map mutation must be a schedule site");
        assert_eq!(decision.site.kind, SiteKind::TensorMapMutation);
        assert!(decision.after_ns > 0, "{:?}", rewrite.report.decisions);

        let mutation = rewrite
            .ptx
            .find("tensormap.replace.tile.global_dim")
            .expect("rewritten PTX must retain the mutation");
        let fence = rewrite
            .ptx
            .find("fence.proxy.tensormap")
            .expect("rewritten PTX must retain the publish fence");
        assert!(
            rewrite.ptx[mutation..fence].contains("nanosleep.u32"),
            "focused perturbation must delay the mutation before the fence"
        );
    }
}
