//! Count-index thresholds: the running counts at which a cell's recommended move flips. These are a
//! count-*independent* property of each `(up-card, shoe, ruleset)` cell — the player's count only says
//! where they sit on the ladder — so they are computed once over the whole reachable count axis
//! ([`ColumnEval`]/[`coalesce_runs`], monotone root-finding sharing one deconvolution across the band)
//! and overlaid on the base chart. [`compute_index_report`] is the background-worker entry point.

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::Sender;

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::count::{CountCmp, CountKind, CountShoe, CountSystem, HiLo, Ko, TC_HALF_UNITS};
use crate::diskcache;
use crate::hand::{HandCategory, Move};
use crate::reach::{reach_weights, summarize_cells};
use crate::rules::Ruleset;
use crate::shoe::Shoe;
use crate::simulation::build_evs;

use super::PANES;
use super::column::{
    COUNT_GROUPS, ColumnSummary, ReachMap, Tree, merge_count_frames, solve_counted,
};
use super::config::{COUNT_PENETRATION, ShoeChoice};

/// One solved frame: the `build_evs` EV tree and its game-time reach weights — the unit the count-index
/// WoO merge reads off (one per band count). Persisted per [`FrameKey`] so a warm relaunch skips the
/// re-solve, which is the dominant cost of a cold fill; in memory it is memoized only *within* a column
/// (the band counts a column's WoO merges share), since distinct up-cards never share a frame.
type Frame = (Tree, ReachMap);

/// What a persisted frame is keyed under: up-card, deck count, exact external count, ruleset, and the
/// counting-system family ([`CountKind`], so a running-count frame and a true-count frame at the same
/// numeric count never alias). The band's comparison is always `Eq` and its penetration prior a fixed
/// constant ([`COUNT_PENETRATION`]), so neither is in the key.
type FrameKey = (Card, u8, i16, Ruleset, CountKind);

/// Disk namespace for persisted frames (distinct from the chart's `"column"` cache).
const FRAME_KIND: &str = "frame";

/// How much occurrence probability the count-index window may drop off **each** tail. The window is the
/// central span of the running-count occurrence distribution
/// ([`CountShoe::external_count_distribution`]) that keeps all but this much mass per side — i.e. we
/// solve every count a player realistically holds and report the rare extremes open-ended (`≤`/`≥`).
/// Counts past it occur under this fraction of the time; their flips are theoretical, not the genuine
/// suggested-play deviations the index is for (for an exact EV at an extreme count, set the count
/// constraint to it directly).
///
/// This is the *whole* tuning knob: a dimensionless probability, not a hand-derived width. Widening or
/// narrowing the window is just a change here — the per-deck `[lo, hi]` recompute themselves off the
/// live distribution, no magic numbers to re-derive. It is also **system-agnostic**: the same threshold
/// carries to any [`CountSystem`](crate::count::CountSystem). The one KO-specific assumption — that the
/// player's *actionable* count is the external *running* count (so the occurrence axis is the running
/// count, independent of penetration depth) — lives in [`ColumnEval`], which sweeps that axis. A
/// *true-count* system (HiLo, once added) acts on running ÷ decks-remaining, so its occurrence axis is
/// the true count (a function of pool size too); generalizing means giving the occurrence distribution
/// that axis, after which this threshold and the trimming in [`occurrence_window`] apply unchanged.
const INDEX_TAIL_MASS: f64 = 0.01;

/// The inclusive external-count window `[lo, hi]` covering the central mass of occurrence distribution
/// `dist` (ascending `(count, P(count))` pairs), dropping at most `tail_each` probability off each end.
/// The trim is purely on probability mass, so it is independent of the count scale or system — only the
/// `dist` passed in is system-specific. A count is dropped from a tail only while the running total of
/// dropped mass stays within budget; the first count that would exceed it becomes the edge (kept).
fn occurrence_window(dist: &[(i16, f64)], tail_each: f64) -> (i16, i16) {
    let mut lo = 0;
    let mut dropped = 0.0;
    while lo + 1 < dist.len() && dropped + dist[lo].1 <= tail_each {
        dropped += dist[lo].1;
        lo += 1;
    }
    let mut hi = dist.len() - 1;
    let mut dropped = 0.0;
    while hi > lo && dropped + dist[hi].1 <= tail_each {
        dropped += dist[hi].1;
        hi -= 1;
    }
    (dist[lo].0, dist[hi].0)
}

/// Max count-index columns solved concurrently in the background. Each is much heavier than a chart
/// column (a handful of count-conditioned solves, splits and all), so this is kept well below the ten
/// chart workers to avoid swamping the cores the in-column split parallelism already wants.
pub(super) const INDEX_FILL_CONCURRENCY: usize = 3;

/// How much occurrence mass the chart `°` marker's *notable* band drops off **each** tail — the
/// commonly-held core of the band, deliberately looser than [`INDEX_TAIL_MASS`] (which bounds the full
/// solved/popup window). Because it is a mass threshold on the live occurrence distribution
/// ([`CountShoe::external_count_distribution`]), this core tracks the deck count and penetration and is
/// system-agnostic — for a balanced count it straddles 0, for an unbalanced one (KO) it slides down to
/// where the running count actually sits (≈ the IRC..pivot span). That is the whole fix for the
/// negative side: at six decks the typical count is ≈ −20..−8, so a deviation at RC −10 is *common*,
/// not extreme, yet a fixed `|RC| ≤ 4` window (centered on 0, as for a balanced count) suppressed its
/// marker. See [`INDEX_MARKER_PIVOT_MARGIN`] for why the mass core alone is not enough on the high side.
const INDEX_MARKER_TAIL_MASS: f64 = 0.10;

/// How far either side of the **pivot** (the zero-edge running count: 0 for a balanced count, +4 for
/// KO) the marker band always reaches, on top of the [`INDEX_MARKER_TAIL_MASS`] mass core. The
/// occurrence distribution is heavily left-skewed for multi-deck KO (its mass piles up near the very
/// negative IRC), so a symmetric mass trim alone clips the high tail down to ≈ pivot and would drop the
/// canonical high-count deviations — insurance, 16vT stand, etc. — which for KO cluster a few counts
/// either side of the pivot regardless of deck count. Anchoring a fixed margin at the pivot (not at 0)
/// keeps those flagged for every deck count while the mass core supplies the deck-dependent common
/// (negative) range. The final band is `[min(mass_lo, pivot − M), max(mass_hi, pivot + M)]`; flips
/// outside it are suppressed on the chart but still shown in full in the always-exhaustive popup.
const INDEX_MARKER_PIVOT_MARGIN: i16 = 5;

/// Whether a move is only available as the opening action on a two-card hand (so it disappears once the
/// hand has been hit). These are exactly the headline moves a [`CategoryIndex`] gives a *fallback*
/// Hit/Stand ladder for: "surrender below RC −1" is incomplete without "…and once you can't surrender,
/// stand at RC ≥ …".
fn is_start_only(mv: Move) -> bool {
    matches!(mv, Move::Double | Move::Split | Move::Surrender)
}

/// Whether an ascending `(move, lo, hi)` run list has a play change whose *flip point* — the running
/// count at which the later move takes over, i.e. the `lo` of each run after the first — falls inside
/// the inclusive band `[band_lo, band_hi]`. Used by the chart marker to ignore flips that only happen
/// out in the occurrence tails. The band is generally asymmetric and off-zero (an unbalanced count's
/// common range does not straddle 0), so it is passed explicitly rather than as a `±` half-width.
///
/// This keys on the boundary, not on which moves are *visible* in the band: a Hit→Stand index at the
/// band's lower edge fires even though its Hit leg lives entirely below the band. Counting distinct
/// in-band moves missed exactly that case — one leg of an in-band flip can sit just outside it.
fn flips_in_band(runs: &[(Move, i16, i16)], band_lo: i16, band_hi: i16) -> bool {
    runs.iter()
        .skip(1)
        .any(|&(_, lo, _)| (band_lo..=band_hi).contains(&lo))
}

/// One chart category's count-dependent move ladder over the index window, in ascending running-count
/// order. `primary` is the headline move's `(move, lo, hi)` inclusive-RC runs (a single run ⇒ the move
/// never changes ⇒ no count dependence). `fallback` is the Hit-vs-Stand ladder that applies once a
/// start-only headline move is unavailable (a hand that has already been hit); it is populated only
/// when `primary` actually contains a start-only move.
///
/// `basic`/`basic_fallback` are the **no-count** (basic-strategy) moves for the two ladders — the
/// headline and the Hit-vs-Stand fallback of the uncounted base chart. They identify which runs are
/// *deviations* (a run whose move differs from basic) so the true-count display can give the deviation
/// the inclusive side of its cutoff (see [`fmt_tc_range`](super::render::fmt_tc_range)). Populated only
/// for a true-count report (the running-count display keeps the offset boundaries); `None` otherwise.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(super) struct CategoryIndex {
    pub(super) primary: Vec<(Move, i16, i16)>,
    pub(super) fallback: Vec<(Move, i16, i16)>,
    pub(super) basic: Option<Move>,
    pub(super) basic_fallback: Option<Move>,
}

impl CategoryIndex {
    /// The cell's right play genuinely shifts with the running count within the *notable* band
    /// `[band_lo, band_hi]` — what the chart `°` marker keys on: either the headline flips or there is a
    /// Hit/Stand flip behind a start-only headline move, somewhere in the band. A ladder that is
    /// constant across the band (its only flips are out in the occurrence tails) is treated as not
    /// count-dependent *for display*; the popup still renders the whole ladder. The band is the
    /// report's [`mark_lo`..=`mark_hi`](IndexReport), tracking the deck count and penetration (see
    /// [`INDEX_MARKER_TAIL_MASS`] and [`INDEX_MARKER_PIVOT_MARGIN`]).
    pub(super) fn count_dependent_in_band(&self, band_lo: i16, band_hi: i16) -> bool {
        flips_in_band(&self.primary, band_lo, band_hi)
            || flips_in_band(&self.fallback, band_lo, band_hi)
    }

    /// The distinct start-only moves the primary ladder recommends somewhere (so the popup can label the
    /// fallback "if can't surrender" etc.).
    pub(super) fn start_only_moves(&self) -> Vec<Move> {
        let mut out: Vec<Move> = Vec::new();
        for &(mv, _, _) in &self.primary {
            if is_start_only(mv) && !out.contains(&mv) {
                out.push(mv);
            }
        }
        out
    }
}

/// One up-card column's full count-index report: each chart category's [`CategoryIndex`] over the
/// shared usable window `[lo, hi]` (the positive-mass external-count range). Count-*independent* — the
/// player's current count only picks where they sit on the ladder — so it is cached per
/// `(up-card, shoe, ruleset)` and overlaid on the base chart. Filled incrementally: Hard/Soft (no split
/// solves) arrive first with `complete = false`, then Pairs complete it, so `cats` may be partial.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct IndexReport {
    pub(super) lo: i16,
    pub(super) hi: i16,
    /// The chart `°` marker's *notable* count band, inside the solved `[lo, hi]` window: the
    /// commonly-held occurrence-mass core ([`INDEX_MARKER_TAIL_MASS`]) unioned with a fixed margin
    /// around the pivot ([`INDEX_MARKER_PIVOT_MARGIN`]). A flip is marked only if its boundary falls in
    /// `[mark_lo, mark_hi]` — for an unbalanced count this sits well off zero and is asymmetric.
    pub(super) mark_lo: i16,
    pub(super) mark_hi: i16,
    pub(super) cats: HashMap<HandCategory, CategoryIndex>,
    pub(super) complete: bool,
}

/// What an [`IndexReport`] is cached/keyed under. Deliberately *without* the player's count (the report
/// is count-independent) and without the chart's count comparison (the index is always exact-count
/// based).
pub(super) type IndexKey = (Card, ShoeChoice, Ruleset, CountKind);

/// A finished (or partial) count-index report, tagged with the index epoch it was computed under so a
/// stale one (the shoe or ruleset changed) is dropped on arrival.
pub(super) struct IndexResult {
    pub(super) epoch: u64,
    pub(super) key: IndexKey,
    pub(super) report: IndexReport,
}

/// Build the `(move, lo, hi)` runs of `move_fn` over the integer running-count window `[lo, hi]` using
/// monotone root-finding: seed a few points, then bisect every adjacent pair whose move differs down to
/// integer adjacency, pinning each flip exactly. The per-move EV differences are monotone in the count,
/// so a flipped pair brackets a single crossing (the recursion still splits both halves, so an
/// intermediate move is handled too); same-move endpoints are taken as constant between them. Far
/// cheaper than sweeping every count — `O(log width)` evaluations per flip — which is the point, since
/// each evaluation is a full count-conditioned solve. The first/last runs are stretched to the window
/// edges so [`fmt_rc_range`](super::render::fmt_rc_range) reads them as open-ended.
fn coalesce_runs(
    lo: i16,
    hi: i16,
    mut move_fn: impl FnMut(i16) -> Option<Move>,
) -> Vec<(Move, i16, i16)> {
    let mut samples: BTreeMap<i16, Move> = BTreeMap::new();
    for s in seed_points(lo, hi) {
        if let Some(mv) = move_fn(s) {
            samples.insert(s, mv);
        }
    }
    if samples.is_empty() {
        return Vec::new();
    }
    let mut stack: Vec<(i16, i16)> = samples
        .keys()
        .copied()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();
    while let Some((a, b)) = stack.pop() {
        if b - a <= 1 || samples[&a] == samples[&b] {
            continue;
        }
        let m = a + (b - a) / 2;
        if let std::collections::btree_map::Entry::Vacant(e) = samples.entry(m) {
            match move_fn(m) {
                Some(mv) => {
                    e.insert(mv);
                }
                None => continue,
            }
        }
        stack.push((a, m));
        stack.push((m, b));
    }
    let mut runs: Vec<(Move, i16, i16)> = Vec::new();
    for (&ext, &mv) in &samples {
        match runs.last_mut() {
            Some((m, _lo, h)) if *m == mv => *h = ext,
            _ => runs.push((mv, ext, ext)),
        }
    }
    if let Some(first) = runs.first_mut() {
        first.1 = lo;
    }
    if let Some(last) = runs.last_mut() {
        last.2 = hi;
    }
    runs
}

/// Initial running counts to evaluate before bisection: the two ends plus a few interior anchors. More
/// than the bare ends so a hidden interior segment (a non-monotone argmax flip) has to dodge several
/// anchored points to be missed; cheap because every category and ladder shares the evaluations.
fn seed_points(lo: i16, hi: i16) -> Vec<i16> {
    if hi <= lo {
        return vec![lo];
    }
    let span = hi - lo;
    if span <= 4 {
        return (lo..=hi).collect();
    }
    let mut v: Vec<i16> = (0..=4).map(|k| lo + (span * k) / 4).collect();
    v.dedup();
    v
}

/// Which move ladder a [`coalesce_runs`] pass reads off each solved column.
#[derive(Clone, Copy)]
enum Ladder {
    /// The cell's headline move (argmax over every legal two-card move).
    Primary,
    /// Hit vs Stand only — the fallback once a start-only move is off the table.
    HitStand,
}

/// A column's count-index evaluator, over either count family (`kind`).
///
/// **Running (KO):** a windowed band of count-conditioned shoes (one per integer external running count,
/// all sharing one deconvolution — see [`CountShoe::band_external`]). Move lookups follow the
/// Wizard-of-Odds convention (the running count *includes* the cell's hand and the up-card — see
/// [`merge_count_frames`]): the move at index-count `ext` is read off a merged column whose hands each
/// come from the band shoe at `ext - map(U) - map(hand)`; those per-band solves (`build_evs` + reach) are
/// memoized in `frames` (disk-backed — see [`ensure_frame`]) and shared across index-counts. The move at
/// `ext` is the best move conditioned on the count being *exactly* `ext`.
///
/// **TrueCount (Hi-Lo):** an exact true count is measure-zero, so there is no exact-count conditioning.
/// Instead the move at integer true count `ext` is read straight off the **one-sided conditioned solve on
/// the side `ext` sits**: `TC ≥ ext` for `ext ≥ 0`, `TC ≤ ext` for `ext < 0` (see
/// [`true_summary`](Self::true_summary)). This is exactly the column the chart shows when the player
/// imposes that same `TC` constraint, so the reported index threshold and the conditioned chart agree by
/// construction — a deviation is declared at the count where `TC ≥ c` first recommends it, not one count
/// later (the earlier slice-differencing was conservative by a count). Each solve is the WoO-merged
/// [`solve_counted`]`::<HiLo>` conditioned column, memoized in `summaries`. No band/`frames` are used here.
///
/// `summaries` memoizes the per-`ext` summary so all categories and both ladders reuse it.
struct ColumnEval {
    kind: CountKind,
    n: u8,
    up: Card,
    rules: Ruleset,
    /// The integer count axis, ascending: usable external running counts (Running) or the true-count
    /// search window (TrueCount). For Running it is aligned with `shoes`.
    externals: Vec<i16>,
    /// Running only: the per-count band shoes aligned with `externals`. Empty for TrueCount.
    shoes: Vec<CountShoe>,
    /// The chart marker's notable count band (mass core ∪ pivot margin — see [`IndexReport::mark_lo`]),
    /// clamped into the usable window. Defaults to the full window for fixed-width test builds.
    mark_lo: i16,
    mark_hi: i16,
    /// Running only: per band external count, the (memoized) all-shift solve there (`build_evs` tree +
    /// reach weights) the WoO merge reads. Filled on demand by [`ensure_frame`], disk-first.
    frames: HashMap<i16, Frame>,
    /// Per index-count `ext`: the per-`ext` chart summary read by the ladders (WoO-merged exact-count for
    /// Running; the one-sided `TC ≥/≤ ext` conditioned column for TrueCount).
    summaries: HashMap<i16, ColumnSummary>,
}

impl ColumnEval {
    /// Build the band over the occurrence-bounded window: the central span of the running-count
    /// occurrence distribution that keeps all but [`INDEX_TAIL_MASS`] of the mass per tail (so we solve
    /// every count realistically held and leave the rare extremes open-ended), widened to always contain
    /// the marker's notable band (mass core ∪ pivot margin — see [`INDEX_MARKER_PIVOT_MARGIN`]) so every
    /// marked flip is visible in the popup. The marker band is then clamped inside the reachable window.
    /// `None` if nothing is reachable.
    fn new(kind: CountKind, n: u8, up: Card, rules: &Ruleset) -> Option<Self> {
        match kind {
            CountKind::Running => Self::new_running(n, up, rules),
            CountKind::TrueCount => Self::new_true(n, up, rules),
        }
    }

    fn new_running(n: u8, up: Card, rules: &Ruleset) -> Option<Self> {
        let dist = CountShoe::external_count_distribution::<Ko>(n, COUNT_PENETRATION);
        let pivot = Ko::pivot(n);
        // The marker's notable band: the commonly-held occurrence-mass core, always extended to cover a
        // fixed margin either side of the pivot so the advantage-region (high-count) deviations stay
        // flagged despite the distribution's skew. See INDEX_MARKER_PIVOT_MARGIN.
        let (mass_lo, mass_hi) = occurrence_window(&dist, INDEX_MARKER_TAIL_MASS);
        let mark_lo = mass_lo.min(pivot - INDEX_MARKER_PIVOT_MARGIN);
        let mark_hi = mass_hi.max(pivot + INDEX_MARKER_PIVOT_MARGIN);
        // Solved/popup window: the wider realistically-reachable span, widened to always contain the
        // marker band (so every marked flip is visible in the popup).
        let (mut lo, mut hi) = occurrence_window(&dist, INDEX_TAIL_MASS);
        lo = lo.min(mark_lo);
        hi = hi.max(mark_hi);
        let mut eval = Self::build_running(n, up, rules, lo, hi)?;
        // Clamp into the actually-solved (reachable) window — `build` may drop unreachable edge counts —
        // so the band never references an unsolved count.
        eval.mark_lo = mark_lo.clamp(eval.lo(), eval.hi());
        eval.mark_hi = mark_hi.clamp(eval.lo(), eval.hi());
        Some(eval)
    }

    /// Build the true-count evaluator over the occurrence-bounded integer-TC window. The window is the
    /// probability-motivated central span of the true-count occurrence distribution (same
    /// [`INDEX_TAIL_MASS`] tail bound as the running-count sweep — the true count's spread is far narrower,
    /// so this is naturally a handful of integers around 0), widened to the marker band (pivot 0 ±
    /// [`INDEX_MARKER_PIVOT_MARGIN`]). No band shoes: each `TC ≥ ext` solve is an on-demand
    /// [`solve_counted`] in [`summary`](Self::summary).
    fn new_true(n: u8, up: Card, rules: &Ruleset) -> Option<Self> {
        let dist = CountShoe::true_count_distribution::<HiLo>(n, COUNT_PENETRATION);
        if dist.is_empty() {
            return None;
        }
        let pivot = 0; // balanced ⇒ the zero-edge true count is 0
        let (mass_lo, mass_hi) = occurrence_window(&dist, INDEX_MARKER_TAIL_MASS);
        let mark_lo = mass_lo.min(pivot - INDEX_MARKER_PIVOT_MARGIN);
        let mark_hi = mass_hi.max(pivot + INDEX_MARKER_PIVOT_MARGIN);
        let (mut lo, mut hi) = occurrence_window(&dist, INDEX_TAIL_MASS);
        lo = lo.min(mark_lo);
        hi = hi.max(mark_hi);
        Some(Self {
            kind: CountKind::TrueCount,
            n,
            up,
            rules: *rules,
            externals: (lo..=hi).collect(),
            shoes: Vec::new(),
            mark_lo: mark_lo.clamp(lo, hi),
            mark_hi: mark_hi.clamp(lo, hi),
            frames: HashMap::new(),
            summaries: HashMap::new(),
        })
    }

    /// Build a **running-count** evaluator over an explicit inclusive external-count window `[lo, hi]`,
    /// dropping the counts whose exact condition has no mass (unreachable under the penetration prior — a
    /// zero draw distribution would make the solve meaningless). The reachable set is contiguous. `None`
    /// if nothing is reachable. KO-specific by construction (KO band shoes, exact-count slicing), so the
    /// `kind` it stamps is always [`CountKind::Running`]; the true-count path shares none of this and
    /// builds its struct inline in [`new_true`](Self::new_true).
    fn build_running(n: u8, up: Card, rules: &Ruleset, lo: i16, hi: i16) -> Option<Self> {
        let window: Vec<i16> = (lo..=hi).collect();
        let shoes = CountShoe::band_external::<Ko>(n, &window, CountCmp::Eq, COUNT_PENETRATION);
        let mut externals = Vec::new();
        let mut usable = Vec::new();
        for (&e, shoe) in window.iter().zip(shoes) {
            // A reachable exact-count shoe's draw distribution sums to 1; an unreachable one is all-zero.
            let mass: f64 = shoe.all_draw_probs().map(|(_, p)| p).sum();
            if mass > 0.5 {
                externals.push(e);
                usable.push(shoe);
            }
        }
        if externals.is_empty() {
            return None;
        }
        // Default the marker band to the full solved window; `new` narrows it to the occurrence-mass
        // band, while fixed-width test builds keep the whole window.
        let (mark_lo, mark_hi) = (externals[0], externals[externals.len() - 1]);
        Some(Self {
            kind: CountKind::Running,
            n,
            up,
            rules: *rules,
            externals,
            shoes: usable,
            mark_lo,
            mark_hi,
            frames: HashMap::new(),
            summaries: HashMap::new(),
        })
    }

    /// As [`new`](Self::new) but over a fixed pivot-centered half-width window, so a measurement can
    /// sweep the window size independent of the occurrence bound. Test-only.
    #[cfg(test)]
    fn new_windowed(n: u8, up: Card, rules: &Ruleset, half_width: i16) -> Option<Self> {
        let pivot = Ko::pivot(n);
        Self::build_running(n, up, rules, pivot - half_width, pivot + half_width)
    }

    fn lo(&self) -> i16 {
        self.externals[0]
    }

    fn hi(&self) -> i16 {
        self.externals[self.externals.len() - 1]
    }

    /// Ensure the all-shift `build_evs` tree and reach weights for the band shoe at external count `c`
    /// are present in the in-column memo, clamping `c` into the usable (contiguous) window so a frame
    /// lookup near a window edge still lands on a solved count. Disk-first: a persisted frame is loaded
    /// (skipping the dominant re-solve cost); a miss solves on the band shoe — preserving the band's
    /// shared deconvolution — and persists the result at frame granularity. Returns the clamped key.
    fn ensure_frame(&mut self, c: i16) -> i16 {
        let c = c.clamp(self.lo(), self.hi());
        if !self.frames.contains_key(&c) {
            let (up, n, rules) = (self.up, self.n, self.rules);
            // The band sweep is solved on KO (running-count) shoes; tag the key so it can never alias a
            // future true-count frame at the same numeric count.
            let key: FrameKey = (up, n, c, rules, Ko::KIND);
            let frame = diskcache::load::<_, Frame>(FRAME_KIND, &key).unwrap_or_else(|| {
                let idx = self.externals.iter().position(|&e| e == c).unwrap();
                let shoe = self.shoes[idx].clone();
                let tree = build_evs(shoe.clone(), up, &rules);
                let reach = reach_weights(shoe, up, &rules, &tree, true);
                let frame = (tree, reach);
                diskcache::store(FRAME_KIND, &key, &frame);
                frame
            });
            self.frames.insert(c, frame);
        }
        c
    }

    /// The per-`ext` chart summary (memoized): WoO-merged exact-count for Running, the one-sided
    /// `TC ≥/≤ ext` conditioned column for TrueCount.
    fn summary(&mut self, ext: i16) -> &ColumnSummary {
        if !self.summaries.contains_key(&ext) {
            let summary = match self.kind {
                CountKind::Running => self.running_summary(ext),
                CountKind::TrueCount => self.true_summary(ext),
            };
            self.summaries.insert(ext, summary);
        }
        &self.summaries[&ext]
    }

    /// The WoO-merged exact-count summary at running count `ext` (Running): each hand read from the band
    /// frame at `ext - map(U) - map(hand)`, so the running count `ext` includes the hand and the up-card.
    fn running_summary(&mut self, ext: i16) -> ColumnSummary {
        let mu = Ko::map(&self.up);
        let (lo, hi) = (self.lo(), self.hi());
        // Solve every frame the merge will read first (mutably), so it can then borrow them all
        // immutably. `frame_key` reads only locals, so it does not borrow `self`.
        for k in COUNT_GROUPS {
            self.ensure_frame(ext - mu - k);
        }
        let frame_key = |k: i16| (ext - mu - k).clamp(lo, hi);
        // `ensure_frame` above guarantees each band frame is present.
        let frame = |k: i16| &self.frames[&frame_key(k)];
        let (mt, mr) = merge_count_frames::<Ko>(|k| &frame(k).0, |k| &frame(k).1);
        summarize_cells(&mt, &mr)
    }

    /// The one-sided conditioned chart summary on the side `ext` sits (TrueCount): the WoO-merged
    /// [`solve_counted`]`::<HiLo>` column at the integer true-count cutoff `ext` (passed in half-units),
    /// compared `TC ≥ ext` for `ext ≥ 0` and `TC ≤ ext` for `ext < 0`. This is the very column the chart
    /// shows under that `TC` constraint, so its headline is read directly as the index move — the reported
    /// threshold and the conditioned chart agree by construction.
    fn true_summary(&self, ext: i16) -> ColumnSummary {
        let cmp = if ext >= 0 { CountCmp::Ge } else { CountCmp::Le };
        solve_counted::<HiLo>(self.n, TC_HALF_UNITS * ext, cmp, self.up, &self.rules).summary
    }

    /// The move `ladder` recommends for `cat` at index-count `ext`, or `None` if the category is absent.
    /// Read straight off the per-`ext` [`summary`](Self::summary) (exact-count for Running, the one-sided
    /// `TC ≥/≤ ext` conditioned column for TrueCount). `coalesce_runs` pins the flips between adjacent
    /// `ext` exactly.
    fn move_at(&mut self, ext: i16, cat: HandCategory, ladder: Ladder) -> Option<Move> {
        let ci = self.summary(ext).get(&cat)?;
        match ladder {
            Ladder::Primary => Some(ci.headline),
            Ladder::HitStand => {
                let h = ci.move_evs.get(&Move::Hit).copied();
                let s = ci.move_evs.get(&Move::Stand).copied();
                match (h, s) {
                    (Some(h), Some(s)) => Some(if h >= s { Move::Hit } else { Move::Stand }),
                    (Some(_), None) => Some(Move::Hit),
                    (None, Some(_)) => Some(Move::Stand),
                    (None, None) => None,
                }
            }
        }
    }

    fn runs(&mut self, cat: HandCategory, ladder: Ladder) -> Vec<(Move, i16, i16)> {
        let (lo, hi) = (self.lo(), self.hi());
        coalesce_runs(lo, hi, |ext| self.move_at(ext, cat, ladder))
    }

    /// The full count-index ladder for one category: the headline runs, plus the Hit/Stand fallback
    /// runs whenever the headline ever recommends a start-only move.
    fn category_index(&mut self, cat: HandCategory) -> CategoryIndex {
        let primary = self.runs(cat, Ladder::Primary);
        let fallback = if primary.iter().any(|&(m, _, _)| is_start_only(m)) {
            self.runs(cat, Ladder::HitStand)
        } else {
            Vec::new()
        };
        // `basic`/`basic_fallback` are filled by the caller from the no-count base chart (they are not a
        // function of the count window). Left `None` here.
        CategoryIndex {
            primary,
            fallback,
            basic: None,
            basic_fallback: None,
        }
    }
}

/// The no-count basic-strategy moves for a category's two ladders: the base-chart headline and its
/// Hit-vs-Stand argmax (the play once a start-only move is off the table). These flag which true-count
/// runs are *deviations*. Read off the uncounted base column (disk-cached — usually the very column the
/// chart already solved, so effectively free).
fn basic_moves(base: &ColumnSummary, cat: HandCategory) -> (Option<Move>, Option<Move>) {
    let Some(cell) = base.get(&cat) else {
        return (None, None);
    };
    let h = cell.move_evs.get(&Move::Hit).copied();
    let s = cell.move_evs.get(&Move::Stand).copied();
    let fallback = match (h, s) {
        (Some(h), Some(s)) => Some(if h >= s { Move::Hit } else { Move::Stand }),
        (Some(_), None) => Some(Move::Hit),
        (None, Some(_)) => Some(Move::Stand),
        (None, None) => None,
    };
    (Some(cell.headline), fallback)
}

/// The chart's categories split into the cheap (Hard/Soft — no split solves) and the expensive (Pairs)
/// halves, so a background index worker can stream the cheap markers first.
fn index_categories() -> (Vec<HandCategory>, Vec<HandCategory>) {
    let mut light = Vec::new();
    let mut pairs = Vec::new();
    for pane in PANES {
        for (cat, _) in pane.rows() {
            match cat {
                HandCategory::Pair(_) => pairs.push(cat),
                _ => light.push(cat),
            }
        }
    }
    (light, pairs)
}

/// Compute one up-card column's count-index report and stream it: the Hard/Soft markers first (a
/// partial report), then the Pairs, completing it. Tagged with `epoch` so a stale basis is dropped.
pub(super) fn compute_index_report(
    n: u8,
    key: IndexKey,
    rules: &Ruleset,
    epoch: u64,
    tx: &Sender<IndexResult>,
) {
    // Disk cache hit: a count-index report is fully determined by its key, so a persisted complete one
    // is reused wholesale — skipping the whole root-finder and its many count-conditioned solves (the
    // dominant cost of the background fill).
    if let Some(report) = diskcache::load::<_, IndexReport>("index", &key)
        && report.complete
    {
        let _ = tx.send(IndexResult { epoch, key, report });
        return;
    }
    let up = key.0;
    // The index family comes from the key (the chart's selected counting system), so the markers match
    // the active system and never alias a cached report from the other.
    let kind = key.3;
    let Some(mut eval) = ColumnEval::new(kind, n, up, rules) else {
        return;
    };
    // The true-count display gives the *deviation* (a play differing from no-count basic strategy) the
    // inclusive side of its cutoff, so a true-count report needs the basic-strategy moves. They come from
    // the uncounted base column (system-independent, disk-cached — usually already solved for the chart).
    // The running-count display keeps its offset boundaries, so it skips this solve.
    let base = (kind == CountKind::TrueCount)
        .then(|| ShoeChoice::Decks(n).solve(up, rules, None).summary);
    let set_basic = |ci: &mut CategoryIndex, cat| {
        if let Some(base) = &base {
            (ci.basic, ci.basic_fallback) = basic_moves(base, cat);
        }
    };
    let (light, pairs) = index_categories();
    let mut report = IndexReport {
        lo: eval.lo(),
        hi: eval.hi(),
        mark_lo: eval.mark_lo,
        mark_hi: eval.mark_hi,
        cats: HashMap::new(),
        complete: false,
    };
    for cat in light {
        let mut ci = eval.category_index(cat);
        set_basic(&mut ci, cat);
        report.cats.insert(cat, ci);
    }
    if tx
        .send(IndexResult {
            epoch,
            key,
            report: report.clone(),
        })
        .is_err()
    {
        return; // receiver gone (app exiting)
    }
    for cat in pairs {
        let mut ci = eval.category_index(cat);
        set_basic(&mut ci, cat);
        report.cats.insert(cat, ci);
    }
    report.complete = true;
    diskcache::store("index", &key, &report);
    let _ = tx.send(IndexResult { epoch, key, report });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The root-finder on a synthetic monotone curve: Hit below 0, Stand at/above, over `[-20, 20]`.
    /// It must return exactly two runs with the flip pinned to the integer boundary (Hit ends at −1,
    /// Stand starts at 0), and stretch the ends to the window edges. Fast (no solving), so not ignored.
    #[test]
    fn coalesce_runs_finds_exact_flip() {
        let runs = coalesce_runs(-20, 20, |rc| {
            Some(if rc < 0 { Move::Hit } else { Move::Stand })
        });
        assert_eq!(runs, vec![(Move::Hit, -20, -1), (Move::Stand, 0, 20)]);
    }

    /// A constant move over the window collapses to a single run (no count dependence).
    #[test]
    fn coalesce_runs_constant_is_single_run() {
        let runs = coalesce_runs(-20, 20, |_| Some(Move::Stand));
        assert_eq!(runs, vec![(Move::Stand, -20, 20)]);
    }

    /// The root-finder pins two flips of a three-segment ladder (Hit / Stand / Hit) to their exact
    /// integer boundaries — the interior segment is found despite both ends being Hit, thanks to the
    /// interior seed anchors.
    #[test]
    fn coalesce_runs_three_segments() {
        let runs = coalesce_runs(-20, 20, |rc| {
            Some(if (-5..5).contains(&rc) {
                Move::Stand
            } else {
                Move::Hit
            })
        });
        assert_eq!(
            runs,
            vec![
                (Move::Hit, -20, -6),
                (Move::Stand, -5, 4),
                (Move::Hit, 5, 20),
            ]
        );
    }

    /// End-to-end count-index regression on the full count-conditioned solve. Pins the canonical KO
    /// deviation for 16 vs Ten (single deck): the primary ladder flips *with the count in the right
    /// direction* — Hit at low counts (deck rich in low cards, hitting stiff 16 is safe) giving way to a
    /// non-Hit (surrender/stand) as the count climbs and the deck richens in tens — and, since the
    /// non-Hit headline is surrender (a start-only move), a Hit/Stand fallback ladder is built behind
    /// it. `#[ignore]` because a count-conditioned column is seconds of work; run `--release --ignored`.
    #[test]
    #[ignore]
    fn count_index_16_vs_ten_flips_with_count() {
        let mut eval = ColumnEval::new(CountKind::Running, 1, Card::Ten, &Ruleset::default())
            .expect("usable window");
        let ci = eval.category_index(HandCategory::Hard(16));
        assert!(
            ci.primary.len() >= 2,
            "expected a count deviation for 16 vs T, got {:?}",
            ci.primary
        );
        assert_eq!(
            ci.primary.first().unwrap().0,
            Move::Hit,
            "low count should Hit"
        );
        assert_ne!(
            ci.primary.last().unwrap().0,
            Move::Hit,
            "high count should deviate off Hit"
        );
        if ci.primary.iter().any(|&(m, _, _)| is_start_only(m)) {
            assert!(
                !ci.fallback.is_empty(),
                "a start-only headline move needs a Hit/Stand fallback ladder"
            );
        }
    }

    /// The **true-count** analog, end-to-end through the Hi-Lo path (`solve_counted::<HiLo>` + the
    /// inequality slice differencing): 16 vs Ten (single deck) must flip off Hit as the true count
    /// climbs, and the boundary must sit near the canonical Hi-Lo index of 0 (Illustrious 18: stand
    /// 16v10 at TC ≥ 0). This exercises the whole TrueCount evaluator — the occurrence window, the
    /// `TC ≥ c` conditioned solves, and `coalesce_runs` over the slice move — that the KO test above does
    /// not. `#[ignore]` (count-conditioned solves); run `--release --ignored`.
    #[test]
    #[ignore]
    fn true_count_index_16_vs_ten_flips_near_zero() {
        let mut eval = ColumnEval::new(CountKind::TrueCount, 1, Card::Ten, &Ruleset::default())
            .expect("usable window");
        let ci = eval.category_index(HandCategory::Hard(16));
        assert!(
            ci.primary.len() >= 2,
            "expected a true-count deviation for 16 vs T, got {:?}",
            ci.primary
        );
        assert_eq!(
            ci.primary.first().unwrap().0,
            Move::Hit,
            "low TC should Hit"
        );
        assert_ne!(
            ci.primary.last().unwrap().0,
            Move::Hit,
            "high TC should deviate off Hit"
        );
        let boundary = ci.primary.first().unwrap().2; // last TC at which Hit is still best
        assert!(
            (-4..=4).contains(&boundary),
            "Hit→deviation boundary should sit near TC 0 (Hi-Lo index), got Hit up to {boundary}"
        );
    }

    /// The bug this redesign fixes: at Hard 12 vs 3 (4 decks) the play flips Hit→Stand near RC −3, but
    /// the old window (centered on the player's +4 count) reported "stand at any RC", contradicting the
    /// no-count base table (Hit). The count-independent report must show both runs with the boundary in
    /// the right place. `#[ignore]` (count-conditioned solve); run `--release --ignored`.
    #[test]
    #[ignore]
    fn count_index_12_vs_3_flips_near_neg3() {
        let mut eval = ColumnEval::new(CountKind::Running, 4, Card::Pip(3), &Ruleset::default())
            .expect("usable window");
        let ci = eval.category_index(HandCategory::Hard(12));
        assert!(
            ci.primary.len() >= 2,
            "expected a Hit→Stand flip for 12 vs 3, got {:?}",
            ci.primary
        );
        assert_eq!(ci.primary.first().unwrap().0, Move::Hit, "low count Hits");
        assert_eq!(
            ci.primary.last().unwrap().0,
            Move::Stand,
            "high count Stands"
        );
        let boundary = ci.primary.first().unwrap().2; // last RC at which Hit is still best
        assert!(
            (-7..=0).contains(&boundary),
            "Hit→Stand boundary should sit near RC −3, got Hit up to {boundary}"
        );
    }

    /// Regression for the soft-double deviation marker: A5 vs 3 (soft 16) deviates Hit→Double as the
    /// count climbs, and that flip must land inside the marker band so the chart draws its `°` — even on
    /// a single deck, where the flip sits at RC +5. The old fixed `|RC| ≤ 4` window dropped exactly that
    /// (the flip was *visible in the popup* but unmarked on the chart), which the pivot-anchored band
    /// (reaching pivot + [`INDEX_MARKER_PIVOT_MARGIN`] = +9) restores. `#[ignore]` (count-conditioned
    /// solve); run `--release --ignored`.
    #[test]
    #[ignore]
    fn count_index_soft16_vs_3_marks_single_deck() {
        let mut eval = ColumnEval::new(CountKind::Running, 1, Card::Pip(3), &Ruleset::default())
            .expect("usable window");
        let (mlo, mhi) = (eval.mark_lo, eval.mark_hi);
        let ci = eval.category_index(HandCategory::Soft(16));
        assert!(
            ci.primary.iter().any(|&(m, _, _)| m == Move::Double),
            "expected a Hit→Double deviation, got primary={:?}",
            ci.primary
        );
        assert!(
            ci.count_dependent_in_band(mlo, mhi),
            "soft 16 vs 3 must be marked; primary={:?} band=[{mlo},{mhi}]",
            ci.primary
        );
    }

    /// PROTOTYPE MEASUREMENT (not an assertion): for each deck count, print the external running-count
    /// occurrence distribution's full reachable span and the [`occurrence_window`] at a few tail
    /// thresholds — i.e. how wide the count-index sweep needs to be to cover all-but-the-practically-
    /// impossible counts. The width (`hi − lo + 1`) is the per-column frame-solve cost, so this is the
    /// data that justifies [`INDEX_TAIL_MASS`] by likelihood. The `*` marks the production threshold.
    /// Fast (no solving — just `CountW` builds).
    /// Run: `cargo test --release occurrence_window_by_decks -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn occurrence_window_by_decks() {
        let pivot_note = "(KO pivot is +4 for every deck count)";
        println!(
            "external running-count occurrence windows, pen={COUNT_PENETRATION:?} {pivot_note}"
        );
        println!("production threshold INDEX_TAIL_MASS = {INDEX_TAIL_MASS} per tail (marked *)");
        println!(
            "marker band: mass core (drop {INDEX_MARKER_TAIL_MASS}/tail) \u{222a} pivot \u{00b1}{INDEX_MARKER_PIVOT_MARGIN} (marked +)"
        );
        for n in [1u8, 2, 4, 6, 8] {
            let dist = CountShoe::external_count_distribution::<Ko>(n, COUNT_PENETRATION);
            let (flo, fhi) = (dist.first().unwrap().0, dist.last().unwrap().0);
            let peak = dist
                .iter()
                .cloned()
                .fold((0i16, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            println!(
                "n={n}: full span=[{flo:>3},{fhi:>3}] (width {:>3})  peak c={:>2} P={:.3}",
                fhi - flo + 1,
                peak.0,
                peak.1
            );
            for tail in [1e-2, 1e-3, 1e-4] {
                let (lo, hi) = occurrence_window(&dist, tail);
                let mark = if tail == INDEX_TAIL_MASS { "*" } else { " " };
                println!(
                    "  {mark} drop {tail:>7} each tail -> [{lo:>3},{hi:>3}]  width {:>3}",
                    hi - lo + 1
                );
            }
            // The production marker band: mass core ∪ pivot ± margin (the actual `°` cue range).
            let pivot = Ko::pivot(n);
            let (mass_lo, mass_hi) = occurrence_window(&dist, INDEX_MARKER_TAIL_MASS);
            let (mlo, mhi) = (
                mass_lo.min(pivot - INDEX_MARKER_PIVOT_MARGIN),
                mass_hi.max(pivot + INDEX_MARKER_PIVOT_MARGIN),
            );
            println!(
                "  + marker band -> [{mlo:>3},{mhi:>3}]  width {:>3}",
                mhi - mlo + 1
            );
        }
    }

    /// One column's frame-origin breakdown at a given window half-width: how many distinct band-count
    /// frames (the expensive `build_evs`+reach solves) were touched, how many of those are the
    /// *seed floor* (forced by the fixed seed grid regardless of any flips — `seed_points` depends only
    /// on `lo`/`hi`, so it is identical for every category), and how many distinct index-counts (`ext`)
    /// were sampled. `flip = frames - seed_floor` is the work attributable to actual play deviations.
    struct ColumnBreakdown {
        externals: usize,
        exts_sampled: usize,
        frames: usize,
        seed_floor: usize,
    }

    /// Fill one up-card column (all categories) at `half_width` and report its [`ColumnBreakdown`].
    fn column_breakdown(n: u8, up: Card, rules: &Ruleset, half_width: i16) -> ColumnBreakdown {
        let mut eval = ColumnEval::new_windowed(n, up, rules, half_width).expect("usable window");
        let (light, pairs) = index_categories();
        for cat in light.into_iter().chain(pairs) {
            let _ = eval.category_index(cat);
        }
        // The seed floor: the band frames the fixed seed grid alone forces, independent of any flip.
        // Every category seeds the same `seed_points(lo, hi)`; each seed `ext` reads COUNT_GROUPS frames
        // at `clamp(ext - mu - k)`. Their union is the unavoidable cost of just *probing* the window.
        let (lo, hi) = (eval.lo(), eval.hi());
        let mu = Ko::map(&up);
        let mut floor: std::collections::BTreeSet<i16> = std::collections::BTreeSet::new();
        for s in seed_points(lo, hi) {
            for k in COUNT_GROUPS {
                floor.insert((s - mu - k).clamp(lo, hi));
            }
        }
        ColumnBreakdown {
            externals: eval.externals.len(),
            exts_sampled: eval.summaries.len(),
            frames: eval.frames.len(),
            seed_floor: floor.len(),
        }
    }

    /// PROTOTYPE MEASUREMENT (not an assertion): for each window half-width, fill every up-card column
    /// and print the per-column frame-solve breakdown plus column-summed totals — the data that decides
    /// how much a narrower window recovers, and how much of the cost is the seed floor vs. real flip
    /// brackets. Counts are exact regardless of disk warmth (a disk hit still records the frame), so this
    /// can run over a warm cache. Run alone:
    /// `cargo test --release frame_origin_breakdown -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore]
    fn frame_origin_breakdown() {
        let rules = Ruleset::default();
        let n = 1u8;
        for half_width in [20i16, 10, 6] {
            let mut tot_frames = 0usize;
            let mut tot_floor = 0usize;
            let mut tot_exts = 0usize;
            println!("=== n={n}  half_width=±{half_width} ===");
            for &up in &crate::tui::UP_CARDS {
                let b = column_breakdown(n, up, &rules, half_width);
                let flip = b.frames.saturating_sub(b.seed_floor);
                println!(
                    "  up={up:?}: window={:>3}  exts={:>3}  frames={:>3}  (seed_floor={:>2}  flip={:>2})",
                    b.externals, b.exts_sampled, b.frames, b.seed_floor, flip
                );
                tot_frames += b.frames;
                tot_floor += b.seed_floor;
                tot_exts += b.exts_sampled;
            }
            println!(
                "  TOTAL: frames={tot_frames}  exts={tot_exts}  seed_floor={tot_floor}  \
                 flip={}\n",
                tot_frames.saturating_sub(tot_floor)
            );
        }
    }
}
