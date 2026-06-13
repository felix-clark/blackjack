//! Count-index thresholds: the running counts at which a cell's recommended move flips. These are a
//! count-*independent* property of each `(up-card, shoe, ruleset)` cell — the player's count only says
//! where they sit on the ladder — so they are computed once over the whole reachable count axis
//! ([`ColumnEval`]/[`coalesce_runs`], monotone root-finding sharing one deconvolution across the band)
//! and overlaid on the base chart. [`compute_index_report`] is the background-worker entry point.

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::Sender;

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::count::{CountCmp, CountShoe, CountSystem, Ko};
use crate::diskcache;
use crate::hand::{HandCategory, Move};
use crate::reach::{reach_weights, summarize_cells};
use crate::rules::Ruleset;
use crate::shoe::Shoe;
use crate::simulation::build_evs;

use super::PANES;
use super::column::{COUNT_GROUPS, ColumnSummary, ReachMap, Tree, merge_count_frames};
use super::config::{COUNT_PENETRATION, ShoeChoice};

/// One solved frame: the `build_evs` EV tree and its game-time reach weights — the unit the count-index
/// WoO merge reads off (one per band count). Persisted per [`FrameKey`] so a warm relaunch skips the
/// re-solve, which is the dominant cost of a cold fill; in memory it is memoized only *within* a column
/// (the band counts a column's WoO merges share), since distinct up-cards never share a frame.
type Frame = (Tree, ReachMap);

/// What a persisted frame is keyed under: up-card, deck count, and exact external running count. The
/// band's comparison is always `Eq` and its penetration prior a fixed constant
/// ([`COUNT_PENETRATION`]), so neither is in the key; the ruleset is.
type FrameKey = (Card, u8, i16, Ruleset);

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

/// Chart `°` markers are only drawn for cells whose play actually shifts within a *notable* running
/// count: roughly `|RC| ≤` this. A flip that only triggers at an extreme count (splitting tens vs 2 at
/// RC ≈ +18, say) is suppressed on the chart — it is vanishingly rare in real play and acting on it
/// would be conspicuous — but the full ladder is still shown in the popup. Stand-in for a future live
/// "marker sensitivity" control; the popup is always exhaustive regardless.
pub(super) const INDEX_MARKER_MAX_RC: i16 = 4;

/// Whether a move is only available as the opening action on a two-card hand (so it disappears once the
/// hand has been hit). These are exactly the headline moves a [`CategoryIndex`] gives a *fallback*
/// Hit/Stand ladder for: "surrender below RC −1" is incomplete without "…and once you can't surrender,
/// stand at RC ≥ …".
fn is_start_only(mv: Move) -> bool {
    matches!(mv, Move::Double | Move::Split | Move::Surrender)
}

/// Whether an ascending `(move, lo, hi)` run list has a play change whose *flip point* — the running
/// count at which the later move takes over, i.e. the `lo` of each run after the first — falls inside
/// the inclusive window `[-max_abs_rc, max_abs_rc]`. Used by the chart marker to ignore flips that only
/// happen at extreme counts.
///
/// This keys on the boundary, not on which moves are *visible* in the window: a Hit→Stand index at
/// RC −4 fires even though its Hit leg lives entirely at RC ≤ −5 (outside the window). Counting
/// distinct in-window moves missed exactly that case — one leg of an in-window flip can sit just
/// outside it.
fn flips_within(runs: &[(Move, i16, i16)], max_abs_rc: i16) -> bool {
    runs.iter()
        .skip(1)
        .any(|&(_, lo, _)| lo.abs() <= max_abs_rc)
}

/// One chart category's count-dependent move ladder over the index window, in ascending running-count
/// order. `primary` is the headline move's `(move, lo, hi)` inclusive-RC runs (a single run ⇒ the move
/// never changes ⇒ no count dependence). `fallback` is the Hit-vs-Stand ladder that applies once a
/// start-only headline move is unavailable (a hand that has already been hit); it is populated only
/// when `primary` actually contains a start-only move.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(super) struct CategoryIndex {
    pub(super) primary: Vec<(Move, i16, i16)>,
    pub(super) fallback: Vec<(Move, i16, i16)>,
}

impl CategoryIndex {
    /// The cell's right play genuinely shifts with the running count within a *notable* window
    /// `|RC| ≤ max_abs_rc` — what the chart `°` marker keys on: either the headline flips or there is a
    /// Hit/Stand flip behind a start-only headline move, somewhere in the window. A ladder that is
    /// constant across the window (its only flips are at extreme, practically-unreachable counts) is
    /// treated as not count-dependent *for display*; the popup still renders the whole ladder.
    pub(super) fn count_dependent_within(&self, max_abs_rc: i16) -> bool {
        flips_within(&self.primary, max_abs_rc) || flips_within(&self.fallback, max_abs_rc)
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
    pub(super) cats: HashMap<HandCategory, CategoryIndex>,
    pub(super) complete: bool,
}

/// What an [`IndexReport`] is cached/keyed under. Deliberately *without* the player's count (the report
/// is count-independent) and without the chart's count comparison (the index is always exact-count
/// based).
pub(super) type IndexKey = (Card, ShoeChoice, Ruleset);

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

/// A column's count-index evaluator: the windowed band of count-conditioned shoes (one per integer
/// external count, all sharing one deconvolution — see [`CountShoe::band`]) plus memos of the work.
///
/// Move lookups follow the Wizard-of-Odds count convention (the running count *includes* the cell's hand
/// and the up-card — see [`merge_count_frames`]). The move at index-count `ext` is therefore read off a
/// merged column whose hands each come from the band shoe at `ext - map(U) - map(hand)`; those per-band
/// solves (`build_evs` + reach) are memoized in `frames` (disk-backed — see [`ensure_frame`]), so each
/// band count is solved at most once and shared across every index-count whose window reaches it.
/// `summaries` memoizes the merged per-`ext` summary so all categories and both ladders reuse it.
struct ColumnEval {
    n: u8,
    up: Card,
    rules: Ruleset,
    /// Usable (positive-mass) external counts, ascending; aligned with `shoes`.
    externals: Vec<i16>,
    shoes: Vec<CountShoe>,
    /// Per band external count: the (memoized) all-shift solve there (`build_evs` tree + reach weights).
    /// The per-frame inputs the WoO merge reads. Filled on demand by [`ensure_frame`], disk-first.
    frames: HashMap<i16, Frame>,
    /// Per index-count `ext`: the merged WoO chart summary read by the ladders.
    summaries: HashMap<i16, ColumnSummary>,
}

impl ColumnEval {
    /// Build the band over the occurrence-bounded window: the central span of the running-count
    /// occurrence distribution that keeps all but [`INDEX_TAIL_MASS`] of the mass per tail (so we solve
    /// every count realistically held and leave the rare extremes open-ended), clamped to always cover
    /// the marker range so no displayable flip can hide outside it. `None` if nothing is reachable.
    fn new(n: u8, up: Card, rules: &Ruleset) -> Option<Self> {
        let dist = CountShoe::external_count_distribution::<Ko>(n, COUNT_PENETRATION);
        let (mut lo, mut hi) = occurrence_window(&dist, INDEX_TAIL_MASS);
        // Never trim inside the marker range — a flip the chart would draw must be inside the window.
        lo = lo.min(-INDEX_MARKER_MAX_RC);
        hi = hi.max(INDEX_MARKER_MAX_RC);
        Self::build(n, up, rules, lo, hi)
    }

    /// Build over an explicit inclusive external-count window `[lo, hi]`, dropping the counts whose exact
    /// condition has no mass (unreachable under the penetration prior — a zero draw distribution would
    /// make the solve meaningless). The reachable set is contiguous. `None` if nothing is reachable.
    fn build(n: u8, up: Card, rules: &Ruleset, lo: i16, hi: i16) -> Option<Self> {
        let window: Vec<i16> = (lo..=hi).collect();
        let shoes = CountShoe::band::<Ko>(n, &window, CountCmp::Eq, COUNT_PENETRATION);
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
        Some(Self {
            n,
            up,
            rules: *rules,
            externals,
            shoes: usable,
            frames: HashMap::new(),
            summaries: HashMap::new(),
        })
    }

    /// As [`new`](Self::new) but over a fixed pivot-centered half-width window, so a measurement can
    /// sweep the window size independent of the occurrence bound. Test-only.
    #[cfg(test)]
    fn new_windowed(n: u8, up: Card, rules: &Ruleset, half_width: i16) -> Option<Self> {
        let pivot = Ko::pivot(n);
        Self::build(n, up, rules, pivot - half_width, pivot + half_width)
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
            let key: FrameKey = (up, n, c, rules);
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

    /// The WoO-merged chart summary at index-count `ext` (memoized): each hand read from the band frame
    /// at `ext - map(U) - map(hand)`, so the running count `ext` includes the hand and the up-card.
    fn summary(&mut self, ext: i16) -> &ColumnSummary {
        if !self.summaries.contains_key(&ext) {
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
            let (mt, mr) = merge_count_frames(|k| &frame(k).0, |k| &frame(k).1);
            let summary = summarize_cells(&mt, &mr);
            self.summaries.insert(ext, summary);
        }
        &self.summaries[&ext]
    }

    /// The move `ladder` recommends for `cat` at external count `ext`, or `None` if the category is
    /// absent (e.g. no Hit/Stand EVs to compare for the fallback).
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
        CategoryIndex { primary, fallback }
    }
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
    let Some(mut eval) = ColumnEval::new(n, up, rules) else {
        return;
    };
    let (light, pairs) = index_categories();
    let mut report = IndexReport {
        lo: eval.lo(),
        hi: eval.hi(),
        cats: HashMap::new(),
        complete: false,
    };
    for cat in light {
        let ci = eval.category_index(cat);
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
        let ci = eval.category_index(cat);
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
        let mut eval = ColumnEval::new(1, Card::Ten, &Ruleset::default()).expect("usable window");
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

    /// The bug this redesign fixes: at Hard 12 vs 3 (4 decks) the play flips Hit→Stand near RC −3, but
    /// the old window (centered on the player's +4 count) reported "stand at any RC", contradicting the
    /// no-count base table (Hit). The count-independent report must show both runs with the boundary in
    /// the right place. `#[ignore]` (count-conditioned solve); run `--release --ignored`.
    #[test]
    #[ignore]
    fn count_index_12_vs_3_flips_near_neg3() {
        let mut eval =
            ColumnEval::new(4, Card::Pip(3), &Ruleset::default()).expect("usable window");
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
        println!("external running-count occurrence windows, pen={COUNT_PENETRATION:?} {pivot_note}");
        println!("production threshold INDEX_TAIL_MASS = {INDEX_TAIL_MASS} per tail (marked *)");
        for n in [1u8, 2, 4, 6, 8] {
            let dist = CountShoe::external_count_distribution::<Ko>(n, COUNT_PENETRATION);
            let (flo, fhi) = (dist.first().unwrap().0, dist.last().unwrap().0);
            let peak = dist.iter().cloned().fold((0i16, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            println!(
                "n={n}: full span=[{flo:>3},{fhi:>3}] (width {:>3})  peak c={:>2} P={:.3}",
                fhi - flo + 1, peak.0, peak.1
            );
            for tail in [1e-2, 1e-3, 1e-4] {
                let (lo, hi) = occurrence_window(&dist, tail);
                let mark = if tail == INDEX_TAIL_MASS { "*" } else { " " };
                println!(
                    "  {mark} drop {tail:>7} each tail -> [{lo:>3},{hi:>3}]  width {:>3}",
                    hi - lo + 1
                );
            }
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
