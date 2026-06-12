//! The terminal UI (the *only* place ratatui is used). It renders the basic-strategy chart as three
//! side-by-side panes (Hard / Soft / Pairs), each a grid of strategy-table rows by dealer up-card,
//! navigable with vim motions; an EV popup shows the per-move EVs for the highlighted hand/up-card;
//! a rules modal edits the [`Ruleset`] (and deck count).
//!
//! Compute is asynchronous: each of the ten up-cards is solved on its own worker thread
//! ([`build_evs`] + [`summarize_cells`]) and the chart fills in column-by-column as results arrive, so
//! the interface never blocks. A monotonic `epoch` tags every batch; results from a superseded epoch
//! (a rules/deck change happened) are discarded rather than interrupting the worker.
//!
//! A card-counting condition (the `c` modal) conditions the solve on a running count: on a finite
//! shoe it swaps the plain [`CardCol`] for a [`CountShoe`] (exact count-conditioned main tree and
//! dealer; mean-field tilt inside splits). The condition is part of the chart cache key, so toggling
//! counts on/off or changing the running count re-solves (or restores from cache) like a rules change.
//! On a finite shoe the chart also shows **count-index thresholds** — the running counts at which the
//! recommended move flips. These are a count-*independent* property of each cell (the player's count
//! only says where they sit on the ladder), so they are computed once per `(up-card, shoe, ruleset)`
//! over the whole reachable count axis ([`ColumnEval`]/[`coalesce_runs`], monotone root-finding sharing
//! one deconvolution across the band), background-filled for every cell, marked with `°` on the chart,
//! and detailed in the popup.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::count::{CountCmp, CountShoe, CountSystem, Ko, Penetration};
use crate::diskcache;
use crate::hand::{HandCategory, Move};
use crate::reach::{CellInfo, reach_weights, summarize_cells};
use crate::rules::{BjPayout, PeekRule, PeekSurrender, Ruleset};
use crate::shoe::{CardCol, InfiniteDeck, Shoe};
use crate::simulation::{EdgeTerm, build_evs, edge_term};

/// Penetration prior used for count conditioning: a flat distribution over deck depth up to 75%
/// penetration (casinos never deal the shoe out). See the count-conditioning architecture notes.
const COUNT_PENETRATION: Penetration = Penetration::FlatPastPercent(25);

/// A card-counting condition the chart is solved under: a counting system (KO for now), the player's
/// external running count, and how it is compared. `None` of this is applied on the infinite deck (an
/// infinite deck has no count) or when counting is toggled off.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct CountSetting {
    /// The player's external running count value being conditioned on.
    external: i16,
    /// How the running count is compared to `external` (`==`, `≥`, `≤`).
    cmp: CountCmp,
}

impl CountSetting {
    fn cmp_label(self) -> &'static str {
        match self.cmp {
            CountCmp::Eq => "==",
            CountCmp::Ge => ">=",
            CountCmp::Le => "<=",
        }
    }
}

/// Dealer up-cards in chart-column order (2..9, T, A).
const UP_CARDS: [Card; 10] = [
    Card::Pip(2),
    Card::Pip(3),
    Card::Pip(4),
    Card::Pip(5),
    Card::Pip(6),
    Card::Pip(7),
    Card::Pip(8),
    Card::Pip(9),
    Card::Ten,
    Card::Ace,
];

/// Column solve order: indices into [`UP_CARDS`], longest-running column first. Measured per-column
/// solve time falls off monotonically from the Ace through the low cards to the Ten (the Ace peeks
/// and low up-cards make the dealer draw deeper), so `A, 2, 3, …, 9, T` is the longest-processing-time
/// schedule — it keeps the heavy work off the tail where it would run under-parallelised.
const SOLVE_ORDER: [usize; 10] = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8];

/// Moves in the fixed order the EV popup lists them.
const MOVE_ORDER: [Move; 5] = [
    Move::Stand,
    Move::Hit,
    Move::Double,
    Move::Split,
    Move::Surrender,
];

/// The shoe the chart is solved against: an infinite (non-depleting) deck or a finite `n`-deck shoe.
/// This is the seam a future card-counting input would adjust.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ShoeChoice {
    Infinite,
    Decks(u8),
}

impl std::fmt::Display for ShoeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShoeChoice::Infinite => write!(f, "\u{221e}"),
            ShoeChoice::Decks(n) => write!(f, "{n}"),
        }
    }
}

impl ShoeChoice {
    /// Solve one up-card's full EV tree on this shoe, collapsing it to the per-category strategy
    /// summary the chart renders and the two-card-root [`EdgeTerm`] the footer's overall edge sums.
    /// Both are read off the same tree, so the edge costs no extra solve. Runs on a worker thread.
    /// `count` conditions the solve on a card-counting running count; it only applies to a finite shoe
    /// (an infinite deck has no count) and is ignored when `None`.
    fn solve(self, up_card: Card, rules: &Ruleset, count: Option<CountSetting>) -> Column {
        // Disk cache: a solved column is fully determined by (up-card, shoe, ruleset, count condition),
        // so persist it — a revisited configuration loads instantly instead of re-solving (splits and
        // all). Best-effort; a miss/error just recomputes.
        let key = (up_card, self, *rules, count);
        if let Some(col) = diskcache::load::<_, Column>("column", &key) {
            return col;
        }
        let column = match self {
            ShoeChoice::Infinite => solve_on(InfiniteDeck {}, up_card, rules),
            ShoeChoice::Decks(n) => match count {
                Some(c) => solve_on(
                    CountShoe::from_external::<Ko>(n, c.external, c.cmp, COUNT_PENETRATION),
                    up_card,
                    rules,
                ),
                None => solve_on(CardCol::from_decks(n), up_card, rules),
            },
        };
        diskcache::store("column", &key, &column);
        column
    }
}

/// Solve and consolidate one up-card column on a concrete shoe `S`. Cells are consolidated by the
/// game-time **reaching weight** ([`reach_weights`] → [`summarize_cells`], split arms folded in): how
/// often each composition is actually the hand in front of a deciding player. Each cell's headline is
/// decided on its two-card decision population (so a start-only move is compared only against the
/// Hit/Stand EVs of hands that can take it), and carries its composition-dependence flag and
/// per-composition breakdown. `p_up` is the up-card's draw probability from the *full* shoe (before
/// `build_evs` removes it).
fn solve_on<S: Shoe + Clone + Eq + Hash + Sync>(shoe: S, up_card: Card, rules: &Ruleset) -> Column {
    let tree = build_evs(shoe.clone(), up_card, rules);
    let weights = reach_weights(shoe.clone(), up_card, rules, &tree, true);
    Column {
        summary: summarize_cells(&tree, &weights),
        p_up: shoe.draw_prob(&up_card),
        edge: edge_term(&tree),
    }
}

/// Half-width of the external-running-count window the count index is computed over, centered on the
/// system pivot (KO: `+4`). The window is count-*independent* — it does not depend on the player's
/// current count — and is wide enough to contain the realistic deviation flips (which cluster near the
/// pivot). Flips beyond it are reported open-ended (`≤`/`≥`). Root-finding keeps the *number of solves*
/// small regardless of width (it bisects to the flip rather than sweeping), so a generous window is
/// cheap; the whole band still shares one deconvolution (see [`CountShoe::band`]).
const INDEX_HALF_WIDTH: i16 = 20;

/// Max count-index columns solved concurrently in the background. Each is much heavier than a chart
/// column (a handful of count-conditioned solves, splits and all), so this is kept well below the ten
/// chart workers to avoid swamping the cores the in-column split parallelism already wants.
const INDEX_FILL_CONCURRENCY: usize = 3;

/// Chart `°` markers are only drawn for cells whose play actually shifts within a *notable* running
/// count: roughly `|RC| ≤` this. A flip that only triggers at an extreme count (splitting tens vs 2 at
/// RC ≈ +18, say) is suppressed on the chart — it is vanishingly rare in real play and acting on it
/// would be conspicuous — but the full ladder is still shown in the popup. Stand-in for a future live
/// "marker sensitivity" control; the popup is always exhaustive regardless.
const INDEX_MARKER_MAX_RC: i16 = 4;

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
struct CategoryIndex {
    primary: Vec<(Move, i16, i16)>,
    fallback: Vec<(Move, i16, i16)>,
}

impl CategoryIndex {
    /// The cell's right play genuinely shifts with the running count within a *notable* window
    /// `|RC| ≤ max_abs_rc` — what the chart `°` marker keys on: either the headline flips or there is a
    /// Hit/Stand flip behind a start-only headline move, somewhere in the window. A ladder that is
    /// constant across the window (its only flips are at extreme, practically-unreachable counts) is
    /// treated as not count-dependent *for display*; the popup still renders the whole ladder.
    fn count_dependent_within(&self, max_abs_rc: i16) -> bool {
        flips_within(&self.primary, max_abs_rc) || flips_within(&self.fallback, max_abs_rc)
    }

    /// The distinct start-only moves the primary ladder recommends somewhere (so the popup can label the
    /// fallback "if can't surrender" etc.).
    fn start_only_moves(&self) -> Vec<Move> {
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
struct IndexReport {
    lo: i16,
    hi: i16,
    cats: HashMap<HandCategory, CategoryIndex>,
    complete: bool,
}

/// What an [`IndexReport`] is cached/keyed under. Deliberately *without* the player's count (the report
/// is count-independent) and without the chart's count comparison (the index is always exact-count
/// based).
type IndexKey = (Card, ShoeChoice, Ruleset);

/// A finished (or partial) count-index report, tagged with the index epoch it was computed under so a
/// stale one (the shoe or ruleset changed) is dropped on arrival.
struct IndexResult {
    epoch: u64,
    key: IndexKey,
    report: IndexReport,
}

/// Build the `(move, lo, hi)` runs of `move_fn` over the integer running-count window `[lo, hi]` using
/// monotone root-finding: seed a few points, then bisect every adjacent pair whose move differs down to
/// integer adjacency, pinning each flip exactly. The per-move EV differences are monotone in the count,
/// so a flipped pair brackets a single crossing (the recursion still splits both halves, so an
/// intermediate move is handled too); same-move endpoints are taken as constant between them. Far
/// cheaper than sweeping every count — `O(log width)` evaluations per flip — which is the point, since
/// each evaluation is a full count-conditioned solve. The first/last runs are stretched to the window
/// edges so [`fmt_rc_range`] reads them as open-ended.
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
/// external count, all sharing one deconvolution — see [`CountShoe::band`]) plus a memo of the per-count
/// solved [`Column`]s. Each count is solved (the expensive [`solve_on`]) at most once and only when the
/// root-finder reaches for it; all categories and both the primary and fallback ladders share the memo.
struct ColumnEval {
    up: Card,
    rules: Ruleset,
    /// Usable (positive-mass) external counts, ascending; aligned with `shoes`.
    externals: Vec<i16>,
    shoes: Vec<CountShoe>,
    memo: HashMap<i16, Column>,
}

impl ColumnEval {
    /// Build the band over the pivot-centered window and drop the counts whose exact condition has no
    /// mass (unreachable under the penetration prior — a zero draw distribution would make `solve_on`
    /// meaningless). The reachable set is contiguous. `None` if nothing in the window is reachable.
    fn new(n: u8, up: Card, rules: &Ruleset) -> Option<Self> {
        let pivot = Ko::pivot(n);
        let window: Vec<i16> = (pivot - INDEX_HALF_WIDTH..=pivot + INDEX_HALF_WIDTH).collect();
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
            up,
            rules: *rules,
            externals,
            shoes: usable,
            memo: HashMap::new(),
        })
    }

    fn lo(&self) -> i16 {
        self.externals[0]
    }

    fn hi(&self) -> i16 {
        self.externals[self.externals.len() - 1]
    }

    /// Solve (memoized) the column at external count `ext`. `ext` must be a usable count.
    fn column(&mut self, ext: i16) -> &Column {
        if !self.memo.contains_key(&ext) {
            let idx = self.externals.iter().position(|&e| e == ext).unwrap();
            let col = solve_on(self.shoes[idx].clone(), self.up, &self.rules);
            self.memo.insert(ext, col);
        }
        &self.memo[&ext]
    }

    /// The move `ladder` recommends for `cat` at external count `ext`, or `None` if the category is
    /// absent (e.g. no Hit/Stand EVs to compare for the fallback).
    fn move_at(&mut self, ext: i16, cat: HandCategory, ladder: Ladder) -> Option<Move> {
        let ci = self.column(ext).summary.get(&cat)?;
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
fn compute_index_report(
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

/// Deck options the rules modal cycles through.
const DECK_OPTIONS: [ShoeChoice; 6] = [
    ShoeChoice::Infinite,
    ShoeChoice::Decks(1),
    ShoeChoice::Decks(2),
    ShoeChoice::Decks(4),
    ShoeChoice::Decks(6),
    ShoeChoice::Decks(8),
];

/// Split-precision options the rules modal cycles through (`split_cards` budget). The fully exact
/// cross-arm search (a budget larger than any reachable draw count) is intentionally not offered — it
/// is combinatorially infeasible on a big shoe and only used in tests.
const SPLIT_OPTIONS: [u8; 7] = [0, 1, 2, 3, 4, 6, 8];

/// One up-card's strategy summary: per chart-row category, its consolidated [`CellInfo`] (recommended
/// move, composition-dependence flag, per-move EVs, and the per-composition breakdown).
type ColumnSummary = HashMap<HandCategory, CellInfo>;

/// Everything a finished up-card column carries: the chart summary, the up-card's draw probability,
/// and its two-card-root edge contribution. Cached whole per `(shoe, ruleset)`.
#[derive(Clone, Serialize, Deserialize)]
struct Column {
    /// Per-category [`CellInfo`], consolidated by the game-time reaching weight (see [`solve_on`]).
    summary: ColumnSummary,
    /// Draw probability of this up-card from the full shoe — its weight in the overall edge.
    p_up: f64,
    edge: EdgeTerm,
}

/// A finished worker result: one solved column, tagged with the epoch it was computed for.
struct ColumnResult {
    epoch: u64,
    col: usize,
    column: Column,
}

/// The three chart panes.
#[derive(Clone, Copy)]
enum Pane {
    Hard,
    Soft,
    Pairs,
}

const PANES: [Pane; 3] = [Pane::Hard, Pane::Soft, Pane::Pairs];

impl Pane {
    fn title(self) -> &'static str {
        match self {
            Pane::Hard => "HARD",
            Pane::Soft => "SOFT",
            Pane::Pairs => "PAIRS",
        }
    }

    /// The rows of this pane: the chart-row category and its display label, top to bottom.
    fn rows(self) -> Vec<(HandCategory, String)> {
        match self {
            Pane::Hard => (5..=21)
                .map(|n| (HandCategory::Hard(n), format!("{n}")))
                .collect(),
            Pane::Soft => (13..=21)
                .map(|n| {
                    let label = if n < 21 {
                        format!("A{}", n - 11)
                    } else {
                        "AT".to_string()
                    };
                    (HandCategory::Soft(n), label)
                })
                .collect(),
            Pane::Pairs => [
                Card::Pip(2),
                Card::Pip(3),
                Card::Pip(4),
                Card::Pip(5),
                Card::Pip(6),
                Card::Pip(7),
                Card::Pip(8),
                Card::Pip(9),
                Card::Ten,
                Card::Ace,
            ]
            .iter()
            .map(|&r| (HandCategory::Pair(r), format!("{r}{r}")))
            .collect(),
        }
    }
}

/// Which overlay (if any) is currently up.
#[derive(PartialEq)]
enum Mode {
    Normal,
    Popup,
    Rules,
    Count,
}

/// Number of fields in the count modal (enabled, comparison, value).
const COUNT_FIELDS: usize = 3;

/// The highlighted cell: which pane, which row within it, which up-card column.
struct Cursor {
    pane: usize,
    row: usize,
    col: usize,
}

/// The full key a chart batch is solved and cached under: shoe, ruleset, and the effective count
/// condition (`None` on the infinite deck or when counting is off).
type ChartKey = (ShoeChoice, Ruleset, Option<CountSetting>);

struct App {
    rules: Ruleset,
    shoe: ShoeChoice,
    /// Whether a count condition is imposed, and its value/comparison. Only applied on a finite shoe.
    count_on: bool,
    count: CountSetting,
    /// Bumped on every recompute; stamped onto worker results so stale ones can be dropped.
    epoch: u64,
    /// The [`ChartKey`] the current epoch is being solved for. Captured at recompute time (not read
    /// from `self` later) so a completed batch is cached under the inputs it was actually computed with,
    /// even if a modal has since edited `self` without re-solving.
    epoch_key: ChartKey,
    /// Finished chart batches, keyed by [`ChartKey`], so flipping back to a prior setting is instant
    /// instead of a re-solve. Populated once all ten columns of an epoch arrive.
    cache: HashMap<ChartKey, [Column; 10]>,
    /// Per up-card column, `None` until that worker finishes (or filled at once from the cache).
    columns: [Option<Column>; 10],
    cursor: Cursor,
    mode: Mode,
    /// Selected field in the rules modal.
    rules_sel: usize,
    /// `(rules, shoe)` captured when the rules modal opened, to detect a change on close.
    rules_snapshot: (Ruleset, ShoeChoice),
    /// Selected field in the count modal.
    count_sel: usize,
    /// `(count_on, count)` captured when the count modal opened, to detect a change on close.
    count_snapshot: (bool, CountSetting),
    tx: Sender<ColumnResult>,
    rx: Receiver<ColumnResult>,
    /// Solved count-index reports, keyed by [`IndexKey`] (count-independent), so they survive count
    /// changes and are reused across the chart marker and the popup.
    index_cache: HashMap<IndexKey, IndexReport>,
    /// Columns whose index report is being solved in the background, so the scheduler doesn't respawn a
    /// column already in flight. Cleared per key when its *complete* report lands.
    index_pending: HashSet<IndexKey>,
    /// The `(shoe, rules)` the cached index reports are for; a change invalidates them (counts don't).
    index_basis: Option<(ShoeChoice, Ruleset)>,
    /// Bumped when `index_basis` changes, stamped on background reports so stale ones are dropped.
    index_epoch: u64,
    index_tx: Sender<IndexResult>,
    index_rx: Receiver<IndexResult>,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let (index_tx, index_rx) = mpsc::channel();
        let count = CountSetting {
            external: 0,
            cmp: CountCmp::Eq,
        };
        Self {
            rules: Ruleset::default(),
            shoe: ShoeChoice::Infinite,
            count_on: false,
            count,
            epoch: 0,
            epoch_key: (ShoeChoice::Infinite, Ruleset::default(), None),
            cache: HashMap::new(),
            columns: std::array::from_fn(|_| None),
            cursor: Cursor {
                pane: 0,
                row: 0,
                col: 0,
            },
            mode: Mode::Normal,
            rules_sel: 0,
            rules_snapshot: (Ruleset::default(), ShoeChoice::Infinite),
            count_sel: 0,
            count_snapshot: (false, count),
            tx,
            rx,
            index_cache: HashMap::new(),
            index_pending: HashSet::new(),
            index_basis: None,
            index_epoch: 0,
            index_tx,
            index_rx,
        }
    }

    /// The count condition actually applied to the solve: `None` on the infinite deck (no count
    /// exists) or when counting is toggled off, else the current [`CountSetting`].
    fn effective_count(&self) -> Option<CountSetting> {
        match self.shoe {
            ShoeChoice::Decks(_) if self.count_on => Some(self.count),
            _ => None,
        }
    }

    /// Refresh the chart for the current `(shoe, rules)`. A cached batch is restored instantly;
    /// otherwise a fresh epoch of ten worker threads (one per up-card) is launched and the chart is
    /// blanked. Old in-flight workers keep running but their results are discarded by epoch on arrival.
    fn recompute(&mut self) {
        self.epoch += 1;
        self.epoch_key = (self.shoe, self.rules, self.effective_count());
        // Count-index reports are count-*independent*, so only a shoe/rules change invalidates them
        // (a count tweak keeps them). Bumping `index_epoch` drops any in-flight worker for the old basis.
        let basis = (self.shoe, self.rules);
        if self.index_basis != Some(basis) {
            self.index_basis = Some(basis);
            self.index_epoch += 1;
            self.index_cache.clear();
            self.index_pending.clear();
        }
        // Cache hit: restore all ten columns at once, no workers (so nothing arrives for this epoch).
        if let Some(cached) = self.cache.get(&self.epoch_key) {
            self.columns = std::array::from_fn(|i| Some(cached[i].clone()));
            return;
        }
        self.columns = std::array::from_fn(|_| None);
        // Spawn the ten columns concurrently, longest-first (the low up-cards and the Ace are the
        // slow ones — the dealer draws more, and the Ace peek-conditions). The heavy work *inside*
        // each column is the pair-split solves, which `build_evs` already fans across cores; running
        // the columns concurrently lets those splits from every column share the machine, so all
        // cores stay busy instead of one column's splits being a single-core tail. On a box with
        // fewer cores than columns the longest-first spawn order is what gets the big columns going
        // first. Results stream to the chart as each finishes; stale epochs are dropped on arrival.
        let count = self.effective_count();
        for &col in &SOLVE_ORDER {
            let tx = self.tx.clone();
            let rules = self.rules;
            let shoe = self.shoe;
            let epoch = self.epoch;
            thread::spawn(move || {
                let column = shoe.solve(UP_CARDS[col], &rules, count);
                // Receiver gone (app exiting) is fine — just drop the result.
                let _ = tx.send(ColumnResult { epoch, col, column });
            });
        }
    }

    /// Drain any finished worker results, applying those from the current epoch. When a batch
    /// completes (all ten columns in), cache it under the epoch's `(shoe, rules)` key so returning to
    /// that ruleset later is instant.
    fn drain_results(&mut self) {
        while let Ok(res) = self.rx.try_recv() {
            if res.epoch == self.epoch {
                self.columns[res.col] = Some(res.column);
            }
        }
        if self.columns.iter().all(Option::is_some) && !self.cache.contains_key(&self.epoch_key) {
            let batch = std::array::from_fn(|i| self.columns[i].clone().unwrap());
            self.cache.insert(self.epoch_key, batch);
        }
    }

    /// The [`IndexKey`] for an up-card on the current basis, or `None` on the infinite deck (which has
    /// no count, so no index). Independent of the player's count and of whether counting is toggled on —
    /// the index is shown on the base chart either way.
    fn index_key(&self, up: Card) -> Option<IndexKey> {
        match self.shoe {
            ShoeChoice::Decks(_) => Some((up, self.shoe, self.rules)),
            ShoeChoice::Infinite => None,
        }
    }

    /// Background-fill the count-index reports for every up-card, up to [`INDEX_FILL_CONCURRENCY`] at a
    /// time. Deferred until the base chart finishes so the markers fill in *after* the chart is up
    /// rather than competing with it. Skips columns already complete or in flight; a worker streams its
    /// Hard/Soft markers first, then completes with Pairs (see [`compute_index_report`]).
    fn schedule_index_fill(&mut self) {
        let ShoeChoice::Decks(n) = self.shoe else {
            return;
        };
        if !self.columns.iter().all(Option::is_some) {
            return;
        }
        for &col in &SOLVE_ORDER {
            if self.index_pending.len() >= INDEX_FILL_CONCURRENCY {
                break;
            }
            let up = UP_CARDS[col];
            let key = (up, self.shoe, self.rules);
            let done = self.index_cache.get(&key).is_some_and(|r| r.complete);
            if done || self.index_pending.contains(&key) {
                continue;
            }
            self.index_pending.insert(key);
            let tx = self.index_tx.clone();
            let rules = self.rules;
            let epoch = self.index_epoch;
            thread::spawn(move || compute_index_report(n, key, &rules, epoch, &tx));
        }
    }

    /// Drain finished (or partial) count-index reports into the cache, dropping any from a superseded
    /// basis and clearing the in-flight marker once a column's *complete* report lands.
    fn drain_index_results(&mut self) {
        while let Ok(res) = self.index_rx.try_recv() {
            if res.epoch != self.index_epoch {
                continue;
            }
            if res.report.complete {
                self.index_pending.remove(&res.key);
            }
            self.index_cache.insert(res.key, res.report);
        }
    }

    /// The count-index report for the popup's current selection, if its column has been solved.
    fn selected_index_report(&self) -> Option<&IndexReport> {
        let (_, up) = self.selection();
        self.index_cache.get(&self.index_key(up)?)
    }

    /// Whether `cat` vs `up` is a count-dependent cell whose report has already been computed and whose
    /// flips fall within the notable [`INDEX_MARKER_MAX_RC`] window — the cue for the chart's `°` marker.
    /// (Extreme-count-only flips are suppressed here but still shown in the popup.)
    fn index_dependent(&self, cat: HandCategory, up: Card) -> bool {
        self.index_key(up)
            .and_then(|key| self.index_cache.get(&key))
            .and_then(|report| report.cats.get(&cat))
            .is_some_and(|ci| ci.count_dependent_within(INDEX_MARKER_MAX_RC))
    }

    fn active_pane(&self) -> Pane {
        PANES[self.cursor.pane]
    }

    /// The category and up-card the cursor is on.
    fn selection(&self) -> (HandCategory, Card) {
        let rows = self.active_pane().rows();
        let (cat, _) = rows[self.cursor.row.min(rows.len() - 1)];
        (cat, UP_CARDS[self.cursor.col])
    }

    /// The consolidated cell for the current selection, if its column has finished computing.
    fn selected_cell(&self) -> Option<&CellInfo> {
        let (cat, _) = self.selection();
        self.columns[self.cursor.col].as_ref()?.summary.get(&cat)
    }

    /// The overall player edge (negative = house edge) under the current rules, available only once
    /// every column has been solved (each contributes its draw-probability-weighted two-card-root
    /// value). `None` while any column is still pending.
    fn total_edge(&self) -> Option<f64> {
        let mut edge = 0.0;
        for col in &self.columns {
            let col = col.as_ref()?;
            edge += col.p_up * col.edge.value();
        }
        Some(edge)
    }

    fn clamp_row(&mut self) {
        let len = self.active_pane().rows().len();
        if self.cursor.row >= len {
            self.cursor.row = len - 1;
        }
    }

    // --- input -------------------------------------------------------------------------------

    /// Handle one key press. Returns `true` to quit.
    fn handle_key(&mut self, code: KeyCode) -> bool {
        match self.mode {
            Mode::Normal => return self.handle_normal(code),
            Mode::Popup => self.handle_popup(code),
            Mode::Rules => self.handle_rules(code),
            Mode::Count => self.handle_count(code),
        }
        false
    }

    fn handle_normal(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Enter | KeyCode::Char(' ') => self.mode = Mode::Popup,
            KeyCode::Char('r') => {
                self.rules_snapshot = (self.rules, self.shoe);
                self.rules_sel = 0;
                self.mode = Mode::Rules;
            }
            KeyCode::Char('c') => {
                self.count_snapshot = (self.count_on, self.count);
                self.count_sel = 0;
                self.mode = Mode::Count;
            }
            _ => self.move_cursor(code),
        }
        false
    }

    /// In the popup, motion keys still move the selection (the popup tracks it live); Esc closes.
    fn handle_popup(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => self.mode = Mode::Normal,
            _ => self.move_cursor(code),
        }
    }

    fn move_cursor(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('h') | KeyCode::Left => {
                self.cursor.col = self.cursor.col.saturating_sub(1)
            }
            KeyCode::Char('l') | KeyCode::Right => {
                self.cursor.col = (self.cursor.col + 1).min(UP_CARDS.len() - 1)
            }
            KeyCode::Char('k') | KeyCode::Up => self.cursor.row = self.cursor.row.saturating_sub(1),
            KeyCode::Char('j') | KeyCode::Down => {
                let len = self.active_pane().rows().len();
                self.cursor.row = (self.cursor.row + 1).min(len - 1);
            }
            KeyCode::Tab => {
                self.cursor.pane = (self.cursor.pane + 1) % PANES.len();
                self.clamp_row();
            }
            KeyCode::BackTab => {
                self.cursor.pane = (self.cursor.pane + PANES.len() - 1) % PANES.len();
                self.clamp_row();
            }
            _ => {}
        }
    }

    fn handle_rules(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('r') | KeyCode::Char('q') => {
                if (self.rules, self.shoe) != self.rules_snapshot {
                    self.recompute();
                }
                self.mode = Mode::Normal;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.rules_sel = (self.rules_sel + 1) % RULES_FIELDS;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.rules_sel = (self.rules_sel + RULES_FIELDS - 1) % RULES_FIELDS;
            }
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Char(' ') => self.edit_rule(1),
            KeyCode::Char('h') | KeyCode::Left => self.edit_rule(-1),
            _ => {}
        }
    }

    /// Change the currently selected rules field by `delta` (booleans toggle regardless of sign).
    fn edit_rule(&mut self, delta: i32) {
        match self.rules_sel {
            0 => {
                let i = DECK_OPTIONS
                    .iter()
                    .position(|&d| d == self.shoe)
                    .unwrap_or(0) as i32;
                let n = DECK_OPTIONS.len() as i32;
                self.shoe = DECK_OPTIONS[(i + delta).rem_euclid(n) as usize];
            }
            1 => self.rules.hs17 = !self.rules.hs17,
            2 => self.rules.das = !self.rules.das,
            3 => {
                // Toggle the peek, carrying the surrender choice across as far as the target state
                // allows: a no-peek game cannot hold *late* surrender, so it lands on early instead.
                self.rules.peek = match self.rules.peek {
                    PeekRule::Peek(PeekSurrender::None) => PeekRule::NoPeek {
                        early_surrender: false,
                    },
                    PeekRule::Peek(_) => PeekRule::NoPeek {
                        early_surrender: true,
                    },
                    PeekRule::NoPeek {
                        early_surrender: false,
                    } => PeekRule::Peek(PeekSurrender::None),
                    PeekRule::NoPeek {
                        early_surrender: true,
                    } => PeekRule::Peek(PeekSurrender::Early),
                };
            }
            4 => {
                self.rules.bj_payout = match self.rules.bj_payout {
                    BjPayout::ThreeToTwo => BjPayout::SixToFive,
                    BjPayout::SixToFive => BjPayout::ThreeToTwo,
                };
            }
            5 => {
                self.rules.peek = match self.rules.peek {
                    PeekRule::Peek(s) => {
                        let order = [
                            PeekSurrender::None,
                            PeekSurrender::Early,
                            PeekSurrender::Late,
                        ];
                        let i = order.iter().position(|&x| x == s).unwrap_or(0) as i32;
                        PeekRule::Peek(order[(i + delta).rem_euclid(3) as usize])
                    }
                    // No peek: late surrender is impossible, so this is just an early-surrender toggle.
                    PeekRule::NoPeek { early_surrender } => PeekRule::NoPeek {
                        early_surrender: !early_surrender,
                    },
                };
            }
            6 => {
                let v = self.rules.max_split_hands as i32 + delta;
                self.rules.max_split_hands = v.clamp(1, 4) as u8;
            }
            7 => {
                let i = SPLIT_OPTIONS
                    .iter()
                    .position(|&s| s == self.rules.split_cards)
                    .unwrap_or(2) as i32;
                let n = SPLIT_OPTIONS.len() as i32;
                self.rules.split_cards = SPLIT_OPTIONS[(i + delta).rem_euclid(n) as usize];
            }
            _ => {}
        }
    }

    fn handle_count(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('c') | KeyCode::Char('q') => {
                if (self.count_on, self.count) != self.count_snapshot {
                    self.recompute();
                }
                self.mode = Mode::Normal;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.count_sel = (self.count_sel + 1) % COUNT_FIELDS;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.count_sel = (self.count_sel + COUNT_FIELDS - 1) % COUNT_FIELDS;
            }
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Char(' ') => self.edit_count(1),
            KeyCode::Char('h') | KeyCode::Left => self.edit_count(-1),
            _ => {}
        }
    }

    /// Change the selected count-modal field by `delta`: toggle enabled, cycle the comparison, or
    /// step the running-count value.
    fn edit_count(&mut self, delta: i32) {
        match self.count_sel {
            0 => self.count_on = !self.count_on,
            1 => {
                let order = [CountCmp::Le, CountCmp::Eq, CountCmp::Ge];
                let i = order.iter().position(|&c| c == self.count.cmp).unwrap_or(1) as i32;
                self.count.cmp = order[(i + delta).rem_euclid(order.len() as i32) as usize];
            }
            2 => self.count.external = (self.count.external as i32 + delta).clamp(-60, 60) as i16,
            _ => {}
        }
    }

    // --- rendering ---------------------------------------------------------------------------

    fn render(&self, f: &mut Frame) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(2)])
            .split(f.area());

        // Lay the panes side by side when there's room for all three (each needs PANE_WIDTH); fall
        // back to a vertical stack on narrower terminals, where the extra height is available.
        let panes: Vec<Rect> = if outer[0].width >= 3 * PANE_WIDTH {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                ])
                .split(outer[0])
                .to_vec()
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(pane_height(Pane::Hard)),
                    Constraint::Length(pane_height(Pane::Soft)),
                    Constraint::Length(pane_height(Pane::Pairs)),
                    Constraint::Min(0),
                ])
                .split(outer[0])
                .to_vec()
        };

        for (i, &pane) in PANES.iter().enumerate() {
            self.render_pane(f, panes[i], pane, i == self.cursor.pane);
        }
        self.render_footer(f, outer[1]);

        match self.mode {
            Mode::Popup => self.render_popup(f),
            Mode::Rules => self.render_rules(f),
            Mode::Count => self.render_count(f),
            Mode::Normal => {}
        }
    }

    fn render_pane(&self, f: &mut Frame, area: Rect, pane: Pane, active: bool) {
        const LBL: usize = 4; // row-label column width

        let border_style = if active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(pane.title());
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines: Vec<Line> = Vec::new();

        // Header: blank label cell, then the up-cards.
        let mut header: Vec<Span> = vec![Span::raw(format!("{:width$}", "", width = LBL))];
        for (c, up) in UP_CARDS.iter().enumerate() {
            let mut style = Style::default().fg(Color::Gray);
            if active && c == self.cursor.col {
                style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
            }
            // Format via a String so the width/centering is honored: Card's Display impl ignores
            // the formatter width, so `{up:^3}` directly would render 1-wide and misalign the header.
            header.push(Span::styled(format!("{:^3}", up.to_string()), style));
        }
        lines.push(Line::from(header));

        for (r, (cat, label)) in pane.rows().into_iter().enumerate() {
            let mut spans: Vec<Span> =
                vec![Span::raw(format!("{label:>width$} ", width = LBL - 1))];
            for (i_c, &upcard) in UP_CARDS.iter().enumerate() {
                let (text, mut style) = match &self.columns[i_c] {
                    None => ("\u{00b7}".to_string(), Style::default().fg(Color::DarkGray)),
                    Some(col) => match col.summary.get(&cat) {
                        Some(cell) => {
                            // The headline move, with a one-char suffix in the 3-wide cell: `*` when the
                            // play varies by composition at this count (takes priority — a composition-
                            // dependent cell has two near-tied EVs, so it is essentially always count-
                            // dependent too, and `*` is the stronger signal), else `°` when the play flips
                            // with the running count in the notable window. The popup carries both. The
                            // leading space keeps the letter centered (" H*" / " H°"); a bare letter
                            // centers to " H " on its own.
                            let mv = cell.headline;
                            let text = if cell.composition_dependent {
                                format!(" {mv}*")
                            } else if self.index_dependent(cat, upcard) {
                                format!(" {mv}\u{00b0}")
                            } else {
                                format!("{mv}")
                            };
                            (text, Style::default().fg(move_color(mv)))
                        }
                        None => (" ".to_string(), Style::default()),
                    },
                };
                if active && r == self.cursor.row && i_c == self.cursor.col {
                    style = style.add_modifier(Modifier::REVERSED | Modifier::BOLD);
                }
                spans.push(Span::styled(format!("{text:^3}"), style));
            }
            lines.push(Line::from(spans));
        }

        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_footer(&self, f: &mut Frame, area: Rect) {
        let r = &self.rules;
        let bj = r.bj_payout.label();
        let surr = r.peek.surrender_label();
        let pending = self.columns.iter().filter(|c| c.is_none()).count();
        // The overall edge needs every column, so show a placeholder until the batch finishes.
        let edge = match self.total_edge() {
            Some(e) => format!("{:+.3}%", e * 100.0),
            None => "\u{2026}".to_string(),
        };
        let computing = if pending > 0 {
            format!("  computing {}/10", 10 - pending)
        } else {
            String::new()
        };

        let (cat, up) = self.selection();
        let sel = match self.selected_cell() {
            Some(cell) => {
                let mv = cell.headline;
                let star = if cell.composition_dependent { "*" } else { "" };
                format!(
                    "{cat} vs {up} \u{2192} {}{star} {:+.3}",
                    move_name(mv),
                    cell.move_evs[&mv]
                )
            }
            None => format!("{cat} vs {up} \u{2192} \u{2026}"),
        };

        // Count status: only meaningful on a finite shoe; shown as e.g. "KO RC>=+4" or "off".
        let count = match self.effective_count() {
            Some(c) => format!("KO RC{}{:+}", c.cmp_label(), c.external),
            None if self.count_on => "n/a(\u{221e})".to_string(),
            None => "off".to_string(),
        };
        let status = format!(
            "decks {} | {} | DAS {} | peek {} | BJ {bj} | surr {surr} | split\u{2264}{} | count {count} | edge {edge}{computing}",
            self.shoe,
            if r.hs17 { "H17" } else { "S17" },
            yn(r.das),
            yn(r.peek.peeks()),
            r.max_split_hands,
        );
        let keys = "hjkl move \u{00b7} Enter EVs \u{00b7} r rules \u{00b7} c count \u{00b7} q quit";

        let lines = vec![
            Line::from(vec![Span::styled(status, Style::default().fg(Color::Cyan))]),
            Line::from(vec![
                Span::styled(format!("{sel}    "), Style::default().fg(Color::White)),
                Span::styled(keys, Style::default().fg(Color::DarkGray)),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), area);
    }

    fn render_popup(&self, f: &mut Frame) {
        let (cat, up) = self.selection();
        let title = format!(" {cat} vs {up} ");

        let width = 48u16;
        let mut lines: Vec<Line> = Vec::new();
        match self.selected_cell() {
            None => lines.push(Line::from("computing\u{2026}")),
            Some(cell) => {
                let best = cell.headline;
                for mv in MOVE_ORDER {
                    if let Some(&ev) = cell.move_evs.get(&mv) {
                        // The best move is bolded; the move name keeps its chart color and the EV is
                        // colored by sign, so each column reads independently.
                        let emphasis = if mv == best {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        };
                        let marker = if mv == best { " *" } else { "" };
                        let name_style = Style::default().fg(move_color(mv)).add_modifier(emphasis);
                        let ev_style = Style::default().fg(ev_color(ev)).add_modifier(emphasis);
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {:<10}", move_name(mv)), name_style),
                            Span::styled(format!("{ev:+.4}{marker}"), ev_style),
                        ]));
                    }
                }
                // Per-composition breakdown: which hands actually prefer which move, ordered by
                // game-time probability. Only worth showing when more than one move wins somewhere.
                if cell.breakdown.len() > 1 {
                    lines.push(Line::from(Span::styled(
                        format!(" {:─^w$}", " by hand ", w = width as usize - 3),
                        Style::default().fg(Color::DarkGray),
                    )));
                    // Budget: inner width minus the "  X  " move-letter prefix.
                    let budget = (width as usize).saturating_sub(2 + 5);
                    for (mv, hands) in &cell.breakdown {
                        let (listed, overflow) = pack_hand_labels(hands, budget);
                        lines.push(Line::from(vec![
                            Span::styled(
                                format!("  {mv}  "),
                                Style::default()
                                    .fg(move_color(*mv))
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::raw(listed),
                            Span::styled(
                                if overflow > 0 {
                                    format!(" +{overflow}")
                                } else {
                                    String::new()
                                },
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                }
            }
        }
        // Count-index thresholds: the running counts at which the recommended play flips. A
        // count-*independent* property of the cell, shown on any finite shoe (with or without a count
        // imposed); background-filled (see `schedule_index_fill`). Layered: the headline ladder first,
        // then — once the headline is a start-only move (surrender/double/split) — the Hit/Stand
        // fallback for a hand that has already been hit and so can no longer take it.
        if self.index_key(up).is_some() {
            lines.push(Line::from(Span::styled(
                format!(
                    " {:─^w$}",
                    " count index (exact RC) ",
                    w = width as usize - 3
                ),
                Style::default().fg(Color::DarkGray),
            )));
            // The player's current running count, so their active run can be highlighted.
            let here = self.count_on.then_some(self.count.external);
            match self.selected_index_report() {
                None => lines.push(Line::from(Span::styled(
                    "  computing\u{2026}",
                    Style::default().fg(Color::DarkGray),
                ))),
                Some(report) => match report.cats.get(&cat) {
                    None => lines.push(Line::from(Span::styled(
                        "  computing\u{2026}",
                        Style::default().fg(Color::DarkGray),
                    ))),
                    Some(ci) => {
                        let (wmin, wmax) = (report.lo, report.hi);
                        for &(mv, lo, hi) in &ci.primary {
                            lines.push(rc_run_line(mv, lo, hi, wmin, wmax, here, 2));
                        }
                        if !ci.fallback.is_empty() {
                            let label = match ci.start_only_moves().as_slice() {
                                [only] => {
                                    format!("  if can't {}:", move_name(*only).to_lowercase())
                                }
                                _ => "  if start move unavailable:".to_string(),
                            };
                            lines.push(Line::from(Span::styled(
                                label,
                                Style::default().fg(Color::DarkGray),
                            )));
                            for &(mv, lo, hi) in &ci.fallback {
                                lines.push(rc_run_line(mv, lo, hi, wmin, wmax, here, 4));
                            }
                        }
                    }
                },
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  hjkl move \u{00b7} Esc close",
            Style::default().fg(Color::DarkGray),
        )));

        let height = lines.len() as u16 + 2;
        let area = centered_rect(width, height, f.area());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(title);
        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_rules(&self, f: &mut Frame) {
        let r = &self.rules;
        let fields = [
            format!("Decks         {}", self.shoe),
            format!("Dealer H17    {}", yn(r.hs17)),
            format!("DAS           {}", yn(r.das)),
            format!("Dealer peek   {}", yn(r.peek.peeks())),
            format!("Blackjack     {}", r.bj_payout.label()),
            format!("Surrender     {}", r.peek.surrender_label()),
            format!("Max hands     {}", r.max_split_hands),
            format!("Split prec.   {}", r.split_cards),
        ];

        let mut lines: Vec<Line> = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            let style = if i == self.rules_sel {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(format!("  {field}  "), style)));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  jk select \u{00b7} hl change \u{00b7} Esc apply",
            Style::default().fg(Color::DarkGray),
        )));

        let width = 34;
        let height = lines.len() as u16 + 2;
        let area = centered_rect(width, height, f.area());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(" Ruleset ");
        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_count(&self, f: &mut Frame) {
        let infinite = matches!(self.shoe, ShoeChoice::Infinite);
        let fields = [
            format!("Counting      {}", yn(self.count_on)),
            "System        KO".to_string(),
            format!(
                "Condition     RC {} {}",
                self.count.cmp_label(),
                self.count.external
            ),
        ];
        // Field 1 (system) is fixed at KO for now, so selection skips it; show it dimmed.
        let mut lines: Vec<Line> = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            let mut style = if i == self.count_sel {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            if i == 1 {
                style = style.fg(Color::DarkGray);
            }
            lines.push(Line::from(Span::styled(format!("  {field}  "), style)));
        }
        lines.push(Line::from(""));
        if infinite {
            lines.push(Line::from(Span::styled(
                "  (count applies to a finite shoe only)",
                Style::default().fg(Color::Yellow),
            )));
        }
        lines.push(Line::from(Span::styled(
            "  jk select \u{00b7} hl change \u{00b7} Esc apply",
            Style::default().fg(Color::DarkGray),
        )));

        let width = 40;
        let height = lines.len() as u16 + 2;
        let area = centered_rect(width, height, f.area());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(" Card counting ");
        f.render_widget(Clear, area);
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> std::io::Result<()> {
        loop {
            self.drain_results();
            self.drain_index_results();
            // Background-fill every cell's count index once the base chart is up (finite shoe only).
            self.schedule_index_fill();
            terminal.draw(|f| self.render(f))?;
            // Poll with a timeout so the chart keeps filling in as workers finish, even with no input.
            if event::poll(Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && self.handle_key(key.code)
            {
                return Ok(());
            }
        }
    }
}

/// Number of editable fields in the rules modal.
const RULES_FIELDS: usize = 8;

/// Width one pane needs to render untruncated: 2 border + 4 row-label + 10 up-cards × 3. Below
/// `3 × PANE_WIDTH` the chart stacks the panes vertically instead of side by side.
const PANE_WIDTH: u16 = 2 + 4 + 10 * 3;

/// Height a pane needs to render all its rows untruncated: 2 border + 1 header + one line per row.
fn pane_height(pane: Pane) -> u16 {
    pane.rows().len() as u16 + 3
}

/// A concrete hand as a compact rank string, e.g. `T5` or `T32`. Aces lead, then tens high→low pips,
/// so the label reads like the hand a player would name. Each rank is a single character, so no
/// separator is needed — and dropping it packs more hands into the breakdown.
fn compact_hand_label(hand: &CardCol) -> String {
    let order = |c: Card| match c {
        Card::Ace => 0,
        Card::Ten => 1,
        Card::Pip(n) => 11 - n as u32,
    };
    let mut cards: Vec<(Card, u16)> = hand.iter().collect();
    cards.sort_by_key(|&(c, _)| order(c));
    let mut parts: Vec<String> = Vec::new();
    for (c, n) in cards {
        for _ in 0..n {
            parts.push(c.to_string());
        }
    }
    parts.concat()
}

/// One count-index run rendered as a popup line: the move letter (indented `indent` columns) and its
/// running-count range. The run the player currently sits on (`here`, their external count, set only
/// when a count is imposed) is bolded and flagged with a `›`.
fn rc_run_line(
    mv: Move,
    lo: i16,
    hi: i16,
    wmin: i16,
    wmax: i16,
    here: Option<i16>,
    indent: usize,
) -> Line<'static> {
    let active = here.is_some_and(|e| lo <= e && e <= hi);
    let emph = if active {
        Modifier::BOLD
    } else {
        Modifier::empty()
    };
    let marker = if active { "\u{203a}" } else { " " };
    let pad = " ".repeat(indent.saturating_sub(1));
    Line::from(vec![
        Span::styled(
            format!("{pad}{marker}{mv}  "),
            Style::default().fg(move_color(mv)).add_modifier(emph),
        ),
        Span::styled(
            fmt_rc_range(lo, hi, wmin, wmax),
            Style::default()
                .fg(if active { Color::White } else { Color::Gray })
                .add_modifier(emph),
        ),
    ])
}

/// A count-index run's running-count range as a counter-friendly threshold, given the swept window
/// `[wmin, wmax]`. A run that reaches a window edge is shown open-ended (`≤`/`≥`) — any actual flip
/// lies outside the window — so e.g. `S  RC ≥ +2` reads "stand once the running count hits +2".
fn fmt_rc_range(lo: i16, hi: i16, wmin: i16, wmax: i16) -> String {
    match (lo <= wmin, hi >= wmax) {
        (true, true) => "any RC".to_string(),
        (true, false) => format!("RC \u{2264} {hi:+}"),
        (false, true) => format!("RC \u{2265} {lo:+}"),
        (false, false) if lo == hi => format!("RC = {lo:+}"),
        (false, false) => format!("RC {lo:+}..{hi:+}"),
    }
}

/// Greedily fit hand labels into `budget` columns, space-separated. Returns the joined string and the
/// count that didn't fit (rendered as a `+N` overflow). Always lists at least the first hand, so a
/// single very long label still shows rather than collapsing to a bare `+N`.
fn pack_hand_labels(hands: &[CardCol], budget: usize) -> (String, usize) {
    let labels: Vec<String> = hands.iter().map(compact_hand_label).collect();
    let mut used = 0usize;
    let mut shown = 0usize;
    for (i, label) in labels.iter().enumerate() {
        let need = if i == 0 { label.len() } else { 1 + label.len() };
        if i > 0 && used + need > budget {
            break;
        }
        used += need;
        shown += 1;
    }
    (labels[..shown].join(" "), labels.len() - shown)
}

fn move_name(mv: Move) -> &'static str {
    match mv {
        Move::Hit => "Hit",
        Move::Stand => "Stand",
        Move::Double => "Double",
        Move::Split => "Split",
        Move::Surrender => "Surrender",
    }
}

fn move_color(mv: Move) -> Color {
    match mv {
        // Hit -> green -> "go", Stand -> red -> "stop".
        Move::Hit => Color::LightGreen,
        Move::Stand => Color::LightRed,
        Move::Double => Color::LightBlue,
        Move::Split => Color::LightYellow,
        Move::Surrender => Color::LightMagenta,
    }
}

/// Sign-coded color for an EV value in the popup: green when favorable, red when not. Kept distinct
/// from the (Light*) move colors by using the plain variants so the two columns don't blur together.
fn ev_color(ev: f64) -> Color {
    if ev > 1e-9 {
        Color::Green
    } else if ev < -1e-9 {
        Color::Red
    } else {
        Color::DarkGray
    }
}

fn yn(b: bool) -> &'static str {
    if b { "\u{2713}" } else { "\u{2717}" }
}

/// A centered `width`x`height` rect within `area`, clamped to fit.
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

/// Launch the TUI: solve the chart asynchronously and run the event loop until the user quits.
pub fn run() -> std::io::Result<()> {
    let mut terminal = ratatui::init();
    let mut app = App::new();
    app.recompute();
    let result = app.event_loop(&mut terminal);
    ratatui::restore();
    result
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
}
