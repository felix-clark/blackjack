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
//! Under a count, the EV popup also shows **count-index thresholds** — the running counts at which the
//! recommended move flips — by sweeping a band of counts around the current one ([`solve_count_band`],
//! sharing one deconvolution across the band) on a background thread while the popup is open.

use std::collections::HashMap;
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

use crate::card::Card;
use crate::count::{CountCmp, CountShoe, Ko, Penetration};
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
        match self {
            ShoeChoice::Infinite => solve_on(InfiniteDeck {}, up_card, rules),
            ShoeChoice::Decks(n) => match count {
                Some(c) => solve_on(
                    CountShoe::from_external::<Ko>(n, c.external, c.cmp, COUNT_PENETRATION),
                    up_card,
                    rules,
                ),
                None => solve_on(CardCol::from_decks(n), up_card, rules),
            },
        }
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

/// How far either side of the player's current running count the count-index sweep reaches. The whole
/// `2·RADIUS+1`-wide band is solved together, sharing one deconvolution across the layers (see
/// [`CountShoe::band`]), so widening it costs only the cheap per-layer arithmetic — but each extra
/// layer still re-runs the (mean-field) splits, so it is kept modest. Thresholds beyond the window are
/// reported as open-ended.
const INDEX_RADIUS: i16 = 6;

/// One up-card's optimal move per chart category across a swept band of external running counts — the
/// raw material for the popup's count-index thresholds. `externals` is ascending; `headline[i]` is the
/// column's per-category recommended move at running count `externals[i]`. Solved under exact-count
/// (`==`) conditioning, the natural "what should I do when the count is exactly this" basis for an
/// index, independent of the chart's own count comparison.
#[derive(Clone)]
struct CountBand {
    externals: Vec<i16>,
    headline: Vec<HashMap<HandCategory, Move>>,
}

/// What a solved [`CountBand`] is cached/keyed under: the up-card, shoe, ruleset, and the band's
/// center (the player's current external count). Indices are exact-count based, so the chart's count
/// *comparison* is deliberately not part of the key.
type IndexKey = (Card, ShoeChoice, Ruleset, i16);

/// A finished count-index band, tagged with the key it was computed for so a stale one is ignored.
struct IndexResult {
    key: IndexKey,
    band: CountBand,
}

/// Solve one up-card's column across the external-count window `center ± INDEX_RADIUS`, sharing the
/// deconvolution across the whole band ([`CountShoe::band`]): the first layer warms the shared
/// draw-distribution cache, the rest read it. Returns each layer's per-category headline move.
fn solve_count_band(n: u8, center: i16, up_card: Card, rules: &Ruleset) -> CountBand {
    let mut externals: Vec<i16> = ((center - INDEX_RADIUS)..=(center + INDEX_RADIUS))
        .map(|e| e.clamp(-60, 60))
        .collect();
    externals.dedup(); // clamping at the ±60 edges can collide; the range is ascending so dups adjoin.
    let shoes = CountShoe::band::<Ko>(n, &externals, CountCmp::Eq, COUNT_PENETRATION);
    let headline = shoes
        .iter()
        .map(|shoe| {
            solve_on(shoe.clone(), up_card, rules)
                .summary
                .iter()
                .map(|(&cat, ci)| (cat, ci.headline))
                .collect()
        })
        .collect();
    CountBand {
        externals,
        headline,
    }
}

/// The count-index runs for `cat` read off a solved band: consecutive `(move, lo, hi)` inclusive
/// running-count ranges over which that move is optimal, in ascending count order. A single run means
/// the move never changes across the window (no index to show); multiple runs are the flip points.
fn category_indices(band: &CountBand, cat: HandCategory) -> Vec<(Move, i16, i16)> {
    let mut runs: Vec<(Move, i16, i16)> = Vec::new();
    for (i, &ext) in band.externals.iter().enumerate() {
        let Some(&mv) = band.headline[i].get(&cat) else {
            continue;
        };
        match runs.last_mut() {
            Some((m, _lo, hi)) if *m == mv => *hi = ext,
            _ => runs.push((mv, ext, ext)),
        }
    }
    runs
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
#[derive(Clone)]
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
    /// Solved count-index bands, keyed by [`IndexKey`], so revisiting a popup cell is instant.
    index_cache: HashMap<IndexKey, CountBand>,
    /// The band currently being solved in the background (one at a time), so a popup that is open over
    /// a count-conditioned cell doesn't respawn the same sweep every tick.
    index_pending: Option<IndexKey>,
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
            index_pending: None,
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
        // Count-index bands are keyed by the old shoe/rules, so they are stale now. A band still in
        // flight will land and be cached under its (now irrelevant) key and simply never be read.
        self.index_cache.clear();
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

    /// The [`IndexKey`] for the popup's current selection, or `None` when count indices don't apply
    /// (counting off, or an infinite deck, which has no count). The band centers on the player's
    /// current external count.
    fn index_key(&self) -> Option<IndexKey> {
        match self.shoe {
            ShoeChoice::Decks(_) if self.count_on => {
                let (_, up) = self.selection();
                Some((up, self.shoe, self.rules, self.count.external))
            }
            _ => None,
        }
    }

    /// While a popup is open over a count-conditioned cell, ensure its column's count-index band is
    /// solving (or already solved). One sweep runs at a time in the background; the popup renders
    /// "computing…" until it lands. A no-op when indices don't apply or the band is already cached/in
    /// flight for this exact selection.
    fn ensure_index(&mut self) {
        let Some(key) = self.index_key() else { return };
        if self.index_cache.contains_key(&key) || self.index_pending.as_ref() == Some(&key) {
            return;
        }
        let ShoeChoice::Decks(n) = key.1 else { return };
        self.index_pending = Some(key);
        let tx = self.index_tx.clone();
        let (up, _, rules, center) = key;
        thread::spawn(move || {
            let band = solve_count_band(n, center, up, &rules);
            let _ = tx.send(IndexResult { key, band });
        });
    }

    /// Drain any finished count-index bands into the cache, clearing the in-flight marker when the one
    /// being awaited arrives.
    fn drain_index_results(&mut self) {
        while let Ok(res) = self.index_rx.try_recv() {
            if self.index_pending.as_ref() == Some(&res.key) {
                self.index_pending = None;
            }
            self.index_cache.insert(res.key, res.band);
        }
    }

    /// The solved count-index band for the popup's current selection, if it has finished.
    fn selected_index_band(&self) -> Option<&CountBand> {
        self.index_cache.get(&self.index_key()?)
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
            for c in 0..UP_CARDS.len() {
                let (text, mut style) = match &self.columns[c] {
                    None => ("\u{00b7}".to_string(), Style::default().fg(Color::DarkGray)),
                    Some(col) => match col.summary.get(&cat) {
                        Some(cell) => {
                            // The headline move, asterisked when the right play genuinely varies by
                            // composition. The letter sits in the middle column either way (a bare
                            // letter centers to " H "); the leading space on the starred form keeps
                            // it there and lets the `*` merely append: " H*".
                            let mv = cell.headline;
                            let text = if cell.composition_dependent {
                                format!(" {mv}*")
                            } else {
                                format!("{mv}")
                            };
                            (text, Style::default().fg(move_color(mv)))
                        }
                        None => (" ".to_string(), Style::default()),
                    },
                };
                if active && r == self.cursor.row && c == self.cursor.col {
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
        let keys =
            "hjkl move \u{00b7} Enter EVs \u{00b7} r rules \u{00b7} c count \u{00b7} q quit";

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
                        format!(
                            " {:─^w$}",
                            " by hand (game-time order) ",
                            w = width as usize - 3
                        ),
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
        // Count-index thresholds: how the recommended move shifts with the running count. Only shown
        // when a count is imposed; the band solves in the background (see `ensure_index`).
        if self.index_key().is_some() {
            lines.push(Line::from(Span::styled(
                format!(" {:─^w$}", " count index (exact RC) ", w = width as usize - 3),
                Style::default().fg(Color::DarkGray),
            )));
            match self.selected_index_band() {
                None => lines.push(Line::from(Span::styled(
                    "  computing\u{2026}",
                    Style::default().fg(Color::DarkGray),
                ))),
                Some(band) => {
                    let runs = category_indices(band, cat);
                    let wmin = band.externals.first().copied().unwrap_or(0);
                    let wmax = band.externals.last().copied().unwrap_or(0);
                    if runs.is_empty() {
                        lines.push(Line::from(Span::styled(
                            "  (no count-consistent hand in window)",
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                    for (mv, lo, hi) in runs {
                        lines.push(Line::from(vec![
                            Span::styled(
                                format!("  {mv}  "),
                                Style::default()
                                    .fg(move_color(mv))
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                fmt_rc_range(lo, hi, wmin, wmax),
                                Style::default().fg(Color::Gray),
                            ),
                        ]));
                    }
                }
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

        let width = 34u16;
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

        let width = 40u16;
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
            // Only solve count-index bands while a popup is actually open over a counted cell.
            if self.mode == Mode::Popup {
                self.ensure_index();
            }
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

    /// End-to-end count-index smoke test on the full TUI band path (`solve_count_band` →
    /// `category_indices`). Pins the qualitative KO deviation for the canonical 16 vs Ten: the band's
    /// externals come back ascending, and the recommended move flips *with the count in the right
    /// direction* — Hit at the bottom of the window (deck rich in low cards, hitting stiff 16 is safe),
    /// giving way to a non-Hit (surrender/stand) as the count climbs and the deck richens in tens.
    /// `#[ignore]` because a full count-conditioned column band is seconds of work (the heavy count
    /// tests all are). Run with `--release --ignored`.
    #[test]
    #[ignore]
    fn count_index_16_vs_ten_flips_with_count() {
        let band = solve_count_band(1, 0, Card::Ten, &Ruleset::default());
        assert!(band.externals.windows(2).all(|w| w[0] < w[1]));
        let runs = category_indices(&band, HandCategory::Hard(16));
        assert!(
            runs.len() >= 2,
            "expected a count deviation for 16 vs T, got {runs:?}"
        );
        assert_eq!(runs.first().unwrap().0, Move::Hit, "low count should Hit");
        assert_ne!(
            runs.last().unwrap().0,
            Move::Hit,
            "high count should deviate off Hit"
        );
    }
}
