//! The application state ([`App`]) and the asynchronous solve lifecycle: launching per-up-card worker
//! threads on a [`recompute`](App::recompute), draining their streamed results, background-filling the
//! count-index reports, and the top-level [`event_loop`](App::event_loop). The input handlers live in
//! [`super::input`] and the rendering in [`super::render`] (both extend `impl App`).

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyEventKind};

use crate::card::Card;
use crate::count::CountCmp;
use crate::hand::HandCategory;
use crate::reach::CellInfo;
use crate::rules::Ruleset;

use super::column::{Column, ColumnResult};
use super::config::{CountSetting, ShoeChoice};
use super::index::{
    INDEX_FILL_CONCURRENCY, INDEX_MARKER_MAX_RC, IndexKey, IndexReport, IndexResult,
    compute_index_report,
};
use super::{PANES, Pane, SOLVE_ORDER, UP_CARDS};

/// Which overlay (if any) is currently up.
#[derive(PartialEq)]
pub(super) enum Mode {
    Normal,
    Popup,
    Rules,
    Count,
}

/// The highlighted cell: which pane, which row within it, which up-card column.
pub(super) struct Cursor {
    pub(super) pane: usize,
    pub(super) row: usize,
    pub(super) col: usize,
}

/// The full key a chart batch is solved and cached under: shoe, ruleset, and the effective count
/// condition (`None` on the infinite deck or when counting is off).
type ChartKey = (ShoeChoice, Ruleset, Option<CountSetting>);

pub(super) struct App {
    pub(super) rules: Ruleset,
    pub(super) shoe: ShoeChoice,
    /// Whether a count condition is imposed, and its value/comparison. Only applied on a finite shoe.
    pub(super) count_on: bool,
    pub(super) count: CountSetting,
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
    pub(super) columns: [Option<Column>; 10],
    pub(super) cursor: Cursor,
    pub(super) mode: Mode,
    /// Selected field in the rules modal.
    pub(super) rules_sel: usize,
    /// `(rules, shoe)` captured when the rules modal opened, to detect a change on close.
    pub(super) rules_snapshot: (Ruleset, ShoeChoice),
    /// Selected field in the count modal.
    pub(super) count_sel: usize,
    /// `(count_on, count)` captured when the count modal opened, to detect a change on close.
    pub(super) count_snapshot: (bool, CountSetting),
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
    pub(super) fn new() -> Self {
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
    pub(super) fn effective_count(&self) -> Option<CountSetting> {
        match self.shoe {
            ShoeChoice::Decks(_) if self.count_on => Some(self.count),
            _ => None,
        }
    }

    /// Refresh the chart for the current `(shoe, rules)`. A cached batch is restored instantly;
    /// otherwise a fresh epoch of ten worker threads (one per up-card) is launched and the chart is
    /// blanked. Old in-flight workers keep running but their results are discarded by epoch on arrival.
    pub(super) fn recompute(&mut self) {
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
    pub(super) fn index_key(&self, up: Card) -> Option<IndexKey> {
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
    pub(super) fn selected_index_report(&self) -> Option<&IndexReport> {
        let (_, up) = self.selection();
        self.index_cache.get(&self.index_key(up)?)
    }

    /// Whether `cat` vs `up` is a count-dependent cell whose report has already been computed and whose
    /// flips fall within the notable [`INDEX_MARKER_MAX_RC`] window — the cue for the chart's `°` marker.
    /// (Extreme-count-only flips are suppressed here but still shown in the popup.)
    pub(super) fn index_dependent(&self, cat: HandCategory, up: Card) -> bool {
        self.index_key(up)
            .and_then(|key| self.index_cache.get(&key))
            .and_then(|report| report.cats.get(&cat))
            .is_some_and(|ci| ci.count_dependent_within(INDEX_MARKER_MAX_RC))
    }

    pub(super) fn active_pane(&self) -> Pane {
        PANES[self.cursor.pane]
    }

    /// The category and up-card the cursor is on.
    pub(super) fn selection(&self) -> (HandCategory, Card) {
        let rows = self.active_pane().rows();
        let (cat, _) = rows[self.cursor.row.min(rows.len() - 1)];
        (cat, UP_CARDS[self.cursor.col])
    }

    /// The consolidated cell for the current selection, if its column has finished computing.
    pub(super) fn selected_cell(&self) -> Option<&CellInfo> {
        let (cat, _) = self.selection();
        self.columns[self.cursor.col].as_ref()?.summary.get(&cat)
    }

    /// The overall player edge (negative = house edge) under the current rules, available only once
    /// every column has been solved (each contributes its draw-probability-weighted two-card-root
    /// value). `None` while any column is still pending.
    pub(super) fn total_edge(&self) -> Option<f64> {
        let mut edge = 0.0;
        for col in &self.columns {
            let col = col.as_ref()?;
            edge += col.p_up * col.edge.value();
        }
        Some(edge)
    }

    pub(super) fn clamp_row(&mut self) {
        let len = self.active_pane().rows().len();
        if self.cursor.row >= len {
            self.cursor.row = len - 1;
        }
    }

    pub(super) fn event_loop(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
    ) -> std::io::Result<()> {
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
