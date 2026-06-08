//! The terminal UI (the *only* place ratatui is used). It renders the basic-strategy chart as three
//! side-by-side panes (Hard / Soft / Pairs), each a grid of strategy-table rows by dealer up-card,
//! navigable with vim motions; an EV popup shows the per-move EVs for the highlighted hand/up-card;
//! a rules modal edits the [`Ruleset`] (and deck count).
//!
//! Compute is asynchronous: each of the ten up-cards is solved on its own worker thread
//! ([`build_evs`] + [`summarize_evs`]) and the chart fills in column-by-column as results arrive, so
//! the interface never blocks. A monotonic `epoch` tags every batch; results from a superseded epoch
//! (a rules/deck change happened) are discarded rather than interrupting the worker.
//!
//! Counting is not modelled by the solver yet, so there is no count input here — the [`ShoeChoice`]
//! seam (finite shoe vs infinite deck) is where a count-adjusted draw distribution would eventually
//! plug in.

use std::collections::HashMap;
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
use crate::hand::{HandCategory, Move};
use crate::rules::{BjPayout, PeekRule, PeekSurrender, Ruleset};
use crate::shoe::{CardCol, InfiniteDeck, Shoe};
use crate::simulation::{EdgeTerm, build_evs, edge_term, summarize_evs};

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
    fn solve(self, up_card: Card, rules: &Ruleset) -> Column {
        // `p_up` is the up-card's draw probability from the *full* shoe (before `build_evs` removes
        // it); it weights this column in the overall edge.
        let (tree, p_up) = match self {
            ShoeChoice::Infinite => {
                let shoe = InfiniteDeck {};
                (build_evs(shoe, up_card, rules), shoe.draw_prob(&up_card))
            }
            ShoeChoice::Decks(n) => {
                let shoe = CardCol::from_decks(n);
                (build_evs(shoe, up_card, rules), shoe.draw_prob(&up_card))
            }
        };
        Column {
            summary: summarize_evs(&tree),
            p_up,
            edge: edge_term(&tree),
        }
    }
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
const SPLIT_OPTIONS: [u8; 4] = [0, 2, 4, 8];

/// One up-card's strategy summary: per chart-row category, the EV of every available move.
type ColumnSummary = HashMap<HandCategory, HashMap<Move, f64>>;

/// Everything a finished up-card column carries: the chart summary, the up-card's draw probability,
/// and its two-card-root edge contribution. Cached whole per `(shoe, ruleset)`.
#[derive(Clone)]
struct Column {
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
                        format!("A,{}", n - 11)
                    } else {
                        "A,T".to_string()
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
            .map(|&r| (HandCategory::Pair(r), format!("{r},{r}")))
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
}

/// The highlighted cell: which pane, which row within it, which up-card column.
struct Cursor {
    pane: usize,
    row: usize,
    col: usize,
}

struct App {
    rules: Ruleset,
    shoe: ShoeChoice,
    /// Bumped on every recompute; stamped onto worker results so stale ones can be dropped.
    epoch: u64,
    /// The `(shoe, rules)` the current epoch is being solved for. Captured at recompute time (not read
    /// from `self` later) so a completed batch is cached under the rules it was actually computed with,
    /// even if the rules modal has since edited `self.rules` without re-solving.
    epoch_key: (ShoeChoice, Ruleset),
    /// Finished chart batches, keyed by `(shoe, rules)`, so flipping back to a prior ruleset is instant
    /// instead of a re-solve. Populated once all ten columns of an epoch arrive.
    cache: HashMap<(ShoeChoice, Ruleset), [Column; 10]>,
    /// Per up-card column, `None` until that worker finishes (or filled at once from the cache).
    columns: [Option<Column>; 10],
    cursor: Cursor,
    mode: Mode,
    /// Selected field in the rules modal.
    rules_sel: usize,
    /// `(rules, shoe)` captured when the rules modal opened, to detect a change on close.
    rules_snapshot: (Ruleset, ShoeChoice),
    tx: Sender<ColumnResult>,
    rx: Receiver<ColumnResult>,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            rules: Ruleset::default(),
            shoe: ShoeChoice::Infinite,
            epoch: 0,
            epoch_key: (ShoeChoice::Infinite, Ruleset::default()),
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
            tx,
            rx,
        }
    }

    /// Refresh the chart for the current `(shoe, rules)`. A cached batch is restored instantly;
    /// otherwise a fresh epoch of ten worker threads (one per up-card) is launched and the chart is
    /// blanked. Old in-flight workers keep running but their results are discarded by epoch on arrival.
    fn recompute(&mut self) {
        self.epoch += 1;
        self.epoch_key = (self.shoe, self.rules);
        // Cache hit: restore all ten columns at once, no workers (so nothing arrives for this epoch).
        if let Some(cached) = self.cache.get(&self.epoch_key) {
            self.columns = std::array::from_fn(|i| Some(cached[i].clone()));
            return;
        }
        self.columns = std::array::from_fn(|_| None);
        for (col, &up_card) in UP_CARDS.iter().enumerate() {
            let tx = self.tx.clone();
            let rules = self.rules;
            let shoe = self.shoe;
            let epoch = self.epoch;
            thread::spawn(move || {
                let column = shoe.solve(up_card, &rules);
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

    fn active_pane(&self) -> Pane {
        PANES[self.cursor.pane]
    }

    /// The category and up-card the cursor is on.
    fn selection(&self) -> (HandCategory, Card) {
        let rows = self.active_pane().rows();
        let (cat, _) = rows[self.cursor.row.min(rows.len() - 1)];
        (cat, UP_CARDS[self.cursor.col])
    }

    /// The per-move EV map for the current selection, if its column has finished computing.
    fn selected_evs(&self) -> Option<&HashMap<Move, f64>> {
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
                    PeekRule::Peek(PeekSurrender::None) => {
                        PeekRule::NoPeek { early_surrender: false }
                    }
                    PeekRule::Peek(_) => PeekRule::NoPeek { early_surrender: true },
                    PeekRule::NoPeek { early_surrender: false } => {
                        PeekRule::Peek(PeekSurrender::None)
                    }
                    PeekRule::NoPeek { early_surrender: true } => {
                        PeekRule::Peek(PeekSurrender::Early)
                    }
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
                        let order =
                            [PeekSurrender::None, PeekSurrender::Early, PeekSurrender::Late];
                        let i = order.iter().position(|&x| x == s).unwrap_or(0) as i32;
                        PeekRule::Peek(order[(i + delta).rem_euclid(3) as usize])
                    }
                    // No peek: late surrender is impossible, so this is just an early-surrender toggle.
                    PeekRule::NoPeek { early_surrender } => {
                        PeekRule::NoPeek { early_surrender: !early_surrender }
                    }
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
                        Some(moves) => {
                            let mv = best_move(moves);
                            (mv.to_string(), Style::default().fg(move_color(mv)))
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
        let sel = match self.selected_evs() {
            Some(moves) => {
                let mv = best_move(moves);
                format!(
                    "{cat} vs {up} \u{2192} {} {:+.3}",
                    move_name(mv),
                    moves[&mv]
                )
            }
            None => format!("{cat} vs {up} \u{2192} \u{2026}"),
        };

        let status = format!(
            "decks {} | {} | DAS {} | peek {} | BJ {bj} | surr {surr} | split\u{2264}{} | edge {edge}{computing}",
            self.shoe,
            if r.hs17 { "H17" } else { "S17" },
            yn(r.das),
            yn(r.peek.peeks()),
            r.max_split_hands,
        );
        let keys =
            "hjkl move \u{00b7} Tab pane \u{00b7} Enter EVs \u{00b7} r rules \u{00b7} q quit";

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

        let mut lines: Vec<Line> = Vec::new();
        match self.selected_evs() {
            None => lines.push(Line::from("computing\u{2026}")),
            Some(moves) => {
                let best = best_move(moves);
                for mv in MOVE_ORDER {
                    if let Some(&ev) = moves.get(&mv) {
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
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  hjkl move \u{00b7} Esc close",
            Style::default().fg(Color::DarkGray),
        )));

        let width = 30u16;
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

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> std::io::Result<()> {
        loop {
            self.drain_results();
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

fn best_move(moves: &HashMap<Move, f64>) -> Move {
    *moves
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .expect("a charted category always has at least one move")
        .0
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
