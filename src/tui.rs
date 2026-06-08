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
use crate::rules::{Ruleset, SurrenderRule};
use crate::shoe::{CardCol, InfiniteDeck};
use crate::simulation::{build_evs, summarize_evs};

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
#[derive(Clone, Copy, PartialEq)]
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
    /// Solve one up-card's full EV tree on this shoe and collapse it to the per-category strategy
    /// summary the chart renders. Runs on a worker thread.
    fn solve(self, up_card: Card, rules: &Ruleset) -> ColumnSummary {
        let tree = match self {
            ShoeChoice::Infinite => build_evs(InfiniteDeck {}, up_card, rules),
            ShoeChoice::Decks(n) => build_evs(CardCol::from_decks(n), up_card, rules),
        };
        summarize_evs(&tree)
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

/// Split-precision options the rules modal cycles through (`split_cards` budget; last is the full
/// exact search — opt-in, can be slow on big shoes).
const SPLIT_OPTIONS: [u8; 5] = [0, 2, 4, 8, Ruleset::EXACT_SPLIT];

/// One up-card's strategy summary: per chart-row category, the EV of every available move.
type ColumnSummary = HashMap<HandCategory, HashMap<Move, f64>>;

/// A finished worker result: the summary for one column, tagged with the epoch it was computed for.
struct ColumnResult {
    epoch: u64,
    col: usize,
    summary: ColumnSummary,
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
    /// Per up-card column summary, `None` until that worker finishes.
    columns: [Option<ColumnSummary>; 10],
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

    /// Keep the ruleset self-consistent (the only invariant the solver enforces): late surrender
    /// requires the dealer to peek, so drop it to early surrender if peek is turned off.
    fn normalize_rules(&mut self) {
        if self.rules.surrender == SurrenderRule::Late && !self.rules.dealer_check {
            self.rules.surrender = SurrenderRule::Early;
        }
    }

    /// Start a fresh batch of ten worker threads, one per up-card, and blank the chart. Old in-flight
    /// workers keep running but their results are discarded by epoch on arrival.
    fn recompute(&mut self) {
        self.normalize_rules();
        self.epoch += 1;
        self.columns = std::array::from_fn(|_| None);
        for (col, &up_card) in UP_CARDS.iter().enumerate() {
            let tx = self.tx.clone();
            let rules = self.rules;
            let shoe = self.shoe;
            let epoch = self.epoch;
            thread::spawn(move || {
                let summary = shoe.solve(up_card, &rules);
                // Receiver gone (app exiting) is fine — just drop the result.
                let _ = tx.send(ColumnResult {
                    epoch,
                    col,
                    summary,
                });
            });
        }
    }

    /// Drain any finished worker results, applying those from the current epoch.
    fn drain_results(&mut self) {
        while let Ok(res) = self.rx.try_recv() {
            if res.epoch == self.epoch {
                self.columns[res.col] = Some(res.summary);
            }
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
        self.columns[self.cursor.col].as_ref()?.get(&cat)
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
                self.normalize_rules();
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
            3 => self.rules.dealer_check = !self.rules.dealer_check,
            4 => {
                self.rules.bj_payout = if (self.rules.bj_payout - 1.5).abs() < 1e-9 {
                    1.2
                } else {
                    1.5
                };
            }
            5 => {
                let order = [
                    SurrenderRule::None,
                    SurrenderRule::Early,
                    SurrenderRule::Late,
                ];
                let i = order
                    .iter()
                    .position(|&s| s == self.rules.surrender)
                    .unwrap_or(0) as i32;
                self.rules.surrender = order[(i + delta).rem_euclid(3) as usize];
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
                    Some(summary) => match summary.get(&cat) {
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
        let bj = if (r.bj_payout - 1.5).abs() < 1e-9 {
            "3:2"
        } else if (r.bj_payout - 1.2).abs() < 1e-9 {
            "6:5"
        } else {
            "?"
        };
        let surr = match r.surrender {
            SurrenderRule::None => "none",
            SurrenderRule::Early => "early",
            SurrenderRule::Late => "late",
        };
        let pending = self.columns.iter().filter(|c| c.is_none()).count();
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
            "decks {} | {} | DAS {} | peek {} | BJ {bj} | surr {surr} | split\u{2264}{}{computing}",
            self.shoe,
            if r.hs17 { "H17" } else { "S17" },
            yn(r.das),
            yn(r.dealer_check),
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
            format!("Dealer peek   {}", yn(r.dealer_check)),
            format!(
                "Blackjack     {}",
                if (r.bj_payout - 1.5).abs() < 1e-9 {
                    "3:2"
                } else {
                    "6:5"
                }
            ),
            format!(
                "Surrender     {}",
                match r.surrender {
                    SurrenderRule::None => "none",
                    SurrenderRule::Early => "early",
                    SurrenderRule::Late => "late",
                }
            ),
            format!("Max hands     {}", r.max_split_hands),
            format!(
                "Split prec.   {}",
                if r.split_cards == Ruleset::EXACT_SPLIT {
                    "exact".to_string()
                } else {
                    r.split_cards.to_string()
                }
            ),
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
