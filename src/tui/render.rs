//! All drawing: the three-pane chart, the footer status line, and the EV / rules / count overlays, plus
//! the small formatting helpers they share. Extends `impl App` (defined in [`super::app`]);
//! [`render`](App::render) is the entry point the [`event_loop`](App::event_loop) draws each frame.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::card::Card;
use crate::hand::{HandState, Move};
use crate::rules::Ruleset;
use crate::shoe::CardCol;

use super::app::{App, Mode};
use super::config::ShoeChoice;
use super::training::{Phase, Training};
use super::{MOVE_ORDER, PANES, Pane, Tab, UP_CARDS};

/// Width one pane needs to render untruncated: 2 border + 4 row-label + 10 up-cards × 3. Below
/// `3 × PANE_WIDTH` the chart stacks the panes vertically instead of side by side.
const PANE_WIDTH: u16 = 2 + 4 + 10 * 3;

impl App {
    pub(super) fn render(&self, f: &mut Frame) {
        // A one-line tab bar tops every view; the body and footer fill the rest.
        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(f.area());
        self.render_tabs(f, root[0]);

        match self.tab {
            Tab::Strategy => self.render_strategy(f, root[1]),
            Tab::Training => self.render_training(f, root[1]),
        }
    }

    /// The top tab bar: the available views with the active one highlighted.
    fn render_tabs(&self, f: &mut Frame, area: Rect) {
        let tab_span = |label: &str, active: bool| {
            let style = if active {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Span::styled(format!(" {label} "), style)
        };
        let line = Line::from(vec![
            tab_span("1 Strategy", self.tab == Tab::Strategy),
            Span::raw(" "),
            tab_span("2 Training", self.tab == Tab::Training),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    /// The strategy tab: the three-pane chart and the status footer.
    fn render_strategy(&self, f: &mut Frame, body: Rect) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)])
            .split(body);

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

        // Row 1: the constant ruleset — everything that's fixed regardless of the running count.
        let rules = format!(
            "decks {} | {} | DAS {} | peek {} | BJ {bj} | surr {surr} | split\u{2264}{}",
            self.shoe,
            if r.hs17 { "H17" } else { "S17" },
            yn(r.das),
            yn(r.peek.peeks()),
            r.max_split_hands,
        );

        // Row 2: everything that moves with the count — the counting system and its current
        // parameterization, the resulting edge, and the insurance EV. Count is only meaningful on a
        // finite shoe; shown as e.g. "KO RC>=+4" or "off".
        let count = match self.effective_count() {
            Some(c) => format!("KO RC{}{:+}", c.cmp_label(), c.external),
            None if self.count_on => "n/a(\u{221e})".to_string(),
            None => "off".to_string(),
        };
        let insurance = format!("{:+.3}", self.insurance_ev);
        let counted = format!("count {count} | edge {edge}{computing} | insurance {insurance}",);

        let keys = "hjkl move \u{00b7} Enter EVs \u{00b7} r rules \u{00b7} c count \u{00b7} q quit";

        let lines = vec![
            Line::from(vec![Span::styled(rules, Style::default().fg(Color::Cyan))]),
            Line::from(vec![Span::styled(
                counted,
                Style::default().fg(Color::Cyan),
            )]),
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

    /// The training tab: the felt (dealer + player hands), a side column of count/feedback/stats
    /// panels, and a key-hint footer, plus the count-quiz overlay when it is open.
    fn render_training(&self, f: &mut Frame, body: Rect) {
        let t = &self.training;
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(2)])
            .split(body);
        // Felt on the left, a fixed-width info column on the right (stacked vertically when narrow).
        let cols = if rows[0].width >= 64 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Length(38)])
                .split(rows[0])
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(10), Constraint::Min(0)])
                .split(rows[0])
        };

        render_felt(f, cols[0], t);
        let side = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Length(9),
                Constraint::Min(0),
            ])
            .split(cols[1]);
        render_count_panel(f, side[0], t);
        render_feedback_panel(f, side[1], t);
        render_stats_panel(f, side[2], t);

        render_training_footer(f, rows[1], t, &self.rules);

        if t.entering_count {
            render_count_quiz(f, t);
        }
        // The rules editor is shared across tabs (opened with `r`), so draw its overlay here too.
        if self.mode == Mode::Rules {
            self.render_rules(f);
        }
    }
}

/// The felt: the dealer's hand, the player's hand(s), and the current status line.
fn render_felt(f: &mut Frame, area: Rect, t: &Training) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .title(" Table ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();

    // Dealer row. The hole card (index 1) is hidden — and uncounted — until the paced dealer turn flips
    // it (so it stays a "?" through the player's turn and only the moment its [`DEALER_STEP`] tick lands).
    let revealed = !t.hole_down;
    let mut dealer: Vec<Span> = vec![Span::styled("Dealer  ", Style::default().fg(Color::Gray))];
    if t.dealer.is_empty() {
        dealer.push(Span::styled("—", Style::default().fg(Color::DarkGray)));
    } else {
        let mut cards_w = 0;
        for (i, &card) in t.dealer.iter().enumerate() {
            let (text, style) = if i == 1 && !revealed {
                ("? ".to_string(), Style::default().fg(Color::DarkGray))
            } else {
                (format!("{card} "), Style::default().fg(Color::White))
            };
            cards_w += text.chars().count();
            dealer.push(Span::styled(text, style));
        }
        let shown: Vec<Card> = if revealed {
            t.dealer.clone()
        } else {
            t.dealer[..1].to_vec()
        };
        push_total(&mut dealer, cards_w, cards_total_label(&shown));
    }
    lines.push(Line::from(dealer));
    lines.push(Line::from(""));

    // Player hand(s).
    if t.hands.is_empty() {
        lines.push(Line::from(Span::styled(
            "Press Enter to deal a hand.",
            Style::default().fg(Color::DarkGray),
        )));
    }
    for (i, hand) in t.hands.iter().enumerate() {
        let active = t.phase == Phase::Player && i == t.active;
        let marker = if active { "\u{203a}" } else { " " };
        let label = if t.hands.len() > 1 {
            format!("{marker}Hand {} ", i + 1)
        } else {
            format!("{marker}You    ")
        };
        let mut spans: Vec<Span> = vec![Span::styled(
            label,
            Style::default()
                .fg(if active { Color::Yellow } else { Color::Gray })
                .add_modifier(if active {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        )];
        let mut cards_w = 0;
        for &card in &hand.cards {
            let text = format!("{card} ");
            cards_w += text.chars().count();
            spans.push(Span::styled(text, Style::default().fg(Color::White)));
        }
        push_total(&mut spans, cards_w, cards_total_label(&hand.cards));
        if hand.doubled {
            spans.push(Span::styled(" x2", Style::default().fg(Color::LightBlue)));
        }
        if let Some(result) = hand.result {
            spans.push(Span::styled(
                format!("  {} {:+.2}", result.label(), hand.net),
                Style::default().fg(ev_color(hand.net)),
            ));
        }
        lines.push(Line::from(spans));
    }

    // Status / feedback line, kept at the bottom of the felt.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        t.message.clone(),
        Style::default().fg(Color::Cyan),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}

/// The count-drill panel. On a finite shoe the true running count is kept hidden (that is the thing
/// being practised) and the player checks themselves with the `n` quiz; penetration is shown since a
/// counter sees it. The infinite deck has no count at all, so it reads as a plain basic-strategy drill.
fn render_count_panel(f: &mut Frame, area: Rect, t: &Training) {
    let title = if t.is_finite() { " Count " } else { " Deck " };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = if let Some(decks_left) = t.decks_remaining() {
        let (q, c) = (t.stats.count_quizzes, t.stats.count_correct);
        let acc = if q > 0 {
            format!("{:.0}%", 100.0 * c as f64 / q as f64)
        } else {
            "—".to_string()
        };
        vec![
            Line::from(Span::styled(
                "RC  hidden  (n to guess)",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(format!("Decks left  {decks_left:.1}")),
            Line::from(format!("Quizzes     {c}/{q}  {acc}")),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                "\u{221e} deck \u{00b7} no count",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "basic-strategy drill",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    };
    f.render_widget(Paragraph::new(lines), inner);
}

/// The last graded decision: the player's move against the basic / indexed / exact-optimal references,
/// and the EV gap. Empty until [`Training::evaluate`](super::training::Training::evaluate) is wired up.
fn render_feedback_panel(f: &mut Frame, area: Rect, t: &Training) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Last decision ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    match &t.last_mark {
        None if t.grading() => lines.push(Line::from(Span::styled(
            "grading\u{2026}",
            Style::default().fg(Color::DarkGray),
        ))),
        None => lines.push(Line::from(Span::styled(
            "play a hand to see feedback",
            Style::default().fg(Color::DarkGray),
        ))),
        Some(m) => {
            lines.push(Line::from(format!("you played  {}", move_name(m.chosen))));
            let ref_line = |name: &str, mv: Move| {
                let agree = mv == m.chosen;
                Line::from(vec![
                    Span::styled(format!("{name:<8}  "), Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!(
                            "{} {}",
                            if agree { "\u{2713}" } else { "\u{2717}" },
                            move_name(mv)
                        ),
                        Style::default().fg(if agree { Color::Green } else { Color::Red }),
                    ),
                ])
            };
            lines.push(ref_line("basic", m.basic));
            match m.indexed {
                Some(mv) => lines.push(ref_line("indexed", mv)),
                None => lines.push(Line::from(vec![
                    Span::styled("indexed   ", Style::default().fg(Color::Gray)),
                    Span::styled("— n/a", Style::default().fg(Color::DarkGray)),
                ])),
            }
            lines.push(ref_line("optimal", m.optimal));
            lines.push(Line::from(vec![
                Span::raw(format!("EV cost   {:+.4}", m.ev_chosen - m.ev_optimal)),
                // A newer decision is still being graded in the background.
                Span::styled(
                    if t.grading() { "  grading\u{2026}" } else { "" },
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The running session scoreboard: decision accuracy against each reference, realised net, and EV gap.
fn render_stats_panel(f: &mut Frame, area: Rect, t: &Training) {
    let block = Block::default().borders(Borders::ALL).title(" Session ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let s = &t.stats;
    let rate = |n: u32| {
        if s.decisions > 0 {
            format!(
                "{n}/{}  {:.0}%",
                s.decisions,
                100.0 * n as f64 / s.decisions as f64
            )
        } else {
            "—".to_string()
        }
    };
    // The indexed reference is count-only, so it is n/a on the infinite deck — show a dash there rather
    // than a misleading 0%.
    let indexed = if t.is_finite() {
        format!("Indexed     {}", rate(s.agree_indexed))
    } else {
        "Indexed     —".to_string()
    };
    let lines = vec![
        Line::from(format!("Rounds      {}", s.rounds)),
        Line::from(format!("Decisions   {}", s.decisions)),
        Line::from(format!("Basic       {}", rate(s.agree_basic))),
        Line::from(indexed),
        Line::from(format!("Optimal     {}", rate(s.agree_optimal))),
        Line::from(vec![
            Span::raw("EV gap      "),
            Span::styled(
                format!("{:+.3}", s.ev_gap),
                Style::default().fg(ev_color(s.ev_gap)),
            ),
        ]),
        Line::from(vec![
            Span::raw("Net         "),
            Span::styled(
                format!("{:+.2} u", s.realized),
                Style::default().fg(ev_color(s.realized)),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// The training footer: the phase tag plus a context-sensitive key map. During the player's turn only
/// the *legal* actions for the active hand are listed (via [`Training::allowed_move`]); otherwise the
/// deal key. The count/tab/quit keys are always shown.
fn render_training_footer(f: &mut Frame, area: Rect, t: &Training, rules: &Ruleset) {
    let phase = match t.phase {
        Phase::Ready => "ready",
        Phase::Dealing => "dealing",
        Phase::Player => "your turn",
        Phase::Dealer => "dealer",
        Phase::Settled => "settled",
    };
    let action = if t.phase == Phase::Player {
        [
            (Move::Hit, "h hit"),
            (Move::Stand, "s stand"),
            (Move::Double, "d double"),
            (Move::Split, "p split"),
            (Move::Surrender, "r surr"),
        ]
        .iter()
        .filter(|(mv, _)| t.allowed_move(*mv, rules))
        .map(|(_, label)| *label)
        .collect::<Vec<_>>()
        .join(" \u{00b7} ")
    } else {
        "Enter deal".to_string()
    };
    // The count quiz is finite-shoe only (the infinite deck has no count to guess).
    let count_key = if t.is_finite() {
        " \u{00b7} n count"
    } else {
        ""
    };
    let keys = format!("{action}{count_key} \u{00b7} 1 strategy \u{00b7} q quit");
    let lines = vec![
        Line::from(Span::styled(
            format!("[{phase}]"),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(Span::styled(keys, Style::default().fg(Color::DarkGray))),
    ];
    f.render_widget(Paragraph::new(lines), area);
}

/// The running-count quiz overlay: the player's working guess, adjusted with `h`/`l` and submitted with
/// Enter.
fn render_count_quiz(f: &mut Frame, t: &Training) {
    let lines = vec![
        Line::from(format!("  Running count guess:  {:+}  ", t.count_entry)),
        Line::from(""),
        Line::from(Span::styled(
            "  hl adjust \u{00b7} Enter submit \u{00b7} Esc cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let width = 40;
    let height = lines.len() as u16 + 2;
    let area = centered_rect(width, height, f.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" What's the count? ");
    f.render_widget(Clear, area);
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// The collapsed total of a set of cards as a short felt label: `bust`, `blackjack`, `soft 18`, or a
/// bare hard total. An empty set reads as `0`.
/// Width of the dealt-cards field on the felt: cards render left-justified within it (each rank is
/// `"X "`, two columns), and the running total is right-justified in [`TOTAL_COL_W`] immediately after,
/// so the total reads as its own column set off from the cards rather than butting up against them.
const CARDS_COL_W: usize = 24;
/// Width of the right-justified hand-total column (fits the widest label, `"blackjack"`).
const TOTAL_COL_W: usize = 9;

/// Append the hand `total` as a right-justified column after the `cards_w`-wide run of card spans,
/// padding the cards field out to [`CARDS_COL_W`] first so every row's total lines up in its own column.
fn push_total(spans: &mut Vec<Span<'static>>, cards_w: usize, total: String) {
    if let Some(pad) = CARDS_COL_W.checked_sub(cards_w).filter(|&p| p > 0) {
        spans.push(Span::raw(" ".repeat(pad)));
    }
    spans.push(Span::styled(
        format!("{total:>TOTAL_COL_W$}"),
        Style::default().fg(Color::Gray),
    ));
}

fn cards_total_label(cards: &[Card]) -> String {
    if cards.is_empty() {
        return "0".to_string();
    }
    match HandState::from(&CardCol::from_hand(cards)) {
        HandState::Bust => "bust".to_string(),
        HandState::Natural => "blackjack".to_string(),
        HandState::Soft(n) => format!("soft {n}"),
        HandState::Hard(n) => format!("{n}"),
    }
}

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
pub(super) fn fmt_rc_range(lo: i16, hi: i16, wmin: i16, wmax: i16) -> String {
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
