//! The terminal UI (the *only* place ratatui is used). It renders the basic-strategy chart as three
//! side-by-side panes (Hard / Soft / Pairs), each a grid of strategy-table rows by dealer up-card,
//! navigable with vim motions; an EV popup shows the per-move EVs for the highlighted hand/up-card;
//! a rules modal edits the [`Ruleset`](crate::rules::Ruleset) (and deck count).
//!
//! Compute is asynchronous: each of the ten up-cards is solved on its own worker thread
//! ([`build_evs`](crate::simulation::build_evs) + [`summarize_cells`](crate::reach::summarize_cells))
//! and the chart fills in column-by-column as results arrive, so the interface never blocks. A
//! monotonic `epoch` tags every batch; results from a superseded epoch (a rules/deck change happened)
//! are discarded rather than interrupting the worker.
//!
//! A card-counting condition (the `c` modal) conditions the solve on a running count: on a finite
//! shoe it swaps the plain [`CardCol`](crate::shoe::CardCol) for a [`CountShoe`](crate::count::CountShoe)
//! (exact count-conditioned main tree and dealer; mean-field tilt inside splits). The condition is part
//! of the chart cache key, so toggling counts on/off or changing the running count re-solves (or
//! restores from cache) like a rules change. On a finite shoe the chart also shows **count-index
//! thresholds** — the running counts at which the recommended move flips (see [`index`]).
//!
//! The interface is organised into top-level [`Tab`]s: the **strategy** tab (the chart described above)
//! and a **training** tab ([`training`]) — a hand-by-hand blackjack drill against the live shoe for
//! practising basic strategy, count-indexed deviations, and the running count. The training tab's game
//! engine is left as a documented seam; the harness (tab switch, layout, rendering, key routing) is wired
//! up around it.
//!
//! ## Module map
//!
//! - [`config`] — the solve configuration (`ShoeChoice`, `CountSetting`) and the per-column solve entry.
//! - [`column`] — a solved up-card column (`Column`) and the generic `solve_on`.
//! - [`index`] — the count-index subsystem (the running counts at which a cell's play flips).
//! - [`training`] — the training-tab model and the (stubbed) game-simulation seam.
//! - [`app`] — the [`App`] state, the async solve lifecycle, and the event loop.
//! - [`input`] — keyboard input and the modal field editors.
//! - [`render`] — all drawing.

mod app;
mod column;
mod config;
mod index;
mod input;
mod render;
mod training;

use crate::card::Card;
use crate::hand::{HandCategory, Move};

use app::App;

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

/// The top-level views, switched with the `1`/`2` keys (and shown in the tab bar).
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Tab {
    /// The basic-strategy chart and EV explorer.
    Strategy,
    /// The hand-by-hand training drill (see [`training`]).
    Training,
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
            Pane::Pairs => UP_CARDS
                .iter()
                .map(|&r| (HandCategory::Pair(r), format!("{r}{r}")))
                .collect(),
        }
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
