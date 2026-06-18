//! Shared `#[cfg(test)]` helpers for the solver's test modules (`main`'s general tests and
//! `split`'s split-specific ones). Kept in one place so the chart-building boilerplate and the
//! float comparison aren't duplicated across modules.

use std::collections::HashMap;

use crate::card::*;
use crate::hand::{HandCategory, Move};
use crate::reach::{bust_weights, reach_weights, summarize_cells};
use crate::rules::Ruleset;
use crate::shoe::*;
use crate::simulation::build_evs;

/// The default ruleset with only the split-accuracy budget overridden — the single knob the tests
/// vary (`0` = independent arms, `u8::MAX` = full exact cross-arm search).
pub(crate) fn ruleset_with(split_cards: u8) -> Ruleset {
    Ruleset {
        split_cards,
        ..Ruleset::default()
    }
}

/// A 2-deck EV tree for `up_card`. Built with `split_cards: 0` (independent arms): the chart/strategy
/// tests assert argmax cells and non-split magnitudes that don't depend on the split-accuracy budget,
/// and a full chart build at the product default (4) is ~minutes per column. The budget itself is
/// covered by the dedicated `*_split_*` tests on focused inputs.
pub(crate) fn ev_tree(up_card: Card) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
    build_evs(CardCol::from_decks(2), up_card, &ruleset_with(0))
}

/// The consolidated recommended-move-per-category strategy for `up_card` on a 2-deck shoe, via the
/// **corrected** game-time consolidation the TUI uses ([`reach_weights`] → [`summarize_cells`]): each
/// cell's headline is decided on its two-card decision population, so a start-only move (Surrender)
/// is compared only against the Hit/Stand EVs of hands that can actually take it.
pub(crate) fn strategy_for(up_card: Card) -> HashMap<HandCategory, Move> {
    let shoe = CardCol::from_decks(2);
    let rules = ruleset_with(0);
    let tree = build_evs(shoe, up_card, &rules);
    let reach = reach_weights(shoe, up_card, &rules, &tree, true);
    let bust = bust_weights(shoe, up_card, &rules, &tree);
    summarize_cells(&tree, &reach, &bust)
        .into_iter()
        .map(|(cat, cell)| (cat, cell.headline))
        .collect()
}

/// The full corrected consolidation for `up_card` on a `decks`-deck shoe, exposing the
/// composition-dependence flag and per-move EVs (not just the argmax) for the chart tests.
pub(crate) fn cells_for(decks: u8, up_card: Card) -> HashMap<HandCategory, crate::reach::CellInfo> {
    let shoe = CardCol::from_decks(decks);
    let rules = ruleset_with(0);
    let tree = build_evs(shoe, up_card, &rules);
    let reach = reach_weights(shoe, up_card, &rules, &tree, true);
    let bust = bust_weights(shoe, up_card, &rules, &tree);
    summarize_cells(&tree, &reach, &bust)
}

#[track_caller]
pub(crate) fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{label}: got {actual}, expected {expected}"
    );
}
