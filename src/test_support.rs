//! Shared `#[cfg(test)]` helpers for the solver's test modules (`main`'s general tests and
//! `split`'s split-specific ones). Kept in one place so the chart-building boilerplate and the
//! float comparison aren't duplicated across modules.

use std::collections::HashMap;

use crate::card::*;
use crate::hand::{HandCategory, Move};
use crate::rules::Ruleset;
use crate::shoe::*;
use crate::simulation::{build_evs, summarize_evs};

/// The default ruleset with only the split-accuracy budget overridden — the single knob the tests
/// vary (`0` = independent arms, [`Ruleset::EXACT_SPLIT`] = full exact cross-arm search).
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

/// The consolidated best-move-per-category strategy for `up_card` on the 2-deck [`ev_tree`].
pub(crate) fn strategy_for(up_card: Card) -> HashMap<HandCategory, Move> {
    best_strategy(&summarize_evs(&ev_tree(up_card)))
}

/// Reduce a per-category move→EV summary (from [`summarize_evs`]) to the single best move per row.
/// Exercised by the strategy tests; the TUI argmaxes per cell inline rather than calling this.
fn best_strategy(
    summary: &HashMap<HandCategory, HashMap<Move, f64>>,
) -> HashMap<HandCategory, Move> {
    summary
        .iter()
        .map(|(&cat, move_evs)| {
            let best = move_evs
                .iter()
                // Panics on a NaN EV, which the solver should never produce.
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(&mv, _)| mv)
                .expect("every category has at least one move");
            (cat, best)
        })
        .collect()
}

#[track_caller]
pub(crate) fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{label}: got {actual}, expected {expected}"
    );
}
