//! One solved up-card column: the per-category strategy summary, the up-card's draw probability, and
//! its two-card-root edge contribution. [`solve_on`] is the generic solve+consolidate the shoe-specific
//! [`ShoeChoice::solve`](super::config::ShoeChoice::solve) dispatches to.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::hand::HandCategory;
use crate::reach::{CellInfo, reach_weights, summarize_cells};
use crate::rules::Ruleset;
use crate::shoe::Shoe;
use crate::simulation::{EdgeTerm, build_evs, edge_term};

/// One up-card's strategy summary: per chart-row category, its consolidated [`CellInfo`] (recommended
/// move, composition-dependence flag, per-move EVs, and the per-composition breakdown).
pub(super) type ColumnSummary = HashMap<HandCategory, CellInfo>;

/// Everything a finished up-card column carries: the chart summary, the up-card's draw probability,
/// and its two-card-root edge contribution. Cached whole per `(shoe, ruleset)`.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Column {
    /// Per-category [`CellInfo`], consolidated by the game-time reaching weight (see [`solve_on`]).
    pub(super) summary: ColumnSummary,
    /// Draw probability of this up-card from the full shoe — its weight in the overall edge.
    pub(super) p_up: f64,
    pub(super) edge: EdgeTerm,
}

/// A finished worker result: one solved column, tagged with the epoch it was computed for.
pub(super) struct ColumnResult {
    pub(super) epoch: u64,
    pub(super) col: usize,
    pub(super) column: Column,
}

/// Solve and consolidate one up-card column on a concrete shoe `S`. Cells are consolidated by the
/// game-time **reaching weight** ([`reach_weights`] → [`summarize_cells`], split arms folded in): how
/// often each composition is actually the hand in front of a deciding player. Each cell's headline is
/// decided on its two-card decision population (so a start-only move is compared only against the
/// Hit/Stand EVs of hands that can take it), and carries its composition-dependence flag and
/// per-composition breakdown. `p_up` is the up-card's draw probability from the *full* shoe (before
/// `build_evs` removes it).
pub(super) fn solve_on<S: Shoe + Clone + Eq + Hash + Sync>(
    shoe: S,
    up_card: Card,
    rules: &Ruleset,
) -> Column {
    let tree = build_evs(shoe.clone(), up_card, rules);
    let weights = reach_weights(shoe.clone(), up_card, rules, &tree, true);
    Column {
        summary: summarize_cells(&tree, &weights),
        p_up: shoe.draw_prob(&up_card),
        edge: edge_term(&tree),
    }
}
