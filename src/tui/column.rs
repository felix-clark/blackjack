//! One solved up-card column: the per-category strategy summary, the up-card's draw probability, and
//! its two-card-root edge contribution. [`solve_on`] is the generic solve+consolidate the shoe-specific
//! [`ShoeChoice::solve`](super::config::ShoeChoice::solve) dispatches to.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::count::{CountCmp, CountShoe, CountSystem, Ko};
use crate::hand::{HandCategory, Move};
use crate::reach::{CellInfo, reach_weights, summarize_cells};
use crate::rules::Ruleset;
use crate::shoe::{CardCol, Shoe};
use crate::simulation::{
    EdgeTerm, build_evs, build_evs_with_splits, edge_term, pair_split_evs_for, splittable_pairs,
};

use super::config::COUNT_PENETRATION;

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

/// One solved EV tree — a per-frame product the count-conditioned solves merge into a display column.
pub(super) type Tree = HashMap<CardCol, (f64, HashMap<Move, f64>)>;
/// Game-time reach weights over a tree's hands (see [`reach_weights`]).
pub(super) type ReachMap = HashMap<CardCol, f64>;

/// The two-card hand count-value groups the Wizard-of-Odds frame merge ranges over. A two-card hand's
/// count value lies in `[-2, 2]`; longer hands clamp into it (see [`merge_count_frames`]).
pub(super) const COUNT_GROUPS: std::ops::RangeInclusive<i16> = -2..=2;

/// The KO running-count value of a whole hand: the sum of its cards' count values. A two-card hand
/// spans `[-2, 2]`; longer hands (multi-card breakdown members) can exceed that and are clamped into the
/// solved range.
fn hand_count_value(hand: &CardCol) -> i16 {
    hand.iter().map(|(c, n)| Ko::map(&c) * n as i16).sum()
}

/// Assemble the merged display tree/reach the chart consolidation reads, under the Wizard-of-Odds count
/// convention: each hand is read from the per-frame solve for *its own* count value `k = map(hand)`
/// (clamped into [`COUNT_GROUPS`]), so it is displayed at the running count that includes the whole hand
/// (see [`solve_counted`]). The caller supplies, per group `k`, the tree and reach solved at that group's
/// frame — the chart drives it with five `build_evs` calls, the count-index sweep with band shoes at the
/// correspondingly shifted counts. Every per-frame tree enumerates the same hand set (the count tilts
/// probabilities, not which ranks the pool holds), so the hand list is taken from group `0`.
pub(super) fn merge_count_frames<'a>(
    tree_for: impl Fn(i16) -> &'a Tree,
    reach_for: impl Fn(i16) -> &'a ReachMap,
) -> (Tree, ReachMap) {
    let (lo, hi) = (*COUNT_GROUPS.start(), *COUNT_GROUPS.end());
    let mut merged_tree = Tree::new();
    let mut merged_reach = ReachMap::new();
    for hand in tree_for(0).keys().copied().collect::<Vec<_>>() {
        let k = hand_count_value(&hand).clamp(lo, hi);
        if let Some(entry) = tree_for(k).get(&hand) {
            merged_tree.insert(hand, entry.clone());
            if let Some(&r) = reach_for(k).get(&hand) {
                merged_reach.insert(hand, r);
            }
        }
    }
    (merged_tree, merged_reach)
}

/// Solve a count-conditioned column under the **Wizard-of-Odds count convention**: the entered running
/// count is taken to *include every card visible at the decision point* — the dealer up-card and the
/// player's own hand. That is the count a player actually holds at the table, and the basis WoO's index
/// numbers are quoted against. (The naive single solve instead anchors the count *before* the round's
/// cards, so it conditions each hand on `external + map(up) + map(hand)` — off by the count value of the
/// very cards the player is looking at.)
///
/// That convention makes each cell's count frame depend on its own hand, which a single shared DP can't
/// represent (two starting hands that hit into the same multiset would demand different count targets on
/// the shared node). The fix exploits an exact identity: a player holding hand `H` (count value
/// `map(H)`) under up-card `U` at running count `external` is equivalent, for the ordinary all-shift
/// engine, to a *pre-exposure* entered count `external - map(U) - map(H)` — feeding that puts `H` at
/// exactly count `external` (the up-card and `H` removed without moving the count, every deeper draw
/// shifting normally; see the count notes). A two-card hand's `map(H)` takes only the five values
/// `[-2, 2]`, so we solve the engine once per `k ∈ [-2, 2]` at `external - map(U) - k` and read each hand
/// from the solve whose `k` matches its own count value. Within a fixed-`k` solve there is no
/// shared-node conflict (the hit-card value on any shared multiset is forced to `map(M) - k`, identical
/// across decompositions), so each per-frame tree is internally exact. Longer (multi-card breakdown)
/// hands clamp into `[-2, 2]`: that only nudges the informational breakdown and composition-dependence
/// flag, never a two-card headline or per-move EV, all of which are exact.
///
/// The overall **edge** and `p_up` keep the *pre-round* reading (the count before this round's cards) —
/// the only coherent frame for a quantity averaged over all up-cards and hands. That is exactly the
/// `k = -map(U)` solve (offset `external`), i.e. the previous single-solve behavior, so the footer edge
/// is unchanged; only the per-cell EVs move.
///
/// The DP runs once per frame (five times per column), but the dominant cost — the pair splits — is
/// solved once per pair rather than per frame: a pair `(r, r)` is only ever read from the frame
/// matching its count value `2·map(r)`, so its split is solved on that frame's shoe alone (see the
/// `split_evs` step below). That keeps the count solve close to the cost of a single uncounted one
/// plus the four extra cheap DP passes, rather than ~5× it.
pub(super) fn solve_counted(
    n_decks: u8,
    external: i16,
    cmp: CountCmp,
    up_card: Card,
    rules: &Ruleset,
) -> Column {
    let mu = Ko::map(&up_card);
    let (lo, hi) = (*COUNT_GROUPS.start(), *COUNT_GROUPS.end());

    // One frame shoe per count group `k`: the entered running count, anchored *before* this round's
    // cards, that lands a hand of count value `k` at exactly `external` (see above).
    let shoes: HashMap<i16, CountShoe> = COUNT_GROUPS
        .map(|k| {
            (
                k,
                CountShoe::from_external::<Ko>(n_decks, external - mu - k, cmp, COUNT_PENETRATION),
            )
        })
        .collect();

    // Splits are the engine's expensive phase (~98% of a solve), and the merge below only ever reads a
    // pair from the frame matching its own count value — so each pair's split is solved exactly once,
    // on that frame's shoe, instead of in all five frames. (A pair `(r, r)` has count value `2·map(r)`,
    // so pairs land only on `k ∈ {-2, 0, 2}`; the ±1 frames hold none.) This is what keeps the count
    // solve from costing ~5× the uncounted one.
    let mut deck = shoes[&0].clone();
    deck.draw(&up_card);
    let pairs = splittable_pairs(&deck);
    let split_evs = pair_split_evs_for(&pairs, up_card, rules, |pair| {
        let k = hand_count_value(pair).clamp(lo, hi);
        shoes[&k].clone()
    });

    // Per-frame DP (the cheap ~2%), each fed only the splits whose pair belongs to that frame.
    let mut trees: HashMap<i16, Tree> = HashMap::new();
    let mut reaches: HashMap<i16, ReachMap> = HashMap::new();
    let mut edge = EdgeTerm {
        weighted_ev: 0.0,
        weight: 0.0,
    };
    let mut p_up = 0.0;
    for k in COUNT_GROUPS {
        let frame_splits: HashMap<CardCol, f64> = split_evs
            .iter()
            .filter(|(pair, _)| hand_count_value(pair).clamp(lo, hi) == k)
            .map(|(&pair, &ev)| (pair, ev))
            .collect();
        let shoe = &shoes[&k];
        let tree = build_evs_with_splits(shoe.clone(), up_card, rules, &frame_splits);
        let reach = reach_weights(shoe.clone(), up_card, rules, &tree, true);
        // The pre-round frame (entered count taken as the count *before* the round) is `k = -map(U)`,
        // which is always in range since `map(U) ∈ [-1, 1]`. The edge and up-card weight read from it.
        if k == -mu {
            edge = edge_term(&tree);
            p_up = shoe.draw_prob(&up_card);
        }
        trees.insert(k, tree);
        reaches.insert(k, reach);
    }

    let (merged_tree, merged_reach) = merge_count_frames(|k| &trees[&k], |k| &reaches[&k]);
    Column {
        summary: summarize_cells(&merged_tree, &merged_reach),
        p_up,
        edge,
    }
}
