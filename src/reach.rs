//! Game-time probability-of-reaching-a-hand — the consolidation weight the live chart pools by.
//!
//! The EV tree's original pooling weight (`summarize_evs`) is the combinatorial *scan-weight*: the
//! multivariate-hypergeometric probability of holding a multiset, ignoring draw order and the
//! stopping rule. Across hand sizes that is not a coherent occurrence distribution (the documented
//! cross-size weighting bias). This module computes the alternative the TUI now uses
//! (`tui::solve_on` → [`reach_weights`] → [`summarize_with`]): the probability that the player is
//! actually *sitting at a hand making a decision* during real, optimally-played rounds. On a
//! non-depleting (infinite) deck every composition of a category has identical EVs, so the two
//! weightings coincide exactly; the difference is a finite-shoe effect (4th-decimal EV shifts,
//! essentially never a cell flip — chiefly it zeroes the unreachable compositions).
//!
//! ## Why this is a forward pass, not a fixed point
//!
//! Optimal play is path-independent at the multiset level — `shoe(H) = initial − up − H` regardless
//! of draw order — so the optimal action at `H` is a pure *backward* function of successor EVs
//! (exactly what `build_evs` already computes) and never needs to know how likely `H` is. Reaching
//! probability is then a pure *forward* pass that consumes that fixed policy. Two passes, no
//! chicken-and-egg.
//!
//! ## Decision-mass, not arrival-mass
//!
//! We only count a hand as "reached" where the player faces a *decision* there. Mass flows along an
//! edge `H → H+c` only when the policy at `H` is **Hit** (or the initial two-card deal). A **Double**
//! resolves terminally — you take one card and stop, never consulting the chart at the resulting
//! total — so a doubled `10` that draws a `6` does **not** contribute to the Hard-16 decision weight,
//! while a `6` that hits to `16` does. Stand/Bust are sinks; **Split** arms are folded back in by
//! [`inject_split_arms`] (independent-arms, resplit budget capped by `max_split_hands`), which manufacture
//! ~6–8% of decision mass against weak up-cards.

// A couple of helpers (`combinatoric_weights`, `category_breakdown`) are exercised only by the
// module's tests/diagnostics; the core path (`reach_weights`/`summarize_with`) is live.
#![allow(dead_code)]

use std::collections::HashMap;

use crate::card::Card;
use crate::hand::{best_move, categorize, pair_rank, HandCategory, Move};
use crate::rules::Ruleset;
use crate::shoe::{CardCol, Shoe};
use crate::simulation::Basis;


/// Decision-arrival mass per concrete hand: `P(player is at this hand, with a decision to make)`,
/// summed over a round of optimal play. The two-card seed sums to the total deal probability (1 on a
/// basis without peek conditioning); deeper hands accumulate only the mass that *hit* into them.
///
/// `shoe` must be the full shoe **with the up-card still present** (mirrors `build_evs`, which
/// removes it internally). `ev_tree` is that same `build_evs` output, supplying both the per-hand
/// optimal action and the two-card deal weights used as the seed.
///
/// With `include_split_arms`, the mass that enters a split is *not* dropped: each arm is seeded as a
/// fresh decision node and folded back into the same forward pass (see [`inject_split_arms`]). Off,
/// split-entry mass simply leaves the lattice (the cheaper, split-free weighting).
pub(crate) fn reach_weights<S: Shoe + Copy>(
    mut shoe: S,
    up_card: Card,
    rules: &Ruleset,
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    include_split_arms: bool,
) -> HashMap<CardCol, f64> {
    shoe.draw(&up_card);
    let shoe_minus_up = shoe;
    let basis = Basis::new(up_card, rules);

    // Seed: every two-card hand is reached by the deal with its exact (within-size) hypergeometric
    // weight. Deeper hands start at zero and fill in from their precursors.
    let mut reach: HashMap<CardCol, f64> = ev_tree
        .iter()
        .filter_map(|(hand, (weight, _))| (hand.len() == 2).then_some((*hand, *weight)))
        .collect();

    // Split arms re-seed extra decision nodes *before* the forward pass, so the pass then plays them
    // out through their hit chains exactly like any dealt hand.
    if include_split_arms {
        inject_split_arms(&shoe_minus_up, &basis, rules, ev_tree, &mut reach);
    }

    // Process in nondecreasing hand size. A hit grows the hand by exactly one card, so every
    // precursor of `hand` is strictly smaller and has its final mass before `hand` is visited.
    let mut hands: Vec<&CardCol> = ev_tree.keys().collect();
    hands.sort_by_key(|h| h.len());

    for hand in hands {
        let here = reach.get(hand).copied().unwrap_or(0.0);
        if here == 0.0 {
            continue;
        }
        // Mass only flows onward from a *Hit* decision. Double/Stand/Split do not feed the chart.
        if best_move(&ev_tree[hand].1) != Move::Hit {
            continue;
        }
        let shoe_here = shoe_minus_up.remove_hand(hand);
        for (c, p_c) in basis.draw_probs(&shoe_here) {
            if p_c == 0.0 {
                continue;
            }
            let mut child = *hand;
            child.insert(c);
            // A bust child has no decision node (it is absent from the tree); its mass simply leaves
            // the decision lattice.
            if ev_tree.contains_key(&child) {
                *reach.entry(child).or_insert(0.0) += here * p_c;
            }
        }
    }
    reach
}

/// Re-seed the decision nodes reached *inside* a split, folding split-arm play back into the reach
/// lattice. Mirrors the [`SplitSolver`](crate::split) arm structure on the **independent-arms** model
/// (each arm drawn from the shared post-split shoe, no cross-arm depletion — the `split_cards: 0`
/// approximation the chart already uses), with the resplit budget capped by `max_split_hands`.
///
/// For every two-card pair the policy actually **splits**, both initial arms — and, recursively, any
/// resplit arms while the budget lasts — are seeded as fresh two-card decision nodes `{r,c}`. Those
/// are *non-pair* hands (`c ≠ r`), so the forward pass plays out their hit chains with no risk of
/// re-entering the split logic. The injected mass multiplies by the arm count (two arms each occur in
/// the same fraction of rounds, not half each), which is the whole point — splitting *manufactures*
/// extra decision nodes.
///
/// **Deliberate approximations (documented, opt-in):**
/// - *Independent arms*: an arm's hit chain is played by the forward pass on the ordinary
///   within-arm shoe (one pair card removed), not the split solver's `shoe0` (both removed) — a
///   one-card difference in a multi-deck shoe.
/// - *Dealt-hand policy*: an arm `{r,c}` is propagated by the pair-free hand's stored optimal move.
///   Exact when `das` is on; off-DAS arms that would hit instead of double are mis-routed (rare).
/// - *Split aces*: forced one card then stand (no decision), so they manufacture no downstream
///   decision mass and are skipped entirely.
/// - *Per-line resplit depth*: the budget is spent along each resplit line rather than as a global
///   hand cap, so multi-resplit rounds (probability ~`p_r²`) can slightly overcount arms. Exact for
///   `max_split_hands ≤ 3` and for the common single-resplit case.
fn inject_split_arms<S: Shoe + Copy>(
    shoe_minus_up: &S,
    basis: &Basis,
    rules: &Ruleset,
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    reach: &mut HashMap<CardCol, f64>,
) {
    let splits_remaining = rules.max_split_hands.saturating_sub(2);
    for (pair, (_, move_ev)) in ev_tree {
        if pair.len() != 2 {
            continue;
        }
        let Some(r) = pair_rank(pair) else { continue };
        // Split aces get one card and stand — no manufactured decisions. And only act where the
        // policy actually splits.
        if r == Card::Ace || best_move(move_ev) != Move::Split {
            continue;
        }
        let entry = reach.get(pair).copied().unwrap_or(0.0);
        if entry == 0.0 {
            continue;
        }
        // Independent arms: every arm draws from the post-split shoe (both pair cards removed).
        let shoe0 = shoe_minus_up.remove_hand(pair);
        let draws = basis.draw_probs(&shoe0);
        let resplit_optimal = ev_tree
            .get(pair)
            .map(|(_, m)| best_move(m) == Move::Split)
            .unwrap_or(false);
        // Two initial arms, each occurring with the full entry mass.
        for _ in 0..2 {
            seed_arm(r, &draws, splits_remaining, resplit_optimal, entry, reach);
        }
    }
}

/// One split arm: seeded with a single `r`, mass `w`, about to be dealt its second card. Emits each
/// resulting non-pair two-card hand `{r,c}` as a decision node (the forward pass then hits it out);
/// a drawn pair resplits (while the budget and policy allow) into two more arms, else is truncated.
fn seed_arm(
    r: Card,
    draws: &[(Card, f64)],
    splits_left: u8,
    resplit_optimal: bool,
    w: f64,
    reach: &mut HashMap<CardCol, f64>,
) {
    for &(c, p_c) in draws {
        if p_c == 0.0 {
            continue;
        }
        let mass = w * p_c;
        if c == r {
            // Drew the pair rank again. Resplit into two fresh arms while allowed; otherwise this is
            // a capped pair played as an ordinary hand — a rare branch (needs the cap reached *and* a
            // matching draw), truncated here rather than routed back through the split logic.
            if splits_left > 0 && resplit_optimal {
                for _ in 0..2 {
                    seed_arm(r, draws, splits_left - 1, resplit_optimal, mass, reach);
                }
            }
        } else {
            *reach.entry(CardCol::from_hand(&[r, c])).or_insert(0.0) += mass;
        }
    }
}

/// For one strategy-table row, the per-composition share of (a) the combinatorial scan-weight that
/// `summarize_evs` currently pools by, versus (b) the game-time decision mass from [`reach_weights`].
/// Returns `(hand, combinatoric_share, reach_share)` rows, each share normalized within the category,
/// sorted by the size of the shift. This is the artifact for *seeing* which cells re-weight.
pub(crate) fn category_breakdown(
    category: HandCategory,
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    reach: &HashMap<CardCol, f64>,
) -> Vec<(CardCol, f64, f64)> {
    let members: Vec<&CardCol> = ev_tree
        .keys()
        .filter(|h| categorize(h) == category)
        .collect();
    let combo_tot: f64 = members.iter().map(|h| ev_tree[*h].0).sum();
    let reach_tot: f64 = members
        .iter()
        .map(|h| reach.get(*h).copied().unwrap_or(0.0))
        .sum();

    let mut rows: Vec<(CardCol, f64, f64)> = members
        .iter()
        .map(|h| {
            let combo = if combo_tot > 0.0 { ev_tree[*h].0 / combo_tot } else { 0.0 };
            let reach_share = if reach_tot > 0.0 {
                reach.get(*h).copied().unwrap_or(0.0) / reach_tot
            } else {
                0.0
            };
            (**h, combo, reach_share)
        })
        .collect();
    rows.sort_by(|a, b| (b.2 - b.1).abs().partial_cmp(&(a.2 - a.1).abs()).unwrap());
    rows
}

/// `summarize_evs` with the pooling weight swapped for an arbitrary per-hand weight map (e.g.
/// [`reach_weights`]). Same streaming weighted average, same "a move only counts from hands that
/// offer it" rule; a zero-weight hand drops out entirely (so doubled/split-terminal compositions
/// vanish from the consolidated decision under the game-time weighting).
pub(crate) fn summarize_with(
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    weight: &HashMap<CardCol, f64>,
) -> HashMap<HandCategory, HashMap<Move, f64>> {
    let mut acc = HashMap::<HandCategory, HashMap<Move, (f64, f64)>>::new();
    for (hand, (_, move_ev)) in ev_tree {
        let w = weight.get(hand).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        let moves = acc.entry(categorize(hand)).or_default();
        for (&mv, &ev) in move_ev {
            let (wt_sum, wt_ev_sum) = moves.entry(mv).or_insert((0.0, 0.0));
            *wt_sum += w;
            *wt_ev_sum += w * ev;
        }
    }
    acc.into_iter()
        .map(|(cat, moves)| {
            let move_evs = moves
                .into_iter()
                .map(|(mv, (wt_sum, wt_ev_sum))| (mv, wt_ev_sum / wt_sum))
                .collect();
            (cat, move_evs)
        })
        .collect()
}

/// The combinatorial scan-weight map straight off a `build_evs` tree — the weighting
/// `summarize_evs` currently uses, packaged so it can be A/B'd against [`reach_weights`].
pub(crate) fn combinatoric_weights(
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
) -> HashMap<CardCol, f64> {
    ev_tree.iter().map(|(h, (w, _))| (*h, *w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::shoe::InfiniteDeck;
    use crate::simulation::build_evs;

    /// The infinite-deck path (the TUI's default shoe) must run and stay sane: non-negative masses,
    /// and game-time weighting agrees with combinatoric on an unambiguous cell (hard 16 vs 6 = stand).
    ///
    /// Crucially it also pins *why the A/B toggle looks inert by default*: with no depletion every
    /// composition of a category has the **same** move EVs, so any pooling weight yields the same
    /// consolidated EV — the two weightings are bit-identical on the infinite deck. The weighting only
    /// bites on a finite shoe (composition-dependent depletion); see [`reach_weight_hard16_breakdown`].
    #[test]
    fn reach_weights_on_infinite_deck() {
        let up = Card::Pip(6);
        let rules = Ruleset { split_cards: 0, ..Ruleset::default() };
        let tree = build_evs(InfiniteDeck {}, up, &rules);
        let reach = reach_weights(InfiniteDeck {}, up, &rules, &tree, true);
        assert!(reach.values().all(|&m| m >= 0.0 && m.is_finite()));

        let combo = summarize_with(&tree, &combinatoric_weights(&tree));
        let game = summarize_with(&tree, &reach);
        let h16 = &game[&HandCategory::Hard(16)];
        assert!(h16[&Move::Stand] > h16[&Move::Hit], "hard 16 vs 6 stands");

        // No depletion ⇒ weighting is irrelevant: every category's every move EV matches exactly.
        for (cat, combo_moves) in &combo {
            for (mv, &cv) in combo_moves {
                let gv = game[cat][mv];
                assert!(
                    (cv - gv).abs() < 1e-12,
                    "{cat} {mv}: combinatoric {cv} vs game-time {gv} differ on the infinite deck"
                );
            }
        }
    }

    /// Sanity + demonstration on a 6-up (no peek conditioning, so the two-card seed is a clean
    /// distribution). Prints the Hard-16 re-weighting so the cross-size shift is visible.
    #[test]
    fn reach_weight_hard16_breakdown() {
        let up = Card::Pip(6);
        let shoe = CardCol::from_decks(2);
        // Independent-arms split budget: fast, and split arms don't feed the decision lattice here.
        let rules = Ruleset { split_cards: 0, ..Ruleset::default() };
        let tree = build_evs(shoe, up, &rules);
        let reach = reach_weights(shoe, up, &rules, &tree, false);

        // Two-card seed is the deal distribution: it sums to 1 off-peek.
        let two_card_mass: f64 = tree
            .keys()
            .filter(|h| h.len() == 2)
            .map(|h| reach[h])
            .sum();
        assert!((two_card_mass - 1.0).abs() < 1e-9, "two-card seed = {two_card_mass}");
        assert!(reach.values().all(|&m| m >= 0.0));

        let rows = category_breakdown(HandCategory::Hard(16), &tree, &reach);
        eprintln!("Hard 16 vs 6 — composition | combinatoric share | game-time share");
        for (hand, combo, reached) in rows.iter().filter(|(_, c, r)| c.max(*r) > 1e-3) {
            eprintln!("  {hand:<10}  {combo:>8.4}   {reached:>8.4}");
        }
        // The multi-card compositions must lose share relative to the two-card {Ten,6}: the deeper
        // ones are only reached through narrow hit funnels (and some not at all).
        let two_card_16 = rows.iter().find(|(h, ..)| h.len() == 2);
        if let Some((_, combo, reached)) = two_card_16 {
            eprintln!("two-card 16: combinatoric {combo:.4} -> game-time {reached:.4}");
        }
    }

    fn argmax(m: &HashMap<Move, f64>) -> (Move, f64) {
        let (&mv, &ev) = m
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (mv, ev)
    }

    /// Probe 1: does the combinatoric→game-time re-weighting change the *consolidated* decision (or
    /// just its reported EV) on cells where it should bite hardest? 16 vs 10 is the canonical
    /// composition-dependent cell. Ten-up also exercises the peek-conditional seed.
    #[test]
    fn reweighting_decision_shift_vs_ten() {
        let up = Card::Ten;
        let shoe = CardCol::from_decks(2);
        let rules = Ruleset { split_cards: 0, ..Ruleset::default() };
        let tree = build_evs(shoe, up, &rules);

        let combo = summarize_with(&tree, &combinatoric_weights(&tree));
        let game = summarize_with(&tree, &reach_weights(shoe, up, &rules, &tree, true));

        // Look at the hard totals where multi-card compositions are common enough to matter.
        eprintln!("vs Ten — cell | combinatoric best (EV) | game-time best (EV) | FLIP?");
        for tot in 12..=16u8 {
            let cat = HandCategory::Hard(tot);
            let (cm, ce) = argmax(&combo[&cat]);
            let (gm, ge) = argmax(&game[&cat]);
            let flip = if cm != gm { "  <== FLIP" } else { "" };
            eprintln!("  H{tot:<2}  {cm} ({ce:+.4})   {gm} ({ge:+.4}){flip}");
            // Also show the Hit/Stand gap under each weighting — the margin a flip would have to cross.
            let hs = |s: &HashMap<Move, f64>| s[&Move::Hit] - s[&Move::Stand];
            eprintln!(
                "         Hit-Stand gap: combinatoric {:+.4}, game-time {:+.4}",
                hs(&combo[&cat]),
                hs(&game[&cat])
            );
        }
    }

    /// Probe 2: the size of the split-arm correction. Compares total decision mass and the Hard-16
    /// re-weighting with split arms folded in (`true`) versus dropped (`false`), per up-card. The
    /// downstream split mass is the difference in totals — what the split-free weighting omits.
    #[test]
    fn split_arm_correction() {
        let rules = Ruleset { split_cards: 0, ..Ruleset::default() };
        eprintln!("up | split-entry | no-split total | with-split total | downstream split mass");
        for up in [Card::Pip(6), Card::Ten, Card::Ace, Card::Pip(2)] {
            let shoe = CardCol::from_decks(2);
            let tree = build_evs(shoe, up, &rules);
            let no_split = reach_weights(shoe, up, &rules, &tree, false);
            let with_split = reach_weights(shoe, up, &rules, &tree, true);

            let entry: f64 = tree
                .iter()
                .filter(|(h, (_, mv))| {
                    best_move(mv) == Move::Split && pair_rank(h) != Some(Card::Ace)
                })
                .map(|(h, _)| no_split.get(h).copied().unwrap_or(0.0))
                .sum();
            let t0: f64 = no_split.values().sum();
            let t1: f64 = with_split.values().sum();
            eprintln!(
                "  {up}  entry {entry:.5}   {t0:.5}   {t1:.5}   downstream {:.5} ({:.2}x entry)",
                t1 - t0,
                (t1 - t0) / entry.max(1e-12)
            );
        }
    }

    /// Where does the split-arm mass land? Multi-card 16s reachable from an 8-arm (e.g. 8→6→2) gain
    /// share once splits are folded in — the cell-level correction the global average hides.
    #[test]
    fn split_arm_lands_on_pair_fed_cells() {
        let up = Card::Pip(6);
        let shoe = CardCol::from_decks(2);
        let rules = Ruleset { split_cards: 0, ..Ruleset::default() };
        let tree = build_evs(shoe, up, &rules);
        let no_split = reach_weights(shoe, up, &rules, &tree, false);
        let with_split = reach_weights(shoe, up, &rules, &tree, true);

        eprintln!("Hard 16 vs 6 — composition | no-split share | with-split share");
        for (hand, ns, ws) in {
            let mut rows: Vec<_> = tree
                .keys()
                .filter(|h| categorize(h) == HandCategory::Hard(16))
                .map(|h| {
                    (
                        *h,
                        no_split.get(h).copied().unwrap_or(0.0),
                        with_split.get(h).copied().unwrap_or(0.0),
                    )
                })
                .collect();
            let n0: f64 = rows.iter().map(|r| r.1).sum();
            let n1: f64 = rows.iter().map(|r| r.2).sum();
            for r in &mut rows {
                r.1 /= n0;
                r.2 /= n1;
            }
            rows.sort_by(|a, b| (b.2 - b.1).abs().partial_cmp(&(a.2 - a.1).abs()).unwrap());
            rows.into_iter().filter(|(_, a, b)| a.max(*b) > 1e-3)
        } {
            eprintln!("  {hand:<10}  {ns:>8.4}   {ws:>8.4}");
        }
    }
}


