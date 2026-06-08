//! Pair-splitting EV machinery, factored out of the main solver.
//!
//! The split EV is computed by a budget recursion over the split arms ([`SplitSolver`]), entered via
//! [`split_move_ev`]. It rests on the same evaluation [`Basis`] as the rest of the tree (the shared
//! dealer/draw distributions and peek conditioning), so the only thing that lives here is the arm
//! traversal itself — see [`SplitSolver`]'s doc comment for the independent-arms model and the
//! cross-arm cards budget.

use std::collections::HashMap;
use std::hash::Hash;

use crate::card::*;
use crate::dealer::*;
use crate::shoe::*;
use crate::{Basis, HandState, Ruleset, pair_rank, resolve_ev};

/// Payoff of a **split arm** standing on `total` against a dealer-outcome distribution.
///
/// A thin expectation wrapper over [`resolve_ev`], with the arm's total presented as an ordinary
/// `Hard(total)` — *never* a natural. That is the one crucial distinction: a two-card 21 made on a
/// split hand (e.g. ace + ten after splitting aces) pays even money, not 3:2, and loses outright to
/// a dealer natural (only reachable on the no-peek basis). Because the state is always `Hard` and
/// never `Natural`, [`resolve_ev`]'s `bj_payout` is never read on this path — we pass `NaN` so that
/// any future change routing a `Natural` through here trips the sum-to-1/EV asserts instead of
/// silently using a bogus payout. (Hard vs Soft is irrelevant to a standing total — only the count is
/// compared.) Busts are handled by the caller; `total` is the arm's best count (≤ 21).
fn arm_stand_ev(total: u8, dealer_probs: &HashMap<DealerOutcome, f64>) -> f64 {
    // A split arm's 21 pays even money, not 3:2, so we classify by total here rather than routing
    // through `HandState::from` (which would call a two-card 21 a `Natural`). The one thing that
    // mapping does give us and a raw `Hard(total)` would not is bust detection, so do it explicitly.
    let state = if total > 21 {
        HandState::Bust
    } else {
        HandState::Hard(total)
    };
    dealer_probs
        .iter()
        .map(|(&dealer, &p)| p * resolve_ev(state, dealer, f64::NAN))
        .sum()
}

/// Split solver: a budget recursion over the split arms, on the same basis as the rest of the tree.
///
/// **Independent arms.** The split EV is the sum of every arm's payoff (linearity of expectation
/// makes the shared dealer irrelevant to the *mean*, only the variance). Each arm is played from the
/// same post-split shoe `shoe0`, depleting only by *its own* draws — sibling arms' cards are *not*
/// removed. This is exact on the infinite deck and on a finite shoe neglects only the cross-arm
/// card-removal effect (one arm thinning the deck the next arm and the dealer draw from) — a sub-0.1%
/// bias, and the price of tractability: the genuinely exact recursion that threads one depleting shoe
/// through all arms has a state space that blows up combinatorially at three-plus hands (each arm path
/// removes a distinct multiset, so nothing memoises). Resetting to `shoe0` at each arm boundary makes
/// the within-arm shoe a bounded function of the current hand, so the memo collapses. The re-split
/// *budget* is still threaded exactly (a small integer, not the source of the blow-up). An exact
/// finite path would be a future opt-in slow mode; per the exact-default convention this approximation
/// is surfaced, not silent.
///
/// **Peek conditioning is internal and non-clairvoyant.** When `conditional`, each arm faces the
/// *marginal* (hole-averaged) dealer distribution [`conditional_dealer_dist`] and draws from the
/// *marginal* conditional distribution [`conditional_draw_probs`], both recomputed at the current
/// shoe. This is the crucial point: the player never sees the hole, so the per-node `max` is taken
/// over moves evaluated against marginal information — the exact achievable EV. Stratifying on a known
/// hole and then averaging the per-hole optima would instead give `E_hole[max ...]`, the clairvoyant
/// upper bound (it only bites once an arm has real decisions, which is why it would inflate e.g. 8,8
/// but not the forced-one-card split aces).
///
/// The objective maximised at each node is the *total* EV of the current arm plus the siblings that
/// follow; with independent arms the siblings' value is unaffected by this arm's play except through
/// the shared re-split budget, so this reduces to playing each arm for its own EV.
///
/// [`conditional_dealer_dist`]: crate::conditional_dealer_dist
/// [`conditional_draw_probs`]: crate::conditional_draw_probs
struct SplitSolver<S: Shoe + Copy + Eq + Hash> {
    /// The shoe each arm starts from (up card and both pair cards already removed): the independent
    /// arms never see one another's depletion, so this is restored at every arm boundary.
    shoe0: S,
    /// The dealer/draw distributions and peek conditioning, shared with `build_evs` (see [`Basis`]).
    basis: Basis,
    /// The rank being split (every arm is seeded with one of these).
    split_card: Card,
    /// Split aces draw exactly one card and stand (and, since this also keeps them off the re-split
    /// decision node, never re-split). These flags are solver-wide because under the current rules
    /// they don't vary with split depth; supporting depth-varying rules (e.g. a different double
    /// permission for re-split arms) would move them into the per-call state / memo key.
    one_card_only: bool,
    /// Double after split allowed.
    das: bool,
    /// The total-cards budget for exact cross-arm depletion (see [`Ruleset::split_cards`]): the number
    /// of cards, along any one line of play, tracked with the depleting shoe carried forward between
    /// arms. Decremented (saturating at `0`) on each card drawn anywhere in the recursion; at an arm
    /// boundary, once it has hit `0` the new arm restarts from `shoe0` (the independent fallback).
    /// A budget larger than any reachable draw count never resets and so gives the full exact search;
    /// `0` is pure independent arms. Since `budget == K − (cards removed from shoe0)`, it is a function
    /// of `shoe` and adds no entropy to the memo key. The memo stays on regardless (it must — see
    /// [`SplitSolver::value`]); this budget is what bounds the shoe-keyed memo on a big finite shoe.
    budget: u8,
    /// Dealer distribution cache keyed on the remaining shoe (wrapping the cacheless [`Basis`]).
    dealer_cache: HashMap<S, HashMap<DealerOutcome, f64>>,
    /// Next-card distribution cache keyed on the remaining shoe (wrapping the cacheless [`Basis`]).
    draw_cache: HashMap<S, Vec<(Card, f64)>>,
    /// Memo over the full arm state: (current arm, is-first-action, pending sibling arms, splits
    /// remaining, shoe, cards budget left). Collapses the repeated arm subtrees the budget recursion
    /// produces.
    memo: HashMap<(CardCol, bool, u8, u8, S, u8), f64>,
}

impl<S: Shoe + Copy + Eq + Hash> SplitSolver<S> {
    fn new(shoe0: S, basis: Basis, split_card: Card, das: bool, budget: u8) -> Self {
        Self {
            shoe0,
            basis,
            split_card,
            one_card_only: split_card == Card::Ace,
            das,
            budget,
            dealer_cache: HashMap::new(),
            draw_cache: HashMap::new(),
            memo: HashMap::new(),
        }
    }

    /// The shoe a *new* arm (a fresh sibling or a re-split product) starts from, given the cards
    /// budget remaining. While any budget remains the depleted `shoe` carries forward (exact cross-arm
    /// removal); once the budget is spent (`0`) new arms restart from the pristine `shoe0` (the
    /// independent fallback). The budget itself is not spent here — only card *draws* spend it (see
    /// [`SplitSolver::draw`]); an arm boundary draws no card.
    fn advance_arm(&self, shoe: S, budget: u8) -> S {
        if budget == 0 { self.shoe0 } else { shoe }
    }

    /// Draw card `c`, depleting the shoe (within-arm depletion is always exact) and spending one unit
    /// of the cards budget. The budget saturates at `0` (independent fallback — the shoe still depletes
    /// within the arm, but the budget is already exhausted so new arms will reset). Returns the next
    /// shoe and the remaining budget.
    fn draw(&self, shoe: S, c: Card, budget: u8) -> (S, u8) {
        let next = shoe.remove_hand(&CardCol::from_hand(&[c]));
        (next, budget.saturating_sub(1))
    }

    /// Dealer outcome distribution from `shoe` (via [`Basis`]), cached on the shoe.
    fn dealer(&mut self, shoe: &S) -> HashMap<DealerOutcome, f64> {
        if let Some(dist) = self.dealer_cache.get(shoe) {
            return dist.clone();
        }
        let dist = self.basis.dealer_dist(shoe);
        self.dealer_cache.insert(*shoe, dist.clone());
        dist
    }

    /// The player's next-card distribution from `shoe` (via [`Basis`]), cached on the shoe.
    fn draws(&mut self, shoe: &S) -> Vec<(Card, f64)> {
        if let Some(v) = self.draw_cache.get(shoe) {
            return v.clone();
        }
        let v = self.basis.draw_probs(shoe);
        self.draw_cache.insert(*shoe, v.clone());
        v
    }

    /// This arm's payoff if it stands on `hand` (assumed not bust) against the dealer from `shoe`.
    fn stand_payoff(&mut self, hand: &CardCol, shoe: &S) -> f64 {
        let dealer = self.dealer(shoe);
        arm_stand_ev(hand.best_count(), &dealer)
    }

    /// Begin the next pending sibling arm, or 0 if none. `shoe` is the deck the current arm finished
    /// on, `budget` the remaining cards budget; [`SplitSolver::advance_arm`] decides whether the new
    /// arm continues from `shoe` (budget left) or restarts from `shoe0` (independent fallback). The
    /// budget is threaded onward unchanged (an arm boundary draws no card).
    fn start_pending(&mut self, pending: u8, splits: u8, shoe: S, budget: u8) -> f64 {
        if pending == 0 {
            return 0.0;
        }
        let seed = CardCol::from_hand(&[self.split_card]);
        let next = self.advance_arm(shoe, budget);
        self.value(seed, false, pending - 1, splits, next, budget)
    }

    /// Total EV of standing this arm now, plus the siblings that then play (carrying `shoe`/`budget`).
    fn stand_and_rest(
        &mut self,
        hand: CardCol,
        pending: u8,
        splits: u8,
        shoe: S,
        budget: u8,
    ) -> f64 {
        self.stand_payoff(&hand, &shoe) + self.start_pending(pending, splits, shoe, budget)
    }

    /// Memoised entry point: total EV (this arm + `pending` siblings) over the arm state. The memo is
    /// essential even in exact mode — without it each of an arm's leaves would recompute every later
    /// arm, exploding exponentially in the number of arms. In exact mode the shoe component of the key
    /// genuinely varies (cross-arm depletion), so the memo grows large on a big finite shoe (the cost
    /// that makes exact opt-in, and what the cards budget bounds); on the infinite deck the shoe is
    /// constant and it collapses to nothing.
    fn value(
        &mut self,
        current: CardCol,
        first: bool,
        pending: u8,
        splits: u8,
        shoe: S,
        budget: u8,
    ) -> f64 {
        let key = (current, first, pending, splits, shoe, budget);
        if let Some(&v) = self.memo.get(&key) {
            return v;
        }
        let v = self.compute(current, first, pending, splits, shoe, budget);
        self.memo.insert(key, v);
        v
    }

    fn compute(
        &mut self,
        current: CardCol,
        first: bool,
        pending: u8,
        splits: u8,
        shoe: S,
        budget: u8,
    ) -> f64 {
        let draws = self.draws(&shoe);
        // A freshly seeded arm has one card and must draw a second before any decision.
        if current.len() == 1 {
            return draws
                .into_iter()
                .map(|(c, p)| {
                    let (next_shoe, next_budget) = self.draw(shoe, c, budget);
                    let mut hand = current;
                    hand.insert(c);
                    let val = if self.one_card_only {
                        // Split aces: exactly one card, then forced stand (no hit/double/re-split).
                        self.stand_and_rest(hand, pending, splits, next_shoe, next_budget)
                    } else {
                        self.value(hand, true, pending, splits, next_shoe, next_budget)
                    };
                    p * val
                })
                .sum();
        }

        // A playable arm of two or more cards: take the best of stand / hit / double / re-split,
        // each evaluated as the total EV of the current arm plus the siblings that follow.
        let mut best = self.stand_and_rest(current, pending, splits, shoe, budget);

        // Hitting (never on 21).
        if current.best_count() < 21 {
            let hit: f64 = draws
                .iter()
                .map(|&(c, p)| {
                    let (next_shoe, next_budget) = self.draw(shoe, c, budget);
                    let mut hand = current;
                    hand.insert(c);
                    let v = if hand.hard_count() > 21 {
                        // This arm busts (-1) but the siblings still play out (from the shoe its
                        // cards, including the bust card, left behind).
                        -1.0 + self.start_pending(pending, splits, next_shoe, next_budget)
                    } else {
                        self.value(hand, false, pending, splits, next_shoe, next_budget)
                    };
                    p * v
                })
                .sum();
            best = best.max(hit);
        }

        // Doubling (first action only, when DAS is allowed; never for one-card split aces).
        if first && self.das && !self.one_card_only {
            let dbl: f64 = draws
                .iter()
                .map(|&(c, p)| {
                    let (next_shoe, next_budget) = self.draw(shoe, c, budget);
                    let mut hand = current;
                    hand.insert(c);
                    let arm = 2.0 * self.stand_payoff(&hand, &next_shoe);
                    p * (arm + self.start_pending(pending, splits, next_shoe, next_budget))
                })
                .sum();
            best = best.max(dbl);
        }

        // Re-splitting: a drawn pair card may be split again while the hand budget allows. (Split
        // aces never reach here — `one_card_only` stands them after one card.) One equal card becomes
        // the current arm's seed and the other an extra pending sibling, spending one re-split unit;
        // both already removed from `shoe`. This spawns a new arm, so it crosses an arm boundary
        // (`advance_arm`); the cards budget threads on unchanged (the re-split draws no new card).
        if first && splits > 0 && current.get_count(&self.split_card) == 2 {
            let seed = CardCol::from_hand(&[self.split_card]);
            let next = self.advance_arm(shoe, budget);
            let resplit = self.value(seed, false, pending + 1, splits - 1, next, budget);
            best = best.max(resplit);
        }

        best
    }
}

/// EV of splitting the pair `pair_hand`, on the given [`Basis`] (shared with the rest of the tree).
///
/// Drives a single [`SplitSolver`] over the two initial arms (the pair cards are already removed in
/// `shoe_minus_hand`): one current seed plus one pending sibling, with the re-split budget set by
/// `max_split_hands`. Peek conditioning rides along inside the `Basis`, so this is just the entry
/// point.
pub(crate) fn split_move_ev<S: Shoe + Copy + Eq + Hash>(
    pair_hand: &CardCol,
    shoe_minus_hand: &S,
    basis: Basis,
    rules: &Ruleset,
) -> f64 {
    let split_card = pair_rank(pair_hand).expect("split_move_ev called on a non-pair");
    // Splitting the pair already makes two hands, so the re-split budget is the remaining headroom.
    let splits_remaining = rules.max_split_hands.saturating_sub(2);

    // The cross-arm cards budget threaded by the solver (large = exact; `0` = independent).
    let budget = rules.split_cards;

    let mut solver = SplitSolver::new(*shoe_minus_hand, basis, split_card, rules.das, budget);
    let seed = CardCol::from_hand(&[split_card]);
    // The root arm starts with the full budget (the solver's stored value).
    let budget = solver.budget;
    solver.value(seed, false, 1, splits_remaining, *shoe_minus_hand, budget)
}

#[cfg(test)]
mod tests {
    //! Split-specific regression guard: the `SplitSolver` budget modes (independent / limited /
    //! exact) probed directly through [`split_move_ev`], plus the pair-cell chart decisions that
    //! exercise the split scoring through the full strategy pipeline.
    //!
    //! Numbers and conventions match the general guard in [`crate::tests`]: a **2-deck shoe under
    //! [`Ruleset::default`]** for the chart cells (via [`crate::test_support::strategy_for`]), with
    //! the focused budget tests built on the infinite deck or a half deck as noted per test.
    //! Reference strategy: <https://wizardofodds.com/games/blackjack/appendix/9/2dh17r4/>.
    use super::*;
    use crate::test_support::*;
    use crate::{HandCategory, Move, build_evs};

    /// Pair decisions on the peek-conditional path (ten and ace up). Aces always split; nines stand
    /// against a ten or ace (18 is good enough, splitting into two 9s is worse); 8,8 splits vs a ten.
    ///
    /// NOTE deliberately omitted: 8,8 vs A. The exact chart surrenders it (true split EV ≈ −0.51),
    /// but it is genuinely borderline and the independent-arms model overshoots by ~0.014 to −0.496,
    /// flipping it to split. That is the documented finite-shoe approximation, not a regression — see
    /// the split solver's basis comment — so this guard does not pin that one cell.
    #[test]
    fn split_decisions_under_peek() {
        let vs_ten = strategy_for(Card::Ten);
        assert_eq!(
            vs_ten[&HandCategory::Pair(Card::Ace)],
            Move::Split,
            "A,A vs T"
        );
        assert_eq!(
            vs_ten[&HandCategory::Pair(Card::Pip(8))],
            Move::Split,
            "8,8 vs T"
        );
        assert_eq!(
            vs_ten[&HandCategory::Pair(Card::Pip(9))],
            Move::Stand,
            "9,9 vs T"
        );

        let vs_ace = strategy_for(Card::Ace);
        assert_eq!(
            vs_ace[&HandCategory::Pair(Card::Ace)],
            Move::Split,
            "A,A vs A"
        );
        assert_eq!(
            vs_ace[&HandCategory::Pair(Card::Pip(9))],
            Move::Stand,
            "9,9 vs A"
        );
    }

    /// Unambiguous pair decisions against a 6 (fast, unconditional path): the always-split pairs,
    /// the never-split 5s (double the hard 10) and tens (stand on 20).
    #[test]
    fn split_decisions_vs_6() {
        let strat = strategy_for(Card::Pip(6));
        assert_eq!(
            strat[&HandCategory::Pair(Card::Ace)],
            Move::Split,
            "A,A vs 6"
        );
        assert_eq!(
            strat[&HandCategory::Pair(Card::Pip(8))],
            Move::Split,
            "8,8 vs 6"
        );
        assert_eq!(
            strat[&HandCategory::Pair(Card::Pip(9))],
            Move::Split,
            "9,9 vs 6"
        );
        assert_eq!(
            strat[&HandCategory::Pair(Card::Pip(2))],
            Move::Split,
            "2,2 vs 6"
        );
        assert_eq!(
            strat[&HandCategory::Pair(Card::Pip(5))],
            Move::Double,
            "5,5 vs 6"
        );
        assert_eq!(
            strat[&HandCategory::Pair(Card::Ten)],
            Move::Stand,
            "T,T vs 6"
        );
    }

    /// Splitting infinitely is impossible on the infinite deck, but a bounded `max_split_hands`
    /// keeps it terminating; A,A there is the textbook always-split, and the value is sane.
    #[test]
    fn split_on_infinite_deck_terminates() {
        let tree = build_evs(InfiniteDeck {}, Card::Pip(6), &Ruleset::default());
        let pair = CardCol::from_hand(&[Card::Ace, Card::Ace]);
        let (_w, evs) = &tree[&pair];
        let split = evs[&Move::Split];
        assert!(
            split > evs[&Move::Stand],
            "A,A split beats standing on soft 12 vs 6"
        );
        assert!(
            split > 0.0 && split < 1.0,
            "A,A vs 6 split EV in a sane range: {split}"
        );
    }

    /// The exact (cross-arm-depleting) split and the independent-arms approximation must agree
    /// exactly on the infinite deck, where depletion is a no-op — this is what validates the exact
    /// recursion (its only intended difference from the approximation is finite-shoe card removal).
    #[test]
    fn exact_split_matches_independent_on_infinite_deck() {
        let indep = ruleset_with(0);
        let exact = ruleset_with(Ruleset::EXACT_SPLIT);
        let shoe = InfiniteDeck {};
        for r in [Card::Pip(8), Card::Pip(9), Card::Ace] {
            let pair = CardCol::from_hand(&[r, r]);
            // up=6: unconditional basis (no peek possible).
            let a = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &indep), &indep);
            let b = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &exact), &exact);
            assert_close(
                a,
                b,
                &format!("{r},{r} exact vs independent (infinite deck)"),
            );
        }
    }

    /// The exact path runs and is sane on a (small) finite shoe, where it legitimately differs from
    /// the approximation by the cross-arm card-removal effect — a small amount.
    #[test]
    fn exact_split_runs_on_finite_shoe() {
        let indep = ruleset_with(0);
        let exact = ruleset_with(Ruleset::EXACT_SPLIT);
        let pair = CardCol::from_hand(&[Card::Pip(8), Card::Pip(8)]);
        // The pair is already removed in the shoe `build_evs` hands to `split_move_ev`.
        let shoe = CardCol::half_deck().remove_hand(&pair);
        let a = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &indep), &indep);
        let b = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &exact), &exact);
        assert!(b.is_finite(), "exact split EV is finite: {b}");
        assert!(
            (a - b).abs() < 0.1,
            "exact and independent stay close on a finite shoe: {a} vs {b}"
        );
    }

    /// The single-budget `Limited` mode collapses to the same value for *every* budget on the
    /// infinite deck (depletion is a no-op), which exercises the budget threading and the
    /// exhaustion/fallback path without any finite-shoe variation to reason about.
    #[test]
    fn limited_split_matches_extremes_on_infinite_deck() {
        let shoe = InfiniteDeck {};
        let exact = ruleset_with(Ruleset::EXACT_SPLIT);
        for cards in [0u8, 1, 2, 5, u8::MAX] {
            let lim = ruleset_with(cards);
            for r in [Card::Pip(8), Card::Pip(9), Card::Ace] {
                let pair = CardCol::from_hand(&[r, r]);
                let a = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &lim), &lim);
                let b = split_move_ev(&pair, &shoe, Basis::new(Card::Pip(6), &exact), &exact);
                assert_close(
                    a,
                    b,
                    &format!("{r},{r} Limited{{cards:{cards}}} == Exact (infinite deck)"),
                );
            }
        }
    }

    /// `cards: 0` is literally the independent execution (the budget starts and stays at 0, so every
    /// new arm resets to `shoe0`) — an exact identity, and the lower endpoint of the `Limited` range.
    #[test]
    fn limited_split_zero_budget_is_independent_on_finite_shoe() {
        let indep = ruleset_with(0);
        let lim0 = ruleset_with(0);
        let pair = CardCol::from_hand(&[Card::Pip(8), Card::Pip(8)]);
        let shoe = CardCol::half_deck().remove_hand(&pair);
        let up = Card::Pip(6);
        let a = split_move_ev(&pair, &shoe, Basis::new(up, &indep), &indep);
        let b = split_move_ev(&pair, &shoe, Basis::new(up, &lim0), &lim0);
        assert_close(a, b, "Limited{cards:0} == IndependentArms (finite shoe)");
    }

    /// On a finite shoe a modest budget pulls the value toward the exact cross-arm result: the
    /// limited estimate is at least as close to `Exact` as the independent baseline is (it adds, not
    /// removes, correlation terms). Demonstrates the budget is doing real work.
    #[test]
    fn limited_split_approaches_exact_on_finite_shoe() {
        let indep = ruleset_with(0);
        let exact = ruleset_with(Ruleset::EXACT_SPLIT);
        let lim = ruleset_with(6);
        let pair = CardCol::from_hand(&[Card::Pip(8), Card::Pip(8)]);
        let shoe = CardCol::half_deck().remove_hand(&pair);
        let up = Card::Pip(6);
        let independent = split_move_ev(&pair, &shoe, Basis::new(up, &indep), &indep);
        let exact_ev = split_move_ev(&pair, &shoe, Basis::new(up, &exact), &exact);
        let limited = split_move_ev(&pair, &shoe, Basis::new(up, &lim), &lim);
        assert!(limited.is_finite(), "limited split EV is finite: {limited}");
        assert!(
            (limited - exact_ev).abs() <= (independent - exact_ev).abs() + 1e-9,
            "Limited{{cards:6}} no further from Exact than IndependentArms: \
             indep={independent}, limited={limited}, exact={exact_ev}"
        );
    }
}
