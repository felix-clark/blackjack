pub(crate) mod card;
pub(crate) mod dealer;
mod legacy;
pub(crate) mod shoe;
mod split;
#[cfg(test)]
mod test_support;

use card::*;
use dealer::*;
use shoe::*;
use split::split_move_ev;

use std::{
    collections::HashMap,
    default::Default,
    fmt::{Debug, Display},
    hash::Hash,
};

#[derive(PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum HandState {
    Bust,
    Soft(u8),
    Hard(u8),
    Natural,
}

impl Display for HandState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandState::Bust => write!(f, "Bust"),
            HandState::Soft(n) => write!(f, "S{}", n),
            HandState::Hard(n) => write!(f, "H{}", n),
            HandState::Natural => write!(f, "Nat"),
        }
    }
}

impl From<&CardCol> for HandState {
    fn from(hand: &CardCol) -> Self {
        if hand.is_nat21() {
            return Self::Natural;
        }
        let has_ace = hand.has_ace();
        let hard_count = hand.hard_count();
        assert!(
            !has_ace || hand.len() != 2 || hard_count != 11,
            "Natural 21 should be taken care of already"
        );
        if hard_count > 21 {
            return Self::Bust;
        }
        if has_ace && hard_count + 10 <= 21 {
            Self::Soft(hard_count + 10)
        } else {
            Self::Hard(hard_count)
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub(crate) enum Move {
    Hit,
    Stand,
    Double,
    Split,
    Surrender,
}

// TODO: similar to Move, we might need an enum for recommended strategy, which encodes DoubleHit
// and DoubleStand (to double if allowed, but Hit/Stand otherwise).

impl Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Move::Hit => write!(f, "H"),
            Move::Stand => write!(f, "S"),
            Move::Double => write!(f, "D"),
            Move::Split => write!(f, "P"),
            Move::Surrender => write!(f, "R"),
        }
    }
}

/// The row a concrete hand occupies in a strategy table.
///
/// Distinct from [`HandState`]: a pair is *also* a hard or soft total, but it is a different
/// decision (split is available, and only here), so it gets its own category rather than being
/// pooled into the corresponding total. `A,A` is `Pair(Ace)` and `T,T` is `Pair(Ten)` — neither
/// falls through to `Soft`/`Hard`/`Natural`. Hard and soft categories still pool every composition
/// (and size) of that total, which is where composition-dependent strategy is averaged out.
#[derive(PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum HandCategory {
    Hard(u8),
    Soft(u8),
    Pair(Card),
    Natural,
}

impl Display for HandCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandCategory::Hard(n) => write!(f, "H{}", n),
            HandCategory::Soft(n) => write!(f, "S{}", n),
            HandCategory::Pair(c) => write!(f, "{},{}", c, c),
            HandCategory::Natural => write!(f, "Nat"),
        }
    }
}

/// When (if ever) the player may forfeit half the bet instead of playing the hand out.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum SurrenderRule {
    /// Surrender is not offered.
    None,
    /// Surrender *before* the dealer peeks for blackjack, escaping the dealer-natural loss too.
    /// EV is an unconditional -0.5.
    Early,
    /// Surrender *after* the dealer peeks and shows no blackjack. Only coherent when the dealer
    /// actually peeks (`dealer_check`), since otherwise there is no "after the check".
    Late,
}

/// The stipulation of miscellaneous rules other than the number of decks (?).
pub(crate) struct Ruleset {
    /// Whether the dealer hits soft 17
    pub(crate) hs17: bool,
    /// Allowed to double after split
    pub(crate) das: bool,
    /// Whether the dealer checks their hole card for blackjack
    /// Note that the worst version of this being false causes a dealer blackjack to take
    /// all splits and doubles.
    pub(crate) dealer_check: bool,
    // /// Double on anything (as opposed to just 10 and 11) -- maybe just assume true
    // doa: bool,
    /// What a player natural (a two-card blackjack) pays, as a multiple of the bet. The good and
    /// near-universal value is `1.5` (3:2); some tables pay `1.2` (6:5), a strictly worse rule.
    /// Only ever applies to a genuine first-deal natural — a split-arm 21 is *not* a blackjack and
    /// pays even money regardless (see [`arm_stand_ev`]).
    pub(crate) bj_payout: f64,
    /// Whether (and when) the player may surrender.
    pub(crate) surrender: SurrenderRule,
    /// Maximum number of hands the player may end up with after splitting (so the number of splits
    /// allowed is `max_split_hands - 1`). Caps the split recursion — and is what keeps the infinite
    /// deck terminating, since otherwise a pair could be re-split without bound. Setting it to
    /// `4 * n_decks` recovers unbounded splitting on a finite shoe. `< 2` disables splitting.
    pub(crate) max_split_hands: u8,
    /// Split accuracy as a single budget on the **total number of cards drawn** (across all arms)
    /// that are tracked with exact cross-arm depletion. While the budget lasts the depleting shoe
    /// carries forward between arms (each arm sees the cards earlier arms removed — the true
    /// finite-shoe correlation); once this many cards have been drawn, further arms restart from the
    /// pristine post-split shoe (within-arm depletion stays exact throughout — it is cheap and
    /// self-contained; only the expensive cross-arm linkage is truncated).
    ///
    /// One total-cards cap because it sets the truncation *order* directly: a draw path of n cards has
    /// probability ~1/13ⁿ, so capping at K neglects the cross-arm correction uniformly at ~1/13^(K+1),
    /// and the budget is spent first on the shallow, high-probability draws where the correction is
    /// largest. The carried-shoe diversity is bounded by ≤K-card removals from the post-split shoe, a
    /// count independent of deck size, so a small K stays tractable even on 8 decks where a full search
    /// is infeasible. `0` is the old independent-arms approximation (no cross-arm correlation); a K
    /// larger than any reachable draw count (see [`Ruleset::EXACT_SPLIT`]) never resets and so is the
    /// full exact search. The default `4` is ~5–10× more accurate than independent (sub-1e-4 vs the
    /// exact value) while staying sub-second per query.
    pub(crate) split_cards: u8,
    // TODO: finer split-aces rules. Currently split aces always get exactly one card and cannot be
    // re-split (the common rule); a future axis could relax either, and "no double on split aces /
    // tens" would refine `das` per split rank — see the `SplitSolver` field comments.
}

impl Ruleset {
    /// A `split_cards` budget larger than any draw count a split can reach, so the cross-arm
    /// truncation never fires: the full exact split search (every drawn card tracked, all arms). Not a
    /// magic sentinel — it is just a large `K` that never decrements to `0` in practice.
    /// Combinatorially infeasible on a big shoe — use it for single-query validation, not whole-chart
    /// builds. See [`Ruleset::split_cards`].
    #[allow(dead_code)] // public knob; currently only the tests construct an exact ruleset
    pub(crate) const EXACT_SPLIT: u8 = u8::MAX;

    /// Reject rule combinations that don't correspond to a real game. Late surrender is defined as
    /// surrendering after the dealer peeks, so it only makes sense when the dealer peeks at all.
    fn validate(&self) {
        if self.surrender == SurrenderRule::Late {
            assert!(
                self.dealer_check,
                "Late surrender requires the dealer to peek for blackjack (dealer_check); \
                 use SurrenderRule::Early for a no-peek game."
            );
        }
    }
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            hs17: true,
            das: true,
            dealer_check: true,
            bj_payout: 1.5,
            surrender: SurrenderRule::Late,
            max_split_hands: 4,
            // This is technically not a ruleset option, but a computational precision vs.
            // investment option that specifies the depth of the exact enumeration in multiple
            // splits.
            split_cards: 4,
        }
    }
}

/// Terminal payoff of a standing/resolved player hand against one dealer outcome, as a multiple of
/// the bet. Keyed on the collapsed [`HandState`] so callers control natural-eligibility: only a
/// hand presented as [`HandState::Natural`] earns `bj_payout` (and pushes a dealer natural) — a
/// split-arm 21 is presented as an ordinary total and so loses to a dealer natural and pays even
/// money (see [`arm_stand_ev`]). `bj_payout` is the table's blackjack payout (3:2 = 1.5, 6:5 = 1.2).
pub(crate) fn resolve_ev(player_state: HandState, dealer_state: DealerOutcome, bj_payout: f64) -> f64 {
    match (player_state, dealer_state) {
        (HandState::Natural, DealerOutcome::Natural) => 0.,
        (_, DealerOutcome::Natural) => -1.,
        (HandState::Natural, _) => bj_payout,
        (HandState::Bust, _) => -1.,
        (_, DealerOutcome::Bust) => 1.,
        (HandState::Hard(p) | HandState::Soft(p), DealerOutcome::Total(d)) => match p.cmp(&d) {
            std::cmp::Ordering::Less => -1.,
            std::cmp::Ordering::Equal => 0.,
            std::cmp::Ordering::Greater => 1.,
        },
    }
}

/// Probability the dealer's down card completes a natural, given the up card and the cards left in
/// `shoe`. Exact because a natural is the only two-card 21: it needs the single rank that pairs with
/// the up card to make ace+ten, and nothing else can.
fn dealer_natural_prob(up_card: Card, shoe: &impl Shoe) -> f64 {
    match up_card {
        Card::Ace => shoe.draw_prob(&Card::Ten),
        Card::Ten => shoe.draw_prob(&Card::Ace),
        _ => 0.0,
    }
}

/// The rank that completes a dealer natural with `up_card` (Ten under an Ace, Ace under a Ten), or
/// `None` for any other up-card, where a natural is impossible and no peek conditioning applies.
fn natural_hole_rank(up_card: Card) -> Option<Card> {
    match up_card {
        Card::Ace => Some(Card::Ten),
        Card::Ten => Some(Card::Ace),
        _ => None,
    }
}

/// Exact dealer outcome distribution **conditioned on the dealer having peeked and shown no
/// natural**, given the deck `shoe` left after the up-card and the player's hand are removed.
///
/// Conditioning on "no dealer natural" is exactly conditioning on the hole card: it cannot be the
/// `bj_rank`. So we stratify on the hole — for every non-`bj_rank` hole we seed the dealer with
/// `(up_card, hole)`, remove that hole from the deck the dealer then draws from, and average the
/// resulting distributions weighted by the conditional hole probability `P(hole | not natural)`.
/// This is more than dropping the natural and renormalising ([`remove_nat21`]): removing the
/// concrete hole before the dealer's later draws is what makes it exact on a finite shoe.
fn conditional_dealer_dist<S: Shoe>(
    up_card: Card,
    bj_rank: Card,
    shoe: &S,
    hs17: bool,
) -> HashMap<DealerOutcome, f64> {
    let norm = 1.0 - shoe.draw_prob(&bj_rank); // P(hole is not the natural-completing rank)
    let mut acc = HashMap::<DealerOutcome, f64>::new();
    for (hole, p_hole) in shoe.all_draw_probs() {
        if hole == bj_rank {
            continue;
        }
        let w = p_hole / norm;
        let seed = CardCol::from_hand(&[up_card, hole]);
        let deck = shoe.remove_hand(&CardCol::from_hand(&[hole]));
        for (outcome, p) in dealer_outcome_probs(seed, &deck, hs17) {
            *acc.entry(outcome).or_insert(0.0) += w * p;
        }
    }
    acc
}

/// The player's next-card distribution **conditioned on the dealer having peeked and shown no
/// natural**, over the deck `shoe` left after the up-card and the player's hand are removed.
///
/// The dealer's (unseen) hole has been ruled out as the `bj_rank`, which removes one non-`bj_rank`
/// card from the deck the player draws from — shifting the composition. Marginalising the hidden
/// hole gives `P(draw c | not natural) = Σ_hole P(hole | not natural) · P(draw c | hole removed)`,
/// computed exactly here. Only `Stand`/`Double`/`Hit` *draw* a card, so this is where the
/// player-side card-removal effect of the peek enters; the recursion's `max` over the child's
/// already-conditioned move EVs keeps the continuation non-clairvoyant (the player never "sees" the
/// hole), so the result is the exact achievable EV rather than an upper bound.
fn conditional_draw_probs<S: Shoe>(bj_rank: Card, shoe: &S) -> Vec<(Card, f64)> {
    let norm = 1.0 - shoe.draw_prob(&bj_rank);
    // Each admissible hole, paired with its conditional weight and the deck it leaves behind.
    let hole_decks: Vec<(f64, S)> = shoe
        .all_draw_probs()
        .filter(|&(hole, _)| hole != bj_rank)
        .map(|(hole, p_hole)| {
            (
                p_hole / norm,
                shoe.remove_hand(&CardCol::from_hand(&[hole])),
            )
        })
        .collect();
    shoe.all_draw_probs()
        .map(|(c, _)| {
            let p_c = hole_decks
                .iter()
                .map(|(w, deck)| w * deck.draw_prob(&c))
                .sum();
            (c, p_c)
        })
        .collect()
}

/// The evaluation **basis** for one up-card: the dealer-outcome distribution and the player's
/// next-card distribution as functions of the remaining shoe.
///
/// This is the shared kernel both player-EV traversals rest on. `build_evs` (its bottom-up
/// partition DP) and [`SplitSolver`] (its top-down arm recursion) are deliberately *different*
/// traversals — one produces the whole weighted tree, the other a single arm's value with a re-split
/// budget — but they ask the deck the *same* two questions, conditioned the *same* way. Centralising
/// that here keeps the subtle peek-conditioning (on which the affine-collapse property depends) in one
/// place instead of three. When `conditional` (the dealer peeks and the up-card can make a natural)
/// both distributions are the marginal, hole-averaged ones conditioned on no dealer natural
/// ([`conditional_dealer_dist`] / [`conditional_draw_probs`]); otherwise they are the plain
/// unconditional ones. It is cacheless by design — `build_evs` queries each shoe once (every hand has
/// a distinct remaining shoe), while `SplitSolver` wraps it in its own caches where arms revisit
/// shoes.
#[derive(Clone, Copy)]
pub(crate) struct Basis {
    up_card: Card,
    /// The hole rank that completes a dealer natural (`Some` only for an Ace/Ten up); `None` off peek.
    bj_rank: Option<Card>,
    /// Whether to condition on a clean dealer peek (no natural).
    conditional: bool,
    hs17: bool,
}

impl Basis {
    pub(crate) fn new(up_card: Card, rules: &Ruleset) -> Self {
        let bj_rank = natural_hole_rank(up_card);
        Self {
            up_card,
            bj_rank,
            conditional: rules.dealer_check && bj_rank.is_some(),
            hs17: rules.hs17,
        }
    }

    /// Dealer outcome distribution from `shoe` on this basis.
    pub(crate) fn dealer_dist<S: Shoe>(&self, shoe: &S) -> HashMap<DealerOutcome, f64> {
        if self.conditional {
            conditional_dealer_dist(self.up_card, self.bj_rank.unwrap(), shoe, self.hs17)
        } else {
            dealer_outcome_probs(CardCol::from_hand(&[self.up_card]), shoe, self.hs17)
        }
    }

    /// The player's next-card distribution from `shoe` on this basis.
    pub(crate) fn draw_probs<S: Shoe>(&self, shoe: &S) -> Vec<(Card, f64)> {
        if self.conditional {
            conditional_draw_probs(self.bj_rank.unwrap(), shoe)
        } else {
            shoe.all_draw_probs().collect()
        }
    }
}

/// Returns a map from a given player hand to a probability weight and an expectation value for each
/// move made with that hand, assuming optimal H/S strategy afterwards.
///
/// **Basis.** When the dealer peeks (`rules.dealer_check`) and the up-card can make a natural
/// (Ace/Ten), every play EV here is *conditioned on the dealer having peeked and shown no natural* —
/// the realistic "the hand is live, how do I play it" value, with both the dealer distribution and
/// the player's own draws conditioned (see [`conditional_dealer_dist`] / [`conditional_draw_probs`]).
/// Otherwise (no peek, or an up-card that can't make a natural) the EVs are unconditional. A player
/// natural is always recorded on the unconditional basis — it has no decision and resolves at the
/// peek, so its honest EV is `(1 - P_bj)·1.5` (a push against a dealer natural) regardless of rules.
/// The unconditional/house-advantage value of a peek game is recoverable from this tree at the
/// two-card root: `-P_bj + (1 - P_bj)·V`, exact because the peek precedes the player's move.
///
/// The weight is the shoe's partition scan-weight for that exact multiset (see
/// [`Shoe::weighted_partitions`]), times `(1 - P_bj)` on the conditional basis so it is the
/// conditional occurrence probability `P(hold hand | no dealer natural)`. Within a fixed hand size
/// it is exact; across sizes the scan-weight is not a coherent distribution (a known deferred
/// imprecision), so it is only a relative pooling weight in [`summarize_evs`].
// TODO: Should this be a struct so it can recursively build the table by demand?
pub(crate) fn build_evs<S: Shoe + Copy + Eq + Hash>(
    mut shoe: S,
    up_card: Card,
    rules: &Ruleset,
) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
    rules.validate();
    // Remove the up card from the deck (a no-op for the infinite deck).
    shoe.draw(&up_card);
    // make into const after draw
    let shoe = shoe;

    // The future dealer draws are not totally independent from the player choices, so to be precise
    // we must wait to resolve the dealer's result conditioned on the players hand. The basis bundles
    // the dealer/draw distributions and the peek conditioning (shared with the split solver).
    let dealer_hand = CardCol::from_hand(&[up_card]);
    let basis = Basis::new(up_card, rules);
    // Peek conditioning only bites when the dealer actually peeks *and* the up-card can make a
    // natural; otherwise the conditional and unconditional bases coincide.
    let conditional = basis.conditional;

    let mut full_ev_tree = HashMap::<CardCol, (f64, HashMap<Move, f64>)>::new();

    // Go down to 2 to get all soft options as well
    for pl_tot in (2..=21).rev() {
        for (weight, pl_hand) in shoe.weighted_partitions(pl_tot) {
            if pl_hand.len() < 2 {
                continue;
            }

            // Assert that we aren't overdrawing; this should be a given if
            // weighted_hard_partitions() is correct (always true for the infinite deck).
            assert!(shoe.contains_hand(&pl_hand));
            let shoe_minus_hand = shoe.remove_hand(&pl_hand);

            // A natural is a terminal two-card hand with no decision to make: it simply resolves
            // (3:2, or a push against a dealer natural). Record it as a leaf so the consolidation
            // and house-advantage layers can account for it, but offer it no playable moves — in
            // particular never a double or surrender. It is always scored on the *unconditional*
            // dealer distribution: a natural resolves at the peek itself, so its honest EV must keep
            // the push-against-a-dealer-natural mass rather than condition it away. A natural is
            // never reached as a hit target (three cards can't total a two-card 21), so the Hit DP
            // below never looks it up.
            if pl_hand.is_nat21() {
                let nat_ev = dealer_outcome_probs(dealer_hand, &shoe_minus_hand, rules.hs17)
                    .iter()
                    .map(|(&dealer, p)| {
                        p * resolve_ev(HandState::from(&pl_hand), dealer, rules.bj_payout)
                    })
                    .sum::<f64>();
                let ev_map = HashMap::from_iter([(Move::Stand, nat_ev)]);
                let ins_res = full_ev_tree.insert(pl_hand, (weight, ev_map));
                assert!(ins_res.is_none());
                continue;
            }

            // Dealer outcome distribution against this hand, conditioned on a clean peek when the
            // peek applies. On the conditional basis there is no `Natural` outcome at all (it was
            // conditioned out), so every move below resolves naturally with no peek special-casing.
            let dealer_probs = basis.dealer_dist(&shoe_minus_hand);
            let stand_ev = dealer_probs
                .iter()
                .map(|(&dealer, p)| {
                    p * resolve_ev(HandState::from(&pl_hand), dealer, rules.bj_payout)
                })
                .sum::<f64>();

            // The player's next-card distribution, conditioned to match the dealer distribution.
            let draw_probs: Vec<(Card, f64)> = basis.draw_probs(&shoe_minus_hand);

            // Hitting draws a card then plays on optimally: the child's *best* move EV. The child
            // is already in the tree (it totals more, computed on an earlier, higher `pl_tot`).
            // Taking the max over the child's already-conditioned move EVs keeps the continuation
            // non-clairvoyant about the hole.
            let hit_ev = draw_probs
                .iter()
                .map(|&(c, p_c)| {
                    let mut pl_hand_hit = pl_hand;
                    pl_hand_hit.insert(c);
                    p_c * match full_ev_tree.get(&pl_hand_hit) {
                        Some((_w, ev_map)) => *ev_map
                            .values()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        None => {
                            assert!(HandState::from(&pl_hand_hit) == HandState::Bust);
                            -1.
                        }
                    }
                })
                .sum::<f64>();
            let mut evs = vec![(Move::Stand, stand_ev), (Move::Hit, hit_ev)];
            // If this is a starting hand (i.e. length two) then we may also have the option to
            // double down, split, or surrender.
            if pl_hand.len() == 2 {
                // Doubling draws exactly one card then stands. The child's stored `Stand` EV already
                // resolves against the dealer distribution with that card removed and on this same
                // (conditional or unconditional) basis, so the doubled payoff is just twice it — no
                // separate dealer recomputation, and no peek special case. On the conditional basis
                // a dealer natural can't occur; on the no-peek basis the child's `Stand` EV already
                // carries -1 per natural, so 2x correctly forfeits the whole doubled bet to it.
                // (Doubling is assumed start-only; allowing it mid-hand would inflate the Hit DP.)
                let double_ev = draw_probs
                    .iter()
                    .map(|&(c, p_c)| {
                        let mut pl_hand_dd = pl_hand;
                        pl_hand_dd.insert(c);
                        let child_stand = match full_ev_tree.get(&pl_hand_dd) {
                            Some((_w, ev_map)) => ev_map[&Move::Stand],
                            None => {
                                assert!(HandState::from(&pl_hand_dd) == HandState::Bust);
                                -1.
                            }
                        };
                        p_c * 2.0 * child_stand
                    })
                    .sum::<f64>();
                evs.push((Move::Double, double_ev));

                // Surrender forfeits half the bet for a flat -0.5 on whichever basis this tree is
                // on: late surrender happens after a clean peek (the conditional basis already
                // excludes the dealer natural), and early surrender escapes the natural before the
                // peek (an unconditional -0.5). NOTE: early surrender *combined with* a peek is the
                // one ragged case — that decision is genuinely pre-peek and would need an
                // unconditional root comparison; it is uncommon and left for later.
                let surrender_ev = match rules.surrender {
                    SurrenderRule::None => None,
                    SurrenderRule::Early | SurrenderRule::Late => Some(-0.5),
                };
                if let Some(surrender_ev) = surrender_ev {
                    evs.push((Move::Surrender, surrender_ev));
                }

                // A pair may be split (creating at least two hands, so only when the rules allow at
                // least two). The arms are played out exactly from the shared depleting shoe; see
                // `split_move_ev`. On the conditional basis this is computed by the same hole
                // stratification as the other moves, so all the EVs in this map stay comparable.
                if rules.max_split_hands >= 2 && pair_rank(&pl_hand).is_some() {
                    let split_ev = split_move_ev(&pl_hand, &shoe_minus_hand, basis, rules);
                    evs.push((Move::Split, split_ev));
                }
            }
            // On the conditional basis the pooling weight becomes the conditional occurrence
            // probability P(hold hand | no dealer natural) ∝ scan-weight · (1 - P_bj).
            let stored_weight = if conditional {
                weight * (1.0 - dealer_natural_prob(up_card, &shoe_minus_hand))
            } else {
                weight
            };
            let ev_map = HashMap::from_iter(evs);
            let ins_res = full_ev_tree.insert(pl_hand, (stored_weight, ev_map));
            assert!(ins_res.is_none());
        }
        // dbg!(pl_tot);
    }
    full_ev_tree
}

/// The rank of a two-card pair, if the hand is one (exactly two cards of the same rank).
pub(crate) fn pair_rank(hand: &CardCol) -> Option<Card> {
    if hand.len() != 2 {
        return None;
    }
    hand.iter().find(|&(_, n)| n == 2).map(|(c, _)| c)
}

/// Route a concrete hand to its strategy-table row (see [`HandCategory`]). Pairs take priority over
/// the hard/soft total they also form; everything else defers to [`HandState`].
fn categorize(hand: &CardCol) -> HandCategory {
    if let Some(rank) = pair_rank(hand) {
        return HandCategory::Pair(rank);
    }
    match HandState::from(hand) {
        HandState::Natural => HandCategory::Natural,
        HandState::Soft(n) => HandCategory::Soft(n),
        HandState::Hard(n) => HandCategory::Hard(n),
        // The tree only holds hands totalling at most 21, so a stored hand is never bust.
        HandState::Bust => unreachable!("a stored hand totals at most 21, so it is never bust"),
    }
}

/// Collapse the per-exact-hand EV tree into one move→EV map per strategy-table [`HandCategory`],
/// pooling every composition (and size) of a category by a weighted average.
///
/// The pooling weight is the tree's combinatorial scan-weight. Within a fixed hand size it is the
/// exact occurrence probability, but across sizes it is not a coherent distribution (see the
/// cross-size weighting note); it is the best stand-in available until a game-time
/// probability-of-hand is implemented, at which point only this weighting need change. A move only
/// contributes from the hands that actually offer it, so e.g. `Double`/`Surrender` for a hard total
/// reflect only its two-card members.
pub(crate) fn summarize_evs(
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
) -> HashMap<HandCategory, HashMap<Move, f64>> {
    // category -> move -> (Σ weight, Σ weight·ev), accumulated for a streaming weighted average.
    let mut acc = HashMap::<HandCategory, HashMap<Move, (f64, f64)>>::new();
    for (hand, (weight, move_ev)) in ev_tree.iter() {
        let moves = acc.entry(categorize(hand)).or_default();
        for (&mv, &ev) in move_ev.iter() {
            let (wt_sum, wt_ev_sum) = moves.entry(mv).or_insert((0.0, 0.0));
            *wt_sum += *weight;
            *wt_ev_sum += *weight * ev;
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

/// Reduce a per-category move→EV summary (from [`summarize_evs`]) to the single best move per row.
pub(crate) fn best_strategy(
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

fn main() {
    println!("Hello, world!");
    println!("{}, {}, {}", Card::Pip(5), Card::Ten, Card::Ace);
    assert!(Card::Pip(2) < Card::Pip(3));
    assert!(Card::Pip(6) < Card::Ten);
    assert!(Card::Pip(9) > Card::Ace);
    assert!(Card::Ten > Card::Ace);

    let dd = CardCol::from_decks(4);
    println!("{} - {} total", dd, dd.len());

    let target_total = 16;
    let parts: Vec<_> = dd.weighted_partitions(target_total).collect();
    // println!("{:#?}", &parts);
    for (weight, hand) in parts.iter() {
        assert!(
            hand.iter()
                .map(|(c, n)| n as usize * c.hard() as usize)
                .sum::<usize>()
                == target_total.into()
        );
        println!("{}:\t{}", weight, hand);
    }
    println!("{} total partitions", parts.len());

    let rules = Ruleset::default();
    let dd = CardCol::from_decks(2);
    // `dealer_outcome_probs` is now always the raw (natural-included) distribution; apply
    // `remove_nat21` to show the peek-conditioned version side by side.
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), &dd, rules.hs17);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", &base_deal_probs, norm);
    println!("{:?}\nnorm: {}", remove_nat21(base_deal_probs), norm);

    let dd = InfiniteDeck {};
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), &dd, rules.hs17);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);

    // NOTE: See https://wizardofodds.com/games/blackjack/appendix/9/1dh17r4/ for precise
    // comparisons
    let dd = CardCol::from_decks(2);
    // let dd = CardCol::half_deck();
    // let ev_map = build_evs(dd, Card::Pip(5), &rules);
    // let ev_map = build_evs(dd, Card::Pip(5), &rules);
    // let ev_map = build_evs(dd, Card::Pip(9), &rules);
    let ev_map = build_evs(dd, Card::Pip(5), &rules);
    let test_hand = CardCol::try_from("9A").unwrap();
    let soft20 = &ev_map[&test_hand];
    dbg!(soft20);

    let summary = summarize_evs(&ev_map);
    let strat = best_strategy(&summary);
    let mut sorted_strat: Vec<_> = strat.into_iter().collect();
    sorted_strat.sort_by_key(|(cat, _m)| *cat);
    for (cat, strat) in sorted_strat.into_iter() {
        println!("{}: {}", cat, strat);
    }
}

#[cfg(test)]
mod tests {
    //! Regression guard pinning the verified general (non-split) solver output. Split-specific tests
    //! — the `SplitSolver` budget modes and the pair-cell chart decisions — live in [`crate::split`].
    //!
    //! All numbers are for a **2-deck shoe under [`Ruleset::default`]** (H17, DAS, dealer peeks,
    //! late surrender) — the configuration every reference value in the design notes was captured
    //! under. EV magnitudes are checked to a tolerance, not bit-exactly: `stand`/`hit` EVs sum over
    //! `HashMap` iteration, whose order is randomized per run, so float non-associativity can move
    //! the last bit. The argmax (chart) cells are robust to that because the competing EVs are not
    //! within ~1e-12 of each other for these decided cells. Reference strategy:
    //! <https://wizardofodds.com/games/blackjack/appendix/9/2dh17r4/>.
    use super::*;
    use crate::test_support::*;

    /// The per-move EVs of soft 20 (9,A) vs a 5 — the unconditional path (P_bj = 0), so a clean
    /// control. These exact magnitudes were the cross-check anchor through the basis redesign.
    #[test]
    fn soft20_vs_5_move_evs() {
        let tree = ev_tree(Card::Pip(5));
        let (_w, evs) = &tree[&CardCol::try_from("9A").unwrap()];
        assert_close(evs[&Move::Stand], 0.674_582_770_421_14, "stand");
        assert_close(evs[&Move::Hit], 0.261_623_004_258_03, "hit");
        assert_close(evs[&Move::Double], 0.523_246_008_516_06, "double");
        assert_close(evs[&Move::Surrender], -0.5, "surrender");
    }

    /// The blackjack payout is a rule parameter. Against a 5 the dealer can't have a natural, so a
    /// player natural always pays out: its Stand EV equals `bj_payout` exactly (3:2 by default, and
    /// 6:5 when configured). This pins the new `Ruleset::bj_payout` axis through the shared resolver.
    #[test]
    fn bj_payout_rule_axis() {
        let nat = CardCol::try_from("TA").unwrap();

        let tree_32 = build_evs(CardCol::from_decks(2), Card::Pip(5), &ruleset_with(0));
        assert_close(tree_32[&nat].1[&Move::Stand], 1.5, "3:2 natural vs 5");

        let six_five = Ruleset {
            bj_payout: 1.2,
            split_cards: 0,
            ..Ruleset::default()
        };
        let tree_65 = build_evs(CardCol::from_decks(2), Card::Pip(5), &six_five);
        assert_close(tree_65[&nat].1[&Move::Stand], 1.2, "6:5 natural vs 5");
    }

    /// Late-surrender cells (hard hands) against a ten and an ace up-card — the peek-conditional
    /// path. The H17-specific H17-vs-A surrender is the H17 tell. Pair cells are checked separately
    /// in `split_decisions_under_peek`, since with split scored 8,8 is split (not surrendered) here.
    #[test]
    fn late_surrender_hard_cells() {
        let vs_ten = strategy_for(Card::Ten);
        assert_eq!(vs_ten[&HandCategory::Hard(15)], Move::Surrender, "H15 vs T");
        assert_eq!(vs_ten[&HandCategory::Hard(16)], Move::Surrender, "H16 vs T");

        let vs_ace = strategy_for(Card::Ace);
        assert_eq!(vs_ace[&HandCategory::Hard(15)], Move::Surrender, "H15 vs A");
        assert_eq!(vs_ace[&HandCategory::Hard(16)], Move::Surrender, "H16 vs A");
        assert_eq!(vs_ace[&HandCategory::Hard(17)], Move::Surrender, "H17 vs A");
    }

    /// A couple of uncontroversial basic-strategy anchors on the unconditional path, so the guard
    /// also covers ordinary hit/stand/double argmax, not just the surrender corners.
    #[test]
    fn basic_strategy_anchors_vs_5() {
        let strat = strategy_for(Card::Pip(5));
        assert_eq!(strat[&HandCategory::Hard(16)], Move::Stand, "H16 vs 5");
        assert_eq!(strat[&HandCategory::Hard(11)], Move::Double, "H11 vs 5");
        assert_eq!(strat[&HandCategory::Hard(8)], Move::Hit, "H8 vs 5");
        assert_eq!(strat[&HandCategory::Soft(18)], Move::Double, "S18 vs 5");
    }

}
