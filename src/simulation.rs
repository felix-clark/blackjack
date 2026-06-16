//! The solver engine: the exact-enumeration EV computation over a shoe.
//!
//! [`build_evs`] is the main driver — a dynamic program over the partition lattice that produces the
//! per-exact-hand move→EV tree for one up-card. [`summarize_evs`] collapses that
//! tree into the per-category move→EV summary behind the strategy chart. [`Basis`] bundles the dealer-outcome and player-draw distributions
//! (and the peek conditioning) shared with the split solver ([`crate::split`]); [`resolve_ev`] is the
//! terminal payoff table. The hand/move vocabulary lives in [`crate::hand`], rule knobs in
//! [`crate::rules`].

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::dealer::{DealerOutcome, dealer_outcome_probs};
use crate::hand::{HandCategory, HandState, Move, best_move, categorize};
use crate::rules::Ruleset;
use crate::shoe::{CardCol, Shoe};
use crate::split::split_move_ev;

/// Terminal payoff of a standing/resolved player hand against one dealer outcome, as a multiple of
/// the bet. Keyed on the collapsed [`HandState`] so callers control natural-eligibility: only a
/// hand presented as [`HandState::Natural`] earns `bj_payout` (and pushes a dealer natural) — a
/// split-arm 21 is presented as an ordinary total and so loses to a dealer natural and pays even
/// money (see [`arm_stand_ev`](crate::split::arm_stand_ev)). `bj_payout` is the table's blackjack
/// payout (3:2 = 1.5, 6:5 = 1.2).
pub(crate) fn resolve_ev(
    player_state: HandState,
    dealer_state: DealerOutcome,
    bj_payout: f64,
) -> f64 {
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
pub(crate) fn dealer_natural_prob(up_card: Card, shoe: &impl Shoe) -> f64 {
    match up_card {
        Card::Ace => shoe.draw_prob(&Card::Ten),
        Card::Ten => shoe.draw_prob(&Card::Ace),
        _ => 0.0,
    }
}

/// EV of taking insurance: a 2:1 side bet that the dealer's hole card completes a natural. `shoe` must
/// be the shoe *with the up-card already removed*, so [`dealer_natural_prob`] reads the hole-card
/// distribution the player faces. Win pays `+2` with probability `P_bj`, lose `-1` otherwise:
/// `2·P_bj − (1 − P_bj) = 3·P_bj − 1`.
pub(crate) fn insurance_ev(up_card: Card, shoe_after_up: &impl Shoe) -> f64 {
    3.0 * dealer_natural_prob(up_card, shoe_after_up) - 1.0
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
/// This is more than dropping the natural and renormalising ([`remove_nat21`](crate::legacy::remove_nat21)):
/// removing the concrete hole before the dealer's later draws is what makes it exact on a finite shoe.
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
/// partition DP) and [`SplitSolver`](crate::split) (its top-down arm recursion) are deliberately
/// *different* traversals — one produces the whole weighted tree, the other a single arm's value with
/// a re-split budget — but they ask the deck the *same* two questions, conditioned the *same* way.
/// Centralising that here keeps the subtle peek-conditioning (on which the affine-collapse property
/// depends) in one place instead of three. When `conditional` (the dealer peeks and the up-card can
/// make a natural) both distributions are the marginal, hole-averaged ones conditioned on no dealer
/// natural ([`conditional_dealer_dist`] / [`conditional_draw_probs`]); otherwise they are the plain
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
            conditional: rules.peek.peeks() && bj_rank.is_some(),
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
pub(crate) fn build_evs<S: Shoe + Clone + Eq + Hash + Sync>(
    shoe: S,
    up_card: Card,
    rules: &Ruleset,
) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
    // The split solves dominate runtime by far (e.g. on a 6-deck shoe the whole non-split DP is
    // ~0.6s while the pair splits sum to ~34s) and each is independent of the DP tree, so they are
    // computed up front across all cores; the DP below just looks the result up. This is the engine's
    // parallelism — what lets a single column use more than one core (one split runs per pair, so the
    // per-level DP loop could never parallelise it). See [`pair_split_evs_for`].
    let split_evs = pair_split_evs(&shoe, up_card, rules);
    build_evs_with_splits(shoe, up_card, rules, &split_evs)
}

/// The [`build_evs`] dynamic program given *precomputed* split EVs — the cheap (~2%) half of the
/// solve, factored out so a caller can supply splits solved elsewhere. The count-frame solve uses
/// this: it solves each pair's split once (in the frame matching the pair's own count value, via
/// [`pair_split_evs_for`]) and then runs this DP per frame, instead of re-solving every split in
/// every frame. `split_evs` need only cover the pairs whose `Move::Split` this tree will actually be
/// read for; a pair absent from the map simply offers no `Split` move.
pub(crate) fn build_evs_with_splits<S: Shoe + Clone + Eq + Hash>(
    mut shoe: S,
    up_card: Card,
    rules: &Ruleset,
    split_evs: &HashMap<CardCol, f64>,
) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
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

    // Go down to 2 to get all soft options as well. The 21→2 order is the DP dependency: a hit/double
    // child totals strictly more, so it is already in the tree when looked up.
    for pl_tot in (2..=21).rev() {
        for (weight, pl_hand) in shoe.weighted_partitions(pl_tot) {
            if pl_hand.len() < 2 {
                continue;
            }
            let (pl_hand, entry) = solve_hand(
                weight,
                pl_hand,
                &shoe,
                &full_ev_tree,
                basis,
                rules,
                dealer_hand,
                up_card,
                conditional,
                split_evs,
            );
            let ins_res = full_ev_tree.insert(pl_hand, entry);
            assert!(ins_res.is_none());
        }
    }
    full_ev_tree
}

/// The splittable pairs the shoe holds two of, one `(r, r)` per rank — the work items
/// [`pair_split_evs_for`] fans out over. Call on the shoe *after* the up-card is removed (matching
/// where the splits are actually played from).
pub(crate) fn splittable_pairs<S: Shoe>(shoe: &S) -> Vec<CardCol> {
    Card::ALL
        .iter()
        .map(|&r| CardCol::from_hand(&[r, r]))
        .filter(|pair| shoe.contains_hand(pair))
        .collect()
}

/// Solve `pairs`' split EVs in parallel — the engine's one expensive, embarrassingly-parallel phase.
/// Each [`split_move_ev`] is self-contained (it spins up its own arm recursion and reads nothing from
/// the main DP tree), so they fan out cleanly across cores via [`par_map`]. `shoe_for(&pair)` yields
/// the shoe that pair's arms are played from, *before* the up-card is removed (this function removes
/// it). That per-pair shoe is the seam the count-frame solve uses: under a count condition each pair
/// is solved at the frame whose entered running count matches the pair's own count value, so its
/// split is conditioned on exactly the count the player holds — and each pair is solved once rather
/// than in every frame. `pairs` must already be filtered to those the shoe holds (see
/// [`splittable_pairs`]). Returns `{pair → split EV}`, empty when splitting is disabled.
pub(crate) fn pair_split_evs_for<S: Shoe + Clone + Eq + Hash + Sync>(
    pairs: &[CardCol],
    up_card: Card,
    rules: &Ruleset,
    shoe_for: impl Fn(&CardCol) -> S + Sync,
) -> HashMap<CardCol, f64> {
    if rules.max_split_hands < 2 {
        return HashMap::new();
    }
    let basis = Basis::new(up_card, rules);
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    par_map(pairs, n_threads, |pair| {
        let mut shoe = shoe_for(pair);
        shoe.draw(&up_card);
        let shoe_minus_hand = shoe.remove_hand(pair);
        (*pair, split_move_ev(pair, &shoe_minus_hand, basis, rules))
    })
    .into_iter()
    .collect()
}

/// Solve every splittable pair's split EV on one shoe (the uncounted single-solve path). Thin wrapper
/// over [`pair_split_evs_for`] that filters the pairs the shoe holds and routes them all to that one
/// shoe. Returns `{pair → split EV}`, empty when splitting is disabled.
fn pair_split_evs<S: Shoe + Clone + Eq + Hash + Sync>(
    shoe: &S,
    up_card: Card,
    rules: &Ruleset,
) -> HashMap<CardCol, f64> {
    if rules.max_split_hands < 2 {
        return HashMap::new();
    }
    let mut after_up = shoe.clone();
    after_up.draw(&up_card);
    let pairs = splittable_pairs(&after_up);
    pair_split_evs_for(&pairs, up_card, rules, |_| shoe.clone())
}

/// Compute the move→EV map for one concrete player hand against the dealer up-card, reading
/// already-solved (higher-total) children out of `full_ev_tree`. Pure and read-only w.r.t. the tree
/// (and everything else), which is what makes the per-level [`par_map`] safe and exact. Returns the
/// hand and its `(pooling weight, move→EV)` entry, ready to insert. Factored out of [`build_evs`]'s
/// level loop verbatim — see there for the running commentary on each branch.
#[allow(clippy::too_many_arguments)]
fn solve_hand<S: Shoe + Clone + Eq + Hash>(
    weight: f64,
    pl_hand: CardCol,
    shoe: &S,
    full_ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    basis: Basis,
    rules: &Ruleset,
    dealer_hand: CardCol,
    up_card: Card,
    conditional: bool,
    split_evs: &HashMap<CardCol, f64>,
) -> (CardCol, (f64, HashMap<Move, f64>)) {
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
                p * resolve_ev(
                    HandState::from(&pl_hand),
                    dealer,
                    rules.bj_payout.multiplier(),
                )
            })
            .sum::<f64>();
        let ev_map = HashMap::from_iter([(Move::Stand, nat_ev)]);
        // Unconditional occurrence probability (the natural resolves at the peek, so it keeps its
        // full weight — no `1 - P_bj` reduction). Via `hand_prob` so a count-tilted shoe weights its
        // (frequent, ten/ace-rich) naturals correctly rather than by the untilted scan-weight; this
        // is the dominant favourable mass in a high-count shoe. Finite/infinite shoes are unchanged.
        return (pl_hand, (shoe.hand_prob(&pl_hand), ev_map));
    }

    // Dealer outcome distribution against this hand, conditioned on a clean peek when the
    // peek applies. On the conditional basis there is no `Natural` outcome at all (it was
    // conditioned out), so every move below resolves naturally with no peek special-casing.
    let dealer_probs = basis.dealer_dist(&shoe_minus_hand);
    let stand_ev = dealer_probs
        .iter()
        .map(|(&dealer, p)| {
            p * resolve_ev(
                HandState::from(&pl_hand),
                dealer,
                rules.bj_payout.multiplier(),
            )
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
        if rules.peek.surrender_offered() {
            evs.push((Move::Surrender, -0.5));
        }

        // A pair may be split (creating at least two hands, so only when the rules allow at
        // least two). The arms are played out exactly from the shared depleting shoe; that solve is
        // done up front in parallel (see `pair_split_evs`), so here we just look it up. On the
        // conditional basis it was computed by the same hole stratification as the other moves, so
        // all the EVs in this map stay comparable.
        if let Some(&split_ev) = split_evs.get(&pl_hand) {
            evs.push((Move::Split, split_ev));
        }
    }
    // The hand's game-time occurrence probability. The partition scan-weight (`weight`) is the
    // *untilted* hypergeometric for a count-conditioned shoe, so a two-card root — the only weight
    // that escapes to the edge/reach integrators — takes its occurrence probability from
    // `Shoe::hand_prob` instead, which carries the count tilt. (For finite/infinite shoes the two
    // coincide, so this is a no-op there.) Deeper hands keep the scan-weight pooling measure: they
    // feed only the combinatorial `summarize_evs` baseline, never the count-conditioned live path.
    let occurrence_weight = if pl_hand.len() == 2 {
        shoe.hand_prob(&pl_hand)
    } else {
        weight
    };
    // On the conditional basis the pooling weight becomes the conditional occurrence
    // probability P(hold hand | no dealer natural) ∝ occurrence-weight · (1 - P_bj).
    let stored_weight = if conditional {
        occurrence_weight * (1.0 - dealer_natural_prob(up_card, &shoe_minus_hand))
    } else {
        occurrence_weight
    };
    let ev_map = HashMap::from_iter(evs);
    (pl_hand, (stored_weight, ev_map))
}

/// Map `f` over `items` across up to `n_threads` scoped worker threads, returning the results in an
/// unspecified order. Work is handed out one item at a time via an atomic cursor, so the very uneven
/// split solves are absorbed by whichever worker is free rather than stalling a fixed shard. Each item
/// here is a whole pair's split solve (seconds of work), so the parallel path is worth taking for as
/// few as two; below that, or single-threaded, it falls back to a plain serial map.
fn par_map<T: Sync, R: Send>(items: &[T], n_threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if n_threads <= 1 || items.len() < 2 {
        return items.iter().map(&f).collect();
    }
    let cursor = AtomicUsize::new(0);
    let n = n_threads.min(items.len());
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..n)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        local.push(f(&items[i]));
                    }
                    local
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    })
}

/// Collapse the per-exact-hand EV tree into one move→EV map per strategy-table [`HandCategory`],
/// pooling every composition (and size) of a category by a weighted average.
///
/// The pooling weight is the tree's combinatorial scan-weight. Within a fixed hand size it is the
/// exact occurrence probability, but across sizes it is not a coherent distribution (see the
/// cross-size weighting note). The **live chart now pools by the game-time reaching weight instead**
/// ([`crate::reach::reach_weights`] → [`crate::reach::summarize_with`], wired in `tui::solve_on`);
/// this combinatorial version is retained as the reference baseline the regression tests pin against
/// (and is `summarize_with` with the scan-weight). A move only contributes from the hands that
/// actually offer it, so e.g. `Double`/`Surrender` for a hard total reflect only its two-card members.
#[cfg_attr(not(test), allow(dead_code))]
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

/// One up-card's two-card-root contribution to the overall player edge, read straight off a
/// [`build_evs`] tree. Keeping it separate from any whole-shoe edge pass lets the TUI accumulate
/// these from the per-up-card trees it already computes for the chart, instead of a second solver pass.
///
/// `weighted_ev` is `Σ weight·max_move_EV` over the *starting* (two-card) hands — the optimal-play
/// value on the tree's basis. `weight` is `Σ weight` over those same hands; it is `1` off-peek but
/// falls short of it on the peek basis by exactly the dealer-natural loss mass (each such hand a flat
/// −1). So the honest unconditional value of this up-card is `weighted_ev − (1 − weight)`: the
/// `−P_bj + (1 − P_bj)·V` two-card-root identity from [`build_evs`], summed over starting hands.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct EdgeTerm {
    pub(crate) weighted_ev: f64,
    pub(crate) weight: f64,
}

impl EdgeTerm {
    /// The unconditional per-up-card edge: the conditional play value with the dealer-natural loss
    /// deficit (`1 − weight`) added back as a flat −1 per non-natural-vs-dealer-natural hand.
    pub(crate) fn value(&self) -> f64 {
        self.weighted_ev - (1.0 - self.weight)
    }
}

/// Accumulate one up-card's two-card-root edge contribution from its EV tree. Only starting
/// (two-card) hands count: their stored weight is the hand's true occurrence probability
/// ([`Shoe::hand_prob`], folded with the peek's `1 − P_bj`), an exact distribution summing to 1
/// off-peek — so this sidesteps the cross-size weighting imprecision that [`summarize_evs`] carries.
/// Crucially that occurrence weight carries the count tilt (the partition scan-weight would not), so
/// the edge stays correct under a count-conditioned shoe. A natural's lone `Stand` EV is its `max`,
/// so naturals fold in correctly.
///
/// [`Shoe::hand_prob`]: crate::shoe::Shoe::hand_prob
pub(crate) fn edge_term(ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>) -> EdgeTerm {
    let mut weighted_ev = 0.0;
    let mut weight = 0.0;
    for (hand, (w, move_ev)) in ev_tree.iter() {
        if hand.len() != 2 {
            continue;
        }
        let best = move_ev.values().copied().fold(f64::NEG_INFINITY, f64::max);
        weight += *w;
        weighted_ev += *w * best;
    }
    EdgeTerm {
        weighted_ev,
        weight,
    }
}

/// The strategy table's recommended move at a *specific node*: the highest-`table`-ranked move that is
/// actually legal here (`legal` = the node's stored move→EV map, whose keys are exactly its available
/// moves). This is what makes the fallback faithful to a printed chart rather than to optimal play:
/// when a start-only move (Surrender/Double) heads the cell but the hand is now multi-card, we drop to
/// the *table's* next preference among `Hit`/`Stand` — e.g. Hard 17 vs Ace, charted Surrender, falls to
/// **Stand** (the table's runner-up), never to Hit, even though Hit might edge it on some composition.
/// Falls back to the node's own argmax only if the table carries no legal move for it (degenerate).
fn table_move(table_evs: Option<&HashMap<Move, f64>>, legal: &HashMap<Move, f64>) -> Move {
    if let Some(t) = table_evs {
        let best = legal
            .keys()
            .filter_map(|m| t.get(m).map(|&v| (*m, v)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((m, _)) = best {
            return m;
        }
    }
    best_move(legal)
}

/// The **basic-strategy value** of every hand in `ev_tree`: the EV of playing the strategy `table` at
/// *every* decision point, not merely the first. `table` maps each [`HandCategory`] to its consolidated
/// per-move EVs (the chart cell's `move_evs`); [`table_move`] turns that into the recommendation legal
/// at each node. The recommendation's value is read straight off the tree for the terminal moves
/// (`Stand`/`Double`/`Surrender`/`Split` — each already a complete continuation), but a **`Hit` recurses
/// into this same BS value** of its children rather than the optimal Hit EV `build_evs` stored. That one
/// branch is the whole difference from [`edge_term`]'s optimal play: under the table you keep consulting
/// the table after every hit, so a multi-card total is played by *its* row, not re-optimised.
///
/// Hands are visited in descending total so every `Hit` child (a strictly larger total) has its BS value
/// ready — the same dependency order as [`build_evs`]'s `21→2` sweep. `shoe` is the full shoe with the
/// up-card still present (this removes it), matching `build_evs`/[`reach_weights`].
///
/// *Split caveat (documented):* a charted `Split` reads the split solver's stored EV, whose arms are
/// played optimally, not table-recursively — the one place the "table at every node" rule is relaxed,
/// inherited from the independent-arms split model the chart already uses. Measured (2026-06, all
/// up-cards × charted-split pairs, occurrence-weighted): playing the arms by basic strategy instead
/// would lower the overall edge by only ~+0.0002% on a 6-deck shoe (≈8% of the ~0.002% optimal-vs-BS
/// gap there, invisible at the footer's 3 decimals), rising to ~+0.0016% single-deck — a pure
/// finite-shoe composition effect that vanishes as decks grow (at 6 decks all but 2,2 and 8,8 are
/// exactly zero). Below the independent-arms approximation already in every split EV, so not worth a
/// second strategy-driven split pass.
pub(crate) fn bs_value_tree<S: Shoe + Clone + Eq + Hash>(
    mut shoe: S,
    up_card: Card,
    rules: &Ruleset,
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    table: &HashMap<HandCategory, HashMap<Move, f64>>,
) -> HashMap<CardCol, f64> {
    shoe.draw(&up_card);
    let shoe = shoe;
    let basis = Basis::new(up_card, rules);

    // Children (larger total) before parents: a Hit grows the total by ≥1, so descending-total order
    // guarantees every Hit child's BS value is computed before the hand that hits into it.
    let mut hands: Vec<&CardCol> = ev_tree.keys().collect();
    hands.sort_by_key(|h| std::cmp::Reverse(h.hard_count()));

    // Fold over descending-total order, accumulating into the BS-value map: each hand's Hit value
    // reads its children's already-folded values, so the accumulator must be threaded, not collected.
    hands.into_iter().fold(
        HashMap::with_capacity(ev_tree.len()),
        |mut bs, hand| {
            let move_ev = &ev_tree[hand].1;
            let mv = table_move(table.get(&categorize(hand)), move_ev);
            let value = if mv == Move::Hit {
                // Replay the hit on the table's continuation: the child's BS value, not its optimal one.
                let shoe_here = shoe.remove_hand(hand);
                basis
                    .draw_probs(&shoe_here)
                    .iter()
                    .map(|&(c, p_c)| {
                        let mut child = *hand;
                        child.insert(c);
                        p_c * bs.get(&child).copied().unwrap_or_else(|| {
                            assert!(HandState::from(&child) == HandState::Bust);
                            -1.0
                        })
                    })
                    .sum()
            } else {
                // Terminal moves carry a complete continuation already — read the stored EV.
                move_ev[&mv]
            };
            bs.insert(*hand, value);
            bs
        },
    )
}

/// Like [`edge_term`], but reading each starting hand's value from a [`bs_value_tree`] — i.e. the
/// two-card-root edge under table play at *every* decision rather than the per-composition optimum.
/// Same weighting and dealer-natural-deficit bookkeeping as [`edge_term`] (so the two are directly
/// comparable); the gap between them is the EV cost of following the chart. `bs_values` must come from
/// [`bs_value_tree`] on this same `ev_tree`.
pub(crate) fn bs_edge_term(
    ev_tree: &HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    bs_values: &HashMap<CardCol, f64>,
) -> EdgeTerm {
    let mut weighted_ev = 0.0;
    let mut weight = 0.0;
    for (hand, (w, _)) in ev_tree.iter() {
        if hand.len() != 2 {
            continue;
        }
        let ev = bs_values.get(hand).copied().unwrap_or(0.0);
        weight += *w;
        weighted_ev += *w * ev;
    }
    EdgeTerm {
        weighted_ev,
        weight,
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
    use crate::rules::BjPayout;
    use crate::test_support::*;

    /// The overall player edge (house edge, signed for the player so it is typically negative) under
    /// `rules`, assuming optimal composition-dependent play. Each up-card's [`build_evs`] tree gives a
    /// two-card-root [`EdgeTerm`]; they are averaged by the up-card's draw probability from the full
    /// shoe. Production code (the TUI) accumulates the per-up-card [`edge_term`]s incrementally from
    /// the chart trees it already computes, so this whole-shoe convenience pass exists only here.
    fn player_edge<S: Shoe + Clone + Eq + Hash + Sync>(shoe: S, rules: &Ruleset) -> f64 {
        shoe.all_draw_probs()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(up, p_up)| {
                let tree = build_evs(shoe.clone(), up, rules);
                p_up * edge_term(&tree).value()
            })
            .sum()
    }

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
            bj_payout: BjPayout::SixToFive,
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

    /// The basic-strategy value pass plays the *table* at every node, not just the first decision.
    /// Pins the user's canonical fallback: Hard 17 vs Ace charts as Surrender (a start-only move), so a
    /// *multi-card* 17 — which can't surrender — must fall to the table's runner-up **Stand**, never to
    /// Hit and never re-optimised. Also pins the headline invariant (a two-card root takes its charted
    /// move's EV) and that the resulting two-card-root edge never beats optimal play.
    #[test]
    fn bs_value_follows_table_at_every_node() {
        use crate::reach::{reach_weights, summarize_cells};
        let up = Card::Ace;
        let shoe = CardCol::from_decks(2);
        let rules = ruleset_with(0);
        let tree = build_evs(shoe, up, &rules);
        let reach = reach_weights(shoe, up, &rules, &tree, true);
        let summary = summarize_cells(&tree, &reach);
        let table: HashMap<HandCategory, HashMap<Move, f64>> = summary
            .iter()
            .map(|(c, cell)| (*c, cell.move_evs.clone()))
            .collect();
        let bs = bs_value_tree(shoe, up, &rules, &tree, &table);

        // Setup tell: the two-card H17-vs-A cell is charted Surrender (the H17 surrender corner).
        assert_eq!(summary[&HandCategory::Hard(17)].headline, Move::Surrender);

        // A three-card hard 17 can't surrender. The table's best *legal* move here is Stand (its
        // runner-up), so its BS value is the stored Stand EV — not Hit, not a re-optimised argmax.
        let multi17 = CardCol::from_hand(&[Card::Ten, Card::Pip(4), Card::Pip(3)]);
        assert_eq!(categorize(&multi17), HandCategory::Hard(17));
        let stand = tree[&multi17].1[&Move::Stand];
        assert_close(bs[&multi17], stand, "multi-card H17 falls back to table Stand");

        // The two-card root takes its charted (Surrender) value, and the edge never beats optimal.
        let nat_free = |hand: &CardCol| hand.len() == 2 && !hand.is_nat21();
        for (hand, (_, mv)) in tree.iter().filter(|(h, _)| nat_free(h)) {
            let charted = table_move(table.get(&categorize(hand)), mv);
            if charted != Move::Hit {
                assert_close(bs[hand], mv[&charted], "two-card root = charted move EV");
            }
            let opt = mv.values().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(bs[hand] <= opt + 1e-12, "BS value {} > optimal {opt}", bs[hand]);
        }

        assert!(
            bs_edge_term(&tree, &bs).value() <= edge_term(&tree).value() + 1e-12,
            "BS edge must not beat optimal edge"
        );
    }

    /// The overall player edge (negative = house advantage) under optimal play, full default rules
    /// (H17, DAS, late surrender, splits) on the infinite deck — fast and deterministic there since
    /// split arms are independent (no depletion). Lands in the canonical sub-percent house-edge band;
    /// the Ten/Ace columns exercise the dealer-natural deficit correction.
    #[test]
    fn player_edge_band() {
        use crate::shoe::InfiniteDeck;
        // split_cards is irrelevant on a non-depleting deck (no cross-arm correlation to track).
        let edge = player_edge(InfiniteDeck {}, &ruleset_with(0));
        assert_close(edge, -0.006_294_265_713_365_8, "infinite-deck edge");
    }

    /// Peek strictly dominates no-peek (ENHC) when every *other* rule is held fixed: off peek the
    /// player forfeits doubled and split bets to a natural revealed at the end, a loss no strategy
    /// adjustment can recover. (The intuition that "no peek helps the player" only appears to hold
    /// when toggling peek silently also upgrades the surrender rule — which the combined [`PeekRule`]
    /// axis now makes impossible.) Surrender is held at `None` on both sides so the comparison is
    /// purely the peek mechanic.
    #[test]
    fn peek_dominates_no_peek() {
        use crate::rules::{PeekRule, PeekSurrender};
        use crate::shoe::InfiniteDeck;
        let with_peek = |peek| Ruleset {
            peek,
            split_cards: 0,
            ..Ruleset::default()
        };
        let peek_edge = player_edge(
            InfiniteDeck {},
            &with_peek(PeekRule::Peek(PeekSurrender::None)),
        );
        let no_peek_edge = player_edge(
            InfiniteDeck {},
            &with_peek(PeekRule::NoPeek {
                early_surrender: false,
            }),
        );
        assert!(
            no_peek_edge < peek_edge,
            "no-peek edge {no_peek_edge} should be below peek edge {peek_edge}"
        );
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

    /// Locks the surrender-cell bug fix: single-deck Hard 15 vs Ten. The old all-sizes argmax charted
    /// this as Surrender by comparing an all-sizes Hit EV (dragged down by un-surrenderable multi-card
    /// 15s) against a two-card-only Surrender EV. Restricted to the two-card 15s where surrender is
    /// actually legal, the pooled Hit EV (~-0.498) beats Surrender's flat -0.5, so the corrected
    /// headline is Hit — matching wizardofodds. The flip is single-deck only (see the 2-deck test).
    #[test]
    fn single_deck_h15_vs_ten_hits() {
        let cells = cells_for(1, Card::Ten);
        let h15 = &cells[&HandCategory::Hard(15)];
        assert_eq!(h15.headline, Move::Hit, "single-deck H15 vs T should hit");
        assert!(
            h15.composition_dependent,
            "H15 vs T varies by composition (8,7 hits; T,5/9,6 surrender)"
        );
    }

    /// The composition-dependence flag is independent of the headline. On 2 decks Hard 15 vs Ten still
    /// *charts* as Surrender, yet the 8,7 fifteen prefers Hit, so the cell is genuinely
    /// composition-dependent and renders `R*` rather than a bare `R`.
    #[test]
    fn two_deck_h15_vs_ten_surrenders_but_flagged() {
        let cells = cells_for(2, Card::Ten);
        let h15 = &cells[&HandCategory::Hard(15)];
        assert_eq!(
            h15.headline,
            Move::Surrender,
            "2-deck H15 vs T still surrenders"
        );
        assert!(
            h15.composition_dependent,
            "2-deck H15 vs T is still composition-dependent"
        );
    }

    /// A cleanly decided, composition-uniform cell must NOT be flagged: hard 20 vs 6 always stands.
    #[test]
    fn uniform_cell_not_flagged() {
        let cells = cells_for(2, Card::Pip(6));
        let h20 = &cells[&HandCategory::Hard(20)];
        assert_eq!(h20.headline, Move::Stand);
        assert!(
            !h20.composition_dependent,
            "hard 20 vs 6 is uniformly stand"
        );
    }

    /// Baseline guard for the combinatorial [`summarize_evs`] (retained as the reference weighting
    /// behind the corrected game-time consolidation): its pooled Stand EV for hard 20 vs 5 is pinned,
    /// so a regression in the scan-weight pooling is caught even though the chart no longer argmaxes it.
    #[test]
    fn summarize_evs_baseline_h20_vs_5() {
        let summary = summarize_evs(&ev_tree(Card::Pip(5)));
        let stand = summary[&HandCategory::Hard(20)][&Move::Stand];
        assert_close(
            stand,
            0.675_691_276_588_821_4,
            "combinatoric H20 vs 5 stand",
        );
    }
}
