mod card;
use card::*;

use counter::Counter;
use std::{
    collections::HashMap,
    default::Default,
    fmt::{Debug, Display},
};

#[derive(PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Clone, Copy)]
enum HandState {
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
enum DealerOutcome {
    Bust, // could also be represented as a zero?
    Total(u8),
    Natural,
}

impl Debug for DealerOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Display for DealerOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DealerOutcome::Bust => write!(f, "Bust"),
            DealerOutcome::Total(n) => write!(f, "{}", n),
            DealerOutcome::Natural => write!(f, "Nat"),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
enum Move {
    Hit,
    Stand,
    Double,
    Split,
    Surrender,
}

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

/// The stipulation of miscellaneous rules other than the number of decks (?).
struct Ruleset {
    /// Whether the dealer hits soft 17
    hs17: bool,
    /// Allowed to double after split
    das: bool,
    /// Whether the dealer checks their hole card for blackjack
    /// Note that the worst version of this being false causes a dealer blackjack to take
    /// all splits and doubles.
    dealer_check: bool,
    // /// Double on anything (as opposed to just 10 and 11) -- maybe just assume true
    // doa: bool,
    // /// Whether surrender is allowed. There are 2 variants, early and late - how to encode this?
    // surrender: bool,
    // TODO: only allowed 1 card after splitting aces? Only allowed to split aces once?
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            hs17: true,
            das: true,
            dealer_check: true,
        }
    }
}

fn dealer_hit(hand: &CardCol, hs17: bool) -> bool {
    // let hard_count: u8 = hand.iter().map(Card::hard).sum();
    let hard_count: u8 = hand.hard_count();
    if hard_count >= 17 {
        return false;
    }
    // let has_ace: bool = hand.iter().any(|&c| c == Card::Ace);
    let has_ace: bool = hand.has_ace();
    if has_ace && hard_count <= 11 {
        let soft_target = if hs17 { 18 } else { 17 };
        if hard_count + 10 >= soft_target {
            return false;
        }
    }
    true
}

/// TODO: Implement this as an iterator to avoid allocations from collections.
/// The weights should be the multivariate hypergeometric probability density. This gives the
/// probability conditioned on the size of the hand; it takes some additional assumptions to infer
/// a marginal distribution.
fn _weighted_partitions(
    mut deck: Counter<Card>,
    hard_total: u8,
    norm_offset: usize,
) -> Vec<(f64, Counter<Card>)> {
    // ) -> impl Iterator<Item = (usize, Counter<Card>)> {

    // TODO: double-check this condition
    // NOTE: This order seemed necessary to get all partitions, but there is an awkwardness with
    // the weights
    if hard_total == 0 {
        // with_capacity?
        return vec![(1., Counter::new())];
        // return std::iter::once((1, Counter::new()));
    }
    if deck.total::<usize>() == 0 {
        return Vec::new();
        // return std::iter::empty::<(usize, Counter<Card>)>();
    }
    // TODO: Should this start with 1. / shoe.total() ?
    // It matters whether this is set before or after the draw
    // let mut weight = 1. / deck.total::<usize>() as f64;
    // let mut weight = 1. / (deck.total::<usize>() - 1) as f64;
    let n_deck = deck.total::<usize>() as f64;
    let mut weight = 1.;

    // NOTE: This can probably be any rank. Keep it now just in case, and verify the results later.
    let top_rank: Card = *deck.keys().map(|c| (c.hard(), c)).max().unwrap().1;
    let n_top = deck.remove(&top_rank).expect("Should be in there");

    let mut k_perms: Vec<Vec<(f64, Counter<Card>)>> = Vec::new();
    for k_top in 0..=n_top {
        let top_cont = top_rank.hard() * k_top as u8;
        // TODO: Can we get the maximum with something like hard_total / top_rank.hard() ? Check 1s
        if top_cont > hard_total {
            break;
        }
        let sub_deck: Counter<Card> = deck
            .clone()
            .into_iter()
            .filter(|(c, _n)| c < &top_rank)
            .collect();
        let sub_parts =
            _weighted_partitions(sub_deck, hard_total - top_cont, norm_offset + n_top - k_top);
        let comb_parts = sub_parts.into_iter().map(|(w, mut cs)| {
            cs[&top_rank] += k_top;
            let new_hand_size = cs.total::<usize>();
            let weight_part = (0..k_top)
                .map(|k| (new_hand_size - k) as f64)
                .product::<f64>();
            (weight * weight_part * w, cs)
        });
        k_perms.push(comb_parts.collect::<Vec<_>>());

        // The weight should be (n_top CHOOSE k_top) (? - or should it?)
        // It should depend on the other elements in the joined iterate.
        // (n_top CHOOSE k_top) is the factor for this given rank in the overall multivariate
        // hypergeometric distribution. However it misses an overall (n_deck CHOOSE n_hand)
        // denominator.
        let k_top = k_top as f64;
        weight *= n_top as f64 - k_top;
        weight /= k_top + 1.;
        // This is part of the overall normalization factor with n_hand multiplicative factors
        // decreasing from n_deck. The other piece (n_hand!) has to be constructed in inner loop
        // since it depends on the hand size, I think.
        weight /= n_deck + norm_offset as f64 - k_top
    }
    k_perms.into_iter().flatten().collect::<Vec<_>>()
}

fn choose(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else if k == 0 || k == n {
        1
    } else if k + 1 == n {
        // Special case to reduce the number of recursive calls, so we don't add a bunch of zeroes.
        // Does it help?
        1 + choose(n - 1, k - 1)
    } else {
        choose(n - 1, k) + choose(n - 1, k - 1)
    }
}

/// NOTE: This is a temporary test function to compare the weights we've computed
fn check_hg_weights(hand: &CardCol, deck: &CardCol) -> usize {
    hand.inner
        .iter()
        .filter(|&(_c, &k)| k > 0)
        .map(|(c, &k)| choose(deck.get_count(c), k))
        .product()
}

#[allow(unused)]
fn check_hg_norm_weights(hand: &CardCol, deck: &CardCol) -> f64 {
    let num = check_hg_weights(hand, deck);
    // This is prohibitively expensive for multiple decks:
    let den = choose(deck.len(), hand.len());
    num as f64 / den as f64
}

fn _dealer_outcome_probs(
    hand: CardCol,
    shoe: impl Shoe,
    // exclude_nat21: bool,
) -> HashMap<DealerOutcome, f64> {
    // TODO: option to exclude natural blackjack?
    let hs17 = true;
    if !dealer_hit(&hand, hs17) {
        let dealer_count = hand.best_count();
        let res = if dealer_count > 21 {
            DealerOutcome::Bust
        } else if hand.is_nat21() {
            DealerOutcome::Natural
        } else {
            DealerOutcome::Total(dealer_count)
        };
        return HashMap::from([(res, 1.0)]);
    }
    let mut prob_map = HashMap::new();
    for (card, weight) in shoe.all_draw_probs() {
        assert!(weight > 0.);
        let mut new_hand = hand;
        new_hand.insert(card);
        let mut new_shoe = shoe.clone();
        new_shoe.draw(&card);
        let draw_probs = _dealer_outcome_probs(new_hand, new_shoe);
        for (res, prob) in draw_probs.into_iter() {
            *prob_map.entry(res).or_insert(0.) += weight * prob;
        }
    }

    prob_map
}

fn remove_nat21(dealer_outcomes: HashMap<DealerOutcome, f64>) -> HashMap<DealerOutcome, f64> {
    let nat_prob: f64 = *dealer_outcomes.get(&DealerOutcome::Natural).unwrap_or(&0.);
    let scale = 1.0 / (1.0 - nat_prob);
    let new_map = HashMap::from_iter(dealer_outcomes.into_iter().filter_map(|(o, p)| {
        if let DealerOutcome::Natural = o {
            None
        } else {
            Some((o, p * scale))
        }
    }));
    assert!((new_map.values().sum::<f64>() - 1.0).abs() < 1e-12);
    new_map
}

fn resolve_ev(player_hand: &CardCol, dealer_state: DealerOutcome) -> f64 {
    let player_state = HandState::from(player_hand);
    match (player_state, dealer_state) {
        (HandState::Natural, DealerOutcome::Natural) => 0.,
        (_, DealerOutcome::Natural) => -1.,
        (HandState::Natural, _) => 1.5, // This can change based on the rules, but should be 3/2
        (HandState::Bust, _) => -1.,
        (_, DealerOutcome::Bust) => 1.,
        (HandState::Hard(p) | HandState::Soft(p), DealerOutcome::Total(d)) => match p.cmp(&d) {
            std::cmp::Ordering::Less => -1.,
            std::cmp::Ordering::Equal => 0.,
            std::cmp::Ordering::Greater => 1.,
        },
    }
}

/// Returns a map from a given player hand to a probability weight (given by purely combinatorial
/// factors, not necessarily realistic probabilities) and an expectation value for each move made
/// with that hand, assuming optimal H/S strategy afterwards.
// TODO: Should this be a struct so it can recursively build the table by demand?
fn build_hard_evs(mut shoe: CardCol, up_card: Card) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
    // Remove the up card from the deck
    // TODO: Think about weighted partitions for infinite deck, then shoe can be impl Shoe instead
    // of CardCol.
    shoe.draw(&up_card);
    // make into const after draw
    let shoe = shoe;
    let dealer_checks_blackjack = true;

    // The future dealer draws are not totally independent from the player choices, so to be precise
    // we must wait to resolve the dealer's result conditioned on the players hand.
    let dealer_hand = CardCol::from_hand(&[up_card]);

    // Counter<Card> doesn't implement Hash
    let mut full_ev_tree = HashMap::<CardCol, (f64, HashMap<Move, f64>)>::new();

    // Go down to 2 to get all soft options as well
    for pl_tot in (2..=21).rev() {
        let pl_hands = _weighted_partitions(shoe.inner.clone(), pl_tot, 0);
        for (weight, pl_hand) in pl_hands.into_iter() {
            let pl_hand = CardCol { inner: pl_hand };
            if pl_hand.len() < 2 {
                continue;
            }
            if pl_hand.is_nat21() {
                continue;
            }

            // // This should hold, but it's expensive to check big combination factors.
            // // It compares the weighs with the hypergeometric distribution terms.
            // let norm_weight = check_hg_norm_weights(&pl_hand, &shoe);
            // assert!((weight - norm_weight).abs() < 1e-10);

            // We want to neglect naturals from this analysis, but these should be excluded from
            // the soft check.
            assert!(!pl_hand.is_nat21());
            // Assert that we aren't overdrawing; this should be a given if
            // weighted_hard_partitions() is correct.
            assert!((pl_hand.inner.clone() - shoe.inner.clone()).is_empty());
            let shoe_minus_hand = shoe.clone() - pl_hand.clone();
            // TODO: Simplify this interface so we don't have to call it like this each time. We
            // can have an option in _dealer_outcome_probs to exclude 21, but should check that it
            // gives the same answer.
            let dealer_probs = if dealer_checks_blackjack {
                remove_nat21(_dealer_outcome_probs(
                    dealer_hand.clone(),
                    shoe_minus_hand.clone(),
                ))
            } else {
                _dealer_outcome_probs(dealer_hand.clone(), shoe_minus_hand.clone())
            };
            let stand_ev = dealer_probs
                .into_iter()
                .map(|(dealer, p)| p * resolve_ev(&pl_hand, dealer))
                .sum::<f64>();
            let hit_ev = shoe_minus_hand
                .all_draw_probs()
                .map(|(c, p_c)| {
                    let mut pl_hand_hit = pl_hand.clone();
                    *pl_hand_hit.inner.entry(c).or_insert(0) += 1;
                    let pl_hand_hit = pl_hand_hit;
                    p_c * match full_ev_tree.get(&pl_hand_hit) {
                        Some((_w, ev_map)) => ev_map
                            .values()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        None => {
                            assert!(HandState::from(&pl_hand_hit) == HandState::Bust);
                            &-1.
                        }
                    }
                })
                .sum::<f64>();
            let ev_map = HashMap::from_iter([(Move::Stand, stand_ev), (Move::Hit, hit_ev)]);
            let ins_res = full_ev_tree.insert(pl_hand, (weight, ev_map));
            assert!(ins_res.is_none());
        }
        // dbg!(pl_tot);
    }
    full_ev_tree
}

// TODO: Add the double-down evs to the full hard/soft strategy tree.
fn add_double_evs(
    ev_tree: HashMap<CardCol, HashMap<Move, f64>>,
) -> HashMap<CardCol, HashMap<Move, f64>> {
    unimplemented!();
}

fn consolidate_strategy(
    ev_tree: HashMap<CardCol, (f64, HashMap<Move, f64>)>,
    // ) -> HashMap<HandState, HashMap<Move, f64>> {
) -> HashMap<HandState, Move> {
    let mut summary_tree = HashMap::new();

    let mut partitioned_evs = HashMap::<HandState, Vec<(f64, CardCol, HashMap<Move, f64>)>>::new();
    for (hand, (weight, move_ev)) in ev_tree.iter() {
        let state = HandState::from(hand);
        partitioned_evs
            .entry(state)
            .or_default()
            .push((*weight, hand.clone(), move_ev.clone()));
    }

    for (state, hands_evs) in partitioned_evs.into_iter() {
        // For each move, collect a list of the weights and evs so we can perform a weighted average
        let mut move_wts_evs = HashMap::<Move, Vec<(f64, f64)>>::new();
        for (weight, _hand, move_ev) in hands_evs.into_iter() {
            for (strat, ev) in move_ev.into_iter() {
                move_wts_evs.entry(strat).or_default().push((weight, ev));
            }
        }
        // Average out the EVs from each specific hand to collapse the move-EV relationship over all
        // specific hands to one summary (e.g. Hard 16).
        // This should represent more complete information than what is returned in this method.
        // TODO: Factor this consolidate_strategy() method into one that generates this move-EV
        // mapping and stores it for each HandState, and one that returns the best move for each
        // HandState given that mapping.
        let move_evs =
            HashMap::<Move, f64>::from_iter(move_wts_evs.into_iter().map(|(strat, wts_evs)| {
                let total_wt: f64 = wts_evs.iter().map(|(w, _)| w).sum();
                let weighted_ev: f64 = wts_evs.iter().map(|(w, ev)| w * ev).sum();
                (strat, weighted_ev / total_wt)
            }));

        let best_move = move_evs
            .into_iter()
            // This will crash on NaNs but we shouldn't be getting any
            .max_by(|(_, eva), (_, evb)| eva.partial_cmp(evb).unwrap())
            .map(|(strat, _)| strat)
            .expect("There should be at least one move here");

        let ins_res = summary_tree.insert(state, best_move);
        assert!(ins_res.is_none());
    }
    summary_tree
}

fn main() {
    println!("Hello, world!");
    println!("{}, {}, {}", Card::Pip(5), Card::Ten, Card::Ace);
    assert!(Card::Pip(2) < Card::Pip(3));
    assert!(Card::Pip(6) < Card::Ten);
    assert!(Card::Pip(9) > Card::Ace);
    assert!(Card::Ten > Card::Ace);

    let dd = CardCol::from_decks(4);
    println!("{} - {} total", dd, dd.inner.total::<usize>());

    let target_total = 16;
    let parts = _weighted_partitions(dd.inner, target_total, 0);
    // println!("{:#?}", &parts);
    for (weight, counter) in parts.iter() {
        assert!(
            counter
                .iter()
                .map(|(&c, &n)| n * c.hard() as usize)
                .sum::<usize>()
                == target_total.into()
        );
        let cc = CardCol {
            inner: counter.clone(),
        };
        println!("{}:\t{}", weight, cc);
    }
    println!("{} total partitions", parts.len());

    let dd = CardCol::from_decks(2);
    let base_deal_probs = _dealer_outcome_probs(CardCol::new(), dd.clone());
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", &base_deal_probs, norm);
    println!("{:?}\nnorm: {}", remove_nat21(base_deal_probs), norm);

    let dd = InfiniteDeck {};
    let base_deal_probs = _dealer_outcome_probs(CardCol::new(), dd);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);

    // NOTE: See https://wizardofodds.com/games/blackjack/appendix/9/1dh17r4/ for precise
    // comparisons
    let dd = CardCol::from_decks(2);
    // let dd = CardCol::half_deck();
    // let ev_map = build_hard_evs(dd, Card::Ace);
    let ev_map = build_hard_evs(dd, Card::Pip(5));
    let test_hand = CardCol::try_from("9A").unwrap();
    let soft20 = &ev_map[&test_hand];
    dbg!(soft20);

    let strat = consolidate_strategy(ev_map);
    let mut sorted_strat: Vec<_> = strat.into_iter().collect();
    sorted_strat.sort_by_key(|(h, _m)| *h);
    for (hand, strat) in sorted_strat.into_iter() {
        println!("{}: {}", hand, strat);
    }
}
