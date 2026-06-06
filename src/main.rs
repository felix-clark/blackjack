pub(crate) mod card;
pub(crate) mod dealer;
mod legacy;
pub(crate) mod shoe;

use card::*;
use dealer::*;
use shoe::*;

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

/// Returns a map from a given player hand to a probability weight and an expectation value for each
/// move made with that hand, assuming optimal H/S strategy afterwards.
///
/// The weight is the shoe's partition scan-weight for that exact multiset (see
/// [`Shoe::weighted_partitions`]). Its meaning depends on the shoe: for the [`InfiniteDeck`] it is
/// the exact multinomial occurrence probability of the hand, but for a finite [`CardCol`] it is the
/// hypergeometric weight of drawing the hand *in isolation* — a purely combinatorial factor, not the
/// realistic probability of holding it in play (which would have to account for the up-card and the
/// dealer's draws depleting the same shoe). It is used only as a relative weight when collapsing
/// per-hand EVs into per-[`HandState`] summaries in [`consolidate_strategy`].
// TODO: Should this be a struct so it can recursively build the table by demand?
fn build_hard_evs(
    mut shoe: impl Shoe,
    up_card: Card,
    rules: &Ruleset,
) -> HashMap<CardCol, (f64, HashMap<Move, f64>)> {
    // Remove the up card from the deck (a no-op for the infinite deck).
    shoe.draw(&up_card);
    // make into const after draw
    let shoe = shoe;

    // The future dealer draws are not totally independent from the player choices, so to be precise
    // we must wait to resolve the dealer's result conditioned on the players hand.
    let dealer_hand = CardCol::from_hand(&[up_card]);

    let mut full_ev_tree = HashMap::<CardCol, (f64, HashMap<Move, f64>)>::new();

    // Go down to 2 to get all soft options as well
    for pl_tot in (2..=21).rev() {
        let pl_hands = shoe.weighted_partitions(pl_tot);
        for (weight, pl_hand) in pl_hands.into_iter() {
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
            // weighted_hard_partitions() is correct (always true for the infinite deck).
            assert!(shoe.contains_hand(&pl_hand));
            let shoe_minus_hand = shoe.remove_hand(&pl_hand);
            // The dealer-natural conditioning (peek rule) is applied inside `dealer_outcome_probs`
            // per `rules.dealer_check`.
            let dealer_probs = dealer_outcome_probs(dealer_hand, shoe_minus_hand.clone(), rules);
            let stand_ev = dealer_probs
                .into_iter()
                .map(|(dealer, p)| p * resolve_ev(&pl_hand, dealer))
                .sum::<f64>();
            let hit_ev = shoe_minus_hand
                .all_draw_probs()
                .map(|(c, p_c)| {
                    let mut pl_hand_hit = pl_hand;
                    pl_hand_hit.insert(c);
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
    _ev_tree: HashMap<CardCol, HashMap<Move, f64>>,
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
            .push((*weight, *hand, move_ev.clone()));
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
    // `dealer_check: false` keeps the raw distribution (natural included) so we can show both the
    // unconditioned probs and the peek-conditioned ones side by side.
    let no_peek = Ruleset {
        dealer_check: false,
        ..Ruleset::default()
    };
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), dd, &no_peek);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", &base_deal_probs, norm);
    println!("{:?}\nnorm: {}", remove_nat21(base_deal_probs), norm);

    let dd = InfiniteDeck {};
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), dd, &no_peek);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);

    // NOTE: See https://wizardofodds.com/games/blackjack/appendix/9/1dh17r4/ for precise
    // comparisons
    let dd = CardCol::from_decks(2);
    // let dd = CardCol::half_deck();
    // let ev_map = build_hard_evs(dd, Card::Ace, &rules);
    let ev_map = build_hard_evs(dd, Card::Pip(5), &rules);
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
