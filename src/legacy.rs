/// Methods we're no longer using, typically because they've been optimized out.
use crate::{card::Card, dealer::DealerOutcome, shoe::CardCol, shoe::Shoe};

use std::collections::HashMap;

/// The weights should be the multivariate hypergeometric probability density. This gives the
/// probability conditioned on the size of the hand; it takes some additional assumptions to infer
/// a marginal distribution.
/// Now this is implemented as a lazy iterator from a Shoe member method.
#[allow(unused)]
fn weighted_partitions_legacy(
    mut deck: CardCol,
    hard_total: u8,
    norm_offset: usize,
) -> Vec<(f64, CardCol)> {
    // TODO: double-check this condition
    // NOTE: This order seemed necessary to get all partitions, but there is an awkwardness with
    // the weights
    if hard_total == 0 {
        return vec![(1., CardCol::new())];
    }
    if deck.is_empty() {
        return Vec::new();
    }
    // TODO: Should this start with 1. / shoe.len() ?
    // It matters whether this is set before or after the draw
    let n_deck = deck.len() as f64;
    let mut weight = 1.;

    // NOTE: This can probably be any rank. Keep it now just in case, and verify the results later.
    // Removing the top (highest) rank leaves exactly the sub-deck of lower ranks for the recursion.
    let top_rank: Card = deck.highest_rank().expect("deck is non-empty");
    let n_top = deck.remove_rank(top_rank) as usize;
    let sub_deck = deck;

    let mut k_perms: Vec<Vec<(f64, CardCol)>> = Vec::new();
    for k_top in 0..=n_top {
        let top_cont = top_rank.hard() * k_top as u8;
        // TODO: Can we get the maximum with something like hard_total / top_rank.hard() ? Check 1s
        if top_cont > hard_total {
            break;
        }
        let sub_parts = weighted_partitions_legacy(
            sub_deck,
            hard_total - top_cont,
            norm_offset + n_top - k_top,
        );
        let comb_parts = sub_parts.into_iter().map(|(w, mut cs)| {
            cs.add_n(top_rank, k_top as u16);
            let new_hand_size = cs.len();
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

fn dealer_hit(hand: &CardCol, hs17: bool) -> bool {
    let hard_count: u8 = hand.hard_count();
    if hard_count >= 17 {
        return false;
    }
    let has_ace: bool = hand.has_ace();
    if has_ace && hard_count <= 11 {
        let soft_target = if hs17 { 18 } else { 17 };
        if hard_count + 10 >= soft_target {
            return false;
        }
    }
    true
}

#[allow(unused)]
pub fn dealer_outcome_probs(
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
        let draw_probs = dealer_outcome_probs(new_hand, new_shoe);
        for (res, prob) in draw_probs.into_iter() {
            *prob_map.entry(res).or_insert(0.) += weight * prob;
        }
    }

    prob_map
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
    hand.iter()
        .map(|(card, k)| choose(deck.get_count(&card) as usize, k as usize))
        .product()
}

#[allow(unused)]
fn check_hg_norm_weights(hand: &CardCol, deck: &CardCol) -> f64 {
    let num = check_hg_weights(hand, deck);
    // This is prohibitively expensive for multiple decks:
    let den = choose(deck.len(), hand.len());
    num as f64 / den as f64
}

/// Used to display the peek-conditioned dealer distribution alongside the raw one; not on the solver
/// hot path (which conditions exactly). The solver applies the peek conditioning once at the 2-card
/// root of the EV tree, so this standalone renormalisation is no longer called.
#[allow(dead_code)]
pub fn remove_nat21(dealer_outcomes: HashMap<DealerOutcome, f64>) -> HashMap<DealerOutcome, f64> {
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
