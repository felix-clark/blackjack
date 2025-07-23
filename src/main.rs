mod card;
use card::*;

use counter::Counter;
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

#[derive(PartialEq, Eq, Debug)]
enum HandState {
    Hard(u8),
    Soft(u8),
    Natural,
    Bust,
}

impl From<&CardCol> for HandState {
    fn from(hand: &CardCol) -> Self {
        if hand.is_nat21() {
            return Self::Natural;
        }
        let has_ace = hand.has_ace();
        let hard_count = hand.hard_count();
        assert!(
            !has_ace || hand.inner.total::<usize>() != 2 || hard_count != 11,
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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

enum Move {
    Hit,
    Stand,
    Double,
    Split,
    Surrender,
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
fn _weighted_hard_partitions(mut deck: Counter<Card>, total: u8) -> Vec<(usize, Counter<Card>)> {
    // ) -> impl Iterator<Item = (usize, Counter<Card>)> {

    // TODO: double-check this condition
    if total == 0 {
        // with_capacity?
        return vec![(1, Counter::new())];
        // return std::iter::once((1, Counter::new()));
    }
    if deck.total::<usize>() == 0 {
        return Vec::new();
        // return std::iter::empty::<(usize, Counter<Card>)>();
    }
    let top_rank: Card = *deck.keys().map(|c| (c.hard(), c)).max().unwrap().1;
    let n_top = deck.remove(&top_rank).expect("Should be in there");

    let mut k_perms: Vec<Vec<(usize, Counter<Card>)>> = Vec::new();
    let mut weight = 1;
    for k_top in 0..=n_top {
        let top_cont = top_rank.hard() * k_top as u8;
        if top_cont > total {
            break;
        }
        let sub_deck: Counter<Card> = deck
            .clone()
            .into_iter()
            .filter(|(c, _n)| c < &top_rank)
            .collect();
        let sub_parts = _weighted_hard_partitions(sub_deck, total - top_cont);
        let comb_parts = sub_parts.into_iter().map(|(w, mut cs)| {
            cs[&top_rank] += k_top;
            (weight * w, cs)
        });
        k_perms.push(comb_parts.collect::<Vec<_>>());

        // The weight should be (n_top CHOOSE k_top)
        weight *= n_top - k_top;
        assert!(weight % (k_top + 1) == 0);
        weight /= k_top + 1;
    }
    k_perms.into_iter().flatten().collect::<Vec<_>>()
}

fn _dealer_outcome_probs(hand: CardCol, shoe: impl Shoe) -> HashMap<DealerOutcome, f64> {
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
        let mut new_hand = hand.clone();
        // new_hand.add(card);
        new_hand.inner[&card] += 1;
        let mut new_shoe = shoe.clone();
        // NOTE: Can we do this operation such that the key is removed when zeroed?
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

fn resolve_ev(player_hand: &CardCol, dealer_hand: &CardCol) -> f64 {
    // TODO: We need a conversion from hand to DealerOutcome, although it should somehow enforce
    // that it's a final outcome (e.g. 17+ or bust)
    let player_state = HandState::from(player_hand);
    let dealer_count = dealer_hand.best_count();
    let dealer_state = if dealer_count > 21 {
        DealerOutcome::Bust
    } else if dealer_hand.is_nat21() {
        DealerOutcome::Natural
    } else {
        assert!((17..=21).contains(&dealer_count));
        DealerOutcome::Total(dealer_count)
    };
    match (player_state, dealer_state) {
        (HandState::Natural, DealerOutcome::Natural) => 0.0,
        (_, DealerOutcome::Natural) => -1.,
        (HandState::Natural, _) => 1.5, // This can change based on the rules, but should be 3/2
        (HandState::Bust, _) => -1.,
        (_, DealerOutcome::Bust) => 1.0,
        (HandState::Hard(p) | HandState::Soft(p), DealerOutcome::Total(d)) => match p.cmp(&d) {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => 1.0,
        },
    }
}

// NOTE: We really only care about hit/stand choices here, so the value could be a tuple?
// Should this be a struct so it can recursively build the table by demand?
fn build_hard_evs(mut shoe: impl Shoe, up_card: Card) -> HashMap<HandState, HashMap<Move, f64>> {
    // Remove the up card from the deck
    shoe.draw(&up_card);
    todo!()
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

    let target_total = 21;
    let parts = _weighted_hard_partitions(dd.inner, target_total);
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
    let base_deal_probs = _dealer_outcome_probs(CardCol::new(), dd);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);

    let dd = InfiniteDeck {};
    let base_deal_probs = _dealer_outcome_probs(CardCol::new(), dd);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);
}
