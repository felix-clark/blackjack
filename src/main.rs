use counter::Counter;
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Card {
    Ace,
    /// a.k.a. Numeral cards, 2-9
    Pip(u8),
    /// In blackjack, tens and face cards are all equivalent.
    Ten,
}

impl Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Card::Pip(n) => write!(f, "{}", n),
            Card::Ten => write!(f, "T"),
            Card::Ace => write!(f, "A"),
        }
    }
}
impl Debug for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Card {
    fn hard(&self) -> u8 {
        match &self {
            Card::Ace => 1,
            Card::Pip(n) if (2..=9).contains(n) => *n,
            Card::Ten => 10,
            _ => unreachable!(),
        }
    }
}

/// Representation of a generic collection of cards
#[derive(PartialEq, Eq, Clone, Debug)]
struct CardCol {
    inner: Counter<Card>,
}

impl CardCol {
    fn new() -> Self {
        Self {
            inner: Counter::with_capacity(10),
        }
    }

    fn from_decks(n: u8) -> Self {
        let mut inner = Counter::<Card>::with_capacity(10);
        let n_per_rank = 4 * n as usize;
        for i in 2..=9 {
            inner.insert(Card::Pip(i), n_per_rank);
        }
        inner.insert(Card::Ten, 4 * n_per_rank);
        inner.insert(Card::Ace, n_per_rank);
        Self { inner }
    }

    fn best_count(&self) -> u8 {
        let hard_count = self.hard_count();
        if hard_count <= 11 && self.has_ace() {
            hard_count + 10
        } else {
            hard_count
        }
    }

    fn hard_count(&self) -> u8 {
        self.inner
            .iter()
            .map(|(c, &n)| n as u8 * c.hard())
            .sum::<u8>()
    }

    fn has_ace(&self) -> bool {
        match self.inner.get(&Card::Ace) {
            Some(&n_a) => n_a > 0,
            None => false,
        }
    }

    // fn from_nat21() -> Self {
    //     let inner = Counter::<Card>::from_iter([Card::Ace, Card::Ten]);
    //     Self { inner }
    // }

    fn is_nat21(&self) -> bool {
        // TODO: Consider if there's a more efficient way of checking this
        self.inner == Counter::<Card>::from_iter([Card::Ace, Card::Ten])
        // *self == Self::from_nat21()
    }
}

impl Display for CardCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (card, num) in self.inner.iter() {
            if num == &0 {
                continue;
            }
            // write!(f, "{}x{} ", num, card)?;
            write!(f, "{}:{} ", card, num)?;
        }
        Ok(())
    }
}

trait Shoe: Clone {
    /// Draw the card and remove it from the shoe
    fn draw(&mut self, card: &Card);

    /// Return the probability of drawing the given card, without changing the deck.
    fn draw_prob(&self, card: &Card) -> f64;

    /// Iterate over all possible cards in the deck with their weights
    fn all_draw_probs(&self) -> impl Iterator<Item = (Card, f64)>;
}

impl Shoe for CardCol {
    fn draw(&mut self, card: &Card) {
        self.inner[card] -= 1;
    }

    fn draw_prob(&self, card: &Card) -> f64 {
        let denom = self.inner.total::<usize>() as f64;
        *self.inner.get(card).unwrap_or(&0) as f64 / denom
    }

    fn all_draw_probs(&self) -> impl Iterator<Item = (Card, f64)> {
        let denom = self.inner.total::<usize>() as f64;
        self.inner
            .iter()
            .filter(|&(_c, &n)| n > 0)
            .map(move |(&c, &n)| (c, n as f64 / denom))
    }
}

#[derive(Copy, Clone)]
struct InfiniteDeck {}

impl Shoe for InfiniteDeck {
    /// Drawing from the infinite deck is a no-op.
    fn draw(&mut self, _card: &Card) {}

    fn draw_prob(&self, card: &Card) -> f64 {
        match card {
            Card::Ace => 1.0 / 13.0,
            Card::Pip(r) if (2 <= *r && *r <= 9) => 1.0 / 13.0,
            Card::Ten => 4.0 / 13.0,
            _ => unreachable!(),
        }
    }

    fn all_draw_probs(&self) -> impl Iterator<Item = (Card, f64)> {
        let mut cards = Vec::with_capacity(10);
        cards.extend((2..=9).map(|n| Card::Pip(n)));
        cards.push(Card::Ten);
        cards.push(Card::Ace);
        cards.into_iter().map(|c| (c, self.draw_prob(&c)))
    }
}

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
        <Self as Display>::fmt(&self, f)
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

// /// The stipulation of miscellaneous rules other than the number of decks (?).
// struct Ruleset {
//     /// Whether the dealer hits soft 17
//     hs17: bool,
//     /// Allowed to double after split
//     das: bool,
//     /// Whether the dealer checks their hole card for blackjack
//     dealer_check: bool,
//     // /// Double on anything (as opposed to just 10 and 11) -- maybe just assume true
//     // doa: bool,
//     // /// Whether surrender is allowed. There are 2 variants, early and late - how to encode this?
//     // surrender: bool,
//     // TODO: only allowed 1 card after splitting aces?
// }

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

    // use Card::*;
    //
    // let mut cc = CardCol {
    //     inner: Counter::<Card>::from_iter([Ace, Ten, Pip(3)].into_iter()),
    // };
    // println!("{}", cc);
    // cc.inner.remove(&Ace);
    // assert!(!cc.inner.contains_key(&Ace));
    // println!("{}", cc);
    // // This will indeed insert a zero
    // // cc.inner.insert(Ace, 0);
    // println!("{}", cc);
    // assert!(!cc.inner.contains_key(&Ace));

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
