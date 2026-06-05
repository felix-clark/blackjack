use counter::Counter;
use std::{
    fmt::{Debug, Display},
    ops::{Add, Sub},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum Card {
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

impl TryFrom<u8> for Card {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ace),
            n if (2..=9).contains(&value) => Ok(Self::Pip(n)),
            10 => Ok(Self::Ten),
            _ => Err(format!("Invald card value {}", value)),
        }
    }
}

impl TryFrom<char> for Card {
    type Error = String;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        use Card::*;
        match value {
            'A' => Ok(Ace),
            '2' => Ok(Pip(2)),
            '3' => Ok(Pip(3)),
            '4' => Ok(Pip(4)),
            '5' => Ok(Pip(5)),
            '6' => Ok(Pip(6)),
            '7' => Ok(Pip(7)),
            '8' => Ok(Pip(8)),
            '9' => Ok(Pip(9)),
            'T' => Ok(Ten),
            _ => Err(format!("Invalid card character {}", value)),
        }
    }
}

impl Card {
    pub fn hard(&self) -> u8 {
        match &self {
            Card::Ace => 1,
            Card::Pip(n) if (2..=9).contains(n) => *n,
            Card::Ten => 10,
            _ => unreachable!(),
        }
    }
}

/// Representation of a generic collection of cards
#[derive(Clone, Eq)]
pub struct CardCol {
    pub inner: Counter<Card>,
}

impl std::hash::Hash for CardCol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // As long as we always use the same order for all cards, it should be OK
        for c in (1..=10).map(Card::try_from) {
            self.inner[&c.unwrap()].hash(state);
        }
    }
}

impl PartialEq for CardCol {
    fn eq(&self, other: &Self) -> bool {
        // NOTE: The inner hash maps can be non-equal due to zero values.
        self.inner.is_subset(&other.inner) && self.inner.is_superset(&other.inner)
    }
}

impl Add for CardCol {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

impl Sub for CardCol {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

impl TryFrom<&str> for CardCol {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let cards: Vec<Card> = value
            .chars()
            .map(|c| c.try_into())
            .collect::<Result<_, _>>()?;
        let inner: Counter<Card> = Counter::from_iter(cards);
        Ok(Self { inner })
    }
}

impl CardCol {
    pub fn new() -> Self {
        Self {
            inner: Counter::with_capacity(10),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.total()
    }

    pub fn get_count(&self, card: &Card) -> usize {
        *self.inner.get(card).unwrap_or(&0)
    }

    pub fn from_decks(n: u8) -> Self {
        let mut inner = Counter::<Card>::with_capacity(10);
        let n_per_rank = 4 * n as usize;
        for i in 2..=9 {
            inner.insert(Card::Pip(i), n_per_rank);
        }
        inner.insert(Card::Ten, 4 * n_per_rank);
        inner.insert(Card::Ace, n_per_rank);
        Self { inner }
    }

    pub fn half_deck() -> Self {
        let mut inner = Counter::<Card>::with_capacity(10);
        let n_per_rank = 2;
        for i in 2..=9 {
            inner.insert(Card::Pip(i), n_per_rank);
        }
        inner.insert(Card::Ten, 4 * n_per_rank);
        inner.insert(Card::Ace, n_per_rank);
        Self { inner }
    }

    pub fn from_hand(hand: &[Card]) -> Self {
        Self {
            inner: hand.iter().cloned().collect::<Counter<_>>(),
        }
    }

    pub fn best_count(&self) -> u8 {
        let hard_count = self.hard_count();
        if hard_count <= 11 && self.has_ace() {
            hard_count + 10
        } else {
            hard_count
        }
    }

    pub fn hard_count(&self) -> u8 {
        self.inner
            .iter()
            .map(|(c, &n)| n as u8 * c.hard())
            .sum::<u8>()
    }

    pub fn has_ace(&self) -> bool {
        match self.inner.get(&Card::Ace) {
            Some(&n_a) => n_a > 0,
            None => false,
        }
    }

    fn from_nat21() -> Self {
        let inner = Counter::<Card>::from_iter([Card::Ace, Card::Ten]);
        Self { inner }
    }

    pub fn is_nat21(&self) -> bool {
        // TODO: Consider if there's a more efficient way of checking this
        // NOTE: It's important that we not simply equate the inner counters, because those
        // consider 0s to be distinct from empties.
        self == &Self::from_nat21()
    }
}

impl Display for CardCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (card, num) in self.inner.iter() {
            if num == &0 {
                continue;
            }
            write!(f, "{}:{} ", card, num)?;
        }
        Ok(())
    }
}

impl Debug for CardCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (card, num) in self.inner.iter() {
            // Consider not skipping zeros for debug, it's actually important esp. with testing equality.
            if num == &0 {
                continue;
            }
            write!(f, "{}x{} ", num, card)?;
        }
        Ok(())
    }
}

pub trait Shoe: Clone {
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
pub struct InfiniteDeck {}

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
        cards.extend((2..=9).map(Card::Pip));
        cards.push(Card::Ten);
        cards.push(Card::Ace);
        cards.into_iter().map(|c| (c, self.draw_prob(&c)))
    }
}
