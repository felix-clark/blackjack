use counter::Counter;
use std::fmt::{Debug, Display};

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
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct CardCol {
    pub inner: Counter<Card>,
}

impl CardCol {
    pub fn new() -> Self {
        Self {
            inner: Counter::with_capacity(10),
        }
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

    pub fn is_nat21(&self) -> bool {
        // TODO: Consider if there's a more efficient way of checking this
        self.inner == Counter::<Card>::from_iter([Card::Ace, Card::Ten])
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
